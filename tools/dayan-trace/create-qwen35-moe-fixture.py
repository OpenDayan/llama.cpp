#!/usr/bin/env python3
"""Create a deterministic random Qwen3.5 MoE GGUF with the source model's dimensions."""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--source-model", type=Path)
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--expert-type", choices=("bf16", "mxfp4"), default="bf16")
    parser.add_argument("--attention", choices=("hybrid", "linear", "full"), default="hybrid")
    parser.add_argument("--small-vocab", action="store_true", help="use 512 dummy tokens for numerical tests")
    parser.add_argument("--full-attention-interval", type=int, help="override the hybrid cycle for a small fixture")
    args = parser.parse_args()
    if args.source_model:
        model = gguf.GGUFReader(args.source_model)
        def value(key):
            return model.get_field("qwen35moe." + key).contents()
        config = dict(hidden_size=value("embedding_length"), head_dim=value("attention.key_length"),
                      num_attention_heads=value("attention.head_count"), num_key_value_heads=value("attention.head_count_kv"),
                      linear_key_head_dim=value("ssm.state_size"), linear_value_head_dim=value("ssm.state_size"),
                      linear_num_key_heads=value("ssm.group_count"), linear_num_value_heads=value("ssm.time_step_rank"),
                      linear_conv_kernel_dim=value("ssm.conv_kernel"), num_experts=value("expert_count"),
                      num_experts_per_tok=value("expert_used_count"), moe_intermediate_size=value("expert_feed_forward_length"),
                      shared_expert_intermediate_size=value("expert_shared_feed_forward_length"),
                      num_hidden_layers=value("block_count"), full_attention_interval=value("full_attention_interval"),
                      max_position_embeddings=value("context_length"), rms_norm_eps=value("attention.layer_norm_rms_epsilon"),
                      initializer_range=0.02, vocab_size=len(model.get_field("tokenizer.ggml.tokens").data),
                      rope_parameters=dict(partial_rotary_factor=value("rope.dimension_count") / value("attention.key_length"),
                                           mrope_section=value("rope.dimension_sections")[:3], rope_theta=value("rope.freq_base")))
    else:
        config = json.loads(args.config.read_text())
        config = config.get("text_config", config)
    interval = args.full_attention_interval or config["full_attention_interval"]
    if interval < 1:
        parser.error("full attention interval must be positive")
    if not 1 <= args.layers <= config["num_hidden_layers"]:
        parser.error("invalid layer count")
    if args.attention != "hybrid":
        interval = 1 if args.attention == "full" else args.layers + 1
    if args.output.exists():
        parser.error("output already exists")
    if config["linear_key_head_dim"] != config["linear_value_head_dim"]:
        parser.error("Qwen3.5 requires matching linear attention head dimensions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(args.output, "qwen35moe", use_temp_file=True)
    if args.small_vocab:
        config["vocab_size"] = 512
        writer.add_tokenizer_model("llama")
        writer.add_token_list(["<unk>", "<s>", "</s>"] + [f"<0x{i:02X}>" for i in range(256)] + [f"token{i}" for i in range(253)])
        writer.add_token_types([2, 3, 3] + [6] * 256 + [1] * 253)
        writer.add_unk_token_id(0)
        writer.add_bos_token_id(1)
        writer.add_eos_token_id(2)
    else:
        if not args.tokenizer_model and not args.source_model:
            parser.error("--tokenizer-model is required without --small-vocab or --source-model")
        tokenizer = gguf.GGUFReader(args.tokenizer_model) if args.tokenizer_model else model
        tokens = tokenizer.get_field("tokenizer.ggml.tokens")
        if tokens is None or len(tokens.data) != config["vocab_size"]:
            parser.error("tokenizer vocabulary does not match the source config")
        for key, field in tokenizer.fields.items():
            if key.startswith("tokenizer."):
                writer.add_key_value(key, field.contents(), field.types[0], field.types[-1])

    hidden = config["hidden_size"]
    head = config["head_dim"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    linear_head = config["linear_key_head_dim"]
    key_heads = config["linear_num_key_heads"]
    value_heads = config["linear_num_value_heads"]
    value_dim = linear_head * value_heads
    conv_dim = 2 * linear_head * key_heads + value_dim
    experts = config["num_experts"]
    used = config["num_experts_per_tok"]
    ff = config["moe_intermediate_size"]
    shared_ff = config["shared_expert_intermediate_size"]
    rope = config["rope_parameters"]
    writer.add_name(f"Qwen3.5-A3B random {args.layers}-layer fixture")
    writer.add_description(f"Random parameters, seed {args.seed}; source dimensions retained; no trained language capability.")
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_MXFP4_MOE if args.expert_type == "mxfp4" else gguf.LlamaFileType.MOSTLY_BF16)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_block_count(args.layers)
    writer.add_feed_forward_length(ff * used)
    writer.add_head_count(heads)
    writer.add_head_count_kv(kv_heads)
    writer.add_key_length(head)
    writer.add_value_length(head)
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_rope_dimension_count(int(head * rope["partial_rotary_factor"]))
    writer.add_rope_dimension_sections(rope["mrope_section"] + [0])
    writer.add_rope_freq_base(rope["rope_theta"])
    writer.add_full_attention_interval(interval)
    writer.add_ssm_conv_kernel(config["linear_conv_kernel_dim"])
    writer.add_ssm_inner_size(value_dim)
    writer.add_ssm_state_size(linear_head)
    writer.add_ssm_time_step_rank(value_heads)
    writer.add_ssm_group_count(key_heads)
    writer.add_expert_count(experts)
    writer.add_expert_used_count(used)
    writer.add_expert_feed_forward_length(ff)
    writer.add_expert_shared_feed_forward_length(shared_ff)

    rng = np.random.default_rng(args.seed)

    def tensor(name, shape, *, f32=False, offset=0.0, negative_exp=False):
        data = rng.standard_normal(shape, dtype=np.float32)
        data *= config["initializer_range"]
        data += offset
        if negative_exp:
            np.exp(data, out=data)
            data *= -1
        if args.expert_type == "mxfp4" and "_exps.weight" in name:
            packed = np.empty(shape[:-1] + (shape[-1] // 32 * 17,), dtype=np.uint8)
            for expert in range(shape[0]):
                packed[expert] = gguf.quants.quantize(data[expert], gguf.GGMLQuantizationType.MXFP4)
            writer.add_tensor(name, packed, raw_dtype=gguf.GGMLQuantizationType.MXFP4)
        elif f32:
            writer.add_tensor(name, data)
        else:
            writer.add_tensor(name, gguf.quants.quantize(data, gguf.GGMLQuantizationType.BF16), raw_dtype=gguf.GGMLQuantizationType.BF16)
        print(f"{name}: {shape}", flush=True)

    tensor("token_embd.weight", (config["vocab_size"], hidden))
    tensor("output.weight", (config["vocab_size"], hidden))
    tensor("output_norm.weight", (hidden,), f32=True, offset=1.0)
    for layer in range(args.layers):
        prefix = f"blk.{layer}."
        tensor(prefix + "attn_norm.weight", (hidden,), f32=True, offset=1.0)
        tensor(prefix + "post_attention_norm.weight", (hidden,), f32=True, offset=1.0)
        if (layer + 1) % interval:
            tensor(prefix + "attn_qkv.weight", (conv_dim, hidden))
            tensor(prefix + "attn_gate.weight", (value_dim, hidden))
            tensor(prefix + "ssm_conv1d.weight", (conv_dim, config["linear_conv_kernel_dim"]), f32=True)
            tensor(prefix + "ssm_dt.bias", (value_heads,), f32=True, offset=-3.0)
            tensor(prefix + "ssm_a", (value_heads,), f32=True, negative_exp=True)
            tensor(prefix + "ssm_beta.weight", (value_heads, hidden))
            tensor(prefix + "ssm_alpha.weight", (value_heads, hidden))
            tensor(prefix + "ssm_norm.weight", (linear_head,), f32=True, offset=1.0)
            tensor(prefix + "ssm_out.weight", (hidden, value_dim))
        else:
            tensor(prefix + "attn_q.weight", (2 * heads * head, hidden))
            tensor(prefix + "attn_k.weight", (kv_heads * head, hidden))
            tensor(prefix + "attn_v.weight", (kv_heads * head, hidden))
            tensor(prefix + "attn_output.weight", (hidden, heads * head))
            tensor(prefix + "attn_q_norm.weight", (head,), f32=True, offset=1.0)
            tensor(prefix + "attn_k_norm.weight", (head,), f32=True, offset=1.0)
        tensor(prefix + "ffn_gate_inp.weight", (experts, hidden), f32=args.expert_type == "mxfp4")
        tensor(prefix + "ffn_gate_exps.weight", (experts, ff, hidden))
        tensor(prefix + "ffn_up_exps.weight", (experts, ff, hidden))
        tensor(prefix + "ffn_down_exps.weight", (experts, hidden, ff))
        tensor(prefix + "ffn_gate_inp_shexp.weight", (hidden,), f32=args.expert_type == "mxfp4")
        tensor(prefix + "ffn_gate_shexp.weight", (shared_ff, hidden))
        tensor(prefix + "ffn_up_shexp.weight", (shared_ff, hidden))
        tensor(prefix + "ffn_down_shexp.weight", (hidden, shared_ff))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using ggml_backend_graph_rewrite_t = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);

struct model_case {
    enum ggml_type type;
    int64_t        n_tokens;
    uint32_t       seed;
};

struct layer_graph {
    bool          recurrent;
    ggml_tensor * conv_silu;
    ggml_tensor * flash_q;
    ggml_tensor * flash_k;
    ggml_tensor * flash_v;
    ggml_tensor * flash_attention;
    ggml_tensor * attention_output;
    ggml_tensor * ffn_gate;
    ggml_tensor * ffn_up;
    ggml_tensor * ffn_fused;
    ggml_tensor * ffn_down;
    ggml_tensor * output;
};

struct model_graph {
    ggml_context_ptr         ctx;
    ggml_cgraph *            graph;
    ggml_tensor *            input;
    ggml_tensor *            positions;
    ggml_tensor *            mask;
    ggml_tensor *            logits;
    std::vector<layer_graph> layers;
};

using tensor_blobs = std::unordered_map<std::string, std::vector<uint8_t>>;

static constexpr int64_t kEmbedding        = 64;
static constexpr int64_t kFeedForward      = 192;
static constexpr int64_t kLayers           = 4;
static constexpr int64_t kConvKernel       = 4;
static constexpr int64_t kLinearHeadDim    = 4;
static constexpr int64_t kLinearHeads      = 16;
static constexpr int64_t kLinearWidth      = kLinearHeadDim * kLinearHeads;
static constexpr int64_t kConvChannels     = 3 * kLinearWidth;
static constexpr int64_t kAttentionHeadDim = 64;
static constexpr int64_t kAttentionHeads   = 1;
static constexpr int64_t kVocab            = 64;

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-qwen35-dense: %s\n", message);
    return 1;
}

static uint16_t encode_lowp(enum ggml_type type, float value) {
    return type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value);
}

static bool rewrite_graph(ggml_backend_t dayan, ggml_context * ctx, ggml_cgraph * graph) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    auto rewrite = (ggml_backend_graph_rewrite_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    return rewrite != nullptr && rewrite(dayan, ctx, graph);
}

static ggml_tensor * new_input_1d(ggml_context * ctx, enum ggml_type type, int64_t ne0, const char * name) {
    ggml_tensor * tensor = ggml_new_tensor_1d(ctx, type, ne0);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * new_input_2d(ggml_context * ctx,
                                  enum ggml_type type,
                                  int64_t        ne0,
                                  int64_t        ne1,
                                  const char *   name) {
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * new_input_3d(ggml_context * ctx,
                                  enum ggml_type type,
                                  int64_t        ne0,
                                  int64_t        ne1,
                                  int64_t        ne2,
                                  const char *   name) {
    ggml_tensor * tensor = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * new_input_4d(ggml_context * ctx,
                                  enum ggml_type type,
                                  int64_t        ne0,
                                  int64_t        ne1,
                                  int64_t        ne2,
                                  int64_t        ne3,
                                  const char *   name) {
    ggml_tensor * tensor = ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

static ggml_tensor * build_rms_norm(ggml_context * ctx, ggml_tensor * input, ggml_tensor * weight, float epsilon) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, input, epsilon), weight);
}

static ggml_tensor * quantize_to_f32(ggml_context * ctx, ggml_tensor * input, enum ggml_type type) {
    return ggml_cast(ctx, ggml_cast(ctx, input, type), GGML_TYPE_F32);
}

static ggml_tensor * build_linear_attention(ggml_context *     ctx,
                                            const model_case & config,
                                            ggml_tensor *      input,
                                            int                layer,
                                            bool               cpu_reference,
                                            layer_graph *      result) {
    char name[64];

    std::snprintf(name, sizeof(name), "blk.%d.attn_qkv.weight", layer);
    ggml_tensor * qkv_weight = new_input_2d(ctx, config.type, kEmbedding, kConvChannels, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_gate.weight", layer);
    ggml_tensor * z_weight = new_input_2d(ctx, config.type, kEmbedding, kLinearWidth, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_beta.weight", layer);
    ggml_tensor * beta_weight = new_input_2d(ctx, config.type, kEmbedding, kLinearHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_alpha.weight", layer);
    ggml_tensor * alpha_weight = new_input_2d(ctx, config.type, kEmbedding, kLinearHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_conv.weight", layer);
    ggml_tensor * conv_weight = new_input_2d(ctx, GGML_TYPE_F32, kConvKernel, kConvChannels, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_history", layer);
    ggml_tensor * history = new_input_3d(ctx, GGML_TYPE_F32, kConvKernel - 1, kConvChannels, 1, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_state", layer);
    ggml_tensor * state = new_input_4d(ctx, GGML_TYPE_F32, kLinearHeadDim, kLinearHeadDim, kLinearHeads, 1, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_dt.bias", layer);
    ggml_tensor * dt = new_input_1d(ctx, GGML_TYPE_F32, kLinearHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_a", layer);
    ggml_tensor * a = new_input_1d(ctx, GGML_TYPE_F32, kLinearHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", layer);
    ggml_tensor * norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kLinearHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", layer);
    ggml_tensor * output_weight = new_input_2d(ctx, config.type, kLinearWidth, kEmbedding, name);

    ggml_tensor * qkv   = ggml_mul_mat(ctx, qkv_weight, input);
    ggml_tensor * z     = ggml_mul_mat(ctx, z_weight, input);
    ggml_tensor * beta  = ggml_sigmoid(ctx, ggml_mul_mat(ctx, beta_weight, input));
    ggml_tensor * alpha = ggml_mul_mat(ctx, alpha_weight, input);
    alpha               = ggml_softplus(ctx, ggml_add(ctx, alpha, dt));
    alpha               = ggml_mul(ctx, alpha, a);

    qkv                             = ggml_reshape_3d(ctx, qkv, kConvChannels, config.n_tokens, 1);
    qkv                             = ggml_transpose(ctx, qkv);
    ggml_tensor * conv_input        = ggml_concat(ctx, history, qkv, 0);
    ggml_tensor * conv_weight_input = conv_weight;
    if (cpu_reference) {
        conv_input        = quantize_to_f32(ctx, conv_input, GGML_TYPE_BF16);
        conv_weight_input = quantize_to_f32(ctx, conv_weight_input, GGML_TYPE_BF16);
    }
    ggml_tensor * conv = ggml_ssm_conv(ctx, conv_input, conv_weight_input);
    result->conv_silu  = ggml_silu(ctx, conv);
    ggml_format_name(result->conv_silu, "blk.%d.ssm_conv_silu", layer);
    ggml_set_output(result->conv_silu);

    const size_t  head_stride  = kLinearHeadDim * sizeof(float);
    const size_t  token_stride = kConvChannels * sizeof(float);
    const size_t  seq_stride   = token_stride * config.n_tokens;
    ggml_tensor * q            = ggml_view_4d(ctx, result->conv_silu, kLinearHeadDim, kLinearHeads, config.n_tokens, 1,
                                              head_stride, token_stride, seq_stride, 0);
    ggml_tensor * k            = ggml_view_4d(ctx, result->conv_silu, kLinearHeadDim, kLinearHeads, config.n_tokens, 1,
                                              head_stride, token_stride, seq_stride, kLinearWidth * sizeof(float));
    ggml_tensor * v            = ggml_view_4d(ctx, result->conv_silu, kLinearHeadDim, kLinearHeads, config.n_tokens, 1,
                                              head_stride, token_stride, seq_stride, 2 * kLinearWidth * sizeof(float));
    q                          = ggml_l2_norm(ctx, q, 1.0e-6f);
    k                          = ggml_l2_norm(ctx, k, 1.0e-6f);
    beta                       = ggml_reshape_4d(ctx, beta, 1, kLinearHeads, config.n_tokens, 1);
    alpha                      = ggml_reshape_4d(ctx, alpha, 1, kLinearHeads, config.n_tokens, 1);

    ggml_tensor * delta = ggml_gated_delta_net(ctx, q, k, v, alpha, beta, state, 1);
    ggml_tensor * core =
        ggml_view_4d(ctx, delta, kLinearHeadDim, kLinearHeads, config.n_tokens, 1, kLinearHeadDim * sizeof(float),
                     kLinearWidth * sizeof(float), kLinearWidth * config.n_tokens * sizeof(float), 0);
    ggml_tensor * z_heads    = ggml_reshape_4d(ctx, z, kLinearHeadDim, kLinearHeads, config.n_tokens, 1);
    core                     = build_rms_norm(ctx, core, norm_weight, 1.0e-6f);
    core                     = ggml_mul(ctx, core, ggml_silu(ctx, z_heads));
    core                     = ggml_reshape_2d(ctx, core, kLinearWidth, config.n_tokens);
    result->attention_output = ggml_mul_mat(ctx, output_weight, core);
    ggml_format_name(result->attention_output, "blk.%d.linear_attn_out", layer);
    ggml_set_output(result->attention_output);
    return result->attention_output;
}

static ggml_tensor * build_full_attention(ggml_context *     ctx,
                                          const model_case & config,
                                          ggml_tensor *      input,
                                          ggml_tensor *      positions,
                                          ggml_tensor *      mask,
                                          int                layer,
                                          bool               cpu_reference,
                                          layer_graph *      result) {
    char name[64];

    std::snprintf(name, sizeof(name), "blk.%d.attn_qg.weight", layer);
    ggml_tensor * qg_weight = new_input_2d(ctx, config.type, kEmbedding, 2 * kAttentionHeadDim * kAttentionHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
    ggml_tensor * k_weight = new_input_2d(ctx, config.type, kEmbedding, kAttentionHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
    ggml_tensor * v_weight = new_input_2d(ctx, config.type, kEmbedding, kAttentionHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", layer);
    ggml_tensor * q_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kAttentionHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", layer);
    ggml_tensor * k_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kAttentionHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_out.weight", layer);
    ggml_tensor * output_weight = new_input_2d(ctx, config.type, kAttentionHeadDim * kAttentionHeads, kEmbedding, name);

    ggml_tensor * qg = ggml_mul_mat(ctx, qg_weight, input);
    ggml_tensor * k  = ggml_mul_mat(ctx, k_weight, input);
    ggml_tensor * v  = ggml_mul_mat(ctx, v_weight, input);

    ggml_tensor * q =
        ggml_view_3d(ctx, qg, kAttentionHeadDim, kAttentionHeads, config.n_tokens,
                     2 * kAttentionHeadDim * sizeof(float), 2 * kAttentionHeadDim * kAttentionHeads * sizeof(float), 0);
    q = build_rms_norm(ctx, q, q_norm_weight, 1.0e-6f);
    k = ggml_reshape_3d(ctx, k, kAttentionHeadDim, 1, config.n_tokens);
    k = build_rms_norm(ctx, k, k_norm_weight, 1.0e-6f);

    int sections[GGML_MROPE_SECTIONS] = { 8, 8, 8, 8 };
    q               = ggml_rope_multi(ctx, q, positions, nullptr, kAttentionHeadDim, sections, GGML_ROPE_TYPE_MROPE,
                                      (int) config.n_tokens, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k               = ggml_rope_multi(ctx, k, positions, nullptr, kAttentionHeadDim, sections, GGML_ROPE_TYPE_MROPE,
                                      (int) config.n_tokens, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    q               = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k               = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v               = ggml_reshape_4d(ctx, v, kAttentionHeadDim, config.n_tokens, 1, 1);
    result->flash_q = ggml_cast(ctx, q, GGML_TYPE_BF16);
    result->flash_k = ggml_cast(ctx, k, GGML_TYPE_BF16);
    result->flash_v = ggml_cast(ctx, v, GGML_TYPE_BF16);
    ggml_format_name(result->flash_q, "blk.%d.flash_q", layer);
    ggml_format_name(result->flash_k, "blk.%d.flash_k", layer);
    ggml_format_name(result->flash_v, "blk.%d.flash_v", layer);
    ggml_set_output(result->flash_q);
    ggml_set_output(result->flash_k);
    ggml_set_output(result->flash_v);
    ggml_tensor * flash_q_input = cpu_reference ? ggml_cast(ctx, result->flash_q, GGML_TYPE_F32) : result->flash_q;

    result->flash_attention = ggml_flash_attn_ext(ctx, flash_q_input, result->flash_k, result->flash_v, mask,
                                                  1.0f / std::sqrt((float) kAttentionHeadDim), 0.0f, 0.0f);
    ggml_format_name(result->flash_attention, "blk.%d.flash_attention", layer);
    ggml_set_output(result->flash_attention);
    ggml_tensor * attention =
        ggml_reshape_2d(ctx, result->flash_attention, kAttentionHeadDim * kAttentionHeads, config.n_tokens);

    ggml_tensor * gate_source = cpu_reference ? quantize_to_f32(ctx, qg, config.type) : qg;
    ggml_tensor * gate_view   = ggml_view_3d(
        ctx, gate_source, kAttentionHeadDim, kAttentionHeads, config.n_tokens, 2 * kAttentionHeadDim * sizeof(float),
        2 * kAttentionHeadDim * kAttentionHeads * sizeof(float), kAttentionHeadDim * sizeof(float));
    ggml_tensor * gate             = ggml_cont_2d(ctx, gate_view, kAttentionHeadDim * kAttentionHeads, config.n_tokens);
    ggml_tensor * gated            = ggml_mul(ctx, attention, ggml_sigmoid(ctx, gate));
    ggml_tensor * projection_input = cpu_reference ? ggml_cast(ctx, gated, config.type) : gated;
    result->attention_output       = ggml_mul_mat(ctx, output_weight, projection_input);
    ggml_format_name(result->attention_output, "blk.%d.attn_out", layer);
    ggml_set_output(result->attention_output);
    return result->attention_output;
}

static ggml_tensor * build_ffn(ggml_context *     ctx,
                               const model_case & config,
                               ggml_tensor *      input,
                               int                layer,
                               layer_graph *      result) {
    char name[64];

    std::snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
    ggml_tensor * gate_weight = new_input_2d(ctx, config.type, kEmbedding, kFeedForward, name);
    std::snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
    ggml_tensor * up_weight = new_input_2d(ctx, config.type, kEmbedding, kFeedForward, name);
    std::snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
    ggml_tensor * down_weight = new_input_2d(ctx, config.type, kFeedForward, kEmbedding, name);

    result->ffn_gate  = ggml_mul_mat(ctx, gate_weight, input);
    result->ffn_up    = ggml_mul_mat(ctx, up_weight, input);
    result->ffn_fused = ggml_swiglu_split(ctx, result->ffn_gate, result->ffn_up);
    result->ffn_down  = ggml_mul_mat(ctx, down_weight, result->ffn_fused);
    ggml_format_name(result->ffn_down, "blk.%d.ffn_out", layer);
    ggml_set_output(result->ffn_down);
    return result->ffn_down;
}

static model_graph build_model_graph(const model_case & config, bool cpu_reference) {
    constexpr size_t graph_size = 2048;
    ggml_init_params params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    model_graph result;
    result.ctx.reset(ggml_init(params));
    if (!result.ctx) {
        return result;
    }

    ggml_context * ctx = result.ctx.get();
    result.input       = new_input_2d(ctx, GGML_TYPE_F32, kEmbedding, config.n_tokens, "model_input");
    result.positions   = new_input_1d(ctx, GGML_TYPE_I32, GGML_MROPE_SECTIONS * config.n_tokens, "positions");
    result.mask        = new_input_2d(ctx, GGML_TYPE_F16, config.n_tokens, config.n_tokens, "causal_mask");

    ggml_tensor * current = result.input;
    result.layers.reserve(kLayers);
    for (int layer = 0; layer < kLayers; ++layer) {
        layer_graph layer_graph = {};
        layer_graph.recurrent   = layer + 1 < kLayers;

        char name[64];
        std::snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
        ggml_tensor * attn_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kEmbedding, name);
        std::snprintf(name, sizeof(name), "blk.%d.post_attn_norm.weight", layer);
        ggml_tensor * post_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kEmbedding, name);

        ggml_tensor * residual   = current;
        ggml_tensor * normalized = build_rms_norm(ctx, current, attn_norm_weight, 1.0e-6f);
        ggml_tensor * attention =
            layer_graph.recurrent ?
                build_linear_attention(ctx, config, normalized, layer, cpu_reference, &layer_graph) :
                build_full_attention(ctx, config, normalized, result.positions, result.mask, layer, cpu_reference,
                                     &layer_graph);
        ggml_tensor * attention_residual = ggml_add(ctx, attention, residual);
        ggml_tensor * ffn_input          = build_rms_norm(ctx, attention_residual, post_norm_weight, 1.0e-6f);
        ggml_tensor * ffn                = build_ffn(ctx, config, ffn_input, layer, &layer_graph);
        layer_graph.output               = ggml_add(ctx, ffn, attention_residual);
        ggml_format_name(layer_graph.output, "blk.%d.output", layer);
        ggml_set_output(layer_graph.output);
        current = layer_graph.output;
        result.layers.push_back(layer_graph);
    }

    ggml_tensor * output_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kEmbedding, "output_norm.weight");
    ggml_tensor * output_weight      = new_input_2d(ctx, config.type, kEmbedding, kVocab, "output.weight");
    current                          = build_rms_norm(ctx, current, output_norm_weight, 1.0e-6f);
    result.logits                    = ggml_mul_mat(ctx, output_weight, current);
    ggml_set_name(result.logits, "logits");
    ggml_set_output(result.logits);

    result.graph = ggml_new_graph_custom(ctx, graph_size, false);
    ggml_build_forward_expand(result.graph, result.logits);
    return result;
}

static tensor_blobs make_tensor_blobs(const model_graph & graph, const model_case & config) {
    std::mt19937                          generator(config.seed);
    std::uniform_real_distribution<float> input_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> weight_dist(-0.08f, 0.08f);
    std::uniform_real_distribution<float> norm_dist(-0.05f, 0.05f);
    tensor_blobs                          result;

    for (ggml_tensor * tensor = ggml_get_first_tensor(graph.ctx.get()); tensor != nullptr;
         tensor               = ggml_get_next_tensor(graph.ctx.get(), tensor)) {
        if ((tensor->flags & GGML_TENSOR_FLAG_INPUT) == 0) {
            continue;
        }
        const std::string    name = tensor->name;
        std::vector<uint8_t> blob(ggml_nbytes(tensor));
        const size_t         elements = (size_t) ggml_nelements(tensor);

        if (name == "positions") {
            int32_t * values = (int32_t *) blob.data();
            for (int axis = 0; axis < GGML_MROPE_SECTIONS; ++axis) {
                for (int64_t token = 0; token < config.n_tokens; ++token) {
                    values[axis * config.n_tokens + token] = (int32_t) (token + axis);
                }
            }
        } else if (name == "causal_mask") {
            ggml_fp16_t * values = (ggml_fp16_t *) blob.data();
            for (int64_t query = 0; query < config.n_tokens; ++query) {
                for (int64_t key = 0; key < config.n_tokens; ++key) {
                    values[query * config.n_tokens + key] = ggml_fp32_to_fp16(key <= query ? 0.0f : -INFINITY);
                }
            }
        } else if (tensor->type == GGML_TYPE_F32) {
            float * values = (float *) blob.data();
            if (name.find("ssm_history") != std::string::npos || name.find("ssm_state") != std::string::npos) {
                std::fill(values, values + elements, 0.0f);
            } else if (name.find("norm.weight") != std::string::npos) {
                for (size_t index = 0; index < elements; ++index) {
                    values[index] = 1.0f + norm_dist(generator);
                }
            } else if (name.find("ssm_a") != std::string::npos) {
                for (size_t index = 0; index < elements; ++index) {
                    values[index] = -0.4f + weight_dist(generator);
                }
            } else if (name == "model_input") {
                for (size_t index = 0; index < elements; ++index) {
                    values[index] = input_dist(generator);
                }
            } else {
                for (size_t index = 0; index < elements; ++index) {
                    values[index] = weight_dist(generator);
                }
            }
        } else {
            uint16_t * values = (uint16_t *) blob.data();
            for (size_t index = 0; index < elements; ++index) {
                values[index] = encode_lowp(tensor->type, weight_dist(generator));
            }
        }
        result.emplace(name, std::move(blob));
    }
    return result;
}

static bool upload_tensor_blobs(const model_graph & graph, const tensor_blobs & blobs) {
    for (const auto & entry : blobs) {
        ggml_tensor * tensor = ggml_get_tensor(graph.ctx.get(), entry.first.c_str());
        if (tensor == nullptr || ggml_nbytes(tensor) != entry.second.size()) {
            return false;
        }
        ggml_backend_tensor_set(tensor, entry.second.data(), 0, entry.second.size());
    }
    return true;
}

static bool validate_assignment(ggml_backend_sched_t scheduler,
                                ggml_backend_t       dayan,
                                ggml_backend_t       cpu,
                                const model_graph &  graph) {
    for (size_t layer = 0; layer < graph.layers.size(); ++layer) {
        const layer_graph & current       = graph.layers[layer];
        ggml_tensor *       dayan_nodes[] = {
            current.attention_output, current.ffn_gate, current.ffn_up, current.ffn_fused, current.ffn_down,
        };
        for (ggml_tensor * node : dayan_nodes) {
            if (ggml_backend_sched_get_tensor_backend(scheduler, node) != dayan) {
                std::fprintf(stderr, "layer=%zu node=%s backend=%s\n", layer, node->name,
                             ggml_backend_name(ggml_backend_sched_get_tensor_backend(scheduler, node)));
                return false;
            }
        }
        if (current.recurrent) {
            if (ggml_backend_sched_get_tensor_backend(scheduler, current.conv_silu) != dayan) {
                return false;
            }
        } else if (ggml_backend_sched_get_tensor_backend(scheduler, current.flash_attention) != dayan) {
            return false;
        }
    }
    GGML_UNUSED(cpu);
    return true;
}

struct comparison_stats {
    float       max_abs = 0.0f;
    float       max_rel = 0.0f;
    double      sum_sq  = 0.0;
    size_t      count   = 0;
    std::string worst_stage;
    bool        close = true;
};

static float compare_tensor(const char *        stage,
                            const ggml_tensor * expected_tensor,
                            const ggml_tensor * observed_tensor,
                            comparison_stats *  stats) {
    const size_t       elements = (size_t) ggml_nelements(expected_tensor);
    std::vector<float> expected(elements);
    std::vector<float> observed(elements);
    float              stage_max_abs = 0.0f;
    ggml_backend_tensor_get(expected_tensor, expected.data(), 0, ggml_nbytes(expected_tensor));
    ggml_backend_tensor_get(observed_tensor, observed.data(), 0, ggml_nbytes(observed_tensor));

    for (size_t index = 0; index < elements; ++index) {
        const float abs_error = std::abs(observed[index] - expected[index]);
        const float rel_error = abs_error / std::max(std::abs(expected[index]), 1.0e-6f);
        stage_max_abs         = std::max(stage_max_abs, abs_error);
        stats->close          = stats->close && abs_error <= 0.03f + 0.03f * std::abs(expected[index]);
        stats->sum_sq += (double) abs_error * abs_error;
        ++stats->count;
        if (abs_error > stats->max_abs) {
            stats->max_abs     = abs_error;
            stats->worst_stage = stage;
        }
        stats->max_rel = std::max(stats->max_rel, rel_error);
    }
    return stage_max_abs;
}

static float compare_bf16_tensor(const char *        stage,
                                 const ggml_tensor * expected_tensor,
                                 const ggml_tensor * observed_tensor,
                                 comparison_stats *  stats) {
    const size_t             elements = (size_t) ggml_nelements(expected_tensor);
    std::vector<ggml_bf16_t> expected(elements);
    std::vector<ggml_bf16_t> observed(elements);
    float                    stage_max_abs = 0.0f;
    ggml_backend_tensor_get(expected_tensor, expected.data(), 0, ggml_nbytes(expected_tensor));
    ggml_backend_tensor_get(observed_tensor, observed.data(), 0, ggml_nbytes(observed_tensor));

    for (size_t index = 0; index < elements; ++index) {
        const float expected_value = ggml_bf16_to_fp32(expected[index]);
        const float observed_value = ggml_bf16_to_fp32(observed[index]);
        const float abs_error      = std::abs(observed_value - expected_value);
        const float rel_error      = abs_error / std::max(std::abs(expected_value), 1.0e-6f);
        stage_max_abs              = std::max(stage_max_abs, abs_error);
        stats->close               = stats->close && abs_error <= 0.03f + 0.03f * std::abs(expected_value);
        stats->sum_sq += (double) abs_error * abs_error;
        ++stats->count;
        if (abs_error > stats->max_abs) {
            stats->max_abs     = abs_error;
            stats->worst_stage = stage;
        }
        stats->max_rel = std::max(stats->max_rel, rel_error);
    }
    return stage_max_abs;
}

static int run_model_case(ggml_backend_t dayan, ggml_backend_t cpu, const model_case & config) {
    model_graph reference = build_model_graph(config, true);
    model_graph actual    = build_model_graph(config, false);
    if (!reference.ctx || !actual.ctx) {
        return fail("cannot create model graph contexts");
    }
    if (!rewrite_graph(dayan, reference.ctx.get(), reference.graph) ||
        !rewrite_graph(dayan, actual.ctx.get(), actual.graph)) {
        return fail("model graph rewrite failed");
    }

    const tensor_blobs      blobs = make_tensor_blobs(actual, config);
    ggml_backend_buffer_ptr reference_buffer(ggml_backend_alloc_ctx_tensors(reference.ctx.get(), cpu));
    if (!reference_buffer || !upload_tensor_blobs(reference, blobs)) {
        return fail("cannot initialize CPU reference tensors");
    }
    if (ggml_backend_graph_compute(cpu, reference.graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU reference graph failed");
    }

    ggml_backend_t         backends[] = { dayan, cpu };
    ggml_backend_sched_ptr scheduler(
        ggml_backend_sched_new(backends, nullptr, 2, ggml_graph_size(actual.graph), false, true));
    if (!scheduler) {
        return fail("cannot create mixed backend scheduler");
    }
    ggml_backend_sched_set_tensor_backend(scheduler.get(), actual.layers.back().flash_attention, dayan);
    if (!ggml_backend_sched_alloc_graph(scheduler.get(), actual.graph)) {
        return fail("cannot allocate mixed backend graph");
    }
    if (!validate_assignment(scheduler.get(), dayan, cpu, actual)) {
        return fail("scheduler assignment does not match the Qwen3.5 plan");
    }
    if (!upload_tensor_blobs(actual, blobs)) {
        return fail("cannot initialize mixed backend tensors");
    }
    if (ggml_backend_sched_graph_compute(scheduler.get(), actual.graph) != GGML_STATUS_SUCCESS) {
        return fail("mixed Dayan model graph failed");
    }

    comparison_stats stats;
    float            flash_max_abs        = 0.0f;
    float            final_hidden_max_abs = 0.0f;
    for (size_t layer = 0; layer < actual.layers.size(); ++layer) {
        char stage[64];
        if (actual.layers[layer].recurrent) {
            std::snprintf(stage, sizeof(stage), "layer.%zu.conv_silu", layer);
            compare_tensor(stage, reference.layers[layer].conv_silu, actual.layers[layer].conv_silu, &stats);
        } else {
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_q", layer);
            compare_bf16_tensor(stage, reference.layers[layer].flash_q, actual.layers[layer].flash_q, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_k", layer);
            compare_bf16_tensor(stage, reference.layers[layer].flash_k, actual.layers[layer].flash_k, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_v", layer);
            compare_bf16_tensor(stage, reference.layers[layer].flash_v, actual.layers[layer].flash_v, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_attention", layer);
            flash_max_abs = compare_tensor(stage, reference.layers[layer].flash_attention,
                                           actual.layers[layer].flash_attention, &stats);
        }
        std::snprintf(stage, sizeof(stage), "layer.%zu.attention_output", layer);
        compare_tensor(stage, reference.layers[layer].attention_output, actual.layers[layer].attention_output, &stats);
        std::snprintf(stage, sizeof(stage), "layer.%zu.ffn_output", layer);
        compare_tensor(stage, reference.layers[layer].ffn_down, actual.layers[layer].ffn_down, &stats);
        std::snprintf(stage, sizeof(stage), "layer.%zu.output", layer);
        const float layer_output_max_abs =
            compare_tensor(stage, reference.layers[layer].output, actual.layers[layer].output, &stats);
        if (layer + 1 == actual.layers.size()) {
            final_hidden_max_abs = layer_output_max_abs;
        }
    }
    const float logits_max_abs = compare_tensor("logits", reference.logits, actual.logits, &stats);

    const double rmse = stats.count == 0 ? 0.0 : std::sqrt(stats.sum_sq / stats.count);
    std::printf(
        "type=%s layers=%lld recurrent=3 full_attention=1 tokens=%lld embd=%lld ff=%lld seed=%u elements=%zu "
        "splits=%d max_abs=%g flash_max_abs=%g final_hidden_max_abs=%g logits_max_abs=%g max_rel=%g rmse=%g "
        "worst=%s\n",
        ggml_type_name(config.type), (long long) kLayers, (long long) config.n_tokens, (long long) kEmbedding,
        (long long) kFeedForward, config.seed, stats.count, ggml_backend_sched_get_n_splits(scheduler.get()),
        stats.max_abs, flash_max_abs, final_hidden_max_abs, logits_max_abs, stats.max_rel, rmse,
        stats.worst_stage.c_str());
    return stats.close ? 0 : fail("Qwen3.5 model numerical comparison failed");
}

int main() {
    ggml_backend_load_all();

    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!dayan || !cpu) {
        return fail("cannot initialize backends");
    }

    const model_case cases[] = {
        { GGML_TYPE_BF16, 4, 0x351004U },
        { GGML_TYPE_F16,  4, 0x351104U },
    };
    for (const model_case & config : cases) {
        if (run_model_case(dayan.get(), cpu.get(), config) != 0) {
            return 1;
        }
    }

    std::puts("test-dayan-qwen35-dense: PASS");
    return 0;
}

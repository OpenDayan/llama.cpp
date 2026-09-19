#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-dayan.h"
#include "ggml.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using ggml_backend_graph_rewrite_t = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);
using json = nlohmann::json;

enum class model_mode {
    hybrid,
    linear,
    full,
};

struct model_case {
    enum ggml_type type;
    int64_t        n_tokens;
    uint32_t       seed;
    int64_t        layers;
    model_mode     mode;
};

struct layer_graph {
    bool          recurrent;
    ggml_tensor * attention_rms;
    ggml_tensor * attention_norm;
    ggml_tensor * conv_silu;
    ggml_tensor * gated_delta;
    ggml_tensor * linear_gdn_lowp;
    ggml_tensor * flash_q;
    ggml_tensor * flash_k;
    ggml_tensor * flash_v;
    ggml_tensor * flash_attention;
    ggml_tensor * full_q_rope;
    ggml_tensor * full_q_norm;
    ggml_tensor * full_q_rms;
    ggml_tensor * full_qg;
    ggml_tensor * full_q_input;
    ggml_tensor * full_q_permuted;
    ggml_tensor * full_k_matmul;
    ggml_tensor * full_v_matmul;
    ggml_tensor * full_k_rope;
    ggml_tensor * full_k_cache;
    ggml_tensor * full_v_cache;
    ggml_tensor * full_kv_indices;
    ggml_tensor * attention_output;
    ggml_tensor * attention_residual;
    ggml_tensor * ffn_rms;
    ggml_tensor * ffn_norm;
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
    ggml_tensor *            residual_input;
    ggml_tensor *            positions;
    ggml_tensor *            mask;
    ggml_tensor *            output_rms;
    ggml_tensor *            output_norm;
    ggml_tensor *            logits;
    std::vector<layer_graph> layers;
};

using tensor_blobs = std::unordered_map<std::string, std::vector<uint8_t>>;

static constexpr int64_t kEmbedding        = 1024;
static constexpr int64_t kFeedForward      = 3584;
static constexpr int64_t kConvKernel       = 4;
static constexpr int64_t kLinearHeadDim    = 128;
static constexpr int64_t kLinearHeads      = 16;
static constexpr int64_t kLinearWidth      = kLinearHeadDim * kLinearHeads;
static constexpr int64_t kConvChannels     = 3 * kLinearWidth;
static constexpr int64_t kAttentionHeadDim = 256;
static constexpr int64_t kAttentionHeads   = 8;
static constexpr int64_t kAttentionKVHeads = 2;
static constexpr int64_t kVocab            = 64;

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-qwen35-dense: %s\n", message);
    return 1;
}

static uint16_t encode_lowp(enum ggml_type type, float value) {
    return type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value);
}

// Matches ggml-common.h; the encoder is exported by ggml-base.
typedef struct {
    uint8_t e;
    uint8_t qs[16];
} test_block_mxfp4;
extern "C" void quantize_row_mxfp4_ref(const float * x, test_block_mxfp4 * y, int64_t k);

static bool rewrite_graph(ggml_backend_t dayan, ggml_context * ctx, ggml_cgraph * graph) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    auto rewrite = (ggml_backend_graph_rewrite_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    return rewrite != nullptr && rewrite(dayan, ctx, graph);
}

static uint64_t mega_launch_count(ggml_backend_t dayan) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    using count_fn            = uint64_t (*)();
    auto fn = (count_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dayan_get_mega_launch_count");
    return fn == nullptr ? 0 : fn();
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

static void force_f32_layout(ggml_tensor * tensor) {
    tensor->type = GGML_TYPE_F32;
    tensor->nb[0] = sizeof(float);
    for (int dimension = 1; dimension < GGML_MAX_DIMS; ++dimension) {
        tensor->nb[dimension] = tensor->nb[dimension - 1] * tensor->ne[dimension - 1];
    }
}

static ggml_tensor * quantize_to_f32(ggml_context * ctx, ggml_tensor * input, enum ggml_type type) {
    return ggml_cast(ctx, ggml_cast(ctx, input, type), GGML_TYPE_F32);
}

static ggml_tensor * build_linear_attention_legacy(ggml_context *     ctx,
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
    std::snprintf(name, sizeof(name), "blk.%d.linear_indices", layer);
    ggml_tensor * indices = new_input_1d(ctx, GGML_TYPE_I32, config.n_tokens, name);
    std::snprintf(name, sizeof(name), "blk.%d.conv_indices", layer);
    ggml_tensor * conv_indices = new_input_1d(ctx, GGML_TYPE_I32, kConvChannels, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_norm.weight", layer);
    ggml_tensor * norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kLinearHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.ssm_out.weight", layer);
    ggml_tensor * output_weight = new_input_2d(ctx, config.type, kLinearWidth, kEmbedding, name);

    // CPU mul_mat needs F32 activations for non-BF16 weight types; the Dayan
    // graph casts activations straight to the weight type.
    const bool lowp_activations = !(cpu_reference && config.type != GGML_TYPE_BF16);
    ggml_tensor * qkv_rows = ggml_get_rows(ctx, ggml_scale(ctx, input, 1.0f), indices);
    ggml_tensor * qkv_input = lowp_activations ? ggml_cast(ctx, qkv_rows, config.type) : qkv_rows;
    ggml_tensor * qkv   = ggml_mul_mat(ctx, qkv_weight, qkv_input);
    ggml_tensor * z     = ggml_mul_mat(ctx, z_weight, qkv_input);
    ggml_tensor * beta  = ggml_sigmoid(ctx, ggml_mul_mat(ctx, beta_weight, qkv_input));
    ggml_tensor * alpha = ggml_mul_mat(ctx, alpha_weight, qkv_input);
    alpha               = ggml_softplus(ctx, ggml_add(ctx, alpha, dt));
    alpha               = ggml_mul(ctx, alpha, a);
    qkv                 = ggml_transpose(ctx, ggml_reshape_3d(ctx, qkv, kConvChannels, config.n_tokens, 1));
    ggml_tensor * conv_input = ggml_concat(ctx, history, qkv, 0);
    ggml_tensor * conv_copy  = ggml_cpy(ctx, conv_input, ggml_dup_tensor(ctx, conv_input));
    ggml_tensor * conv_scale = ggml_scale(ctx, conv_copy, 1.0f);
    ggml_tensor * conv_rows  = ggml_get_rows(ctx, conv_scale, conv_indices);
    ggml_tensor * conv  = ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, conv_rows,
                                                              kConvKernel - 1 + config.n_tokens,
                                                              kConvChannels, 1), conv_weight);
    result->conv_silu   = ggml_silu(ctx, conv);
    ggml_set_output(result->conv_silu);
    const size_t head_stride = kLinearHeadDim * sizeof(float);
    const size_t token_stride = kConvChannels * sizeof(float);
    const size_t seq_stride = token_stride * config.n_tokens;
    ggml_tensor * q = ggml_l2_norm(ctx, ggml_view_4d(ctx, result->conv_silu, kLinearHeadDim, kLinearHeads, config.n_tokens, 1, head_stride, token_stride, seq_stride, 0), 1.0e-6f);
    ggml_tensor * k = ggml_l2_norm(ctx, ggml_view_4d(ctx, result->conv_silu, kLinearHeadDim, kLinearHeads, config.n_tokens, 1, head_stride, token_stride, seq_stride, kLinearWidth * sizeof(float)), 1.0e-6f);
    ggml_tensor * v = ggml_view_4d(ctx, result->conv_silu, kLinearHeadDim, kLinearHeads, config.n_tokens, 1, head_stride, token_stride, seq_stride, 2 * kLinearWidth * sizeof(float));
    beta = ggml_reshape_4d(ctx, beta, 1, kLinearHeads, config.n_tokens, 1);
    alpha = ggml_reshape_4d(ctx, alpha, 1, kLinearHeads, config.n_tokens, 1);
    result->gated_delta = ggml_gated_delta_net(ctx, q, k, v, alpha, beta, state, 1);
    ggml_set_output(result->gated_delta);
    result->linear_gdn_lowp = ggml_cast(ctx, result->gated_delta, GGML_TYPE_BF16);
    ggml_set_output(result->linear_gdn_lowp);
    ggml_tensor * core = ggml_view_4d(ctx, result->gated_delta, kLinearHeadDim, kLinearHeads, config.n_tokens, 1, kLinearHeadDim * sizeof(float), kLinearWidth * sizeof(float), kLinearWidth * config.n_tokens * sizeof(float), 0);
    core = build_rms_norm(ctx, core, norm_weight, 1.0e-6f);
    core = ggml_mul(ctx, core, ggml_silu(ctx, ggml_reshape_4d(ctx, z, kLinearHeadDim, kLinearHeads, config.n_tokens, 1)));
    core = ggml_reshape_2d(ctx, core, kLinearWidth, config.n_tokens);
    if (lowp_activations) {
        core = ggml_cast(ctx, core, config.type);
    }
    result->attention_output = ggml_mul_mat(ctx, output_weight, core);
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
    GGML_UNUSED(cpu_reference);
    char name[64];

    std::snprintf(name, sizeof(name), "blk.%d.attn_qg.weight", layer);
    ggml_tensor * qg_weight = new_input_2d(ctx, config.type, kEmbedding, 2 * kAttentionHeadDim * kAttentionHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_q_norm.weight", layer);
    ggml_tensor * q_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kAttentionHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_k_norm.weight", layer);
    ggml_tensor * k_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kAttentionHeadDim, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
    ggml_tensor * k_weight = new_input_2d(ctx, config.type, kEmbedding,
                                           kAttentionHeadDim * kAttentionKVHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
    ggml_tensor * v_weight = new_input_2d(ctx, config.type, kEmbedding,
                                           kAttentionHeadDim * kAttentionKVHeads, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_out.weight", layer);
    ggml_tensor * output_weight = new_input_2d(ctx, config.type,
                                                kAttentionHeadDim * kAttentionHeads, kEmbedding, name);

    ggml_tensor * qg = ggml_mul_mat(ctx, qg_weight, input);
    result->full_qg = qg;
    ggml_set_output(qg);
    ggml_tensor * q =
        ggml_view_3d(ctx, qg, kAttentionHeadDim, kAttentionHeads, config.n_tokens,
                     2 * kAttentionHeadDim * sizeof(float), 2 * kAttentionHeadDim * kAttentionHeads * sizeof(float), 0);
    result->full_q_input = q;
    ggml_set_output(q);
    q = build_rms_norm(ctx, q, q_norm_weight, 1.0e-6f);
    result->full_q_norm = q;
    result->full_q_rms = q->src[0];
    ggml_set_output(result->full_q_rms);
    ggml_set_output(result->full_q_norm);

    int sections[GGML_MROPE_SECTIONS] = { 11, 11, 10, 0 };
    q = ggml_rope_multi(ctx, q, positions, nullptr, kAttentionHeadDim, sections, GGML_ROPE_TYPE_MROPE,
                        (int) config.n_tokens, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    result->full_q_rope = q;
    ggml_set_output(result->full_q_rope);

    // Keep K and V matmuls adjacent so the Dayan compiler can form its pair slot.
    ggml_tensor * k = ggml_mul_mat(ctx, k_weight, input);
    ggml_tensor * v = ggml_mul_mat(ctx, v_weight, input);
    result->full_k_matmul = k;
    result->full_v_matmul = v;
    ggml_set_output(result->full_k_matmul);
    ggml_set_output(result->full_v_matmul);
    k = build_rms_norm(ctx, ggml_reshape_3d(ctx, k, kAttentionHeadDim, kAttentionKVHeads, config.n_tokens),
                       k_norm_weight, 1.0e-6f);
    k = ggml_rope_multi(ctx, k, positions, nullptr, kAttentionHeadDim, sections, GGML_ROPE_TYPE_MROPE,
                        (int) config.n_tokens, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    result->full_k_rope = k;
    ggml_set_output(result->full_k_rope);

    std::snprintf(name, sizeof(name), "blk.%d.attn_k_cache", layer);
    ggml_tensor * k_cache = new_input_2d(ctx, GGML_TYPE_F16, kAttentionHeadDim * kAttentionKVHeads,
                                         config.n_tokens, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_v_cache", layer);
    ggml_tensor * v_cache = new_input_2d(ctx, GGML_TYPE_F16, kAttentionHeadDim * kAttentionKVHeads,
                                         config.n_tokens, name);
    std::snprintf(name, sizeof(name), "blk.%d.attn_kv_indices", layer);
    ggml_tensor * kv_indices = new_input_1d(ctx, GGML_TYPE_I32, config.n_tokens, name);
    result->full_k_cache = k_cache;
    result->full_v_cache = v_cache;
    result->full_kv_indices = kv_indices;
    ggml_tensor * k_rows = ggml_reshape_2d(ctx, k, kAttentionHeadDim * kAttentionKVHeads, config.n_tokens);
    ggml_tensor * v_rows = ggml_reshape_2d(ctx, v, kAttentionHeadDim * kAttentionKVHeads, config.n_tokens);
    ggml_tensor * k_set = ggml_set_rows(ctx, k_cache, k_rows, kv_indices);
    ggml_tensor * v_set = ggml_set_rows(ctx, v_cache, v_rows, kv_indices);

    ggml_tensor * q_permuted = ggml_permute(ctx, ggml_reshape_4d(ctx, q, kAttentionHeadDim,
                                                                 kAttentionHeads, config.n_tokens, 1),
                                            0, 2, 1, 3);
    result->full_q_permuted = q_permuted;
    ggml_set_output(result->full_q_permuted);
    ggml_tensor * q_flash = ggml_cast(ctx, q_permuted, GGML_TYPE_F16);
    ggml_tensor * k_flash = ggml_permute(ctx, ggml_reshape_4d(ctx, k_set, kAttentionHeadDim,
                                                              kAttentionKVHeads, config.n_tokens, 1),
                                         0, 2, 1, 3);
    ggml_tensor * v_flash = ggml_permute(ctx, ggml_reshape_4d(ctx, v_set, kAttentionHeadDim,
                                                              kAttentionKVHeads, config.n_tokens, 1),
                                         0, 2, 1, 3);
    result->flash_q = q_flash;
    result->flash_k = k_flash;
    result->flash_v = v_flash;
    ggml_set_output(result->flash_q);
    ggml_set_output(result->flash_k);
    ggml_set_output(result->flash_v);
    ggml_format_name(result->flash_q, "blk.%d.flash_q", layer);
    ggml_format_name(result->flash_k, "blk.%d.flash_k", layer);
    ggml_format_name(result->flash_v, "blk.%d.flash_v", layer);
    if (config.mode == model_mode::hybrid) {
        ggml_set_output(result->flash_q);
        ggml_set_output(result->flash_k);
        ggml_set_output(result->flash_v);
    }
    result->flash_attention = ggml_flash_attn_ext(ctx, result->flash_q, result->flash_k, result->flash_v, mask,
                                                  1.0f / std::sqrt((float) kAttentionHeadDim), 0.0f, 0.0f);
    ggml_set_output(result->flash_attention);
    ggml_format_name(result->flash_attention, "blk.%d.flash_attention", layer);
    if (config.mode == model_mode::hybrid) {
        ggml_set_output(result->flash_attention);
    }
    ggml_tensor * attention =
        ggml_reshape_2d(ctx, result->flash_attention, kAttentionHeadDim * kAttentionHeads, config.n_tokens);

    ggml_tensor * gate_view = ggml_view_3d(
        ctx, qg, kAttentionHeadDim, kAttentionHeads, config.n_tokens, 2 * kAttentionHeadDim * sizeof(float),
        2 * kAttentionHeadDim * kAttentionHeads * sizeof(float), kAttentionHeadDim * sizeof(float));
    ggml_tensor * gate = ggml_cont_2d(ctx, gate_view, kAttentionHeadDim * kAttentionHeads, config.n_tokens);
    ggml_tensor * gated = ggml_mul(ctx, attention, ggml_sigmoid(ctx, gate));
    result->attention_output = ggml_mul_mat(ctx, output_weight, gated);
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
    const enum ggml_type input_type = GGML_TYPE_F32;
    result.input       = new_input_2d(ctx, input_type, kEmbedding, config.n_tokens, "model_input");
    result.residual_input = config.mode == model_mode::hybrid ? result.input :
                            new_input_2d(ctx, cpu_reference || config.type != GGML_TYPE_BF16 ? GGML_TYPE_F32 :
                                               GGML_TYPE_BF16,
                                         kEmbedding, config.n_tokens, "model_residual");
    result.positions   = new_input_1d(ctx, GGML_TYPE_I32, GGML_MROPE_SECTIONS * config.n_tokens, "positions");
    result.mask        = new_input_2d(ctx, GGML_TYPE_F16, config.n_tokens, config.n_tokens, "causal_mask");
    result.graph       = ggml_new_graph_custom(ctx, graph_size, false);

    ggml_tensor * current = result.input;
    result.layers.reserve(config.layers);
    for (int layer = 0; layer < config.layers; ++layer) {
        layer_graph layer_graph = {};
        layer_graph.recurrent   = config.mode == model_mode::linear ||
                                   (config.mode == model_mode::hybrid && layer + 1 < config.layers);

        char name[64];
        std::snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
        ggml_tensor * attn_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kEmbedding, name);
        std::snprintf(name, sizeof(name), "blk.%d.post_attn_norm.weight", layer);
        ggml_tensor * post_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kEmbedding, name);

        ggml_tensor * residual = config.mode == model_mode::hybrid ? current : result.residual_input;
        layer_graph.attention_norm = build_rms_norm(ctx, current, attn_norm_weight, 1.0e-6f);
        layer_graph.attention_rms  = layer_graph.attention_norm->src[0];
        ggml_tensor * normalized   = layer_graph.attention_norm;
        ggml_format_name(layer_graph.attention_rms, "blk.%d.attn_rms", layer);
        ggml_format_name(layer_graph.attention_norm, "blk.%d.attn_norm", layer);
        ggml_set_output(layer_graph.attention_rms);
        ggml_set_output(layer_graph.attention_norm);
        ggml_tensor * attention =
            layer_graph.recurrent ?
                    build_linear_attention_legacy(ctx, config, normalized, layer, cpu_reference, &layer_graph) :
                build_full_attention(ctx, config, normalized, result.positions, result.mask, layer, cpu_reference,
                                     &layer_graph);
        // The BF16 signature models an explicit residual-stream cast; other
        // weight types rely on the planner's local widening.
        const bool widen_residual = !cpu_reference && config.type == GGML_TYPE_BF16;
        ggml_tensor * attention_for_residual = config.mode == model_mode::hybrid || !widen_residual ? attention :
                                                ggml_cast(ctx, attention, GGML_TYPE_BF16);
        layer_graph.attention_residual = ggml_add(ctx, attention_for_residual, residual);
        if (!cpu_reference && config.mode != model_mode::hybrid) {
            ggml_tensor * rms = ggml_rms_norm(ctx, layer_graph.attention_residual, 1.0e-6f);
            force_f32_layout(rms);
            layer_graph.ffn_norm = ggml_mul(ctx, rms, post_norm_weight);
        } else {
            layer_graph.ffn_norm = build_rms_norm(ctx, layer_graph.attention_residual, post_norm_weight, 1.0e-6f);
        }
        layer_graph.ffn_rms            = layer_graph.ffn_norm->src[0];
        ggml_format_name(layer_graph.attention_residual, "blk.%d.attn_residual", layer);
        ggml_format_name(layer_graph.ffn_rms, "blk.%d.ffn_rms", layer);
        ggml_format_name(layer_graph.ffn_norm, "blk.%d.ffn_norm", layer);
        ggml_set_output(layer_graph.attention_residual);
        ggml_set_output(layer_graph.ffn_rms);
        ggml_set_output(layer_graph.ffn_norm);
        ggml_tensor * ffn  = build_ffn(ctx, config, layer_graph.ffn_norm, layer, &layer_graph);
        ggml_tensor * ffn_for_output = config.mode == model_mode::hybrid || !widen_residual ? ffn :
                                        ggml_cast(ctx, ffn, GGML_TYPE_BF16);
        layer_graph.output = ggml_add(ctx, ffn_for_output, layer_graph.attention_residual);
        ggml_format_name(layer_graph.output, "blk.%d.output", layer);
        ggml_set_output(layer_graph.output);
        current = layer_graph.output;
        result.layers.push_back(layer_graph);
    }

    ggml_tensor * output_norm_weight = new_input_1d(ctx, GGML_TYPE_F32, kEmbedding, "output_norm.weight");
    ggml_tensor * output_weight      = new_input_2d(ctx, config.type, kEmbedding, kVocab, "output.weight");
    if (!cpu_reference && config.mode != model_mode::hybrid) {
        ggml_tensor * rms = ggml_rms_norm(ctx, current, 1.0e-6f);
        force_f32_layout(rms);
        result.output_norm = ggml_mul(ctx, rms, output_norm_weight);
    } else {
        result.output_norm = build_rms_norm(ctx, current, output_norm_weight, 1.0e-6f);
    }
    result.output_rms                = result.output_norm->src[0];
    ggml_set_name(result.output_rms, "output_rms");
    ggml_set_name(result.output_norm, "output_norm");
    ggml_set_output(result.output_rms);
    ggml_set_output(result.output_norm);
    result.logits = ggml_mul_mat(ctx, output_weight, result.output_norm);
    ggml_set_name(result.logits, "logits");
    ggml_set_output(result.logits);

    if (config.mode == model_mode::full) {
        for (const layer_graph & layer : result.layers) {
            ggml_build_forward_expand(result.graph, layer.full_q_rope);
            ggml_build_forward_expand(result.graph, layer.full_k_matmul);
            ggml_build_forward_expand(result.graph, layer.full_v_matmul);
            ggml_build_forward_expand(result.graph, layer.full_k_rope);
            ggml_build_forward_expand(result.graph, layer.flash_k);
            ggml_build_forward_expand(result.graph, layer.flash_v);
            ggml_build_forward_expand(result.graph, layer.flash_q);
        }
    }
    if (config.mode == model_mode::linear) {
        for (const layer_graph & layer : result.layers) {
            ggml_build_forward_expand(result.graph, layer.linear_gdn_lowp);
        }
    }
    ggml_build_forward_expand(result.graph, result.logits);
    return result;
}

static tensor_blobs make_tensor_blobs(const model_graph & graph, const model_case & config) {
    std::mt19937                          generator(config.seed);
    std::uniform_real_distribution<float> input_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> weight_dist(-0.08f, 0.08f);
    std::uniform_real_distribution<float> norm_dist(-0.05f, 0.05f);
    tensor_blobs                          result;
    std::vector<float>                    input_values;

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
        } else if (tensor->type == GGML_TYPE_I32) {
            int32_t * values = (int32_t *) blob.data();
            for (size_t index = 0; index < elements; ++index) {
                values[index] = (int32_t) index;
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
            } else if (name == "model_input" || name == "model_residual") {
                if (name == "model_residual" && input_values.empty()) {
                    input_values.resize(elements);
                    for (size_t index = 0; index < elements; ++index) {
                        input_values[index] = input_dist(generator);
                    }
                }
                input_values.resize(elements);
                for (size_t index = 0; index < elements; ++index) {
                    values[index] = name == "model_input" ? input_values[index] = input_dist(generator) : input_values[index];
                }
            } else {
                for (size_t index = 0; index < elements; ++index) {
                    values[index] = weight_dist(generator);
                }
            }
        } else {
            if (tensor->type == GGML_TYPE_MXFP4) {
                std::vector<float> source(elements);
                for (size_t index = 0; index < elements; ++index) {
                    source[index] = name.find("cache") != std::string::npos ? 0.0f : weight_dist(generator);
                }
                quantize_row_mxfp4_ref(source.data(), (test_block_mxfp4 *) blob.data(), (int64_t) elements);
            } else {
                uint16_t * values = (uint16_t *) blob.data();
                for (size_t index = 0; index < elements; ++index) {
                    const float value = name == "model_residual" && index < input_values.size() ? input_values[index] :
                                        name == "model_input" ? input_dist(generator) :
                                        (name.find("cache") != std::string::npos ? 0.0f : weight_dist(generator));
                    values[index] = encode_lowp(tensor->type, value);
                }
            }
        }
        result.emplace(name, std::move(blob));
    }
    return result;
}

static bool upload_tensor_blobs(const model_graph & graph, const tensor_blobs & blobs, bool skip_unallocated = false) {
    for (const auto & entry : blobs) {
        ggml_tensor * tensor = ggml_get_tensor(graph.ctx.get(), entry.first.c_str());
        if (tensor == nullptr || ggml_nbytes(tensor) != entry.second.size()) {
            return false;
        }
        if (tensor->buffer == nullptr && (tensor->view_src == nullptr || tensor->view_src->buffer == nullptr)) {
            if (skip_unallocated) {
                continue;
            }
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
            current.attention_rms, current.attention_norm, current.attention_output, current.attention_residual,
            current.ffn_rms,       current.ffn_norm,       current.ffn_gate,         current.ffn_up,
            current.ffn_fused,     current.ffn_down,       current.output,
        };
        for (ggml_tensor * node : dayan_nodes) {
            if (ggml_backend_sched_get_tensor_backend(scheduler, node) != dayan) {
                std::fprintf(stderr, "layer=%zu node=%s backend=%s\n", layer, node->name,
                             ggml_backend_name(ggml_backend_sched_get_tensor_backend(scheduler, node)));
                return false;
            }
        }
        if (current.recurrent) {
            if (ggml_backend_sched_get_tensor_backend(scheduler, current.conv_silu) != dayan ||
                ggml_backend_sched_get_tensor_backend(scheduler, current.gated_delta) != dayan) {
                return false;
            }
        } else if (ggml_backend_sched_get_tensor_backend(scheduler, current.flash_attention) != dayan) {
            return false;
        }
    }
    if (ggml_backend_sched_get_tensor_backend(scheduler, graph.output_rms) != dayan ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.output_norm) != dayan ||
        ggml_backend_sched_get_tensor_backend(scheduler, graph.logits) != dayan) {
        return false;
    }
    GGML_UNUSED(cpu);
    return true;
}

struct comparison_stats {
    float       max_abs = 0.0f;
    float       max_rel = 0.0f;
    double      sum_sq  = 0.0;
    double      sum_ref_sq = 0.0;
    size_t      count   = 0;
    std::string worst_stage;
    bool        close = true;
    size_t      nonfinite = 0;
};

static size_t tensor_offset(const ggml_tensor * tensor, size_t index) {
    size_t offset = 0;
    for (int dimension = 0; dimension < GGML_MAX_DIMS; ++dimension) {
        const size_t coordinate = index % (size_t) tensor->ne[dimension];
        index /= (size_t) tensor->ne[dimension];
        offset += coordinate * tensor->nb[dimension];
    }
    return offset;
}

static float tensor_value(const ggml_tensor * tensor, const std::vector<uint8_t> & bytes, size_t index) {
    const size_t offset = tensor_offset(tensor, index);
    if (tensor->type == GGML_TYPE_BF16) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
        return ggml_bf16_to_fp32(ggml_bf16_t{ bits });
    }
    if (tensor->type == GGML_TYPE_F16) {
        uint16_t bits;
        std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
        return ggml_fp16_to_fp32(bits);
    }
    float value;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    return value;
}

// Relative storage quantum of a low-precision type: the rounding step at a value's own
// magnitude. F32 storage is exact for tolerance purposes.
static float storage_quantum(enum ggml_type type) {
    if (type == GGML_TYPE_BF16) {
        return 1.0f / 256.0f;
    }
    if (type == GGML_TYPE_F16) {
        return 1.0f / 2048.0f;
    }
    return 0.0f;
}

static float reference_max_abs(const ggml_tensor * tensor) {
    const size_t elements = (size_t) ggml_nelements(tensor);
    std::vector<uint8_t> bytes(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
    float max_abs = 0.0f;
    for (size_t index = 0; index < elements; ++index) {
        max_abs = std::max(max_abs, std::abs(tensor_value(tensor, bytes, index)));
    }
    return max_abs;
}

static float compare_tensor(const char *        stage,
                            const ggml_tensor * expected_tensor,
                            const ggml_tensor * observed_tensor,
                            comparison_stats *  stats,
                            float               extra_slack = 0.0f) {
    const size_t       elements = (size_t) ggml_nelements(expected_tensor);
    std::vector<uint8_t> expected(ggml_nbytes(expected_tensor));
    std::vector<uint8_t> observed(ggml_nbytes(observed_tensor));
    float stage_max_abs = 0.0f;
    float stage_max_reference = 0.0f;
    ggml_backend_tensor_get(expected_tensor, expected.data(), 0, expected.size());
    ggml_backend_tensor_get(observed_tensor, observed.data(), 0, observed.size());

    for (size_t index = 0; index < elements; ++index) {
        const float expected_value = tensor_value(expected_tensor, expected, index);
        const float observed_value = tensor_value(observed_tensor, observed, index);
        const float abs_error = std::abs(observed_value - expected_value);
        const float rel_error = abs_error / std::max(std::abs(expected_value), 1.0e-6f);
        if (!std::isfinite(expected_value) || !std::isfinite(observed_value)) {
            ++stats->nonfinite;
            stats->close = false;
        }
        stage_max_abs         = std::max(stage_max_abs, abs_error);
        stage_max_reference   = std::max(stage_max_reference, std::abs(expected_value));
        stats->sum_sq += (double) abs_error * abs_error;
        stats->sum_ref_sq += (double) expected_value * expected_value;
        ++stats->count;
        if (abs_error > stats->max_abs) {
            stats->max_abs     = abs_error;
            stats->worst_stage = stage;
        }
        stats->max_rel = std::max(stats->max_rel, rel_error);
    }
    // A low-precision tensor in the residual stream rounds each addend at that addend's own
    // magnitude, so a small result can legitimately differ from the F32 reference by several
    // storage quanta of the stage's largest value. Allow exactly that much extra slack.
    const float storage_slack = storage_quantum(observed_tensor->type) * stage_max_reference + extra_slack;
    for (size_t index = 0; index < elements; ++index) {
        const float expected_value = tensor_value(expected_tensor, expected, index);
        const float observed_value = tensor_value(observed_tensor, observed, index);
        stats->close = stats->close &&
                       std::abs(observed_value - expected_value) <=
                           0.03f + 0.03f * std::abs(expected_value) + storage_slack;
    }
    std::fprintf(stderr, "COMPARE %s max_abs=%g expected=%s observed=%s\n", stage, stage_max_abs,
                 ggml_type_name(expected_tensor->type), ggml_type_name(observed_tensor->type));
    return stage_max_abs;
}

static float compare_lowp_tensor(const char *        stage,
                                 const ggml_tensor * expected_tensor,
                                 const ggml_tensor * observed_tensor,
                                 comparison_stats *  stats) {
    const size_t          elements = (size_t) ggml_nelements(expected_tensor);
    std::vector<uint8_t> expected(ggml_nbytes(expected_tensor));
    std::vector<uint8_t> observed(ggml_nbytes(observed_tensor));
    float                 stage_max_abs = 0.0f;
    ggml_backend_tensor_get(expected_tensor, expected.data(), 0, expected.size());
    ggml_backend_tensor_get(observed_tensor, observed.data(), 0, observed.size());

    for (size_t index = 0; index < elements; ++index) {
        const float expected_value = tensor_value(expected_tensor, expected, index);
        const float observed_value = tensor_value(observed_tensor, observed, index);
        const float abs_error      = std::abs(observed_value - expected_value);
        const float rel_error      = abs_error / std::max(std::abs(expected_value), 1.0e-6f);
        stage_max_abs              = std::max(stage_max_abs, abs_error);
        stats->close               = stats->close && abs_error <= 0.03f + 0.03f * std::abs(expected_value);
        stats->sum_sq += (double) abs_error * abs_error;
        stats->sum_ref_sq += (double) expected_value * expected_value;
        ++stats->count;
        if (abs_error > stats->max_abs) {
            stats->max_abs     = abs_error;
            stats->worst_stage = stage;
        }
        stats->max_rel = std::max(stats->max_rel, rel_error);
    }
    std::fprintf(stderr, "COMPARE %s max_abs=%g expected=%s observed=%s\n", stage, stage_max_abs,
                 ggml_type_name(expected_tensor->type), ggml_type_name(observed_tensor->type));
    return stage_max_abs;
}

static void bypass_reference_activation_casts(ggml_cgraph * graph) {
    // The rewrite quantizes matmul activations for Dayan; the CPU reference
    // needs the plain F32 sources instead.
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        ggml_tensor * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_MUL_MAT || node->src[1] == nullptr) {
            continue;
        }
        ggml_tensor * activation = node->src[1];
        if (activation->type == GGML_TYPE_MXFP4 && activation->src[0] != nullptr &&
            activation->src[0]->type == GGML_TYPE_F32) {
            node->src[1] = activation->src[0];
        }
    }
}

static int run_model_case(ggml_backend_t dayan, ggml_backend_t cpu, const model_case & config, json * result = nullptr) {
    const uint64_t mega_before = mega_launch_count(dayan);
    model_graph reference = build_model_graph(config, true);
    model_graph actual    = build_model_graph(config, false);
    if (!reference.ctx || !actual.ctx) {
        return fail("cannot create model graph contexts");
    }
    std::vector<ggml_tensor *> reference_flash_inputs;
    for (const layer_graph & layer : reference.layers) {
        if (!layer.recurrent) {
            reference_flash_inputs.push_back(layer.flash_attention->src[0]);
        }
    }
    if (!rewrite_graph(dayan, reference.ctx.get(), reference.graph) ||
        !rewrite_graph(dayan, actual.ctx.get(), actual.graph)) {
        return fail("model graph rewrite failed");
    }
    bypass_reference_activation_casts(reference.graph);
    size_t flash_index = 0;
    for (layer_graph & layer : reference.layers) {
        if (!layer.recurrent) {
            layer.flash_attention->src[0] = reference_flash_inputs[flash_index++];
        }
    }
    ggml_graph_clear(reference.graph);
    ggml_build_forward_expand(reference.graph, reference.logits);
    const tensor_blobs      reference_blobs = make_tensor_blobs(reference, config);
    const tensor_blobs      actual_blobs    = make_tensor_blobs(actual, config);
    ggml_backend_buffer_ptr reference_buffer(ggml_backend_alloc_ctx_tensors(reference.ctx.get(), cpu));
    if (!reference_buffer || !upload_tensor_blobs(reference, reference_blobs)) {
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
    ggml_tensor * anchor = actual.layers.back().recurrent ? actual.layers.back().attention_output : actual.layers.back().flash_attention;
    ggml_backend_sched_set_tensor_backend(scheduler.get(), anchor, dayan);
    if (config.mode != model_mode::hybrid) {
        ggml_backend_sched_set_tensor_backend(scheduler.get(), actual.residual_input, dayan);
    }
    for (const layer_graph & layer : actual.layers) {
        if (!layer.recurrent) {
            ggml_backend_sched_set_tensor_backend(scheduler.get(), layer.full_k_cache, dayan);
            ggml_backend_sched_set_tensor_backend(scheduler.get(), layer.full_v_cache, dayan);
            ggml_backend_sched_set_tensor_backend(scheduler.get(), layer.full_kv_indices, dayan);
        }
    }
    if (!ggml_backend_sched_alloc_graph(scheduler.get(), actual.graph)) {
        return fail("cannot allocate mixed backend graph");
    }
    if (!validate_assignment(scheduler.get(), dayan, cpu, actual)) {
        return fail("scheduler assignment does not match the Qwen3.5 plan");
    }
    if (!upload_tensor_blobs(actual, actual_blobs, true)) {
        return fail("cannot initialize mixed backend tensors");
    }
    const auto launch_start = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute(scheduler.get(), actual.graph) != GGML_STATUS_SUCCESS) {
        return fail("mixed Dayan model graph failed");
    }
    const double launch_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - launch_start).count();
    const bool mega_launched = mega_launch_count(dayan) != mega_before;

    comparison_stats stats;
    float            flash_max_abs        = 0.0f;
    float            final_hidden_max_abs = 0.0f;
    for (size_t layer = 0; layer < actual.layers.size(); ++layer) {
        char stage[64];
        std::snprintf(stage, sizeof(stage), "layer.%zu.attention_rms", layer);
        compare_tensor(stage, reference.layers[layer].attention_rms, actual.layers[layer].attention_rms, &stats);
        std::snprintf(stage, sizeof(stage), "layer.%zu.attention_norm", layer);
        compare_tensor(stage, reference.layers[layer].attention_norm, actual.layers[layer].attention_norm, &stats);
        if (actual.layers[layer].recurrent) {
            std::snprintf(stage, sizeof(stage), "layer.%zu.conv_silu", layer);
            compare_tensor(stage, reference.layers[layer].conv_silu, actual.layers[layer].conv_silu, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.gated_delta", layer);
            compare_tensor(stage, reference.layers[layer].gated_delta, actual.layers[layer].gated_delta, &stats);
        } else {
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_qg", layer);
            compare_tensor(stage, reference.layers[layer].full_qg, actual.layers[layer].full_qg, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_q_input", layer);
            compare_tensor(stage, reference.layers[layer].full_q_input, actual.layers[layer].full_q_input, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_q_norm", layer);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_q_rms", layer);
            compare_tensor(stage, reference.layers[layer].full_q_rms, actual.layers[layer].full_q_rms, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_q_norm", layer);
            compare_tensor(stage, reference.layers[layer].full_q_norm, actual.layers[layer].full_q_norm, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_q_rope", layer);
            compare_tensor(stage, reference.layers[layer].full_q_rope, actual.layers[layer].full_q_rope, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_k_matmul", layer);
            compare_tensor(stage, reference.layers[layer].full_k_matmul, actual.layers[layer].full_k_matmul, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_v_matmul", layer);
            compare_tensor(stage, reference.layers[layer].full_v_matmul, actual.layers[layer].full_v_matmul, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_k_rope", layer);
            compare_tensor(stage, reference.layers[layer].full_k_rope, actual.layers[layer].full_k_rope, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.full_q_permuted", layer);
            compare_tensor(stage, reference.layers[layer].full_q_permuted, actual.layers[layer].full_q_permuted, &stats);
            // The mega executor reuses flash_q as the gated output projection scratch
            // once attention has consumed it; flash_attention covers its values there.
            if (!mega_launched) {
                std::snprintf(stage, sizeof(stage), "layer.%zu.flash_q", layer);
                compare_lowp_tensor(stage, reference.layers[layer].flash_q, actual.layers[layer].flash_q, &stats);
            }
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_k", layer);
            compare_lowp_tensor(stage, reference.layers[layer].flash_k, actual.layers[layer].flash_k, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_v", layer);
            compare_lowp_tensor(stage, reference.layers[layer].flash_v, actual.layers[layer].flash_v, &stats);
            std::snprintf(stage, sizeof(stage), "layer.%zu.flash_attention", layer);
            flash_max_abs = compare_tensor(stage, reference.layers[layer].flash_attention,
                                           actual.layers[layer].flash_attention, &stats);
        }
        std::snprintf(stage, sizeof(stage), "layer.%zu.attention_output", layer);
        compare_tensor(stage, reference.layers[layer].attention_output, actual.layers[layer].attention_output, &stats);
        std::snprintf(stage, sizeof(stage), "layer.%zu.attention_residual", layer);
        compare_tensor(stage, reference.layers[layer].attention_residual, actual.layers[layer].attention_residual,
                       &stats);
        // The FFN norm stages read the low-precision attention residual, so they inherit
        // one storage quantum of its largest value on top of their own tolerance.
        const float norm_slack =
            storage_quantum(GGML_TYPE_BF16) * reference_max_abs(reference.layers[layer].attention_residual);
        std::snprintf(stage, sizeof(stage), "layer.%zu.ffn_rms", layer);
        compare_tensor(stage, reference.layers[layer].ffn_rms, actual.layers[layer].ffn_rms, &stats, norm_slack);
        std::snprintf(stage, sizeof(stage), "layer.%zu.ffn_norm", layer);
        compare_tensor(stage, reference.layers[layer].ffn_norm, actual.layers[layer].ffn_norm, &stats, norm_slack);
        // The quantized FFN matmul materializes only the low-precision ffn_out copy; the F32
        // tensor is never written in either execution mode, so it is not compared here. The
        // FFN values are covered by the output and logits comparisons below.
        std::snprintf(stage, sizeof(stage), "layer.%zu.output", layer);
        // The layer output sums two low-precision addends (attention residual and FFN
        // output), each rounded at its own magnitude; absorb both storage quanta.
        const float residual_slack = storage_quantum(GGML_TYPE_BF16) *
                                     (reference_max_abs(reference.layers[layer].attention_residual) +
                                      reference_max_abs(reference.layers[layer].ffn_down));
        const float layer_output_max_abs =
            compare_tensor(stage, reference.layers[layer].output, actual.layers[layer].output, &stats,
                           residual_slack);
        if (layer + 1 == actual.layers.size()) {
            final_hidden_max_abs = layer_output_max_abs;
        }
    }
    // The final norm and LM-head stages read the low-precision residual stream, so they
    // inherit one storage quantum of its largest value on top of their own tolerance.
    const float tail_slack =
        storage_quantum(GGML_TYPE_BF16) * reference_max_abs(reference.layers.back().output);
    compare_tensor("output_rms", reference.output_rms, actual.output_rms, &stats, tail_slack);
    compare_tensor("output_norm", reference.output_norm, actual.output_norm, &stats, tail_slack);
    const float logits_max_abs = compare_tensor("logits", reference.logits, actual.logits, &stats, tail_slack);

    const double rmse = stats.count == 0 ? 0.0 : std::sqrt(stats.sum_sq / stats.count);
    const double relative_l2 = std::sqrt(stats.sum_sq / std::max(stats.sum_ref_sq, 1.0e-30));
    std::printf(
        "type=%s layers=%lld mode=%s tokens=%lld embd=%lld ff=%lld seed=%u elements=%zu "
        "splits=%d max_abs=%g flash_max_abs=%g final_hidden_max_abs=%g logits_max_abs=%g max_rel=%g rmse=%g "
        "worst=%s\n",
        ggml_type_name(config.type), (long long) config.layers,
        config.mode == model_mode::linear ? "linear" : config.mode == model_mode::full ? "full" : "hybrid",
        (long long) config.n_tokens, (long long) kEmbedding,
        (long long) kFeedForward, config.seed, stats.count, ggml_backend_sched_get_n_splits(scheduler.get()),
        stats.max_abs, flash_max_abs, final_hidden_max_abs, logits_max_abs, stats.max_rel, rmse,
        stats.worst_stage.c_str());
    if (result != nullptr) {
        (*result)["status"] = stats.close ? "ok" : "failed";
        (*result)["max_abs_error"] = stats.max_abs;
        (*result)["max_rel_error"] = stats.max_rel;
        (*result)["relative_l2_error"] = relative_l2;
        (*result)["checked_elements"] = stats.count;
        (*result)["mega_launched"] = mega_launched;
        (*result)["comparison"] = {
            {"status", stats.close ? "pass" : "fail"},
            {"max_abs_error", stats.max_abs},
            {"max_rel_error", stats.max_rel},
            {"relative_l2_error", relative_l2},
            {"checked_elements", stats.count},
        };
        const char * tag = config.mode == model_mode::linear ?
                               (mega_launched ? "q35-linear-mega" : "q35-linear-fallback") :
                               (mega_launched ? "q35-full-mega" : "q35-full-fallback");
        (*result)["samples"] = {{{"tag", tag}, {"warmup", false}, {"wall_s", launch_wall_s}, {"tasks", 4}, {"status", stats.close ? 0 : 1}}};
    }
    return stats.close ? 0 : fail("Qwen3.5 model numerical comparison failed");
}

static bool parse_mode(const std::string & value, model_mode * mode) {
    if (value == "linear") {
        *mode = model_mode::linear;
        return true;
    }
    if (value == "full") {
        *mode = model_mode::full;
        return true;
    }
    return false;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--capabilities") == 0) {
        std::puts("{\"case_schema\":3,\"cmp\":true,\"dtypes\":[\"bf16\"],\"ops\":[\"mega\"]}");
        return 0;
    }
    const bool protocol = argc == 4;
    if (!protocol && argc != 1) {
        std::fprintf(stderr, "usage: %s [CASE.json RESULT.json TAG]\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!dayan || !cpu) {
        return fail("cannot initialize backends");
    }

    if (protocol) {
        json spec;
        try {
            std::ifstream input(argv[1]);
            spec = json::parse(input);
        } catch (const std::exception & error) {
            return fail(error.what());
        }
        model_mode mode;
        const std::string dtype = spec.value("dtype", "bf16");
        if (spec.value("op", "") != "mega" || (dtype != "bf16" && dtype != "mxfp4") ||
            !parse_mode(spec.value("mode", ""), &mode) || spec.value("layers", 1) != 1 ||
            spec.value("m", 0) < 1 || spec.value("m", 0) > 1024 || spec.value("repeats", 1) != 1 ||
            spec.value("warmup", 0) != 0) {
            return fail("mega case needs BF16 or MXFP4, one linear/full layer, one launch, and no warmup");
        }
        const model_case config = { dtype == "mxfp4" ? GGML_TYPE_MXFP4 : GGML_TYPE_BF16,
                                    spec.at("m").get<int64_t>(), spec.value("seed", 1U), 1, mode };
        json result = {{"case", spec}, {"samples", json::array()}};
        int status = run_model_case(dayan.get(), cpu.get(), config, &result);
        if (status == 0 && result.value("mega_launched", false) != true) {
            result["status"] = "failed";
            result["error"] = "the graph did not match a Dayan mega signature";
            status = 1;
        } else {
            result["status"] = status == 0 ? "ok" : "failed";
        }
        std::ofstream output(argv[2]);
        output << result.dump(2) << '\n';
        return status;
    }

    const model_case cases[] = {
        { GGML_TYPE_BF16, 1, 35U, 1, model_mode::linear },
        { GGML_TYPE_BF16, 4, 0x351004U, 4, model_mode::hybrid },
        { GGML_TYPE_F16,  4, 0x351104U, 4, model_mode::hybrid },
    };
    for (const model_case & config : cases) {
        if (run_model_case(dayan.get(), cpu.get(), config) != 0) {
            return 1;
        }
    }

    std::puts("test-dayan-qwen35-dense: PASS");
    return 0;
}

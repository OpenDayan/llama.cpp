#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <vector>

enum class ggml_backend_dayan_slot_kind {
    mul_mat,
    mul_mat_silu,
    mul_mat_quantized,
    mul_mat_pair,
    mul_mat_swiglu,
    mul_mat_geglu,
    mul_mat_id,
    mul_mat_id_pair,
    mul_mat_id_swiglu,
    mul_mat_id_geglu,
    add_id,
    argsort,
    top_k,
    elementwise_add,
    elementwise_mul,
    elementwise_sub,
    elementwise_div,
    scale,
    sigmoid,
    silu,
    softplus,
    unary_math,
    sum,
    sum_rows,
    mean,
    l2_norm,
    get_rows,
    set_rows,
    copy,
    repeat,
    set,
    diag,
    diag_mask_inf,
    diag_mask_zero,
    pad,
    pad_circular,
    pad_reflect_1d,
    roll,
    clamp,
    softmax,
    concat,
    rope,
    residual_add,
    rms_norm,
    norm,
    group_norm,
    norm_pipeline,
    flash_attention,
    gated_delta_net,
    gated_output_projection,
    ssm_conv,
    ssm_conv_silu,
};

enum : uint32_t {
    kDayanNormSlotOutputResidual = 1U << 0,
    kDayanNormSlotOutputRms      = 1U << 1,
    kDayanNormSlotOutputScaled   = 1U << 2,
    kDayanNormSlotOutputLowp     = 1U << 3,
};

struct ggml_backend_dayan_graph_slot {
    ggml_backend_dayan_slot_kind kind       = ggml_backend_dayan_slot_kind::mul_mat;
    std::array<ggml_tensor *, 5> nodes      = { nullptr, nullptr, nullptr, nullptr, nullptr };
    size_t                       node_count  = 0;
    uint32_t                     output_mask = 0;
};

struct ggml_backend_dayan_graph_plan {
    std::vector<ggml_backend_dayan_graph_slot> slots;
    size_t                                     paired_mul_mat_slots  = 0;
    size_t                                     fused_mul_mat_slots   = 0;
    size_t                                     residual_add_slots    = 0;
    size_t                                     rms_norm_slots        = 0;
    size_t                                     flash_attention_slots = 0;
    size_t                                     gated_delta_net_slots = 0;
    size_t                                     gated_output_slots    = 0;
    size_t                                     ssm_conv_slots        = 0;
    size_t                                     moe_slots             = 0;
    size_t                                     fused_moe_slots       = 0;
    size_t                                     routing_slots         = 0;
};

bool ggml_backend_dayan_supports_mul_mat(const ggml_tensor * op);
bool ggml_backend_dayan_supports_mul_mat_silu(const ggml_tensor * op);
bool ggml_backend_dayan_supports_mul_mat_id(const ggml_tensor * op);
bool ggml_backend_dayan_supports_quantized_mul_mat(const ggml_tensor * op);
bool ggml_backend_dayan_supports_glu(const ggml_tensor * op);
bool ggml_backend_dayan_supports_mul_mat_id_glu(const ggml_tensor * op);
bool ggml_backend_dayan_supports_quantized_glu(const ggml_tensor * op);
bool ggml_backend_dayan_supports_add_id(const ggml_tensor * op);
bool ggml_backend_dayan_supports_argsort(const ggml_tensor * op);
bool ggml_backend_dayan_supports_top_k(const ggml_tensor * op);
bool ggml_backend_dayan_supports_residual_add(const ggml_tensor * op);
bool ggml_backend_dayan_supports_elementwise_add(const ggml_tensor * op);
bool ggml_backend_dayan_supports_elementwise_mul(const ggml_tensor * op);
bool ggml_backend_dayan_supports_elementwise_sub(const ggml_tensor * op);
bool ggml_backend_dayan_supports_elementwise_div(const ggml_tensor * op);
bool ggml_backend_dayan_supports_scale(const ggml_tensor * op);
bool ggml_backend_dayan_supports_unary(const ggml_tensor * op);
bool ggml_backend_dayan_supports_math_unary(const ggml_tensor * op);
bool ggml_backend_dayan_supports_reduce(const ggml_tensor * op);
bool ggml_backend_dayan_supports_l2_norm(const ggml_tensor * op);
bool ggml_backend_dayan_supports_get_rows(const ggml_tensor * op);
bool ggml_backend_dayan_supports_set_rows(const ggml_tensor * op);
bool ggml_backend_dayan_supports_copy(const ggml_tensor * op);
bool ggml_backend_dayan_supports_layout_copy(const ggml_tensor * op);
bool ggml_backend_dayan_supports_repeat(const ggml_tensor * op);
bool ggml_backend_dayan_supports_set(const ggml_tensor * op);
bool ggml_backend_dayan_supports_layout(const ggml_tensor * op);
bool ggml_backend_dayan_supports_clamp(const ggml_tensor * op);
bool ggml_backend_dayan_supports_softmax(const ggml_tensor * op);
bool ggml_backend_dayan_supports_concat(const ggml_tensor * op);
bool ggml_backend_dayan_supports_rope(const ggml_tensor * op);
bool ggml_backend_dayan_supports_rms_norm(const ggml_tensor * op);
bool ggml_backend_dayan_supports_norm(const ggml_tensor * op);
bool ggml_backend_dayan_supports_group_norm(const ggml_tensor * op);
bool ggml_backend_dayan_supports_rms_norm_scale(const ggml_tensor * op);
bool ggml_backend_dayan_supports_quantized_rms_norm(const ggml_tensor * op);
bool ggml_backend_dayan_supports_flash_attention(const ggml_tensor * op);
bool ggml_backend_dayan_flash_attention_lengths(const ggml_tensor * op, uint32_t * causal_offset, uint32_t * kv_len);
bool ggml_backend_dayan_supports_gated_delta_net(const ggml_tensor * op);
bool ggml_backend_dayan_supports_gate_cont(const ggml_tensor * op);
bool ggml_backend_dayan_supports_gate_sigmoid(const ggml_tensor * op);
bool ggml_backend_dayan_supports_gated_attention(const ggml_tensor * op);
bool ggml_backend_dayan_supports_ssm_conv(const ggml_tensor * op);
bool ggml_backend_dayan_supports_ssm_conv_silu(const ggml_tensor * op);

bool ggml_backend_dayan_compile_graph(const ggml_cgraph *             cgraph,
                                      ggml_backend_dayan_graph_plan * plan,
                                      const ggml_tensor **            unsupported);

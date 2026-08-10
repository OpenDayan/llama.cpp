#pragma once

#include "ggml.h"

#include <array>
#include <cstddef>
#include <vector>

enum class ggml_backend_dayan_slot_kind {
    mul_mat,
    mul_mat_quantized,
    mul_mat_pair,
    mul_mat_swiglu,
    mul_mat_geglu,
    flash_attention,
    gated_output_projection,
    ssm_conv,
    ssm_conv_silu,
};

struct ggml_backend_dayan_graph_slot {
    ggml_backend_dayan_slot_kind kind       = ggml_backend_dayan_slot_kind::mul_mat;
    std::array<ggml_tensor *, 4> nodes      = { nullptr, nullptr, nullptr, nullptr };
    size_t                       node_count = 0;
};

struct ggml_backend_dayan_graph_plan {
    std::vector<ggml_backend_dayan_graph_slot> slots;
    size_t                                     paired_mul_mat_slots  = 0;
    size_t                                     fused_mul_mat_slots   = 0;
    size_t                                     flash_attention_slots = 0;
    size_t                                     gated_output_slots    = 0;
    size_t                                     ssm_conv_slots        = 0;
};

bool ggml_backend_dayan_supports_mul_mat(const ggml_tensor * op);
bool ggml_backend_dayan_supports_quantized_mul_mat(const ggml_tensor * op);
bool ggml_backend_dayan_supports_glu(const ggml_tensor * op);
bool ggml_backend_dayan_supports_quantized_glu(const ggml_tensor * op);
bool ggml_backend_dayan_supports_flash_attention(const ggml_tensor * op);
bool ggml_backend_dayan_flash_attention_lengths(const ggml_tensor * op,
                                                uint32_t *          causal_offset,
                                                uint32_t *          kv_len);
bool ggml_backend_dayan_supports_gate_cont(const ggml_tensor * op);
bool ggml_backend_dayan_supports_gate_sigmoid(const ggml_tensor * op);
bool ggml_backend_dayan_supports_gated_attention(const ggml_tensor * op);
bool ggml_backend_dayan_supports_ssm_conv(const ggml_tensor * op);
bool ggml_backend_dayan_supports_ssm_conv_silu(const ggml_tensor * op);

bool ggml_backend_dayan_compile_graph(const ggml_cgraph *             cgraph,
                                      ggml_backend_dayan_graph_plan * plan,
                                      const ggml_tensor **            unsupported);

#pragma once

#include "../ggml-dayan-mega-executor.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_mega_detail {

struct Qwen35DenseMxfp4LinearGraph {
    enum : uint16_t {
        p_norm0     = 0,
        p_scale0    = NextParameter<p_norm0, 8>::value,
        p_rows0     = NextParameter<p_scale0, 3>::value,
        p_quant0    = NextParameter<p_rows0, 4>::value,
        p_qkv       = NextParameter<p_quant0, 4>::value,
        p_scale1    = NextParameter<p_qkv, 9>::value,
        p_rows1     = NextParameter<p_scale1, 3>::value,
        p_conv_split = NextParameter<p_rows1, 4>::value,
        p_l2q       = NextParameter<p_conv_split, 6>::value,
        p_l2k       = NextParameter<p_l2q, 3>::value,
        p_alpha     = NextParameter<p_l2k, 3>::value,
        p_alpha_add = NextParameter<p_alpha, 9>::value,
        p_softplus  = NextParameter<p_alpha_add, 4>::value,
        p_alpha_mul = NextParameter<p_softplus, 3>::value,
        p_beta      = NextParameter<p_alpha_mul, 4>::value,
        p_sigmoid   = NextParameter<p_beta, 9>::value,
        p_gdn       = NextParameter<p_sigmoid, 3>::value,
        p_copy1     = NextParameter<p_gdn, 8>::value,
        p_gdn_norm  = NextParameter<p_copy1, 3>::value,
        p_z         = NextParameter<p_gdn_norm, 8>::value,
        p_silu      = NextParameter<p_z, 9>::value,
        p_gate_mul  = NextParameter<p_silu, 3>::value,
        p_quant1    = NextParameter<p_gate_mul, 4>::value,
        p_out       = NextParameter<p_quant1, 4>::value,
        p_ffn_norm  = NextParameter<p_out, 9>::value,
        p_quant2    = NextParameter<p_ffn_norm, 8>::value,
        p_ffn       = NextParameter<p_quant2, 4>::value,
        p_down      = NextParameter<p_ffn, 9>::value,
        p_residual  = NextParameter<p_down, 9>::value,
        parameter_count = NextParameter<p_residual, 4>::value,
    };

    using B = Arguments<>;

    static const Operation operations[];
    static constexpr uint16_t operation_count = 56;
};

const Operation Qwen35DenseMxfp4LinearGraph::operations[] = {
    { norm_pipeline, Arguments<p_norm0 + 0, p_norm0 + 1, p_norm0 + 2, p_norm0 + 3, p_norm0 + 4, p_norm0 + 5, p_norm0 + 6, p_norm0 + 7>::values },
    { nullptr, nullptr },
    { scale, Arguments<p_scale0 + 0, p_scale0 + 1, p_scale0 + 2>::values },
    { nullptr, nullptr },
    { get_rows, Arguments<p_rows0 + 0, p_rows0 + 1, p_rows0 + 2, p_rows0 + 3>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quant0 + 0, p_quant0 + 1, p_quant0 + 2, p_quant0 + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_qkv + 0, p_qkv + 1, p_qkv + 2, p_qkv + 3, p_qkv + 4, p_qkv + 5, p_qkv + 6, p_qkv + 7, p_qkv + 8>::values },
    { nullptr, nullptr },
    { scale, Arguments<p_scale1 + 0, p_scale1 + 1, p_scale1 + 2>::values },
    { nullptr, nullptr },
    { get_rows, Arguments<p_rows1 + 0, p_rows1 + 1, p_rows1 + 2, p_rows1 + 3>::values },
    { nullptr, nullptr },
    { ssm_conv_split, Arguments<p_conv_split + 0, p_conv_split + 1, p_conv_split + 2, p_conv_split + 3,
                                p_conv_split + 4, p_conv_split + 5>::values },
    { nullptr, nullptr },
    { l2_norm, Arguments<p_l2q + 0, p_l2q + 1, p_l2q + 2>::values },
    { l2_norm, Arguments<p_l2k + 0, p_l2k + 1, p_l2k + 2>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_alpha + 0, p_alpha + 1, p_alpha + 2, p_alpha + 3, p_alpha + 4, p_alpha + 5, p_alpha + 6, p_alpha + 7, p_alpha + 8>::values },
    { nullptr, nullptr },
    { add, Arguments<p_alpha_add + 0, p_alpha_add + 1, p_alpha_add + 2, p_alpha_add + 3>::values },
    { nullptr, nullptr },
    { softplus, Arguments<p_softplus + 0, p_softplus + 1, p_softplus + 2>::values },
    { nullptr, nullptr },
    { multiply, Arguments<p_alpha_mul + 0, p_alpha_mul + 1, p_alpha_mul + 2, p_alpha_mul + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_beta + 0, p_beta + 1, p_beta + 2, p_beta + 3, p_beta + 4, p_beta + 5, p_beta + 6, p_beta + 7, p_beta + 8>::values },
    { nullptr, nullptr },
    { sigmoid, Arguments<p_sigmoid + 0, p_sigmoid + 1, p_sigmoid + 2>::values },
    { nullptr, nullptr },
    { gated_delta_net, Arguments<p_gdn + 0, p_gdn + 1, p_gdn + 2, p_gdn + 3, p_gdn + 4, p_gdn + 5, p_gdn + 6, p_gdn + 7>::values },
    { nullptr, nullptr },
    { copy, Arguments<p_copy1 + 0, p_copy1 + 1, p_copy1 + 2>::values },
    { nullptr, nullptr },
    { norm_pipeline, Arguments<p_gdn_norm + 0, p_gdn_norm + 1, p_gdn_norm + 2, p_gdn_norm + 3, p_gdn_norm + 4, p_gdn_norm + 5, p_gdn_norm + 6, p_gdn_norm + 7>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_z + 0, p_z + 1, p_z + 2, p_z + 3, p_z + 4, p_z + 5, p_z + 6, p_z + 7, p_z + 8>::values },
    { nullptr, nullptr },
    { silu, Arguments<p_silu + 0, p_silu + 1, p_silu + 2>::values },
    { nullptr, nullptr },
    { multiply, Arguments<p_gate_mul + 0, p_gate_mul + 1, p_gate_mul + 2, p_gate_mul + 3>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quant1 + 0, p_quant1 + 1, p_quant1 + 2, p_quant1 + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_out + 0, p_out + 1, p_out + 2, p_out + 3, p_out + 4, p_out + 5, p_out + 6, p_out + 7, p_out + 8>::values },
    { nullptr, nullptr },
    { norm_pipeline, Arguments<p_ffn_norm + 0, p_ffn_norm + 1, p_ffn_norm + 2, p_ffn_norm + 3, p_ffn_norm + 4, p_ffn_norm + 5, p_ffn_norm + 6, p_ffn_norm + 7>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quant2 + 0, p_quant2 + 1, p_quant2 + 2, p_quant2 + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_ffn + 0, p_ffn + 1, p_ffn + 2, p_ffn + 3, p_ffn + 4, p_ffn + 5, p_ffn + 6, p_ffn + 7, p_ffn + 8>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_down + 0, p_down + 1, p_down + 2, p_down + 3, p_down + 4, p_down + 5, p_down + 6, p_down + 7, p_down + 8>::values },
    { nullptr, nullptr },
    { residual_add, Arguments<p_residual + 0, p_residual + 1, p_residual + 2, p_residual + 3>::values },
};

struct Qwen35DenseMxfp4FullGraph {
    enum : uint16_t {
        p_norm0     = 0,
        p_quant0    = NextParameter<p_norm0, 8>::value,
        p_qg        = NextParameter<p_quant0, 4>::value,
        p_q_norm    = NextParameter<p_qg, 9>::value,
        p_q_scale   = NextParameter<p_q_norm, 3>::value,
        p_q_rope    = NextParameter<p_q_scale, 4>::value,
        p_k         = NextParameter<p_q_rope, 5>::value,
        p_v         = NextParameter<p_k, 9>::value,
        p_k_norm    = NextParameter<p_v, 9>::value,
        p_k_rope    = NextParameter<p_k_norm, 8>::value,
        p_set_k     = NextParameter<p_k_rope, 5>::value,
        p_set_v     = NextParameter<p_set_k, 4>::value,
        p_q_copy    = NextParameter<p_set_v, 4>::value,
        p_attention = NextParameter<p_q_copy, 3>::value,
        p_cont      = NextParameter<p_attention, 5>::value,
        p_sigmoid   = NextParameter<p_cont, 3>::value,
        p_gate_mul  = NextParameter<p_sigmoid, 3>::value,
        p_quant1    = NextParameter<p_gate_mul, 4>::value,
        p_out       = NextParameter<p_quant1, 4>::value,
        p_ffn_norm  = NextParameter<p_out, 9>::value,
        p_quant2    = NextParameter<p_ffn_norm, 8>::value,
        p_ffn       = NextParameter<p_quant2, 4>::value,
        p_down      = NextParameter<p_ffn, 9>::value,
        p_residual  = NextParameter<p_down, 9>::value,
        parameter_count = NextParameter<p_residual, 4>::value,
    };

    static const Operation operations[];
    static constexpr uint16_t operation_count = 45;
};

const Operation Qwen35DenseMxfp4FullGraph::operations[] = {
    { norm_pipeline, Arguments<p_norm0 + 0, p_norm0 + 1, p_norm0 + 2, p_norm0 + 3, p_norm0 + 4, p_norm0 + 5, p_norm0 + 6, p_norm0 + 7>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quant0 + 0, p_quant0 + 1, p_quant0 + 2, p_quant0 + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_qg + 0, p_qg + 1, p_qg + 2, p_qg + 3, p_qg + 4, p_qg + 5, p_qg + 6, p_qg + 7, p_qg + 8>::values },
    { nullptr, nullptr },
    { rms_norm, Arguments<p_q_norm + 0, p_q_norm + 1, p_q_norm + 2>::values },
    { nullptr, nullptr },
    { multiply, Arguments<p_q_scale + 0, p_q_scale + 1, p_q_scale + 2, p_q_scale + 3>::values },
    { nullptr, nullptr },
    { rope, Arguments<p_q_rope + 0, p_q_rope + 1, p_q_rope + 2, p_q_rope + 3, p_q_rope + 4>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_k + 0, p_k + 1, p_k + 2, p_k + 3, p_k + 4, p_k + 5, p_k + 6, p_k + 7, p_k + 8>::values },
    { matmul_mxfp4, Arguments<p_v + 0, p_v + 1, p_v + 2, p_v + 3, p_v + 4, p_v + 5, p_v + 6, p_v + 7, p_v + 8>::values },
    { nullptr, nullptr },
    { norm_pipeline, Arguments<p_k_norm + 0, p_k_norm + 1, p_k_norm + 2, p_k_norm + 3, p_k_norm + 4, p_k_norm + 5, p_k_norm + 6, p_k_norm + 7>::values },
    { nullptr, nullptr },
    { rope, Arguments<p_k_rope + 0, p_k_rope + 1, p_k_rope + 2, p_k_rope + 3, p_k_rope + 4>::values },
    { nullptr, nullptr },
    { set_rows, Arguments<p_set_k + 0, p_set_k + 1, p_set_k + 2, p_set_k + 3>::values },
    { set_rows, Arguments<p_set_v + 0, p_set_v + 1, p_set_v + 2, p_set_v + 3>::values },
    { nullptr, nullptr },
    { copy, Arguments<p_q_copy + 0, p_q_copy + 1, p_q_copy + 2>::values },
    { nullptr, nullptr },
    { flash_attention_fp16, Arguments<p_attention + 0, p_attention + 1, p_attention + 2, p_attention + 3, p_attention + 4>::values },
    { nullptr, nullptr },
    { copy, Arguments<p_cont + 0, p_cont + 1, p_cont + 2>::values },
    { nullptr, nullptr },
    { sigmoid, Arguments<p_sigmoid + 0, p_sigmoid + 1, p_sigmoid + 2>::values },
    { nullptr, nullptr },
    { multiply, Arguments<p_gate_mul + 0, p_gate_mul + 1, p_gate_mul + 2, p_gate_mul + 3>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quant1 + 0, p_quant1 + 1, p_quant1 + 2, p_quant1 + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_out + 0, p_out + 1, p_out + 2, p_out + 3, p_out + 4, p_out + 5, p_out + 6, p_out + 7, p_out + 8>::values },
    { nullptr, nullptr },
    { norm_pipeline, Arguments<p_ffn_norm + 0, p_ffn_norm + 1, p_ffn_norm + 2, p_ffn_norm + 3, p_ffn_norm + 4, p_ffn_norm + 5, p_ffn_norm + 6, p_ffn_norm + 7>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quant2 + 0, p_quant2 + 1, p_quant2 + 2, p_quant2 + 3>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_ffn + 0, p_ffn + 1, p_ffn + 2, p_ffn + 3, p_ffn + 4, p_ffn + 5, p_ffn + 6, p_ffn + 7, p_ffn + 8>::values },
    { nullptr, nullptr },
    { matmul_mxfp4, Arguments<p_down + 0, p_down + 1, p_down + 2, p_down + 3, p_down + 4, p_down + 5, p_down + 6, p_down + 7, p_down + 8>::values },
    { nullptr, nullptr },
    { residual_add, Arguments<p_residual + 0, p_residual + 1, p_residual + 2, p_residual + 3>::values },
};

static_assert(Qwen35DenseMxfp4LinearGraph::parameter_count <= kDayanMegaMaxParameters, "Linear graph parameter table is too large");
static_assert(Qwen35DenseMxfp4FullGraph::parameter_count <= kDayanMegaMaxParameters, "Full graph parameter table is too large");
static_assert(Qwen35DenseMxfp4LinearGraph::parameter_count == kDayanMegaQwen35DenseMxfp4LinearParameters, "Linear graph parameter ABI mismatch");
static_assert(Qwen35DenseMxfp4FullGraph::parameter_count == kDayanMegaQwen35DenseMxfp4FullParameters, "Full graph parameter ABI mismatch");
static_assert(sizeof(Qwen35DenseMxfp4LinearGraph::operations) / sizeof(Operation) == Qwen35DenseMxfp4LinearGraph::operation_count, "Linear graph operation count mismatch");
static_assert(sizeof(Qwen35DenseMxfp4FullGraph::operations) / sizeof(Operation) == Qwen35DenseMxfp4FullGraph::operation_count, "Full graph operation count mismatch");

} // namespace ggml_dayan_mega_detail

#endif

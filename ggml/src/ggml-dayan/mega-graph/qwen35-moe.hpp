#pragma once

#include "qwen35-dense-bf16.hpp"

#if defined(__DAYAN_DEVICE__)
namespace ggml_dayan_mega_detail {

template <uint16_t Base>
struct Qwen35MoeFfnGraph {
    enum : uint16_t {
        p_norm = Base,
        p_router = NextParameter<p_norm, 8>::value,
        p_probs = NextParameter<p_router, 4>::value,
        p_sort = NextParameter<p_probs, 3>::value,
        p_weights = NextParameter<p_sort, 3>::value,
        p_sum = NextParameter<p_weights, 4>::value,
        p_clamp = NextParameter<p_sum, 3>::value,
        p_divide = NextParameter<p_clamp, 3>::value,
        p_quantize = NextParameter<p_divide, 4>::value,
        p_experts = NextParameter<p_quantize, 4>::value,
        p_down = NextParameter<p_experts, 10>::value,
        p_weighted = NextParameter<p_down, 10>::value,
        p_add0 = NextParameter<p_weighted, 4>::value,
        p_add1 = NextParameter<p_add0, 4>::value,
        p_add2 = NextParameter<p_add1, 4>::value,
        p_add3 = NextParameter<p_add2, 4>::value,
        p_add4 = NextParameter<p_add3, 4>::value,
        p_add5 = NextParameter<p_add4, 4>::value,
        p_add6 = NextParameter<p_add5, 4>::value,
        p_copy = NextParameter<p_add6, 4>::value,
        p_shared = NextParameter<p_copy, 3>::value,
        p_shared_down = NextParameter<p_shared, 5>::value,
        p_shared_router = NextParameter<p_shared_down, 4>::value,
        p_sigmoid = NextParameter<p_shared_router, 4>::value,
        p_shared_gate = NextParameter<p_sigmoid, 3>::value,
        p_combine = NextParameter<p_shared_gate, 4>::value,
        p_residual = NextParameter<p_combine, 4>::value,
        parameter_count = NextParameter<p_residual, 4>::value,
    };
    static const Operation operations[];
    static constexpr uint16_t operation_count = 53;
};

template <uint16_t Base>
const Operation Qwen35MoeFfnGraph<Base>::operations[] = {
    { norm_pipeline, Arguments<p_norm + 0, p_norm + 1, p_norm + 2, p_norm + 3, p_norm + 4, p_norm + 5, p_norm + 6, p_norm + 7>::values },
    { nullptr, nullptr },
    { matmul_tf32, Arguments<p_router + 0, p_router + 1, p_router + 2, p_router + 3>::values },
    { nullptr, nullptr },
    { softmax, Arguments<p_probs + 0, p_probs + 1, p_probs + 2>::values },
    { nullptr, nullptr },
    { argsort, Arguments<p_sort + 0, p_sort + 1, p_sort + 2>::values },
    { nullptr, nullptr },
    { get_rows, Arguments<p_weights + 0, p_weights + 1, p_weights + 2, p_weights + 3>::values },
    { nullptr, nullptr },
    { sum_rows, Arguments<p_sum + 0, p_sum + 1, p_sum + 2>::values },
    { nullptr, nullptr },
    { clamp, Arguments<p_clamp + 0, p_clamp + 1, p_clamp + 2>::values },
    { nullptr, nullptr },
    { divide, Arguments<p_divide + 0, p_divide + 1, p_divide + 2, p_divide + 3>::values },
    { nullptr, nullptr },
    { quantize_mxfp4, Arguments<p_quantize + 0, p_quantize + 1, p_quantize + 2, p_quantize + 3>::values },
    { nullptr, nullptr },
    { matmul_id_mxfp4, Arguments<p_experts + 0, p_experts + 1, p_experts + 2, p_experts + 3, p_experts + 4, p_experts + 5, p_experts + 6, p_experts + 7, p_experts + 8, p_experts + 9>::values },
    { nullptr, nullptr },
    { matmul_id_mxfp4, Arguments<p_down + 0, p_down + 1, p_down + 2, p_down + 3, p_down + 4, p_down + 5, p_down + 6, p_down + 7, p_down + 8, p_down + 9>::values },
    { nullptr, nullptr },
    { multiply, Arguments<p_weighted + 0, p_weighted + 1, p_weighted + 2, p_weighted + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add0 + 0, p_add0 + 1, p_add0 + 2, p_add0 + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add1 + 0, p_add1 + 1, p_add1 + 2, p_add1 + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add2 + 0, p_add2 + 1, p_add2 + 2, p_add2 + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add3 + 0, p_add3 + 1, p_add3 + 2, p_add3 + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add4 + 0, p_add4 + 1, p_add4 + 2, p_add4 + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add5 + 0, p_add5 + 1, p_add5 + 2, p_add5 + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_add6 + 0, p_add6 + 1, p_add6 + 2, p_add6 + 3>::values },
    { nullptr, nullptr },
    { copy, Arguments<p_copy + 0, p_copy + 1, p_copy + 2>::values },
    { nullptr, nullptr },
    { matmul_swiglu_bf16, Arguments<p_shared + 0, p_shared + 1, p_shared + 2, p_shared + 3, p_shared + 4>::values },
    { nullptr, nullptr },
    { matmul_bf16, Arguments<p_shared_down + 0, p_shared_down + 1, p_shared_down + 2, p_shared_down + 3>::values },
    { nullptr, nullptr },
    { matmul_tf32, Arguments<p_shared_router + 0, p_shared_router + 1, p_shared_router + 2, p_shared_router + 3>::values },
    { nullptr, nullptr },
    { sigmoid, Arguments<p_sigmoid + 0, p_sigmoid + 1, p_sigmoid + 2>::values },
    { nullptr, nullptr },
    { multiply, Arguments<p_shared_gate + 0, p_shared_gate + 1, p_shared_gate + 2, p_shared_gate + 3>::values },
    { nullptr, nullptr },
    { add, Arguments<p_combine + 0, p_combine + 1, p_combine + 2, p_combine + 3>::values },
    { nullptr, nullptr },
    { residual_add, Arguments<p_residual + 0, p_residual + 1, p_residual + 2, p_residual + 3>::values },
};

using Qwen35MoeLinearGraph = Qwen35MoeFfnGraph<Qwen35DenseBf16LinearGraph::p_copy4>;
using Qwen35MoeFullGraph = Qwen35MoeFfnGraph<Qwen35DenseBf16FullGraph::p_cont>;
static_assert(Qwen35MoeLinearGraph::parameter_count == kDayanMegaQwen35MoeLinearParameters, "MoE linear parameter ABI mismatch");
static_assert(Qwen35MoeFullGraph::parameter_count == kDayanMegaQwen35MoeFullParameters, "MoE full parameter ABI mismatch");
static_assert(sizeof(Qwen35MoeLinearGraph::operations) / sizeof(Operation) == Qwen35MoeLinearGraph::operation_count, "MoE operation count mismatch");
} // namespace ggml_dayan_mega_detail
#endif

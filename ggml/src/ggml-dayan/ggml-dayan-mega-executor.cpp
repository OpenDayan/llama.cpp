#include "ggml-dayan-mega-executor.h"

#include "ggml-dayan-common.h"
#include "ggml-dayan-attention.h"
#include "ggml-dayan-conv.h"
#include "ggml-dayan-elementwise.h"
#include "ggml-dayan-layout.h"
#include "ggml-dayan-matmul-4c.h"
#include "ggml-dayan-memory.h"
#include "ggml-dayan-mxfp4.h"
#include "ggml-dayan-moe.h"
#include "ggml-dayan-softmax.h"
#include "ggml-dayan-normalization.h"
#include "ggml-dayan-rope.h"

#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#    include <dayan_sys.h>

static inline uint32_t float_as_u32(float value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

static inline float u32_as_float(uint32_t value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = value;
    return bits.f;
}

#    include "ggml-dayan-flash-attention.h"
#    include "ggml-dayan-gated-delta-net.h"

namespace ggml_dayan_mega_detail {

static inline void reset_local_resources() {
    // The mega descriptor owns physical Local Resources 0..31.
    __builtin_riscv_tx_base_tcsr_write(UINT32_MAX, 0, 0x035U);
}

static int prepare_matmul_4c(uint64_t workspace) {
    reset_local_resources();
    ggml_dayan_matmul_4c_detail::initialize_resources(workspace);
    return 0;
}

int norm_pipeline(const DayanNormPipelineConfig * config, const void * input, const void * right, const float * scale, void * residual, float * rms, float * scaled, uint16_t * lowp) {
    using namespace ggml_dayan_matmul_detail;
    run_rms_norm_pipeline(config, input, right, scale, residual, rms, scaled, lowp);
    return 0;
}

int residual_add(const DayanElementwiseConfig * config, const void * left, const void * right, void * output) {
    using namespace ggml_dayan_matmul_detail;
    run_residual_add<Bf16Format, kDayanDataTypeF32, normalization_load_f32>(config, left, right, output);
    return 0;
}

int residual_add_bf16(const DayanElementwiseConfig * config, const void * left, const void * right, void * output) {
    using namespace ggml_dayan_matmul_detail;
    run_residual_add<Bf16Format, kDayanDataTypeBF16,
                     normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>>(config, left, right, output);
    return 0;
}

int rms_norm(const DayanNormConfig * config, const void * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_rms_norm(config, input, output);
    return 0;
}

int scale(const DayanScaleConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_scale_f32(config, input, output);
    return 0;
}

int get_rows(const DayanRowsConfig * config, const void * input, const void * indices, void * output) {
    return ggml_dayan_matmul_detail::run_get_rows(config, input, indices, output);
}

int quantize_mxfp4(const DayanMxfp4Config * config, const void * input, void * output, void * scales) {
    ggml_dayan_mxfp4_detail::convert(config, input, output, scales);
    return 0;
}

int matmul_mxfp4(const DayanMxfp4Config * config, const void * input, const void * input_scales, const void * weight, const void * weight_scales, const void * up, const void * up_scales, void * output, void * output_scales) {
    reset_local_resources();
    return ggml_dayan_mxfp4_detail::run(config, input, input_scales, weight, weight_scales, up, up_scales, output, output_scales);
}

int concat(const DayanConcatConfig * config, const void * left, const void * right, void * output) {
    ggml_dayan_matmul_detail::run_concat(config, left, right, output);
    return 0;
}

int copy(const DayanCopyConfig * config, const void * input, void * output) {
    ggml_dayan_matmul_detail::run_copy(config, input, output);
    return 0;
}

int ssm_conv(const DayanSSMConvConfig * config, const float * input, const float * weight, float * output) {
    ggml_dayan_matmul_detail::run_ssm_conv(config, input, weight, output);
    return 0;
}

int ssm_conv_split(const DayanSSMConvConfig * config, const float * input, const float * weight, float * output, const float * current, float * state) {
    ggml_dayan_matmul_detail::run_ssm_conv(config, input, weight, output, current, state);
    return 0;
}

int l2_norm(const DayanNormConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_l2_norm(config, input, output);
    return 0;
}

int add(const DayanBinaryConfig * config, const float * left, const float * right, float * output) {
    ggml_dayan_matmul_detail::run_binary_f32<__tx_vec_vv_fp32_add<0, 0, 16, 0>>(config, left, right, output);
    return 0;
}

int multiply(const DayanBinaryConfig * config, const float * left, const float * right, float * output) {
    ggml_dayan_matmul_detail::run_binary_f32<__tx_vec_vv_fp32_mul<0, 0, 16, 0>>(config, left, right, output);
    return 0;
}

int softplus(const DayanElementwiseConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_unary_f32<ggml_dayan_matmul_detail::softplus_tile>(config, input, output);
    return 0;
}

int sigmoid(const DayanElementwiseConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_unary_f32<ggml_dayan_matmul_detail::sigmoid_tile>(config, input, output);
    return 0;
}

int silu(const DayanElementwiseConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_unary_f32<ggml_dayan_matmul_detail::silu_tile>(config, input, output);
    return 0;
}

int gated_delta_net(const DayanGatedDeltaNetConfig * config, const float * q, const float * k, const float * v, const float * g, const float * beta, const float * state, float * output) {
    reset_local_resources();
    return (int) ggml_dayan_gdn_detail::run_gated_delta_net(config, q, k, v, g, beta, state, output);
}

int rope(const DayanRopeConfig * config, const float * input, const int32_t * positions, const float * factors, float * output) {
    ggml_dayan_matmul_detail::run_rope_f32(config, input, positions, factors, output);
    return 0;
}

int set_rows(const DayanRowsConfig * config, const float * input, const void * indices, void * output) {
    ggml_dayan_matmul_detail::run_set_rows(config, input, indices, output);
    return 0;
}

int flash_attention_fp16(const DayanFlashAttentionConfig * config, const uint16_t * q, const uint16_t * k, const uint16_t * v, float * output) {
    reset_local_resources();
    if (dayan_flash_attention_use_split_kv(config->q_len, config->kv_len)) {
        return ggml_dayan_flash_attention_detail::run<ggml_dayan_matmul_detail::Fp16Format, 4, true>(
            config, q, k, v, output);
    }
    return ggml_dayan_flash_attention_detail::run<ggml_dayan_matmul_detail::Fp16Format, 1>(
        config, q, k, v, output);
}

int matmul_bf16(const DayanMatmulConfig * config, const uint16_t * input, const uint16_t * weight, float * output) {
    const int status = prepare_matmul_4c(config->cluster_workspace);
    if (status < 0) {
        return status;
    }
    return config->m <= 512 ?
               ggml_dayan_matmul_4c_detail::run_matmul_4c<ggml_dayan_matmul_detail::Bf16Format, true>(
                   config, input, weight, output) :
               ggml_dayan_matmul_4c_detail::run_matmul_4c<ggml_dayan_matmul_detail::Bf16Format>(
                   config, input, weight, output);
}

int matmul_pair_bf16(const DayanMatmulPairConfig * config, const uint16_t * input, const uint16_t * first, const uint16_t * second, float * first_output, float * second_output) {
    const int status = prepare_matmul_4c(config->first.cluster_workspace);
    if (status < 0) {
        return status;
    }
    return ggml_dayan_matmul_4c_detail::run_matmul_pair_4c<ggml_dayan_matmul_detail::Bf16Format>(config, input, first, second, first_output, second_output);
}

int matmul_swiglu_bf16(const DayanMatmulConfig * config, const uint16_t * input, const uint16_t * gate, const uint16_t * up, uint16_t * output) {
    const int status = prepare_matmul_4c(config->cluster_workspace);
    if (status < 0) {
        return status;
    }
    return config->m <= 512 ?
               ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Bf16Format, false, true>(
                   config, input, gate, up, output) :
               ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Bf16Format, false>(
                   config, input, gate, up, output);
}

int matmul_quantized_bf16(const DayanMatmulConfig * config, const uint16_t * input, const uint16_t * weight, uint16_t * output) {
    const int status = prepare_matmul_4c(config->cluster_workspace);
    if (status < 0) {
        return status;
    }
    return config->m <= 512 ?
               ggml_dayan_matmul_4c_detail::run_matmul_quantized_4c<ggml_dayan_matmul_detail::Bf16Format, true>(
                   config, input, weight, output) :
               ggml_dayan_matmul_4c_detail::run_matmul_quantized_4c<ggml_dayan_matmul_detail::Bf16Format>(
                   config, input, weight, output);
}

int gated_output_projection_bf16(const DayanGatedProjectionConfig * config, const float * input, const float * gate, const uint16_t * weight, float * output, uint16_t * scratch) {
    const int status = prepare_matmul_4c(config->matmul.cluster_workspace);
    if (status < 0) {
        return status;
    }
    return ggml_dayan_matmul_4c_detail::run_gated_output_projection_f32_gate_4c<
        ggml_dayan_matmul_detail::Bf16Format>(
            config,
            input,
            gate,
            weight,
            output,
            scratch);
}

int matmul_tf32(const DayanMatmulConfig * config, const float * input, const float * weight, float * output) {
    ggml_dayan_matmul_detail::run_router_f32(config, input, weight, output);
    return 0;
}

int moe_sort(const DayanMoeSortConfig * config, const float * probabilities, int32_t * selected, void * workspace) {
    return ggml_dayan_matmul_detail::run_moe_sort(config, probabilities, selected, workspace);
}

int matmul_id_mxfp4(const DayanMxfp4MoeConfig * config, const void * input, const void * input_scales, const void * weight, const void * weight_scales, const void * up, const void * up_scales, const int32_t * selected, void * output, void * output_scales, void * workspace) {
    reset_local_resources();
    return ggml_dayan_matmul_detail::run_mul_mat_id_mxfp4(config, input, input_scales, weight, weight_scales, up, up_scales, selected, output, output_scales, workspace);
}


int softmax(const DayanSoftmaxConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_softmax_f32(
        config,
        input, nullptr, nullptr,
        output);
    return 0;
}

int sum_rows(const DayanReduceConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_reduce_rows_f32(config, input, output);
    return 0;
}

int clamp(const DayanClampConfig * config, const float * input, float * output) {
    ggml_dayan_matmul_detail::run_clamp_f32(config, input, output);
    return 0;
}

int divide(const DayanBinaryConfig * config, const float * left, const float * right, float * output) {
    ggml_dayan_matmul_detail::run_div_f32(config, left, right, output);
    return 0;
}

int noop() { return 0; }

#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) namespace Name { static const Operation operations[] = {
#define MEGA_STEP(Name, Kind, Function, Barriers, Mode, ...) \
    { Binding<decltype(&ggml_dayan_mega_detail::Function), &ggml_dayan_mega_detail::Function>::call, Signature<decltype(&ggml_dayan_mega_detail::Function)>::count, Barriers, #Name },
#define MEGA_GRAPH_END() }; }
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END

template <unsigned Count>
static int execute(const Operation (&operations)[Count], const uintptr_t * parameters, dayan_cluster_barrier_t * barrier, uint64_t workspace) {
    for (const Operation & operation : operations) {
        int status = dayan_set_proc_name(operation.name);
        if (status < 0) {
            return status;
        }
        status = operation.function(parameters, workspace);
        if (status < 0) {
            return status;
        }
        parameters += operation.argument_count;
        for (uint16_t index = 0; index < operation.barriers; ++index) {
            status = dayan_cluster_barrier(4, barrier);
            if (status < 0) {
                return status;
            }
        }
    }
    return 0;
}

template <bool Moe>
static void run(const DayanMegaConfig * config, const void * const * sources) {
    DAYAN_PARAMS_IF(config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    dayan_cache_invalidate(config, sizeof(*config));
    const DayanMegaParameter * encoded = (const DayanMegaParameter *) (config + 1);
    DAYAN_PARAMS_IF(shared == nullptr || dayan_task_mode() != 4U || dayan_task_dim(DAYAN_TASK_X) != 4U ||
                    dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        DAYAN_PARAMS_ERROR();
    }
    DAYAN_PARAMS_IF(config->parameter_count > kDayanMegaMaxParameters ||
                    config->total_bytes < sizeof(*config) + (uint32_t) config->parameter_count * sizeof(DayanMegaParameter)) {
        DAYAN_PARAMS_ERROR();
    }
    if (config->workspace_offset % 64 || config->workspace_offset < sizeof(*config) +
            (uint32_t) config->parameter_count * sizeof(DayanMegaParameter) ||
        config->total_bytes < sizeof(DayanClusterState) ||
        config->workspace_offset > config->total_bytes - sizeof(DayanClusterState)) {
        dayan_thread_exit(-22);
    }
    dayan_cache_invalidate(config, config->total_bytes);

    uintptr_t parameters[kDayanMegaMaxParameters];
    for (uint16_t index = 0; index < config->parameter_count; ++index) {
        DAYAN_PARAMS_IF(encoded[index].source > kDayanMegaRootCount || sources[encoded[index].source] == nullptr ||
                        (encoded[index].source == 0 && encoded[index].offset >= config->total_bytes)) {
            DAYAN_PARAMS_ERROR();
        }
        parameters[index] = (uintptr_t) sources[encoded[index].source] + encoded[index].offset;
    }

    auto * barrier = (dayan_cluster_barrier_t *) shared;
    const uint64_t workspace = (uintptr_t) config + config->workspace_offset;
    int status = -22;
    switch (config->graph) {
#define MEGA_GRAPH_BEGIN(Name, Id, IsMoe, Linear) \
        case Id: \
            if constexpr (Moe == IsMoe) { \
                DAYAN_PARAMS_IF(config->parameter_count != Name::parameter_count) { DAYAN_PARAMS_ERROR(); } \
                status = execute(Name::operations, parameters, barrier, workspace); \
            } \
            break;
#define MEGA_STEP(...)
#define MEGA_GRAPH_END()
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END
        default: break;
    }
    DAYAN_PARAMS_IF(status < 0) {
        DAYAN_PARAMS_ERROR();
    }
    (void) status;
}

} // namespace ggml_dayan_mega_detail
#endif

__kernel__ void qwen35_dense_mega(
    const DayanMegaConfig * config,
    const void * root0,
    const void * root1,
    const void * root2,
    const void * root3,
    const void * root4,
    const void * root5,
    const void * root6,
    const void * root7,
    const void * root8,
    const void * root9,
    const void * root10,
    const void * root11,
    const void * root12,
    const void * root13,
    const void * root14,
    const void * root15) {
    const void * sources[] = {
        config, root0, root1, root2, root3, root4, root5, root6, root7,
        root8, root9, root10, root11, root12, root13, root14, root15,
    };
    ggml_dayan_mega_detail::run<false>(config, sources);
}

DAYAN_KERNEL_RESOURCE_DESC_EX(
    qwen35_dense_mega, 1, DAYAN_KERNEL_MAX_CHILD_THREADS, 0,
    sizeof(dayan_cluster_barrier_t), MATMUL_DAYAN_4C_URAM_BYTES,
    DAYAN_TCSR_LOGICAL_GROUPS, DAYAN_TCSR_LOGICAL_GROUPS, 4,
    DAYAN_TCSR_LOGICAL_GROUPS, DAYAN_RESOURCE_LOGICAL_COUNT, DAYAN_RESOURCE_LOGICAL_COUNT);

__kernel__ void qwen35_moe_mega(
    const DayanMegaConfig * config,
    const void * root0,
    const void * root1,
    const void * root2,
    const void * root3,
    const void * root4,
    const void * root5,
    const void * root6,
    const void * root7,
    const void * root8,
    const void * root9,
    const void * root10,
    const void * root11,
    const void * root12,
    const void * root13,
    const void * root14,
    const void * root15) {
    const void * sources[] = {
        config, root0, root1, root2, root3, root4, root5, root6, root7,
        root8, root9, root10, root11, root12, root13, root14, root15,
    };
    ggml_dayan_mega_detail::run<true>(config, sources);
}

DAYAN_KERNEL_RESOURCE_DESC_EX(
    qwen35_moe_mega, 1, DAYAN_KERNEL_MAX_CHILD_THREADS, 0,
    sizeof(dayan_cluster_barrier_t), MATMUL_DAYAN_4C_URAM_BYTES,
    DAYAN_TCSR_LOGICAL_GROUPS, DAYAN_TCSR_LOGICAL_GROUPS, 4,
    DAYAN_TCSR_LOGICAL_GROUPS, DAYAN_RESOURCE_LOGICAL_COUNT, DAYAN_RESOURCE_LOGICAL_COUNT);

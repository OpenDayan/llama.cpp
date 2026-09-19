#include "ggml-dayan-mega-executor.h"

#include "ggml-dayan-common.h"
#include "ggml-dayan-attention.h"
#include "ggml-dayan-conv.h"
#include "ggml-dayan-elementwise.h"
#include "ggml-dayan-layout.h"
#include "ggml-dayan-matmul-4c.h"
#include "ggml-dayan-memory.h"
#include "ggml-dayan-mxfp4.h"
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
#    include "mega-graph/qwen35-dense-bf16.hpp"
#    include "mega-graph/qwen35-dense-mxfp4.hpp"

namespace ggml_dayan_mega_detail {

static_assert(ggml_dayan_flash_attention_detail::kAtomicBytes <= MATMUL_DAYAN_4C_ATOMIC_BYTES,
              "Flash Attention shared state overlaps the mega barrier");
static_assert(MATMUL_DAYAN_4C_ATOMIC_BYTES % alignof(dayan_cluster_barrier_t) == 0,
              "Mega barrier is not aligned");

template <typename T>
static inline T parameter(const uintptr_t * parameters, const uint16_t * indices, uint32_t index) {
    return (T) parameters[indices[index]];
}

static inline void reset_local_resources() {
    // The mega descriptor owns physical Local Resources 0..31.
    __builtin_riscv_tx_base_tcsr_write(UINT32_MAX, 0, 0x035U);
}

static int clear_atomic_cache(uint32_t bytes) {
    if (dayan_task_id(DAYAN_TASK_X) == 0) {
        volatile uint64_t * words = (volatile uint64_t *) shared;
        for (uint32_t index = 0; index < (bytes + sizeof(*words) - 1) / sizeof(*words); ++index) {
            words[index] = 0;
        }
    }
    dayan_cluster_barrier_t * barrier =
        (dayan_cluster_barrier_t *) ((char *) shared + MATMUL_DAYAN_4C_ATOMIC_BYTES);
    return dayan_cluster_barrier(4, barrier);
}

static int prepare_matmul_4c() {
    reset_local_resources();
    const int status = clear_atomic_cache(MATMUL_DAYAN_4C_ATOMIC_BYTES);
    if (status < 0) {
        return status;
    }
    ggml_dayan_matmul_4c_detail::initialize_resources();
    return 0;
}

int norm_pipeline(const uintptr_t * parameters, const uint16_t * indices) {
    using namespace ggml_dayan_matmul_detail;
    run_rms_norm_pipeline(parameter<const DayanNormPipelineConfig *>(parameters, indices, 0),
                          parameter<const void *>(parameters, indices, 1),
                          parameter<const void *>(parameters, indices, 2),
                          parameter<const float *>(parameters, indices, 3),
                          parameter<void *>(parameters, indices, 4),
                          parameter<float *>(parameters, indices, 5),
                          parameter<float *>(parameters, indices, 6),
                          parameter<uint16_t *>(parameters, indices, 7));
    return 0;
}

int residual_add(const uintptr_t * parameters, const uint16_t * indices) {
    using namespace ggml_dayan_matmul_detail;
    run_residual_add<Bf16Format, kDayanDataTypeF32, normalization_load_f32>(
        parameter<const DayanElementwiseConfig *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<const void *>(parameters, indices, 2),
        parameter<void *>(parameters, indices, 3));
    return 0;
}

int residual_add_bf16(const uintptr_t * parameters, const uint16_t * indices) {
    using namespace ggml_dayan_matmul_detail;
    run_residual_add<Bf16Format, kDayanDataTypeBF16,
                     normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>>(
        parameter<const DayanElementwiseConfig *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<const void *>(parameters, indices, 2),
        parameter<void *>(parameters, indices, 3));
    return 0;
}

int rms_norm(const uintptr_t * parameters, const uint16_t * indices) {
    using namespace ggml_dayan_matmul_detail;
    run_rms_norm(parameter<const DayanNormConfig *>(parameters, indices, 0),
                 parameter<const void *>(parameters, indices, 1),
                 parameter<float *>(parameters, indices, 2));
    return 0;
}

int scale(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_scale_f32(
        parameter<const DayanScaleConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<float *>(parameters, indices, 2));
    return 0;
}

int get_rows(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_get_rows(
        parameter<const DayanRowsConfig *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<const void *>(parameters, indices, 2),
        parameter<void *>(parameters, indices, 3));
    return 0;
}

int quantize_mxfp4(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_mxfp4_detail::convert(
        parameter<const DayanMxfp4Config *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<void *>(parameters, indices, 2),
        parameter<void *>(parameters, indices, 3));
    return 0;
}

int matmul_mxfp4(const uintptr_t * parameters, const uint16_t * indices) {
    reset_local_resources();
    return ggml_dayan_mxfp4_detail::run(
        parameter<const DayanMxfp4Config *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<const void *>(parameters, indices, 2),
        parameter<const void *>(parameters, indices, 3),
        parameter<const void *>(parameters, indices, 4),
        parameter<const void *>(parameters, indices, 5),
        parameter<const void *>(parameters, indices, 6),
        parameter<void *>(parameters, indices, 7),
        parameter<void *>(parameters, indices, 8));
}

int concat(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_concat(
        parameter<const DayanConcatConfig *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<const void *>(parameters, indices, 2),
        parameter<void *>(parameters, indices, 3));
    return 0;
}

int copy(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_copy(
        parameter<const DayanCopyConfig *>(parameters, indices, 0),
        parameter<const void *>(parameters, indices, 1),
        parameter<void *>(parameters, indices, 2));
    return 0;
}

int ssm_conv(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_ssm_conv(
        parameter<const DayanSSMConvConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<const float *>(parameters, indices, 2),
        parameter<float *>(parameters, indices, 3));
    return 0;
}

int l2_norm(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_l2_norm(
        parameter<const DayanNormConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<float *>(parameters, indices, 2));
    return 0;
}

int add(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_binary_f32<__tx_vec_vv_fp32_add<0, 0, 16, 0>>(
        parameter<const DayanBinaryConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<const float *>(parameters, indices, 2),
        parameter<float *>(parameters, indices, 3));
    return 0;
}

int multiply(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_binary_f32<__tx_vec_vv_fp32_mul<0, 0, 16, 0>>(
        parameter<const DayanBinaryConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<const float *>(parameters, indices, 2),
        parameter<float *>(parameters, indices, 3));
    return 0;
}

int softplus(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_unary_f32<ggml_dayan_matmul_detail::softplus_tile>(
        parameter<const DayanElementwiseConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<float *>(parameters, indices, 2));
    return 0;
}

int sigmoid(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_unary_f32<ggml_dayan_matmul_detail::sigmoid_tile>(
        parameter<const DayanElementwiseConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<float *>(parameters, indices, 2));
    return 0;
}

int silu(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_unary_f32<ggml_dayan_matmul_detail::silu_tile>(
        parameter<const DayanElementwiseConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<float *>(parameters, indices, 2));
    return 0;
}

int gated_delta_net(const uintptr_t * parameters, const uint16_t * indices) {
    reset_local_resources();
    return (int) ggml_dayan_gdn_detail::run_gated_delta_net(
        parameter<const DayanGatedDeltaNetConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<const float *>(parameters, indices, 2),
        parameter<const float *>(parameters, indices, 3),
        parameter<const float *>(parameters, indices, 4),
        parameter<const float *>(parameters, indices, 5),
        parameter<const float *>(parameters, indices, 6),
        parameter<float *>(parameters, indices, 7));
}

int rope(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_rope_f32(
        parameter<const DayanRopeConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<const int32_t *>(parameters, indices, 2),
        parameter<const float *>(parameters, indices, 3),
        parameter<float *>(parameters, indices, 4));
    return 0;
}

int set_rows(const uintptr_t * parameters, const uint16_t * indices) {
    ggml_dayan_matmul_detail::run_set_rows(
        parameter<const DayanRowsConfig *>(parameters, indices, 0),
        parameter<const float *>(parameters, indices, 1),
        parameter<const void *>(parameters, indices, 2),
        parameter<void *>(parameters, indices, 3));
    return 0;
}

int flash_attention_fp16(const uintptr_t * parameters, const uint16_t * indices) {
    reset_local_resources();
    const int clear_status = clear_atomic_cache(ggml_dayan_flash_attention_detail::kAtomicBytes);
    if (clear_status < 0) {
        return clear_status;
    }
    const DayanFlashAttentionConfig * config =
        parameter<const DayanFlashAttentionConfig *>(parameters, indices, 0);
    const uint16_t * q = parameter<const uint16_t *>(parameters, indices, 1);
    const uint16_t * k = parameter<const uint16_t *>(parameters, indices, 2);
    const uint16_t * v = parameter<const uint16_t *>(parameters, indices, 3);
    float * output = parameter<float *>(parameters, indices, 4);
    return ggml_dayan_flash_attention_detail::run<ggml_dayan_matmul_detail::Fp16Format, 1>(
        config, q, k, v, output);
}

int matmul_bf16(const uintptr_t * parameters, const uint16_t * indices) {
    const int status = prepare_matmul_4c();
    if (status < 0) {
        return status;
    }
    const DayanMatmulConfig * config = parameter<const DayanMatmulConfig *>(parameters, indices, 0);
    const uint16_t * input = parameter<const uint16_t *>(parameters, indices, 1);
    const uint16_t * weight = parameter<const uint16_t *>(parameters, indices, 2);
    float * output = parameter<float *>(parameters, indices, 3);
    return config->m <= 512 ?
               ggml_dayan_matmul_4c_detail::run_matmul_4c<ggml_dayan_matmul_detail::Bf16Format, true>(
                   config, input, weight, output) :
               ggml_dayan_matmul_4c_detail::run_matmul_4c<ggml_dayan_matmul_detail::Bf16Format>(
                   config, input, weight, output);
}

int matmul_pair_bf16(const uintptr_t * parameters, const uint16_t * indices) {
    const int status = prepare_matmul_4c();
    if (status < 0) {
        return status;
    }
    return ggml_dayan_matmul_4c_detail::run_matmul_pair_4c<ggml_dayan_matmul_detail::Bf16Format>(
        parameter<const DayanMatmulPairConfig *>(parameters, indices, 0),
        parameter<const uint16_t *>(parameters, indices, 1),
        parameter<const uint16_t *>(parameters, indices, 2),
        parameter<const uint16_t *>(parameters, indices, 3),
        parameter<float *>(parameters, indices, 4),
        parameter<float *>(parameters, indices, 5));
}

int matmul_swiglu_bf16(const uintptr_t * parameters, const uint16_t * indices) {
    const int status = prepare_matmul_4c();
    if (status < 0) {
        return status;
    }
    const DayanMatmulConfig * config = parameter<const DayanMatmulConfig *>(parameters, indices, 0);
    const uint16_t * input = parameter<const uint16_t *>(parameters, indices, 1);
    const uint16_t * gate = parameter<const uint16_t *>(parameters, indices, 2);
    const uint16_t * up = parameter<const uint16_t *>(parameters, indices, 3);
    uint16_t * output = parameter<uint16_t *>(parameters, indices, 4);
    return config->m <= 512 ?
               ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Bf16Format, false, true>(
                   config, input, gate, up, output) :
               ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Bf16Format, false>(
                   config, input, gate, up, output);
}

int matmul_quantized_bf16(const uintptr_t * parameters, const uint16_t * indices) {
    const int status = prepare_matmul_4c();
    if (status < 0) {
        return status;
    }
    const DayanMatmulConfig * config = parameter<const DayanMatmulConfig *>(parameters, indices, 0);
    const uint16_t * input = parameter<const uint16_t *>(parameters, indices, 1);
    const uint16_t * weight = parameter<const uint16_t *>(parameters, indices, 2);
    uint16_t * output = parameter<uint16_t *>(parameters, indices, 3);
    return config->m <= 512 ?
               ggml_dayan_matmul_4c_detail::run_matmul_quantized_4c<ggml_dayan_matmul_detail::Bf16Format, true>(
                   config, input, weight, output) :
               ggml_dayan_matmul_4c_detail::run_matmul_quantized_4c<ggml_dayan_matmul_detail::Bf16Format>(
                   config, input, weight, output);
}

int gated_output_projection_bf16(const uintptr_t * parameters, const uint16_t * indices) {
    const int status = prepare_matmul_4c();
    if (status < 0) {
        return status;
    }
    return ggml_dayan_matmul_4c_detail::run_gated_output_projection_f32_gate_4c<
        ggml_dayan_matmul_detail::Bf16Format>(
            parameter<const DayanGatedProjectionConfig *>(parameters, indices, 0),
            parameter<const float *>(parameters, indices, 1),
            parameter<const float *>(parameters, indices, 2),
            parameter<const uint16_t *>(parameters, indices, 3),
            parameter<float *>(parameters, indices, 4),
            parameter<uint16_t *>(parameters, indices, 5));
}

template <typename Graph>
static int execute(const uintptr_t * parameters, dayan_cluster_barrier_t * barrier) {
    for (uint16_t index = 0; index < Graph::operation_count; ++index) {
        const Operation & operation = Graph::operations[index];
        if (operation.function == nullptr) {
            const int status = dayan_cluster_barrier(4, barrier);
            if (status < 0) {
                return status;
            }
            continue;
        }
        const int status = operation.function(parameters, operation.arguments);
        if (status < 0) {
            return status;
        }
    }
    return 0;
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
    dayan_cache_invalidate(config, config->total_bytes);

    const void * sources[] = {
        config, root0, root1, root2, root3, root4, root5, root6, root7,
        root8, root9, root10, root11, root12, root13, root14, root15,
    };
    uintptr_t parameters[kDayanMegaMaxParameters];
    for (uint16_t index = 0; index < config->parameter_count; ++index) {
        DAYAN_PARAMS_IF(encoded[index].source > kDayanMegaRootCount || sources[encoded[index].source] == nullptr ||
                        (encoded[index].source == 0 && encoded[index].offset >= config->total_bytes)) {
            DAYAN_PARAMS_ERROR();
        }
        parameters[index] = (uintptr_t) sources[encoded[index].source] + encoded[index].offset;
    }

    dayan_cluster_barrier_t * barrier =
        (dayan_cluster_barrier_t *) ((char *) shared + MATMUL_DAYAN_4C_ATOMIC_BYTES);
    int status;
    switch (config->graph) {
        case kDayanMegaQwen35DenseMxfp4Linear:
            DAYAN_PARAMS_IF(config->parameter_count != ggml_dayan_mega_detail::Qwen35DenseMxfp4LinearGraph::parameter_count) {
                DAYAN_PARAMS_ERROR();
            }
            status = ggml_dayan_mega_detail::execute<ggml_dayan_mega_detail::Qwen35DenseMxfp4LinearGraph>(
                parameters, barrier);
            break;
        case kDayanMegaQwen35DenseMxfp4Full:
            DAYAN_PARAMS_IF(config->parameter_count != ggml_dayan_mega_detail::Qwen35DenseMxfp4FullGraph::parameter_count) {
                DAYAN_PARAMS_ERROR();
            }
            status = ggml_dayan_mega_detail::execute<ggml_dayan_mega_detail::Qwen35DenseMxfp4FullGraph>(
                parameters, barrier);
            break;
        case kDayanMegaQwen35DenseBf16Linear:
            DAYAN_PARAMS_IF(config->parameter_count != ggml_dayan_mega_detail::Qwen35DenseBf16LinearGraph::parameter_count) {
                DAYAN_PARAMS_ERROR();
            }
            status = ggml_dayan_mega_detail::execute<ggml_dayan_mega_detail::Qwen35DenseBf16LinearGraph>(
                parameters, barrier);
            break;
        case kDayanMegaQwen35DenseBf16Full:
            DAYAN_PARAMS_IF(config->parameter_count != ggml_dayan_mega_detail::Qwen35DenseBf16FullGraph::parameter_count) {
                DAYAN_PARAMS_ERROR();
            }
            status = ggml_dayan_mega_detail::execute<ggml_dayan_mega_detail::Qwen35DenseBf16FullGraph>(
                parameters, barrier);
            break;
        default:
            status = -22;
            break;
    }
    DAYAN_PARAMS_IF(status < 0) {
        DAYAN_PARAMS_ERROR();
    }
    (void) status;
}

DAYAN_KERNEL_RESOURCE_DESC_EX(
    qwen35_dense_mega, 1, DAYAN_KERNEL_MAX_CHILD_THREADS, 0,
    MATMUL_DAYAN_4C_ATOMIC_BYTES + sizeof(dayan_cluster_barrier_t), MATMUL_DAYAN_4C_URAM_BYTES,
    DAYAN_TCSR_LOGICAL_GROUPS, DAYAN_TCSR_LOGICAL_GROUPS, 4,
    DAYAN_TCSR_LOGICAL_GROUPS, DAYAN_RESOURCE_LOGICAL_COUNT, DAYAN_RESOURCE_LOGICAL_COUNT);

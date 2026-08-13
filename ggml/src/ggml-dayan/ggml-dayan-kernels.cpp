#include "ggml-dayan-kernels.h"

#include "ggml-dayan-common.h"
#include "ggml-dayan-attention.h"
#include "ggml-dayan-conv.h"
#include "ggml-dayan-elementwise.h"
#include "ggml-dayan-layout.h"
#include "ggml-dayan-matmul.h"
#include "ggml-dayan-memory.h"
#include "ggml-dayan-moe.h"
#include "ggml-dayan-matmul-4c.h"
#include "ggml-dayan-normalization.h"
#include "ggml-dayan-rope.h"
#include "ggml-dayan-softmax.h"

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
#endif

__kernel__ void matmul_bf16_1c(const DayanMatmulConfig * config,
                               const uint16_t *          a,
                               const uint16_t *          bt,
                               float *                   output,
                               DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Bf16Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity>(config, a, bt, nullptr, output, perf);
}

__kernel__ void matmul_fp16_1c(const DayanMatmulConfig * config,
                               const uint16_t *          a,
                               const uint16_t *          bt,
                               float *                   output,
                               DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Fp16Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity>(config, a, bt, nullptr, output, perf);
}

__kernel__ void matmul_bf16_4c(const DayanMatmulConfig * config,
                               const uint16_t *          a,
                               const uint16_t *          bt,
                               float *                   output,
                               DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, a, bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_fp16_4c(const DayanMatmulConfig * config,
                               const uint16_t *          a,
                               const uint16_t *          bt,
                               float *                   output,
                               DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, a, bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_pair_bf16_4c(const DayanMatmulPairConfig * config,
                                    const uint16_t *              a,
                                    const uint16_t *              first_bt,
                                    const uint16_t *              second_bt,
                                    float *                       first,
                                    float *                       second,
                                    DayanMatmulPerf *             perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_pair_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, a, first_bt, second_bt, first, second, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_pair_fp16_4c(const DayanMatmulPairConfig * config,
                                    const uint16_t *              a,
                                    const uint16_t *              first_bt,
                                    const uint16_t *              second_bt,
                                    float *                       first,
                                    float *                       second,
                                    DayanMatmulPerf *             perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_pair_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, a, first_bt, second_bt, first, second, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_quantized_bf16_1c(const DayanMatmulConfig * config,
                                         const uint16_t *          a,
                                         const uint16_t *          bt,
                                         uint16_t *                output,
                                         DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Bf16Format, 1, sizeof(uint16_t), true, epilogue_identity_bf16<0>, epilogue_identity_bf16<4>,
                      epilogue_identity_bf16<2>, epilogue_identity_bf16<0>, epilogue_identity_bf16<4>,
                      epilogue_identity_bf16<2>>(config, a, bt, nullptr, output, perf);
}

__kernel__ void matmul_quantized_fp16_1c(const DayanMatmulConfig * config,
                                         const uint16_t *          a,
                                         const uint16_t *          bt,
                                         uint16_t *                output,
                                         DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Fp16Format, 1, sizeof(uint16_t), true, epilogue_identity_fp16<0>, epilogue_identity_fp16<4>,
                      epilogue_identity_fp16<2>, epilogue_identity_fp16<0>, epilogue_identity_fp16<4>,
                      epilogue_identity_fp16<2>>(config, a, bt, nullptr, output, perf);
}

__kernel__ void matmul_quantized_bf16_4c(const DayanMatmulConfig * config,
                                         const uint16_t *          a,
                                         const uint16_t *          bt,
                                         uint16_t *                output,
                                         DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_quantized_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, a, bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_quantized_fp16_4c(const DayanMatmulConfig * config,
                                         const uint16_t *          a,
                                         const uint16_t *          bt,
                                         uint16_t *                output,
                                         DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_quantized_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, a, bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_swiglu_bf16_1c(const DayanMatmulConfig * config,
                                      const uint16_t *          a,
                                      const uint16_t *          gate_bt,
                                      const uint16_t *          up_bt,
                                      uint16_t *                output,
                                      DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Bf16Format, 2, sizeof(uint16_t), true, epilogue_swiglu_bf16<0>, epilogue_swiglu_bf16<4>,
                      epilogue_swiglu_bf16<2>, epilogue_swiglu_bf16<0>, epilogue_swiglu_bf16<4>,
                      epilogue_swiglu_bf16<2>>(config, a, gate_bt, up_bt, output, perf);
}

__kernel__ void matmul_swiglu_fp16_1c(const DayanMatmulConfig * config,
                                      const uint16_t *          a,
                                      const uint16_t *          gate_bt,
                                      const uint16_t *          up_bt,
                                      uint16_t *                output,
                                      DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Fp16Format, 2, sizeof(uint16_t), true, epilogue_swiglu_fp16<0>, epilogue_swiglu_fp16<4>,
                      epilogue_swiglu_fp16<2>, epilogue_swiglu_fp16<0>, epilogue_swiglu_fp16<4>,
                      epilogue_swiglu_fp16<2>>(config, a, gate_bt, up_bt, output, perf);
}

__kernel__ void matmul_geglu_bf16_1c(const DayanMatmulConfig * config,
                                     const uint16_t *          a,
                                     const uint16_t *          gate_bt,
                                     const uint16_t *          up_bt,
                                     uint16_t *                output,
                                     DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Bf16Format, 2, sizeof(uint16_t), true, epilogue_geglu_bf16<0>, epilogue_geglu_bf16<4>,
                      epilogue_geglu_bf16<2>, epilogue_geglu_bf16<0>, epilogue_geglu_bf16<4>, epilogue_geglu_bf16<2>>(
        config, a, gate_bt, up_bt, output, perf);
}

__kernel__ void matmul_geglu_fp16_1c(const DayanMatmulConfig * config,
                                     const uint16_t *          a,
                                     const uint16_t *          gate_bt,
                                     const uint16_t *          up_bt,
                                     uint16_t *                output,
                                     DayanMatmulPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;

    run_matmul_kernel<Fp16Format, 2, sizeof(uint16_t), true, epilogue_geglu_fp16<0>, epilogue_geglu_fp16<4>,
                      epilogue_geglu_fp16<2>, epilogue_geglu_fp16<0>, epilogue_geglu_fp16<4>, epilogue_geglu_fp16<2>>(
        config, a, gate_bt, up_bt, output, perf);
}

__kernel__ void matmul_swiglu_bf16_4c(const DayanMatmulConfig * config,
                                      const uint16_t *          a,
                                      const uint16_t *          gate_bt,
                                      const uint16_t *          up_bt,
                                      uint16_t *                output,
                                      DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Bf16Format, false>(
        config, a, gate_bt, up_bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_swiglu_fp16_4c(const DayanMatmulConfig * config,
                                      const uint16_t *          a,
                                      const uint16_t *          gate_bt,
                                      const uint16_t *          up_bt,
                                      uint16_t *                output,
                                      DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Fp16Format, false>(
        config, a, gate_bt, up_bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_geglu_bf16_4c(const DayanMatmulConfig * config,
                                     const uint16_t *          a,
                                     const uint16_t *          gate_bt,
                                     const uint16_t *          up_bt,
                                     uint16_t *                output,
                                     DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Bf16Format, true>(
        config, a, gate_bt, up_bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void matmul_geglu_fp16_4c(const DayanMatmulConfig * config,
                                     const uint16_t *          a,
                                     const uint16_t *          gate_bt,
                                     const uint16_t *          up_bt,
                                     uint16_t *                output,
                                     DayanMatmulPerf *         perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_matmul_glu_4c<ggml_dayan_matmul_detail::Fp16Format, true>(
        config, a, gate_bt, up_bt, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_bf16_1c(const DayanMulMatIdConfig * config,
                                   const uint16_t *            activation,
                                   const uint16_t *            weight,
                                   const int32_t *             ids,
                                   float *                     output,
                                   DayanMatmulPerf *           perf) {
    using namespace ggml_dayan_matmul_detail;
    run_mul_mat_id<Bf16Format>(config, activation, weight, ids, output, perf);
}

__kernel__ void mul_mat_id_fp16_1c(const DayanMulMatIdConfig * config,
                                   const uint16_t *            activation,
                                   const uint16_t *            weight,
                                   const int32_t *             ids,
                                   float *                     output,
                                   DayanMatmulPerf *           perf) {
    using namespace ggml_dayan_matmul_detail;
    run_mul_mat_id<Fp16Format>(config, activation, weight, ids, output, perf);
}

__kernel__ void mul_mat_id_bf16_4c(const DayanMulMatIdConfig * config,
                                   const uint16_t *            activation,
                                   const uint16_t *            weight,
                                   const int32_t *             ids,
                                   float *                     output,
                                   DayanMatmulPerf *           perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, activation, weight, ids, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_fp16_4c(const DayanMulMatIdConfig * config,
                                   const uint16_t *            activation,
                                   const uint16_t *            weight,
                                   const int32_t *             ids,
                                   float *                     output,
                                   DayanMatmulPerf *           perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, activation, weight, ids, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_pair_bf16_4c(const DayanMulMatIdPairConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                first_weight,
                                        const uint16_t *                second_weight,
                                        const int32_t *                 ids,
                                        float *                         first_output,
                                        float *                         second_output,
                                        DayanMatmulPerf *               perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_pair_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, activation, first_weight, second_weight, ids, first_output, second_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_pair_fp16_4c(const DayanMulMatIdPairConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                first_weight,
                                        const uint16_t *                second_weight,
                                        const int32_t *                 ids,
                                        float *                         first_output,
                                        float *                         second_output,
                                        DayanMatmulPerf *               perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_pair_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, activation, first_weight, second_weight, ids, first_output, second_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_swiglu_bf16_1c(const DayanMulMatIdConfig * config,
                                          const uint16_t *            activation,
                                          const uint16_t *            gate_weight,
                                          const uint16_t *            up_weight,
                                          const int32_t *             ids,
                                          uint16_t *                  output,
                                          DayanMatmulPerf *           perf) {
    using namespace ggml_dayan_matmul_detail;
    run_mul_mat_id_glu<Bf16Format, epilogue_swiglu_bf16<0>, epilogue_swiglu_bf16<4>, epilogue_swiglu_bf16<2>,
                       epilogue_swiglu_bf16<0>, epilogue_swiglu_bf16<4>, epilogue_swiglu_bf16<2>>(
        config, activation, gate_weight, up_weight, ids, output, perf);
}

__kernel__ void mul_mat_id_swiglu_fp16_1c(const DayanMulMatIdConfig * config,
                                          const uint16_t *            activation,
                                          const uint16_t *            gate_weight,
                                          const uint16_t *            up_weight,
                                          const int32_t *             ids,
                                          uint16_t *                  output,
                                          DayanMatmulPerf *           perf) {
    using namespace ggml_dayan_matmul_detail;
    run_mul_mat_id_glu<Fp16Format, epilogue_swiglu_fp16<0>, epilogue_swiglu_fp16<4>, epilogue_swiglu_fp16<2>,
                       epilogue_swiglu_fp16<0>, epilogue_swiglu_fp16<4>, epilogue_swiglu_fp16<2>>(
        config, activation, gate_weight, up_weight, ids, output, perf);
}

__kernel__ void mul_mat_id_geglu_bf16_1c(const DayanMulMatIdConfig * config,
                                         const uint16_t *            activation,
                                         const uint16_t *            gate_weight,
                                         const uint16_t *            up_weight,
                                         const int32_t *             ids,
                                         uint16_t *                  output,
                                         DayanMatmulPerf *           perf) {
    using namespace ggml_dayan_matmul_detail;
    run_mul_mat_id_glu<Bf16Format, epilogue_geglu_bf16<0>, epilogue_geglu_bf16<4>, epilogue_geglu_bf16<2>,
                       epilogue_geglu_bf16<0>, epilogue_geglu_bf16<4>, epilogue_geglu_bf16<2>>(
        config, activation, gate_weight, up_weight, ids, output, perf);
}

__kernel__ void mul_mat_id_geglu_fp16_1c(const DayanMulMatIdConfig * config,
                                         const uint16_t *            activation,
                                         const uint16_t *            gate_weight,
                                         const uint16_t *            up_weight,
                                         const int32_t *             ids,
                                         uint16_t *                  output,
                                         DayanMatmulPerf *           perf) {
    using namespace ggml_dayan_matmul_detail;
    run_mul_mat_id_glu<Fp16Format, epilogue_geglu_fp16<0>, epilogue_geglu_fp16<4>, epilogue_geglu_fp16<2>,
                       epilogue_geglu_fp16<0>, epilogue_geglu_fp16<4>, epilogue_geglu_fp16<2>>(
        config, activation, gate_weight, up_weight, ids, output, perf);
}

__kernel__ void mul_mat_id_swiglu_bf16_4c(const DayanMulMatIdConfig * config,
                                          const uint16_t *            activation,
                                          const uint16_t *            gate_weight,
                                          const uint16_t *            up_weight,
                                          const int32_t *             ids,
                                          uint16_t *                  output,
                                          DayanMatmulPerf *           perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_glu_4c<
        ggml_dayan_matmul_detail::Bf16Format, false>(
        config, activation, gate_weight, up_weight, ids, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_swiglu_fp16_4c(const DayanMulMatIdConfig * config,
                                          const uint16_t *            activation,
                                          const uint16_t *            gate_weight,
                                          const uint16_t *            up_weight,
                                          const int32_t *             ids,
                                          uint16_t *                  output,
                                          DayanMatmulPerf *           perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_glu_4c<
        ggml_dayan_matmul_detail::Fp16Format, false>(
        config, activation, gate_weight, up_weight, ids, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_geglu_bf16_4c(const DayanMulMatIdConfig * config,
                                         const uint16_t *            activation,
                                         const uint16_t *            gate_weight,
                                         const uint16_t *            up_weight,
                                         const int32_t *             ids,
                                         uint16_t *                  output,
                                         DayanMatmulPerf *           perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_glu_4c<
        ggml_dayan_matmul_detail::Bf16Format, true>(
        config, activation, gate_weight, up_weight, ids, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void mul_mat_id_geglu_fp16_4c(const DayanMulMatIdConfig * config,
                                         const uint16_t *            activation,
                                         const uint16_t *            gate_weight,
                                         const uint16_t *            up_weight,
                                         const int32_t *             ids,
                                         uint16_t *                  output,
                                         DayanMatmulPerf *           perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_mul_mat_id_glu_4c<
        ggml_dayan_matmul_detail::Fp16Format, true>(
        config, activation, gate_weight, up_weight, ids, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void flash_attention_bf16_1c(const DayanFlashAttentionConfig * config,
                                        const uint16_t *                  q,
                                        const uint16_t *                  k,
                                        const uint16_t *                  v,
                                        float *                           output,
                                        int64_t *                         result) {
#if defined(__DAYAN_DEVICE__)
    DAYAN_PARAMS_IF(result == nullptr) {
        return;
    }
#if defined(ENABLE_PARAMS_CHECK)
    *result = -22;
#endif
    DAYAN_PARAMS_IF(config == nullptr) {
        return;
    }

    const long status = flash_attention_lowp_f32<ggml_dayan_matmul_detail::Bf16Format>(
        (void *) q, (void *) k, (void *) v, output, config->batch, config->q_heads, config->kv_heads, config->q_len,
        config->kv_len, config->head_dim, config->scale_bits, config->causal_offset, config->flags,
        config->q_row_stride, config->q_head_stride, config->q_batch_stride, config->k_row_stride,
        config->k_head_stride, config->k_batch_stride, config->v_row_stride, config->v_head_stride,
        config->v_batch_stride);
    if (status < 0) {
        dayan_proc_exit(status);
    }
    if (dayan_task_id(DAYAN_TASK_X) == 0U) {
        *result = status;
    }
#else
    (void) config;
    (void) q;
    (void) k;
    (void) v;
    (void) output;
    (void) result;
#endif
}

__kernel__ void flash_attention_fp16_1c(const DayanFlashAttentionConfig * config,
                                        const uint16_t *                  q,
                                        const uint16_t *                  k,
                                        const uint16_t *                  v,
                                        float *                           output,
                                        int64_t *                         result) {
#if defined(__DAYAN_DEVICE__)
    DAYAN_PARAMS_IF(result == nullptr) {
        return;
    }
#if defined(ENABLE_PARAMS_CHECK)
    *result = -22;
#endif
    DAYAN_PARAMS_IF(config == nullptr) {
        return;
    }

    const long status = flash_attention_lowp_f32<ggml_dayan_matmul_detail::Fp16Format>(
        (void *) q, (void *) k, (void *) v, output, config->batch, config->q_heads, config->kv_heads, config->q_len,
        config->kv_len, config->head_dim, config->scale_bits, config->causal_offset, config->flags,
        config->q_row_stride, config->q_head_stride, config->q_batch_stride, config->k_row_stride,
        config->k_head_stride, config->k_batch_stride, config->v_row_stride, config->v_head_stride,
        config->v_batch_stride);
    if (status < 0) {
        dayan_proc_exit(status);
    }
    if (dayan_task_id(DAYAN_TASK_X) == 0U) {
        *result = status;
    }
#else
    (void) config;
    (void) q;
    (void) k;
    (void) v;
    (void) output;
    (void) result;
#endif
}

__kernel__ void gated_delta_net_f32_1c(const DayanGatedDeltaNetConfig * config,
                                       const float *                    q,
                                       const float *                    k,
                                       const float *                    v,
                                       const float *                    g,
                                       const float *                    beta,
                                       const float *                    state,
                                       float *                          output) {
#if defined(__DAYAN_DEVICE__)
    const long status = ggml_dayan_gdn_detail::run_gated_delta_net(config, q, k, v, g, beta, state, output);
    if (status < 0) {
        dayan_proc_exit(status);
    }
#else
    (void) config;
    (void) q;
    (void) k;
    (void) v;
    (void) g;
    (void) beta;
    (void) state;
    (void) output;
#endif
}

__kernel__ void project_qg_bf16_1c(const DayanAttentionProjectionConfig * config,
                                   const uint16_t *                       activation,
                                   const uint16_t *                       weight,
                                   const float *                          norm,
                                   const int32_t *                        positions,
                                   uint16_t *                             qg_output,
                                   uint16_t *                             q_output,
                                   DayanMatmulPerf *                      perf) {
    using namespace ggml_dayan_matmul_detail;
    run_attention_projection<Bf16Format, true>(config, activation, weight, norm, positions, qg_output, q_output, perf);
}

__kernel__ void project_qg_fp16_1c(const DayanAttentionProjectionConfig * config,
                                   const uint16_t *                       activation,
                                   const uint16_t *                       weight,
                                   const float *                          norm,
                                   const int32_t *                        positions,
                                   uint16_t *                             qg_output,
                                   uint16_t *                             q_output,
                                   DayanMatmulPerf *                      perf) {
    using namespace ggml_dayan_matmul_detail;
    run_attention_projection<Fp16Format, true>(config, activation, weight, norm, positions, qg_output, q_output, perf);
}

__kernel__ void project_qg_bf16_4c(const DayanAttentionProjectionConfig * config,
                                   const uint16_t *                       activation,
                                   const uint16_t *                       weight,
                                   const float *                          norm,
                                   const int32_t *                        positions,
                                   uint16_t *                             qg_output,
                                   uint16_t *                             q_output,
                                   DayanMatmulPerf *                      perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_attention_projection_4c<
        ggml_dayan_matmul_detail::Bf16Format, true>(
        config, activation, weight, norm, positions, qg_output, q_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_qg_fp16_4c(const DayanAttentionProjectionConfig * config,
                                   const uint16_t *                       activation,
                                   const uint16_t *                       weight,
                                   const float *                          norm,
                                   const int32_t *                        positions,
                                   uint16_t *                             qg_output,
                                   uint16_t *                             q_output,
                                   DayanMatmulPerf *                      perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_attention_projection_4c<
        ggml_dayan_matmul_detail::Fp16Format, true>(
        config, activation, weight, norm, positions, qg_output, q_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_k_bf16_1c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  const float *                          norm,
                                  const int32_t *                        positions,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             k_output,
                                  DayanMatmulPerf *                      perf) {
    using namespace ggml_dayan_matmul_detail;
    run_attention_projection<Bf16Format, false>(config, activation, weight, norm, positions, raw_output, k_output,
                                                perf);
}

__kernel__ void project_k_fp16_1c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  const float *                          norm,
                                  const int32_t *                        positions,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             k_output,
                                  DayanMatmulPerf *                      perf) {
    using namespace ggml_dayan_matmul_detail;
    run_attention_projection<Fp16Format, false>(config, activation, weight, norm, positions, raw_output, k_output,
                                                perf);
}

__kernel__ void project_k_bf16_4c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  const float *                          norm,
                                  const int32_t *                        positions,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             k_output,
                                  DayanMatmulPerf *                      perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_attention_projection_4c<
        ggml_dayan_matmul_detail::Bf16Format, false>(
        config, activation, weight, norm, positions, raw_output, k_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_k_fp16_4c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  const float *                          norm,
                                  const int32_t *                        positions,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             k_output,
                                  DayanMatmulPerf *                      perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_attention_projection_4c<
        ggml_dayan_matmul_detail::Fp16Format, false>(
        config, activation, weight, norm, positions, raw_output, k_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_v_bf16_1c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             v_output,
                                  DayanMatmulPerf *                      perf) {
    using namespace ggml_dayan_matmul_detail;
    run_value_projection<Bf16Format>(config, activation, weight, raw_output, v_output, perf);
}

__kernel__ void project_v_fp16_1c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             v_output,
                                  DayanMatmulPerf *                      perf) {
    using namespace ggml_dayan_matmul_detail;
    run_value_projection<Fp16Format>(config, activation, weight, raw_output, v_output, perf);
}

__kernel__ void project_v_bf16_4c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             v_output,
                                  DayanMatmulPerf *                      perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_value_projection_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, activation, weight, raw_output, v_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_v_fp16_4c(const DayanAttentionProjectionConfig * config,
                                  const uint16_t *                       activation,
                                  const uint16_t *                       weight,
                                  uint16_t *                             raw_output,
                                  uint16_t *                             v_output,
                                  DayanMatmulPerf *                      perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_value_projection_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, activation, weight, raw_output, v_output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_k_cache_bf16_1c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const float *                   norm,
                                        const int32_t *                 positions_indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    using namespace ggml_dayan_matmul_detail;
    run_k_cache_projection<Bf16Format>(config, activation, weight, norm, positions_indices, workspace, cache, perf);
}

__kernel__ void project_k_cache_fp16_1c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const float *                   norm,
                                        const int32_t *                 positions_indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    using namespace ggml_dayan_matmul_detail;
    run_k_cache_projection<Fp16Format>(config, activation, weight, norm, positions_indices, workspace, cache, perf);
}

__kernel__ void project_k_cache_bf16_4c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const float *                   norm,
                                        const int32_t *                 positions_indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_k_cache_projection_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, activation, weight, norm, positions_indices, workspace, cache, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_k_cache_fp16_4c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const float *                   norm,
                                        const int32_t *                 positions_indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_k_cache_projection_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, activation, weight, norm, positions_indices, workspace, cache, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_v_cache_bf16_1c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const int32_t *                 indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    using namespace ggml_dayan_matmul_detail;
    run_v_cache_projection<Bf16Format>(config, activation, weight, indices, workspace, cache, perf);
}

__kernel__ void project_v_cache_fp16_1c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const int32_t *                 indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    using namespace ggml_dayan_matmul_detail;
    run_v_cache_projection<Fp16Format>(config, activation, weight, indices, workspace, cache, perf);
}

__kernel__ void project_v_cache_bf16_4c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const int32_t *                 indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_v_cache_projection_4c<ggml_dayan_matmul_detail::Bf16Format>(
        config, activation, weight, indices, workspace, cache, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_v_cache_fp16_4c(const DayanKVProjectionConfig * config,
                                        const uint16_t *                activation,
                                        const uint16_t *                weight,
                                        const int32_t *                 indices,
                                        uint16_t *                      workspace,
                                        uint16_t *                      cache,
                                        DayanMatmulPerf *               perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_v_cache_projection_4c<ggml_dayan_matmul_detail::Fp16Format>(
        config, activation, weight, indices, workspace, cache, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_gated_output_bf16_1c(const DayanGatedProjectionConfig * config,
                                             const float *                      attention,
                                             const uint16_t *                   gate,
                                             const uint16_t *                   weight,
                                             float *                            output,
                                             DayanMatmulPerf *                  perf) {
    using namespace ggml_dayan_matmul_detail;
    run_gated_output_projection<Bf16Format>(config, attention, gate, weight, output, perf);
}

__kernel__ void project_gated_output_fp16_1c(const DayanGatedProjectionConfig * config,
                                             const float *                      attention,
                                             const uint16_t *                   gate,
                                             const uint16_t *                   weight,
                                             float *                            output,
                                             DayanMatmulPerf *                  perf) {
    using namespace ggml_dayan_matmul_detail;
    run_gated_output_projection<Fp16Format>(config, attention, gate, weight, output, perf);
}

__kernel__ void project_gated_output_bf16_4c(const DayanGatedProjectionConfig * config,
                                             const float *                      attention,
                                             const uint16_t *                   gate,
                                             const uint16_t *                   weight,
                                             float *                            output,
                                             DayanMatmulPerf *                  perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_gated_output_projection_4c<
        ggml_dayan_matmul_detail::Bf16Format>(config, attention, gate, weight, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void project_gated_output_fp16_4c(const DayanGatedProjectionConfig * config,
                                             const float *                      attention,
                                             const uint16_t *                   gate,
                                             const uint16_t *                   weight,
                                             float *                            output,
                                             DayanMatmulPerf *                  perf) {
    const int status = ggml_dayan_matmul_4c_detail::run_gated_output_projection_4c<
        ggml_dayan_matmul_detail::Fp16Format>(config, attention, gate, weight, output, perf);
    if (status < 0) {
        dayan_proc_exit(status);
    }
}

__kernel__ void ssm_conv_f32_1c(const DayanSSMConvConfig * config,
                                const float *              input,
                                const float *              weight,
                                float *                    output,
                                DayanSSMConvPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;
    run_ssm_conv(config, input, weight, output, perf);
}

__kernel__ void residual_add_f32_1c(const DayanElementwiseConfig * config,
                                    const float *                  left,
                                    const float *                  right,
                                    float *                        output) {
    using namespace ggml_dayan_matmul_detail;
    run_residual_add(config, left, right, output);
}

__kernel__ void elementwise_add_f32_1c(const DayanElementwiseConfig * config,
                                       const float *                  left,
                                       const float *                  right,
                                       float *                        output) {
    using namespace ggml_dayan_matmul_detail;
    run_binary_f32<elementwise_add_tile>(config, left, right, output);
}

__kernel__ void elementwise_mul_f32_1c(const DayanElementwiseConfig * config,
                                       const float *                  left,
                                       const float *                  right,
                                       float *                        output) {
    using namespace ggml_dayan_matmul_detail;
    run_binary_f32<elementwise_mul_tile>(config, left, right, output);
}

__kernel__ void elementwise_sub_f32_1c(const DayanElementwiseConfig * config,
                                       const float *                  left,
                                       const float *                  right,
                                       float *                        output) {
    using namespace ggml_dayan_matmul_detail;
    run_binary_f32<elementwise_sub_tile>(config, left, right, output);
}

__kernel__ void elementwise_div_f32_1c(const DayanElementwiseConfig * config,
                                       const float *                  left,
                                       const float *                  right,
                                       float *                        output) {
    using namespace ggml_dayan_matmul_detail;
    run_binary_f32<elementwise_div_tile>(config, left, right, output);
}

__kernel__ void scale_f32_1c(const DayanScaleConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_scale_f32(config, input, output);
}

__kernel__ void sigmoid_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<sigmoid_tile>(config, input, output);
}

__kernel__ void silu_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<silu_tile>(config, input, output);
}

__kernel__ void softplus_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<softplus_tile>(config, input, output);
}

__kernel__ void square_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<square_tile>(config, input, output);
}

__kernel__ void sqrt_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<sqrt_tile>(config, input, output);
}

__kernel__ void log_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<log_tile>(config, input, output);
}

__kernel__ void sin_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<sin_tile>(config, input, output);
}

__kernel__ void cos_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<cos_tile>(config, input, output);
}

__kernel__ void abs_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<abs_tile>(config, input, output);
}

__kernel__ void neg_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<neg_tile>(config, input, output);
}

__kernel__ void tanh_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<tanh_tile>(config, input, output);
}

__kernel__ void relu_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<relu_tile>(config, input, output);
}

__kernel__ void gelu_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<gelu_tile>(config, input, output);
}

__kernel__ void gelu_quick_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<gelu_quick_tile>(config, input, output);
}

__kernel__ void exp_f32_1c(const DayanElementwiseConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_unary_f32<exp_tile>(config, input, output);
}

__kernel__ void sum_f32_1c(const DayanReduceConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_reduce_rows_f32(config, input, output);
}

__kernel__ void sum_rows_f32_1c(const DayanReduceConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_reduce_rows_f32(config, input, output);
}

__kernel__ void mean_f32_1c(const DayanReduceConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_reduce_rows_f32(config, input, output);
}

__kernel__ void l2_norm_f32_1c(const DayanNormConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_l2_norm(config, input, output);
}

__kernel__ void get_rows_1c(const DayanRowsConfig * config, const void * source, const void * indices, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_get_rows(config, source, indices, output);
}

__kernel__ void set_rows_1c(const DayanRowsConfig * config,
                            const float *           source,
                            const void *            indices,
                            void *                  destination) {
    using namespace ggml_dayan_matmul_detail;
    run_set_rows(config, source, indices, destination);
}

__kernel__ void copy_f32_1c(const DayanCopyConfig * config, const void * source, void * destination) {
    using namespace ggml_dayan_matmul_detail;
    run_copy(config, source, destination);
}

__kernel__ void repeat_1c(const DayanRepeatConfig * config, const void * source, void * output) {
    using namespace ggml_dayan_matmul_detail;
    run_repeat(config, source, output);
}

__kernel__ void set_1c(const DayanSetConfig * config, const void * base, const void * source, void * output) {
    using namespace ggml_dayan_matmul_detail;
    run_set(config, base, source, output);
}

__kernel__ void diag_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_diag_f32(config, input, output);
}

__kernel__ void diag_mask_inf_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_diag_mask_f32<0xff800000U>(config, input, output);
}

__kernel__ void diag_mask_zero_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_diag_mask_f32<0U>(config, input, output);
}

__kernel__ void pad_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_pad_f32<false>(config, input, output);
}

__kernel__ void pad_circular_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_pad_f32<true>(config, input, output);
}

__kernel__ void pad_reflect_1d_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_pad_reflect_1d_f32(config, input, output);
}

__kernel__ void roll_f32_1c(const DayanLayoutConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_roll_f32(config, input, output);
}

__kernel__ void add_id_f32_1c(const DayanAddIdConfig * config,
                              const float *            input,
                              const float *            bias,
                              const int32_t *          ids,
                              float *                  output) {
    using namespace ggml_dayan_matmul_detail;
    run_add_id(config, input, bias, ids, output);
}

__kernel__ void argsort_f32_1c(const DayanSortConfig * config, const float * input, int32_t * output) {
    using namespace ggml_dayan_matmul_detail;
    run_sort<false>(config, input, output);
}

__kernel__ void top_k_f32_1c(const DayanSortConfig * config, const float * input, int32_t * output) {
    using namespace ggml_dayan_matmul_detail;
    run_sort<true>(config, input, output);
}

__kernel__ void clamp_f32_1c(const DayanClampConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_clamp_f32(config, input, output);
}

__kernel__ void softmax_f32_1c(const DayanSoftmaxConfig * config,
                               const float *              input,
                               const void *               mask,
                               const float *              sinks,
                               float *                    output) {
    using namespace ggml_dayan_matmul_detail;
    run_softmax_f32(config, input, mask, sinks, output);
}

__kernel__ void concat_1c(const DayanConcatConfig * config, const void * left, const void * right, void * output) {
    using namespace ggml_dayan_matmul_detail;
    run_concat(config, left, right, output);
}

__kernel__ void rope_f32_1c(const DayanRopeConfig * config,
                            const float *           input,
                            const int32_t *         positions,
                            const float *           freq_factors,
                            float *                 output) {
    using namespace ggml_dayan_matmul_detail;
    run_rope_f32(config, input, positions, freq_factors, output);
}

__kernel__ void rms_norm_f32_1c(const DayanNormConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_rms_norm(config, input, output);
}

__kernel__ void norm_f32_1c(const DayanNormConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_norm(config, input, output);
}

__kernel__ void group_norm_f32_1c(const DayanGroupNormConfig * config, const float * input, float * output) {
    using namespace ggml_dayan_matmul_detail;
    run_group_norm(config, input, output);
}

__kernel__ void rms_norm_pipeline_1c(const DayanNormPipelineConfig * config,
                                     const float *                   left,
                                     const float *                   right,
                                     const float *                   scale,
                                     float *                         residual,
                                     float *                         rms,
                                     float *                         scaled,
                                     uint16_t *                      lowp) {
    using namespace ggml_dayan_matmul_detail;
    run_rms_norm_pipeline(config, left, right, scale, residual, rms, scaled, lowp);
}

DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_pair_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_pair_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_quantized_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_quantized_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_quantized_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_quantized_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_swiglu_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_swiglu_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_geglu_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(matmul_geglu_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_swiglu_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_swiglu_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_geglu_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_geglu_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(mul_mat_id_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(mul_mat_id_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_pair_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_pair_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(mul_mat_id_swiglu_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(mul_mat_id_swiglu_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(mul_mat_id_geglu_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(mul_mat_id_geglu_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_swiglu_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_swiglu_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_geglu_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(mul_mat_id_geglu_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(flash_attention_bf16_1c, 3, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(flash_attention_fp16_1c, 3, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(gated_delta_net_f32_1c, 0, 0, DAYAN_GATED_DELTA_NET_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_qg_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_qg_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_qg_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_qg_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_k_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_k_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_v_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_v_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_k_cache_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_k_cache_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_cache_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_cache_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_v_cache_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_v_cache_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_cache_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_cache_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_gated_output_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(project_gated_output_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_gated_output_bf16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_gated_output_fp16_4c, MATMUL_DAYAN_4C_WORKERS, MATMUL_DAYAN_4C_ATOMIC_BYTES,
                  MATMUL_DAYAN_4C_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(ssm_conv_f32_1c, 0, 0, DAYAN_SSM_CONV_TASK_URAM_BYTES);
#define DAYAN_NATIVE_KERNEL_DESC(kernel) \
    DAYAN_KERNEL_LOCKED_TASK_DESC(kernel, 0, 0, DAYAN_NATIVE_TASK_URAM_BYTES)

DAYAN_NATIVE_KERNEL_DESC(residual_add_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(elementwise_add_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(elementwise_mul_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(elementwise_sub_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(elementwise_div_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(scale_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(sigmoid_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(silu_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(softplus_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(square_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(sqrt_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(log_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(sin_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(cos_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(abs_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(neg_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(tanh_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(relu_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(gelu_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(gelu_quick_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(exp_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(sum_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(sum_rows_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(mean_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(l2_norm_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(get_rows_1c);
DAYAN_NATIVE_KERNEL_DESC(set_rows_1c);
DAYAN_NATIVE_KERNEL_DESC(copy_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(repeat_1c);
DAYAN_NATIVE_KERNEL_DESC(set_1c);
DAYAN_NATIVE_KERNEL_DESC(diag_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(diag_mask_inf_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(diag_mask_zero_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(pad_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(pad_circular_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(pad_reflect_1d_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(roll_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(add_id_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(clamp_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(softmax_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(concat_1c);
DAYAN_NATIVE_KERNEL_DESC(rope_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(rms_norm_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(norm_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(group_norm_f32_1c);
DAYAN_NATIVE_KERNEL_DESC(rms_norm_pipeline_1c);

#undef DAYAN_NATIVE_KERNEL_DESC

DAYAN_KERNEL_LOCKED_TASK_DESC(argsort_f32_1c, 0, 0, DAYAN_SORT_TASK_URAM_BYTES);
DAYAN_KERNEL_LOCKED_TASK_DESC(top_k_f32_1c, 0, 0, DAYAN_SORT_TASK_URAM_BYTES);

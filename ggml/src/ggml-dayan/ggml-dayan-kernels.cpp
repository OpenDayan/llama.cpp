#include "ggml-dayan-kernels.h"

#include "ggml-dayan-attention.h"
#include "ggml-dayan-conv.h"
#include "ggml-dayan-matmul.h"

#include <dayan_kernel.h>

#if defined(__DAYAN_DEVICE__)
#    include <dayan_sys.h>
#    define _NPU_KERNEL_H_

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

__kernel__ void flash_attention_bf16_1c(const DayanFlashAttentionConfig * config,
                                        const uint16_t *                  q,
                                        const uint16_t *                  k,
                                        const uint16_t *                  v,
                                        float *                           output,
                                        int64_t *                         result) {
#if defined(__DAYAN_DEVICE__)
    if (result == nullptr) {
        return;
    }
    *result = -22;
    if (config == nullptr) {
        return;
    }

    const long status = flash_attention_bf16_f32(
        (void *) q, (void *) k, (void *) v, output, config->batch, config->q_heads, config->kv_heads, config->q_len,
        config->kv_len, config->head_dim, config->scale_bits, config->causal_offset, config->flags);
    *result = status;
#else
    (void) config;
    (void) q;
    (void) k;
    (void) v;
    (void) output;
    (void) result;
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

__kernel__ void project_gated_output_bf16_1c(const DayanGatedProjectionConfig * config,
                                             const float *                      attention,
                                             const uint16_t *                   gate,
                                             uint16_t *                         activation,
                                             const uint16_t *                   weight,
                                             float *                            output,
                                             DayanMatmulPerf *                  perf) {
    using namespace ggml_dayan_matmul_detail;
    run_gated_output_projection<Bf16Format>(config, attention, gate, activation, weight, output, perf);
}

__kernel__ void project_gated_output_fp16_1c(const DayanGatedProjectionConfig * config,
                                             const float *                      attention,
                                             const uint16_t *                   gate,
                                             uint16_t *                         activation,
                                             const uint16_t *                   weight,
                                             float *                            output,
                                             DayanMatmulPerf *                  perf) {
    using namespace ggml_dayan_matmul_detail;
    run_gated_output_projection<Fp16Format>(config, attention, gate, activation, weight, output, perf);
}

__kernel__ void ssm_conv_f32_1c(const DayanSSMConvConfig * config,
                                const float *              input,
                                const float *              weight,
                                float *                    output,
                                DayanSSMConvPerf *         perf) {
    using namespace ggml_dayan_matmul_detail;
    run_ssm_conv(config, input, weight, output, perf);
}

DAYAN_KERNEL_DESC(matmul_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_quantized_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_quantized_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_swiglu_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_swiglu_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_geglu_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(matmul_geglu_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(flash_attention_bf16_1c, 0, 0, 1024U * 1024U);
DAYAN_KERNEL_DESC(project_qg_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_qg_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_cache_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_k_cache_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_cache_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_v_cache_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_gated_output_bf16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(project_gated_output_fp16_1c, MATMUL_DAYAN_1C_SLICES, 0, MATMUL_DAYAN_1C_URAM_BYTES);
DAYAN_KERNEL_DESC(ssm_conv_f32_1c, 0, 0, MATMUL_DAYAN_1C_URAM_BYTES);

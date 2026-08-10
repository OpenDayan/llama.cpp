#pragma once

#include <dayan_kernel.h>
#include <stdint.h>

#ifndef MATMUL_DAYAN_1C_SLICES
#    define MATMUL_DAYAN_1C_SLICES 3
#endif

#ifndef MATMUL_DAYAN_1C_URAM_BYTES
#    define MATMUL_DAYAN_1C_URAM_BYTES (1024U * 1024U)
#endif

struct DayanMatmulConfig {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t output_stride;
};

enum : uint64_t {
    kDayanMatmulPerfMagic = 0x4d4d314350455246ULL,
};

struct DayanMatmulPerf {
    uint64_t magic;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t kernel_cycles;
    uint64_t pipeline_cycles;
};

struct DayanFlashAttentionConfig {
    uint32_t batch;
    uint32_t q_heads;
    uint32_t kv_heads;
    uint32_t q_len;
    uint32_t kv_len;
    uint32_t head_dim;
    uint32_t scale_bits;
    uint32_t causal_offset;
    uint32_t flags;
};

struct DayanAttentionProjectionConfig {
    DayanMatmulConfig matmul;
    uint32_t          tokens;
    uint32_t          heads;
    uint32_t          head_dim;
    uint32_t          n_rot;
    uint32_t          rope_mode;
    uint32_t          norm_eps_bits;
    uint32_t          theta_scale_bits;
    uint32_t          freq_scale_bits;
    uint32_t          attn_factor_bits;
    int32_t           sections[4];
};

struct DayanGatedProjectionConfig {
    DayanMatmulConfig matmul;
    uint32_t          tokens;
    uint32_t          heads;
    uint32_t          head_dim;
};

struct DayanKVProjectionConfig {
    DayanAttentionProjectionConfig projection;
    uint32_t                       cache_rows;
};

struct DayanSSMConvConfig {
    uint32_t input_length;
    uint32_t channels;
    uint32_t tokens;
    uint32_t sequences;
    uint32_t kernel_size;
    uint32_t flags;
};

enum : uint64_t {
    kDayanSSMConvPerfMagic = 0x434f4e5631435046ULL,
};

struct DayanSSMConvPerf {
    uint64_t magic;
    uint64_t cycles;
};

#if !defined(__DAYAN_DEVICE__)
int matmul_bf16_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   npu_buffer_t *                    perf,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_fp16_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   npu_buffer_t *                    perf,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_bf16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             npu_buffer_t *                    perf,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_fp16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             npu_buffer_t *                    perf,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_bf16_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          npu_buffer_t *                    perf,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_fp16_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          npu_buffer_t *                    perf,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_bf16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         npu_buffer_t *                    perf,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_fp16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         npu_buffer_t *                    perf,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_qg_bf16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    norm,
                       npu_buffer_t *                    positions,
                       npu_buffer_t *                    qg_output,
                       npu_buffer_t *                    q_output,
                       npu_buffer_t *                    perf,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int project_qg_fp16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    norm,
                       npu_buffer_t *                    positions,
                       npu_buffer_t *                    qg_output,
                       npu_buffer_t *                    q_output,
                       npu_buffer_t *                    perf,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_bf16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    norm,
                      npu_buffer_t *                    positions,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    k_output,
                      npu_buffer_t *                    perf,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_fp16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    norm,
                      npu_buffer_t *                    positions,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    k_output,
                      npu_buffer_t *                    perf,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_bf16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    v_output,
                      npu_buffer_t *                    perf,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_fp16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    v_output,
                      npu_buffer_t *                    perf,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_cache_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    norm,
                            npu_buffer_t *                    positions_indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            npu_buffer_t *                    perf,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_cache_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    norm,
                            npu_buffer_t *                    positions_indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            npu_buffer_t *                    perf,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_cache_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            npu_buffer_t *                    perf,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_cache_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            npu_buffer_t *                    perf,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_bf16_1c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    attention,
                                 npu_buffer_t *                    gate,
                                 npu_buffer_t *                    activation,
                                 npu_buffer_t *                    weight,
                                 npu_buffer_t *                    output,
                                 npu_buffer_t *                    perf,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_fp16_1c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    attention,
                                 npu_buffer_t *                    gate,
                                 npu_buffer_t *                    activation,
                                 npu_buffer_t *                    weight,
                                 npu_buffer_t *                    output,
                                 npu_buffer_t *                    perf,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int ssm_conv_f32_1c(npu_buffer_t *                    config,
                    npu_buffer_t *                    input,
                    npu_buffer_t *                    weight,
                    npu_buffer_t *                    output,
                    npu_buffer_t *                    perf,
                    const npu_kernel_launch_attrs_t * launch_attrs);
#endif

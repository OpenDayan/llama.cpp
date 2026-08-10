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

struct DayanGatedDeltaNetConfig {
    uint32_t head_dim;
    uint32_t heads;
    uint32_t tokens;
    uint32_t sequences;
    uint32_t snapshots;
    uint32_t gate_width;
    uint32_t scale_bits;
    uint32_t reserved;
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t g_offset;
    uint64_t beta_offset;
    uint64_t input_bytes;
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

struct DayanElementwiseConfig {
    uint64_t elements;
};

struct DayanReduceConfig {
    uint32_t width;
    uint32_t rows;
    uint32_t scale_bits;
    uint32_t reserved;
};

struct DayanScaleConfig {
    uint64_t elements;
    uint32_t scale_bits;
    uint32_t bias_bits;
};

enum : uint32_t {
    kDayanDataTypeF32  = 0,
    kDayanDataTypeBF16 = 1,
    kDayanDataTypeF16  = 2,
};

struct DayanRowsConfig {
    uint32_t row_elements;
    uint32_t rows;
    uint64_t source_rows;
    uint64_t destination_rows;
    uint32_t source_type;
    uint32_t destination_type;
};

struct DayanCopyConfig {
    uint64_t elements;
    uint32_t source_type;
    uint32_t destination_type;
};

struct DayanRepeatConfig {
    uint32_t source_ne[4];
    uint32_t output_ne[4];
    uint32_t type;
    uint32_t reserved;
};

struct DayanLayoutConfig {
    uint32_t input_ne[4];
    uint32_t output_ne[4];
    int32_t  parameters[8];
    uint32_t type;
    uint32_t flags;
};

struct DayanSetConfig {
    uint64_t output_bytes;
    uint32_t source_ne[4];
    uint32_t element_bytes;
    uint32_t destination_nb[3];
    uint32_t destination_offset;
};

struct DayanMulMatIdConfig {
    uint32_t k;
    uint32_t n;
    uint32_t experts;
    uint32_t experts_used;
    uint32_t tokens;
    uint32_t activation_slots;
};

struct DayanAddIdConfig {
    uint32_t width;
    uint32_t experts;
    uint32_t experts_used;
    uint32_t tokens;
};

struct DayanSortConfig {
    uint32_t width;
    uint32_t rows;
    uint32_t output_width;
    uint32_t order;
};

struct DayanClampConfig {
    uint64_t elements;
    uint32_t min_bits;
    uint32_t max_bits;
};

enum : uint32_t {
    kDayanSoftmaxHasMask  = 1U << 0,
    kDayanSoftmaxHasSinks = 1U << 1,
};

struct DayanSoftmaxConfig {
    uint32_t width;
    uint32_t rows;
    uint32_t heads;
    uint32_t batches;
    uint32_t mask_rows;
    uint32_t mask_heads;
    uint32_t mask_batches;
    uint32_t mask_type;
    uint32_t scale_bits;
    uint32_t max_bias_bits;
    uint32_t flags;
    uint32_t reserved;
};

struct DayanConcatConfig {
    uint32_t left_ne[4];
    uint32_t right_ne[4];
    uint32_t output_ne[4];
    uint32_t axis;
    uint32_t type;
};

struct DayanRopeConfig {
    uint32_t width;
    uint32_t heads;
    uint32_t tokens;
    uint32_t batches;
    uint32_t n_dims;
    uint32_t mode;
    uint32_t theta_scale_bits;
    uint32_t freq_scale_bits;
    uint32_t ext_factor_bits;
    uint32_t mscale_bits;
    uint32_t corr_dim0_bits;
    uint32_t corr_dim1_bits;
    int32_t  sections[4];
};

struct DayanNormConfig {
    uint32_t width;
    uint32_t rows;
    uint32_t epsilon_bits;
    uint32_t reserved;
};

struct DayanGroupNormConfig {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t batches;
    uint32_t groups;
    uint32_t epsilon_bits;
};

struct DayanNormMatmulConfig {
    DayanNormConfig   norm;
    DayanMatmulConfig matmul;
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
int mul_mat_id_bf16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    ids,
                       npu_buffer_t *                    output,
                       npu_buffer_t *                    perf,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_fp16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    ids,
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
int gated_delta_net_f32_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    input,
                           npu_buffer_t *                    state,
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
int residual_add_f32_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    left,
                        npu_buffer_t *                    right,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int elementwise_add_f32_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    left,
                           npu_buffer_t *                    right,
                           npu_buffer_t *                    output,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int elementwise_mul_f32_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    left,
                           npu_buffer_t *                    right,
                           npu_buffer_t *                    output,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int elementwise_sub_f32_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    left,
                           npu_buffer_t *                    right,
                           npu_buffer_t *                    output,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int elementwise_div_f32_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    left,
                           npu_buffer_t *                    right,
                           npu_buffer_t *                    output,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int scale_f32_1c(npu_buffer_t *                    config,
                 npu_buffer_t *                    input,
                 npu_buffer_t *                    output,
                 const npu_kernel_launch_attrs_t * launch_attrs);
int sigmoid_f32_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    input,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int silu_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int softplus_f32_1c(npu_buffer_t *                    config,
                    npu_buffer_t *                    input,
                    npu_buffer_t *                    output,
                    const npu_kernel_launch_attrs_t * launch_attrs);
int square_f32_1c(npu_buffer_t *                    config,
                  npu_buffer_t *                    input,
                  npu_buffer_t *                    output,
                  const npu_kernel_launch_attrs_t * launch_attrs);
int sqrt_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int log_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int sin_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int cos_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int abs_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int neg_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int tanh_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int relu_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int gelu_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int gelu_quick_f32_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    input,
                      npu_buffer_t *                    output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int exp_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int sum_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int sum_rows_f32_1c(npu_buffer_t *                    config,
                    npu_buffer_t *                    input,
                    npu_buffer_t *                    output,
                    const npu_kernel_launch_attrs_t * launch_attrs);
int mean_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int l2_norm_f32_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    input,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int get_rows_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    source,
                npu_buffer_t *                    indices,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int set_rows_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    source,
                npu_buffer_t *                    indices,
                npu_buffer_t *                    destination,
                const npu_kernel_launch_attrs_t * launch_attrs);
int copy_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    source,
                npu_buffer_t *                    destination,
                const npu_kernel_launch_attrs_t * launch_attrs);
int repeat_1c(npu_buffer_t *                    config,
              npu_buffer_t *                    source,
              npu_buffer_t *                    output,
              const npu_kernel_launch_attrs_t * launch_attrs);
int set_1c(npu_buffer_t *                    config,
           npu_buffer_t *                    base,
           npu_buffer_t *                    source,
           npu_buffer_t *                    output,
           const npu_kernel_launch_attrs_t * launch_attrs);
int diag_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int diag_mask_inf_f32_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    input,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int diag_mask_zero_f32_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    input,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int pad_f32_1c(npu_buffer_t *                    config,
               npu_buffer_t *                    input,
               npu_buffer_t *                    output,
               const npu_kernel_launch_attrs_t * launch_attrs);
int pad_circular_f32_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    input,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int pad_reflect_1d_f32_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    input,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int roll_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int add_id_f32_1c(npu_buffer_t *                    config,
                  npu_buffer_t *                    input,
                  npu_buffer_t *                    bias,
                  npu_buffer_t *                    ids,
                  npu_buffer_t *                    output,
                  const npu_kernel_launch_attrs_t * launch_attrs);
int argsort_f32_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    input,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int top_k_f32_1c(npu_buffer_t *                    config,
                 npu_buffer_t *                    input,
                 npu_buffer_t *                    output,
                 const npu_kernel_launch_attrs_t * launch_attrs);
int clamp_f32_1c(npu_buffer_t *                    config,
                 npu_buffer_t *                    input,
                 npu_buffer_t *                    output,
                 const npu_kernel_launch_attrs_t * launch_attrs);
int softmax_f32_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    input,
                   npu_buffer_t *                    mask,
                   npu_buffer_t *                    sinks,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int concat_1c(npu_buffer_t *                    config,
              npu_buffer_t *                    left,
              npu_buffer_t *                    right,
              npu_buffer_t *                    output,
              const npu_kernel_launch_attrs_t * launch_attrs);
int rope_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    positions,
                npu_buffer_t *                    freq_factors,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int rms_norm_f32_1c(npu_buffer_t *                    config,
                    npu_buffer_t *                    input,
                    npu_buffer_t *                    output,
                    const npu_kernel_launch_attrs_t * launch_attrs);
int norm_f32_1c(npu_buffer_t *                    config,
                npu_buffer_t *                    input,
                npu_buffer_t *                    output,
                const npu_kernel_launch_attrs_t * launch_attrs);
int group_norm_f32_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    input,
                      npu_buffer_t *                    output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int rms_norm_scale_f32_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    input,
                          npu_buffer_t *                    scale,
                          npu_buffer_t *                    rms,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int rms_norm_scale_bf16_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    input,
                           npu_buffer_t *                    scale,
                           npu_buffer_t *                    rms,
                           npu_buffer_t *                    output,
                           npu_buffer_t *                    lowp,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int rms_norm_scale_fp16_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    input,
                           npu_buffer_t *                    scale,
                           npu_buffer_t *                    rms,
                           npu_buffer_t *                    output,
                           npu_buffer_t *                    lowp,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int add_rms_norm_scale_f32_1c(npu_buffer_t *                    config,
                              npu_buffer_t *                    left,
                              npu_buffer_t *                    right,
                              npu_buffer_t *                    scale,
                              npu_buffer_t *                    residual,
                              npu_buffer_t *                    rms,
                              npu_buffer_t *                    output,
                              const npu_kernel_launch_attrs_t * launch_attrs);
int add_rms_norm_scale_bf16_1c(npu_buffer_t *                    config,
                               npu_buffer_t *                    left,
                               npu_buffer_t *                    right,
                               npu_buffer_t *                    scale,
                               npu_buffer_t *                    residual,
                               npu_buffer_t *                    rms,
                               npu_buffer_t *                    output,
                               npu_buffer_t *                    lowp,
                               const npu_kernel_launch_attrs_t * launch_attrs);
int add_rms_norm_scale_fp16_1c(npu_buffer_t *                    config,
                               npu_buffer_t *                    left,
                               npu_buffer_t *                    right,
                               npu_buffer_t *                    scale,
                               npu_buffer_t *                    residual,
                               npu_buffer_t *                    rms,
                               npu_buffer_t *                    output,
                               npu_buffer_t *                    lowp,
                               const npu_kernel_launch_attrs_t * launch_attrs);
int rms_norm_matmul_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    input,
                            npu_buffer_t *                    scale,
                            npu_buffer_t *                    rms,
                            npu_buffer_t *                    scaled,
                            npu_buffer_t *                    lowp,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    output,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int rms_norm_matmul_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    input,
                            npu_buffer_t *                    scale,
                            npu_buffer_t *                    rms,
                            npu_buffer_t *                    scaled,
                            npu_buffer_t *                    lowp,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    output,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int add_rms_norm_matmul_bf16_1c(npu_buffer_t *                    config,
                                npu_buffer_t *                    left,
                                npu_buffer_t *                    right,
                                npu_buffer_t *                    scale,
                                npu_buffer_t *                    intermediates,
                                npu_buffer_t *                    weight,
                                npu_buffer_t *                    output,
                                const npu_kernel_launch_attrs_t * launch_attrs);
int add_rms_norm_matmul_fp16_1c(npu_buffer_t *                    config,
                                npu_buffer_t *                    left,
                                npu_buffer_t *                    right,
                                npu_buffer_t *                    scale,
                                npu_buffer_t *                    intermediates,
                                npu_buffer_t *                    weight,
                                npu_buffer_t *                    output,
                                const npu_kernel_launch_attrs_t * launch_attrs);
#endif

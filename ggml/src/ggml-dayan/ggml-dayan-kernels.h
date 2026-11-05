#pragma once

#include "ggml-dayan-cluster.h"

#include <dayan_kernel.h>
#include <stdint.h>

#ifndef MATMUL_DAYAN_1C_URAM_BYTES
#    define MATMUL_DAYAN_1C_URAM_BYTES (768U * 1024U)
#endif

#ifndef MATMUL_DAYAN_4C_ATOMIC_BYTES
#    define MATMUL_DAYAN_4C_ATOMIC_BYTES 256U
#endif

#ifndef MATMUL_DAYAN_4C_URAM_BYTES
#    define MATMUL_DAYAN_4C_URAM_BYTES (1536U * 1024U)
#endif

#ifndef DAYAN_NATIVE_TASK_URAM_BYTES
#    define DAYAN_NATIVE_TASK_URAM_BYTES (512U * 1024U)
#endif

#ifndef DAYAN_NORM_TASK_URAM_BYTES
#    define DAYAN_NORM_TASK_URAM_BYTES (512U * 1024U)
#endif

#ifndef DAYAN_GET_ROWS_URAM_BYTES
#    define DAYAN_GET_ROWS_URAM_BYTES (512U * 1024U)
#endif

enum : uint32_t {
    kDayanNativeTaskTileElements = 8192U,
    kDayanNativeMaxTasks         = 4U,
    kDayanSSMConvMaxTasks         = 16U,
    kDayanSSMConvTokenTile        = 64U,
    kDayanGatedDeltaNetMaxTasks   = 16U,
};

#ifndef DAYAN_SORT_TASK_URAM_BYTES
#    define DAYAN_SORT_TASK_URAM_BYTES (512U * 1024U)
#endif

#ifndef DAYAN_SSM_CONV_TASK_URAM_BYTES
#    define DAYAN_SSM_CONV_TASK_URAM_BYTES (512U * 1024U)
#endif

#ifndef DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES
#    define DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES (512U * 1024U)
#endif

#ifndef DAYAN_GATED_DELTA_NET_URAM_BYTES
#    define DAYAN_GATED_DELTA_NET_URAM_BYTES DAYAN_GATED_DELTA_NET_TASK_URAM_BYTES
#endif

#ifndef DAYAN_GATED_DELTA_NET_CHUNK_URAM_BYTES
#    define DAYAN_GATED_DELTA_NET_CHUNK_URAM_BYTES (768U * 1024U)
#endif

#ifndef DAYAN_ROPE_MAX_PAIRS
#    define DAYAN_ROPE_MAX_PAIRS 256U
#endif

static inline uint64_t dayan_partition_begin(uint64_t elements, uint32_t index, uint32_t count, uint64_t alignment = 64) {
    const uint64_t blocks = (elements + alignment - 1) / alignment;
    const uint64_t offset = blocks * index / count * alignment;
    return offset < elements ? offset : elements;
}

#if !defined(__DAYAN_DEVICE__)
struct DayanTopology {
    uint32_t cores = 0;
    uint32_t clusters = 0;

    bool init(const npu_runtime_topology_t & info) {
        cores = clusters = 0;
        if (info.cluster_count == 0 || info.cluster_count > 32 || info.cores_per_cluster == 0 || info.cores_per_cluster > 32 ||
            !info.enabled_cluster_mask || !info.enabled_core_mask ||
            (info.cluster_count < 32 && (info.enabled_cluster_mask >> info.cluster_count)) ||
            (info.cores_per_cluster < 32 && (info.enabled_core_mask >> info.cores_per_cluster))) {
            return false;
        }
        uint32_t enabled_clusters = 0, enabled_cores = 0;
        for (uint32_t mask = info.enabled_cluster_mask; mask; mask &= mask - 1) { ++enabled_clusters; }
        for (uint32_t mask = info.enabled_core_mask; mask; mask &= mask - 1) { ++enabled_cores; }
        cores = enabled_clusters * enabled_cores;
        clusters = info.cores_per_cluster == 4 && info.enabled_core_mask == 0xf ? enabled_clusters : 0;
        return true;
    }

    NPULaunchConfig launch(bool cooperative, uint64_t work = UINT64_MAX) const {
        const bool four_core = cooperative && clusters != 0;
        uint32_t count = four_core ? clusters : cores;
        if (work < count) { count = (uint32_t) work; }
        if (count == 0) { count = 1; }
        return {four_core ? NPUTaskDim3{4, count, 1} : NPUTaskDim3{count, 1, 1},
                four_core ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore};
    }
};
#endif

struct DayanMatmulConfig {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t a_stride;
    uint64_t bt_stride;
    uint64_t output_stride;
    uint64_t batch_ne2            = 0;
    uint64_t batch_ne3            = 0;
    uint64_t weight_ne2           = 0;
    uint64_t weight_ne3           = 0;
    uint64_t a_batch_stride2      = 0;
    uint64_t a_batch_stride3      = 0;
    uint64_t bt_batch_stride2     = 0;
    uint64_t bt_batch_stride3     = 0;
    uint64_t output_batch_stride2 = 0;
    uint64_t output_batch_stride3 = 0;
    uint64_t cluster_workspace = 0;
    uint64_t uram_offset = 0;
    uint64_t uram_size = 0;
};

struct DayanMatmulPairConfig {
    DayanMatmulConfig first;
    DayanMatmulConfig second;
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
    uint64_t q_row_stride;
    uint64_t q_head_stride;
    uint64_t q_batch_stride;
    uint64_t k_row_stride;
    uint64_t k_head_stride;
    uint64_t k_batch_stride;
    uint64_t v_row_stride;
    uint64_t v_head_stride;
    uint64_t v_batch_stride;
    uint64_t cluster_workspace = 0;
};

static inline bool dayan_flash_attention_use_split_kv(uint32_t q_len, uint32_t kv_len) {
    return q_len == 1 && kv_len >= 512;
}

struct DayanGatedDeltaNetConfig {
    uint32_t head_dim;
    uint32_t heads;
    uint32_t tokens;
    uint32_t sequences;
    uint32_t snapshots;
    uint32_t gate_width;
    uint32_t scale_bits;
    uint32_t q_heads;
    uint32_t k_heads;
    uint32_t q_sequences;
    uint32_t k_sequences;
    uint32_t reserved;
    uint64_t q_nb[4];
    uint64_t k_nb[4];
    uint64_t v_nb[4];
    uint64_t g_nb[4];
    uint64_t beta_nb[4];
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
    uint32_t neg_log2e_bits;
    uint32_t one_bits;
};

enum : uint32_t {
    kDayanSSMConvFlagSilu        = 1U << 0,
    kDayanSSMConvFlagNoStateCopy = 1U << 1,
};

static inline uint32_t dayan_ssm_conv_task_count(uint32_t channels, uint32_t tokens, uint32_t sequences, uint32_t max_tasks = kDayanSSMConvMaxTasks) {
    const uint64_t channel_groups = ((uint64_t) channels + 63U) / 64U;
    const uint64_t row_groups     = ((uint64_t) tokens + kDayanSSMConvTokenTile - 1U) / kDayanSSMConvTokenTile * sequences;
    const uint64_t tasks = (channel_groups < max_tasks ? channel_groups : max_tasks) *
                           (row_groups < max_tasks ? row_groups : max_tasks);
    return (uint32_t) (tasks < max_tasks ? tasks : max_tasks);
}

enum : uint32_t {
    kDayanSSMConvMaxKernelSize = 63,
};

struct DayanElementwiseConfig {
    uint64_t elements;
};

// Zero row strides select contiguous storage. Each right_ne[d] must divide ne[d].
struct DayanBinaryConfig {
    uint64_t elements;
    uint32_t ne[4];
    uint32_t right_ne[4];
    uint64_t left_nb[3];
    uint64_t right_nb[3];
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
    kDayanDataTypeI32  = 3,
    kDayanDataTypeI64  = 4,
    kDayanDataTypeMXFP4 = 5,
};

struct DayanMxfp4Config {
    uint64_t m, n, k;
    uint64_t a_stride, b_stride, output_stride;
    uint64_t a_scale_stride, b_scale_stride, output_scale_stride;
    uint32_t source_type, weight_type, output_type, operation;
    uint32_t share_inputs;
    uint64_t cluster_workspace = 0;
};

struct DayanRowsConfig {
    uint32_t row_elements;
    uint32_t rows;
    uint64_t source_rows;
    uint64_t destination_rows;
    uint32_t source_type;
    uint32_t destination_type;
    uint32_t index_type;
    uint32_t flags;
    uint32_t index_ne[3];
    uint64_t index_nb[3];
    uint32_t source_ne[3];
    uint32_t destination_ne[3];
    uint64_t source_row_stride;
    uint64_t source_scale_stride;
    uint64_t source_scale_offset;
};

enum : uint32_t {
    kDayanRowsSet = 1U << 0,
};

struct DayanCopyConfig {
    uint64_t elements;
    uint32_t source_type;
    uint32_t destination_type;
    uint32_t source_ne[4];
    uint64_t source_nb[4];
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
    uint64_t ids_row_stride;
};

struct DayanMulMatIdPairConfig {
    DayanMulMatIdConfig first;
    DayanMulMatIdConfig second;
};

struct DayanMxfp4MoeConfig {
    DayanMxfp4Config matmul;
    DayanMulMatIdConfig routing;
    uint32_t grouped;
    uint32_t reserved;
    uint64_t workspace_bytes;
};

struct DayanFfnConfig {
    DayanMxfp4Config gate;
    DayanMxfp4Config down;
    uint32_t experts;
    uint32_t experts_used;
};

struct DayanMoeDispatchLayout {
    uint64_t count_stride;
    uint64_t position_stride;
    // Per-core maps from local token/rank order to global grouped rows.
    uint64_t positions_offset;
    uint64_t offsets_offset;
    uint64_t bytes;

    static DayanMoeDispatchLayout make(uint32_t tokens, uint32_t experts, uint32_t used, uint32_t cores = 4) {
        const uint64_t counts = ((uint64_t) experts * sizeof(uint32_t) + 255U) & ~255ULL;
        const uint64_t positions = ((((uint64_t) tokens + cores - 1U) / cores) * used * sizeof(uint32_t) + 63U) & ~63ULL;
        const uint64_t offsets = (((uint64_t) experts + 1U) * sizeof(uint32_t) + 255U) & ~255ULL;
        return {counts, positions, 4U * counts, 4U * counts + cores * positions, 4U * counts + cores * positions + offsets};
    }
};

struct DayanAddIdConfig {
    uint32_t width;
    uint32_t experts;
    uint32_t experts_used;
    uint32_t tokens;
    uint64_t ids_row_stride;
};

struct DayanSortConfig {
    uint32_t width;
    uint32_t rows;
    uint32_t output_width;
    uint32_t order;
    uint32_t output_stride;
};

struct DayanMoeSortConfig {
    DayanSortConfig sort;
    uint32_t experts_used;
    uint64_t workspace_bytes;
};

enum : uint32_t {
    kDayanSortMaxWidth = 4096,
    kDayanSortOrderAsc  = 0,
    kDayanSortOrderDesc = 1,
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
    uint64_t left_nb[4];
    uint64_t right_nb[4];
    uint64_t output_nb[4];
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
    uint32_t axis_values[3];
    uint32_t axes[DAYAN_ROPE_MAX_PAIRS];
    uint32_t multipliers[DAYAN_ROPE_MAX_PAIRS];
};

struct DayanNormConfig {
    uint32_t width;
    uint32_t rows;
    uint32_t epsilon_bits;
    uint32_t inv_width_bits;
    uint32_t input_type;
    uint32_t input_ne[4];
    uint64_t input_nb[4];
};

enum : uint32_t {
    kDayanNormHasResidual   = 1U << 0,
    kDayanNormWriteResidual = 1U << 1,
    kDayanNormWriteRms      = 1U << 2,
    kDayanNormWriteScaled   = 1U << 3,
    kDayanNormWriteLowp     = 1U << 4,
};

struct DayanNormPipelineConfig {
    DayanNormConfig norm;
    uint32_t        flags;
    uint32_t        output_type;
};

struct DayanGdnZGateConfig {
    DayanGatedDeltaNetConfig gdn;
    DayanCopyConfig          state_copy;
    DayanNormPipelineConfig  norm_pipeline;
    DayanMatmulConfig        z_bf16;
    DayanElementwiseConfig   z_silu;
    DayanBinaryConfig        gate_mul;
};

struct DayanGroupNormConfig {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t batches;
    uint32_t groups;
    uint32_t epsilon_bits;
    uint32_t inv_group_bits;
    uint32_t inv_last_group_bits;
};

#if !defined(__DAYAN_DEVICE__)
int moe_ffn_decode(npu_buffer_t * config, npu_buffer_t * input, npu_buffer_t * input_scales,
                   npu_buffer_t * gate, npu_buffer_t * gate_scales, npu_buffer_t * up, npu_buffer_t * up_scales,
                   npu_buffer_t * down, npu_buffer_t * down_scales, npu_buffer_t * selected, npu_buffer_t * weights,
                   npu_buffer_t * output, const npu_kernel_launch_attrs_t * attrs);
int router_f32(npu_buffer_t * config, npu_buffer_t * activation, npu_buffer_t * weight, npu_buffer_t * output,
               const npu_kernel_launch_attrs_t * attrs);
int mul_mat_id_mxfp4(npu_buffer_t * config, npu_buffer_t * a, npu_buffer_t * as, npu_buffer_t * b, npu_buffer_t * bs,
                     npu_buffer_t * up, npu_buffer_t * ups, npu_buffer_t * ids, npu_buffer_t * output, npu_buffer_t * scales,
                     npu_buffer_t * workspace,
                     const npu_kernel_launch_attrs_t * attrs);
int mul_mat_id_mxfp4_small(npu_buffer_t * config, npu_buffer_t * a, npu_buffer_t * as, npu_buffer_t * b, npu_buffer_t * bs,
                     npu_buffer_t * up, npu_buffer_t * ups, npu_buffer_t * ids, npu_buffer_t * output, npu_buffer_t * scales,
                     npu_buffer_t * workspace,
                     const npu_kernel_launch_attrs_t * attrs);
int quantize_mxfp4(npu_buffer_t * config, npu_buffer_t * input, npu_buffer_t * output, npu_buffer_t * scales,
                   const npu_kernel_launch_attrs_t * attrs);
int matmul_mxfp4_1c(npu_buffer_t * config, npu_buffer_t * a, npu_buffer_t * as, npu_buffer_t * b, npu_buffer_t * bs,
                     npu_buffer_t * up, npu_buffer_t * ups, npu_buffer_t * output, npu_buffer_t * scales,
                     const npu_kernel_launch_attrs_t * attrs);
int matmul_mxfp4_4c(npu_buffer_t * config, npu_buffer_t * a, npu_buffer_t * as, npu_buffer_t * b, npu_buffer_t * bs,
                     npu_buffer_t * up, npu_buffer_t * ups, npu_buffer_t * output, npu_buffer_t * scales,
                     const npu_kernel_launch_attrs_t * attrs);
int matmul_bf16_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_inference_bf16_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_fp16_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_inference_fp16_1c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_bf16_4c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_inference_bf16_4c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_fp16_4c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_inference_fp16_4c(npu_buffer_t *                    config,
                   npu_buffer_t *                    a,
                   npu_buffer_t *                    bt,
                   npu_buffer_t *                    output,
                   const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_bf16_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_bf16_4c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_inference_bf16_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_inference_bf16_4c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_fp16_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_fp16_4c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_inference_fp16_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_silu_inference_fp16_4c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    bt,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_pair_bf16_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    first_bt,
                        npu_buffer_t *                    second_bt,
                        npu_buffer_t *                    first,
                        npu_buffer_t *                    second,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_pair_bf16_4c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    first_bt,
                        npu_buffer_t *                    second_bt,
                        npu_buffer_t *                    first,
                        npu_buffer_t *                    second,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_pair_fp16_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    first_bt,
                        npu_buffer_t *                    second_bt,
                        npu_buffer_t *                    first,
                        npu_buffer_t *                    second,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_pair_fp16_4c(npu_buffer_t *                    config,
                        npu_buffer_t *                    a,
                        npu_buffer_t *                    first_bt,
                        npu_buffer_t *                    second_bt,
                        npu_buffer_t *                    first,
                        npu_buffer_t *                    second,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_bf16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_inference_bf16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_fp16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_inference_fp16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_bf16_4c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_inference_bf16_4c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_fp16_4c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_quantized_inference_fp16_4c(npu_buffer_t *                    config,
                             npu_buffer_t *                    a,
                             npu_buffer_t *                    bt,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_bf16_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_inference_bf16_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_fp16_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_inference_fp16_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_bf16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_inference_bf16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_fp16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_inference_fp16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_bf16_4c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_inference_bf16_4c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_fp16_4c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_swiglu_inference_fp16_4c(npu_buffer_t *                    config,
                          npu_buffer_t *                    a,
                          npu_buffer_t *                    gate_bt,
                          npu_buffer_t *                    up_bt,
                          npu_buffer_t *                    output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_bf16_4c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_inference_bf16_4c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_fp16_4c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int matmul_geglu_inference_fp16_4c(npu_buffer_t *                    config,
                         npu_buffer_t *                    a,
                         npu_buffer_t *                    gate_bt,
                         npu_buffer_t *                    up_bt,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_bf16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    ids,
                       npu_buffer_t *                    output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_fp16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    ids,
                       npu_buffer_t *                    output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_bf16_4c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    ids,
                       npu_buffer_t *                    output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_fp16_4c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    ids,
                       npu_buffer_t *                    output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_pair_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    first_weight,
                            npu_buffer_t *                    second_weight,
                            npu_buffer_t *                    ids,
                            npu_buffer_t *                    first_output,
                            npu_buffer_t *                    second_output,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_pair_bf16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    first_weight,
                            npu_buffer_t *                    second_weight,
                            npu_buffer_t *                    ids,
                            npu_buffer_t *                    first_output,
                            npu_buffer_t *                    second_output,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_pair_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    first_weight,
                            npu_buffer_t *                    second_weight,
                            npu_buffer_t *                    ids,
                            npu_buffer_t *                    first_output,
                            npu_buffer_t *                    second_output,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_pair_fp16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    first_weight,
                            npu_buffer_t *                    second_weight,
                            npu_buffer_t *                    ids,
                            npu_buffer_t *                    first_output,
                            npu_buffer_t *                    second_output,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_swiglu_bf16_1c(npu_buffer_t *                    config,
                              npu_buffer_t *                    activation,
                              npu_buffer_t *                    gate_weight,
                              npu_buffer_t *                    up_weight,
                              npu_buffer_t *                    ids,
                              npu_buffer_t *                    output,
                              const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_swiglu_fp16_1c(npu_buffer_t *                    config,
                              npu_buffer_t *                    activation,
                              npu_buffer_t *                    gate_weight,
                              npu_buffer_t *                    up_weight,
                              npu_buffer_t *                    ids,
                              npu_buffer_t *                    output,
                              const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_geglu_bf16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    activation,
                             npu_buffer_t *                    gate_weight,
                             npu_buffer_t *                    up_weight,
                             npu_buffer_t *                    ids,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_geglu_fp16_1c(npu_buffer_t *                    config,
                             npu_buffer_t *                    activation,
                             npu_buffer_t *                    gate_weight,
                             npu_buffer_t *                    up_weight,
                             npu_buffer_t *                    ids,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_swiglu_bf16_4c(npu_buffer_t *                    config,
                              npu_buffer_t *                    activation,
                              npu_buffer_t *                    gate_weight,
                              npu_buffer_t *                    up_weight,
                              npu_buffer_t *                    ids,
                              npu_buffer_t *                    output,
                              const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_swiglu_fp16_4c(npu_buffer_t *                    config,
                              npu_buffer_t *                    activation,
                              npu_buffer_t *                    gate_weight,
                              npu_buffer_t *                    up_weight,
                              npu_buffer_t *                    ids,
                              npu_buffer_t *                    output,
                              const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_geglu_bf16_4c(npu_buffer_t *                    config,
                             npu_buffer_t *                    activation,
                             npu_buffer_t *                    gate_weight,
                             npu_buffer_t *                    up_weight,
                             npu_buffer_t *                    ids,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int mul_mat_id_geglu_fp16_4c(npu_buffer_t *                    config,
                             npu_buffer_t *                    activation,
                             npu_buffer_t *                    gate_weight,
                             npu_buffer_t *                    up_weight,
                             npu_buffer_t *                    ids,
                             npu_buffer_t *                    output,
                             const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_bf16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_fp16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_decode_bf16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int flash_attention_decode_fp16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    q,
                            npu_buffer_t *                    k,
                            npu_buffer_t *                    v,
                            npu_buffer_t *                    output,
                            npu_buffer_t *                    result,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int gated_delta_net_f32_1c(npu_buffer_t *                    config,
                           npu_buffer_t *                    q,
                           npu_buffer_t *                    k,
                           npu_buffer_t *                    v,
                           npu_buffer_t *                    g,
                           npu_buffer_t *                    beta,
                           npu_buffer_t *                    state,
                           npu_buffer_t *                    output,
                           const npu_kernel_launch_attrs_t * launch_attrs);
int gated_delta_net_chunk_f32_1c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    q,
                                 npu_buffer_t *                    k,
                                 npu_buffer_t *                    v,
                                 npu_buffer_t *                    g,
                                 npu_buffer_t *                    beta,
                                 npu_buffer_t *                    state,
                                 npu_buffer_t *                    output,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int project_qg_bf16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    norm,
                       npu_buffer_t *                    positions,
                       npu_buffer_t *                    qg_output,
                       npu_buffer_t *                    q_output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int project_qg_fp16_1c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    norm,
                       npu_buffer_t *                    positions,
                       npu_buffer_t *                    qg_output,
                       npu_buffer_t *                    q_output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int project_qg_bf16_4c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    norm,
                       npu_buffer_t *                    positions,
                       npu_buffer_t *                    qg_output,
                       npu_buffer_t *                    q_output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int project_qg_fp16_4c(npu_buffer_t *                    config,
                       npu_buffer_t *                    activation,
                       npu_buffer_t *                    weight,
                       npu_buffer_t *                    norm,
                       npu_buffer_t *                    positions,
                       npu_buffer_t *                    qg_output,
                       npu_buffer_t *                    q_output,
                       const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_bf16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    norm,
                      npu_buffer_t *                    positions,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    k_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_fp16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    norm,
                      npu_buffer_t *                    positions,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    k_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_bf16_4c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    norm,
                      npu_buffer_t *                    positions,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    k_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_fp16_4c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    norm,
                      npu_buffer_t *                    positions,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    k_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_bf16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    v_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_fp16_1c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    v_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_bf16_4c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    v_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_fp16_4c(npu_buffer_t *                    config,
                      npu_buffer_t *                    activation,
                      npu_buffer_t *                    weight,
                      npu_buffer_t *                    raw_output,
                      npu_buffer_t *                    v_output,
                      const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_cache_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    norm,
                            npu_buffer_t *                    positions_indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_cache_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    norm,
                            npu_buffer_t *                    positions_indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_cache_bf16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    norm,
                            npu_buffer_t *                    positions_indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_k_cache_fp16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    norm,
                            npu_buffer_t *                    positions_indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_cache_bf16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_cache_fp16_1c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_cache_bf16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_v_cache_fp16_4c(npu_buffer_t *                    config,
                            npu_buffer_t *                    activation,
                            npu_buffer_t *                    weight,
                            npu_buffer_t *                    indices,
                            npu_buffer_t *                    workspace,
                            npu_buffer_t *                    cache,
                            const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_bf16_1c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    attention,
                                 npu_buffer_t *                    gate,
                                 npu_buffer_t *                    weight,
                                 npu_buffer_t *                    output,
                                 npu_buffer_t *                    scratch,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_fp16_1c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    attention,
                                 npu_buffer_t *                    gate,
                                 npu_buffer_t *                    weight,
                                 npu_buffer_t *                    output,
                                 npu_buffer_t *                    scratch,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_bf16_4c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    attention,
                                 npu_buffer_t *                    gate,
                                 npu_buffer_t *                    weight,
                                 npu_buffer_t *                    output,
                                 npu_buffer_t *                    scratch,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_fp16_4c(npu_buffer_t *                    config,
                                 npu_buffer_t *                    attention,
                                 npu_buffer_t *                    gate,
                                 npu_buffer_t *                    weight,
                                 npu_buffer_t *                    output,
                                 npu_buffer_t *                    scratch,
                                 const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_f32_gate_bf16_1c(npu_buffer_t *                    config,
                                          npu_buffer_t *                    attention,
                                          npu_buffer_t *                    gate,
                                          npu_buffer_t *                    weight,
                                          npu_buffer_t *                    output,
                                          npu_buffer_t *                    scratch,
                                          const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_f32_gate_bf16_4c(npu_buffer_t *                    config,
                                          npu_buffer_t *                    attention,
                                          npu_buffer_t *                    gate,
                                          npu_buffer_t *                    weight,
                                          npu_buffer_t *                    output,
                                          npu_buffer_t *                    scratch,
                                          const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_f32_gate_fp16_1c(npu_buffer_t *                    config,
                                          npu_buffer_t *                    attention,
                                          npu_buffer_t *                    gate,
                                          npu_buffer_t *                    weight,
                                          npu_buffer_t *                    output,
                                          npu_buffer_t *                    scratch,
                                          const npu_kernel_launch_attrs_t * launch_attrs);
int project_gated_output_f32_gate_fp16_4c(npu_buffer_t *                    config,
                                          npu_buffer_t *                    attention,
                                          npu_buffer_t *                    gate,
                                          npu_buffer_t *                    weight,
                                          npu_buffer_t *                    output,
                                          npu_buffer_t *                    scratch,
                                          const npu_kernel_launch_attrs_t * launch_attrs);
int ssm_conv_f32_1c(npu_buffer_t *                    config,
                    npu_buffer_t *                    input,
                    npu_buffer_t *                    weight,
                    npu_buffer_t *                    output,
                    const npu_kernel_launch_attrs_t * launch_attrs);
int ssm_conv_split_f32_1c(npu_buffer_t *                    config,
                          npu_buffer_t *                    state,
                          npu_buffer_t *                    weight,
                          npu_buffer_t *                    output,
                          npu_buffer_t *                    qkv,
                          npu_buffer_t *                    state_output,
                          const npu_kernel_launch_attrs_t * launch_attrs);
int residual_add_f32_1c(npu_buffer_t *                    config,
                        npu_buffer_t *                    left,
                        npu_buffer_t *                    right,
                        npu_buffer_t *                    output,
                        const npu_kernel_launch_attrs_t * launch_attrs);
int residual_add_bf16_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    left,
                         npu_buffer_t *                    right,
                         npu_buffer_t *                    output,
                         const npu_kernel_launch_attrs_t * launch_attrs);
int residual_add_fp16_1c(npu_buffer_t *                    config,
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
int rms_norm_pipeline_1c(npu_buffer_t *                    config,
                         npu_buffer_t *                    left,
                         npu_buffer_t *                    right,
                         npu_buffer_t *                    scale,
                         npu_buffer_t *                    residual,
                         npu_buffer_t *                    rms,
                         npu_buffer_t *                    scaled,
                         npu_buffer_t *                    lowp,
                         const npu_kernel_launch_attrs_t * launch_attrs);
#endif

#pragma once

#include "ggml-dayan-attention.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct RopeUramLayout {
    enum : uint64_t {
        kScalarStride = 32,
        kHeadStride   = 2 * Constants::kVecBytes,
        kMaxHeads     = 13,
        kValues       = 0,
        kAxes         = kValues + kMaxHeads * kHeadStride,
        kMultiplier   = kAxes + Constants::kVecBytes,
        kFactors      = kMultiplier + Constants::kVecBytes,
        kPositionScratch = kFactors + Constants::kVecBytes,
        kPositions    = kPositionScratch + 2 * kScalarStride,
        kTotal        = kPositions + 4 * kScalarStride,
    };
};

static_assert(RopeUramLayout::kTotal <= DAYAN_NATIVE_TASK_URAM_BYTES, "RoPE URAM layout is too large");

template <long Group>
static inline void configure_rope_lsu_2d(uint32_t dim0,
                                         uint64_t input_dim0_stride,
                                         uint64_t output_dim0_stride,
                                         uint32_t dim1,
                                         uint64_t input_dim1_stride,
                                         uint64_t output_dim1_stride,
                                         uint32_t read_bytes,
                                         uint32_t write_bytes) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(dim1 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(6);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(read_bytes - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(write_bytes - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <long Group>
static inline void configure_rope_lsu(uint32_t dim0,
                                      uint64_t input_stride,
                                      uint64_t output_stride,
                                      uint32_t read_bytes,
                                      uint32_t write_bytes) {
    configure_rope_lsu_2d<Group>(dim0, input_stride, output_stride, 1, 0, 0, read_bytes, write_bytes);
}

static inline void rope_load_triplet(uint64_t output0,
                                     uint64_t input0,
                                     uint64_t output1,
                                     uint64_t input1,
                                     uint64_t output2,
                                     uint64_t input2,
                                     uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (full != 0) {
        configure_rope_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_d2u<0>(output0, input0);
        dayan_wait_lsu();
        __tx_lsu_dma_d2u<0>(output1, input1);
        dayan_wait_lsu();
        __tx_lsu_dma_d2u<0>(output2, input2);
        dayan_wait_lsu();
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu<0>(1, 0, 0, tail, 64);
        __tx_lsu_dma_d2u<0>(output0 + offset, input0 + offset);
        dayan_wait_lsu();
        __tx_lsu_dma_d2u<0>(output1 + offset, input1 + offset);
        dayan_wait_lsu();
        __tx_lsu_dma_d2u<0>(output2 + offset, input2 + offset);
        dayan_wait_lsu();
    }
}

template <bool Pair>
static inline void rope_load_rows(uint64_t output0,
                                  uint64_t input0,
                                  uint64_t output1,
                                  uint64_t input1,
                                  uint32_t bytes,
                                  uint32_t rows,
                                  uint32_t input_row_stride) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (full != 0) {
        configure_rope_lsu<0>(full, 64, 64, 64, 64);
        for (uint32_t row = 0; row < rows; ++row) {
            __tx_lsu_dma_d2u<0>(output0 + (uint64_t) row * RopeUramLayout::kHeadStride,
                                input0 + (uint64_t) row * input_row_stride);
            dayan_wait_lsu();
            if constexpr (Pair) {
                __tx_lsu_dma_d2u<0>(output1 + (uint64_t) row * RopeUramLayout::kHeadStride,
                                    input1 + (uint64_t) row * input_row_stride);
                dayan_wait_lsu();
            }
        }
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu<0>(1, 0, 0, tail, 64);
        for (uint32_t row = 0; row < rows; ++row) {
            __tx_lsu_dma_d2u<0>(output0 + (uint64_t) row * RopeUramLayout::kHeadStride + offset,
                                input0 + (uint64_t) row * input_row_stride + offset);
            dayan_wait_lsu();
            if constexpr (Pair) {
                __tx_lsu_dma_d2u<0>(output1 + (uint64_t) row * RopeUramLayout::kHeadStride + offset,
                                    input1 + (uint64_t) row * input_row_stride + offset);
                dayan_wait_lsu();
            }
        }
    }
}

template <bool Pair>
static inline void rope_store_rows(uint64_t output0,
                                   uint64_t input0,
                                   uint64_t output1,
                                   uint64_t input1,
                                   uint32_t bytes,
                                   uint32_t rows,
                                   uint32_t output_row_stride) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (full != 0) {
        configure_rope_lsu<1>(full, 64, 64, 64, 64);
        for (uint32_t row = 0; row < rows; ++row) {
            __tx_lsu_dma_u2d<1>(output0 + (uint64_t) row * output_row_stride,
                                input0 + (uint64_t) row * RopeUramLayout::kHeadStride);
            dayan_wait_lsu();
            if constexpr (Pair) {
                __tx_lsu_dma_u2d<1>(output1 + (uint64_t) row * output_row_stride,
                                    input1 + (uint64_t) row * RopeUramLayout::kHeadStride);
                dayan_wait_lsu();
            }
        }
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu<1>(1, 0, 0, 64, tail);
        for (uint32_t row = 0; row < rows; ++row) {
            __tx_lsu_dma_u2d<1>(output0 + (uint64_t) row * output_row_stride + offset,
                                input0 + (uint64_t) row * RopeUramLayout::kHeadStride + offset);
            dayan_wait_lsu();
            if constexpr (Pair) {
                __tx_lsu_dma_u2d<1>(output1 + (uint64_t) row * output_row_stride + offset,
                                    input1 + (uint64_t) row * RopeUramLayout::kHeadStride + offset);
                dayan_wait_lsu();
            }
        }
    }
}

static inline void rope_copy_rows(uint64_t output,
                                  uint64_t input,
                                  uint32_t bytes,
                                  uint32_t rows,
                                  uint32_t row_stride) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;
    if (full != 0) {
        configure_rope_lsu_2d<0>(full, 64, 64, rows, row_stride, row_stride, 64, 64);
        __tx_lsu_dma_d2d<0>(output, input);
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu_2d<0>(1, 0, 0, rows, row_stride, row_stride, tail, tail);
        __tx_lsu_dma_d2d<0>(output + offset, input + offset);
    }
}

static inline void emit_rope_axis_select(uint64_t axes, uint32_t axis_value, uint64_t position) {
    __tx_vec_lsu_u2v<1, 2>(axes);
    __tx_vec_lsu_g2v_broadcast32<1, 1>(axis_value);
    __tx_vec_vv_int32_cmp<16, 17, 0, 0b010>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(position);
    __tx_vec_vv_mux2<0, 0, 16, 0>();
}

static inline void emit_rope_common(uint64_t axes,
                                    uint64_t positions,
                                    uint64_t multiplier,
                                    uint64_t factors,
                                    uint32_t mscale_bits) {
    __tx_vec_lsu_u2v_broadcast32<0, 2>(positions + 3U * RopeUramLayout::kScalarStride);
    emit_rope_axis_select(axes, 2, positions + 2U * RopeUramLayout::kScalarStride);
    emit_rope_axis_select(axes, 1, positions + RopeUramLayout::kScalarStride);
    emit_rope_axis_select(axes, 0, positions);
    __tx_vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(multiplier);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(factors);
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 2>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>(mscale_bits);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_shuffle_xor<1, 2>(0);
    __tx_vec_v_fp32_sfu<1, 16, 3>();
    __tx_vec_vv_fp32_mul<1, 16, 18, 0>();
}

static inline void issue_rope_split(uint64_t values,
                                    uint64_t axes,
                                    uint64_t positions,
                                    uint64_t multiplier,
                                    uint64_t factors,
                                    uint32_t mscale_bits) {
    constexpr long instructions = 38;
    begin_vec_block(instructions);
    emit_rope_common(axes, positions, multiplier, factors, mscale_bits);
    __tx_vec_lsu_u2v<0, 0>(values);
    __tx_vec_lsu_u2v<1, 0>(values + Constants::kVecBytes);
    __tx_vec_vv_fp32_mul<0, 0, 17, 0>();
    __tx_vec_vv_fp32_mul<1, 16, 2, 0>();
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_vv_fp32_mul<1, 2, 3, 0>();
    __tx_vec_vv_fp32_mul<0, 18, 19, 0>();
    __tx_vec_vv_fp32_add<1, 16, 0, 0>();
    __tx_vec_lsu_v2u<1, 3>(values);
    __tx_vec_lsu_v2u<16, 3>(values + Constants::kVecBytes);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void issue_rope_interleaved(uint64_t values,
                                          uint64_t axes,
                                          uint64_t positions,
                                          uint64_t multiplier,
                                          uint64_t factors,
                                          uint32_t mscale_bits) {
    constexpr long instructions = 39;
    begin_vec_block(instructions);
    emit_rope_common(axes, positions, multiplier, factors, mscale_bits);
    __tx_vec_shuffle_compress<0, 0>(32);
    __tx_vec_shuffle_compress<1, 16>(32);
    __tx_vec_lsu_u2v<0, 0>(values);
    __tx_vec_shuffle_xor<1, 0>(1);
    __tx_vec_vv_fp32_mul<0, 0, 17, 0>();
    __tx_vec_vv_fp32_mul<1, 16, 2, 0>();
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_vv_fp32_add<1, 1, 16, 0>();
    __tx_vec_lsu_g2s<0>((long) 0xaaaaaaaaaaaaaaaaULL);
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>(values);
    start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void run_rope_f32(const DayanRopeConfig * config,
                                const float *           input,
                                const int32_t *         positions,
                                const float *           freq_factors,
                                float *                 output) {
    DAYAN_PARAMS_IF(config == nullptr || input == nullptr || positions == nullptr || freq_factors == nullptr ||
                    output == nullptr || config->width == 0 || config->heads == 0 || config->tokens == 0 ||
                    config->batches == 0 || config->n_dims == 0 || config->n_dims > config->width ||
                    config->n_dims / 2U > DAYAN_ROPE_MAX_PAIRS || (config->n_dims & 1U) != 0U ||
                    (config->mode != 0U && config->mode != 2U && config->mode != 8U && config->mode != 40U)) {
        DAYAN_PARAMS_ERROR();
    }
    if (dayan_uram_base() == nullptr || dayan_uram_size() < RopeUramLayout::kTotal) {
        dayan_thread_exit(-12);
    }

    const uint32_t position_axes = (config->mode & 8U) != 0U ? 4U : 1U;
    const uint32_t row_bytes     = config->width * sizeof(float);

    const uint64_t uram        = reinterpret_cast<uint64_t>(dayan_uram_base());
    const uint64_t values      = uram + RopeUramLayout::kValues;
    const uint64_t axes        = uram + RopeUramLayout::kAxes;
    const uint64_t multiplier  = uram + RopeUramLayout::kMultiplier;
    const uint64_t factors     = uram + RopeUramLayout::kFactors;
    const uint64_t position_scratch = uram + RopeUramLayout::kPositionScratch;
    const uint64_t position_in = uram + RopeUramLayout::kPositions;

    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    const uint32_t half  = config->n_dims / 2U;
    const uint64_t token_rows = (uint64_t) config->batches * config->tokens;
    const uint32_t heads_per_tile = min_int((int) config->heads, (int) RopeUramLayout::kMaxHeads);
    const uint32_t head_tiles = (config->heads + heads_per_tile - 1U) / heads_per_tile;
    const uint64_t work_items = token_rows * head_tiles;

    configure_vec_iterator<kVecV0Base>(0, 0, 0, RopeUramLayout::kHeadStride);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, RopeUramLayout::kHeadStride);

    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint64_t token_row = work / head_tiles;
        const uint32_t head_base = (uint32_t) (work % head_tiles) * heads_per_tile;
        const uint32_t head_count = min_int((int) (config->heads - head_base), (int) heads_per_tile);
        const uint32_t token = (uint32_t) (token_row % config->tokens);
        const uint64_t row = token_row * config->heads + head_base;
        const float *  source = input + row * config->width;
        float *        destination = output + row * config->width;

        configure_rope_lsu<0>(1, 0, 0, 64, 64);
        for (uint32_t axis = 0; axis < position_axes; ++axis) {
            const int32_t * position_source = positions + token + (uint64_t) axis * config->tokens;
            const uint64_t position_address = reinterpret_cast<uint64_t>(position_source);
            __tx_lsu_dma_d2u<0>(position_scratch, position_address & ~63ULL);
            dayan_wait_lsu();
            dayan_cache_invalidate(reinterpret_cast<void *>(position_scratch), 64);
            *reinterpret_cast<int32_t *>(position_in + (uint64_t) axis * RopeUramLayout::kScalarStride) =
                *reinterpret_cast<const int32_t *>(position_scratch + (position_address & 63U));
        }
        dayan_cache_flush(reinterpret_cast<void *>(position_in), position_axes * RopeUramLayout::kScalarStride);

        const uint32_t pairs_per_tile = config->mode == 0U ? Constants::kVecLanes / 2U : Constants::kVecLanes;
        for (uint32_t pair_base = 0; pair_base < half; pair_base += pairs_per_tile) {
            const uint32_t pair_count =
                half - pair_base < pairs_per_tile ? half - pair_base : pairs_per_tile;
            const uint32_t pair_bytes = pair_count * sizeof(float);
            rope_load_triplet(axes, reinterpret_cast<uint64_t>(config->axes + pair_base), multiplier,
                              reinterpret_cast<uint64_t>(config->multipliers + pair_base), factors,
                              reinterpret_cast<uint64_t>(freq_factors + pair_base), pair_bytes);
            dayan_wait_lsu();
            if (config->mode == 0U) {
                rope_load_rows<false>(values, reinterpret_cast<uint64_t>(source + 2U * pair_base), 0, 0,
                                      2U * pair_bytes, head_count, row_bytes);
            } else {
                rope_load_rows<true>(values, reinterpret_cast<uint64_t>(source + pair_base),
                                     values + Constants::kVecBytes,
                                     reinterpret_cast<uint64_t>(source + half + pair_base), pair_bytes, head_count,
                                     row_bytes);
            }

            configure_vec_grid(1, head_count);

            if (config->mode == 0U) {
                issue_rope_interleaved(values, axes, position_in, multiplier, factors, config->mscale_bits);
                rope_store_rows<false>(reinterpret_cast<uint64_t>(destination + 2U * pair_base), values, 0, 0,
                                       2U * pair_bytes, head_count, row_bytes);
            } else {
                issue_rope_split(values, axes, position_in, multiplier, factors, config->mscale_bits);
                rope_store_rows<true>(reinterpret_cast<uint64_t>(destination + pair_base), values,
                                      reinterpret_cast<uint64_t>(destination + half + pair_base),
                                      values + Constants::kVecBytes, pair_bytes, head_count, row_bytes);
            }
        }

        const uint32_t tail_bytes = (config->width - config->n_dims) * sizeof(float);
        if (tail_bytes != 0) {
            rope_copy_rows(reinterpret_cast<uint64_t>(destination + config->n_dims),
                           reinterpret_cast<uint64_t>(source + config->n_dims), tail_bytes, head_count, row_bytes);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

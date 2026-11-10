/*
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#pragma once

#include "ggml-dayan-attention.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct RopeUramLayout {
    enum : uint64_t {
        kScalarStride = 32,
        kRowStride   = 2 * Constants::kVecBytes,
        kTileRows     = 256,
        kValues       = 0,
        kAxes         = kValues + kTileRows * kRowStride,
        kMultiplier   = kAxes + DAYAN_ROPE_MAX_PAIRS * sizeof(float),
        kFactors      = kMultiplier + DAYAN_ROPE_MAX_PAIRS * sizeof(float),
        kTokenStride  = 4 * kScalarStride,
        kPositions    = kFactors + DAYAN_ROPE_MAX_PAIRS * sizeof(float),
        kTotal        = kPositions + kTileRows * kTokenStride,
    };
};

static_assert(RopeUramLayout::kTotal <= 256U * 1024U, "RoPE worker URAM layout is too large");

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
        __tx_lsu_dma_d2u<0>(output1, input1);
        __tx_lsu_dma_d2u<0>(output2, input2);
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu<0>(1, 0, 0, tail, 64);
        __tx_lsu_dma_d2u<0>(output0 + offset, input0 + offset);
        __tx_lsu_dma_d2u<0>(output1 + offset, input1 + offset);
        __tx_lsu_dma_d2u<0>(output2 + offset, input2 + offset);
    }
    dayan_wait_lsu();
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
        configure_rope_lsu_2d<0>(full, 64, 64, rows, input_row_stride, RopeUramLayout::kRowStride, 64, 64);
        __tx_lsu_dma_d2u<0>(output0, input0);
        if constexpr (Pair) {
            __tx_lsu_dma_d2u<0>(output1, input1);
        }
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu_2d<0>(1, 0, 0, rows, input_row_stride, RopeUramLayout::kRowStride, tail, 64);
        __tx_lsu_dma_d2u<0>(output0 + offset, input0 + offset);
        if constexpr (Pair) {
            __tx_lsu_dma_d2u<0>(output1 + offset, input1 + offset);
        }
    }
    dayan_wait_lsu();
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
        configure_rope_lsu_2d<1>(full, 64, 64, rows, RopeUramLayout::kRowStride, output_row_stride, 64, 64);
        __tx_lsu_dma_u2d<1>(output0, input0);
        if constexpr (Pair) {
            __tx_lsu_dma_u2d<1>(output1, input1);
        }
    }
    if (tail != 0) {
        const uint64_t offset = (uint64_t) full * 64U;
        configure_rope_lsu_2d<1>(1, 0, 0, rows, RopeUramLayout::kRowStride, output_row_stride, 64, tail);
        __tx_lsu_dma_u2d<1>(output0 + offset, input0 + offset);
        if constexpr (Pair) {
            __tx_lsu_dma_u2d<1>(output1 + offset, input1 + offset);
        }
    }
    dayan_wait_lsu();
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
    __tx_vec_lsu_u2v<1, 1>(axes);
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
    emit_rope_reduce_angle();
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
    constexpr long instructions = 38 + kRopeReduceInstructions;
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
    constexpr long instructions = 39 + kRopeReduceInstructions;
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
                                float *                 output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {(uint64_t) config->width * config->heads * config->tokens * config->batches, (uint64_t) config->tokens * config->batches * ((config->heads + RopeUramLayout::kTileRows - 1) / RopeUramLayout::kTileRows), 256U * 1024U}, [&](const NativeContext & worker) {
            run_rope_f32(config, input, positions, freq_factors, output, worker);
        })) { return; }

    DAYAN_PARAMS_IF(config == nullptr || input == nullptr || positions == nullptr || freq_factors == nullptr ||
                    output == nullptr || config->width == 0 || config->heads == 0 || config->tokens == 0 ||
                    config->batches == 0 || config->n_dims == 0 || config->n_dims > config->width ||
                    config->n_dims / 2U > DAYAN_ROPE_MAX_PAIRS || (config->n_dims & 1U) != 0U ||
                    (config->mode != 0U && config->mode != 2U && config->mode != 8U && config->mode != 40U)) {
        DAYAN_PARAMS_ERROR();
    }
    DAYAN_PARAMS_IF(native_uram(context) == 0 || native_uram_size(context) < RopeUramLayout::kTotal) {
        DAYAN_PARAMS_ERROR();
    }

    const uint32_t position_axes = (config->mode & 8U) != 0U ? 4U : 1U;
    const uint32_t row_bytes     = config->width * sizeof(float);

    const uint64_t uram        = native_uram(context);
    const uint64_t values      = uram + RopeUramLayout::kValues;
    const uint64_t axes        = uram + RopeUramLayout::kAxes;
    const uint64_t multiplier  = uram + RopeUramLayout::kMultiplier;
    const uint64_t factors     = uram + RopeUramLayout::kFactors;
    const uint64_t position_in = uram + RopeUramLayout::kPositions;

    const uint32_t task  = dayan_native_task_id(context);
    const uint32_t tasks = dayan_native_task_count(context);
    const uint32_t half  = config->n_dims / 2U;
    const uint32_t heads_per_tile = min_int((int) config->heads, (int) RopeUramLayout::kTileRows);
    const uint32_t head_tiles = (config->heads + heads_per_tile - 1U) / heads_per_tile;
    const uint64_t task_tokens = ((uint64_t) config->tokens * config->batches * head_tiles + tasks - 1) / tasks;
    const uint32_t capacity_tokens = RopeUramLayout::kTileRows / heads_per_tile;
    const uint32_t tokens_per_tile = task_tokens < capacity_tokens ? task_tokens : capacity_tokens;
    const uint32_t token_tiles = (config->tokens + tokens_per_tile - 1U) / tokens_per_tile;
    const uint64_t work_items = (uint64_t) config->batches * token_tiles * head_tiles;
    if (task >= work_items) { return; }

    rope_load_triplet(axes, reinterpret_cast<uint64_t>(config->axes), multiplier,
                      reinterpret_cast<uint64_t>(config->multipliers), factors,
                      reinterpret_cast<uint64_t>(freq_factors), half * sizeof(float));
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, RopeUramLayout::kTokenStride);

    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint64_t token_tile = work / head_tiles;
        const uint32_t head_base = (uint32_t) (work % head_tiles) * heads_per_tile;
        const uint32_t head_count = min_int((int) (config->heads - head_base), (int) heads_per_tile);
        const uint32_t token = (uint32_t) (token_tile % token_tiles) * tokens_per_tile;
        const uint32_t token_count = min_int((int) (config->tokens - token), (int) tokens_per_tile);
        const uint64_t batch = token_tile / token_tiles;
        const uint64_t row = (batch * config->tokens + token) * config->heads + head_base;
        const uint32_t rows = head_count * token_count;
        const float * source = input + row * config->width;
        float * destination = output + row * config->width;

        configure_rope_lsu_2d<0>(token_count, sizeof(int32_t), RopeUramLayout::kTokenStride, 4,
                                  position_axes == 1 ? 0 : (uint64_t) config->tokens * sizeof(int32_t),
                                  RopeUramLayout::kScalarStride, sizeof(int32_t), sizeof(int32_t));
        __tx_lsu_dma_d2u<0>(position_in, reinterpret_cast<uint64_t>(positions + token));
        dayan_wait_lsu();
        configure_vec_iterator<kVecV0Base>(0, 0, RopeUramLayout::kRowStride, head_count * RopeUramLayout::kRowStride);
        configure_vec_iterator<kVecV3Base>(0, 0, RopeUramLayout::kRowStride, head_count * RopeUramLayout::kRowStride);

        const uint32_t pairs_per_tile = config->mode == 0U ? Constants::kVecLanes / 2U : Constants::kVecLanes;
        for (uint32_t pair_base = 0; pair_base < half; pair_base += pairs_per_tile) {
            const uint32_t pair_count =
                half - pair_base < pairs_per_tile ? half - pair_base : pairs_per_tile;
            const uint32_t pair_bytes = pair_count * sizeof(float);
            if (config->mode == 0U) {
                rope_load_rows<false>(values, reinterpret_cast<uint64_t>(source + 2U * pair_base), 0, 0,
                                      2U * pair_bytes, rows, row_bytes);
            } else {
                rope_load_rows<true>(values, reinterpret_cast<uint64_t>(source + pair_base),
                                     values + Constants::kVecBytes,
                                     reinterpret_cast<uint64_t>(source + half + pair_base), pair_bytes, rows,
                                     row_bytes);
            }

            configure_vec_grid(head_count, token_count);

            if (config->mode == 0U) {
                issue_rope_interleaved(values, axes + pair_base * sizeof(float), position_in, multiplier + pair_base * sizeof(float), factors + pair_base * sizeof(float), config->mscale_bits);
                rope_store_rows<false>(reinterpret_cast<uint64_t>(destination + 2U * pair_base), values, 0, 0,
                                       2U * pair_bytes, rows, row_bytes);
            } else {
                issue_rope_split(values, axes + pair_base * sizeof(float), position_in, multiplier + pair_base * sizeof(float), factors + pair_base * sizeof(float), config->mscale_bits);
                rope_store_rows<true>(reinterpret_cast<uint64_t>(destination + pair_base), values,
                                      reinterpret_cast<uint64_t>(destination + half + pair_base),
                                      values + Constants::kVecBytes, pair_bytes, rows, row_bytes);
            }
        }

        const uint32_t tail_bytes = (config->width - config->n_dims) * sizeof(float);
        if (tail_bytes != 0) {
            rope_copy_rows(reinterpret_cast<uint64_t>(destination + config->n_dims),
                           reinterpret_cast<uint64_t>(source + config->n_dims), tail_bytes, rows, row_bytes);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

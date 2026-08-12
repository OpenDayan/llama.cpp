#pragma once

#include "ggml-dayan-attention.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct RopeUramLayout {
    enum : uint64_t {
        kX0         = 0,
        kX1         = kX0 + Constants::kVecBytes,
        kOutput0    = kX1 + Constants::kVecBytes,
        kOutput1    = kOutput0 + Constants::kVecBytes,
        kFactors    = kOutput1 + Constants::kVecBytes,
        kAxes       = kFactors + Constants::kVecBytes,
        kMultiplier = kAxes + Constants::kVecBytes,
        kPosition   = kMultiplier + Constants::kVecBytes,
        kTheta      = kPosition + Constants::kVecBytes,
        kSine       = kTheta + Constants::kVecBytes,
        kCosine     = kSine + Constants::kVecBytes,
        kPositions  = kCosine + Constants::kVecBytes,
        kAxisValues = kPositions + Constants::kVecBytes,
        kMscale     = kAxisValues + Constants::kVecBytes,
        kTotal      = kMscale + Constants::kVecBytes,
    };
};

static_assert(RopeUramLayout::kTotal <= DAYAN_NATIVE_TASK_URAM_BYTES, "RoPE URAM layout is too large");

template <long Group>
static inline void configure_rope_lsu(uint32_t dim0,
                                      uint64_t input_stride,
                                      uint64_t output_stride,
                                      uint32_t read_bytes,
                                      uint32_t write_bytes) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0 - 1U);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(0);
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

static inline void rope_load_linear(uint64_t output, uint64_t input, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;

    if (full != 0) {
        configure_rope_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_d2u<0>(output, input);
    }
    if (tail != 0) {
        configure_rope_lsu<0>(1, 0, 0, tail, 64);
        __tx_lsu_dma_d2u<0>(output + (uint64_t) full * 64U, input + (uint64_t) full * 64U);
    }
}

static inline void rope_store_linear(uint64_t output, uint64_t input, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;

    if (full != 0) {
        configure_rope_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_u2d<0>(output, input);
    }
    if (tail != 0) {
        configure_rope_lsu<0>(1, 0, 0, 64, tail);
        __tx_lsu_dma_u2d<0>(output + (uint64_t) full * 64U, input + (uint64_t) full * 64U);
    }
}

static inline void rope_copy_linear(uint64_t output, uint64_t input, uint32_t bytes) {
    const uint32_t full = bytes / 64U;
    const uint32_t tail = bytes % 64U;

    if (full != 0) {
        configure_rope_lsu<0>(full, 64, 64, 64, 64);
        __tx_lsu_dma_d2d<0>(output, input);
    }
    if (tail != 0) {
        configure_rope_lsu<0>(1, 0, 0, tail, tail);
        __tx_lsu_dma_d2d<0>(output + (uint64_t) full * 64U, input + (uint64_t) full * 64U);
    }
}

static inline void rope_load_strided(uint64_t output,
                                     uint64_t input,
                                     uint32_t count,
                                     uint64_t input_stride) {
    configure_rope_lsu<0>(count, input_stride, sizeof(uint32_t), sizeof(uint32_t), sizeof(uint32_t));
    __tx_lsu_dma_d2u<0>(output, input);
}

static inline void rope_store_strided(uint64_t output,
                                      uint64_t input,
                                      uint32_t count,
                                      uint64_t output_stride) {
    configure_rope_lsu<0>(count, sizeof(uint32_t), output_stride, sizeof(uint32_t), sizeof(uint32_t));
    __tx_lsu_dma_u2d<0>(output, input);
}

static inline void emit_rope_axis_select(uint64_t axes, uint64_t axis_value, uint64_t position) {
    __tx_vec_lsu_u2v<1, 0>(axes);
    __tx_vec_lsu_u2v_broadcast32<1, 1>(axis_value);
    __tx_vec_vv_int32_cmp<16, 17, 0, 0b010>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(position);
    __tx_vec_vv_mux2<0, 16, 0, 0>();
}

static inline void issue_rope_position(uint64_t axes,
                                       uint64_t positions,
                                       uint64_t axis_values,
                                       uint64_t output) {
    constexpr long instructions = 18;

    configure_vec_dims(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>(positions + 3U * sizeof(int32_t));
    emit_rope_axis_select(axes, axis_values + 2U * sizeof(uint32_t), positions + 2U * sizeof(int32_t));
    emit_rope_axis_select(axes, axis_values + sizeof(uint32_t), positions + sizeof(int32_t));
    emit_rope_axis_select(axes, axis_values, positions);
    __tx_vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output);
    start_vec_block(instructions);
}

static inline void issue_rope_theta(uint64_t position,
                                    uint64_t multiplier,
                                    uint64_t factors,
                                    uint64_t output) {
    constexpr long instructions = 7;

    configure_vec_dims(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(position);
    __tx_vec_lsu_u2v<1, 1>(multiplier);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(factors);
    __tx_vec_v_fp32_sfu<1, 5, 16>();
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output);
    start_vec_block(instructions);
}

static inline void issue_rope_sincos(uint64_t theta,
                                     uint64_t mscale,
                                     uint64_t sine,
                                     uint64_t cosine) {
    constexpr long instructions = 10;

    configure_vec_dims(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(theta);
    __tx_vec_v_fp32_sfu<0, 2, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(mscale);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(sine);
    __tx_vec_lsu_u2v<0, 0>(theta);
    __tx_vec_v_fp32_sfu<0, 3, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>(mscale);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(cosine);
    start_vec_block(instructions);
}

static inline void issue_rope_rotation(uint64_t x0,
                                       uint64_t x1,
                                       uint64_t sine,
                                       uint64_t cosine,
                                       uint64_t output0,
                                       uint64_t output1) {
    constexpr long instructions = 16;

    configure_vec_dims(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(x0);
    __tx_vec_lsu_u2v<1, 1>(cosine);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(x1);
    __tx_vec_lsu_u2v<1, 1>(sine);
    __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
    __tx_vec_vv_fp32_sub<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output0);
    __tx_vec_lsu_u2v<0, 0>(x0);
    __tx_vec_lsu_u2v<1, 1>(sine);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_u2v<1, 1>(x1);
    __tx_vec_lsu_u2v<1, 1>(cosine);
    __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<3, 0>(output1);
    start_vec_block(instructions);
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
        dayan_proc_exit(-12);
    }

    const uint64_t elements      = (uint64_t) config->width * config->heads * config->tokens * config->batches;
    const uint32_t position_axes = (config->mode & 8U) != 0U ? 4U : 1U;
    cache_inval_range(input, (size_t) elements * sizeof(float));
    cache_inval_range(positions, (size_t) position_axes * config->tokens * sizeof(int32_t));
    cache_inval_range(freq_factors, (size_t) config->n_dims / 2U * sizeof(float));

    const uint64_t uram        = reinterpret_cast<uint64_t>(dayan_uram_base());
    const uint64_t x0          = uram + RopeUramLayout::kX0;
    const uint64_t x1          = uram + RopeUramLayout::kX1;
    const uint64_t output0     = uram + RopeUramLayout::kOutput0;
    const uint64_t output1     = uram + RopeUramLayout::kOutput1;
    const uint64_t factors     = uram + RopeUramLayout::kFactors;
    const uint64_t axes        = uram + RopeUramLayout::kAxes;
    const uint64_t multiplier  = uram + RopeUramLayout::kMultiplier;
    const uint64_t position    = uram + RopeUramLayout::kPosition;
    const uint64_t theta       = uram + RopeUramLayout::kTheta;
    const uint64_t sine        = uram + RopeUramLayout::kSine;
    const uint64_t cosine      = uram + RopeUramLayout::kCosine;
    const uint64_t position_in = uram + RopeUramLayout::kPositions;
    const uint64_t axis_values = uram + RopeUramLayout::kAxisValues;
    const uint64_t mscale      = uram + RopeUramLayout::kMscale;

    rope_load_linear(axis_values, reinterpret_cast<uint64_t>(config->axis_values), sizeof(config->axis_values));
    rope_load_linear(mscale, reinterpret_cast<uint64_t>(&config->mscale_bits), sizeof(config->mscale_bits));
    dayan_wait_lsu();

    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    const uint64_t rows  = (uint64_t) config->batches * config->tokens * config->heads;
    const uint32_t half  = config->n_dims / 2U;
    for (uint64_t row = task; row < rows; row += tasks) {
        const uint32_t token = (uint32_t) ((row / config->heads) % config->tokens);
        const float *  source = input + row * config->width;
        float *        destination = output + row * config->width;

        if (position_axes == 4U) {
            rope_load_strided(position_in, reinterpret_cast<uint64_t>(positions + token), 4,
                              (uint64_t) config->tokens * sizeof(int32_t));
        } else {
            rope_load_linear(position_in, reinterpret_cast<uint64_t>(positions + token), sizeof(int32_t));
        }

        for (uint32_t pair_base = 0; pair_base < half; pair_base += Constants::kVecLanes) {
            const uint32_t pair_count =
                half - pair_base < Constants::kVecLanes ? half - pair_base : Constants::kVecLanes;
            const uint32_t pair_bytes = pair_count * sizeof(float);
            rope_load_linear(axes, reinterpret_cast<uint64_t>(config->axes + pair_base), pair_bytes);
            rope_load_linear(multiplier, reinterpret_cast<uint64_t>(config->multipliers + pair_base), pair_bytes);
            rope_load_linear(factors, reinterpret_cast<uint64_t>(freq_factors + pair_base), pair_bytes);
            if (config->mode == 0U) {
                rope_load_strided(x0, reinterpret_cast<uint64_t>(source + 2U * pair_base), pair_count,
                                  2U * sizeof(float));
                rope_load_strided(x1, reinterpret_cast<uint64_t>(source + 2U * pair_base + 1U), pair_count,
                                  2U * sizeof(float));
            } else {
                rope_load_linear(x0, reinterpret_cast<uint64_t>(source + pair_base), pair_bytes);
                rope_load_linear(x1, reinterpret_cast<uint64_t>(source + half + pair_base), pair_bytes);
            }
            dayan_wait_lsu();

            issue_rope_position(axes, position_in, axis_values, position);
            dayan_wait_vec();
            issue_rope_theta(position, multiplier, factors, theta);
            dayan_wait_vec();
            issue_rope_sincos(theta, mscale, sine, cosine);
            dayan_wait_vec();
            issue_rope_rotation(x0, x1, sine, cosine, output0, output1);
            dayan_wait_vec();

            if (config->mode == 0U) {
                rope_store_strided(reinterpret_cast<uint64_t>(destination + 2U * pair_base), output0, pair_count,
                                   2U * sizeof(float));
                rope_store_strided(reinterpret_cast<uint64_t>(destination + 2U * pair_base + 1U), output1, pair_count,
                                   2U * sizeof(float));
            } else {
                rope_store_linear(reinterpret_cast<uint64_t>(destination + pair_base), output0, pair_bytes);
                rope_store_linear(reinterpret_cast<uint64_t>(destination + half + pair_base), output1, pair_bytes);
            }
            dayan_wait_lsu();
        }

        const uint32_t tail_bytes = (config->width - config->n_dims) * sizeof(float);
        if (tail_bytes != 0) {
            rope_copy_linear(reinterpret_cast<uint64_t>(destination + config->n_dims),
                             reinterpret_cast<uint64_t>(source + config->n_dims), tail_bytes);
            dayan_wait_lsu();
        }
    }
    cache_bypass_flush();
}

}  // namespace ggml_dayan_matmul_detail

#endif

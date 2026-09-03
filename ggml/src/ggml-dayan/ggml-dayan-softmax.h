#pragma once

#include "ggml-dayan-memory.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline bool softmax_positive_bits(uint32_t value) {
    return (value & 0x80000000U) == 0U && (value & 0x7fffffffU) != 0U;
}

static inline void softmax_slope(uint64_t output,
                                 uint32_t head,
                                 uint32_t heads,
                                 uint32_t max_bias_bits) {
    if (!softmax_positive_bits(max_bias_bits)) {
        constexpr long instructions = 2;
        ggml_dayan_vec_detail::configure_vec_grid(1);
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_g2v_broadcast32<0, 0>(0x3f800000U);
        __tx_vec_lsu_v2u<0, 3>((long) output);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        return;
    }

    uint32_t head_log2 = 1U;
    while (head_log2 <= heads / 2U) {
        head_log2 <<= 1U;
    }

    uint32_t numerator;
    uint32_t factor_bits;
    if (head < head_log2) {
        numerator   = head + 1U;
        factor_bits = 0xbf800000U;
    } else {
        numerator   = 2U * (head - head_log2) + 1U;
        factor_bits = 0xbf000000U;
    }

    constexpr long instructions = 12;
    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) numerator);
    __tx_vec_cvt_fp32_int32<0, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) head_log2);
    __tx_vec_cvt_fp32_int32<1, 16, 0>();
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) max_bias_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) factor_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline uint64_t softmax_mask_row(const DayanSoftmaxConfig * config,
                                        uint32_t                   row,
                                        uint32_t                   head,
                                        uint32_t                   batch) {
    return ((uint64_t) (batch % config->mask_batches) * config->mask_heads + head % config->mask_heads) *
               config->mask_rows +
           row;
}

static inline void softmax_load_tile(const DayanSoftmaxConfig * config,
                                     const float *              input,
                                     const void *               mask,
                                     uint64_t                   input_row,
                                     uint64_t                   mask_row,
                                     uint32_t                   offset,
                                     uint32_t                   elements,
                                     uint64_t                   logits_uram,
                                     uint64_t                   mask_f32_uram) {
    normalization_load_bits(
        logits_uram, input + input_row * config->width + offset, elements, 0xff7fffffU);
    if ((config->flags & kDayanSoftmaxHasMask) != 0U) {
        const uint32_t mask_bytes = memory_type_bytes(config->mask_type);
        memory_load_f32(mask_f32_uram,
                        (const char *) mask + (mask_row * config->width + offset) * mask_bytes,
                        elements,
                        config->mask_type);
        normalization_pad_tail_bits(mask_f32_uram, elements, 0U);
    }
}

static inline void emit_softmax_logits(const DayanSoftmaxConfig * config,
                                       uint64_t                   logits,
                                       uint64_t                   mask,
                                       uint64_t                   slope) {
    __tx_vec_lsu_u2v<0, 0>((long) logits);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->scale_bits);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    if ((config->flags & kDayanSoftmaxHasMask) != 0U) {
        __tx_vec_lsu_u2v<1, 1>((long) mask);
        __tx_vec_lsu_u2v_broadcast32<1, 2>((long) slope);
        __tx_vec_vv_fp32_mul<1, 16, 17, 0>();
        __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    }
}

static inline void softmax_max_accumulate(const DayanSoftmaxConfig * config,
                                          uint64_t                   accumulator,
                                          uint64_t                   input,
                                          uint64_t                   mask,
                                          uint64_t                   slope,
                                          uint64_t                   sink,
                                          uint32_t                   elements,
                                          bool                       initialize) {
    const bool has_mask          = (config->flags & kDayanSoftmaxHasMask) != 0U;
    const bool has_sink          = (config->flags & kDayanSoftmaxHasSinks) != 0U;
    const long chunks            = ggml_dayan_vec_detail::f32_chunks(elements);
    const long chunk_instructions = has_mask ? 27 : 23;
    const long instructions       = 2 + chunk_instructions * chunks;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV1Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV2Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    if (initialize) {
        if (has_sink) {
            __tx_vec_lsu_u2v_broadcast32<1, 1>((long) sink);
        } else {
            __tx_vec_lsu_g2v_broadcast32<1, 1>(0xff7fffffU);
        }
    } else {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) accumulator);
    }
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const uint64_t chunk_offset = (uint64_t) chunk * Constants::kVecBytes;
        emit_softmax_logits(config, input + chunk_offset, mask + chunk_offset, slope);
        ggml_dayan_vec_detail::emit_f32_max_stages();
        if (has_mask) {
            __tx_vec_vv_fp32_cmp<0, 26, 0, 0b100>();
            __tx_vec_vv_mux2<1, 0, 26, 0>();
        } else {
            __tx_vec_vv_fp32_cmp<0, 23, 0, 0b100>();
            __tx_vec_vv_mux2<1, 0, 23, 0>();
        }
    }
    __tx_vec_lsu_v2u<16, 3>((long) accumulator);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void softmax_exp_sum_accumulate(const DayanSoftmaxConfig * config,
                                              uint64_t                   accumulator,
                                              uint64_t                   maximum,
                                              uint64_t                   exponent,
                                              uint64_t                   logits,
                                              uint64_t                   mask,
                                              uint64_t                   slope,
                                              uint64_t                   sink,
                                              uint32_t                   elements,
                                              bool                       initialize,
                                              bool                       finalize) {
    const bool has_mask                 = (config->flags & kDayanSoftmaxHasMask) != 0U;
    const bool has_sink                 = (config->flags & kDayanSoftmaxHasSinks) != 0U;
    const long chunks                   = ggml_dayan_vec_detail::f32_chunks(elements);
    const long initialize_instructions = initialize && has_sink ? 6 : 1;
    const long chunk_instructions      = has_mask ? 26 : 22;
    const long finalize_instructions   = finalize ? 1 + 3 * chunks : 0;
    const long instructions = initialize_instructions + chunk_instructions * chunks + finalize_instructions + 1;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV1Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV2Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    if (initialize) {
        if (has_sink) {
            __tx_vec_lsu_u2v_broadcast32<1, 1>((long) sink);
            __tx_vec_lsu_u2v_broadcast32<0, 0>((long) maximum);
            __tx_vec_vv_fp32_sub<1, 16, 0, 0>();
            __tx_vec_lsu_g2v_broadcast32<0, 0>(
                (long) ggml_dayan_vec_detail::f32_bits(1.4426950408889634f));
            __tx_vec_vv_fp32_mul<1, 16, 0, 0>();
            __tx_vec_v_fp32_sfu<1, 16, 1>();
        } else {
            __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
        }
    } else {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) accumulator);
    }
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const uint64_t chunk_offset = (uint64_t) chunk * Constants::kVecBytes;
        emit_softmax_logits(config, logits + chunk_offset, mask + chunk_offset, slope);
        __tx_vec_lsu_u2v_broadcast32<0, 0>((long) maximum);
        __tx_vec_vv_fp32_sub<0, 1, 0, 0>();
        __tx_vec_lsu_g2v_broadcast32<0, 0>((long) ggml_dayan_vec_detail::f32_bits(1.4426950408889634f));
        __tx_vec_vv_fp32_mul<0, 0, 1, 0>();
        __tx_vec_v_fp32_sfu<0, 0, 1>();
        __tx_vec_lsu_v2u<0, 3>((long) (exponent + chunk_offset));
        ggml_dayan_vec_detail::emit_f32_sum_stages();
        if (has_mask) {
            __tx_vec_vv_fp32_add<1, 0, 26, 0>();
        } else {
            __tx_vec_vv_fp32_add<1, 0, 23, 0>();
        }
    }
    if (finalize) {
        __tx_vec_v_fp32_sfu<1, 16, 5>();
        for (long chunk = 0; chunk < chunks; ++chunk) {
            const uint64_t chunk_offset = (uint64_t) chunk * Constants::kVecBytes;
            __tx_vec_lsu_u2v<0, 0>((long) (exponent + chunk_offset));
            __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 3>((long) (exponent + chunk_offset));
        }
    }
    __tx_vec_lsu_v2u<16, 3>((long) accumulator);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void softmax_scale_output(uint64_t output,
                                        uint64_t exponent,
                                        uint64_t reciprocal,
                                        uint32_t elements) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) exponent);
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) reciprocal);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void run_softmax_f32(const DayanSoftmaxConfig * config,
                                   const float *              input,
                                   const void *               mask,
                                   const float *              sinks,
                                   float *                    output) {
    memory_validate(config);
    if (input == nullptr || output == nullptr || config->width == 0 || config->rows == 0 || config->heads == 0 ||
        config->batches == 0 || !softmax_positive_bits(config->scale_bits) ||
        (((config->flags & kDayanSoftmaxHasMask) != 0U) &&
         (mask == nullptr || config->mask_rows < config->rows || config->mask_heads == 0 || config->mask_batches == 0 ||
          (config->mask_type != kDayanDataTypeF32 && config->mask_type != kDayanDataTypeF16))) ||
        (((config->flags & kDayanSoftmaxHasSinks) != 0U) && sinks == nullptr) ||
        (softmax_positive_bits(config->max_bias_bits) && (config->flags & kDayanSoftmaxHasMask) == 0U)) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint64_t uram          = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t logits_uram   = uram + NormalizationLayout::kInput;
    const uint64_t sink_uram     = uram + NormalizationLayout::kAux;
    const uint64_t mask_f32_uram = uram + NormalizationLayout::kValue;
    const uint64_t exp_uram      = uram + NormalizationLayout::kSquare;
    const uint64_t maximum       = uram + NormalizationLayout::kRms;
    const uint64_t slope         = uram + NormalizationLayout::kScaled;
    const uint64_t accumulator   = uram + NormalizationLayout::kVectorScalar;

    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    const uint64_t work_items = (uint64_t) config->batches * config->heads * config->rows;
    for (uint64_t input_row = task; input_row < work_items; input_row += tasks) {
        uint64_t       work = input_row;
        const uint32_t row  = (uint32_t) (work % config->rows);
        work /= config->rows;
        const uint32_t head  = (uint32_t) (work % config->heads);
        const uint32_t batch = (uint32_t) (work / config->heads);
        softmax_slope(slope, head, config->heads, config->max_bias_bits);
        const uint64_t mask_row =
            (config->flags & kDayanSoftmaxHasMask) != 0U ? softmax_mask_row(config, row, head, batch) : 0U;

        if ((config->flags & kDayanSoftmaxHasSinks) != 0U) {
            normalization_load(sink_uram, sinks + head, 1);
        }

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            softmax_load_tile(
                config, input, mask, input_row, mask_row, offset, elements, logits_uram, mask_f32_uram);
            softmax_max_accumulate(
                config, maximum, logits_uram, mask_f32_uram, slope, sink_uram, elements, offset == 0);
        }

        const uint32_t final_offset = ((config->width - 1U) / NormalizationLayout::kTileElements) *
                                      NormalizationLayout::kTileElements;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            softmax_load_tile(
                config, input, mask, input_row, mask_row, offset, elements, logits_uram, mask_f32_uram);
            softmax_exp_sum_accumulate(config,
                                       accumulator,
                                       maximum,
                                       exp_uram,
                                       logits_uram,
                                       mask_f32_uram,
                                       slope,
                                       sink_uram,
                                       elements,
                                       offset == 0,
                                       offset == final_offset);
            normalization_store_f32(output + input_row * config->width + offset, exp_uram, elements);
            dayan_wait_lsu();
        }

        for (uint32_t offset = 0; offset < final_offset; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(exp_uram, output + input_row * config->width + offset, elements);
            softmax_scale_output(exp_uram, exp_uram, accumulator, elements);
            normalization_store_f32(output + input_row * config->width + offset, exp_uram, elements);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

#pragma once

#include "ggml-dayan-memory.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline bool softmax_positive_bits(uint32_t value) {
    return (value & 0x80000000U) == 0U && (value & 0x7fffffffU) != 0U;
}

static inline void softmax_slope(uint64_t output,
                                 uint64_t temporary,
                                 uint32_t head,
                                 uint32_t heads,
                                 uint32_t max_bias_bits) {
    if (!softmax_positive_bits(max_bias_bits)) {
        ggml_dayan_vec_detail::f32_fill_bits(output, 0x3f800000U, Constants::kVecLanes);
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

    ggml_dayan_vec_detail::f32_int32_scalar(output, numerator);
    ggml_dayan_vec_detail::f32_int32_scalar(temporary, head_log2);
    ggml_dayan_vec_detail::f32_div(output, output, temporary, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_scalar_mul_bits(output, max_bias_bits, output, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_scalar_mul_bits(output, factor_bits, output, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_sfu<1>(output, output, Constants::kVecLanes);
}

static inline uint64_t softmax_mask_row(const DayanSoftmaxConfig * config,
                                        uint32_t                   row,
                                        uint32_t                   head,
                                        uint32_t                   batch) {
    return ((uint64_t) (batch % config->mask_batches) * config->mask_heads + head % config->mask_heads) *
               config->mask_rows +
           row;
}

static inline void softmax_load_logits(const DayanSoftmaxConfig * config,
                                       const float *              input,
                                       const void *               mask,
                                       uint64_t                   input_row,
                                       uint64_t                   mask_row,
                                       uint32_t                   offset,
                                       uint32_t                   elements,
                                       uint64_t                   slope,
                                       uint64_t                   logits_uram,
                                       uint64_t                   mask_f32_uram) {
    normalization_load_bits(
        logits_uram, input + input_row * config->width + offset, elements, 0xff7fffffU);
    ggml_dayan_vec_detail::f32_scalar_mul_bits(
        logits_uram, config->scale_bits, logits_uram, elements);
    if ((config->flags & kDayanSoftmaxHasMask) != 0U) {
        const uint32_t mask_bytes = memory_type_bytes(config->mask_type);
        normalization_pad_tail_bits(mask_f32_uram, elements, 0U);
        memory_load_f32(mask_f32_uram,
                        (const char *) mask + (mask_row * config->width + offset) * mask_bytes,
                        elements,
                        config->mask_type);
        ggml_dayan_vec_detail::f32_mul_scalar_address(
            mask_f32_uram, mask_f32_uram, slope, elements);
        ggml_dayan_vec_detail::f32_add(logits_uram, logits_uram, mask_f32_uram, elements);
    }
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
    const uint64_t uram          = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t logits_uram   = uram + NormalizationLayout::kInput;
    const uint64_t sink_uram     = uram + NormalizationLayout::kAux;
    const uint64_t mask_f32_uram = uram + NormalizationLayout::kValue;
    const uint64_t exp_uram      = uram + NormalizationLayout::kSquare;
    const uint64_t maximum       = uram + NormalizationLayout::kRms;
    const uint64_t slope         = uram + NormalizationLayout::kScaled;
    const uint64_t accumulator   = uram + NormalizationLayout::kVectorScalar;
    const uint64_t reduce        = uram + NormalizationLayout::kVecReduce;

    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    const uint64_t work_items = (uint64_t) config->batches * config->heads * config->rows;
    for (uint64_t input_row = task; input_row < work_items; input_row += tasks) {
        uint64_t       work = input_row;
        const uint32_t row  = (uint32_t) (work % config->rows);
        work /= config->rows;
        const uint32_t head  = (uint32_t) (work % config->heads);
        const uint32_t batch = (uint32_t) (work / config->heads);
        softmax_slope(slope, sink_uram, head, config->heads, config->max_bias_bits);
        const uint64_t mask_row =
            (config->flags & kDayanSoftmaxHasMask) != 0U ? softmax_mask_row(config, row, head, batch) : 0U;

        if ((config->flags & kDayanSoftmaxHasSinks) != 0U) {
            normalization_load(sink_uram, sinks + head, 1);
            ggml_dayan_vec_detail::f32_fill_address(sink_uram, sink_uram, Constants::kVecLanes);
        }

        ggml_dayan_vec_detail::f32_fill_bits(accumulator, 0xff7fffffU, Constants::kVecLanes);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            softmax_load_logits(config,
                                input,
                                mask,
                                input_row,
                                mask_row,
                                offset,
                                elements,
                                slope,
                                logits_uram,
                                mask_f32_uram);
            ggml_dayan_vec_detail::f32_reduce_max(reduce, logits_uram, elements);
            ggml_dayan_vec_detail::f32_max(accumulator, accumulator, reduce, Constants::kVecLanes);
        }
        if ((config->flags & kDayanSoftmaxHasSinks) != 0U) {
            ggml_dayan_vec_detail::f32_max(accumulator, accumulator, sink_uram, Constants::kVecLanes);
        }
        ggml_dayan_vec_detail::f32_fill_address(maximum, accumulator, Constants::kVecLanes);

        ggml_dayan_vec_detail::f32_fill_bits(accumulator, 0U, Constants::kVecLanes);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            softmax_load_logits(config,
                                input,
                                mask,
                                input_row,
                                mask_row,
                                offset,
                                elements,
                                slope,
                                logits_uram,
                                mask_f32_uram);
            ggml_dayan_vec_detail::f32_sub_scalar_address(logits_uram, logits_uram, maximum, elements);
            ggml_dayan_vec_detail::f32_scalar_mul(logits_uram, 1.4426950408889634f, logits_uram, elements);
            ggml_dayan_vec_detail::f32_sfu<1>(exp_uram, logits_uram, elements);
            ggml_dayan_vec_detail::f32_reduce_sum(reduce, exp_uram, elements);
            ggml_dayan_vec_detail::f32_add(accumulator, accumulator, reduce, Constants::kVecLanes);
            normalization_store_f32(output + input_row * config->width + offset, exp_uram, elements);
            dayan_wait_lsu();
        }
        if ((config->flags & kDayanSoftmaxHasSinks) != 0U) {
            ggml_dayan_vec_detail::f32_sub(sink_uram, sink_uram, maximum, Constants::kVecLanes);
            ggml_dayan_vec_detail::f32_scalar_mul(sink_uram, 1.4426950408889634f, sink_uram, Constants::kVecLanes);
            ggml_dayan_vec_detail::f32_sfu<1>(sink_uram, sink_uram, Constants::kVecLanes);
            ggml_dayan_vec_detail::f32_add(accumulator, accumulator, sink_uram, Constants::kVecLanes);
        }

        ggml_dayan_vec_detail::f32_sfu<5>(accumulator, accumulator, Constants::kVecLanes);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(exp_uram, output + input_row * config->width + offset, elements);
            ggml_dayan_vec_detail::f32_mul_scalar_address(exp_uram, exp_uram, accumulator, elements);
            normalization_store_f32(output + input_row * config->width + offset, exp_uram, elements);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

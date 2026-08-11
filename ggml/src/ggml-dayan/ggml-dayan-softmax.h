#pragma once

#include "ggml-dayan-memory.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline float softmax_exp2_scalar(float value, float * scratch) {
    scratch[0] = value;
    configure_normalization_macro<1>(1);
    __tx_macro_sfu_exp2_fp32<1>(scratch + 1, scratch);
    dayan_wait_macro();
    return scratch[1];
}

static inline float softmax_slope(uint32_t head, uint32_t heads, float max_bias, float * scratch) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }

    uint32_t head_log2 = 1U;
    while (head_log2 <= heads / 2U) {
        head_log2 <<= 1U;
    }
    const float exponent = head < head_log2 ?
                               -max_bias * (float) (head + 1U) / (float) head_log2 :
                               -0.5f * max_bias * (float) (2U * (head - head_log2) + 1U) / (float) head_log2;
    return softmax_exp2_scalar(exponent, scratch);
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
                                       float                      slope,
                                       uint64_t                   logits_uram,
                                       uint64_t                   mask_uram,
                                       uint64_t                   mask_f32_uram) {
    normalization_load(logits_uram, input + input_row * config->width + offset, elements);
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(logits_uram, normalization_bits_float(config->scale_bits), logits_uram);
    if ((config->flags & kDayanSoftmaxHasMask) != 0U) {
        const uint32_t mask_bytes = memory_type_bytes(config->mask_type);
        memory_load(mask_uram, (const char *) mask + (mask_row * config->width + offset) * mask_bytes, elements,
                    config->mask_type);
        memory_convert_to_f32(mask_f32_uram, mask_uram, elements, config->mask_type);
        elementwise_scalar_mul<0>(mask_f32_uram, slope, mask_f32_uram);
        __tx_macro_elemwise_add_fp32<0, 0>((void *) logits_uram, (void *) logits_uram, (void *) mask_f32_uram);
    }
    dayan_wait_macro();
}

static inline void run_softmax_f32(const DayanSoftmaxConfig * config,
                                   const float *              input,
                                   const void *               mask,
                                   const float *              sinks,
                                   float *                    output) {
    memory_validate(config);
    if (input == nullptr || output == nullptr || config->width == 0 || config->rows == 0 || config->heads == 0 ||
        config->batches == 0 ||
        (((config->flags & kDayanSoftmaxHasMask) != 0U) &&
         (mask == nullptr || config->mask_rows < config->rows || config->mask_heads == 0 || config->mask_batches == 0 ||
          (config->mask_type != kDayanDataTypeF32 && config->mask_type != kDayanDataTypeF16))) ||
        (((config->flags & kDayanSoftmaxHasSinks) != 0U) && sinks == nullptr) ||
        (normalization_bits_float(config->max_bias_bits) > 0.0f && (config->flags & kDayanSoftmaxHasMask) == 0U)) {
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    const uint64_t uram          = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t logits_uram   = uram + NormalizationLayout::kInput;
    const uint64_t mask_uram     = uram + NormalizationLayout::kAux;
    const uint64_t mask_f32_uram = uram + NormalizationLayout::kValue;
    const uint64_t exp_uram      = uram + NormalizationLayout::kSquare;
    float *        scratch       = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar);
    const float    max_bias      = normalization_bits_float(config->max_bias_bits);

    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    const uint64_t work_items = (uint64_t) config->batches * config->heads * config->rows;
    for (uint64_t input_row = task; input_row < work_items; input_row += tasks) {
        uint64_t       work = input_row;
        const uint32_t row  = (uint32_t) (work % config->rows);
        work /= config->rows;
        const uint32_t head  = (uint32_t) (work % config->heads);
        const uint32_t batch = (uint32_t) (work / config->heads);
        const float    slope = softmax_slope(head, config->heads, max_bias, scratch + 4);
        const uint64_t mask_row =
            (config->flags & kDayanSoftmaxHasMask) != 0U ? softmax_mask_row(config, row, head, batch) : 0U;
        float maximum = -3.402823466e+38f;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            softmax_load_logits(config, input, mask, input_row, mask_row, offset, elements, slope, logits_uram,
                                mask_uram, mask_f32_uram);
            configure_normalization_macro<0>(elements);
            __tx_macro_reduce_max_fp32<0, 0>(scratch, (void *) logits_uram, (void *) 0);
            dayan_wait_macro();
            if (scratch[0] > maximum) {
                maximum = scratch[0];
            }
        }
        if ((config->flags & kDayanSoftmaxHasSinks) != 0U && sinks[head] > maximum) {
            maximum = sinks[head];
        }

        float sum = 0.0f;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            softmax_load_logits(config, input, mask, input_row, mask_row, offset, elements, slope, logits_uram,
                                mask_uram, mask_f32_uram);
            configure_normalization_macro<0>(elements);
            elementwise_scalar_add<0>(logits_uram, -maximum, logits_uram);
            elementwise_scalar_mul<0>(logits_uram, kDayanLog2E, logits_uram);
            __tx_macro_sfu_exp2_fp32<0>((void *) exp_uram, (void *) logits_uram);
            __tx_macro_reduce_add_fp32<0>(scratch + 1, (void *) exp_uram);
            dayan_wait_macro();
            sum += scratch[1];
            normalization_store_f32(output + input_row * config->width + offset, exp_uram, elements);
            dayan_wait_lsu();
        }
        if ((config->flags & kDayanSoftmaxHasSinks) != 0U) {
            sum += softmax_exp2_scalar((sinks[head] - maximum) * kDayanLog2E, scratch + 4);
        }

        scratch[0] = sum;
        configure_normalization_macro<1>(1);
        __tx_macro_sfu_recip_fp32<1>(scratch + 1, scratch);
        dayan_wait_macro();
        const float inverse_sum = scratch[1];
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(exp_uram, output + input_row * config->width + offset, elements);
            configure_normalization_macro<0>(elements);
            elementwise_scalar_mul<0>(exp_uram, inverse_sum, exp_uram);
            dayan_wait_macro();
            normalization_store_f32(output + input_row * config->width + offset, exp_uram, elements);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

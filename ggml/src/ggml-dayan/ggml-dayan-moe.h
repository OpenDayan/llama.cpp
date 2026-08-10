#pragma once

#include "ggml-dayan-elementwise.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

template <typename Format>
static void run_mul_mat_id(const DayanMulMatIdConfig * config,
                           const uint16_t *            activation,
                           const uint16_t *            weight,
                           const int32_t *             ids,
                           float *                     output,
                           DayanMatmulPerf *           perf) {
    if (config == nullptr || activation == nullptr || weight == nullptr || ids == nullptr || output == nullptr ||
        perf == nullptr || config->k == 0 || config->n == 0 || config->experts == 0 || config->experts_used == 0 ||
        config->tokens == 0 || config->activation_slots == 0 || config->experts_used % config->activation_slots != 0) {
        dayan_proc_exit(-22);
    }

    const DayanMatmulConfig matmul = {
        1,
        config->n,
        config->k,
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->n * sizeof(float),
    };
    const uint64_t weight_elements = (uint64_t) config->n * config->k;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t expert = ids[(uint64_t) token * config->experts_used + used];
            if (expert < 0 || expert >= (int32_t) config->experts) {
                dayan_proc_exit(-22);
            }
            const uint32_t activation_slot = used % config->activation_slots;
            const uint64_t activation_offset = ((uint64_t) token * config->activation_slots + activation_slot) *
                                               config->k;
            const uint64_t weight_offset = (uint64_t) expert * weight_elements;
            const uint64_t output_offset = ((uint64_t) token * config->experts_used + used) * config->n;
            run_matmul_kernel<Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity,
                              epilogue_identity, epilogue_identity, epilogue_identity, epilogue_identity>(
                &matmul, activation + activation_offset, weight + weight_offset, nullptr, output + output_offset,
                perf);
        }
    }
}

static inline void run_add_id(const DayanAddIdConfig * config,
                              const float *            input,
                              const float *            bias,
                              const int32_t *          ids,
                              float *                  output) {
    if (config == nullptr || input == nullptr || bias == nullptr || ids == nullptr || output == nullptr ||
        config->width == 0 || config->experts == 0 || config->experts_used == 0 || config->tokens == 0) {
        dayan_proc_exit(-22);
    }
    elementwise_validate((uint64_t) config->width * config->experts_used * config->tokens, input, output);
    configure_normalization_lsu();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t bias_uram  = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t expert = ids[(uint64_t) token * config->experts_used + used];
            if (expert < 0 || expert >= (int32_t) config->experts) {
                dayan_proc_exit(-22);
            }
            const uint64_t row = (uint64_t) token * config->experts_used + used;
            for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
                const uint32_t elements = min_int((int) (config->width - offset),
                                                  (int) NormalizationLayout::kTileElements);
                normalization_load(input_uram, input + row * config->width + offset, elements);
                normalization_load(bias_uram, bias + (uint64_t) expert * config->width + offset, elements);
                elementwise_add_tile(out_uram, input_uram, bias_uram, elements);
                dayan_wait_macro();
                normalization_store_f32(output + row * config->width + offset, out_uram, elements);
                dayan_wait_lsu();
            }
        }
    }
}

template <bool TopK>
static void run_sort(const DayanSortConfig * config, const float * input, int32_t * output) {
    memory_validate(config);
    if (input == nullptr || output == nullptr || config->width == 0 || config->width > 256 || config->rows == 0 ||
        config->output_width == 0 || config->output_width > config->width || config->order > 1U ||
        (TopK && config->order != GGML_SORT_ORDER_DESC)) {
        dayan_proc_exit(-22);
    }

    int32_t * indices = (int32_t *) ((uintptr_t) dayan_uram_base() + NormalizationLayout::kInput);
    for (uint32_t row = 0; row < config->rows; ++row) {
        const float * values = input + (uint64_t) row * config->width;
        for (uint32_t index = 0; index < config->width; ++index) {
            indices[index] = (int32_t) index;
        }
        for (uint32_t index = 1; index < config->width; ++index) {
            const int32_t key = indices[index];
            uint32_t      position = index;
            while (position > 0) {
                const int32_t previous = indices[position - 1U];
                const bool before = config->order == GGML_SORT_ORDER_ASC ? values[key] < values[previous] :
                                                                           values[key] > values[previous];
                if (!before) {
                    break;
                }
                indices[position] = previous;
                --position;
            }
            indices[position] = key;
        }
        if constexpr (TopK) {
            if (config->output_width > 1) {
                const int32_t first = indices[0];
                indices[0]          = indices[1];
                indices[1]          = first;
            }
        }
        for (uint32_t index = 0; index < config->output_width; ++index) {
            output[(uint64_t) row * config->output_width + index] = indices[index];
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

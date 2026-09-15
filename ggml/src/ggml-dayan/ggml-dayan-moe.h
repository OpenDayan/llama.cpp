#pragma once

#include "ggml-dayan-memory.h"
#include "ggml-dayan-ffn.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct SortUramLayout {
    enum : uint64_t {
        kValues  = 0,
        kIndices = kValues + (uint64_t) kDayanSortMaxWidth * sizeof(float),
        kTotal   = kIndices + (uint64_t) kDayanSortMaxWidth * sizeof(int32_t),
    };
};

static_assert(SortUramLayout::kTotal <= DAYAN_SORT_TASK_URAM_BYTES, "Sort URAM layout is too large");

static inline bool moe_allocation_fits(uint64_t element_size, uint64_t ne0, uint64_t ne1, uint64_t ne2) {
    uint64_t       size          = element_size;
    const uint64_t dimensions[3] = { ne0, ne1, ne2 };
    for (int index = 0; index < 3; ++index) {
        if (dimensions[index] == 0 || dimensions[index] > UINT64_MAX / size) {
            return false;
        }
        size *= dimensions[index];
    }
    return true;
}

static inline int32_t moe_read_expert(const int32_t * ids, uint64_t index) {
    const uint64_t scratch = (uint64_t) (uintptr_t) dayan_uram_base();
    if (scratch == 0 || dayan_uram_size() < sizeof(int32_t)) {
        dayan_thread_exit(-12);
    }
    return (int32_t) memory_read_index(ids, (uint32_t) index, kDayanDataTypeI32, scratch);
}

template <typename Format>
static void run_mul_mat_id(const DayanMulMatIdConfig * config,
                           const uint16_t *            activation,
                           const uint16_t *            weight,
                           const int32_t *             ids,
                           float *                     output, MatmulPhases * phases = nullptr) {
    if (config == nullptr || activation == nullptr || weight == nullptr || ids == nullptr || output == nullptr ||
        config->k == 0 || config->n == 0 || config->experts == 0 || config->experts > INT32_MAX ||
        config->experts_used == 0 || config->tokens == 0 || config->activation_slots == 0 ||
        config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        config->ids_row_stride > UINT64_MAX / config->tokens ||
        config->experts_used % config->activation_slots != 0 ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->n, config->experts) ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->activation_slots, config->tokens) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1) ||
        !moe_allocation_fits(sizeof(float), config->n, config->experts_used, config->tokens)) {
        DAYAN_PARAMS_ERROR();
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
    MatmulPhases local_phases;
    MatmulPhases & resource_phases = phases ? *phases : local_phases;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * config->ids_row_stride);
            const int32_t expert = moe_read_expert(row_ids, used);
            if (expert < 0 || expert >= (int32_t) config->experts) {
                DAYAN_PARAMS_ERROR();
            }
            const uint32_t activation_slot = used % config->activation_slots;
            const uint64_t activation_offset =
                ((uint64_t) token * config->activation_slots + activation_slot) * config->k;
            const uint64_t weight_offset = (uint64_t) expert * weight_elements;
            const uint64_t output_offset = ((uint64_t) token * config->experts_used + used) * config->n;
            matmul_check(run_matmul<Format, 1, 1, sizeof(float), false, epilogue_identity, nullptr, 0, true>(
                &matmul, activation + activation_offset, weight + weight_offset, nullptr, output + output_offset, nullptr, &resource_phases));
        }
    }
}

template <typename Format, bool GeGLU>
static void run_mul_mat_id_glu(const DayanMulMatIdConfig * config,
                               const uint16_t *            activation,
                               const uint16_t *            gate_weight,
                               const uint16_t *            up_weight,
                               const int32_t *             ids,
                               uint16_t *                  output) {
    if (config == nullptr || activation == nullptr || gate_weight == nullptr || up_weight == nullptr ||
        ids == nullptr || output == nullptr || config->k == 0 || config->n == 0 ||
        config->experts == 0 || config->experts > INT32_MAX || config->experts_used == 0 || config->tokens == 0 ||
        config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        config->ids_row_stride > UINT64_MAX / config->tokens ||
        config->activation_slots == 0 || config->experts_used % config->activation_slots != 0 ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->n, config->experts) ||
        !moe_allocation_fits(sizeof(uint16_t), config->k, config->activation_slots, config->tokens) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1) ||
        !moe_allocation_fits(sizeof(uint16_t), config->n, config->experts_used, config->tokens)) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanMatmulConfig matmul = {
        1,
        config->n,
        config->k,
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->k * sizeof(uint16_t),
        (uint64_t) config->n * sizeof(uint16_t),
    };
    const uint64_t weight_elements = (uint64_t) config->n * config->k;
    MatmulPhases resource_phases;
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t used = 0; used < config->experts_used; ++used) {
            const int32_t * row_ids = (const int32_t *) ((const char *) ids + token * config->ids_row_stride);
            const int32_t expert = moe_read_expert(row_ids, used);
            if (expert < 0 || expert >= (int32_t) config->experts) {
                DAYAN_PARAMS_ERROR();
            }
            const uint32_t activation_slot = used % config->activation_slots;
            const uint64_t activation_offset =
                ((uint64_t) token * config->activation_slots + activation_slot) * config->k;
            const uint64_t weight_offset = (uint64_t) expert * weight_elements;
            const uint64_t output_offset = ((uint64_t) token * config->experts_used + used) * config->n;
            matmul_check(ggml_dayan_ffn_detail::run<Format, 1, GeGLU, true>(
                &matmul, activation + activation_offset, gate_weight + weight_offset, up_weight + weight_offset,
                output + output_offset, &resource_phases));
        }
    }
}

static inline void run_add_id(const DayanAddIdConfig * config,
                              const float *            input,
                              const float *            bias,
                              const int32_t *          ids,
                              float *                  output) {
    if (config == nullptr || input == nullptr || bias == nullptr || ids == nullptr || output == nullptr ||
        config->width == 0 || config->experts == 0 || config->experts > INT32_MAX || config->experts_used == 0 ||
        config->tokens == 0 || config->ids_row_stride < (uint64_t) config->experts_used * sizeof(int32_t) ||
        !moe_allocation_fits(sizeof(float), config->width, config->experts_used, config->tokens) ||
        !moe_allocation_fits(sizeof(float), config->width, config->experts, 1) ||
        !moe_allocation_fits(sizeof(int32_t), config->experts_used, config->tokens, 1)) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate((uint64_t) config->width * config->experts_used * config->tokens, input, output);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t bias_uram  = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    const uint64_t rows       = (uint64_t) config->tokens * config->experts_used;
    for (uint64_t row = task; row < rows; row += tasks) {
        const uint32_t used   = (uint32_t) (row % config->experts_used);
        const uint32_t token  = (uint32_t) (row / config->experts_used);
        const int32_t * token_ids =
            (const int32_t *) ((const char *) ids + (uint64_t) token * config->ids_row_stride);
        const int32_t expert = moe_read_expert(token_ids, used);
        if (expert < 0 || expert >= (int32_t) config->experts) {
            DAYAN_PARAMS_ERROR();
        }
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + row * config->width + offset, elements);
            normalization_load(bias_uram, bias + (uint64_t) expert * config->width + offset, elements);
            ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
            constexpr long instructions = 4;
            ggml_dayan_vec_detail::begin_vec_block(instructions);
            __tx_vec_lsu_u2v<0, 0>((long) input_uram);
            __tx_vec_lsu_u2v<1, 1>((long) bias_uram);
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
            __tx_vec_lsu_v2u<0, 3>((long) out_uram);
            ggml_dayan_vec_detail::start_vec_block(instructions);
            dayan_wait_vec();
            normalization_store_f32(output + row * config->width + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

template <bool TopK> static void run_sort(const DayanSortConfig * config, const float * input, int32_t * output) {
    memory_validate(config);
    if (input == nullptr || output == nullptr || config->width == 0 || config->rows == 0 || config->output_width == 0 ||
        config->width > kDayanSortMaxWidth || config->output_width > config->width || config->order > 1U ||
        config->output_stride < config->output_width ||
        (TopK && config->order != kDayanSortOrderDesc)) {
        DAYAN_PARAMS_ERROR();
    }

    if (dayan_uram_base() == nullptr || dayan_uram_size() < SortUramLayout::kTotal) {
        dayan_thread_exit(-12);
    }

    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    const uint64_t uram  = (uint64_t) (uintptr_t) dayan_uram_base();
    for (uint32_t row = task; row < config->rows; row += tasks) {
        memory_load_bytes(uram + SortUramLayout::kValues, input + (uint64_t) row * config->width,
                          (uint64_t) config->width * sizeof(float));
        const float * values  = (const float *) (uintptr_t) (uram + SortUramLayout::kValues);
        int32_t *     indices = (int32_t *) (uintptr_t) (uram + SortUramLayout::kIndices);
        for (uint32_t index = 0; index < config->width; ++index) {
            const uint32_t count    = index < config->output_width ? index : config->output_width;
            uint32_t       position = count;
            while (position > 0) {
                const int32_t previous = indices[position - 1U];
                const bool    before   = config->order == kDayanSortOrderAsc ? values[index] < values[previous] :
                                                                               values[index] > values[previous];
                if (!before) {
                    break;
                }
                if (position < config->output_width) {
                    indices[position] = previous;
                }
                --position;
            }
            if (position < config->output_width) {
                indices[position] = (int32_t) index;
            }
        }
        if constexpr (TopK) {
            if (config->output_width > 1) {
                const int32_t first = indices[0];
                indices[0]          = indices[1];
                indices[1]          = first;
            }
        }
        memory_store_bytes(output + (uint64_t) row * config->output_stride, uram + SortUramLayout::kIndices,
                           (uint64_t) config->output_width * sizeof(int32_t));
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

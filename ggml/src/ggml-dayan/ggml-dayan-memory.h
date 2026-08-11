#pragma once

#include "ggml-dayan-elementwise.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline uint32_t memory_type_bytes(uint32_t type) {
    return type == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
}

static inline float memory_bf16_to_f32(uint16_t value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = (uint32_t) value << 16;
    return bits.f;
}

static inline float memory_f16_to_f32(uint16_t value) {
    const uint32_t sign     = (uint32_t) (value & 0x8000U) << 16;
    uint32_t       exponent = (value >> 10) & 0x1fU;
    uint32_t       fraction = value & 0x03ffU;
    uint32_t       result;

    if (exponent == 0U) {
        if (fraction == 0U) {
            result = sign;
        } else {
            exponent = 113U;
            while ((fraction & 0x0400U) == 0U) {
                fraction <<= 1;
                --exponent;
            }
            result = sign | (exponent << 23) | ((fraction & 0x03ffU) << 13);
        }
    } else if (exponent == 0x1fU) {
        result = sign | 0x7f800000U | (fraction << 13);
    } else {
        result = sign | ((exponent + 112U) << 23) | (fraction << 13);
    }

    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = result;
    return bits.f;
}

static inline uint64_t memory_read_index(const void * indices, uint32_t index, uint32_t type) {
    if (type == kDayanDataTypeI32) {
        const int32_t value = ((const int32_t *) indices)[index];
        if (value < 0) {
            dayan_proc_exit(-22);
        }
        return (uint64_t) value;
    }
    if (type == kDayanDataTypeI64) {
        const int64_t value = ((const int64_t *) indices)[index];
        if (value < 0) {
            dayan_proc_exit(-22);
        }
        return (uint64_t) value;
    }
    dayan_proc_exit(-22);
    return 0;
}

static inline void memory_validate(const void * config) {
    if (config == nullptr) {
        dayan_proc_exit(-22);
    }
    (void) dayan_native_task_id();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }
}

static inline void memory_load(uint64_t uram, const void * dram, uint32_t elements, uint32_t type) {
    const uint64_t bytes = (uint64_t) elements * memory_type_bytes(type);
    load_matrix<0>(bytes, 1, NormalizationLayout::kFloatBytes, 1, uram, (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
}

static inline void memory_store(void * dram, uint64_t uram, uint32_t elements, uint32_t type) {
    const uint64_t bytes = (uint64_t) elements * memory_type_bytes(type);
    store_matrix<0>(bytes, 1, NormalizationLayout::kFloatBytes, 1, uram, (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
}

static inline void memory_convert_to_f32(uint64_t output, uint64_t input, uint32_t elements, uint32_t type) {
    if (type == kDayanDataTypeF32) {
        float *       destination = (float *) (uintptr_t) output;
        const float * source      = (const float *) (uintptr_t) input;
        for (uint32_t index = 0; index < elements; ++index) {
            destination[index] = source[index];
        }
        return;
    }
    if (type != kDayanDataTypeBF16 && type != kDayanDataTypeF16) {
        dayan_proc_exit(-22);
    }

    float *          destination = (float *) (uintptr_t) output;
    const uint16_t * source      = (const uint16_t *) (uintptr_t) input;
    for (uint32_t index = 0; index < elements; ++index) {
        destination[index] =
            type == kDayanDataTypeBF16 ? memory_bf16_to_f32(source[index]) : memory_f16_to_f32(source[index]);
    }
}

static inline uint64_t memory_convert_from_f32(uint64_t output, uint64_t input, uint32_t elements, uint32_t type) {
    if (type == kDayanDataTypeF32) {
        float *       destination = (float *) (uintptr_t) output;
        const float * source      = (const float *) (uintptr_t) input;
        for (uint32_t index = 0; index < elements; ++index) {
            destination[index] = source[index];
        }
        return output;
    }

    configure_normalization_macro<0>(elements);
    if (type == kDayanDataTypeBF16) {
        __tx_macro_convert_fp32_bf16<0>((void *) output, (void *) input);
    } else if (type == kDayanDataTypeF16) {
        __tx_macro_convert_fp32_fp16<0>((void *) output, (void *) input);
    } else {
        dayan_proc_exit(-22);
    }
    dayan_wait_macro();
    return output;
}

static inline void run_get_rows(const DayanRowsConfig * config,
                                const void *            source,
                                const void *            indices,
                                float *                 output) {
    memory_validate(config);
    if (source == nullptr || indices == nullptr || output == nullptr || config->row_elements == 0 ||
        config->rows == 0 || (config->source_type != kDayanDataTypeF32 && config->source_type != kDayanDataTypeBF16)) {
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint64_t row_bytes  = (uint64_t) config->row_elements * memory_type_bytes(config->source_type);
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    for (uint32_t row = task; row < config->rows; row += tasks) {
        uint32_t       linear   = row / config->index_ne[0];
        const uint32_t index1   = linear % config->index_ne[1];
        const uint32_t index2   = linear / config->index_ne[1];
        const uint64_t selected = memory_read_index(indices, row, config->index_type);
        const uint64_t source_index =
            selected + (uint64_t) config->source_ne[0] * (index1 + (uint64_t) config->source_ne[1] * index2);
        if (selected >= config->source_ne[0] || source_index >= config->source_rows) {
            dayan_proc_exit(-22);
        }
        const char * source_row = (const char *) source + source_index * row_bytes;
        float *      output_row = output + (uint64_t) row * config->row_elements;
        for (uint32_t offset = 0; offset < config->row_elements; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements =
                min_int((int) (config->row_elements - offset), (int) NormalizationLayout::kTileElements);
            memory_load(input_uram, source_row + (uint64_t) offset * memory_type_bytes(config->source_type), elements,
                        config->source_type);
            memory_convert_to_f32(out_uram, input_uram, elements, config->source_type);
            normalization_store_f32(output_row + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void run_set_rows(const DayanRowsConfig * config,
                                const float *           source,
                                const void *            indices,
                                void *                  destination) {
    memory_validate(config);
    if (source == nullptr || indices == nullptr || destination == nullptr || config->row_elements == 0 ||
        config->rows == 0 || config->source_type != kDayanDataTypeF32 ||
        (config->destination_type != kDayanDataTypeF32 && config->destination_type != kDayanDataTypeBF16 &&
         config->destination_type != kDayanDataTypeF16)) {
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = config->destination_type == kDayanDataTypeF32 ? uram + NormalizationLayout::kValue :
                                                                                uram + NormalizationLayout::kLowp;
    const uint64_t row_bytes  = (uint64_t) config->row_elements * memory_type_bytes(config->destination_type);
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    for (uint32_t row = task; row < config->rows; row += tasks) {
        uint32_t       linear  = row;
        const uint32_t source1 = linear % config->source_ne[0];
        linear /= config->source_ne[0];
        const uint32_t source2  = linear % config->source_ne[1];
        const uint32_t source3  = linear / config->source_ne[1];
        const uint64_t index    = source1 + (uint64_t) config->index_ne[0] *
                                                ((source2 % config->index_ne[1]) +
                                                 (uint64_t) config->index_ne[1] * (source3 % config->index_ne[2]));
        const uint64_t selected = memory_read_index(indices, (uint32_t) index, config->index_type);
        const uint64_t destination_index = selected + (uint64_t) config->destination_ne[0] *
                                                          (source2 + (uint64_t) config->destination_ne[1] * source3);
        if (selected >= config->destination_ne[0] || destination_index >= config->destination_rows) {
            dayan_proc_exit(-22);
        }
        const float * source_row      = source + (uint64_t) row * config->row_elements;
        char *        destination_row = (char *) destination + destination_index * row_bytes;
        for (uint32_t offset = 0; offset < config->row_elements; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements =
                min_int((int) (config->row_elements - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, source_row + offset, elements);
            memory_convert_from_f32(out_uram, input_uram, elements, config->destination_type);
            memory_store(destination_row + (uint64_t) offset * memory_type_bytes(config->destination_type), out_uram,
                         elements, config->destination_type);
        }
    }
}

static inline void run_copy(const DayanCopyConfig * config, const void * source, void * destination) {
    memory_validate(config);
    if (source == nullptr || destination == nullptr || config->elements == 0 ||
        (config->source_type != kDayanDataTypeF32 && config->source_type != kDayanDataTypeBF16 &&
         config->source_type != kDayanDataTypeF16) ||
        (config->destination_type != kDayanDataTypeF32 && config->destination_type != kDayanDataTypeBF16 &&
         config->destination_type != kDayanDataTypeF16)) {
        dayan_proc_exit(-22);
    }
    uint64_t source_elements = 1;
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->source_ne[dimension] == 0 || config->source_nb[dimension] == 0) {
            dayan_proc_exit(-22);
        }
        source_elements *= config->source_ne[dimension];
    }
    if (source_elements != config->elements) {
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t value_uram = uram + NormalizationLayout::kValue;
    const uint64_t out_uram =
        config->destination_type == kDayanDataTypeF32 ? value_uram : uram + NormalizationLayout::kLowp;
    const uint32_t source_bytes      = memory_type_bytes(config->source_type);
    const uint32_t destination_bytes = memory_type_bytes(config->destination_type);
    const uint32_t task              = dayan_native_task_id();
    const uint32_t tasks             = dayan_native_task_count();
    const uint64_t source_rows       = config->elements / config->source_ne[0];
    for (uint64_t row = task; row < source_rows; row += tasks) {
        uint64_t       index  = row;
        const uint32_t index1 = (uint32_t) (index % config->source_ne[1]);
        index /= config->source_ne[1];
        const uint32_t index2     = (uint32_t) (index % config->source_ne[2]);
        const uint32_t index3     = (uint32_t) (index / config->source_ne[2]);
        const char *   source_row = (const char *) source + (uint64_t) index1 * config->source_nb[1] +
                                    (uint64_t) index2 * config->source_nb[2] + (uint64_t) index3 * config->source_nb[3];
        const uint64_t output_row = row * config->source_ne[0];
        for (uint32_t offset = 0; offset < config->source_ne[0]; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements =
                min_int((int) (config->source_ne[0] - offset), (int) NormalizationLayout::kTileElements);
            if (config->source_nb[0] == source_bytes) {
                memory_load(input_uram, source_row + (uint64_t) offset * source_bytes, elements, config->source_type);
            } else {
                if (config->source_type == kDayanDataTypeF32) {
                    for (uint32_t element = 0; element < elements; ++element) {
                        ((float *) (uintptr_t) input_uram)[element] =
                            *(const float *) (source_row + (uint64_t) (offset + element) * config->source_nb[0]);
                    }
                } else {
                    for (uint32_t element = 0; element < elements; ++element) {
                        ((uint16_t *) (uintptr_t) input_uram)[element] =
                            *(const uint16_t *) (source_row + (uint64_t) (offset + element) * config->source_nb[0]);
                    }
                }
            }
            const uint64_t output_offset = output_row + offset;
            if (config->source_type == config->destination_type) {
                memory_store((char *) destination + output_offset * destination_bytes, input_uram, elements,
                             config->destination_type);
                continue;
            }
            memory_convert_to_f32(value_uram, input_uram, elements, config->source_type);
            memory_convert_from_f32(out_uram, value_uram, elements, config->destination_type);
            memory_store((char *) destination + output_offset * destination_bytes, out_uram, elements,
                         config->destination_type);
        }
    }
}

static inline void run_repeat(const DayanRepeatConfig * config, const void * source, void * output) {
    memory_validate(config);
    if (source == nullptr || output == nullptr ||
        (config->type != kDayanDataTypeF32 && config->type != kDayanDataTypeBF16 &&
         config->type != kDayanDataTypeF16)) {
        dayan_proc_exit(-22);
    }
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->source_ne[dimension] == 0 || config->output_ne[dimension] == 0 ||
            config->output_ne[dimension] % config->source_ne[dimension] != 0) {
            dayan_proc_exit(-22);
        }
    }

    configure_normalization_lsu();
    const uint64_t input_uram  = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kInput;
    const uint32_t type_bytes  = memory_type_bytes(config->type);
    const uint64_t source_rows = (uint64_t) config->source_ne[1] * config->source_ne[2] * config->source_ne[3];
    const uint64_t output_rows = (uint64_t) config->output_ne[1] * config->output_ne[2] * config->output_ne[3];
    const uint32_t task        = dayan_native_task_id();
    const uint32_t tasks       = dayan_native_task_count();
    for (uint64_t output_row = task; output_row < output_rows; output_row += tasks) {
        uint64_t       index  = output_row;
        const uint32_t index1 = (uint32_t) (index % config->output_ne[1]);
        index /= config->output_ne[1];
        const uint32_t index2 = (uint32_t) (index % config->output_ne[2]);
        const uint32_t index3 = (uint32_t) (index / config->output_ne[2]);
        const uint64_t source_row =
            ((uint64_t) (index3 % config->source_ne[3]) * config->source_ne[2] + index2 % config->source_ne[2]) *
                config->source_ne[1] +
            index1 % config->source_ne[1];
        if (source_row >= source_rows) {
            dayan_proc_exit(-22);
        }

        for (uint32_t output0 = 0; output0 < config->output_ne[0];) {
            const uint32_t source0       = output0 % config->source_ne[0];
            uint32_t       elements      = config->source_ne[0] - source0;
            elements                     = min_int((int) elements, (int) (config->output_ne[0] - output0));
            elements                     = min_int((int) elements, (int) NormalizationLayout::kTileElements);
            const uint64_t source_offset = source_row * config->source_ne[0] + source0;
            const uint64_t output_offset = output_row * config->output_ne[0] + output0;
            memory_load(input_uram, (const char *) source + source_offset * type_bytes, elements, config->type);
            memory_store((char *) output + output_offset * type_bytes, input_uram, elements, config->type);
            output0 += elements;
        }
    }
}

static inline uint64_t memory_strided_offset(const uint64_t nb[4], const uint32_t indices[4]) {
    return (uint64_t) indices[0] * nb[0] + (uint64_t) indices[1] * nb[1] + (uint64_t) indices[2] * nb[2] +
           (uint64_t) indices[3] * nb[3];
}

static inline void run_concat(const DayanConcatConfig * config, const void * left, const void * right, void * output) {
    memory_validate(config);
    if (left == nullptr || right == nullptr || output == nullptr || config->axis >= 4 ||
        (config->type != kDayanDataTypeF32 && config->type != kDayanDataTypeBF16 &&
         config->type != kDayanDataTypeF16)) {
        dayan_proc_exit(-22);
    }
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->left_ne[dimension] == 0 || config->right_ne[dimension] == 0 || config->output_ne[dimension] == 0 ||
            (dimension == config->axis ?
                 config->output_ne[dimension] != config->left_ne[dimension] + config->right_ne[dimension] :
                 (config->output_ne[dimension] != config->left_ne[dimension] ||
                  config->output_ne[dimension] != config->right_ne[dimension]))) {
            dayan_proc_exit(-22);
        }
    }

    const uint32_t element_bytes = memory_type_bytes(config->type);
    const uint64_t elements =
        (uint64_t) config->output_ne[0] * config->output_ne[1] * config->output_ne[2] * config->output_ne[3];
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    for (uint64_t output_index = task; output_index < elements; output_index += tasks) {
        uint64_t value = output_index;
        uint32_t indices[4];
        for (uint32_t dimension = 0; dimension < 4; ++dimension) {
            indices[dimension] = (uint32_t) (value % config->output_ne[dimension]);
            value /= config->output_ne[dimension];
        }

        const uint64_t   output_offset = memory_strided_offset(config->output_nb, indices);
        const void *     source;
        const uint64_t * source_nb;
        if (indices[config->axis] < config->left_ne[config->axis]) {
            source    = left;
            source_nb = config->left_nb;
        } else {
            indices[config->axis] -= config->left_ne[config->axis];
            source    = right;
            source_nb = config->right_nb;
        }
        const uint64_t source_offset = memory_strided_offset(source_nb, indices);
        if (element_bytes == sizeof(float)) {
            *(uint32_t *) ((char *) output + output_offset) =
                *(const uint32_t *) ((const char *) source + source_offset);
        } else {
            *(uint16_t *) ((char *) output + output_offset) =
                *(const uint16_t *) ((const char *) source + source_offset);
        }
    }
    cache_bypass_flush();
}

}  // namespace ggml_dayan_matmul_detail

#endif

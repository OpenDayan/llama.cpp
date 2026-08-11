#pragma once

#include "ggml-dayan-memory.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline uint64_t layout_elements(const uint32_t ne[4]) {
    return (uint64_t) ne[0] * ne[1] * ne[2] * ne[3];
}

static inline uint64_t layout_index(const uint32_t ne[4],
                                    uint32_t       index0,
                                    uint32_t       index1,
                                    uint32_t       index2,
                                    uint32_t       index3) {
    return (((uint64_t) index3 * ne[2] + index2) * ne[1] + index1) * ne[0] + index0;
}

static inline uint32_t layout_wrap(int64_t index, uint32_t size) {
    int64_t wrapped = index % (int64_t) size;
    if (wrapped < 0) {
        wrapped += size;
    }
    return (uint32_t) wrapped;
}

static inline float layout_bits_float(uint32_t value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = value;
    return bits.f;
}

static inline void layout_validate_f32(const DayanLayoutConfig * config, const float * input, float * output) {
    memory_validate(config);
    if (input == nullptr || output == nullptr || config->type != kDayanDataTypeF32) {
        dayan_proc_exit(-22);
    }
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->input_ne[dimension] == 0 || config->output_ne[dimension] == 0) {
            dayan_proc_exit(-22);
        }
    }
}

static inline void layout_copy_bytes(void * destination, const void * source, uint64_t bytes) {
    const uint64_t uram = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kInput;
    for (uint64_t offset = 0; offset < bytes; offset += NormalizationLayout::kFloatBytes) {
        const uint64_t remaining = bytes - offset;
        const uint64_t count =
            remaining < NormalizationLayout::kFloatBytes ? remaining : NormalizationLayout::kFloatBytes;
        load_matrix<0>(count, 1, NormalizationLayout::kFloatBytes, 1, uram,
                       (uint64_t) (uintptr_t) ((const char *) source + offset));
        dayan_wait_lsu();
        store_matrix<0>(count, 1, NormalizationLayout::kFloatBytes, 1, uram,
                        (uint64_t) (uintptr_t) ((char *) destination + offset));
        dayan_wait_lsu();
    }
}

static inline void run_set(const DayanSetConfig * config, const void * base, const void * source, void * output) {
    memory_validate(config);
    if (base == nullptr || source == nullptr || output == nullptr || config->output_bytes == 0 ||
        config->element_bytes == 0) {
        dayan_proc_exit(-22);
    }
    (void) base;

    const uint64_t row_bytes = (uint64_t) config->source_ne[0] * config->element_bytes;
    const uint64_t rows      = (uint64_t) config->source_ne[1] * config->source_ne[2] * config->source_ne[3];
    const uint32_t task      = dayan_native_task_id();
    const uint32_t tasks     = dayan_native_task_count();
    for (uint64_t row = task; row < rows; row += tasks) {
        uint64_t       index  = row;
        const uint32_t index1 = (uint32_t) (index % config->source_ne[1]);
        index /= config->source_ne[1];
        const uint32_t index2             = (uint32_t) (index % config->source_ne[2]);
        const uint32_t index3             = (uint32_t) (index / config->source_ne[2]);
        const uint64_t destination_offset = config->destination_offset + (uint64_t) index1 * config->destination_nb[0] +
                                            (uint64_t) index2 * config->destination_nb[1] +
                                            (uint64_t) index3 * config->destination_nb[2];
        if (destination_offset + row_bytes > config->output_bytes) {
            dayan_proc_exit(-22);
        }
        layout_copy_bytes((char *) output + destination_offset, (const char *) source + row * row_bytes, row_bytes);
    }
}

static inline void run_diag_f32(const DayanLayoutConfig * config, const float * input, float * output) {
    layout_validate_f32(config, input, output);
    if (config->input_ne[1] != 1 || config->output_ne[0] != config->input_ne[0] ||
        config->output_ne[1] != config->input_ne[0] || config->output_ne[2] != config->input_ne[2] ||
        config->output_ne[3] != config->input_ne[3]) {
        dayan_proc_exit(-22);
    }

    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    const uint64_t rows  = (uint64_t) config->output_ne[1] * config->output_ne[2] * config->output_ne[3];
    for (uint64_t row = task; row < rows; row += tasks) {
        uint64_t       index  = row;
        const uint32_t index1 = (uint32_t) (index % config->output_ne[1]);
        index /= config->output_ne[1];
        const uint32_t index2     = (uint32_t) (index % config->output_ne[2]);
        const uint32_t index3     = (uint32_t) (index / config->output_ne[2]);
        const uint64_t input_base = layout_index(config->input_ne, 0, 0, index2, index3);
        float *        output_row = output + layout_index(config->output_ne, 0, index1, index2, index3);
        for (uint32_t index0 = 0; index0 < config->output_ne[0]; ++index0) {
            output_row[index0] = index0 == index1 ? input[input_base + index0] : 0.0f;
        }
    }
}

template <uint32_t MaskBits>
static void run_diag_mask_f32(const DayanLayoutConfig * config, const float * input, float * output) {
    layout_validate_f32(config, input, output);
    if (layout_elements(config->input_ne) != layout_elements(config->output_ne) ||
        config->input_ne[0] != config->output_ne[0] || config->input_ne[1] != config->output_ne[1] ||
        config->parameters[0] < 0) {
        dayan_proc_exit(-22);
    }

    const float    mask   = layout_bits_float(MaskBits);
    const uint64_t planes = (uint64_t) config->output_ne[2] * config->output_ne[3];
    const uint32_t task   = dayan_native_task_id();
    const uint32_t tasks  = dayan_native_task_count();
    const uint64_t rows   = planes * config->output_ne[1];
    for (uint64_t work = task; work < rows; work += tasks) {
        const uint32_t row    = (uint32_t) (work % config->output_ne[1]);
        const uint64_t offset = work * config->output_ne[0];
        for (uint32_t column = 0; column < config->output_ne[0]; ++column) {
            output[offset + column] = column > (uint32_t) config->parameters[0] + row ? mask : input[offset + column];
        }
    }
}

template <bool Circular>
static void run_pad_f32(const DayanLayoutConfig * config, const float * input, float * output) {
    layout_validate_f32(config, input, output);
    const uint32_t task     = dayan_native_task_id();
    const uint32_t tasks    = dayan_native_task_count();
    const uint64_t elements = layout_elements(config->output_ne);
    for (uint64_t destination = task; destination < elements; destination += tasks) {
        uint64_t       index  = destination;
        const uint32_t index0 = (uint32_t) (index % config->output_ne[0]);
        index /= config->output_ne[0];
        const uint32_t index1 = (uint32_t) (index % config->output_ne[1]);
        index /= config->output_ne[1];
        const uint32_t index2  = (uint32_t) (index % config->output_ne[2]);
        const uint32_t index3  = (uint32_t) (index / config->output_ne[2]);
        const int64_t  source0 = (int64_t) index0 - config->parameters[0];
        const int64_t  source1 = (int64_t) index1 - config->parameters[2];
        const int64_t  source2 = (int64_t) index2 - config->parameters[4];
        const int64_t  source3 = (int64_t) index3 - config->parameters[6];
        if constexpr (Circular) {
            output[destination] = input[layout_index(
                config->input_ne, layout_wrap(source0, config->input_ne[0]), layout_wrap(source1, config->input_ne[1]),
                layout_wrap(source2, config->input_ne[2]), layout_wrap(source3, config->input_ne[3]))];
        } else if (source0 >= 0 && source0 < config->input_ne[0] && source1 >= 0 && source1 < config->input_ne[1] &&
                   source2 >= 0 && source2 < config->input_ne[2] && source3 >= 0 && source3 < config->input_ne[3]) {
            output[destination] = input[layout_index(config->input_ne, (uint32_t) source0, (uint32_t) source1,
                                                     (uint32_t) source2, (uint32_t) source3)];
        } else {
            output[destination] = 0.0f;
        }
    }
}

static inline void run_pad_reflect_1d_f32(const DayanLayoutConfig * config, const float * input, float * output) {
    layout_validate_f32(config, input, output);
    const int32_t left  = config->parameters[0];
    const int32_t right = config->parameters[1];
    if (left < 0 || right < 0 || left >= (int32_t) config->input_ne[0] || right >= (int32_t) config->input_ne[0] ||
        config->output_ne[0] != config->input_ne[0] + (uint32_t) left + (uint32_t) right) {
        dayan_proc_exit(-22);
    }

    const uint64_t rows  = (uint64_t) config->output_ne[1] * config->output_ne[2] * config->output_ne[3];
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    for (uint64_t row = task; row < rows; row += tasks) {
        for (uint32_t output0 = 0; output0 < config->output_ne[0]; ++output0) {
            uint32_t source0;
            if (output0 < (uint32_t) left) {
                source0 = (uint32_t) left - output0;
            } else if (output0 < (uint32_t) left + config->input_ne[0]) {
                source0 = output0 - (uint32_t) left;
            } else {
                source0 = 2U * config->input_ne[0] + (uint32_t) left - 2U - output0;
            }
            output[row * config->output_ne[0] + output0] = input[row * config->input_ne[0] + source0];
        }
    }
}

static inline void run_roll_f32(const DayanLayoutConfig * config, const float * input, float * output) {
    layout_validate_f32(config, input, output);
    if (layout_elements(config->input_ne) != layout_elements(config->output_ne)) {
        dayan_proc_exit(-22);
    }
    const uint32_t task     = dayan_native_task_id();
    const uint32_t tasks    = dayan_native_task_count();
    const uint64_t elements = layout_elements(config->output_ne);
    for (uint64_t destination = task; destination < elements; destination += tasks) {
        uint64_t       index  = destination;
        const uint32_t index0 = (uint32_t) (index % config->output_ne[0]);
        index /= config->output_ne[0];
        const uint32_t index1 = (uint32_t) (index % config->output_ne[1]);
        index /= config->output_ne[1];
        const uint32_t index2 = (uint32_t) (index % config->output_ne[2]);
        const uint32_t index3 = (uint32_t) (index / config->output_ne[2]);
        output[destination]   = input[layout_index(
            config->input_ne, layout_wrap((int64_t) index0 - config->parameters[0], config->input_ne[0]),
            layout_wrap((int64_t) index1 - config->parameters[1], config->input_ne[1]),
            layout_wrap((int64_t) index2 - config->parameters[2], config->input_ne[2]),
            layout_wrap((int64_t) index3 - config->parameters[3], config->input_ne[3]))];
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

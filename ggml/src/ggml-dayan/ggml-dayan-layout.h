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

static inline void layout_validate_f32(const DayanLayoutConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    memory_validate(config, context);
    if (input == nullptr || output == nullptr || config->type != kDayanDataTypeF32) {
        DAYAN_PARAMS_ERROR();
    }
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->input_ne[dimension] == 0 || config->output_ne[dimension] == 0) {
            DAYAN_PARAMS_ERROR();
        }
    }
}

static inline uint64_t layout_scratch(const NativeContext & context = {}) {
    return native_uram(context) + NormalizationLayout::kInput;
}

static inline void layout_make_constant(uint32_t bits, const NativeContext & context = {}) {
    const uint64_t scratch = layout_scratch(context);
    uint32_t *     scalar  = (uint32_t *) (native_uram(context) + NormalizationLayout::kScalar);
    *scalar                = bits;

    ggml_dayan_vec_detail::configure_vec_grid(NormalizationLayout::kTileElements / Constants::kVecLanes);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, 0);
    begin_vec_block(2);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) (uintptr_t) scalar);
    __tx_vec_lsu_v2u<0, 3>((long) scratch);
    start_vec_block(2);
    dayan_wait_vec();
}

static inline void layout_contiguous_strides(const uint32_t ne[4], uint64_t nb[4]) {
    nb[0] = sizeof(float);
    for (uint32_t dimension = 1; dimension < 4; ++dimension) {
        nb[dimension] = nb[dimension - 1U] * ne[dimension - 1U];
    }
}

static inline uint64_t layout_offset(const uint64_t nb[4], const uint32_t index[4]) {
    return (uint64_t) index[0] * nb[0] + (uint64_t) index[1] * nb[1] +
           (uint64_t) index[2] * nb[2] + (uint64_t) index[3] * nb[3];
}

static inline bool layout_issue_wrapped_copy(const DayanLayoutConfig * config,
                                             const float *             input,
                                             float *                   output,
                                             const int32_t             offsets[4], const NativeContext & context = {}) {
    uint64_t input_nb[4];
    uint64_t output_nb[4];
    layout_contiguous_strides(config->input_ne, input_nb);
    layout_contiguous_strides(config->output_ne, output_nb);

    uint32_t destination[4] = {};
    bool     issued         = false;
    for (;;) {
        uint32_t source[4];
        uint32_t extent[4];
        for (uint32_t dimension = 0; dimension < 4; ++dimension) {
            source[dimension] =
                layout_wrap((int64_t) destination[dimension] - offsets[dimension], config->input_ne[dimension]);
            const uint32_t input_remaining  = config->input_ne[dimension] - source[dimension];
            const uint32_t output_remaining = config->output_ne[dimension] - destination[dimension];
            extent[dimension] = input_remaining < output_remaining ? input_remaining : output_remaining;
        }

        issued = memory_issue_tensor_copy(reinterpret_cast<uint64_t>(output) + layout_offset(output_nb, destination),
                                          reinterpret_cast<uint64_t>(input) + layout_offset(input_nb, source), extent,
                                          input_nb, output_nb, sizeof(float), context) ||
                 issued;

        bool done = true;
        for (uint32_t dimension = 0; dimension < 4; ++dimension) {
            destination[dimension] += extent[dimension];
            if (destination[dimension] < config->output_ne[dimension]) {
                done = false;
                break;
            }
            destination[dimension] = 0;
        }
        if (done) {
            return issued;
        }
    }
}

static inline void run_set(const DayanSetConfig * config, const void * base, const void * source, void * output, const NativeContext & context = {}) {
    memory_validate(config, context);
    if (base == nullptr || source == nullptr || output == nullptr || config->output_bytes == 0 ||
        config->element_bytes == 0) {
        DAYAN_PARAMS_ERROR();
    }
    uint64_t last = config->destination_offset + config->element_bytes;
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        if (config->source_ne[dimension] == 0) {
            DAYAN_PARAMS_ERROR();
        }
        const uint64_t stride = dimension == 0 ? config->element_bytes : config->destination_nb[dimension - 1U];
        last += (uint64_t) (config->source_ne[dimension] - 1U) * stride;
    }
    if (last > config->output_bytes) {
        DAYAN_PARAMS_ERROR();
    }
    (void) base;

    if (dayan_native_task_id(context) != 0U) {
        return;
    }
    const uint64_t source_nb[4] = {
        config->element_bytes,
        (uint64_t) config->source_ne[0] * config->element_bytes,
        (uint64_t) config->source_ne[0] * config->source_ne[1] * config->element_bytes,
        (uint64_t) config->source_ne[0] * config->source_ne[1] * config->source_ne[2] * config->element_bytes,
    };
    const uint64_t destination_nb[4] = {
        config->element_bytes,
        config->destination_nb[0],
        config->destination_nb[1],
        config->destination_nb[2],
    };
    const bool issued = memory_issue_tensor_copy(reinterpret_cast<uint64_t>(output) + config->destination_offset,
                                                 reinterpret_cast<uint64_t>(source), config->source_ne, source_nb,
                                                 destination_nb, config->element_bytes, context);
    if (issued) {
        dayan_wait_lsu();
    }
}

static inline void run_diag_f32(const DayanLayoutConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    layout_validate_f32(config, input, output, context);
    if (config->input_ne[1] != 1 || config->output_ne[0] != config->input_ne[0] ||
        config->output_ne[1] != config->input_ne[0] || config->output_ne[2] != config->input_ne[2] ||
        config->output_ne[3] != config->input_ne[3]) {
        DAYAN_PARAMS_ERROR();
    }

    if (dayan_native_task_id(context) != 0U) {
        return;
    }
    layout_make_constant(0U, context);
    memory_issue_fill(reinterpret_cast<uint64_t>(output), layout_scratch(context), layout_elements(config->output_ne) * sizeof(float));
    MemoryCopyDimension dimensions[3] = {
        { config->input_ne[0], sizeof(float), (uint64_t) (config->output_ne[0] + 1U) * sizeof(float) },
        { config->input_ne[2], (uint64_t) config->input_ne[0] * sizeof(float),
          (uint64_t) config->output_ne[0] * config->output_ne[1] * sizeof(float) },
        { config->input_ne[3], (uint64_t) config->input_ne[0] * config->input_ne[2] * sizeof(float),
          (uint64_t) config->output_ne[0] * config->output_ne[1] * config->output_ne[2] * sizeof(float) },
    };
    memory_issue_copy_dimensions(reinterpret_cast<uint64_t>(output), reinterpret_cast<uint64_t>(input), sizeof(float),
                                 dimensions, 3);
    dayan_wait_lsu();
}

template <uint32_t MaskBits>
static void run_diag_mask_f32(const DayanLayoutConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    if (!config) { DAYAN_PARAMS_ERROR(); }
    if (native_dispatch(context, {layout_elements(config->output_ne), config->output_ne[1], 256U * 1024U}, [&](const NativeContext & worker) {
            run_diag_mask_f32<MaskBits>(config, input, output, worker);
        })) { return; }

    layout_validate_f32(config, input, output, context);
    if (layout_elements(config->input_ne) != layout_elements(config->output_ne) ||
        config->input_ne[0] != config->output_ne[0] || config->input_ne[1] != config->output_ne[1] ||
        config->parameters[0] < 0) {
        DAYAN_PARAMS_ERROR();
    }

    const uint64_t planes       = (uint64_t) config->output_ne[2] * config->output_ne[3];
    const uint32_t task         = dayan_native_task_id(context);
    const uint32_t tasks        = dayan_native_task_count(context);
    const uint64_t plane_stride = (uint64_t) config->output_ne[0] * config->output_ne[1] * sizeof(float);
    if (task >= config->output_ne[1]) {
        return;
    }
    layout_make_constant(MaskBits, context);
    MemoryCopyDimension plane_dimension = { planes, plane_stride, plane_stride };
    bool                issued          = false;
    for (uint32_t row = task; row < config->output_ne[1]; row += tasks) {
        const uint64_t offset   = (uint64_t) row * config->output_ne[0];
        const uint64_t boundary = (uint64_t) (uint32_t) config->parameters[0] + row + 1U;
        const uint32_t prefix   = boundary < config->output_ne[0] ? (uint32_t) boundary : config->output_ne[0];
        if (prefix != 0U) {
            memory_issue_copy_dimensions(reinterpret_cast<uint64_t>(output + offset),
                                         reinterpret_cast<uint64_t>(input + offset),
                                         (uint64_t) prefix * sizeof(float), &plane_dimension, 1);
            issued = true;
        }
        if (prefix != config->output_ne[0]) {
            memory_issue_fill_dimensions(reinterpret_cast<uint64_t>(output + offset + prefix), layout_scratch(context),
                                         (uint64_t) (config->output_ne[0] - prefix) * sizeof(float), &plane_dimension,
                                         1);
            issued = true;
        }
    }
    if (issued) {
        dayan_wait_lsu();
    }
}

template <bool Circular>
static void run_pad_f32(const DayanLayoutConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    layout_validate_f32(config, input, output, context);
    if (dayan_native_task_id(context) != 0U) {
        return;
    }
    const int32_t offsets[4] = {
        config->parameters[0], config->parameters[2], config->parameters[4], config->parameters[6],
    };
    if constexpr (Circular) {
        if (layout_issue_wrapped_copy(config, input, output, offsets, context)) {
            dayan_wait_lsu();
        }
        return;
    }

    layout_make_constant(0U, context);
    memory_issue_fill(reinterpret_cast<uint64_t>(output), layout_scratch(context), layout_elements(config->output_ne) * sizeof(float));

    uint64_t input_nb[4];
    uint64_t output_nb[4];
    layout_contiguous_strides(config->input_ne, input_nb);
    layout_contiguous_strides(config->output_ne, output_nb);
    uint32_t source_start[4];
    uint32_t output_start[4];
    uint32_t extent[4];
    bool     intersects = true;
    for (uint32_t dimension = 0; dimension < 4; ++dimension) {
        source_start[dimension] = offsets[dimension] < 0 ? (uint32_t) -(int64_t) offsets[dimension] : 0U;
        output_start[dimension] = offsets[dimension] > 0 ? (uint32_t) offsets[dimension] : 0U;
        if (source_start[dimension] >= config->input_ne[dimension] ||
            output_start[dimension] >= config->output_ne[dimension]) {
            extent[dimension] = 0U;
            intersects        = false;
            continue;
        }
        const uint32_t input_remaining  = config->input_ne[dimension] - source_start[dimension];
        const uint32_t output_remaining = config->output_ne[dimension] - output_start[dimension];
        extent[dimension] = input_remaining < output_remaining ? input_remaining : output_remaining;
    }
    if (intersects) {
        memory_issue_tensor_copy(reinterpret_cast<uint64_t>(output) + layout_offset(output_nb, output_start),
                                 reinterpret_cast<uint64_t>(input) + layout_offset(input_nb, source_start), extent,
                                 input_nb, output_nb, sizeof(float), context);
    }
    dayan_wait_lsu();
}

static inline void run_pad_reflect_1d_f32(const DayanLayoutConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    layout_validate_f32(config, input, output, context);
    const int32_t left  = config->parameters[0];
    const int32_t right = config->parameters[1];
    if (left < 0 || right < 0 || left >= (int32_t) config->input_ne[0] || right >= (int32_t) config->input_ne[0] ||
        config->output_ne[0] != config->input_ne[0] + (uint32_t) left + (uint32_t) right) {
        DAYAN_PARAMS_ERROR();
    }

    if (dayan_native_task_id(context) != 0U) {
        return;
    }

    const uint64_t rows              = (uint64_t) config->output_ne[1] * config->output_ne[2] * config->output_ne[3];
    const uint64_t input_row_bytes   = (uint64_t) config->input_ne[0] * sizeof(float);
    const uint64_t output_row_bytes  = (uint64_t) config->output_ne[0] * sizeof(float);
    MemoryCopyDimension row_dimension = { rows, input_row_bytes, output_row_bytes };
    memory_issue_copy_dimensions(reinterpret_cast<uint64_t>(output + left), reinterpret_cast<uint64_t>(input),
                                 input_row_bytes, &row_dimension, 1);
    for (uint32_t output0 = 0; output0 < (uint32_t) left; ++output0) {
        const uint32_t source0 = (uint32_t) left - output0;
        memory_issue_copy_dimensions(reinterpret_cast<uint64_t>(output + output0),
                                     reinterpret_cast<uint64_t>(input + source0), sizeof(float), &row_dimension, 1);
    }
    for (uint32_t reflected = 0; reflected < (uint32_t) right; ++reflected) {
        const uint32_t output0 = (uint32_t) left + config->input_ne[0] + reflected;
        const uint32_t source0 = config->input_ne[0] - 2U - reflected;
        memory_issue_copy_dimensions(reinterpret_cast<uint64_t>(output + output0),
                                     reinterpret_cast<uint64_t>(input + source0), sizeof(float), &row_dimension, 1);
    }
    dayan_wait_lsu();
}

static inline void run_roll_f32(const DayanLayoutConfig * config, const float * input, float * output, const NativeContext & context = {}) {
    layout_validate_f32(config, input, output, context);
    if (layout_elements(config->input_ne) != layout_elements(config->output_ne)) {
        DAYAN_PARAMS_ERROR();
    }
    if (dayan_native_task_id(context) != 0U) {
        return;
    }
    if (layout_issue_wrapped_copy(config, input, output, config->parameters, context)) {
        dayan_wait_lsu();
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

#pragma once

#include "ggml-dayan-normalization.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static constexpr float kDayanLog2E       = 1.4426950408889634f;
static constexpr float kDayanLn2         = 0.6931471805599453f;
static constexpr float kDayanGeluA       = 0.044715f;
static constexpr float kDayanGeluQuick   = 1.702f;
static constexpr float kDayanSqrt2OverPi = 0.7978845608028654f;

static inline void elementwise_validate(uint64_t elements, const void * input, void * output) {
    if (elements == 0 || input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    (void) dayan_native_task_id();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }
}

static inline void elementwise_add_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    ggml_dayan_vec_detail::f32_add(output, left, right, elements);
}

static inline void elementwise_mul_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    ggml_dayan_vec_detail::f32_mul(output, left, right, elements);
}

static inline void elementwise_sub_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    ggml_dayan_vec_detail::f32_sub(output, left, right, elements);
}

static inline void elementwise_div_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    ggml_dayan_vec_detail::f32_div(output, left, right, elements);
}

template <void (*Operation)(uint64_t, uint64_t, uint64_t, uint32_t)>
static void run_binary_f32(const DayanElementwiseConfig * config,
                           const float *                  left,
                           const float *                  right,
                           float *                        output) {
    if (config == nullptr || right == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, left, output);
    configure_normalization_lsu();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t left_uram  = uram + NormalizationLayout::kInput;
    const uint64_t right_uram = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(left_uram, left + offset, elements);
        normalization_load(right_uram, right + offset, elements);
        Operation(out_uram, left_uram, right_uram, elements);
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void sigmoid_tile(uint64_t output,
                                uint64_t input,
                                uint64_t temporary,
                                uint64_t auxiliary,
                                uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, -kDayanLog2E, input, elements);
    ggml_dayan_vec_detail::f32_sfu<1>(auxiliary, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_add(temporary, 1.0f, auxiliary, elements);
    ggml_dayan_vec_detail::f32_sfu<5>(output, temporary, elements);
}

static inline void silu_tile(uint64_t output,
                             uint64_t input,
                             uint64_t temporary,
                             uint64_t auxiliary,
                             uint32_t elements) {
    sigmoid_tile(output, input, temporary, auxiliary, elements);
    ggml_dayan_vec_detail::f32_mul(output, input, output, elements);
}

static inline void softplus_tile(uint64_t output,
                                 uint64_t input,
                                 uint64_t temporary,
                                 uint64_t auxiliary,
                                 uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, -1.0f, input, elements);
    ggml_dayan_vec_detail::f32_min(temporary, input, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, kDayanLog2E, temporary, elements);
    ggml_dayan_vec_detail::f32_sfu<1>(temporary, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_add(temporary, 1.0f, temporary, elements);
    ggml_dayan_vec_detail::f32_sfu<0>(temporary, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, kDayanLn2, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_max(auxiliary, 0.0f, input, elements);
    ggml_dayan_vec_detail::f32_add(output, auxiliary, temporary, elements);
}

static inline void square_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_mul(output, input, input, elements);
}

static inline void sqrt_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_sqrt(output, input, elements);
    (void) temporary;
}

static inline void log_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_sfu<0>(output, input, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(output, kDayanLn2, output, elements);
}

static inline void sin_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_sfu<2>(output, input, elements);
}

static inline void cos_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_sfu<3>(output, input, elements);
}

static inline void abs_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, -1.0f, input, elements);
    ggml_dayan_vec_detail::f32_max(output, input, temporary, elements);
}

static inline void neg_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(output, -1.0f, input, elements);
}

static inline void relu_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_max(output, 0.0f, input, elements);
}

static inline void exp_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, kDayanLog2E, input, elements);
    ggml_dayan_vec_detail::f32_sfu<1>(output, temporary, elements);
}

static inline void tanh_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, -2.0f * kDayanLog2E, input, elements);
    ggml_dayan_vec_detail::f32_sfu<1>(output, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_add(output, 1.0f, output, elements);
    ggml_dayan_vec_detail::f32_sfu<5>(output, output, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(output, 2.0f, output, elements);
    ggml_dayan_vec_detail::f32_scalar_add(output, -1.0f, output, elements);
}

static inline void gelu_tile(uint64_t output,
                             uint64_t input,
                             uint64_t temporary,
                             uint64_t auxiliary,
                             uint32_t elements) {
    ggml_dayan_vec_detail::f32_mul(auxiliary, input, input, elements);
    ggml_dayan_vec_detail::f32_mul(auxiliary, auxiliary, input, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(auxiliary, kDayanGeluA, auxiliary, elements);
    ggml_dayan_vec_detail::f32_add(temporary, input, auxiliary, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, -2.0f * kDayanLog2E * kDayanSqrt2OverPi, temporary,
                                          elements);
    ggml_dayan_vec_detail::f32_sfu<1>(output, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_add(output, 1.0f, output, elements);
    ggml_dayan_vec_detail::f32_sfu<5>(output, output, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(output, 2.0f, output, elements);
    ggml_dayan_vec_detail::f32_scalar_mul(output, 0.5f, output, elements);
    ggml_dayan_vec_detail::f32_mul(output, input, output, elements);
}

static inline void gelu_quick_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    ggml_dayan_vec_detail::f32_scalar_mul(temporary, -kDayanGeluQuick * kDayanLog2E, input, elements);
    ggml_dayan_vec_detail::f32_sfu<1>(output, temporary, elements);
    ggml_dayan_vec_detail::f32_scalar_add(output, 1.0f, output, elements);
    ggml_dayan_vec_detail::f32_sfu<5>(output, output, elements);
    ggml_dayan_vec_detail::f32_mul(output, input, output, elements);
}

template <void (*Operation)(uint64_t, uint64_t, uint64_t, uint64_t, uint32_t)>
static void run_unary_f32(const DayanElementwiseConfig * config, const float * input, float * output) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output);
    configure_normalization_lsu();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t temporary  = uram + NormalizationLayout::kAux;
    const uint64_t auxiliary  = uram + NormalizationLayout::kValue;
    const uint64_t out_uram   = uram + NormalizationLayout::kScaled;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(input_uram, input + offset, elements);
        Operation(out_uram, input_uram, temporary, auxiliary, elements);
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void run_reduce_rows_f32(const DayanReduceConfig * config, const float * input, float * output) {
    if (config == nullptr || config->width == 0 || config->rows == 0 || input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }

    configure_normalization_lsu();
    const uint64_t uram        = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram  = uram + NormalizationLayout::kInput;
    const uint64_t accumulator = uram + NormalizationLayout::kVectorScalar;
    const uint64_t reduce      = uram + NormalizationLayout::kVecReduce;
    for (uint32_t row = task; row < config->rows; row += tasks) {
        ggml_dayan_vec_detail::f32_fill_bits(accumulator, 0U, Constants::kVecLanes);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + (uint64_t) row * config->width + offset, elements);
            ggml_dayan_vec_detail::f32_reduce_sum(reduce, input_uram, elements);
            ggml_dayan_vec_detail::f32_add(accumulator, accumulator, reduce, Constants::kVecLanes);
        }
        ggml_dayan_vec_detail::f32_scalar_mul_bits(
            accumulator, config->scale_bits, accumulator, Constants::kVecLanes);
        normalization_store_f32(output + row, accumulator, 1);
        dayan_wait_lsu();
    }
}

static inline void run_scale_f32(const DayanScaleConfig * config, const float * input, float * output) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output);
    configure_normalization_lsu();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(input_uram, input + offset, elements);
        ggml_dayan_vec_detail::f32_scalar_mul_bits(out_uram, config->scale_bits, input_uram, elements);
        if ((config->bias_bits & 0x7fffffffU) != 0U) {
            ggml_dayan_vec_detail::f32_scalar_add_bits(out_uram, config->bias_bits, out_uram, elements);
        }
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void run_clamp_f32(const DayanClampConfig * config, const float * input, float * output) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output);
    configure_normalization_lsu();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(input_uram, input + offset, elements);
        ggml_dayan_vec_detail::f32_scalar_max_bits(out_uram, config->min_bits, input_uram, elements);
        ggml_dayan_vec_detail::f32_scalar_min_bits(out_uram, config->max_bits, out_uram, elements);
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void run_l2_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t square     = uram + NormalizationLayout::kSquare;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint64_t scalar     = uram + NormalizationLayout::kVectorScalar;
    const uint64_t reduce     = uram + NormalizationLayout::kVecReduce;

    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float *  input_row  = normalization_input_row(config, input, row);
        ggml_dayan_vec_detail::f32_fill_bits(scalar, 0U, Constants::kVecLanes);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements);
            ggml_dayan_vec_detail::f32_mul(square, input_uram, input_uram, elements);
            ggml_dayan_vec_detail::f32_reduce_sum(reduce, square, elements);
            ggml_dayan_vec_detail::f32_add(scalar, scalar, reduce, Constants::kVecLanes);
        }

        ggml_dayan_vec_detail::f32_fill_bits(reduce, config->epsilon_bits, Constants::kVecLanes);
        ggml_dayan_vec_detail::f32_mul(reduce, reduce, reduce, Constants::kVecLanes);
        ggml_dayan_vec_detail::f32_max(scalar, scalar, reduce, Constants::kVecLanes);
        ggml_dayan_vec_detail::f32_sfu<4>(scalar, scalar, Constants::kVecLanes);

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements);
            ggml_dayan_vec_detail::f32_mul_scalar_address(out_uram, input_uram, scalar, elements);
            normalization_store_f32(output + row_offset + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

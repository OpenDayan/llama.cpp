#pragma once

#include "ggml-dayan-normalization.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static constexpr float kDayanLog2E       = 1.4426950408889634f;
static constexpr float kDayanLn2         = 0.6931471805599453f;
static constexpr float kDayanGeluA       = 0.044715f;
static constexpr float kDayanGeluQuick   = 1.702f;
static constexpr float kDayanSqrt2OverPi = 0.7978845608028654f;

static inline uint32_t elementwise_float_bits(float value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

template <long Group> static inline void elementwise_scalar_add(uint64_t output, float scalar, uint64_t input) {
    __tx_macro_elemwise_add_fp32<Group, 2>((void *) output, (void *) (uintptr_t) elementwise_float_bits(scalar),
                                           (void *) input);
}

template <long Group> static inline void elementwise_scalar_mul(uint64_t output, float scalar, uint64_t input) {
    __tx_macro_elemwise_mul_fp32<Group, 2>((void *) output, (void *) (uintptr_t) elementwise_float_bits(scalar),
                                           (void *) input);
}

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
    configure_normalization_macro<0>(elements);
    __tx_macro_elemwise_add_fp32<0, 0>((void *) output, (void *) left, (void *) right);
}

static inline void elementwise_mul_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) output, (void *) left, (void *) right);
}

static inline void elementwise_sub_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_elemwise_sub_fp32<0, 0>((void *) output, (void *) left, (void *) right);
}

static inline void elementwise_div_tile(uint64_t output, uint64_t left, uint64_t right, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_sfu_recip_fp32<0>((void *) output, (void *) right);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) output, (void *) left, (void *) output);
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
        dayan_wait_macro();
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void sigmoid_tile(uint64_t output,
                                uint64_t input,
                                uint64_t temporary,
                                uint64_t auxiliary,
                                uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(temporary, -kDayanLog2E, input);
    __tx_macro_sfu_exp2_fp32<0>((void *) auxiliary, (void *) temporary);
    elementwise_scalar_add<0>(temporary, 1.0f, auxiliary);
    __tx_macro_sfu_recip_fp32<0>((void *) output, (void *) temporary);
}

static inline void silu_tile(uint64_t output,
                             uint64_t input,
                             uint64_t temporary,
                             uint64_t auxiliary,
                             uint32_t elements) {
    sigmoid_tile(output, input, temporary, auxiliary, elements);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) output, (void *) input, (void *) output);
}

static inline void softplus_tile(uint64_t output,
                                 uint64_t input,
                                 uint64_t temporary,
                                 uint64_t auxiliary,
                                 uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(temporary, -1.0f, input);
    __tx_macro_elemwise_min_fp32<0, 0>((void *) temporary, (void *) input, (void *) temporary);
    elementwise_scalar_mul<0>(temporary, kDayanLog2E, temporary);
    __tx_macro_sfu_exp2_fp32<0>((void *) temporary, (void *) temporary);
    elementwise_scalar_add<0>(temporary, 1.0f, temporary);
    __tx_macro_sfu_log2_fp32<0>((void *) temporary, (void *) temporary);
    elementwise_scalar_mul<0>(temporary, kDayanLn2, temporary);
    __tx_macro_elemwise_max_fp32<0, 2>((void *) auxiliary, (void *) (uintptr_t) elementwise_float_bits(0.0f),
                                       (void *) input);
    __tx_macro_elemwise_add_fp32<0, 0>((void *) output, (void *) auxiliary, (void *) temporary);
}

static inline void square_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) output, (void *) input, (void *) input);
}

static inline void sqrt_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_sfu_rsqrt_fp32<0>((void *) temporary, (void *) input);
    __tx_macro_sfu_recip_fp32<0>((void *) output, (void *) temporary);
}

static inline void log_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_sfu_log2_fp32<0>((void *) output, (void *) input);
    elementwise_scalar_mul<0>(output, kDayanLn2, output);
}

static inline void sin_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_sfu_sin_fp32<0>((void *) output, (void *) input);
}

static inline void cos_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_sfu_cos_fp32<0>((void *) output, (void *) input);
}

static inline void abs_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(temporary, -1.0f, input);
    __tx_macro_elemwise_max_fp32<0, 0>((void *) output, (void *) input, (void *) temporary);
}

static inline void neg_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(output, -1.0f, input);
}

static inline void relu_tile(uint64_t output, uint64_t input, uint64_t, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_elemwise_max_fp32<0, 2>((void *) output, (void *) (uintptr_t) elementwise_float_bits(0.0f),
                                       (void *) input);
}

static inline void exp_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(temporary, kDayanLog2E, input);
    __tx_macro_sfu_exp2_fp32<0>((void *) output, (void *) temporary);
}

static inline void tanh_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(temporary, -2.0f * kDayanLog2E, input);
    __tx_macro_sfu_exp2_fp32<0>((void *) output, (void *) temporary);
    elementwise_scalar_add<0>(output, 1.0f, output);
    __tx_macro_sfu_recip_fp32<0>((void *) output, (void *) output);
    elementwise_scalar_mul<0>(output, 2.0f, output);
    elementwise_scalar_add<0>(output, -1.0f, output);
}

static inline void gelu_tile(uint64_t output,
                             uint64_t input,
                             uint64_t temporary,
                             uint64_t auxiliary,
                             uint32_t elements) {
    configure_normalization_macro<0>(elements);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) auxiliary, (void *) input, (void *) input);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) auxiliary, (void *) auxiliary, (void *) input);
    elementwise_scalar_mul<0>(auxiliary, kDayanGeluA, auxiliary);
    __tx_macro_elemwise_add_fp32<0, 0>((void *) temporary, (void *) input, (void *) auxiliary);
    elementwise_scalar_mul<0>(temporary, -2.0f * kDayanLog2E * kDayanSqrt2OverPi, temporary);
    __tx_macro_sfu_exp2_fp32<0>((void *) output, (void *) temporary);
    elementwise_scalar_add<0>(output, 1.0f, output);
    __tx_macro_sfu_recip_fp32<0>((void *) output, (void *) output);
    elementwise_scalar_mul<0>(output, 2.0f, output);
    elementwise_scalar_add<0>(output, -1.0f, output);
    elementwise_scalar_add<0>(output, 1.0f, output);
    elementwise_scalar_mul<0>(output, 0.5f, output);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) output, (void *) input, (void *) output);
}

static inline void gelu_quick_tile(uint64_t output, uint64_t input, uint64_t temporary, uint64_t, uint32_t elements) {
    configure_normalization_macro<0>(elements);
    elementwise_scalar_mul<0>(temporary, -kDayanGeluQuick * kDayanLog2E, input);
    __tx_macro_sfu_exp2_fp32<0>((void *) output, (void *) temporary);
    elementwise_scalar_add<0>(output, 1.0f, output);
    __tx_macro_sfu_recip_fp32<0>((void *) output, (void *) output);
    __tx_macro_elemwise_mul_fp32<0, 0>((void *) output, (void *) input, (void *) output);
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
        dayan_wait_macro();
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
    const float    scale      = normalization_bits_float(config->scale_bits);
    const uint64_t input_uram = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kInput;
    float *        partial    = (float *) ((uintptr_t) dayan_uram_base() + NormalizationLayout::kScalar);
    for (uint32_t row = task; row < config->rows; row += tasks) {
        float sum = 0.0f;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + (uint64_t) row * config->width + offset, elements);
            configure_normalization_macro<0>(elements);
            __tx_macro_reduce_add_fp32<0>(partial, (void *) input_uram);
            dayan_wait_macro();
            sum += partial[0];
        }
        partial[0] = sum * scale;
        normalization_store_f32(output + row, (uint64_t) (uintptr_t) partial, 1);
        dayan_wait_lsu();
    }
}

static inline void run_scale_f32(const DayanScaleConfig * config, const float * input, float * output) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output);
    configure_normalization_lsu();

    const float    scale      = normalization_bits_float(config->scale_bits);
    const float    bias       = normalization_bits_float(config->bias_bits);
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
        configure_normalization_macro<0>(elements);
        elementwise_scalar_mul<0>(out_uram, scale, input_uram);
        if (bias != 0.0f) {
            elementwise_scalar_add<0>(out_uram, bias, out_uram);
        }
        dayan_wait_macro();
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

    const float    minimum    = normalization_bits_float(config->min_bits);
    const float    maximum    = normalization_bits_float(config->max_bits);
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
        configure_normalization_macro<0>(elements);
        __tx_macro_elemwise_max_fp32<0, 2>((void *) out_uram, (void *) (uintptr_t) elementwise_float_bits(minimum),
                                           (void *) input_uram);
        __tx_macro_elemwise_min_fp32<0, 2>((void *) out_uram, (void *) (uintptr_t) elementwise_float_bits(maximum),
                                           (void *) out_uram);
        dayan_wait_macro();
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
    float *        scalar     = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar);

    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float *  input_row  = normalization_input_row(config, input, row);
        float          sum        = 0.0f;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements);
            configure_normalization_macro<0>(elements);
            __tx_macro_elemwise_mul_fp32<0, 0>((void *) square, (void *) input_uram, (void *) input_uram);
            __tx_macro_reduce_add_fp32<0>(scalar, (void *) square);
            dayan_wait_macro();
            sum += scalar[0];
        }

        const float epsilon = normalization_bits_float(config->epsilon_bits);
        scalar[0]           = sum;
        scalar[1]           = epsilon * epsilon;
        configure_normalization_macro<1>(1);
        __tx_macro_elemwise_max_fp32<1, 0>(scalar + 2, scalar, scalar + 1);
        __tx_macro_sfu_rsqrt_fp32<1>(scalar + 3, scalar + 2);
        dayan_wait_macro();

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements);
            configure_normalization_macro<0>(elements);
            __tx_macro_elemwise_mul_fp32<0, 1>((void *) out_uram, scalar + 3, (void *) input_uram);
            dayan_wait_macro();
            normalization_store_f32(output + row_offset + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

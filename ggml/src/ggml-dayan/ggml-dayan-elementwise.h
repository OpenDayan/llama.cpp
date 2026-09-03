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
        dayan_thread_exit(-12);
    }
}

using BinaryOperation = void (*)();

template <BinaryOperation Operation>
static void run_binary_f32(const DayanElementwiseConfig * config,
                           const float *                  left,
                           const float *                  right,
                           float *                        output) {
    if (config == nullptr || right == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, left, output);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t left_uram  = uram + NormalizationLayout::kInput;
    const uint64_t right_uram = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    ggml_dayan_vec_detail::configure_f32_tile(NormalizationLayout::kTileElements);
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, left_uram,
                       (uint64_t) (uintptr_t) (left + offset));
        load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                       (uint64_t) (uintptr_t) (right + offset));
        dayan_wait_lsu();
        constexpr long instructions = 4;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) left_uram);
        __tx_vec_lsu_u2v<1, 1>((long) right_uram);
        Operation();
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        store_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, out_uram,
                        (uint64_t) (uintptr_t) (output + offset));
        dayan_wait_lsu();
    }
}

static void run_div_f32(const DayanElementwiseConfig * config,
                        const float *                  left,
                        const float *                  right,
                        float *                        output) {
    if (config == nullptr || right == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, left, output);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t left_uram  = uram + NormalizationLayout::kInput;
    const uint64_t right_uram = uram + NormalizationLayout::kAux;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint32_t task       = dayan_native_task_id();
    const uint32_t tasks      = dayan_native_task_count();
    ggml_dayan_vec_detail::configure_f32_tile(NormalizationLayout::kTileElements);
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, left_uram,
                       (uint64_t) (uintptr_t) (left + offset));
        load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, right_uram,
                       (uint64_t) (uintptr_t) (right + offset));
        dayan_wait_lsu();
        constexpr long instructions = 5;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) left_uram);
        __tx_vec_lsu_u2v<1, 1>((long) right_uram);
        __tx_vec_v_fp32_sfu<1, 16, 5>();
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        store_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, out_uram,
                        (uint64_t) (uintptr_t) (output + offset));
        dayan_wait_lsu();
    }
}

static inline void sigmoid_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 8;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void silu_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 10;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void softplus_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 20;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 16, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLn2));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(0.0f));
    __tx_vec_vv_fp32_cmp<16, 17, 0, 0b100>();
    __tx_vec_vv_mux2<1, 16, 17, 0>();
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void square_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void sqrt_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 4>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void log_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 5;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLn2));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void sin_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 2>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void cos_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 3;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_v_fp32_sfu<0, 0, 3>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void abs_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 7;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void neg_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void relu_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 5;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(0.0f));
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void exp_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 5;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void tanh_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 12;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-2.0f * kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(2.0f));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(-1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void gelu_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 18;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(kDayanGeluA));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>(
        (long) ggml_dayan_vec_detail::f32_bits(-2.0f * kDayanLog2E * kDayanSqrt2OverPi));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void gelu_quick_tile(uint64_t output, uint64_t input) {
    constexpr long instructions = 10;

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_g2v_broadcast32<1, 1>(
        (long) ggml_dayan_vec_detail::f32_bits(-kDayanGeluQuick * kDayanLog2E));
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 1>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) ggml_dayan_vec_detail::f32_bits(1.0f));
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 5>();
    __tx_vec_lsu_u2v<1, 0>((long) input);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

using UnaryOperation = void (*)(uint64_t, uint64_t);

template <UnaryOperation Operation>
static void run_unary_f32(const DayanElementwiseConfig * config, const float * input, float * output) {
    if (config == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    elementwise_validate(config->elements, input, output);
    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();

    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
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
        ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
        Operation(out_uram, input_uram);
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
        dayan_thread_exit(-12);
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint64_t uram        = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram  = uram + NormalizationLayout::kInput;
    const uint64_t accumulator = uram + NormalizationLayout::kVectorScalar;
    for (uint32_t row = task; row < config->rows; row += tasks) {
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input + (uint64_t) row * config->width + offset, elements);
            normalization_sum_accumulate(accumulator, input_uram, elements, offset == 0);
        }
        constexpr long instructions = 4;
        ggml_dayan_vec_detail::configure_vec_grid(1);
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v_broadcast32<0, 0>((long) accumulator);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->scale_bits);
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 3>((long) accumulator);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
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
    ggml_dayan_vec_detail::configure_f32_iterators();

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
        ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
        const bool has_bias     = (config->bias_bits & 0x7fffffffU) != 0U;
        const long instructions = has_bias ? 6 : 4;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) input_uram);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->scale_bits);
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        if (has_bias) {
            __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->bias_bits);
            __tx_vec_vv_fp32_add<0, 0, 16, 0>();
        }
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
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
    ggml_dayan_vec_detail::configure_f32_iterators();

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
        ggml_dayan_vec_detail::configure_vec_grid(ggml_dayan_vec_detail::f32_chunks(elements));
        constexpr long instructions = 8;
        ggml_dayan_vec_detail::begin_vec_block(instructions);
        __tx_vec_lsu_u2v<0, 0>((long) input_uram);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->min_bits);
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->max_bits);
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
        __tx_vec_vv_mux2<0, 16, 0, 0>();
        __tx_vec_lsu_v2u<0, 3>((long) out_uram);
        ggml_dayan_vec_detail::start_vec_block(instructions);
        dayan_wait_vec();
        normalization_store_f32(output + offset, out_uram, elements);
        dayan_wait_lsu();
    }
}

static inline void l2_norm_fused(uint64_t input, uint64_t output, uint32_t elements, uint32_t epsilon_bits) {
    const long     chunks = ggml_dayan_vec_detail::f32_chunks(elements);
    const uint32_t tail   = elements % Constants::kVecLanes;
    const long     instructions = 6 + 18 * chunks + (tail == 0 ? 0 : 2);

    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_g2v_broadcast32<1, 0>(0);
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const uint64_t chunk_input = input + (uint64_t) chunk * Constants::kVecBytes;
        if (tail != 0 && chunk + 1 == chunks) {
            __tx_vec_lsu_g2v_broadcast32<0, 0>(0);
            __tx_vec_lsu_g2s<0>((long) ((uint64_t{ 1 } << tail) - 1U));
            __tx_vec_lsu_u2v_mask<0, 0, 0>((long) chunk_input);
        } else {
            __tx_vec_lsu_u2v<0, 0>((long) chunk_input);
        }
        __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
        ggml_dayan_vec_detail::emit_f32_sum_stages();
        __tx_vec_vv_fp32_add<1, 0, 22, 0>();
    }
    __tx_vec_lsu_g2v_broadcast32<0, 0>((long) epsilon_bits);
    __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
    __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
    __tx_vec_vv_mux2<1, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<1, 16, 4>();
    for (long chunk = 0; chunk < chunks; ++chunk) {
        const uint64_t offset = (uint64_t) chunk * Constants::kVecBytes;
        __tx_vec_lsu_u2v<0, 0>((long) (input + offset));
        __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
        __tx_vec_lsu_v2u<0, 0>((long) (output + offset));
    }
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void run_l2_norm_generic(const DayanNormConfig * config, const float * input, float * output) {
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t out_uram   = uram + NormalizationLayout::kValue;
    const uint64_t scalar     = uram + NormalizationLayout::kVectorScalar;

    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float *  input_row  = (const float *) normalization_input_row(config, input, row);
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements);
            normalization_square_sum_accumulate(scalar, input_uram, elements, offset == 0);
        }

        constexpr long finalize_instructions = 7;
        ggml_dayan_vec_detail::configure_vec_grid(1);
        ggml_dayan_vec_detail::begin_vec_block(finalize_instructions);
        __tx_vec_lsu_u2v_broadcast32<0, 0>((long) scalar);
        __tx_vec_lsu_g2v_broadcast32<1, 1>((long) config->epsilon_bits);
        __tx_vec_vv_fp32_mul<1, 16, 16, 0>();
        __tx_vec_vv_fp32_cmp<0, 16, 0, 0b100>();
        __tx_vec_vv_mux2<0, 0, 16, 0>();
        __tx_vec_v_fp32_sfu<0, 0, 4>();
        __tx_vec_lsu_v2u<0, 3>((long) scalar);
        ggml_dayan_vec_detail::start_vec_block(finalize_instructions);
        dayan_wait_vec();

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(input_uram, input_row + offset, elements);
            normalization_apply_rms(out_uram, input_uram, scalar, elements);
            normalization_store_f32(output + row_offset + offset, out_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void run_l2_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate<kDayanDataTypeF32>(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    if (config->width > 4U * Constants::kVecLanes) {
        run_l2_norm_generic(config, input, output);
        return;
    }

    const uint64_t uram        = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram  = uram + NormalizationLayout::kInput;
    if ((config->width % Constants::kVecLanes) == 0U) {
        const uint32_t row_bytes     = config->width * sizeof(float);
        const uint32_t rows_per_tile = DAYAN_NATIVE_TASK_URAM_BYTES / (2U * row_bytes);
        const uint64_t output_uram   = uram + (uint64_t) rows_per_tile * row_bytes;
        configure_vec_iterator<kVecV0Base>(0, 0, 0, row_bytes);
        configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
        configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
        configure_vec_iterator<kVecV3Base>(0, 0, 0, row_bytes);

        const uint32_t task  = dayan_native_task_id();
        const uint32_t tasks = dayan_native_task_count();
        for (uint64_t row_base = (uint64_t) task * rows_per_tile; row_base < config->rows;
             row_base += (uint64_t) tasks * rows_per_tile) {
            const uint32_t rows = min_int((int) (config->rows - row_base), (int) rows_per_tile);
            for (uint32_t row = 0; row < rows; ++row) {
                const float * input_row = (const float *) normalization_input_row(config, input, row_base + row);
                load_matrix<0>(row_bytes, 1, row_bytes, 1, input_uram + (uint64_t) row * row_bytes,
                               (uint64_t) (uintptr_t) input_row);
            }
            dayan_wait_lsu();
            ggml_dayan_vec_detail::configure_vec_grid(1, rows);
            l2_norm_fused(input_uram, output_uram, config->width, config->epsilon_bits);
            for (uint32_t row = 0; row < rows; ++row) {
                store_matrix<0>(row_bytes, 1, row_bytes, 1, output_uram + (uint64_t) row * row_bytes,
                                (uint64_t) (uintptr_t) (output + (row_base + row) * config->width));
            }
            dayan_wait_lsu();
        }
        return;
    }

    const uint64_t output_uram = uram + NormalizationLayout::kValue;
    ggml_dayan_vec_detail::configure_vec_grid(1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float *  input_row  = (const float *) normalization_input_row(config, input, row);
        load_matrix<0>((uint64_t) config->width * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, input_uram,
                       (uint64_t) (uintptr_t) input_row);
        dayan_wait_lsu();
        l2_norm_fused(input_uram, output_uram, config->width, config->epsilon_bits);
        normalization_store_f32(output + row_offset, output_uram, config->width);
        dayan_wait_lsu();
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

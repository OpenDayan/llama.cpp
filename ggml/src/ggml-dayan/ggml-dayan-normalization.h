#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct NormalizationLayout {
    enum : uint64_t {
        kTileElements = kDayanNativeTaskTileElements,
        kFloatBytes   = kTileElements * sizeof(float),
        kInput        = 0,
        kAux          = kInput + kFloatBytes,
        kValue        = kAux + kFloatBytes,
        kSquare       = kValue + kFloatBytes,
        kRms          = kSquare + kFloatBytes,
        kScaled       = kRms + kFloatBytes,
        kLowp         = kScaled + kFloatBytes,
        kScalar       = kLowp + kTileElements * sizeof(uint16_t),
        kVectorScalar = kScalar + 64,
        kVecReduce    = kVectorScalar + Constants::kVecBytes,
        kTotalBytes   = kVecReduce + kFloatBytes,
    };
};

static_assert(NormalizationLayout::kTotalBytes <= DAYAN_NATIVE_TASK_URAM_BYTES, "Native Task URAM layout is too large");
static_assert((NormalizationLayout::kScalar & 63U) == 0U, "Normalization scalar storage must be 64-byte aligned");

static inline uint32_t dayan_native_task_id() {
    const uint32_t tasks = dayan_task_dim(DAYAN_TASK_X);
    const uint32_t task  = dayan_task_id(DAYAN_TASK_X);
    if (dayan_task_mode() != 1U || tasks == 0U || tasks > kDayanNativeMaxTasks || task >= tasks ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        DAYAN_PARAMS_ERROR();
    }
    return task;
}

static inline uint32_t dayan_native_task_count() {
    return dayan_task_dim(DAYAN_TASK_X);
}

static inline void configure_normalization_lsu() {
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, NormalizationLayout::kFloatBytes, NormalizationLayout::kFloatBytes,
                      NormalizationLayout::kFloatBytes);
}

static inline bool normalization_negative_bits(uint32_t value) {
    return (value & 0x80000000U) != 0U && (value & 0x7fffffffU) != 0U;
}

static inline void normalization_pad_tail_bits(uint64_t uram, uint32_t elements, uint32_t value_bits) {
    const uint64_t mask = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kScalar;
    ggml_dayan_vec_detail::f32_pad_tail_bits(uram, mask, value_bits, elements);
}

static inline void normalization_pad_tail_address(uint64_t uram, uint32_t elements, uint64_t scalar) {
    const uint64_t mask = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kScalar;
    ggml_dayan_vec_detail::f32_pad_tail_address(uram, mask, scalar, elements);
}

static inline void normalization_load_bits(uint64_t uram,
                                           const float * dram,
                                           uint32_t      elements,
                                           uint32_t      padding_bits) {
    load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
    normalization_pad_tail_bits(uram, elements, padding_bits);
}

static inline void normalization_load(uint64_t uram, const float * dram, uint32_t elements) {
    normalization_load_bits(uram, dram, elements, 0U);
}

using NormalizationLoad = void (*)(uint64_t, const void *, uint32_t);

static inline void normalization_load_f32(uint64_t uram, const void * dram, uint32_t elements) {
    normalization_load(uram, (const float *) dram, elements);
}

template <void (*Convert)()>
static inline void normalization_load_lowp(uint64_t uram, const void * dram, uint32_t elements) {
    const uint64_t staging = (uint64_t) (uintptr_t) dayan_uram_base() + NormalizationLayout::kLowp;
    load_matrix<0>((uint64_t) elements * sizeof(uint16_t), 1,
                   NormalizationLayout::kTileElements * sizeof(uint16_t), 1, staging,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(uram, staging, elements);
    normalization_pad_tail_bits(uram, elements, 0U);
}

static inline void normalization_load_scalar_pad(uint64_t uram,
                                                 const float * dram,
                                                 uint32_t      elements,
                                                 uint64_t      scalar) {
    load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
    normalization_pad_tail_address(uram, elements, scalar);
}

static inline void normalization_store_f32(float * dram, uint64_t uram, uint32_t elements) {
    store_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                    (uint64_t) (uintptr_t) dram);
}

static inline void normalization_store_lowp(uint16_t * dram, uint64_t uram, uint32_t elements) {
    store_matrix<0>((uint64_t) elements * sizeof(uint16_t), 1, NormalizationLayout::kFloatBytes, 1, uram,
                    (uint64_t) (uintptr_t) dram);
}

static inline void normalization_vec_add(uint64_t left, uint64_t right, uint64_t output, uint32_t elements) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) left);
    __tx_vec_lsu_u2v<1, 1>((long) right);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

template <typename Format>
static inline void normalization_quantize(uint64_t output, uint64_t input, uint64_t scratch, uint32_t elements) {
    ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
        { input, (long) elements * (long) sizeof(float) },
        { scratch, (long) elements * (long) sizeof(float) },
        { output, (long) elements * (long) sizeof(uint16_t) }, 1, elements);
}

template <uint32_t InputType>
static inline void normalization_validate(const DayanNormConfig * config) {
    static_assert(InputType == kDayanDataTypeF32 || InputType == kDayanDataTypeBF16 ||
                  InputType == kDayanDataTypeF16);
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);

    if (config == nullptr || config->width == 0 || config->rows == 0 ||
        normalization_negative_bits(config->epsilon_bits) || config->input_ne[0] != config->width ||
        config->input_type != InputType || config->input_nb[0] != input_bytes || config->input_ne[1] == 0 ||
        config->input_ne[2] == 0 || config->input_ne[3] == 0 ||
        (uint64_t) config->input_ne[1] * config->input_ne[2] * config->input_ne[3] != config->rows) {
        DAYAN_PARAMS_ERROR();
    }
    (void) dayan_native_task_id();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }
}

static inline const void * normalization_input_row(const DayanNormConfig * config, const void * input, uint32_t row) {
    uint32_t       index  = row;
    const uint32_t index1 = index % config->input_ne[1];
    index /= config->input_ne[1];
    const uint32_t index2 = index % config->input_ne[2];
    const uint32_t index3 = index / config->input_ne[2];
    return (const char *) input + (uint64_t) index1 * config->input_nb[1] +
           (uint64_t) index2 * config->input_nb[2] + (uint64_t) index3 * config->input_nb[3];
}

static inline void normalization_sum_accumulate(uint64_t accumulator,
                                                uint64_t input,
                                                uint32_t elements,
                                                bool initialize) {
    const long chunks       = ggml_dayan_vec_detail::f32_chunks(elements);
    const long instructions = 2 + 14 * chunks;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV1Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV2Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    if (initialize) {
        __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
    } else {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) accumulator);
    }
    for (long chunk = 0; chunk < chunks; ++chunk) {
        __tx_vec_lsu_u2v<0, 0>((long) (input + (uint64_t) chunk * Constants::kVecBytes));
        ggml_dayan_vec_detail::emit_f32_sum_stages();
        __tx_vec_vv_fp32_add<1, 0, 22, 0>();
    }
    __tx_vec_lsu_v2u<16, 3>((long) accumulator);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_square_sum_accumulate(uint64_t accumulator,
                                                       uint64_t input,
                                                       uint32_t elements,
                                                       bool initialize) {
    const long chunks       = ggml_dayan_vec_detail::f32_chunks(elements);
    const long instructions = 2 + 15 * chunks;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV1Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV2Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    if (initialize) {
        __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
    } else {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) accumulator);
    }
    for (long chunk = 0; chunk < chunks; ++chunk) {
        __tx_vec_lsu_u2v<0, 0>((long) (input + (uint64_t) chunk * Constants::kVecBytes));
        __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
        ggml_dayan_vec_detail::emit_f32_sum_stages();
        __tx_vec_vv_fp32_add<1, 0, 22, 0>();
    }
    __tx_vec_lsu_v2u<16, 3>((long) accumulator);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_centered_square_sum_accumulate(uint64_t accumulator,
                                                                 uint64_t input,
                                                                 uint64_t mean,
                                                                 uint32_t elements,
                                                                 bool initialize) {
    const long chunks       = ggml_dayan_vec_detail::f32_chunks(elements);
    const long instructions = 2 + 17 * chunks;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV0Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV1Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV2Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::configure_vec_iterator<ggml_dayan_vec_detail::kVecV3Base>(0, 0, 0, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    if (initialize) {
        __tx_vec_lsu_g2v_broadcast32<1, 1>(0);
    } else {
        __tx_vec_lsu_u2v_broadcast32<1, 1>((long) accumulator);
    }
    for (long chunk = 0; chunk < chunks; ++chunk) {
        __tx_vec_lsu_u2v<0, 0>((long) (input + (uint64_t) chunk * Constants::kVecBytes));
        __tx_vec_lsu_u2v_broadcast32<0, 0>((long) mean);
        __tx_vec_vv_fp32_sub<0, 0, 1, 0>();
        __tx_vec_vv_fp32_mul<0, 0, 0, 0>();
        ggml_dayan_vec_detail::emit_f32_sum_stages();
        __tx_vec_vv_fp32_add<1, 0, 22, 0>();
    }
    __tx_vec_lsu_v2u<16, 3>((long) accumulator);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_mean(uint64_t output, uint64_t sum, uint32_t width) {
    constexpr long instructions = 6;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) sum);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) width);
    __tx_vec_cvt_fp32_int32<1, 16, 0>();
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_inverse_rms_finalize(uint64_t inverse, uint32_t width, uint32_t epsilon_bits) {
    constexpr long instructions = 9;

    ggml_dayan_vec_detail::configure_vec_grid(1);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v_broadcast32<0, 0>((long) inverse);
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) width);
    __tx_vec_cvt_fp32_int32<1, 16, 0>();
    __tx_vec_v_fp32_sfu<1, 16, 5>();
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_g2v_broadcast32<1, 1>((long) epsilon_bits);
    __tx_vec_vv_fp32_add<0, 0, 16, 0>();
    __tx_vec_v_fp32_sfu<0, 0, 4>();
    __tx_vec_lsu_v2u<0, 3>((long) inverse);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_apply_rms(uint64_t output, uint64_t input, uint64_t inverse, uint32_t elements) {
    constexpr long instructions = 4;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) inverse);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_apply_rms_scale(uint64_t rms,
                                                 uint64_t scaled,
                                                 uint64_t input,
                                                 uint64_t inverse,
                                                 uint64_t scale,
                                                 uint32_t elements,
                                                 bool write_rms) {
    const long instructions = write_rms ? 8 : 6;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v_broadcast32<1, 2>((long) inverse);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    if (write_rms) {
        __tx_vec_shuffle_xor<1, 0>(0);
    }
    __tx_vec_lsu_u2v<1, 1>((long) scale);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) scaled);
    if (write_rms) {
        __tx_vec_lsu_v2u<17, 3>((long) rms);
    }
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

static inline void normalization_center_scale(uint64_t output,
                                              uint64_t input,
                                              uint64_t mean,
                                              uint64_t inverse,
                                              uint32_t elements) {
    constexpr long instructions = 6;

    ggml_dayan_vec_detail::configure_f32_iterators();
    ggml_dayan_vec_detail::configure_f32_tile(elements, 0);
    ggml_dayan_vec_detail::begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>((long) input);
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) mean);
    __tx_vec_vv_fp32_sub<0, 0, 16, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 1>((long) inverse);
    __tx_vec_vv_fp32_mul<0, 0, 16, 0>();
    __tx_vec_lsu_v2u<0, 3>((long) output);
    ggml_dayan_vec_detail::start_vec_block(instructions);
    dayan_wait_vec();
}

template <typename Format, void (*Convert)(), uint32_t InputType, NormalizationLoad Load, bool HasResidual>
static uint64_t normalization_inverse_rms(const DayanNormConfig * config,
                                          const void *            left,
                                          const void *            right,
                                          void *                  residual,
                                          bool                    write_residual) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const uint64_t uram    = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input   = uram + NormalizationLayout::kInput;
    const uint64_t aux     = uram + NormalizationLayout::kAux;
    const uint64_t value   = uram + NormalizationLayout::kValue;
    const uint64_t inverse = uram + NormalizationLayout::kVectorScalar;

    for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
        const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
        if constexpr (HasResidual) {
            Load(input, (const char *) left + (uint64_t) offset * input_bytes, elements);
            Load(aux, (const char *) right + (uint64_t) offset * input_bytes, elements);
            normalization_vec_add(input, aux, value, elements);
            if constexpr (InputType != kDayanDataTypeF32) {
                const uint64_t lowp = uram + NormalizationLayout::kLowp;
                normalization_quantize<Format>(lowp, value, input, elements);
                configure_normalization_lsu();
                ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(value, lowp, elements);
                if (write_residual) {
                    normalization_store_lowp((uint16_t *) residual + offset, lowp, elements);
                    dayan_wait_lsu();
                }
            } else if (write_residual) {
                    normalization_store_f32((float *) residual + offset, value, elements);
                dayan_wait_lsu();
            }
        } else {
            Load(value, (const char *) left + (uint64_t) offset * input_bytes, elements);
        }

        normalization_square_sum_accumulate(inverse, value, elements, offset == 0);
    }

    normalization_inverse_rms_finalize(inverse, config->width, config->epsilon_bits);
    return inverse;
}

template <typename Format, void (*Convert)(), uint32_t InputType, NormalizationLoad Load, bool HasResidual, bool Quantize>
static void run_rms_norm_scale(const DayanNormPipelineConfig * config,
                               const void *                    left,
                               const void *                    right,
                               const float *                   scale,
                               void *                          residual,
                               float *                         rms,
                               float *                         scaled,
                               uint16_t *                      lowp) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const bool write_residual = (config->flags & kDayanNormWriteResidual) != 0U;
    const bool write_rms      = (config->flags & kDayanNormWriteRms) != 0U;
    const bool write_scaled   = (config->flags & kDayanNormWriteScaled) != 0U;

    const DayanNormConfig * norm = &config->norm;
    normalization_validate<InputType>(norm);
    if (left == nullptr || scale == nullptr || (HasResidual && right == nullptr) ||
        (write_residual && residual == nullptr) || (write_rms && rms == nullptr) ||
        (write_scaled && scaled == nullptr) || (Quantize && lowp == nullptr)) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    for (uint32_t row = dayan_native_task_id(); row < norm->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset   = (uint64_t) row * norm->width;
        const void *    left_row     = normalization_input_row(norm, left, row);
        const void *    right_row    = HasResidual ? normalization_input_row(norm, right, row) : nullptr;
        void * residual_row = HasResidual && write_residual ?
                                  (char *) residual + row_offset * input_bytes :
                                  nullptr;
        const uint64_t inverse_uram = normalization_inverse_rms<Format, Convert, InputType, Load, HasResidual>(
            norm, left_row, right_row, residual_row, write_residual);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t input        = uram + NormalizationLayout::kInput;
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t aux          = uram + NormalizationLayout::kAux;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        const uint64_t scaled_uram  = uram + NormalizationLayout::kScaled;
        const uint64_t lowp_uram    = uram + NormalizationLayout::kLowp;
        for (uint32_t offset = 0; offset < norm->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (norm->width - offset), (int) NormalizationLayout::kTileElements);
            if constexpr (HasResidual) {
                if (write_residual) {
                    Load(value, (const char *) residual_row + (uint64_t) offset * input_bytes, elements);
                } else {
                    Load(input, (const char *) left_row + (uint64_t) offset * input_bytes, elements);
                    Load(aux, (const char *) right_row + (uint64_t) offset * input_bytes, elements);
                    normalization_vec_add(input, aux, value, elements);
                    if constexpr (InputType != kDayanDataTypeF32) {
                        normalization_quantize<Format>(lowp_uram, value, input, elements);
                        configure_normalization_lsu();
                        ggml_dayan_vec_detail::issue_lowp_to_f32<Convert>(value, lowp_uram, elements);
                    }
                }
            } else {
                Load(value, (const char *) left_row + (uint64_t) offset * input_bytes, elements);
            }
            normalization_load(aux, scale + offset, elements);

            normalization_apply_rms_scale(
                rms_uram, scaled_uram, value, inverse_uram, aux, elements, write_rms);
            if constexpr (Quantize) {
                normalization_quantize<Format>(lowp_uram, scaled_uram, input, elements);
                configure_normalization_lsu();
            }

            bool issued_store = false;
            if (write_rms) {
                normalization_store_f32(rms + row_offset + offset, rms_uram, elements);
                issued_store = true;
            }
            if (write_scaled) {
                normalization_store_f32(scaled + row_offset + offset, scaled_uram, elements);
                issued_store = true;
            }
            if constexpr (Quantize) {
                normalization_store_lowp(lowp + row_offset + offset, lowp_uram, elements);
                issued_store = true;
            }
            if (issued_store) {
                dayan_wait_lsu();
            }
        }
    }
}

template <uint32_t InputType, NormalizationLoad Load>
static inline void run_rms_norm_typed(const DayanNormConfig * config, const void * input, float * output) {
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    normalization_validate<InputType>(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset   = (uint64_t) row * config->width;
        const void *    input_row    = normalization_input_row(config, input, row);
        const uint64_t inverse_uram = normalization_inverse_rms<
            Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, InputType, Load, false>(
            config, input_row, nullptr, nullptr, false);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            Load(value, (const char *) input_row + (uint64_t) offset * input_bytes, elements);
            normalization_apply_rms(rms_uram, value, inverse_uram, elements);
            normalization_store_f32(output + row_offset + offset, rms_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void run_rms_norm(const DayanNormConfig * config, const void * input, float * output) {
    if (config->input_type == kDayanDataTypeF32) {
        run_rms_norm_typed<kDayanDataTypeF32, normalization_load_f32>(config, input, output);
    } else if (config->input_type == kDayanDataTypeBF16) {
        run_rms_norm_typed<kDayanDataTypeBF16,
                           normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>>(config, input, output);
    } else if (config->input_type == kDayanDataTypeF16) {
        run_rms_norm_typed<kDayanDataTypeF16,
                           normalization_load_lowp<__tx_vec_cvt_fp32_fp16<0, 0, 0>>>(config, input, output);
    } else {
        DAYAN_PARAMS_ERROR();
    }
}

static inline void normalization_centered(const float * input,
                                          float *       output,
                                          uint64_t      elements,
                                          uint32_t      epsilon_bits) {
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t mean       = uram + NormalizationLayout::kRms;
    const uint64_t scaled     = uram + NormalizationLayout::kScaled;
    const uint64_t inverse    = uram + NormalizationLayout::kVectorScalar;

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        normalization_sum_accumulate(inverse, input_uram, count, offset == 0);
    }

    normalization_mean(mean, inverse, (uint32_t) elements);
    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load_scalar_pad(input_uram, input + offset, count, mean);
        normalization_centered_square_sum_accumulate(inverse, input_uram, mean, count, offset == 0);
    }

    normalization_inverse_rms_finalize(inverse, (uint32_t) elements, epsilon_bits);

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        normalization_center_scale(scaled, input_uram, mean, inverse, count);
        normalization_store_f32(output + offset, scaled, count);
        dayan_wait_lsu();
    }
}

static inline void run_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate<kDayanDataTypeF32>(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t offset = (uint64_t) row * config->width;
        normalization_centered(
            (const float *) normalization_input_row(config, input, row), output + offset, config->width,
            config->epsilon_bits);
    }
}

static inline void run_group_norm(const DayanGroupNormConfig * config, const float * input, float * output) {
    if (config == nullptr || config->width == 0 || config->height == 0 || config->channels == 0 ||
        config->batches == 0 || config->groups == 0 || config->groups > config->channels || input == nullptr ||
        output == nullptr || normalization_negative_bits(config->epsilon_bits)) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint32_t channels_per_group = (config->channels + config->groups - 1U) / config->groups;
    const uint64_t channel_elements   = (uint64_t) config->width * config->height;
    const uint64_t work_items         = (uint64_t) config->batches * config->groups;
    for (uint64_t work = task; work < work_items; work += tasks) {
        const uint32_t batch         = (uint32_t) (work / config->groups);
        const uint32_t group         = (uint32_t) (work % config->groups);
        const uint32_t channel_start = group * channels_per_group;
        const uint32_t channel_end   = channel_start + channels_per_group < config->channels ?
                                           channel_start + channels_per_group :
                                           config->channels;
        const uint64_t offset        = ((uint64_t) batch * config->channels + channel_start) * channel_elements;
        const uint64_t elements      = (uint64_t) (channel_end - channel_start) * channel_elements;
        normalization_centered(input + offset, output + offset, elements, config->epsilon_bits);
    }
}

template <typename Format, uint32_t InputType, NormalizationLoad Load>
static inline void run_residual_add(const DayanElementwiseConfig * config,
                                    const void *                   left,
                                    const void *                   right,
                                    void *                         output) {
    if (config == nullptr || config->elements == 0 || left == nullptr || right == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    constexpr uint32_t input_bytes = InputType == kDayanDataTypeF32 ? sizeof(float) : sizeof(uint16_t);
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_thread_exit(-12);
    }

    configure_normalization_lsu();
    ggml_dayan_vec_detail::configure_f32_iterators();
    const uint64_t uram  = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input = uram + NormalizationLayout::kInput;
    const uint64_t aux   = uram + NormalizationLayout::kAux;
    const uint64_t value = uram + NormalizationLayout::kValue;
    for (uint64_t offset = (uint64_t) task * NormalizationLayout::kTileElements; offset < config->elements;
         offset += (uint64_t) tasks * NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements  = remaining < NormalizationLayout::kTileElements ?
                                       (uint32_t) remaining :
                                       (uint32_t) NormalizationLayout::kTileElements;
        Load(input, (const char *) left + offset * input_bytes, elements);
        Load(aux, (const char *) right + offset * input_bytes, elements);
        normalization_vec_add(input, aux, value, elements);
        if constexpr (InputType == kDayanDataTypeF32) {
            normalization_store_f32((float *) output + offset, value, elements);
        } else {
            const uint64_t lowp = uram + NormalizationLayout::kLowp;
            normalization_quantize<Format>(lowp, value, input, elements);
            configure_normalization_lsu();
            normalization_store_lowp((uint16_t *) output + offset, lowp, elements);
        }
        dayan_wait_lsu();
    }
}

template <typename Format, void (*Convert)(), uint32_t InputType, NormalizationLoad Load, bool HasResidual>
static inline void run_rms_norm_pipeline_typed(const DayanNormPipelineConfig * config,
                                               const void *                    left,
                                               const void *                    right,
                                               const float *                   scale,
                                               void *                          residual,
                                               float *                         rms,
                                               float *                         scaled,
                                               uint16_t *                      lowp) {
    const bool write_lowp = (config->flags & kDayanNormWriteLowp) != 0U;
    if (!write_lowp) {
        run_rms_norm_scale<Format, Convert, InputType, Load, HasResidual, false>(
            config, left, right, scale, residual, rms, scaled, lowp);
    } else {
        run_rms_norm_scale<Format, Convert, InputType, Load, HasResidual, true>(
            config, left, right, scale, residual, rms, scaled, lowp);
    }
}

static inline void run_rms_norm_pipeline(const DayanNormPipelineConfig * config,
                                         const void *                    left,
                                         const void *                    right,
                                         const float *                   scale,
                                         void *                          residual,
                                         float *                         rms,
                                         float *                         scaled,
                                         uint16_t *                      lowp) {
    const uint32_t write_flags = kDayanNormWriteResidual | kDayanNormWriteRms | kDayanNormWriteScaled |
                                 kDayanNormWriteLowp;
    const uint32_t allowed = kDayanNormHasResidual | write_flags;
    if (config == nullptr || (config->flags & write_flags) == 0U || (config->flags & ~allowed) != 0U ||
        ((config->flags & kDayanNormWriteResidual) != 0U && (config->flags & kDayanNormHasResidual) == 0U)) {
        DAYAN_PARAMS_ERROR();
    }

    const bool has_residual = (config->flags & kDayanNormHasResidual) != 0U;
    const bool write_lowp   = (config->flags & kDayanNormWriteLowp) != 0U;
    if ((write_lowp && config->output_type != kDayanDataTypeBF16 && config->output_type != kDayanDataTypeF16) ||
        (has_residual && write_lowp && config->norm.input_type != kDayanDataTypeF32 &&
         config->norm.input_type != config->output_type)) {
        DAYAN_PARAMS_ERROR();
    }

    if (config->norm.input_type == kDayanDataTypeF32) {
        if (write_lowp && config->output_type == kDayanDataTypeF16) {
            if (has_residual) {
                run_rms_norm_pipeline_typed<
                    Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, true>(
                    config, left, right, scale, residual, rms, scaled, lowp);
            } else {
                run_rms_norm_pipeline_typed<
                    Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, false>(
                    config, left, right, scale, residual, rms, scaled, lowp);
            }
        } else if (has_residual) {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, true>(
                config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeF32, normalization_load_f32, false>(
                config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else if (config->norm.input_type == kDayanDataTypeBF16) {
        if (has_residual) {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeBF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>, true>(
                config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_pipeline_typed<
                Bf16Format, __tx_vec_cvt_fp32_bf16<0, 0, 0>, kDayanDataTypeBF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_bf16<0, 0, 0>>, false>(
                config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else if (config->norm.input_type == kDayanDataTypeF16) {
        if (has_residual) {
            run_rms_norm_pipeline_typed<
                Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_fp16<0, 0, 0>>, true>(
                config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_pipeline_typed<
                Fp16Format, __tx_vec_cvt_fp32_fp16<0, 0, 0>, kDayanDataTypeF16,
                normalization_load_lowp<__tx_vec_cvt_fp32_fp16<0, 0, 0>>, false>(
                config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else {
        DAYAN_PARAMS_ERROR();
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

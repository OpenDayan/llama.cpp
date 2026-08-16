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
static_assert((NormalizationLayout::kScalar & 63U) == 0U, "Normalization scalar storage must be cache-line aligned");

static inline void normalization_scalar_acquire(const void * scalar) {
    dayan_cache_invalidate(scalar, 64);
}

static inline void normalization_scalar_release(const void * scalar) {
    dayan_cache_flush(scalar, 64);
}

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
    ggml_dayan_vec_detail::f32_add(output, left, right, elements);
}

template <typename Format>
static inline void normalization_quantize(uint64_t output, uint64_t input, uint64_t scratch, uint32_t elements) {
    ggml_dayan_vec_detail::convert_f32_to_lowp<Format::kMatDataType, 0>(
        { input, (long) elements * (long) sizeof(float) },
        { scratch, (long) elements * (long) sizeof(float) },
        { output, (long) elements * (long) sizeof(uint16_t) }, 1, elements);
}

static inline void normalization_validate(const DayanNormConfig * config) {
    if (config == nullptr || config->width == 0 || config->rows == 0 ||
        normalization_negative_bits(config->epsilon_bits) || config->input_ne[0] != config->width ||
        config->input_nb[0] != sizeof(float) || config->input_ne[1] == 0 || config->input_ne[2] == 0 ||
        config->input_ne[3] == 0 ||
        (uint64_t) config->input_ne[1] * config->input_ne[2] * config->input_ne[3] != config->rows) {
        DAYAN_PARAMS_ERROR();
    }
    (void) dayan_native_task_id();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }
}

static inline const float * normalization_input_row(const DayanNormConfig * config, const float * input, uint32_t row) {
    uint32_t       index  = row;
    const uint32_t index1 = index % config->input_ne[1];
    index /= config->input_ne[1];
    const uint32_t index2 = index % config->input_ne[2];
    const uint32_t index3 = index / config->input_ne[2];
    return (const float *) ((const char *) input + (uint64_t) index1 * config->input_nb[1] +
                            (uint64_t) index2 * config->input_nb[2] + (uint64_t) index3 * config->input_nb[3]);
}

template <bool HasResidual>
static uint64_t normalization_inverse_rms(const DayanNormConfig * config,
                                          const float *           left,
                                          const float *           right,
                                          float *                 residual,
                                          bool                    write_residual) {
    const uint64_t uram    = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input   = uram + NormalizationLayout::kInput;
    const uint64_t aux     = uram + NormalizationLayout::kAux;
    const uint64_t value   = uram + NormalizationLayout::kValue;
    const uint64_t square  = uram + NormalizationLayout::kSquare;
    const uint64_t inverse = uram + NormalizationLayout::kVectorScalar;
    const uint64_t reduce  = uram + NormalizationLayout::kVecReduce;

    ggml_dayan_vec_detail::f32_fill_bits(inverse, 0U, Constants::kVecLanes);

    for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
        const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
        if constexpr (HasResidual) {
            normalization_load(input, left + offset, elements);
            normalization_load(aux, right + offset, elements);
            normalization_vec_add(input, aux, value, elements);
            if (write_residual) {
                normalization_store_f32(residual + offset, value, elements);
                dayan_wait_lsu();
            }
        } else {
            normalization_load(value, left + offset, elements);
        }

        ggml_dayan_vec_detail::f32_mul(square, value, value, elements);
        ggml_dayan_vec_detail::f32_reduce_sum(reduce, square, elements);
        ggml_dayan_vec_detail::f32_add(inverse, inverse, reduce, Constants::kVecLanes);
    }

    ggml_dayan_vec_detail::f32_int32_scalar(square, config->width);
    ggml_dayan_vec_detail::f32_div(inverse, inverse, square, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_scalar_add_bits(inverse, config->epsilon_bits, inverse, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_sfu<4>(inverse, inverse, Constants::kVecLanes);
    return inverse;
}

template <typename Format, bool HasResidual, bool Quantize>
static void run_rms_norm_scale(const DayanNormPipelineConfig * config,
                               const float *                   left,
                               const float *                   right,
                               const float *                   scale,
                               float *                         residual,
                               float *                         rms,
                               float *                         scaled,
                               uint16_t *                      lowp) {
    const bool write_residual = (config->flags & kDayanNormWriteResidual) != 0U;
    const bool write_rms      = (config->flags & kDayanNormWriteRms) != 0U;
    const bool write_scaled   = (config->flags & kDayanNormWriteScaled) != 0U;

    const DayanNormConfig * norm = &config->norm;
    normalization_validate(norm);
    if (left == nullptr || scale == nullptr || (HasResidual && right == nullptr) ||
        (write_residual && residual == nullptr) || (write_rms && rms == nullptr) ||
        (write_scaled && scaled == nullptr) || (Quantize && lowp == nullptr)) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    for (uint32_t row = dayan_native_task_id(); row < norm->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset   = (uint64_t) row * norm->width;
        const float *  left_row     = normalization_input_row(norm, left, row);
        const float *  right_row    = HasResidual ? normalization_input_row(norm, right, row) : nullptr;
        float * residual_row = HasResidual && write_residual ? residual + row_offset : nullptr;
        const uint64_t inverse_uram =
            normalization_inverse_rms<HasResidual>(norm, left_row, right_row, residual_row, write_residual);
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
                    normalization_load(value, residual_row + offset, elements);
                } else {
                    normalization_load(input, left_row + offset, elements);
                    normalization_load(aux, right_row + offset, elements);
                    normalization_vec_add(input, aux, value, elements);
                }
            } else {
                normalization_load(value, left_row + offset, elements);
            }
            normalization_load(aux, scale + offset, elements);

            ggml_dayan_vec_detail::f32_mul_scalar_address(rms_uram, value, inverse_uram, elements);
            ggml_dayan_vec_detail::f32_mul(scaled_uram, rms_uram, aux, elements);
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

static inline void run_rms_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t row_offset   = (uint64_t) row * config->width;
        const float *  input_row    = normalization_input_row(config, input, row);
        const uint64_t inverse_uram = normalization_inverse_rms<false>(config, input_row, nullptr, nullptr, false);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(value, input_row + offset, elements);
            ggml_dayan_vec_detail::f32_mul_scalar_address(rms_uram, value, inverse_uram, elements);
            normalization_store_f32(output + row_offset + offset, rms_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void normalization_centered(const float * input,
                                          float *       output,
                                          uint64_t      elements,
                                          uint32_t      epsilon_bits) {
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t width      = uram + NormalizationLayout::kAux;
    const uint64_t centered   = uram + NormalizationLayout::kValue;
    const uint64_t square     = uram + NormalizationLayout::kSquare;
    const uint64_t mean       = uram + NormalizationLayout::kRms;
    const uint64_t scaled     = uram + NormalizationLayout::kScaled;
    const uint64_t inverse    = uram + NormalizationLayout::kVectorScalar;
    const uint64_t reduce     = uram + NormalizationLayout::kVecReduce;

    ggml_dayan_vec_detail::f32_fill_bits(inverse, 0U, Constants::kVecLanes);

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        ggml_dayan_vec_detail::f32_reduce_sum(reduce, input_uram, count);
        ggml_dayan_vec_detail::f32_add(inverse, inverse, reduce, Constants::kVecLanes);
    }

    ggml_dayan_vec_detail::f32_int32_scalar(width, (uint32_t) elements);
    ggml_dayan_vec_detail::f32_div(mean, inverse, width, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_fill_bits(inverse, 0U, Constants::kVecLanes);
    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load_scalar_pad(input_uram, input + offset, count, mean);
        ggml_dayan_vec_detail::f32_sub_scalar_address(centered, input_uram, mean, count);
        ggml_dayan_vec_detail::f32_mul(square, centered, centered, count);
        ggml_dayan_vec_detail::f32_reduce_sum(reduce, square, count);
        ggml_dayan_vec_detail::f32_add(inverse, inverse, reduce, Constants::kVecLanes);
    }

    ggml_dayan_vec_detail::f32_div(inverse, inverse, width, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_scalar_add_bits(inverse, epsilon_bits, inverse, Constants::kVecLanes);
    ggml_dayan_vec_detail::f32_sfu<4>(inverse, inverse, Constants::kVecLanes);

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        ggml_dayan_vec_detail::f32_sub_scalar_address(centered, input_uram, mean, count);
        ggml_dayan_vec_detail::f32_mul_scalar_address(scaled, centered, inverse, count);
        normalization_store_f32(output + offset, scaled, count);
        dayan_wait_lsu();
    }
}

static inline void run_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate(config);
    if (input == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    configure_normalization_lsu();
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t offset = (uint64_t) row * config->width;
        normalization_centered(
            normalization_input_row(config, input, row), output + offset, config->width, config->epsilon_bits);
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
        dayan_proc_exit(-12);
    }

    configure_normalization_lsu();
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

static inline void run_residual_add(const DayanElementwiseConfig * config,
                                    const float *                  left,
                                    const float *                  right,
                                    float *                        output) {
    if (config == nullptr || config->elements == 0 || left == nullptr || right == nullptr || output == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    const uint32_t task  = dayan_native_task_id();
    const uint32_t tasks = dayan_native_task_count();
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }

    configure_normalization_lsu();
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
        normalization_load(input, left + offset, elements);
        normalization_load(aux, right + offset, elements);
        normalization_vec_add(input, aux, value, elements);
        normalization_store_f32(output + offset, value, elements);
        dayan_wait_lsu();
    }
}

static inline void run_rms_norm_pipeline(const DayanNormPipelineConfig * config,
                                         const float *                   left,
                                         const float *                   right,
                                         const float *                   scale,
                                         float *                         residual,
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
    if (write_lowp && config->output_type != kDayanDataTypeBF16 && config->output_type != kDayanDataTypeF16) {
        DAYAN_PARAMS_ERROR();
    }

    if (has_residual) {
        if (!write_lowp) {
            run_rms_norm_scale<Bf16Format, true, false>(config, left, right, scale, residual, rms, scaled, lowp);
        } else if (config->output_type == kDayanDataTypeBF16) {
            run_rms_norm_scale<Bf16Format, true, true>(config, left, right, scale, residual, rms, scaled, lowp);
        } else {
            run_rms_norm_scale<Fp16Format, true, true>(config, left, right, scale, residual, rms, scaled, lowp);
        }
    } else if (!write_lowp) {
        run_rms_norm_scale<Bf16Format, false, false>(config, left, right, scale, residual, rms, scaled, lowp);
    } else if (config->output_type == kDayanDataTypeBF16) {
        run_rms_norm_scale<Bf16Format, false, true>(config, left, right, scale, residual, rms, scaled, lowp);
    } else {
        run_rms_norm_scale<Fp16Format, false, true>(config, left, right, scale, residual, rms, scaled, lowp);
    }
}

}  // namespace ggml_dayan_matmul_detail

#endif

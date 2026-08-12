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
        kTotalBytes   = kScalar + 32,
    };
};

static_assert(NormalizationLayout::kTotalBytes <= DAYAN_NATIVE_TASK_URAM_BYTES, "Native Task URAM layout is too large");

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

static inline float normalization_bits_float(uint32_t value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = value;
    return bits.f;
}

static inline void configure_normalization_lsu() {
    long checksum = 0;
    set_lsu_matrix<0>(checksum, 1, NormalizationLayout::kFloatBytes, NormalizationLayout::kFloatBytes,
                      NormalizationLayout::kFloatBytes);
}

template <long Group> static inline void configure_normalization_macro(uint32_t elements) {
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM0>(elements - 1U);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM1>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM2>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DIM3>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0DIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1DIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM1STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM2STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTDIM3STRIDE>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC0SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_SRC1SWIZZLEMBS>(0);
    __tx_macro_tcsr_write<Group * MACRO_GROUPSZ + MACRO_DSTSWIZZLEMBS>(0);
}

static inline void normalization_load(uint64_t uram, const float * dram, uint32_t elements) {
    load_matrix<0>((uint64_t) elements * sizeof(float), 1, NormalizationLayout::kFloatBytes, 1, uram,
                   (uint64_t) (uintptr_t) dram);
    dayan_wait_lsu();
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
    const long chunks = (elements + Constants::kVecLanes - 1U) / Constants::kVecLanes;

    __tx_vec_tcsr_write<kVecSync>(0);
    configure_vec_dims(1, chunks);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, Constants::kVecBytes);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, Constants::kVecBytes);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(4);
    __tx_vec_lsu_u2v<0, 0>((long) left);
    __tx_vec_lsu_u2v<1, 1>((long) right);
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>((long) output);
    start_vec_block(4);
    long done;
    __tx_vec_sb_bar(done);
    __tx_vec_sb_bar(done);
}

template <typename Format>
static inline void normalization_quantize(uint64_t output, uint64_t input, uint32_t elements) {
    ggml_dayan_vec_detail::issue_f32_to_lowp<Format::kMatDataType>(
        { input, (long) elements * (long) sizeof(float) },
        { output, (long) elements * (long) sizeof(uint16_t) }, 1, elements);
}

static inline void normalization_validate(const DayanNormConfig * config) {
    if (config == nullptr || config->width == 0 || config->rows == 0 ||
        normalization_bits_float(config->epsilon_bits) < 0.0f || config->input_ne[0] != config->width ||
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
static float normalization_inverse_rms(const DayanNormConfig * config,
                                       const float *           left,
                                       const float *           right,
                                       float *                 residual,
                                       bool                    write_residual) {
    const uint64_t uram    = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input   = uram + NormalizationLayout::kInput;
    const uint64_t aux     = uram + NormalizationLayout::kAux;
    const uint64_t value   = uram + NormalizationLayout::kValue;
    const uint64_t square  = uram + NormalizationLayout::kSquare;
    float *        partial = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar);
    float          sum     = 0.0f;

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

        configure_normalization_macro<0>(elements);
        __tx_macro_elemwise_mul_fp32<0, 0>((void *) square, (void *) value, (void *) value);
        __tx_macro_reduce_add_fp32<0>(partial, (void *) square);
        dayan_wait_macro();
        sum += *partial;
    }

    float * mean    = partial;
    float * inverse = partial + 1;
    *mean           = sum / (float) config->width + normalization_bits_float(config->epsilon_bits);
    configure_normalization_macro<1>(1);
    __tx_macro_sfu_rsqrt_fp32<1>(inverse, mean);
    dayan_wait_macro();
    return *inverse;
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
        const float inverse =
            normalization_inverse_rms<HasResidual>(norm, left_row, right_row, residual_row, write_residual);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t input        = uram + NormalizationLayout::kInput;
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t aux          = uram + NormalizationLayout::kAux;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        const uint64_t scaled_uram  = uram + NormalizationLayout::kScaled;
        const uint64_t lowp_uram    = uram + NormalizationLayout::kLowp;
        float *        inverse_uram = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar + sizeof(float));
        *inverse_uram               = inverse;

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

            configure_normalization_macro<0>(elements);
            __tx_macro_elemwise_mul_fp32<0, 1>((void *) rms_uram, inverse_uram, (void *) value);
            __tx_macro_elemwise_mul_fp32<0, 0>((void *) scaled_uram, (void *) rms_uram, (void *) aux);
            dayan_wait_macro();
            if constexpr (Quantize) {
                normalization_quantize<Format>(lowp_uram, scaled_uram, elements);
                dayan_wait_vec();
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
        const float    inverse      = normalization_inverse_rms<false>(config, input_row, nullptr, nullptr, false);
        const uint64_t uram         = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t value        = uram + NormalizationLayout::kValue;
        const uint64_t rms_uram     = uram + NormalizationLayout::kRms;
        float *        inverse_uram = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar + sizeof(float));
        *inverse_uram               = inverse;

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset), (int) NormalizationLayout::kTileElements);
            normalization_load(value, input_row + offset, elements);
            configure_normalization_macro<0>(elements);
            __tx_macro_elemwise_mul_fp32<0, 1>((void *) rms_uram, inverse_uram, (void *) value);
            dayan_wait_macro();
            normalization_store_f32(output + row_offset + offset, rms_uram, elements);
            dayan_wait_lsu();
        }
    }
}

static inline void normalization_centered(const float * input, float * output, uint64_t elements, float epsilon) {
    const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input_uram = uram + NormalizationLayout::kInput;
    const uint64_t centered   = uram + NormalizationLayout::kValue;
    const uint64_t square     = uram + NormalizationLayout::kSquare;
    const uint64_t scaled     = uram + NormalizationLayout::kScaled;
    float *        scalar     = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar);
    float          sum        = 0.0f;

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        configure_normalization_macro<0>(count);
        __tx_macro_reduce_add_fp32<0>(scalar, (void *) input_uram);
        dayan_wait_macro();
        sum += scalar[0];
    }

    float * mean   = scalar + 1;
    *mean          = sum / (float) elements;
    float variance = 0.0f;
    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        configure_normalization_macro<0>(count);
        __tx_macro_elemwise_sub_fp32<0, 1>((void *) centered, mean, (void *) input_uram);
        __tx_macro_elemwise_mul_fp32<0, 0>((void *) square, (void *) centered, (void *) centered);
        __tx_macro_reduce_add_fp32<0>(scalar, (void *) square);
        dayan_wait_macro();
        variance += scalar[0];
    }

    float * variance_uram = scalar + 2;
    float * inverse       = scalar + 3;
    *variance_uram        = variance / (float) elements + epsilon;
    configure_normalization_macro<1>(1);
    __tx_macro_sfu_rsqrt_fp32<1>(inverse, variance_uram);
    dayan_wait_macro();
    *inverse = -*inverse;

    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count =
            (uint32_t) (elements - offset < NormalizationLayout::kTileElements ? elements - offset :
                                                                                 NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        configure_normalization_macro<0>(count);
        __tx_macro_elemwise_sub_fp32<0, 1>((void *) centered, mean, (void *) input_uram);
        __tx_macro_elemwise_mul_fp32<0, 1>((void *) scaled, inverse, (void *) centered);
        dayan_wait_macro();
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
    const float epsilon = normalization_bits_float(config->epsilon_bits);
    for (uint32_t row = dayan_native_task_id(); row < config->rows; row += dayan_native_task_count()) {
        const uint64_t offset = (uint64_t) row * config->width;
        normalization_centered(normalization_input_row(config, input, row), output + offset, config->width, epsilon);
    }
}

static inline void run_group_norm(const DayanGroupNormConfig * config, const float * input, float * output) {
    if (config == nullptr || config->width == 0 || config->height == 0 || config->channels == 0 ||
        config->batches == 0 || config->groups == 0 || config->groups > config->channels || input == nullptr ||
        output == nullptr || normalization_bits_float(config->epsilon_bits) < 0.0f) {
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
    const float    epsilon            = normalization_bits_float(config->epsilon_bits);
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
        normalization_centered(input + offset, output + offset, elements, epsilon);
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

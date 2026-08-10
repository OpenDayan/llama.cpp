#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

struct NormalizationLayout {
    enum : uint64_t {
        kTileElements = 256,
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

template <typename Format> static inline void normalization_quantize(uint64_t output, uint64_t input);

template <> inline void normalization_quantize<Bf16Format>(uint64_t output, uint64_t input) {
    __tx_macro_convert_fp32_bf16<0>((void *) output, (void *) input);
}

template <> inline void normalization_quantize<Fp16Format>(uint64_t output, uint64_t input) {
    __tx_macro_convert_fp32_fp16<0>((void *) output, (void *) input);
}

static inline void normalization_validate(const DayanNormConfig * config) {
    if (config == nullptr || config->width == 0 || config->rows == 0 ||
        normalization_bits_float(config->epsilon_bits) < 0.0f || dayan_task_mode() != 1U ||
        dayan_task_dim(DAYAN_TASK_X) != 1U || dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        dayan_proc_exit(-22);
    }
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }
}

template <bool HasResidual>
static float normalization_inverse_rms(const DayanNormConfig * config,
                                       const float *           left,
                                       const float *           right,
                                       float *                 residual) {
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
            normalization_store_f32(residual + offset, value, elements);
            dayan_wait_lsu();
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
    *mean = sum / (float) config->width + normalization_bits_float(config->epsilon_bits);
    configure_normalization_macro<1>(1);
    __tx_macro_sfu_rsqrt_fp32<1>(inverse, mean);
    dayan_wait_macro();
    return *inverse;
}

template <typename Format, bool HasResidual, bool Quantize>
static void run_rms_norm_scale(const DayanNormConfig * config,
                               const float *           left,
                               const float *           right,
                               const float *           scale,
                               float *                 residual,
                               float *                 rms,
                               float *                 scaled,
                               uint16_t *              lowp) {
    normalization_validate(config);
    if (left == nullptr || scale == nullptr || rms == nullptr || scaled == nullptr ||
        (HasResidual && (right == nullptr || residual == nullptr)) || (Quantize && lowp == nullptr)) {
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    for (uint32_t row = 0; row < config->rows; ++row) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        float *        residual_row = HasResidual ? residual + row_offset : nullptr;
        const float inverse = normalization_inverse_rms<HasResidual>(config, left + row_offset,
                                                                      HasResidual ? right + row_offset : nullptr,
                                                                      residual_row);
        const uint64_t uram       = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t value      = uram + NormalizationLayout::kValue;
        const uint64_t aux        = uram + NormalizationLayout::kAux;
        const uint64_t rms_uram   = uram + NormalizationLayout::kRms;
        const uint64_t scaled_uram = uram + NormalizationLayout::kScaled;
        const uint64_t lowp_uram  = uram + NormalizationLayout::kLowp;
        float *        inverse_uram = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar + sizeof(float));
        *inverse_uram = inverse;

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset),
                                              (int) NormalizationLayout::kTileElements);
            if constexpr (HasResidual) {
                normalization_load(value, residual_row + offset, elements);
            } else {
                normalization_load(value, left + row_offset + offset, elements);
            }
            normalization_load(aux, scale + offset, elements);

            configure_normalization_macro<0>(elements);
            __tx_macro_elemwise_mul_fp32<0, 1>((void *) rms_uram, inverse_uram, (void *) value);
            __tx_macro_elemwise_mul_fp32<0, 0>((void *) scaled_uram, (void *) rms_uram, (void *) aux);
            if constexpr (Quantize) {
                normalization_quantize<Format>(lowp_uram, scaled_uram);
            }
            dayan_wait_macro();

            normalization_store_f32(rms + row_offset + offset, rms_uram, elements);
            normalization_store_f32(scaled + row_offset + offset, scaled_uram, elements);
            if constexpr (Quantize) {
                normalization_store_lowp(lowp + row_offset + offset, lowp_uram, elements);
            }
            dayan_wait_lsu();
        }
    }
}

static inline void run_rms_norm(const DayanNormConfig * config, const float * input, float * output) {
    normalization_validate(config);
    if (input == nullptr || output == nullptr) {
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    for (uint32_t row = 0; row < config->rows; ++row) {
        const uint64_t row_offset = (uint64_t) row * config->width;
        const float    inverse = normalization_inverse_rms<false>(config, input + row_offset, nullptr, nullptr);
        const uint64_t uram     = (uint64_t) (uintptr_t) dayan_uram_base();
        const uint64_t value    = uram + NormalizationLayout::kValue;
        const uint64_t rms_uram = uram + NormalizationLayout::kRms;
        float * inverse_uram = (float *) (uintptr_t) (uram + NormalizationLayout::kScalar + sizeof(float));
        *inverse_uram = inverse;

        for (uint32_t offset = 0; offset < config->width; offset += NormalizationLayout::kTileElements) {
            const uint32_t elements = min_int((int) (config->width - offset),
                                              (int) NormalizationLayout::kTileElements);
            normalization_load(value, input + row_offset + offset, elements);
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
        const uint32_t count = (uint32_t) (elements - offset < NormalizationLayout::kTileElements ?
                                               elements - offset :
                                               NormalizationLayout::kTileElements);
        normalization_load(input_uram, input + offset, count);
        configure_normalization_macro<0>(count);
        __tx_macro_reduce_add_fp32<0>(scalar, (void *) input_uram);
        dayan_wait_macro();
        sum += scalar[0];
    }

    float * mean = scalar + 1;
    *mean        = sum / (float) elements;
    float variance = 0.0f;
    for (uint64_t offset = 0; offset < elements; offset += NormalizationLayout::kTileElements) {
        const uint32_t count = (uint32_t) (elements - offset < NormalizationLayout::kTileElements ?
                                               elements - offset :
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
        const uint32_t count = (uint32_t) (elements - offset < NormalizationLayout::kTileElements ?
                                               elements - offset :
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
        dayan_proc_exit(-22);
    }

    configure_normalization_lsu();
    const float epsilon = normalization_bits_float(config->epsilon_bits);
    for (uint32_t row = 0; row < config->rows; ++row) {
        const uint64_t offset = (uint64_t) row * config->width;
        normalization_centered(input + offset, output + offset, config->width, epsilon);
    }
}

static inline void run_group_norm(const DayanGroupNormConfig * config, const float * input, float * output) {
    if (config == nullptr || config->width == 0 || config->height == 0 || config->channels == 0 ||
        config->batches == 0 || config->groups == 0 || config->groups > config->channels || input == nullptr ||
        output == nullptr || normalization_bits_float(config->epsilon_bits) < 0.0f || dayan_task_mode() != 1U ||
        dayan_task_dim(DAYAN_TASK_X) != 1U || dayan_task_dim(DAYAN_TASK_Y) != 1U ||
        dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        dayan_proc_exit(-22);
    }
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }

    configure_normalization_lsu();
    const uint32_t channels_per_group = (config->channels + config->groups - 1U) / config->groups;
    const uint64_t channel_elements   = (uint64_t) config->width * config->height;
    const float    epsilon            = normalization_bits_float(config->epsilon_bits);
    for (uint32_t batch = 0; batch < config->batches; ++batch) {
        for (uint32_t group = 0; group < config->groups; ++group) {
            const uint32_t channel_start = group * channels_per_group;
            const uint32_t channel_end = channel_start + channels_per_group < config->channels ?
                                             channel_start + channels_per_group :
                                             config->channels;
            const uint64_t offset = ((uint64_t) batch * config->channels + channel_start) * channel_elements;
            const uint64_t elements = (uint64_t) (channel_end - channel_start) * channel_elements;
            normalization_centered(input + offset, output + offset, elements, epsilon);
        }
    }
}

static inline void run_residual_add(const DayanElementwiseConfig * config,
                                    const float *                  left,
                                    const float *                  right,
                                    float *                        output) {
    if (config == nullptr || config->elements == 0 || left == nullptr ||
        right == nullptr || output == nullptr || dayan_task_mode() != 1U || dayan_task_dim(DAYAN_TASK_X) != 1U ||
        dayan_task_dim(DAYAN_TASK_Y) != 1U || dayan_task_dim(DAYAN_TASK_Z) != 1U) {
        dayan_proc_exit(-22);
    }
    if (dayan_uram_base() == nullptr || dayan_uram_size() < NormalizationLayout::kTotalBytes) {
        dayan_proc_exit(-12);
    }

    configure_normalization_lsu();
    const uint64_t uram  = (uint64_t) (uintptr_t) dayan_uram_base();
    const uint64_t input = uram + NormalizationLayout::kInput;
    const uint64_t aux   = uram + NormalizationLayout::kAux;
    const uint64_t value = uram + NormalizationLayout::kValue;
    for (uint64_t offset = 0; offset < config->elements; offset += NormalizationLayout::kTileElements) {
        const uint64_t remaining = config->elements - offset;
        const uint32_t elements = remaining < NormalizationLayout::kTileElements ?
                                      (uint32_t) remaining :
                                      (uint32_t) NormalizationLayout::kTileElements;
        normalization_load(input, left + offset, elements);
        normalization_load(aux, right + offset, elements);
        normalization_vec_add(input, aux, value, elements);
        normalization_store_f32(output + offset, value, elements);
        dayan_wait_lsu();
    }
}

template <typename Format>
static void run_rms_norm_matmul(const DayanNormMatmulConfig * config,
                                const float *                 input,
                                const float *                 scale,
                                float *                       rms,
                                float *                       scaled,
                                uint16_t *                    lowp,
                                const uint16_t *              weight,
                                float *                       output) {
    if (config == nullptr || config->norm.rows != config->matmul.m || config->norm.width != config->matmul.k) {
        dayan_proc_exit(-22);
    }

    run_rms_norm_scale<Format, false, true>(&config->norm, input, nullptr, scale, nullptr, rms, scaled, lowp);
    DayanMatmulPerf perf;
    run_matmul_kernel<Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity>(&config->matmul, lowp, weight, nullptr,
                                                                               output, &perf);
}

template <typename Format>
static void run_add_rms_norm_matmul(const DayanNormMatmulConfig * config,
                                    const float *                 left,
                                    const float *                 right,
                                    const float *                 scale,
                                    void *                        intermediates,
                                    const uint16_t *              weight,
                                    float *                       output) {
    if (config == nullptr || config->norm.rows != config->matmul.m || config->norm.width != config->matmul.k ||
        intermediates == nullptr) {
        dayan_proc_exit(-22);
    }

    const uint64_t elements = (uint64_t) config->norm.width * config->norm.rows;
    float *        residual = (float *) intermediates;
    float *        rms      = residual + elements;
    float *        scaled   = rms + elements;
    uint16_t *     lowp     = (uint16_t *) (scaled + elements);
    run_rms_norm_scale<Format, true, true>(&config->norm, left, right, scale, residual, rms, scaled, lowp);
    DayanMatmulPerf perf;
    run_matmul_kernel<Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity>(&config->matmul, lowp, weight, nullptr,
                                                                               output, &perf);
}

}  // namespace ggml_dayan_matmul_detail

#endif

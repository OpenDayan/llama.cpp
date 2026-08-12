#pragma once

#include "ggml-dayan-matmul.h"

#if defined(__DAYAN_DEVICE__)

#    include <dayan_sys.h>
#    include <npu_cache.h>

extern "C" float sqrtf(float value);

namespace ggml_dayan_matmul_detail {

static inline uint32_t scalar_float_bits(float value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.f = value;
    return bits.u;
}

static inline float scalar_bits_float(uint32_t value) {
    union {
        float    f;
        uint32_t u;
    } bits;

    bits.u = value;
    return bits.f;
}

static inline void scalar_sincos(float angle, float * sine, float * cosine) {
    constexpr float pi         = 3.14159265358979323846f;
    constexpr float half_pi    = 1.57079632679489661923f;
    constexpr float two_pi     = 6.28318530717958647692f;
    constexpr float inv_two_pi = 0.15915494309189533577f;
    int             turns      = (int) (angle * inv_two_pi);
    angle -= (float) turns * two_pi;
    if (angle > pi) {
        angle -= two_pi;
    } else if (angle < -pi) {
        angle += two_pi;
    }

    float cosine_sign = 1.0f;
    if (angle > half_pi) {
        angle       = pi - angle;
        cosine_sign = -1.0f;
    } else if (angle < -half_pi) {
        angle       = -pi - angle;
        cosine_sign = -1.0f;
    }

    const float square = angle * angle;
    *sine =
        angle * (1.0f + square * (-0.16666666666666666f +
                                  square * (0.008333333333333333f +
                                            square * (-0.0001984126984126984f + square * 0.0000027557319223985893f))));
    *cosine = cosine_sign *
              (1.0f + square * (-0.5f + square * (0.041666666666666664f + square * (-0.001388888888888889f +
                                                                                    square * 0.0000248015873015873f))));
}

template <typename Format> struct ScalarLowp;

template <> struct ScalarLowp<Bf16Format> {
    static inline float decode(uint16_t value) { return scalar_bits_float((uint32_t) value << 16); }

    static inline uint16_t encode(float value) {
        uint32_t bits = scalar_float_bits(value);
        bits += 0x7fffU + ((bits >> 16) & 1U);
        return (uint16_t) (bits >> 16);
    }
};

template <> struct ScalarLowp<Fp16Format> {
    static inline float decode(uint16_t value) {
        const uint32_t sign     = (uint32_t) (value & 0x8000U) << 16;
        uint32_t       exponent = (value >> 10) & 0x1fU;
        uint32_t       mantissa = value & 0x03ffU;

        if (exponent == 0) {
            if (mantissa == 0) {
                return scalar_bits_float(sign);
            }
            int shift = 0;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x03ffU;
            exponent = (uint32_t) (127 - 15 - shift + 1);
        } else if (exponent == 0x1fU) {
            exponent = 0xffU;
        } else {
            exponent += 127 - 15;
        }
        return scalar_bits_float(sign | (exponent << 23) | (mantissa << 13));
    }

    static inline uint16_t encode(float value) {
        const uint32_t bits     = scalar_float_bits(value);
        const uint16_t sign     = (uint16_t) ((bits >> 16) & 0x8000U);
        const uint32_t exponent = (bits >> 23) & 0xffU;
        const uint32_t mantissa = bits & 0x7fffffU;

        if (exponent == 0xffU) {
            return (uint16_t) (sign | (mantissa ? 0x7e00U : 0x7c00U));
        }

        const int half_exponent = (int) exponent - 127 + 15;
        if (half_exponent >= 0x1f) {
            return (uint16_t) (sign | 0x7c00U);
        }
        if (half_exponent <= 0) {
            if (half_exponent < -10) {
                return sign;
            }
            const uint32_t normalized = mantissa | 0x800000U;
            const int      shift      = 14 - half_exponent;
            uint32_t       rounded    = normalized >> shift;
            const uint32_t remainder  = normalized & ((1U << shift) - 1U);
            const uint32_t halfway    = 1U << (shift - 1);
            if (remainder > halfway || (remainder == halfway && (rounded & 1U))) {
                ++rounded;
            }
            return (uint16_t) (sign | rounded);
        }

        uint32_t rounded         = mantissa + 0x0fffU + ((mantissa >> 13) & 1U);
        uint32_t result_exponent = (uint32_t) half_exponent;
        if (rounded & 0x800000U) {
            rounded = 0;
            ++result_exponent;
            if (result_exponent >= 0x1fU) {
                return (uint16_t) (sign | 0x7c00U);
            }
        }
        return (uint16_t) (sign | (result_exponent << 10) | (rounded >> 13));
    }
};

static inline bool attention_projection_config_valid(const DayanAttentionProjectionConfig * config, bool has_gate) {
    if (config == nullptr || config->tokens == 0 || config->heads == 0 || config->head_dim == 0 || config->n_rot == 0 ||
        config->n_rot > config->head_dim || (config->n_rot & 1U) != 0 ||
        (config->rope_mode != 0U && config->rope_mode != 2U && config->rope_mode != 8U)) {
        return false;
    }
    const uint64_t projection_width = (uint64_t) config->heads * config->head_dim * (has_gate ? 2U : 1U);
    return config->matmul.m == config->tokens && config->matmul.n == projection_width &&
           config->matmul.output_stride >= projection_width * sizeof(uint16_t);
}

template <typename Format>
static inline void run_projection_matmul(const DayanMatmulConfig * config,
                                         const uint16_t *          activation,
                                         const uint16_t *          weight,
                                         uint16_t *                output,
                                         DayanMatmulPerf *         perf) {
    run_matmul_kernel<Format, 1, sizeof(uint16_t), true, epilogue_identity_bf16<0>, epilogue_identity_bf16<4>,
                      epilogue_identity_bf16<2>, epilogue_identity_bf16<0>, epilogue_identity_bf16<4>,
                      epilogue_identity_bf16<2>>(config, activation, weight, nullptr, output, perf);
}

template <>
inline void run_projection_matmul<Fp16Format>(const DayanMatmulConfig * config,
                                              const uint16_t *          activation,
                                              const uint16_t *          weight,
                                              uint16_t *                output,
                                              DayanMatmulPerf *         perf) {
    run_matmul_kernel<Fp16Format, 1, sizeof(uint16_t), true, epilogue_identity_fp16<0>, epilogue_identity_fp16<4>,
                      epilogue_identity_fp16<2>, epilogue_identity_fp16<0>, epilogue_identity_fp16<4>,
                      epilogue_identity_fp16<2>>(config, activation, weight, nullptr, output, perf);
}

static inline int32_t rope_position(const DayanAttentionProjectionConfig * config,
                                    const int32_t *                        positions,
                                    uint32_t                               token,
                                    uint32_t                               pair) {
    if (config->rope_mode != 8U) {
        return positions[token];
    }

    const int32_t section_count = config->sections[0] + config->sections[1] + config->sections[2] + config->sections[3];
    if (section_count <= 0) {
        return positions[token];
    }

    const int32_t sector = (int32_t) pair % section_count;
    int32_t       axis   = 0;
    if (sector >= config->sections[0]) {
        axis = 1;
    }
    if (sector >= config->sections[0] + config->sections[1]) {
        axis = 2;
    }
    if (sector >= config->sections[0] + config->sections[1] + config->sections[2]) {
        axis = 3;
    }
    return positions[token + (uint32_t) axis * config->tokens];
}

template <typename Format>
static inline void postprocess_attention_head(const DayanAttentionProjectionConfig * config,
                                              const float *                          norm,
                                              const int32_t *                        positions,
                                              uint32_t                               token,
                                              const uint16_t *                       source,
                                              uint16_t *                             destination) {
    const uint32_t head_dim    = config->head_dim;
    const float    epsilon     = scalar_bits_float(config->norm_eps_bits);
    const float    theta_scale = scalar_bits_float(config->theta_scale_bits);
    const float    freq_scale  = scalar_bits_float(config->freq_scale_bits);
    const float    attn_factor = scalar_bits_float(config->attn_factor_bits);
    float          sum_squares = 0.0f;
    for (uint32_t dim = 0; dim < head_dim; ++dim) {
        const float value = ScalarLowp<Format>::decode(source[dim]);
        sum_squares += value * value;
    }
    const float inverse_rms = 1.0f / sqrtf(sum_squares / (float) head_dim + epsilon);

    if (config->rope_mode == 0U) {
        float theta_factor = 1.0f;
        for (uint32_t dim = 0; dim < config->n_rot; dim += 2) {
            const uint32_t pair     = dim / 2;
            const float    position = (float) rope_position(config, positions, token, pair);
            const float    theta    = position * theta_factor * freq_scale;
            float          sine;
            float          cosine;
            scalar_sincos(theta, &sine, &cosine);
            sine *= attn_factor;
            cosine *= attn_factor;
            const float x0       = ScalarLowp<Format>::decode(source[dim]) * norm[dim] * inverse_rms;
            const float x1       = ScalarLowp<Format>::decode(source[dim + 1]) * norm[dim + 1] * inverse_rms;
            destination[dim]     = ScalarLowp<Format>::encode(x0 * cosine - x1 * sine);
            destination[dim + 1] = ScalarLowp<Format>::encode(x0 * sine + x1 * cosine);
            theta_factor *= theta_scale;
        }
    } else {
        const uint32_t half         = config->n_rot / 2;
        float          theta_factor = 1.0f;
        for (uint32_t pair = 0; pair < half; ++pair) {
            const float position = (float) rope_position(config, positions, token, pair);
            const float theta    = position * theta_factor * freq_scale;
            float       sine;
            float       cosine;
            scalar_sincos(theta, &sine, &cosine);
            sine *= attn_factor;
            cosine *= attn_factor;
            const float x0    = ScalarLowp<Format>::decode(source[pair]) * norm[pair] * inverse_rms;
            const float x1    = ScalarLowp<Format>::decode(source[pair + half]) * norm[pair + half] * inverse_rms;
            destination[pair] = ScalarLowp<Format>::encode(x0 * cosine - x1 * sine);
            destination[pair + half] = ScalarLowp<Format>::encode(x0 * sine + x1 * cosine);
            theta_factor *= theta_scale;
        }
    }

    for (uint32_t dim = config->n_rot; dim < head_dim; ++dim) {
        const float value = ScalarLowp<Format>::decode(source[dim]) * norm[dim] * inverse_rms;
        destination[dim]  = ScalarLowp<Format>::encode(value);
    }
}

template <typename Format, bool HasGate>
static inline void postprocess_attention_projection(const DayanAttentionProjectionConfig * config,
                                                    const float *                          norm,
                                                    const int32_t *                        positions,
                                                    const uint16_t *                       raw_output,
                                                    uint16_t *                             output) {
    const uint32_t head_dim        = config->head_dim;
    const uint32_t projection_step = head_dim * (HasGate ? 2U : 1U);

    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t head = 0; head < config->heads; ++head) {
            const uint16_t * source = raw_output + (uint64_t) token * config->matmul.output_stride / sizeof(uint16_t) +
                                      head * projection_step;
            uint16_t *       destination = output + ((uint64_t) head * config->tokens + token) * head_dim;
            postprocess_attention_head<Format>(config, norm, positions, token, source, destination);
        }
    }
    cache_bypass_flush();
}

template <typename Format, bool HasGate>
static inline void run_attention_projection(const DayanAttentionProjectionConfig * config,
                                            const uint16_t *                       activation,
                                            const uint16_t *                       weight,
                                            const float *                          norm,
                                            const int32_t *                        positions,
                                            uint16_t *                             raw_output,
                                            uint16_t *                             output,
                                            DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, HasGate) || activation == nullptr || weight == nullptr ||
        norm == nullptr || positions == nullptr || raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    run_projection_matmul<Format>(&config->matmul, activation, weight, raw_output, perf);
    cache_inval_range(raw_output, (size_t) config->tokens * config->matmul.n * sizeof(uint16_t));
    postprocess_attention_projection<Format, HasGate>(config, norm, positions, raw_output, output);
}

template <>
inline void run_attention_projection<Fp16Format, true>(const DayanAttentionProjectionConfig * config,
                                                       const uint16_t *                       activation,
                                                       const uint16_t *                       weight,
                                                       const float *                          norm,
                                                       const int32_t *                        positions,
                                                       uint16_t *                             raw_output,
                                                       uint16_t *                             output,
                                                       DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, true) || activation == nullptr || weight == nullptr ||
        norm == nullptr || positions == nullptr || raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    run_projection_matmul<Fp16Format>(&config->matmul, activation, weight, raw_output, perf);
    cache_inval_range(raw_output, (size_t) config->tokens * config->matmul.n * sizeof(uint16_t));
    postprocess_attention_projection<Fp16Format, true>(config, norm, positions, raw_output, output);
}

template <>
inline void run_attention_projection<Fp16Format, false>(const DayanAttentionProjectionConfig * config,
                                                        const uint16_t *                       activation,
                                                        const uint16_t *                       weight,
                                                        const float *                          norm,
                                                        const int32_t *                        positions,
                                                        uint16_t *                             raw_output,
                                                        uint16_t *                             output,
                                                        DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, false) || activation == nullptr || weight == nullptr ||
        norm == nullptr || positions == nullptr || raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    run_projection_matmul<Fp16Format>(&config->matmul, activation, weight, raw_output, perf);
    cache_inval_range(raw_output, (size_t) config->tokens * config->matmul.n * sizeof(uint16_t));
    postprocess_attention_projection<Fp16Format, false>(config, norm, positions, raw_output, output);
}

template <typename Format>
static inline void run_value_projection(const DayanAttentionProjectionConfig * config,
                                        const uint16_t *                       activation,
                                        const uint16_t *                       weight,
                                        uint16_t *                             raw_output,
                                        uint16_t *                             output,
                                        DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, false) || activation == nullptr || weight == nullptr ||
        raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    run_projection_matmul<Format>(&config->matmul, activation, weight, raw_output, perf);
    cache_inval_range(raw_output, (size_t) config->tokens * config->matmul.n * sizeof(uint16_t));
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t head = 0; head < config->heads; ++head) {
            for (uint32_t dim = 0; dim < config->head_dim; ++dim) {
                const uint64_t source =
                    (uint64_t) token * config->matmul.output_stride / sizeof(uint16_t) + head * config->head_dim + dim;
                const uint64_t target = ((uint64_t) head * config->tokens + token) * config->head_dim + dim;
                output[target]        = raw_output[source];
            }
        }
    }
    cache_bypass_flush();
}

template <>
inline void run_value_projection<Fp16Format>(const DayanAttentionProjectionConfig * config,
                                             const uint16_t *                       activation,
                                             const uint16_t *                       weight,
                                             uint16_t *                             raw_output,
                                             uint16_t *                             output,
                                             DayanMatmulPerf *                      perf) {
    if (!attention_projection_config_valid(config, false) || activation == nullptr || weight == nullptr ||
        raw_output == nullptr || output == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }
    run_projection_matmul<Fp16Format>(&config->matmul, activation, weight, raw_output, perf);
    cache_inval_range(raw_output, (size_t) config->tokens * config->matmul.n * sizeof(uint16_t));
    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t head = 0; head < config->heads; ++head) {
            for (uint32_t dim = 0; dim < config->head_dim; ++dim) {
                const uint64_t source =
                    (uint64_t) token * config->matmul.output_stride / sizeof(uint16_t) + head * config->head_dim + dim;
                const uint64_t target = ((uint64_t) head * config->tokens + token) * config->head_dim + dim;
                output[target]        = raw_output[source];
            }
        }
    }
    cache_bypass_flush();
}

static inline bool kv_projection_config_valid(const DayanKVProjectionConfig * config) {
    return config != nullptr && config->cache_rows > 0 && attention_projection_config_valid(&config->projection, false);
}

template <typename Format>
static inline void run_k_cache_projection(const DayanKVProjectionConfig * config,
                                          const uint16_t *                activation,
                                          const uint16_t *                weight,
                                          const float *                   norm,
                                          const int32_t *                 positions_indices,
                                          uint16_t *                      workspace,
                                          uint16_t *                      cache,
                                          DayanMatmulPerf *               perf) {
    if (!kv_projection_config_valid(config) || activation == nullptr || weight == nullptr || norm == nullptr ||
        positions_indices == nullptr || workspace == nullptr || cache == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanAttentionProjectionConfig * projection    = &config->projection;
    const uint32_t                         position_axes = projection->rope_mode == 8U ? 4U : 1U;
    const int32_t *                        indices = positions_indices + (uint64_t) position_axes * projection->tokens;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        if (indices[token] < 0 || (uint32_t) indices[token] >= config->cache_rows) {
            DAYAN_PARAMS_ERROR();
        }
    }

    run_projection_matmul<Format>(&projection->matmul, activation, weight, workspace, perf);
    cache_inval_range(workspace, (size_t) projection->tokens * projection->matmul.output_stride);
    cache_inval_range(cache, (size_t) config->cache_rows * projection->heads * projection->head_dim * sizeof(uint16_t));
    const uint32_t cache_width = projection->heads * projection->head_dim;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        for (uint32_t head = 0; head < projection->heads; ++head) {
            const uint16_t * source = workspace +
                                      (uint64_t) token * projection->matmul.output_stride / sizeof(uint16_t) +
                                      head * projection->head_dim;
            uint16_t * destination  = cache + (uint64_t) indices[token] * cache_width + head * projection->head_dim;
            postprocess_attention_head<Format>(projection, norm, positions_indices, token, source, destination);
        }
    }
    cache_bypass_flush();
}

template <typename Format>
static inline void run_v_cache_projection(const DayanKVProjectionConfig * config,
                                          const uint16_t *                activation,
                                          const uint16_t *                weight,
                                          const int32_t *                 indices,
                                          uint16_t *                      workspace,
                                          uint16_t *                      cache,
                                          DayanMatmulPerf *               perf) {
    if (!kv_projection_config_valid(config) || activation == nullptr || weight == nullptr || indices == nullptr ||
        workspace == nullptr || cache == nullptr || perf == nullptr) {
        DAYAN_PARAMS_ERROR();
    }

    const DayanAttentionProjectionConfig * projection = &config->projection;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        if (indices[token] < 0 || (uint32_t) indices[token] >= config->cache_rows) {
            DAYAN_PARAMS_ERROR();
        }
    }

    run_projection_matmul<Format>(&projection->matmul, activation, weight, workspace, perf);
    cache_inval_range(workspace, (size_t) projection->tokens * projection->matmul.output_stride);
    cache_inval_range(cache, (size_t) config->cache_rows * projection->heads * projection->head_dim * sizeof(uint16_t));
    const uint32_t cache_width = projection->heads * projection->head_dim;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        for (uint32_t dim = 0; dim < cache_width; ++dim) {
            const uint64_t source = (uint64_t) token * projection->matmul.output_stride / sizeof(uint16_t) + dim;
            cache[(uint64_t) indices[token] * cache_width + dim] = workspace[source];
        }
    }
    cache_bypass_flush();
}

template <long Group>
static inline void configure_gated_lsu(long block,
                                       long dim0,
                                       long dim1,
                                       long input_block_stride,
                                       long input_dim0_stride,
                                       long input_dim1_stride,
                                       long output_block_stride,
                                       long output_dim0_stride,
                                       long output_dim1_stride,
                                       long transpose,
                                       long read_size,
                                       long write_size) {
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCK>(block);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM0>(dim0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM1>(dim1);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_DIM2>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INBLOCKSTRIDE>(input_block_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM0STRIDE>(input_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM1STRIDE>(input_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_INDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTBLOCKSTRIDE>(output_block_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM0STRIDE>(output_dim0_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM1STRIDE>(output_dim1_stride);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKTRANSPOSE>(transpose);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_BLOCKPADDING>(block);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTREADSIZE>(read_size);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_PADDINGVALUE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMWRITESWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_URAMREADSWIZZLEMBS>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_REQUESTWRITESIZE>(write_size);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESIZE>(0);
    __tx_lsu_tcsr_write<Group * LSU_GROUPSZ + LSU_TABLESTRIDE>(64);
}

template <typename Format> static inline void issue_gated_lowp_to_f32(uint64_t input, long rows, long chunks) {
    constexpr long instructions = 3;
    constexpr long row_stride   = Constants::kTileK * sizeof(float);

    ggml_dayan_vec_detail::configure_vec_grid(chunks, rows);
    configure_vec_iterator<kVecV0Base>(0, 0, Constants::kVecBytes, row_stride);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, Constants::kVecBytes, row_stride);
    begin_vec_block(instructions);
    __tx_vec_lsu_u2v<0, 0>(input);
    if constexpr (Format::kMatDataType == BaseNPUInstBuilder::DT_BF16) {
        __tx_vec_cvt_fp32_bf16<0, 0, 0>();
    } else {
        __tx_vec_cvt_fp32_fp16<0, 0, 0>();
    }
    __tx_vec_lsu_v2u<3, 0>(input);
    start_vec_block(instructions);
}

template <typename Format>
static inline void issue_gated_f32_to_lowp(uint64_t input, uint64_t output, long rows, long cols) {
    ggml_dayan_vec_detail::issue_f32_to_lowp<Format::kMatDataType>(
        { input, Constants::kTileK * (int) sizeof(float) },
        { output, Constants::kTileK * (int) sizeof(uint16_t) }, rows, cols);
}

struct GatedInputContext {
    const DayanGatedProjectionConfig * config;
    const float *                      attention;
    const uint16_t *                   gate;
};

enum : uint64_t {
    kGatedInputMatrixBytes  = Constants::kTileM * Constants::kTileK * sizeof(float),
    kGatedInputScratchBytes = 2 * kGatedInputMatrixBytes + Constants::kConstantStride,
};

template <typename Format> static int load_gated_input(const MatmulInputArgs * args) {
    const GatedInputContext *          context        = (const GatedInputContext *) args->context;
    const DayanGatedProjectionConfig * config         = context->config;
    const uint64_t                     attention_uram = args->scratch;
    const uint64_t                     gate_uram      = attention_uram + kGatedInputMatrixBytes;
    const uint64_t                     constants      = gate_uram + kGatedInputMatrixBytes;
    const uint32_t                     width          = config->heads * config->head_dim;
    const uint32_t                     heads          = args->cols / config->head_dim;
    const uint32_t                     first_head     = args->k_base / config->head_dim;
    const uint64_t                     attention_base = (uint64_t) args->row_base * width + args->k_base;
    const uint64_t                     gate_base =
        (uint64_t) args->row_base * 2U * width + first_head * 2U * config->head_dim + config->head_dim;

    *(volatile float *) (uintptr_t) constants                   = -1.4426950408889634f;
    *(volatile float *) (uintptr_t) (constants + sizeof(float)) = 1.0f;

    long checksum = 0;
    set_lsu_matrix<Constants::kLsuShared>(checksum, Constants::kTileM, Constants::kTileK * (int) sizeof(float),
                                          width * sizeof(float), Constants::kTileK * (int) sizeof(float));
    load_matrix<Constants::kLsuShared>((uint64_t) args->cols * sizeof(float), args->rows,
                                       Constants::kTileK * sizeof(float), Constants::kTileM, attention_uram,
                                       (uint64_t) (uintptr_t) (context->attention + attention_base));
    configure_gated_lsu<Constants::kLsuShared>(
        0, config->head_dim - 1, args->rows - 1, sizeof(uint16_t), sizeof(uint16_t), 2U * width * sizeof(uint16_t),
        sizeof(float), sizeof(float), Constants::kTileK * sizeof(float), 6, sizeof(uint16_t) - 1, sizeof(float) - 1);
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_DIM2>(heads - 1);
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_INDIM2STRIDE>(2U * config->head_dim *
                                                                                sizeof(uint16_t));
    __tx_lsu_tcsr_write<Constants::kLsuShared * LSU_GROUPSZ + LSU_OUTDIM2STRIDE>(config->head_dim * sizeof(float));
    __tx_lsu_dma_d2u<Constants::kLsuShared>(gate_uram, reinterpret_cast<uint64_t>(context->gate + gate_base));
    dayan_wait_lsu();

    const long chunks = args->cols / Constants::kVecLanes;
    issue_gated_lowp_to_f32<Format>(gate_uram, args->rows, chunks);
    dayan_wait_vec();
    ggml_dayan_vec_detail::issue_sigmoid_mul({ gate_uram, Constants::kTileK * (int) sizeof(float) },
                                             { attention_uram, Constants::kTileK * (int) sizeof(float) },
                                             { attention_uram, Constants::kTileK * (int) sizeof(float) },
                                             { constants, constants + sizeof(float), 0, 0, 0, 0, 0 }, args->rows,
                                             args->cols);
    dayan_wait_vec();
    issue_gated_f32_to_lowp<Format>(attention_uram, args->output, args->rows, args->cols);
    dayan_wait_vec();
    return 0;
}

template <typename Format>
static inline void run_gated_output_projection(const DayanGatedProjectionConfig * config,
                                               const float *                      attention,
                                               const uint16_t *                   gate,
                                               const uint16_t *                   weight,
                                               float *                            output,
                                               DayanMatmulPerf *                  perf) {
    DAYAN_PARAMS_IF(config == nullptr || attention == nullptr || gate == nullptr || weight == nullptr ||
                    output == nullptr || perf == nullptr || config->tokens == 0 || config->heads == 0 ||
                    config->head_dim == 0 || config->matmul.m != config->tokens ||
                    config->matmul.k != (uint64_t) config->heads * config->head_dim ||
                    config->head_dim > Constants::kTileK || config->head_dim % Constants::kVecLanes != 0 ||
                    Constants::kTileK % config->head_dim != 0 || config->matmul.m >= config->matmul.n) {
        DAYAN_PARAMS_ERROR();
    }

    cache_inval_range(attention, (size_t) config->tokens * config->matmul.k * sizeof(float));
    cache_inval_range(gate, (size_t) config->tokens * 2U * config->matmul.k * sizeof(uint16_t));
    const GatedInputContext input_context = { config, attention, gate };
    run_matmul_kernel<Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity, load_gated_input<Format>,
                      kGatedInputScratchBytes>(&config->matmul, (const uint16_t *) attention, weight, nullptr, output,
                                               perf, &input_context);
}

}  // namespace ggml_dayan_matmul_detail

#endif

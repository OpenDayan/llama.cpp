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
        dayan_proc_exit(-22);
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
        dayan_proc_exit(-22);
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
        dayan_proc_exit(-22);
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
        dayan_proc_exit(-22);
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
        dayan_proc_exit(-22);
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
        dayan_proc_exit(-22);
    }

    const DayanAttentionProjectionConfig * projection    = &config->projection;
    const uint32_t                         position_axes = projection->rope_mode == 8U ? 4U : 1U;
    const int32_t *                        indices = positions_indices + (uint64_t) position_axes * projection->tokens;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        if (indices[token] < 0 || (uint32_t) indices[token] >= config->cache_rows) {
            dayan_proc_exit(-22);
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
        dayan_proc_exit(-22);
    }

    const DayanAttentionProjectionConfig * projection = &config->projection;
    for (uint32_t token = 0; token < projection->tokens; ++token) {
        if (indices[token] < 0 || (uint32_t) indices[token] >= config->cache_rows) {
            dayan_proc_exit(-22);
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

static inline void issue_gated_attention(uint64_t attention, uint64_t gate, uint64_t output, uint64_t constants) {
    enum : long {
        kInstructions = 10,
    };

    configure_vec_dims(1, 1);
    configure_vec_iterator<kVecV0Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV1Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV2Base>(0, 0, 0, 0);
    configure_vec_iterator<kVecV3Base>(0, 0, 0, 0);
    begin_vec_block(kInstructions);
    __tx_vec_lsu_u2v<0, 0>(gate);
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 1, 0>();
    __tx_vec_lsu_u2v_broadcast32<1, 2>(constants + sizeof(float));
    __tx_vec_vv_fp32_add<0, 16, 0, 0>();
    __tx_vec_v_fp32_sfu<0, 5, 0>();
    __tx_vec_lsu_u2v<1, 0>(attention);
    __tx_vec_vv_fp32_mul<0, 16, 0, 0>();
    __tx_vec_lsu_v2u<0, 0>(output);
    start_vec_block(kInstructions);
}

template <typename Format>
static inline void prepare_gated_attention(const DayanGatedProjectionConfig * config,
                                           const float *                      attention,
                                           const uint16_t *                   gate,
                                           uint16_t *                         activation) {
    float * uram           = (float *) dayan_uram_base();
    float * attention_uram = uram;
    float * gate_uram      = uram + Constants::kVecLanes;
    float * output_uram    = uram + 2 * Constants::kVecLanes;
    float * constants      = uram + 3 * Constants::kVecLanes;
    constants[0]           = -1.4426950408889634f;
    constants[1]           = 1.0f;

    const uint32_t width      = config->heads * config->head_dim;
    const uint32_t gate_width = 2 * width;
    cache_inval_range(attention, (size_t) config->tokens * width * sizeof(float));
    cache_inval_range(gate, (size_t) config->tokens * gate_width * sizeof(uint16_t));

    for (uint32_t token = 0; token < config->tokens; ++token) {
        for (uint32_t head = 0; head < config->heads; ++head) {
            for (uint32_t chunk = 0; chunk < config->head_dim; chunk += Constants::kVecLanes) {
                const uint32_t count =
                    config->head_dim - chunk < Constants::kVecLanes ? config->head_dim - chunk : Constants::kVecLanes;
                const uint64_t attention_base = (uint64_t) token * width + head * config->head_dim + chunk;
                const uint64_t gate_base =
                    (uint64_t) token * gate_width + head * 2U * config->head_dim + config->head_dim + chunk;
                for (uint32_t lane = 0; lane < Constants::kVecLanes; ++lane) {
                    attention_uram[lane] = lane < count ? attention[attention_base + lane] : 0.0f;
                    gate_uram[lane]      = lane < count ? ScalarLowp<Format>::decode(gate[gate_base + lane]) : -32.0f;
                }
                issue_gated_attention((uint64_t) attention_uram, (uint64_t) gate_uram, (uint64_t) output_uram,
                                      (uint64_t) constants);
                dayan_wait_vec();
                for (uint32_t lane = 0; lane < count; ++lane) {
                    activation[attention_base + lane] = ScalarLowp<Format>::encode(output_uram[lane]);
                }
            }
        }
    }
    cache_bypass_flush();
}

template <typename Format>
static inline void run_gated_output_projection(const DayanGatedProjectionConfig * config,
                                               const float *                      attention,
                                               const uint16_t *                   gate,
                                               uint16_t *                         activation,
                                               const uint16_t *                   weight,
                                               float *                            output,
                                               DayanMatmulPerf *                  perf) {
    if (config == nullptr || attention == nullptr || gate == nullptr || activation == nullptr || weight == nullptr ||
        output == nullptr || perf == nullptr || config->tokens == 0 || config->heads == 0 || config->head_dim == 0 ||
        config->matmul.m != config->tokens || config->matmul.k != (uint64_t) config->heads * config->head_dim) {
        dayan_proc_exit(-22);
    }

    prepare_gated_attention<Format>(config, attention, gate, activation);
    run_matmul_kernel<Format, 1, sizeof(float), false, epilogue_identity, epilogue_identity, epilogue_identity,
                      epilogue_identity, epilogue_identity, epilogue_identity>(&config->matmul, activation, weight,
                                                                               nullptr, output, perf);
}

}  // namespace ggml_dayan_matmul_detail

#endif

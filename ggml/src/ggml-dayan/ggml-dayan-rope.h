#pragma once

#include "ggml-dayan-attention.h"

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_matmul_detail {

static inline uint32_t rope_axis(const DayanRopeConfig * config, uint32_t pair) {
    const int32_t section_count = config->sections[0] + config->sections[1] + config->sections[2] + config->sections[3];
    if (section_count <= 0) {
        return 0;
    }

    const int32_t sector = (int32_t) pair % section_count;
    if (config->mode == 40U) {
        if (sector % 3 == 1 && sector < 3 * config->sections[1]) {
            return 1;
        }
        if (sector % 3 == 2 && sector < 3 * config->sections[2]) {
            return 2;
        }
        if (sector % 3 == 0 && sector < 3 * config->sections[0]) {
            return 0;
        }
        return 3;
    }
    if (sector >= config->sections[0] + config->sections[1] + config->sections[2]) {
        return 3;
    }
    if (sector >= config->sections[0] + config->sections[1]) {
        return 2;
    }
    if (sector >= config->sections[0]) {
        return 1;
    }
    return 0;
}

static inline void rope_sincos(const DayanRopeConfig * config,
                               const int32_t *         positions,
                               const float *           freq_factors,
                               uint32_t                token,
                               uint32_t                pair,
                               float                   theta_factor,
                               float *                 sine,
                               float *                 cosine) {
    const uint32_t axis         = (config->mode & 8U) != 0U ? rope_axis(config, pair) : 0U;
    const float    position     = (float) positions[token + (uint64_t) axis * config->tokens];
    const float    theta_extrap = position * theta_factor / freq_factors[pair];
    const float    freq_scale   = scalar_bits_float(config->freq_scale_bits);
    float          theta        = theta_extrap * freq_scale;
    const float    ext_factor   = scalar_bits_float(config->ext_factor_bits);
    if (ext_factor != 0.0f) {
        const float low   = scalar_bits_float(config->corr_dim0_bits);
        const float high  = scalar_bits_float(config->corr_dim1_bits);
        const float range = high - low > 0.001f ? high - low : 0.001f;
        float       ramp  = ((float) pair - low) / range;
        ramp              = ramp < 0.0f ? 0.0f : (ramp > 1.0f ? 1.0f : ramp);
        const float mix   = (1.0f - ramp) * ext_factor;
        theta             = theta * (1.0f - mix) + theta_extrap * mix;
    }
    scalar_sincos(theta, sine, cosine);
    const float mscale = scalar_bits_float(config->mscale_bits);
    *sine *= mscale;
    *cosine *= mscale;
}

static inline void run_rope_f32(const DayanRopeConfig * config,
                                const float *           input,
                                const int32_t *         positions,
                                const float *           freq_factors,
                                float *                 output) {
    if (config == nullptr || input == nullptr || positions == nullptr || freq_factors == nullptr || output == nullptr ||
        config->width == 0 || config->heads == 0 || config->tokens == 0 || config->batches == 0 ||
        config->n_dims == 0 || config->n_dims > config->width || (config->n_dims & 1U) != 0U ||
        (config->mode != 0U && config->mode != 2U && config->mode != 8U && config->mode != 40U)) {
        dayan_proc_exit(-22);
    }

    const uint64_t elements      = (uint64_t) config->width * config->heads * config->tokens * config->batches;
    const uint32_t position_axes = (config->mode & 8U) != 0U ? 4U : 1U;
    cache_inval_range(input, (size_t) elements * sizeof(float));
    cache_inval_range(positions, (size_t) position_axes * config->tokens * sizeof(int32_t));
    cache_inval_range(freq_factors, (size_t) config->n_dims / 2 * sizeof(float));

    const float    theta_scale = scalar_bits_float(config->theta_scale_bits);
    const uint32_t task        = dayan_native_task_id();
    const uint32_t tasks       = dayan_native_task_count();
    const uint64_t rows        = (uint64_t) config->batches * config->tokens * config->heads;
    for (uint64_t row = task; row < rows; row += tasks) {
        uint64_t       index       = row / config->heads;
        const uint32_t token       = (uint32_t) (index % config->tokens);
        const float *  source      = input + row * config->width;
        float *        destination = output + row * config->width;
        for (uint32_t dim = config->n_dims; dim < config->width; ++dim) {
            destination[dim] = source[dim];
        }

        float          theta_factor = 1.0f;
        const uint32_t half         = config->n_dims / 2;
        for (uint32_t pair = 0; pair < half; ++pair) {
            float sine;
            float cosine;
            rope_sincos(config, positions, freq_factors, token, pair, theta_factor, &sine, &cosine);
            const uint32_t first  = config->mode == 0U ? pair * 2U : pair;
            const uint32_t second = config->mode == 0U ? first + 1U : pair + half;
            const float    x0     = source[first];
            const float    x1     = source[second];
            destination[first]    = x0 * cosine - x1 * sine;
            destination[second]   = x0 * sine + x1 * cosine;
            theta_factor *= theta_scale;
        }
    }
    cache_bypass_flush();
}

}  // namespace ggml_dayan_matmul_detail

#endif

#include "ggml-dayan-kernels.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

struct runtime_context {
    npu_runtime_t * runtime = nullptr;
    npu_context_t * context = nullptr;
    npu_stream_t *  stream  = nullptr;
};

struct device_buffers {
    npu_buffer_t * config     = nullptr;
    npu_buffer_t * activation = nullptr;
    npu_buffer_t * weight     = nullptr;
    npu_buffer_t * norm       = nullptr;
    npu_buffer_t * positions  = nullptr;
    npu_buffer_t * raw        = nullptr;
    npu_buffer_t * output     = nullptr;
    npu_buffer_t * perf       = nullptr;
};

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-attention-projection: %s\n", message);
    return 1;
}

static void free_runtime(runtime_context * runtime) {
    if (runtime->stream != nullptr) {
        npu_stream_destroy(runtime->stream);
    }
    if (runtime->context != nullptr) {
        npu_context_destroy(runtime->context);
    }
    if (runtime->runtime != nullptr) {
        npu_runtime_close(runtime->runtime);
    }
}

static void free_buffers(device_buffers * buffers) {
    npu_buffer_t * list[] = { buffers->perf, buffers->output, buffers->raw,        buffers->positions,
                              buffers->norm, buffers->weight, buffers->activation, buffers->config };
    for (npu_buffer_t * buffer : list) {
        if (buffer != nullptr) {
            npu_buffer_destroy(buffer);
        }
    }
}

static int create_buffer(npu_context_t * context, size_t size, npu_buffer_t ** buffer) {
    npu_buffer_desc_t desc;
    npu_buffer_desc_init(&desc, size);
    return npu_buffer_create(context, &desc, buffer);
}

static int wait_event(npu_event_t * event) {
    if (event == nullptr) {
        return -1;
    }
    npu_event_info_t info;
    int              status         = npu_event_wait(event, 3600000U, &info);
    const int        destroy_status = npu_event_destroy(event);
    if (status == 0) {
        status = info.status < 0 ? (int) info.status : destroy_status;
    }
    return status;
}

static int upload(npu_stream_t * stream, npu_buffer_t * buffer, const void * data, size_t size) {
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_h2d_async(stream, buffer, 0, data, size, &event);
    return status < 0 ? status : wait_event(event);
}

static int download(npu_stream_t * stream, void * data, npu_buffer_t * buffer, size_t size) {
    npu_event_t * event  = nullptr;
    const int     status = npu_memcpy_d2h_async(stream, data, buffer, 0, size, &event);
    return status < 0 ? status : wait_event(event);
}

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static uint16_t encode_lowp(enum ggml_type type, float value) {
    return type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value);
}

static float decode_lowp(enum ggml_type type, uint16_t value) {
    return type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(ggml_bf16_t{ value }) : ggml_fp16_to_fp32(value);
}

static int32_t rope_position(const DayanAttentionProjectionConfig & config,
                             const std::vector<int32_t> &           positions,
                             uint32_t                               token,
                             uint32_t                               pair) {
    if (config.rope_mode != GGML_ROPE_TYPE_MROPE) {
        return positions[token];
    }
    const int32_t total  = config.sections[0] + config.sections[1] + config.sections[2] + config.sections[3];
    const int32_t sector = (int32_t) pair % total;
    int32_t       axis   = 0;
    if (sector >= config.sections[0]) {
        axis = 1;
    }
    if (sector >= config.sections[0] + config.sections[1]) {
        axis = 2;
    }
    if (sector >= config.sections[0] + config.sections[1] + config.sections[2]) {
        axis = 3;
    }
    return positions[token + (uint32_t) axis * config.tokens];
}

static void reference_matmul(enum ggml_type                type,
                             const DayanMatmulConfig &     config,
                             const std::vector<uint16_t> & activation,
                             const std::vector<uint16_t> & weight,
                             std::vector<uint16_t> *       output) {
    output->resize((size_t) (config.m * config.n));
    for (uint64_t row = 0; row < config.m; ++row) {
        for (uint64_t col = 0; col < config.n; ++col) {
            float sum = 0.0f;
            for (uint64_t inner = 0; inner < config.k; ++inner) {
                sum += decode_lowp(type, activation[(size_t) (row * config.k + inner)]) *
                       decode_lowp(type, weight[(size_t) (col * config.k + inner)]);
            }
            (*output)[(size_t) (row * config.n + col)] = encode_lowp(type, sum);
        }
    }
}

static void reference_norm_rope(enum ggml_type                         type,
                                const DayanAttentionProjectionConfig & config,
                                bool                                   has_gate,
                                const std::vector<float> &             norm,
                                const std::vector<int32_t> &           positions,
                                const std::vector<uint16_t> &          raw,
                                std::vector<uint16_t> *                output) {
    output->resize((size_t) config.tokens * config.heads * config.head_dim);
    const uint32_t step  = config.head_dim * (has_gate ? 2U : 1U);
    const uint32_t width = config.heads * step;
    float          epsilon;
    float          theta_scale;
    float          freq_scale;
    float          attn_factor;
    std::memcpy(&epsilon, &config.norm_eps_bits, sizeof(epsilon));
    std::memcpy(&theta_scale, &config.theta_scale_bits, sizeof(theta_scale));
    std::memcpy(&freq_scale, &config.freq_scale_bits, sizeof(freq_scale));
    std::memcpy(&attn_factor, &config.attn_factor_bits, sizeof(attn_factor));

    for (uint32_t token = 0; token < config.tokens; ++token) {
        for (uint32_t head = 0; head < config.heads; ++head) {
            const uint16_t * source      = raw.data() + (size_t) token * width + head * step;
            uint16_t *       destination = output->data() + ((size_t) head * config.tokens + token) * config.head_dim;
            float            sum_squares = 0.0f;
            for (uint32_t dim = 0; dim < config.head_dim; ++dim) {
                const float value = decode_lowp(type, source[dim]);
                sum_squares += value * value;
            }
            const float    inverse_rms  = 1.0f / std::sqrt(sum_squares / config.head_dim + epsilon);
            const uint32_t half         = config.n_rot / 2;
            float          theta_factor = 1.0f;
            for (uint32_t pair = 0; pair < half; ++pair) {
                const float position     = (float) rope_position(config, positions, token, pair);
                const float theta        = position * theta_factor * freq_scale;
                const float cosine       = std::cos(theta) * attn_factor;
                const float sine         = std::sin(theta) * attn_factor;
                const float x0           = decode_lowp(type, source[pair]) * norm[pair] * inverse_rms;
                const float x1           = decode_lowp(type, source[pair + half]) * norm[pair + half] * inverse_rms;
                destination[pair]        = encode_lowp(type, x0 * cosine - x1 * sine);
                destination[pair + half] = encode_lowp(type, x0 * sine + x1 * cosine);
                theta_factor *= theta_scale;
            }
            for (uint32_t dim = config.n_rot; dim < config.head_dim; ++dim) {
                destination[dim] = encode_lowp(type, decode_lowp(type, source[dim]) * norm[dim] * inverse_rms);
            }
        }
    }
}

static bool compare_lowp(enum ggml_type                type,
                         const std::vector<uint16_t> & expected,
                         const std::vector<uint16_t> & observed,
                         float *                       max_abs) {
    bool close = expected.size() == observed.size();
    *max_abs   = 0.0f;
    for (size_t index = 0; index < expected.size() && index < observed.size(); ++index) {
        const float expected_value = decode_lowp(type, expected[index]);
        const float observed_value = decode_lowp(type, observed[index]);
        const float error          = std::abs(expected_value - observed_value);
        *max_abs                   = std::max(*max_abs, error);
        close                      = close && error <= 0.02f + 0.02f * std::abs(expected_value);
    }
    return close;
}

static bool compare_fp32(const std::vector<float> & expected, const std::vector<float> & observed, float * max_abs) {
    bool close = expected.size() == observed.size();
    *max_abs   = 0.0f;
    for (size_t index = 0; index < expected.size() && index < observed.size(); ++index) {
        const float error = std::abs(expected[index] - observed[index]);
        *max_abs          = std::max(*max_abs, error);
        close             = close && error <= 0.03f + 0.03f * std::abs(expected[index]);
    }
    return close;
}

static int run_case(runtime_context * runtime, enum ggml_type type) {
    constexpr uint32_t tokens     = 3;
    constexpr uint32_t embd       = 64;
    constexpr uint32_t head_dim   = 64;
    constexpr uint32_t q_heads    = 2;
    constexpr uint32_t kv_heads   = 1;
    constexpr uint32_t output_dim = 64;
    constexpr uint32_t cache_rows = 7;
    constexpr uint32_t qg_width   = q_heads * head_dim * 2;
    constexpr uint32_t kv_width   = kv_heads * head_dim;

    std::mt19937                          generator(type == GGML_TYPE_BF16 ? 0x351bfU : 0x351f1U);
    std::uniform_real_distribution<float> input_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> weight_dist(-0.08f, 0.08f);
    std::uniform_real_distribution<float> norm_dist(-0.04f, 0.04f);
    std::vector<uint16_t>                 activation((size_t) tokens * embd);
    std::vector<uint16_t>                 qg_weight((size_t) qg_width * embd);
    std::vector<uint16_t>                 k_weight((size_t) kv_width * embd);
    std::vector<uint16_t>                 v_weight((size_t) kv_width * embd);
    std::vector<uint16_t>                 output_weight((size_t) output_dim * q_heads * head_dim);
    std::vector<float>                    attention((size_t) tokens * q_heads * head_dim);
    std::vector<float>                    q_norm(head_dim);
    std::vector<float>                    k_norm(head_dim);
    std::vector<int32_t>                  positions(tokens * 4);
    for (uint16_t & value : activation) {
        value = encode_lowp(type, input_dist(generator));
    }
    for (uint16_t & value : qg_weight) {
        value = encode_lowp(type, weight_dist(generator));
    }
    for (uint16_t & value : k_weight) {
        value = encode_lowp(type, weight_dist(generator));
    }
    for (uint16_t & value : v_weight) {
        value = encode_lowp(type, weight_dist(generator));
    }
    for (uint16_t & value : output_weight) {
        value = encode_lowp(type, weight_dist(generator));
    }
    for (float & value : q_norm) {
        value = 1.0f + norm_dist(generator);
    }
    for (float & value : k_norm) {
        value = 1.0f + norm_dist(generator);
    }
    for (uint32_t axis = 0; axis < 4; ++axis) {
        for (uint32_t token = 0; token < tokens; ++token) {
            positions[token + axis * tokens] = (int32_t) (token + axis);
        }
    }

    const auto make_config = [](uint32_t heads, uint32_t width) {
        DayanAttentionProjectionConfig config = {};
        config.matmul                         = {
            tokens, width, embd, embd * sizeof(uint16_t), embd * sizeof(uint16_t), width * sizeof(uint16_t)
        };
        config.tokens           = tokens;
        config.heads            = heads;
        config.head_dim         = head_dim;
        config.n_rot            = head_dim;
        config.rope_mode        = GGML_ROPE_TYPE_MROPE;
        config.norm_eps_bits    = float_bits(1.0e-6f);
        config.theta_scale_bits = float_bits(std::pow(10000.0f, -2.0f / (float) head_dim));
        config.freq_scale_bits  = float_bits(1.0f);
        config.attn_factor_bits = float_bits(1.0f);
        config.sections[0]      = 8;
        config.sections[1]      = 8;
        config.sections[2]      = 8;
        config.sections[3]      = 8;
        return config;
    };

    const DayanAttentionProjectionConfig q_config  = make_config(q_heads, qg_width);
    const DayanAttentionProjectionConfig kv_config = make_config(kv_heads, kv_width);
    std::vector<uint16_t>                qg_expected;
    std::vector<uint16_t>                k_raw_expected;
    std::vector<uint16_t>                v_raw_expected;
    std::vector<uint16_t>                q_expected;
    std::vector<uint16_t>                k_expected;
    std::vector<uint16_t>                v_expected((size_t) tokens * kv_width);
    reference_matmul(type, q_config.matmul, activation, qg_weight, &qg_expected);
    reference_matmul(type, kv_config.matmul, activation, k_weight, &k_raw_expected);
    reference_matmul(type, kv_config.matmul, activation, v_weight, &v_raw_expected);
    reference_norm_rope(type, q_config, true, q_norm, positions, qg_expected, &q_expected);
    reference_norm_rope(type, kv_config, false, k_norm, positions, k_raw_expected, &k_expected);
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                v_expected[((size_t) head * tokens + token) * head_dim + dim] =
                    v_raw_expected[((size_t) token * kv_heads + head) * head_dim + dim];
            }
        }
    }

    const auto launch_projection = [&](const DayanAttentionProjectionConfig & config,
                                       const std::vector<uint16_t> & weight, const std::vector<float> * norm, bool qg,
                                       bool value, std::vector<uint16_t> * raw_result,
                                       std::vector<uint16_t> * output_result) {
        device_buffers  buffers;
        DayanMatmulPerf perf        = {};
        const size_t    raw_size    = (size_t) config.tokens * config.matmul.n * sizeof(uint16_t);
        const size_t    output_size = (size_t) config.tokens * config.heads * config.head_dim * sizeof(uint16_t);
        int             status      = create_buffer(runtime->context, sizeof(config), &buffers.config);
        if (status == 0) {
            status = create_buffer(runtime->context, activation.size() * sizeof(uint16_t), &buffers.activation);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, weight.size() * sizeof(uint16_t), &buffers.weight);
        }
        if (status == 0 && norm != nullptr) {
            status = create_buffer(runtime->context, norm->size() * sizeof(float), &buffers.norm);
        }
        if (status == 0 && !value) {
            status = create_buffer(runtime->context, positions.size() * sizeof(int32_t), &buffers.positions);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, raw_size, &buffers.raw);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, output_size, &buffers.output);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, sizeof(perf), &buffers.perf);
        }
        if (status == 0) {
            status = upload(runtime->stream, buffers.config, &config, sizeof(config));
        }
        if (status == 0) {
            status =
                upload(runtime->stream, buffers.activation, activation.data(), activation.size() * sizeof(uint16_t));
        }
        if (status == 0) {
            status = upload(runtime->stream, buffers.weight, weight.data(), weight.size() * sizeof(uint16_t));
        }
        if (status == 0 && norm != nullptr) {
            status = upload(runtime->stream, buffers.norm, norm->data(), norm->size() * sizeof(float));
        }
        if (status == 0 && !value) {
            status = upload(runtime->stream, buffers.positions, positions.data(), positions.size() * sizeof(int32_t));
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            npu_kernel_launch_attrs_init(&attrs, runtime->context, runtime->stream,
                                         qg    ? "qg-proj" :
                                         value ? "v-proj" :
                                                 "k-proj");
            attrs.config.task_dim  = NPUTaskDim3{ 1, 1, 1 };
            attrs.config.task_mode = NPUTaskMode::kSingleCore;
            attrs.timeout_ms       = 3600000U;
            if (qg) {
                status = type == GGML_TYPE_BF16 ?
                             project_qg_bf16_1c(buffers.config, buffers.activation, buffers.weight, buffers.norm,
                                                buffers.positions, buffers.raw, buffers.output, buffers.perf, &attrs) :
                             project_qg_fp16_1c(buffers.config, buffers.activation, buffers.weight, buffers.norm,
                                                buffers.positions, buffers.raw, buffers.output, buffers.perf, &attrs);
            } else if (value) {
                status = type == GGML_TYPE_BF16 ? project_v_bf16_1c(buffers.config, buffers.activation, buffers.weight,
                                                                    buffers.raw, buffers.output, buffers.perf, &attrs) :
                                                  project_v_fp16_1c(buffers.config, buffers.activation, buffers.weight,
                                                                    buffers.raw, buffers.output, buffers.perf, &attrs);
            } else {
                status = type == GGML_TYPE_BF16 ?
                             project_k_bf16_1c(buffers.config, buffers.activation, buffers.weight, buffers.norm,
                                               buffers.positions, buffers.raw, buffers.output, buffers.perf, &attrs) :
                             project_k_fp16_1c(buffers.config, buffers.activation, buffers.weight, buffers.norm,
                                               buffers.positions, buffers.raw, buffers.output, buffers.perf, &attrs);
            }
        }
        raw_result->resize(raw_size / sizeof(uint16_t));
        output_result->resize(output_size / sizeof(uint16_t));
        if (status == 0) {
            status = download(runtime->stream, raw_result->data(), buffers.raw, raw_size);
        }
        if (status == 0) {
            status = download(runtime->stream, output_result->data(), buffers.output, output_size);
        }
        if (status == 0) {
            status = download(runtime->stream, &perf, buffers.perf, sizeof(perf));
        }
        free_buffers(&buffers);
        return status;
    };

    std::vector<uint16_t> qg_observed;
    std::vector<uint16_t> q_observed;
    std::vector<uint16_t> k_raw_observed;
    std::vector<uint16_t> k_observed;
    std::vector<uint16_t> v_raw_observed;
    std::vector<uint16_t> v_observed;
    const int q_status = launch_projection(q_config, qg_weight, &q_norm, true, false, &qg_observed, &q_observed);
    const int k_status = launch_projection(kv_config, k_weight, &k_norm, false, false, &k_raw_observed, &k_observed);
    const int v_status = launch_projection(kv_config, v_weight, nullptr, false, true, &v_raw_observed, &v_observed);
    if (q_status < 0 || k_status < 0 || v_status < 0) {
        std::fprintf(stderr, "type=%s q_status=%d k_status=%d v_status=%d\n", ggml_type_name(type), q_status, k_status,
                     v_status);
        return fail("projection kernel launch failed");
    }

    const std::vector<uint16_t> & q_flash           = q_observed;
    const std::vector<uint16_t> & k_flash           = k_observed;
    const std::vector<uint16_t> & v_flash           = v_observed;
    const std::vector<uint16_t> & q_flash_reference = q_expected;
    const std::vector<uint16_t> & k_flash_reference = k_expected;
    const std::vector<uint16_t> & v_flash_reference = v_expected;
    const float                   attention_scale   = 1.0f / std::sqrt((float) head_dim);
    std::vector<float>            attention_expected(attention.size());
    for (uint32_t query = 0; query < tokens; ++query) {
        for (uint32_t head = 0; head < q_heads; ++head) {
            const uint32_t kv_head = head / (q_heads / kv_heads);
            float          scores[tokens];
            float          max_score = -std::numeric_limits<float>::infinity();
            for (uint32_t key = 0; key < tokens; ++key) {
                if (key > query) {
                    scores[key] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                float score = 0.0f;
                for (uint32_t dim = 0; dim < head_dim; ++dim) {
                    score += decode_lowp(type, q_flash_reference[((size_t) head * tokens + query) * head_dim + dim]) *
                             decode_lowp(type, k_flash_reference[((size_t) kv_head * tokens + key) * head_dim + dim]);
                }
                scores[key] = score * attention_scale;
                max_score   = std::max(max_score, scores[key]);
            }
            float sum = 0.0f;
            for (uint32_t key = 0; key <= query; ++key) {
                scores[key] = std::exp(scores[key] - max_score);
                sum += scores[key];
            }
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                float value = 0.0f;
                for (uint32_t key = 0; key <= query; ++key) {
                    value += scores[key] / sum *
                             decode_lowp(type, v_flash_reference[((size_t) kv_head * tokens + key) * head_dim + dim]);
                }
                attention_expected[((size_t) query * q_heads + head) * head_dim + dim] = value;
            }
        }
    }

    DayanFlashAttentionConfig flash_config = {};
    flash_config.batch                     = 1;
    flash_config.q_heads                   = q_heads;
    flash_config.kv_heads                  = kv_heads;
    flash_config.q_len                     = tokens;
    flash_config.kv_len                    = tokens;
    flash_config.head_dim                  = head_dim;
    flash_config.scale_bits                = float_bits(attention_scale);
    flash_config.flags                     = 1U;
    flash_config.q_row_stride              = head_dim * sizeof(uint16_t);
    flash_config.q_head_stride             = tokens * flash_config.q_row_stride;
    flash_config.q_batch_stride            = q_heads * flash_config.q_head_stride;
    flash_config.k_row_stride              = head_dim * sizeof(uint16_t);
    flash_config.k_head_stride             = tokens * flash_config.k_row_stride;
    flash_config.k_batch_stride            = kv_heads * flash_config.k_head_stride;
    flash_config.v_row_stride              = head_dim * sizeof(uint16_t);
    flash_config.v_head_stride             = tokens * flash_config.v_row_stride;
    flash_config.v_batch_stride            = kv_heads * flash_config.v_head_stride;
    device_buffers flash_buffers;
    int64_t        flash_result = -1;
    int            flash_status = create_buffer(runtime->context, sizeof(flash_config), &flash_buffers.config);
    if (flash_status == 0) {
        flash_status = create_buffer(runtime->context, q_flash.size() * sizeof(uint16_t), &flash_buffers.weight);
    }
    if (flash_status == 0) {
        flash_status = create_buffer(runtime->context, k_flash.size() * sizeof(uint16_t), &flash_buffers.norm);
    }
    if (flash_status == 0) {
        flash_status = create_buffer(runtime->context, v_flash.size() * sizeof(uint16_t), &flash_buffers.positions);
    }
    if (flash_status == 0) {
        flash_status = create_buffer(runtime->context, attention.size() * sizeof(float), &flash_buffers.output);
    }
    if (flash_status == 0) {
        flash_status = create_buffer(runtime->context, sizeof(flash_result), &flash_buffers.perf);
    }
    if (flash_status == 0) {
        flash_status = upload(runtime->stream, flash_buffers.config, &flash_config, sizeof(flash_config));
    }
    if (flash_status == 0) {
        flash_status = upload(runtime->stream, flash_buffers.weight, q_flash.data(), q_flash.size() * sizeof(uint16_t));
    }
    if (flash_status == 0) {
        flash_status = upload(runtime->stream, flash_buffers.norm, k_flash.data(), k_flash.size() * sizeof(uint16_t));
    }
    if (flash_status == 0) {
        flash_status =
            upload(runtime->stream, flash_buffers.positions, v_flash.data(), v_flash.size() * sizeof(uint16_t));
    }
    if (flash_status == 0) {
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, runtime->context, runtime->stream, "fattn");
        attrs.config.task_dim  = NPUTaskDim3{ 1, 1, 1 };
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms       = 3600000U;
        flash_status =
            type == GGML_TYPE_BF16 ?
                flash_attention_bf16_1c(flash_buffers.config, flash_buffers.weight, flash_buffers.norm,
                                        flash_buffers.positions, flash_buffers.output, flash_buffers.perf, &attrs) :
                flash_attention_fp16_1c(flash_buffers.config, flash_buffers.weight, flash_buffers.norm,
                                        flash_buffers.positions, flash_buffers.output, flash_buffers.perf, &attrs);
    }
    std::vector<float> flash_layout(attention.size());
    if (flash_status == 0) {
        flash_status = download(runtime->stream, &flash_result, flash_buffers.perf, sizeof(flash_result));
    }
    if (flash_status == 0 && flash_result < 0) {
        flash_status = (int) flash_result;
    }
    if (flash_status == 0) {
        flash_status =
            download(runtime->stream, flash_layout.data(), flash_buffers.output, flash_layout.size() * sizeof(float));
    }
    free_buffers(&flash_buffers);
    if (flash_status < 0) {
        std::fprintf(stderr, "type=%s flash_status=%d device_status=%lld\n", ggml_type_name(type), flash_status,
                     (long long) flash_result);
        return fail("Flash-Attention kernel launch failed");
    }
    attention = flash_layout;

    const std::vector<int32_t> cache_indices     = { 5, 1, 6 };
    std::vector<int32_t>       positions_indices = positions;
    positions_indices.insert(positions_indices.end(), cache_indices.begin(), cache_indices.end());
    std::vector<uint16_t> k_cache_expected((size_t) cache_rows * kv_width);
    std::vector<uint16_t> v_cache_expected((size_t) cache_rows * kv_width);
    for (uint16_t & value : k_cache_expected) {
        value = encode_lowp(type, input_dist(generator));
    }
    for (uint16_t & value : v_cache_expected) {
        value = encode_lowp(type, input_dist(generator));
    }
    const std::vector<uint16_t> k_cache_initial = k_cache_expected;
    const std::vector<uint16_t> v_cache_initial = v_cache_expected;
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < kv_heads; ++head) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                const size_t source      = ((size_t) head * tokens + token) * head_dim + dim;
                const size_t target      = (size_t) cache_indices[token] * kv_width + head * head_dim + dim;
                k_cache_expected[target] = k_expected[source];
                v_cache_expected[target] = v_expected[source];
            }
        }
    }

    const auto launch_cache_projection = [&](bool k_projection, const std::vector<uint16_t> & weight,
                                             const std::vector<uint16_t> & initial_cache,
                                             std::vector<uint16_t> *       output_cache) {
        DayanKVProjectionConfig config        = {};
        config.projection                     = kv_config;
        config.cache_rows                     = cache_rows;
        const std::vector<int32_t> & metadata = k_projection ? positions_indices : cache_indices;
        device_buffers               buffers;
        DayanMatmulPerf              perf   = {};
        int                          status = create_buffer(runtime->context, sizeof(config), &buffers.config);
        if (status == 0) {
            status = create_buffer(runtime->context, activation.size() * sizeof(uint16_t), &buffers.activation);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, weight.size() * sizeof(uint16_t), &buffers.weight);
        }
        if (status == 0 && k_projection) {
            status = create_buffer(runtime->context, q_norm.size() * sizeof(float), &buffers.norm);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, metadata.size() * sizeof(int32_t), &buffers.positions);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, (size_t) tokens * kv_width * sizeof(uint16_t), &buffers.raw);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, initial_cache.size() * sizeof(uint16_t), &buffers.output);
        }
        if (status == 0) {
            status = create_buffer(runtime->context, sizeof(perf), &buffers.perf);
        }
        if (status == 0) {
            status = upload(runtime->stream, buffers.config, &config, sizeof(config));
        }
        if (status == 0) {
            status =
                upload(runtime->stream, buffers.activation, activation.data(), activation.size() * sizeof(uint16_t));
        }
        if (status == 0) {
            status = upload(runtime->stream, buffers.weight, weight.data(), weight.size() * sizeof(uint16_t));
        }
        if (status == 0 && k_projection) {
            status = upload(runtime->stream, buffers.norm, k_norm.data(), k_norm.size() * sizeof(float));
        }
        if (status == 0) {
            status = upload(runtime->stream, buffers.positions, metadata.data(), metadata.size() * sizeof(int32_t));
        }
        if (status == 0) {
            status =
                upload(runtime->stream, buffers.output, initial_cache.data(), initial_cache.size() * sizeof(uint16_t));
        }
        if (status == 0) {
            npu_kernel_launch_attrs_t attrs;
            npu_kernel_launch_attrs_init(&attrs, runtime->context, runtime->stream,
                                         k_projection ? "k-cache" : "v-cache");
            attrs.config.task_dim  = NPUTaskDim3{ 1, 1, 1 };
            attrs.config.task_mode = NPUTaskMode::kSingleCore;
            attrs.timeout_ms       = 3600000U;
            if (k_projection) {
                status =
                    type == GGML_TYPE_BF16 ?
                        project_k_cache_bf16_1c(buffers.config, buffers.activation, buffers.weight, buffers.norm,
                                                buffers.positions, buffers.raw, buffers.output, buffers.perf, &attrs) :
                        project_k_cache_fp16_1c(buffers.config, buffers.activation, buffers.weight, buffers.norm,
                                                buffers.positions, buffers.raw, buffers.output, buffers.perf, &attrs);
            } else {
                status =
                    type == GGML_TYPE_BF16 ?
                        project_v_cache_bf16_1c(buffers.config, buffers.activation, buffers.weight, buffers.positions,
                                                buffers.raw, buffers.output, buffers.perf, &attrs) :
                        project_v_cache_fp16_1c(buffers.config, buffers.activation, buffers.weight, buffers.positions,
                                                buffers.raw, buffers.output, buffers.perf, &attrs);
            }
        }
        output_cache->resize(initial_cache.size());
        if (status == 0) {
            status = download(runtime->stream, output_cache->data(), buffers.output,
                              output_cache->size() * sizeof(uint16_t));
        }
        if (status == 0) {
            status = download(runtime->stream, &perf, buffers.perf, sizeof(perf));
        }
        free_buffers(&buffers);
        return status;
    };

    std::vector<uint16_t> k_cache_observed;
    std::vector<uint16_t> v_cache_observed;
    const int             k_cache_status = launch_cache_projection(true, k_weight, k_cache_initial, &k_cache_observed);
    const int             v_cache_status = launch_cache_projection(false, v_weight, v_cache_initial, &v_cache_observed);
    if (k_cache_status < 0 || v_cache_status < 0) {
        std::fprintf(stderr, "type=%s k_cache_status=%d v_cache_status=%d\n", ggml_type_name(type), k_cache_status,
                     v_cache_status);
        return fail("KV cache projection kernel launch failed");
    }

    DayanGatedProjectionConfig gated_config = {};
    gated_config.matmul                     = { tokens,
                                                output_dim,
                                                q_heads * head_dim,
                                                q_heads * head_dim * sizeof(uint16_t),
                                                q_heads * head_dim * sizeof(uint16_t),
                                                output_dim * sizeof(float) };
    gated_config.tokens                     = tokens;
    gated_config.heads                      = q_heads;
    gated_config.head_dim                   = head_dim;
    std::vector<uint16_t> gated_activation(attention.size());
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t head = 0; head < q_heads; ++head) {
            for (uint32_t dim = 0; dim < head_dim; ++dim) {
                const size_t index      = ((size_t) token * q_heads + head) * head_dim + dim;
                const size_t gate_index = (size_t) token * qg_width + head * 2U * head_dim + head_dim + dim;
                const float  gate       = decode_lowp(type, qg_expected[gate_index]);
                gated_activation[index] = encode_lowp(type, attention_expected[index] / (1.0f + std::exp(-gate)));
            }
        }
    }
    std::vector<float> gated_expected((size_t) tokens * output_dim);
    for (uint32_t token = 0; token < tokens; ++token) {
        for (uint32_t output = 0; output < output_dim; ++output) {
            float sum = 0.0f;
            for (uint32_t inner = 0; inner < q_heads * head_dim; ++inner) {
                sum += decode_lowp(type, gated_activation[(size_t) token * q_heads * head_dim + inner]) *
                       decode_lowp(type, output_weight[(size_t) output * q_heads * head_dim + inner]);
            }
            gated_expected[(size_t) token * output_dim + output] = sum;
        }
    }

    device_buffers  gated_buffers;
    DayanMatmulPerf gated_perf   = {};
    int             gated_status = create_buffer(runtime->context, sizeof(gated_config), &gated_buffers.config);
    if (gated_status == 0) {
        gated_status = create_buffer(runtime->context, attention.size() * sizeof(float), &gated_buffers.raw);
    }
    if (gated_status == 0) {
        gated_status = create_buffer(runtime->context, qg_observed.size() * sizeof(uint16_t), &gated_buffers.norm);
    }
    if (gated_status == 0) {
        gated_status =
            create_buffer(runtime->context, gated_activation.size() * sizeof(uint16_t), &gated_buffers.activation);
    }
    if (gated_status == 0) {
        gated_status = create_buffer(runtime->context, output_weight.size() * sizeof(uint16_t), &gated_buffers.weight);
    }
    if (gated_status == 0) {
        gated_status = create_buffer(runtime->context, gated_expected.size() * sizeof(float), &gated_buffers.output);
    }
    if (gated_status == 0) {
        gated_status = create_buffer(runtime->context, sizeof(gated_perf), &gated_buffers.perf);
    }
    if (gated_status == 0) {
        gated_status = upload(runtime->stream, gated_buffers.config, &gated_config, sizeof(gated_config));
    }
    if (gated_status == 0) {
        gated_status = upload(runtime->stream, gated_buffers.raw, attention.data(), attention.size() * sizeof(float));
    }
    if (gated_status == 0) {
        gated_status =
            upload(runtime->stream, gated_buffers.norm, qg_observed.data(), qg_observed.size() * sizeof(uint16_t));
    }
    if (gated_status == 0) {
        gated_status = upload(runtime->stream, gated_buffers.weight, output_weight.data(),
                              output_weight.size() * sizeof(uint16_t));
    }
    if (gated_status == 0) {
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, runtime->context, runtime->stream, "gate-out");
        attrs.config.task_dim  = NPUTaskDim3{ 1, 1, 1 };
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms       = 3600000U;
        gated_status = type == GGML_TYPE_BF16 ?
                           project_gated_output_bf16_1c(gated_buffers.config, gated_buffers.raw, gated_buffers.norm,
                                                        gated_buffers.activation, gated_buffers.weight,
                                                        gated_buffers.output, gated_buffers.perf, &attrs) :
                           project_gated_output_fp16_1c(gated_buffers.config, gated_buffers.raw, gated_buffers.norm,
                                                        gated_buffers.activation, gated_buffers.weight,
                                                        gated_buffers.output, gated_buffers.perf, &attrs);
    }
    std::vector<float> gated_observed(gated_expected.size());
    std::vector<uint16_t> gated_activation_observed(gated_activation.size());
    if (gated_status == 0) {
        gated_status = download(runtime->stream, gated_observed.data(), gated_buffers.output,
                                gated_observed.size() * sizeof(float));
    }
    if (gated_status == 0) {
        gated_status = download(runtime->stream, gated_activation_observed.data(), gated_buffers.activation,
                                gated_activation_observed.size() * sizeof(uint16_t));
    }
    if (gated_status == 0) {
        gated_status = download(runtime->stream, &gated_perf, gated_buffers.perf, sizeof(gated_perf));
    }
    free_buffers(&gated_buffers);
    if (gated_status < 0) {
        std::fprintf(stderr, "type=%s gated_status=%d\n", ggml_type_name(type), gated_status);
        return fail("gated output projection kernel launch failed");
    }

    float      qg_error        = 0.0f;
    float      q_error         = 0.0f;
    float      k_error         = 0.0f;
    float      v_error         = 0.0f;
    float      k_cache_error   = 0.0f;
    float      v_cache_error   = 0.0f;
    float      attention_error = 0.0f;
    float      gated_activation_error = 0.0f;
    float      gated_error     = 0.0f;
    const bool close           = compare_lowp(type, qg_expected, qg_observed, &qg_error) &&
                                 compare_lowp(type, q_expected, q_observed, &q_error) &&
                                 compare_lowp(type, k_expected, k_observed, &k_error) &&
                                 compare_lowp(type, v_expected, v_observed, &v_error) &&
                                 compare_lowp(type, k_cache_expected, k_cache_observed, &k_cache_error) &&
                                 compare_lowp(type, v_cache_expected, v_cache_observed, &v_cache_error) &&
                                 compare_fp32(attention_expected, attention, &attention_error) &&
                                 compare_lowp(type, gated_activation, gated_activation_observed,
                                              &gated_activation_error) &&
                                 compare_fp32(gated_expected, gated_observed, &gated_error);
    std::printf(
        "type=%s tokens=%u embd=%u head_dim=%u q_heads=%u kv_heads=%u qg=%g q=%g k=%g v=%g kc=%g vc=%g attn=%g "
        "gated-activation=%g gated=%g\n",
        ggml_type_name(type), tokens, embd, head_dim, q_heads, kv_heads, qg_error, q_error, k_error, v_error,
        k_cache_error, v_cache_error, attention_error, gated_activation_error, gated_error);
    return close ? 0 : fail("projection numerical comparison failed");
}

int main() {
    runtime_context       runtime;
    npu_runtime_options_t options;
    npu_runtime_options_init(&options);
    options.default_timeout_ms = 3600000U;
    const char * socket        = std::getenv("DAYAN_SERVER_SOCKET");
    if (socket != nullptr && socket[0] != '\0') {
        options.server_socket = socket;
    }
    int status = npu_runtime_open(&runtime.runtime, &options);
    if (status == 0) {
        status = npu_context_create(runtime.runtime, nullptr, &runtime.context);
    }
    if (status == 0) {
        status = npu_stream_create(runtime.context, nullptr, &runtime.stream);
    }
    if (status < 0) {
        free_runtime(&runtime);
        return fail("cannot initialize the Dayan runtime");
    }

    const int result = run_case(&runtime, GGML_TYPE_BF16) != 0 || run_case(&runtime, GGML_TYPE_F16) != 0;
    free_runtime(&runtime);
    if (result != 0) {
        return 1;
    }
    std::puts("test-dayan-attention-projection: PASS");
    return 0;
}

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
    npu_buffer_t * config = nullptr;
    npu_buffer_t * q      = nullptr;
    npu_buffer_t * k      = nullptr;
    npu_buffer_t * v      = nullptr;
    npu_buffer_t * output = nullptr;
    npu_buffer_t * result = nullptr;
};

struct attention_case {
    const char *   name;
    enum ggml_type type;
    uint32_t       batch;
    uint32_t       q_heads;
    uint32_t       kv_heads;
    uint32_t       q_len;
    uint32_t       kv_len;
    uint32_t       head_dim;
    uint32_t       causal_offset;
    bool           causal;
    bool           interleaved_kv = false;
    uint32_t       tasks          = 1;
    bool           four_core      = false;
    float          qk_range       = 0.25f;
};

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-attention-pipeline: %s\n", message);
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
    npu_buffer_t * list[] = { buffers->result, buffers->output, buffers->v, buffers->k, buffers->q, buffers->config };
    for (npu_buffer_t * buffer : list) {
        if (buffer != nullptr) {
            npu_buffer_destroy(buffer);
        }
    }
}

static int wait_event(npu_event_t * event) {
    if (event == nullptr) {
        return -1;
    }
    npu_event_info_t info;
    int              status         = npu_event_wait(event, 3600000U, &info);
    const int        destroy_status = npu_event_destroy(event);
    if (status == 0) {
        status = info.status < 0 ? static_cast<int>(info.status) : destroy_status;
    }
    return status;
}

static int create_buffer(npu_context_t * context, size_t size, npu_buffer_t ** buffer) {
    npu_buffer_desc_t desc;
    npu_buffer_desc_init(&desc, size);
    return npu_buffer_create(context, &desc, buffer);
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

static size_t q_index(const attention_case & test, uint32_t b, uint32_t h, uint32_t row, uint32_t dim) {
    return (((static_cast<size_t>(b) * test.q_heads + h) * test.q_len + row) * test.head_dim + dim);
}

static size_t kv_index(const attention_case & test, uint32_t b, uint32_t h, uint32_t row, uint32_t dim) {
    if (test.interleaved_kv) {
        return (((static_cast<size_t>(b) * test.kv_len + row) * test.kv_heads + h) * test.head_dim + dim);
    }
    return (((static_cast<size_t>(b) * test.kv_heads + h) * test.kv_len + row) * test.head_dim + dim);
}

static size_t output_index(const attention_case & test, uint32_t b, uint32_t row, uint32_t h, uint32_t dim) {
    return (((static_cast<size_t>(b) * test.q_len + row) * test.q_heads + h) * test.head_dim + dim);
}

static void reference_attention(const attention_case &        test,
                                const std::vector<uint16_t> & q,
                                const std::vector<uint16_t> & k,
                                const std::vector<uint16_t> & v,
                                std::vector<float> *          output) {
    const float        scale = 1.0f / std::sqrt(static_cast<float>(test.head_dim));
    std::vector<float> scores(test.kv_len);
    for (uint32_t b = 0; b < test.batch; ++b) {
        for (uint32_t qh = 0; qh < test.q_heads; ++qh) {
            const uint32_t kvh = qh / (test.q_heads / test.kv_heads);
            for (uint32_t qr = 0; qr < test.q_len; ++qr) {
                const uint32_t keys      = test.causal ? test.causal_offset + qr + 1 : test.kv_len;
                float          max_score = -std::numeric_limits<float>::infinity();
                for (uint32_t kr = 0; kr < keys; ++kr) {
                    float score = 0.0f;
                    for (uint32_t dim = 0; dim < test.head_dim; ++dim) {
                        score += decode_lowp(test.type, q[q_index(test, b, qh, qr, dim)]) *
                                 decode_lowp(test.type, k[kv_index(test, b, kvh, kr, dim)]);
                    }
                    scores[kr] = score * scale;
                    max_score  = std::max(max_score, scores[kr]);
                }

                float sum = 0.0f;
                for (uint32_t kr = 0; kr < keys; ++kr) {
                    scores[kr] = std::exp(scores[kr] - max_score);
                    sum += scores[kr];
                }
                for (uint32_t dim = 0; dim < test.head_dim; ++dim) {
                    float value = 0.0f;
                    for (uint32_t kr = 0; kr < keys; ++kr) {
                        value += scores[kr] * decode_lowp(test.type, v[kv_index(test, b, kvh, kr, dim)]);
                    }
                    (*output)[output_index(test, b, qr, qh, dim)] = value / sum;
                }
            }
        }
    }
}

static int run_case(runtime_context * runtime, const attention_case & test, uint32_t seed) {
    const size_t          q_elements  = static_cast<size_t>(test.batch) * test.q_heads * test.q_len * test.head_dim;
    const size_t          kv_elements = static_cast<size_t>(test.batch) * test.kv_heads * test.kv_len * test.head_dim;
    const size_t          output_elements = static_cast<size_t>(test.batch) * test.q_len * test.q_heads * test.head_dim;
    std::vector<uint16_t> q(q_elements);
    std::vector<uint16_t> k(kv_elements);
    std::vector<uint16_t> v(kv_elements);
    std::vector<float>    expected(output_elements);
    std::vector<float>    observed(output_elements, std::numeric_limits<float>::quiet_NaN());

    std::mt19937                          generator(seed);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::uniform_real_distribution<float> qk_distribution(-test.qk_range, test.qk_range);
    for (uint16_t & value : q) {
        value = encode_lowp(test.type, qk_distribution(generator));
    }
    for (uint16_t & value : k) {
        value = encode_lowp(test.type, qk_distribution(generator));
    }
    for (uint16_t & value : v) {
        value = encode_lowp(test.type, distribution(generator));
    }
    reference_attention(test, q, k, v, &expected);

    DayanFlashAttentionConfig config = {};
    config.batch                     = test.batch;
    config.q_heads                   = test.q_heads;
    config.kv_heads                  = test.kv_heads;
    config.q_len                     = test.q_len;
    config.kv_len                    = test.kv_len;
    config.head_dim                  = test.head_dim;
    config.scale_bits                = float_bits(1.0f / std::sqrt(static_cast<float>(test.head_dim)));
    config.causal_offset             = test.causal_offset;
    config.flags                     = test.causal ? 1U : 0U;
    config.q_row_stride              = test.head_dim * sizeof(uint16_t);
    config.q_head_stride             = static_cast<uint64_t>(test.q_len) * config.q_row_stride;
    config.q_batch_stride            = static_cast<uint64_t>(test.q_heads) * config.q_head_stride;
    if (test.interleaved_kv) {
        config.k_head_stride  = test.head_dim * sizeof(uint16_t);
        config.k_row_stride   = static_cast<uint64_t>(test.kv_heads) * config.k_head_stride;
        config.k_batch_stride = static_cast<uint64_t>(test.kv_len) * config.k_row_stride;
        config.v_head_stride  = test.head_dim * sizeof(uint16_t);
        config.v_row_stride   = static_cast<uint64_t>(test.kv_heads) * config.v_head_stride;
        config.v_batch_stride = static_cast<uint64_t>(test.kv_len) * config.v_row_stride;
    } else {
        config.k_row_stride   = test.head_dim * sizeof(uint16_t);
        config.k_head_stride  = static_cast<uint64_t>(test.kv_len) * config.k_row_stride;
        config.k_batch_stride = static_cast<uint64_t>(test.kv_heads) * config.k_head_stride;
        config.v_row_stride   = test.head_dim * sizeof(uint16_t);
        config.v_head_stride  = static_cast<uint64_t>(test.kv_len) * config.v_row_stride;
        config.v_batch_stride = static_cast<uint64_t>(test.kv_heads) * config.v_head_stride;
    }

    device_buffers buffers;
    int64_t        device_result = -1;
    int            status        = create_buffer(runtime->context, sizeof(config), &buffers.config);
    if (status == 0) {
        status = create_buffer(runtime->context, q.size() * sizeof(uint16_t), &buffers.q);
    }
    if (status == 0) {
        status = create_buffer(runtime->context, k.size() * sizeof(uint16_t), &buffers.k);
    }
    if (status == 0) {
        status = create_buffer(runtime->context, v.size() * sizeof(uint16_t), &buffers.v);
    }
    if (status == 0) {
        status = create_buffer(runtime->context, observed.size() * sizeof(float), &buffers.output);
    }
    if (status == 0) {
        status = create_buffer(runtime->context, sizeof(device_result), &buffers.result);
    }
    if (status == 0) {
        status = upload(runtime->stream, buffers.config, &config, sizeof(config));
    }
    if (status == 0) {
        status = upload(runtime->stream, buffers.q, q.data(), q.size() * sizeof(uint16_t));
    }
    if (status == 0) {
        status = upload(runtime->stream, buffers.k, k.data(), k.size() * sizeof(uint16_t));
    }
    if (status == 0) {
        status = upload(runtime->stream, buffers.v, v.data(), v.size() * sizeof(uint16_t));
    }
    if (status == 0) {
        status = upload(runtime->stream, buffers.output, observed.data(), observed.size() * sizeof(float));
    }
    if (status == 0) {
        status = upload(runtime->stream, buffers.result, &device_result, sizeof(device_result));
    }
    if (status == 0) {
        npu_kernel_launch_attrs_t attrs;
        const char *              launch_name = test.type == GGML_TYPE_BF16 ? "flash-attn-bf16" : "flash-attn-fp16";
        npu_kernel_launch_attrs_init(&attrs, runtime->context, runtime->stream, launch_name);
        attrs.config.task_dim  = NPUTaskDim3{ test.four_core ? 4U : test.tasks, 1, 1 };
        attrs.config.task_mode = test.four_core ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
        attrs.timeout_ms       = 3600000U;
        const auto kernel = test.type == GGML_TYPE_BF16 ?
            (test.four_core ? flash_attention_bf16_4c : flash_attention_bf16_1c) :
            (test.four_core ? flash_attention_fp16_4c : flash_attention_fp16_1c);
        status = kernel(buffers.config, buffers.q, buffers.k, buffers.v, buffers.output, buffers.result, &attrs);
    }
    if (status == 0) {
        status = download(runtime->stream, &device_result, buffers.result, sizeof(device_result));
    }
    if (status == 0 && device_result < 0) {
        status = static_cast<int>(device_result);
    }
    if (status == 0) {
        status = download(runtime->stream, observed.data(), buffers.output, observed.size() * sizeof(float));
    }
    free_buffers(&buffers);
    if (status < 0) {
        std::fprintf(stderr, "case=%s launch_status=%d device_status=%lld\n", test.name, status,
                     static_cast<long long>(device_result));
        return fail("Attention kernel launch failed");
    }

    float  max_abs   = 0.0f;
    float  max_rel   = 0.0f;
    size_t worst     = 0;
    size_t nonfinite = 0;
    bool   close     = true;
    for (size_t index = 0; index < expected.size(); ++index) {
        const float abs_error = std::abs(observed[index] - expected[index]);
        const float rel_error = abs_error / std::max(std::abs(expected[index]), 1.0e-6f);
        if (!std::isfinite(observed[index])) {
            ++nonfinite;
            close = false;
            continue;
        }
        if (abs_error > 0.001f + 0.01f * std::abs(expected[index])) {
            close = false;
        }
        if (abs_error > max_abs) {
            max_abs = abs_error;
            worst   = index;
        }
        max_rel = std::max(max_rel, rel_error);
    }
    std::printf(
        "case=%s type=%s batch=%u q_heads=%u kv_heads=%u q_len=%u kv_len=%u head_dim=%u max_abs=%g "
        "max_rel=%g nonfinite=%zu worst=%zu expected=%g observed=%g\n",
        test.name, ggml_type_name(test.type), test.batch, test.q_heads, test.kv_heads, test.q_len, test.kv_len,
        test.head_dim, max_abs, max_rel, nonfinite, worst, expected[worst], observed[worst]);
    if (!close) {
        for (uint32_t row = 0; row < test.q_len; ++row) {
            float row_max = 0.0f;
            for (uint32_t head = 0; head < test.q_heads; ++head) {
                for (uint32_t dim = 0; dim < test.head_dim; ++dim) {
                    const size_t index = output_index(test, 0, row, head, dim);
                    row_max            = std::max(row_max, std::abs(observed[index] - expected[index]));
                }
            }
            std::printf("case=%s row=%u max_abs=%g\n", test.name, row, row_max);
        }
    }
    return close ? 0 : fail("Attention numerical comparison failed");
}

int main(int argc, char ** argv) {
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

    const attention_case cases[] = {
        { "pipeline-full-bf16", GGML_TYPE_BF16, 1, 2, 1, 64, 128, 64, 0, false },
        { "pipeline-two-full-bf16", GGML_TYPE_BF16, 1, 1, 1, 1, 256, 64, 0, false },
        { "pipeline-tail-bf16", GGML_TYPE_BF16, 1, 1, 1, 17, 64, 64, 0, false },
        { "pipeline-mask-bf16", GGML_TYPE_BF16, 1, 1, 1, 17, 64, 64, 47, true },
        { "pipeline-causal-bf16", GGML_TYPE_BF16, 1, 1, 1, 129, 256, 64, 127, true },
        { "pipeline-batch-gqa-fp16", GGML_TYPE_F16, 2, 4, 2, 17, 151, 64, 134, true },
        { "pipeline-trace-fp16", GGML_TYPE_F16, 1, 8, 2, 128, 256, 256, 128, true, true, 4 },
        { "pipeline-wrap-bf16-1c", GGML_TYPE_BF16, 1, 1, 1, 257, 641, 128, 384, true },
        { "pipeline-partial-fp16-4c", GGML_TYPE_F16, 1, 2, 1, 65, 151, 192, 86, true, true, 1, true },
        { "pipeline-wrap-bf16-4c", GGML_TYPE_BF16, 1, 1, 1, 1025, 1153, 64, 128, true, false, 1, true },
        { "pipeline-wide-fp16-4c", GGML_TYPE_F16, 1, 2, 1, 513, 641, 256, 0, false, true, 1, true },
        { "pipeline-rescale-bf16-1c", GGML_TYPE_BF16, 1, 1, 1, 129, 385, 128, 0, false, false, 1, false, 2.0f },
        { "pipeline-rescale-fp16-4c", GGML_TYPE_F16, 1, 1, 1, 257, 641, 64, 384, true, false, 1, true, 2.0f },
        { "pipeline-short-fp16-1c", GGML_TYPE_F16, 2, 4, 2, 17, 23, 64, 6, true, false, 4 },
        { "pipeline-state-reuse-bf16-1c", GGML_TYPE_BF16, 2, 1, 1, 257, 257, 256, 0, true },
        { "pipeline-state-reuse-fp16-4c", GGML_TYPE_F16, 2, 1, 1, 513, 641, 64, 128, true, false, 1, true },
    };

    int      result = 0;
    uint32_t seed   = 0x35a128U;
    bool     ran    = false;
    for (const attention_case & test : cases) {
        if (argc > 1 && std::strcmp(argv[1], test.name) != 0) {
            ++seed;
            continue;
        }
        ran = true;
        if (run_case(&runtime, test, seed++) != 0) {
            result = 1;
            break;
        }
    }
    free_runtime(&runtime);
    if (!ran) {
        return fail("unknown test case");
    }
    if (result != 0) {
        return result;
    }
    std::puts("test-dayan-attention-pipeline: PASS");
    return 0;
}

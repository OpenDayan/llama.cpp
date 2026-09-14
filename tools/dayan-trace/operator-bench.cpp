#include "ggml-dayan-kernels.h"
#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

static void check(int status, const std::string & operation) {
    if (status < 0) {
        throw std::runtime_error(operation + ": Dayan status " + std::to_string(status));
    }
}

static void wait_event(int status, npu_event_t * event, uint32_t timeout_ms, const std::string & operation) {
    check(status, operation);
    npu_event_info_t info{};
    status = npu_event_wait(event, timeout_ms, &info);
    const int destroy_status = npu_event_destroy(event);
    check(status, operation + " wait");
    check((int) info.status, operation + " completion");
    check(destroy_status, "npu_event_destroy");
}

struct device_context {
    npu_runtime_t * runtime = nullptr;
    npu_context_t * context = nullptr;
    npu_stream_t * stream = nullptr;
    std::vector<npu_buffer_t *> buffers;
    uint32_t timeout_ms = 0;

    ~device_context() {
        for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
            npu_buffer_destroy(*it);
        }
        if (stream)  { npu_stream_destroy(stream); }
        if (context) { npu_context_destroy(context); }
        if (runtime) { npu_runtime_close(runtime); }
    }

    npu_buffer_t * upload(const void * data, size_t bytes) {
        npu_buffer_desc_t desc;
        npu_buffer_desc_init(&desc, bytes);
        npu_buffer_t * buffer = nullptr;
        check(npu_buffer_create(context, &desc, &buffer), "npu_buffer_create(" + std::to_string(bytes) + " bytes)");
        buffers.push_back(buffer);
        npu_event_t * event = nullptr;
        int status = npu_memcpy_h2d_async(stream, buffer, 0, data, bytes, &event);
        wait_event(status, event, timeout_ms, "npu_memcpy_h2d_async(" + std::to_string(bytes) + " bytes)");
        return buffer;
    }

    npu_buffer_t * filled(size_t count, float value, bool bf16 = false) {
        if (bf16) {
            std::vector<ggml_bf16_t> values(count, ggml_fp32_to_bf16(value));
            return upload(values.data(), count * sizeof(values[0]));
        }
        std::vector<float> values(count, value);
        return upload(values.data(), count * sizeof(values[0]));
    }

    void download(npu_buffer_t * buffer, void * data, size_t bytes) {
        npu_event_t * event = nullptr;
        int status = npu_memcpy_d2h_async(stream, data, buffer, 0, bytes, &event);
        wait_event(status, event, timeout_ms, "npu_memcpy_d2h_async(" + std::to_string(bytes) + " bytes)");
    }
};

static uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void strides(uint64_t nb[4], uint32_t width, uint32_t heads, uint32_t tokens) {
    nb[0] = sizeof(float);
    nb[1] = nb[0] * width;
    nb[2] = nb[1] * heads;
    nb[3] = nb[2] * tokens;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " CASE.json RESULT.json TAG\n";
        return 1;
    }
    json result;
    int exit_status = 0;
    try {
        std::ifstream input(argv[1]);
        const json spec = json::parse(input);
        result["case"] = spec;
        result["samples"] = json::array();
        const std::string tag = argv[3];
        const std::string op = spec.at("op");
        const uint32_t m = spec.at("m");
        const uint32_t n = spec.value("n", 1024U);
        const uint32_t k = spec.value("k", 1024U);
        const int repeats = spec.value("repeats", 1);
        const int warmup = spec.value("warmup", 0);
        if (m == 0 || m > 1024 || n == 0 || n > 65536 || k == 0 || k > 65536 ||
            (uint64_t) n * k > 64U * 1024U * 1024U || repeats < 1 || warmup < 0 ||
            repeats + warmup > 100 || tag.empty() || tag.size() > 11) {
            throw std::runtime_error("invalid shape, repetition count, or tag");
        }
        device_context device;
        device.timeout_ms = spec.at("timeout_ms");
        npu_runtime_options_t options;
        npu_runtime_options_init(&options);
        options.default_timeout_ms = device.timeout_ms;
        const char * socket = std::getenv("DAYAN_SERVER_SOCKET");
        if (socket && socket[0]) { options.server_socket = socket; }
        check(npu_runtime_open(&device.runtime, &options), "npu_runtime_open");
        check(npu_context_create(device.runtime, nullptr, &device.context), "npu_context_create");
        check(npu_stream_create(device.context, nullptr, &device.stream), "npu_stream_create");
        npu_kernel_launch_attrs_t attrs;
        npu_kernel_launch_attrs_init(&attrs, device.context, device.stream, tag.c_str());
        attrs.config.task_dim = NPUTaskDim3{4, 1, 1};
        attrs.config.task_mode = NPUTaskMode::kSingleCore;
        attrs.timeout_ms = device.timeout_ms;

        npu_buffer_t * output = nullptr;
        npu_buffer_t * kernel_result = nullptr;
        std::vector<float> reference;
        bool lowp_output = false;
        std::function<int()> launch;
        const float nan = std::numeric_limits<float>::quiet_NaN();

        if (op == "matmul" || op == "ffn") {
            lowp_output = op == "ffn";
            attrs.config.task_mode = NPUTaskMode::kFourCore;
            DayanMatmulConfig config{m, n, k, k * 2ULL, k * 2ULL, n * (lowp_output ? 2ULL : 4ULL)};
            auto cfg = device.upload(&config, sizeof(config));
            auto a = device.filled((size_t) m * k, 1.0f / 64, true);
            auto b = device.filled((size_t) n * k, 1.0f / 64, true);
            output = device.filled((size_t) m * n, nan, lowp_output);
            float value = k / 4096.0f;
            if (lowp_output) {
                auto up = device.filled((size_t) n * k, 1.0f / 64, true);
                launch = [&, cfg, a, b, up]() {
                    return (m <= 512 ? matmul_swiglu_inference_bf16_4c : matmul_swiglu_bf16_4c)(cfg, a, b, up, output, &attrs);
                };
                value = value * value / (1.0f + std::exp(-value));
            } else {
                launch = [&, cfg, a, b]() {
                    return (m <= 512 ? matmul_inference_bf16_4c : matmul_bf16_4c)(cfg, a, b, output, &attrs);
                };
            }
            reference.assign((size_t) m * n, value);
        } else if (op == "ssm_conv") {
            DayanSSMConvConfig config{m + 3, 6144, m, 1, 4, 1, float_bits(-1.4426950408889634f), float_bits(1.0f)};
            attrs.config.task_dim = NPUTaskDim3{dayan_ssm_conv_task_count(config.channels, config.tokens, config.sequences), 1, 1};
            auto cfg = device.upload(&config, sizeof(config));
            auto a = device.filled((size_t) (m + 3) * config.channels, 1.0f / 64);
            auto b = device.filled(4 * config.channels, 1.0f / 32);
            output = device.filled((size_t) m * config.channels, nan);
            const float value = 4.0f / (64 * 32);
            reference.assign((size_t) m * config.channels, value / (1.0f + std::exp(-value)));
            launch = [&, cfg, a, b]() { return ssm_conv_f32_1c(cfg, a, b, output, &attrs); };
        } else if (op == "gdn") {
            DayanGatedDeltaNetConfig config{};
            config.head_dim = 128;
            config.heads = config.q_heads = config.k_heads = 16;
            config.tokens = m;
            config.sequences = config.snapshots = config.gate_width = config.q_sequences = config.k_sequences = 1;
            config.scale_bits = float_bits(1.0f / std::sqrt(128.0f));
            attrs.config.task_dim = NPUTaskDim3{std::min(config.heads * config.sequences, (uint32_t) kDayanGatedDeltaNetMaxTasks), 1, 1};
            for (auto nb : {config.q_nb, config.k_nb, config.v_nb}) { strides(nb, 128, 16, m); }
            for (auto nb : {config.g_nb, config.beta_nb}) { strides(nb, 1, 16, m); }
            auto cfg = device.upload(&config, sizeof(config));
            auto q = device.filled((size_t) m * 2048, 1.0f / 16);
            auto key = device.filled((size_t) m * 2048, 1.0f / 16);
            auto v = device.filled((size_t) m * 2048, 1.0f / 32);
            auto g = device.filled((size_t) m * 16, -0.1f);
            auto beta = device.filled((size_t) m * 16, 0.5f);
            auto state = device.filled(128 * 128 * 16, 0.0f);
            reference.resize((size_t) m * 2048 + 128 * 128 * 16);
            float s = 0.0f;
            for (uint32_t token = 0; token < m; ++token) {
                s *= std::exp(-0.1f);
                s += (1.0f / 32 - 128 * s / 16) * 0.5f / 16;
                std::fill_n(reference.begin() + token * 2048, 2048, 128 * s / (16 * std::sqrt(128.0f)));
            }
            std::fill(reference.begin() + m * 2048, reference.end(), s);
            output = device.filled(reference.size(), nan);
            launch = [&, cfg, q, key, v, g, beta, state]() {
                return gated_delta_net_f32_1c(cfg, q, key, v, g, beta, state, output, &attrs);
            };
        } else if (op == "attention") {
            DayanFlashAttentionConfig config{};
            config.batch = 1;
            config.q_heads = 8;
            config.kv_heads = 2;
            config.q_len = m;
            config.kv_len = spec.at("kv_len");
            if (config.kv_len < m || config.kv_len > 65536) { throw std::runtime_error("invalid kv_len"); }
            config.head_dim = 256;
            config.scale_bits = float_bits(1.0f / 16);
            config.causal_offset = config.kv_len - m;
            config.flags = 1;
            config.q_row_stride = config.k_row_stride = config.v_row_stride = 512;
            config.q_head_stride = m * 512ULL;
            config.q_batch_stride = 8 * config.q_head_stride;
            config.k_head_stride = config.v_head_stride = config.kv_len * 512ULL;
            config.k_batch_stride = config.v_batch_stride = 2 * config.k_head_stride;
            auto cfg = device.upload(&config, sizeof(config));
            auto q = device.filled((size_t) m * 8 * 256, 1.0f / 64, true);
            auto key = device.filled((size_t) config.kv_len * 2 * 256, 1.0f / 64, true);
            auto v = device.filled((size_t) config.kv_len * 2 * 256, 1.0f / 32, true);
            reference.assign((size_t) m * 8 * 256, 1.0f / 32);
            output = device.filled(reference.size(), nan);
            const int64_t initial_status = -1;
            kernel_result = device.upload(&initial_status, sizeof(initial_status));
            const bool split_kv = dayan_flash_attention_use_split_kv(m, config.kv_len);
            const bool four_core = m > 64 || split_kv;
            attrs.config.task_mode = four_core ? NPUTaskMode::kFourCore : NPUTaskMode::kSingleCore;
            launch = [&, cfg, q, key, v, four_core, split_kv]() {
                const auto kernel = split_kv ? flash_attention_decode_bf16_4c : four_core ? flash_attention_bf16_4c : flash_attention_bf16_1c;
                return kernel(cfg, q, key, v, output, kernel_result, &attrs);
            };
        } else if (op == "norm") {
            attrs.config.task_dim.x = std::min<uint32_t>(m, kDayanNativeMaxTasks);
            DayanNormPipelineConfig config{};
            config.norm.width = n;
            config.norm.rows = m;
            config.norm.epsilon_bits = float_bits(1e-6f);
            config.norm.inv_width_bits = float_bits(1.0f / (float) n);
            config.norm.input_ne[0] = n;
            config.norm.input_ne[1] = m;
            config.norm.input_ne[2] = config.norm.input_ne[3] = 1;
            strides(config.norm.input_nb, n, m, 1);
            config.flags = kDayanNormHasResidual | kDayanNormWriteResidual | kDayanNormWriteLowp;
            config.output_type = 1;
            auto cfg = device.upload(&config, sizeof(config));
            auto left = device.filled((size_t) m * n, 0.125f);
            auto right = device.filled((size_t) m * n, 0.0625f);
            auto scale = device.filled(n, 1.0f);
            auto residual = device.filled((size_t) m * n, nan);
            lowp_output = true;
            output = device.filled((size_t) m * n, nan, true);
            reference.assign((size_t) m * n, 0.1875f / std::sqrt(0.1875f * 0.1875f + 1e-6f));
            launch = [&, cfg, left, right, scale, residual]() {
                return rms_norm_pipeline_1c(cfg, left, right, scale, residual, residual, residual, output, &attrs);
            };
        } else {
            throw std::runtime_error("unknown operator: " + op);
        }

        for (int iteration = 0; iteration < warmup + repeats; ++iteration) {
            const std::string name = tag + (iteration < warmup ? "w" : "r") + std::to_string(iteration);
            std::memset(attrs.task_name, 0, sizeof(attrs.task_name));
            std::memcpy(attrs.task_name, name.c_str(), name.size());
            const auto start = std::chrono::steady_clock::now();
            const int status = launch();
            const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result["samples"].push_back({{"tag", name}, {"warmup", iteration < warmup}, {"wall_s", wall_s}, {"status", status}, {"tasks", attrs.config.task_dim.x * attrs.config.task_dim.y * attrs.config.task_dim.z}});
            check(status, "kernel launch " + name);
        }
        if (kernel_result) {
            int64_t status;
            device.download(kernel_result, &status, sizeof(status));
            if (status != 0) { throw std::runtime_error("attention result " + std::to_string(status)); }
        }
        std::vector<float> observed(reference.size());
        if (lowp_output) {
            std::vector<ggml_bf16_t> values(reference.size());
            device.download(output, values.data(), values.size() * sizeof(values[0]));
            for (size_t i = 0; i < values.size(); ++i) { observed[i] = ggml_bf16_to_fp32(values[i]); }
        } else {
            device.download(output, observed.data(), observed.size() * sizeof(observed[0]));
        }
        double max_abs_error = 0;
        for (size_t i = 0; i < reference.size(); ++i) {
            const double error = std::abs(observed[i] - reference[i]);
            if (!std::isfinite(observed[i]) || error > 1e-5 + 0.02 * std::abs(reference[i])) {
                throw std::runtime_error("reference mismatch at " + std::to_string(i) + ": observed=" + std::to_string(observed[i]) + ", expected=" + std::to_string(reference[i]));
            }
            max_abs_error = std::max(max_abs_error, error);
        }
        result["max_abs_error"] = max_abs_error;
        result["checked_elements"] = reference.size();
        result["validation"] = "constant-input analytical reference";
        result["status"] = "ok";
    } catch (const std::exception & error) {
        result["status"] = "failed";
        result["error"] = error.what();
        exit_status = 1;
    }
    std::ofstream output(argv[2]);
    output << result.dump(2) << '\n';
    std::cout << result.dump(2) << '\n';
    return output.good() ? exit_status : 1;
}

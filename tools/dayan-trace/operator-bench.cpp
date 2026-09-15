#include "ggml-dayan-kernels.h"
#include "ggml-dayan-mxfp4-layout.h"
#include "ggml-cpp.h"
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
#include <thread>
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
    bool compare = false;
    uint32_t seed = 1;
    std::vector<std::vector<float>> inputs;

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

    std::vector<float> values(size_t count, float value, bool positive = false) {
        std::vector<float> data(count, value);
        if (compare) {
            for (size_t i = 0; i < count; ++i) {
                seed = seed * 1664525U + 1013904223U;
                const float uniform = (seed >> 8) * (1.0f / 16777216);
                data[i] = positive ? value * (0.5f + uniform) :
                    (uniform * 2 - 1) * (value == 0 ? 0.02f : value * 2);
            }
        }
        return data;
    }

    npu_buffer_t * input(size_t count, float value, bool bf16 = false, bool positive = false) {
        auto data = values(count, value, positive);
        npu_buffer_t * buffer;
        if (bf16) {
            std::vector<ggml_bf16_t> encoded(count);
            for (size_t i = 0; i < count; ++i) {
                encoded[i] = ggml_fp32_to_bf16(data[i]);
                data[i] = ggml_bf16_to_fp32(encoded[i]);
            }
            buffer = upload(encoded.data(), encoded.size() * sizeof(encoded[0]));
        } else {
            buffer = upload(data.data(), data.size() * sizeof(data[0]));
        }
        if (compare) { inputs.push_back(std::move(data)); }
        return buffer;
    }

    npu_buffer_t * view(npu_buffer_t * buffer, size_t offset, size_t bytes) {
        npu_buffer_t * result = nullptr;
        check(npu_buffer_create_view(buffer, offset, bytes, &result), "npu_buffer_create_view");
        buffers.push_back(result);
        return result;
    }

    void download(npu_buffer_t * buffer, void * data, size_t bytes) {
        npu_event_t * event = nullptr;
        int status = npu_memcpy_d2h_async(stream, data, buffer, 0, bytes, &event);
        wait_event(status, event, timeout_ms, "npu_memcpy_d2h_async(" + std::to_string(bytes) + " bytes)");
    }

    void poison(npu_buffer_t * buffer, size_t bytes) {
        std::vector<uint8_t> data(bytes, 255);
        npu_event_t * event = nullptr;
        int status = npu_memcpy_h2d_async(stream, buffer, 0, data.data(), bytes, &event);
        wait_event(status, event, timeout_ms, "poison output");
    }
};

static float mxfp4_value(uint8_t scale, uint8_t code) {
    static const float values[] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
    return scale == 255 ? std::numeric_limits<float>::quiet_NaN() :
        std::ldexp(values[code & 7] * (code & 8 ? -1 : 1), (int) scale - 127);
}

static float mxfp4_constant(float value, uint8_t & scale, uint8_t & code) {
    value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
    int exponent = value == 0 ? 0 : -127;
    while (exponent < 127 && std::abs(value) > std::ldexp(6.0f, exponent)) { ++exponent; }
    scale = exponent + 127;
    code = 0;
    while (code < 7 && std::abs(value) >= mxfp4_value(scale, code + 1)) { ++code; }
    if (std::signbit(value)) { code |= 8; }
    return mxfp4_value(scale, code);
}

static std::vector<uint8_t> quantize_mxfp4_reference(std::vector<float> & data) {
    std::vector<uint8_t> logical(data.size() / 32 * 17, 0);
    for (size_t block = 0; block < data.size() / 32; ++block) {
        float largest = 0;
        for (size_t i = 0; i < 32; ++i) {
            float & value = data[block * 32 + i];
            value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
            largest = std::max(largest, std::abs(value));
        }
        uint8_t scale, unused;
        mxfp4_constant(largest, scale, unused);
        logical[block * 17] = scale;
        for (size_t i = 0; i < 32; ++i) {
            float & value = data[block * 32 + i];
            uint8_t code = 0;
            while (code < 7 && std::abs(value) >= mxfp4_value(scale, code + 1)) { ++code; }
            if (std::signbit(value)) { code |= 8; }
            logical[block * 17 + 1 + i % 16] |= code << (i / 16 * 4);
            value = mxfp4_value(scale, code);
        }
    }
    return logical;
}

static npu_buffer_t * mxfp4_input(device_context & device, const DayanMxfp4Layout & layout, float value) {
    auto data = device.values(layout.rows * layout.columns, value);
    if (device.compare) {
        for (size_t i = 0; i < data.size(); ++i) { data[i] = std::ldexp(data[i], (int) (i / 32 % 7) - 3); }
    }
    auto logical = quantize_mxfp4_reference(data);
    std::vector<uint8_t> packed(layout.bytes, 0);
    dayan_mxfp4_pack(layout, logical.data(), packed.data());
    if (device.compare) { device.inputs.push_back(std::move(data)); }
    return device.upload(packed.data(), packed.size());
}

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

static std::vector<float> cpu_reference(const json & spec, const std::vector<std::vector<float>> & inputs) {
    const std::string op = spec.at("op");
    if (op == "quantize") { return inputs.at(0); }
    const int64_t m = spec.at("m"), n = spec.value("n", 1024), k = spec.value("k", 1024);
    ggml_backend_load_all();
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cpu) { throw std::runtime_error("--cmp requires the GGML CPU backend"); }
    const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(cpu.get()));
    const auto set_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_threads) { set_threads(cpu.get(), std::max(1U, std::min(8U, std::thread::hardware_concurrency()))); }
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true}));
    if (!ctx) { throw std::runtime_error("cannot allocate CPU reference context"); }
    std::vector<ggml_tensor *> tensors;
    auto tensor = [&](int64_t d0, int64_t d1 = 1, int64_t d2 = 1, int64_t d3 = 1) {
        auto t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d0, d1, d2, d3);
        tensors.push_back(t);
        return t;
    };
    ggml_tensor * output = nullptr;
    ggml_tensor * mask = nullptr;
    if (op == "matmul" || op == "ffn") {
        auto a = tensor(k, m), b = tensor(k, n);
        output = ggml_mul_mat(ctx.get(), b, a);
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        if (op == "ffn") {
            auto up = ggml_mul_mat(ctx.get(), tensor(k, n), a);
            ggml_mul_mat_set_prec(up, GGML_PREC_F32);
            output = ggml_mul(ctx.get(), ggml_silu(ctx.get(), output), up);
        }
    } else if (op == "ssm_conv") {
        const int64_t channels = spec.value("channels", 6144), kernel = spec.value("kernel", 4);
        auto a = tensor(m + kernel - 1, channels), b = tensor(kernel, channels);
        output = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), a, b));
    } else if (op == "gdn") {
        const int64_t heads = spec.value("heads", 16), dim = spec.value("head_dim", 128);
        auto q = tensor(dim, heads, m), key = tensor(dim, heads, m), v = tensor(dim, heads, m);
        auto g = tensor(1, heads, m), beta = tensor(1, heads, m), state = tensor(dim, dim, heads);
        output = ggml_gated_delta_net(ctx.get(), q, key, v, g, beta, state, 1);
    } else if (op == "attention") {
        const int64_t heads = spec.value("q_heads", 8), kv_heads = spec.value("kv_heads", 2);
        const int64_t dim = spec.value("head_dim", 256), kv_len = spec.at("kv_len");
        auto q = tensor(dim, m, heads), key = tensor(dim, kv_len, kv_heads), v = tensor(dim, kv_len, kv_heads);
        mask = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, kv_len, m);
        output = ggml_flash_attn_ext(ctx.get(), q, key, v, mask, 1.0f / std::sqrt((float) dim), 0, 0);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
    } else if (op == "norm") {
        auto left = tensor(n, m), right = tensor(n, m), scale = tensor(n);
        output = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), ggml_add(ctx.get(), left, right), 1e-6f), scale);
    } else {
        throw std::runtime_error("no CPU reference for " + op);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()));
    if (!buffer || tensors.size() != inputs.size()) { throw std::runtime_error("invalid CPU reference inputs or allocation"); }
    for (size_t i = 0; i < tensors.size(); ++i) {
        if ((size_t) ggml_nelements(tensors[i]) != inputs[i].size()) { throw std::runtime_error("CPU reference input size mismatch"); }
        ggml_backend_tensor_set(tensors[i], inputs[i].data(), 0, ggml_nbytes(tensors[i]));
    }
    if (mask) {
        std::vector<ggml_fp16_t> values((size_t) ggml_nelements(mask));
        for (int64_t row = 0; row < m; ++row) {
            for (int64_t col = 0; col < mask->ne[0]; ++col) {
                values[row * mask->ne[0] + col] = ggml_fp32_to_fp16(col <= mask->ne[0] - m + row ? 0 : -INFINITY);
            }
        }
        ggml_backend_tensor_set(mask, values.data(), 0, ggml_nbytes(mask));
    }
    if (ggml_backend_graph_compute(cpu.get(), graph) != GGML_STATUS_SUCCESS) { throw std::runtime_error("CPU reference graph failed"); }
    std::vector<float> reference((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, reference.data(), 0, ggml_nbytes(output));
    return reference;
}

static json compare_output(const std::vector<float> & observed, const std::vector<float> & reference, double atol, double rtol,
                           const std::vector<uint8_t> & quantized = {}) {
    double max_abs = 0, max_rel = 0, squared_error = 0, squared_reference = 0;
    size_t mismatches = 0, nonfinite = 0;
    json first = nullptr;
    for (size_t i = 0; i < reference.size(); ++i) {
        const bool finite = std::isfinite(observed[i]) && std::isfinite(reference[i]);
        const double error = finite ? std::abs((double) observed[i] - reference[i]) : 0;
        if (!finite) { ++nonfinite; }
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(std::abs((double) reference[i]), 1e-12));
        squared_error += error * error;
        if (finite) { squared_reference += (double) reference[i] * reference[i]; }
        const double tolerance = quantized.empty() ? atol + rtol * std::abs(reference[i]) :
            std::ldexp(2.0, (int) quantized[i / 32 * 17] - 127);
        if (!finite || error > tolerance) {
            ++mismatches;
            if (first.is_null()) { first = {{"index", i}, {"observed", observed[i]}, {"expected", reference[i]}}; }
        }
    }
    return {{"status", mismatches == 0 ? "pass" : "fail"}, {"checked_elements", reference.size()},
            {"max_abs_error", max_abs}, {"max_rel_error", max_rel},
            {"relative_l2_error", std::sqrt(squared_error / std::max(squared_reference, 1e-30))},
            {"atol", atol}, {"rtol", rtol}, {"mismatched_elements", mismatches}, {"nonfinite_elements", nonfinite}, {"first_mismatch", first}};
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--capabilities") == 0) {
        std::cout << "{\"case_schema\":2,\"cmp\":true,\"dtypes\":[\"f32\",\"bf16\",\"mxfp4\"]}\n";
        return 0;
    }
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
        const std::string dtype = spec.value("dtype", op == "matmul" || op == "ffn" || op == "attention" ? "bf16" : "f32");
        const std::string output_dtype = spec.value("output_dtype", op == "ffn" || op == "norm" ? "bf16" : op == "quantize" ? "mxfp4" : "f32");
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
        const bool matrix = op == "matmul" || op == "ffn";
        const bool mx = dtype == "mxfp4";
        if ((matrix && (dtype != "bf16" && !mx)) ||
            (matrix && !mx && output_dtype != (op == "ffn" ? "bf16" : "f32")) ||
            (!matrix && dtype != (op == "attention" ? "bf16" : "f32")) ||
            (output_dtype != "f32" && output_dtype != "bf16" && output_dtype != "mxfp4") ||
            (mx && k % 32 != 0) || (output_dtype == "mxfp4" && n % 32 != 0) ||
            (op == "quantize" && output_dtype != "mxfp4") ||
            (op == "norm" && output_dtype != "f32" && output_dtype != "bf16") ||
            ((op == "gdn" || op == "ssm_conv" || op == "attention") && output_dtype != "f32")) {
            throw std::runtime_error("unsupported operator dtype or MXFP4 block alignment");
        }
        result["dtype"] = dtype;
        result["output_dtype"] = output_dtype;
        device_context device;
        device.compare = spec.value("cmp", false);
        device.seed = spec.value("seed", 1U);
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
        npu_buffer_t * residual_output = nullptr;
        std::vector<float> reference;
        std::vector<float> residual_reference;
        const bool lowp_output = output_dtype == "bf16";
        const bool mxfp4_output = output_dtype == "mxfp4";
        const auto output_layout = DayanMxfp4Layout::make(m, n);
        std::function<int()> launch;
        const float nan = std::numeric_limits<float>::quiet_NaN();

        if ((matrix && mx) || op == "quantize") {
            DayanMxfp4Config config{};
            config.m = m; config.n = n; config.k = k;
            config.output_stride = mxfp4_output ? output_layout.row_stride : n * (lowp_output ? 2ULL : 4ULL);
            config.output_scale_stride = output_layout.scale_stride;
            config.output_type = mxfp4_output ? kDayanDataTypeMXFP4 : lowp_output ? kDayanDataTypeBF16 : kDayanDataTypeF32;
            npu_buffer_t * scales = nullptr;
            if (mxfp4_output) {
                std::vector<uint8_t> sentinel(output_layout.bytes, 255);
                output = device.upload(sentinel.data(), sentinel.size());
                scales = device.view(output, output_layout.scale_offset, output_layout.bytes - output_layout.scale_offset);
            } else {
                output = device.filled((size_t) m * n, nan, lowp_output);
                scales = output;
            }
            float value = 1.0f / 64;
            if (op == "quantize") {
                config.source_type = kDayanDataTypeF32;
                config.a_stride = n * 4ULL;
                auto cfg = device.upload(&config, sizeof(config));
                auto data = device.values((size_t) m * n, value);
                if (device.compare) {
                    for (size_t i = 0; i < data.size(); ++i) { data[i] = std::ldexp(data[i], (int) (i / 32 % 13) - 6); }
                    device.inputs.push_back(data);
                }
                auto a = device.upload(data.data(), data.size() * sizeof(data[0]));
                result["kernel"] = "quantize_mxfp4";
                launch = [&, cfg, a, scales]() { return quantize_mxfp4(cfg, a, output, scales, &attrs); };
            } else {
                attrs.config.task_mode = NPUTaskMode::kFourCore;
                const auto a_layout = DayanMxfp4Layout::make(m, k), b_layout = DayanMxfp4Layout::make(n, k);
                config.source_type = config.weight_type = kDayanDataTypeMXFP4;
                config.a_stride = a_layout.row_stride; config.b_stride = b_layout.row_stride;
                config.a_scale_stride = a_layout.scale_stride; config.b_scale_stride = b_layout.scale_stride;
                config.operation = op == "ffn" ? 2 : 0;
                auto cfg = device.upload(&config, sizeof(config));
                auto a = mxfp4_input(device, a_layout, value), b = mxfp4_input(device, b_layout, value);
                auto as = device.view(a, a_layout.scale_offset, a_layout.bytes - a_layout.scale_offset);
                auto bs = device.view(b, b_layout.scale_offset, b_layout.bytes - b_layout.scale_offset);
                auto up = op == "ffn" ? mxfp4_input(device, b_layout, value) : b;
                auto ups = op == "ffn" ? device.view(up, b_layout.scale_offset, b_layout.bytes - b_layout.scale_offset) : bs;
                result["kernel"] = "matmul_mxfp4_4c";
                launch = [&, cfg, a, as, b, bs, up, ups, scales]() {
                    return matmul_mxfp4_4c(cfg, a, as, b, bs, up, ups, output, scales, &attrs);
                };
                value = k / 4096.0f;
                if (op == "ffn") { value = value * value / (1.0f + std::exp(-value)); }
            }
            if (mxfp4_output) {
                uint8_t scale, code;
                value = mxfp4_constant(value, scale, code);
            }
            reference.assign((size_t) m * n, value);
        } else if (matrix) {
            attrs.config.task_mode = NPUTaskMode::kFourCore;
            DayanMatmulConfig config{m, n, k, k * 2ULL, k * 2ULL, n * (lowp_output ? 2ULL : 4ULL)};
            auto cfg = device.upload(&config, sizeof(config));
            auto a = device.input((size_t) m * k, 1.0f / 64, true);
            auto b = device.input((size_t) n * k, 1.0f / 64, true);
            output = device.filled((size_t) m * n, nan, lowp_output);
            float value = k / 4096.0f;
            if (lowp_output) {
                auto up = device.input((size_t) n * k, 1.0f / 64, true);
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
            const uint32_t channels = spec.value("channels", 6144U), kernel = spec.value("kernel", 4U);
            if (channels == 0 || channels > 65536 || kernel == 0 || kernel > 63) { throw std::runtime_error("invalid convolution shape"); }
            DayanSSMConvConfig config{m + kernel - 1, channels, m, 1, kernel, 1, float_bits(-1.4426950408889634f), float_bits(1.0f)};
            attrs.config.task_dim = NPUTaskDim3{dayan_ssm_conv_task_count(config.channels, config.tokens, config.sequences), 1, 1};
            auto cfg = device.upload(&config, sizeof(config));
            auto a = device.input((size_t) config.input_length * config.channels, 1.0f / 64);
            auto b = device.input(kernel * config.channels, 1.0f / 32);
            output = device.filled((size_t) m * config.channels, nan);
            const float value = kernel / (64.0f * 32);
            reference.assign((size_t) m * config.channels, value / (1.0f + std::exp(-value)));
            launch = [&, cfg, a, b]() { return ssm_conv_f32_1c(cfg, a, b, output, &attrs); };
        } else if (op == "gdn") {
            DayanGatedDeltaNetConfig config{};
            config.head_dim = spec.value("head_dim", 128U);
            config.heads = config.q_heads = config.k_heads = spec.value("heads", 16U);
            if (config.head_dim != 128 || config.heads == 0 || config.heads > 128 || spec.value("snapshots", 1U) != 1) {
                throw std::runtime_error("GDN needs head_dim=128, 1..128 heads, and one snapshot");
            }
            const size_t width = config.head_dim * config.heads, state_elements = width * config.head_dim;
            config.tokens = m;
            config.sequences = config.snapshots = config.gate_width = config.q_sequences = config.k_sequences = 1;
            config.scale_bits = float_bits(1.0f / std::sqrt(128.0f));
            attrs.config.task_dim = NPUTaskDim3{std::min(config.heads * config.sequences, (uint32_t) kDayanGatedDeltaNetMaxTasks), 1, 1};
            for (auto nb : {config.q_nb, config.k_nb, config.v_nb}) { strides(nb, 128, config.heads, m); }
            for (auto nb : {config.g_nb, config.beta_nb}) { strides(nb, 1, config.heads, m); }
            auto cfg = device.upload(&config, sizeof(config));
            auto q = device.input(m * width, 1.0f / 16);
            auto key = device.input(m * width, 1.0f / 16);
            auto v = device.input(m * width, 1.0f / 32);
            auto g = device.input((size_t) m * config.heads, -0.1f, false, true);
            auto beta = device.input((size_t) m * config.heads, 0.5f, false, true);
            auto state = device.input(state_elements, 0.0f);
            reference.resize(m * width + state_elements);
            float s = 0.0f;
            for (uint32_t token = 0; token < m; ++token) {
                s *= std::exp(-0.1f);
                s += (1.0f / 32 - 128 * s / 16) * 0.5f / 16;
                std::fill_n(reference.begin() + token * width, width, 128 * s / (16 * std::sqrt(128.0f)));
            }
            std::fill(reference.begin() + m * width, reference.end(), s);
            output = device.filled(reference.size(), nan);
            launch = [&, cfg, q, key, v, g, beta, state]() {
                return gated_delta_net_f32_1c(cfg, q, key, v, g, beta, state, output, &attrs);
            };
        } else if (op == "attention") {
            DayanFlashAttentionConfig config{};
            config.batch = 1;
            config.q_heads = spec.value("q_heads", 8U);
            config.kv_heads = spec.value("kv_heads", 2U);
            config.q_len = m;
            config.kv_len = spec.at("kv_len");
            if (config.kv_len < m || config.kv_len > 65536) { throw std::runtime_error("invalid kv_len"); }
            config.head_dim = spec.value("head_dim", 256U);
            if (config.head_dim != 256 || config.q_heads == 0 || config.q_heads > 128 || config.kv_heads == 0 || config.q_heads % config.kv_heads != 0) {
                throw std::runtime_error("attention needs head_dim=256 and 1..128 query heads divisible by KV heads");
            }
            config.scale_bits = float_bits(1.0f / 16);
            config.causal_offset = config.kv_len - m;
            config.flags = 1;
            config.q_row_stride = config.k_row_stride = config.v_row_stride = 512;
            config.q_head_stride = m * 512ULL;
            config.q_batch_stride = config.q_heads * config.q_head_stride;
            config.k_head_stride = config.v_head_stride = config.kv_len * 512ULL;
            config.k_batch_stride = config.v_batch_stride = config.kv_heads * config.k_head_stride;
            auto cfg = device.upload(&config, sizeof(config));
            auto q = device.input((size_t) m * config.q_heads * 256, 1.0f / 64, true);
            auto key = device.input((size_t) config.kv_len * config.kv_heads * 256, 1.0f / 64, true);
            auto v = device.input((size_t) config.kv_len * config.kv_heads * 256, 1.0f / 32, true);
            reference.assign((size_t) m * config.q_heads * 256, 1.0f / 32);
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
            config.flags = kDayanNormHasResidual | kDayanNormWriteResidual | (lowp_output ? kDayanNormWriteLowp : kDayanNormWriteScaled);
            config.output_type = lowp_output ? kDayanDataTypeBF16 : kDayanDataTypeF32;
            auto cfg = device.upload(&config, sizeof(config));
            auto left = device.input((size_t) m * n, 0.125f);
            auto right = device.input((size_t) m * n, 0.0625f);
            auto scale = device.input(n, 1.0f, false, true);
            auto residual = device.filled((size_t) m * n, nan);
            residual_output = residual;
            output = device.filled((size_t) m * n, nan, lowp_output);
            reference.assign((size_t) m * n, 0.1875f / std::sqrt(0.1875f * 0.1875f + 1e-6f));
            launch = [&, cfg, left, right, scale, residual]() {
                return rms_norm_pipeline_1c(cfg, left, right, scale, residual, residual, lowp_output ? residual : output, output, &attrs);
            };
        } else {
            throw std::runtime_error("unknown operator: " + op);
        }

        std::vector<uint8_t> reference_bytes;
        if (device.compare) {
            const auto start = std::chrono::steady_clock::now();
            reference = cpu_reference(spec, device.inputs);
            if (mxfp4_output) { reference_bytes = quantize_mxfp4_reference(reference); }
            if (lowp_output) {
                for (float & value : reference) { value = ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)); }
            }
            if (residual_output) {
                residual_reference.resize(reference.size());
                for (size_t i = 0; i < reference.size(); ++i) { residual_reference[i] = device.inputs[0][i] + device.inputs[1][i]; }
            }
            result["reference_wall_s"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result["input_seed"] = spec.value("seed", 1U);
        }
        result["validation"] = device.compare ? "GGML CPU F32 reference with stored input values and device output conversion" : "constant-input analytical reference";
        result["comparisons"] = json::array();
        result["max_abs_error"] = 0.0;
        for (int iteration = 0; iteration < warmup + repeats; ++iteration) {
            const std::string name = tag + (iteration < warmup ? "w" : "r") + std::to_string(iteration);
            std::memset(attrs.task_name, 0, sizeof(attrs.task_name));
            std::memcpy(attrs.task_name, name.c_str(), name.size());
            if (device.compare && iteration > 0) {
                device.poison(output, mxfp4_output ? output_layout.bytes : reference.size() * (lowp_output ? 2 : 4));
                if (residual_output) { device.poison(residual_output, residual_reference.size() * sizeof(float)); }
                if (kernel_result) { device.poison(kernel_result, sizeof(int64_t)); }
            }
            const auto start = std::chrono::steady_clock::now();
            const int status = launch();
            const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            result["samples"].push_back({{"tag", name}, {"warmup", iteration < warmup}, {"wall_s", wall_s}, {"status", status}, {"tasks", attrs.config.task_dim.x * attrs.config.task_dim.y * attrs.config.task_dim.z}});
            check(status, "kernel launch " + name);
            if (!device.compare && iteration + 1 < warmup + repeats) { continue; }
            if (kernel_result) {
                int64_t status;
                device.download(kernel_result, &status, sizeof(status));
                if (status != 0) { throw std::runtime_error("attention result " + std::to_string(status)); }
            }
            std::vector<float> observed(reference.size());
            size_t byte_mismatches = 0;
            if (mxfp4_output) {
                std::vector<uint8_t> packed(output_layout.bytes), logical(reference.size() / 32 * 17);
                device.download(output, packed.data(), packed.size());
                dayan_mxfp4_unpack(output_layout, packed.data(), logical.data());
                if (device.compare) {
                    for (size_t i = 0; i < logical.size(); ++i) { byte_mismatches += logical[i] != reference_bytes[i]; }
                }
                for (size_t i = 0; i < observed.size(); ++i) {
                    const size_t block = i / 32 * 17;
                    const uint8_t code = (logical[block + 1 + i % 16] >> (i % 32 / 16 * 4)) & 15;
                    observed[i] = mxfp4_value(logical[block], code);
                }
            } else if (lowp_output) {
                std::vector<ggml_bf16_t> values(reference.size());
                device.download(output, values.data(), values.size() * sizeof(values[0]));
                for (size_t i = 0; i < values.size(); ++i) { observed[i] = ggml_bf16_to_fp32(values[i]); }
            } else {
                device.download(output, observed.data(), observed.size() * sizeof(observed[0]));
            }
            double atol = 1e-5, rtol = 0.02, l2_limit = 0.02;
            if (device.compare) {
                atol = 1e-6;
                rtol = lowp_output ? 0.01 : 0.001;
                l2_limit = rtol;
                if (op == "attention") { atol = 2e-5; rtol = l2_limit = 0.003; }
                if (op == "gdn") { atol = m >= 16 ? 2e-4 : 1e-6; rtol = l2_limit = m >= 16 ? 0.003 : 1e-4; }
                if (mxfp4_output) { l2_limit = 0.01; }
            }
            auto comparison = compare_output(observed, reference, atol, rtol, device.compare ? reference_bytes : std::vector<uint8_t>{});
            if (device.compare) {
                comparison["relative_l2_limit"] = l2_limit;
                if (comparison["relative_l2_error"].get<double>() > l2_limit) { comparison["status"] = "fail"; }
            }
            if (device.compare && mxfp4_output) {
                comparison["mismatched_bytes"] = byte_mismatches;
                if (op == "quantize") {
                    comparison["criterion"] = "exact MXFP4 payload and scale bytes";
                    comparison["status"] = byte_mismatches == 0 && comparison["nonfinite_elements"] == 0 ? "pass" : "fail";
                } else {
                    comparison["criterion"] = "finite values, each error <= 2 * its E8M0 scale, relative L2 <= 0.01; activation approximation can cross FP4 rounding thresholds";
                }
            }
            if (device.compare && residual_output) {
                std::vector<float> residual(residual_reference.size());
                device.download(residual_output, residual.data(), residual.size() * sizeof(residual[0]));
                comparison["residual"] = compare_output(residual, residual_reference, 1e-6, 1e-5);
                if (comparison["residual"]["status"] != "pass") { comparison["status"] = "fail"; }
            }
            comparison["tag"] = name;
            result["max_abs_error"] = std::max(result["max_abs_error"].get<double>(), comparison["max_abs_error"].get<double>());
            result["checked_elements"] = reference.size() + residual_reference.size();
            result["comparisons"].push_back(comparison);
            result["comparison"] = comparison;
            if (comparison["status"] != "pass") { throw std::runtime_error("reference mismatch: " + comparison.dump()); }
        }
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

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-dayan.h"
#include "ggml-dayan-mxfp4-layout.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) { std::fprintf(stderr, "MXFP4: %s\n", message); std::exit(1); }
}

static std::vector<float> decode(const std::vector<uint8_t> & bytes) {
    static const float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    std::vector<float> result(bytes.size() / 17 * 32);
    for (size_t b = 0; b < bytes.size() / 17; ++b) {
        for (int i = 0; i < 32; ++i) {
            const int code = (bytes[b * 17 + 1 + i % 16] >> (i / 16 * 4)) & 15;
            result[b * 32 + i] = std::ldexp(values[code & 7] * (code & 8 ? -1 : 1), (int) bytes[b * 17] - 127);
        }
    }
    return result;
}

static std::vector<uint8_t> quantize(const std::vector<float> & input) {
    static const float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    std::vector<uint8_t> result(input.size() / 32 * 17, 0);
    for (size_t b = 0; b < input.size() / 32; ++b) {
        uint16_t bits[32];
        float largest = 0;
        for (int i = 0; i < 32; ++i) {
            bits[i] = ggml_fp32_to_bf16(input[b * 32 + i]).bits;
            const int exp = (bits[i] >> 7) & 255;
            if (exp > 0 && exp < 255) { largest = std::max(largest, std::fabs(ggml_bf16_to_fp32({bits[i]}))); }
        }
        int exponent = 0;
        if (largest > 0) {
            exponent = -127;
            while (exponent < 127 && largest > std::ldexp(6.0f, exponent)) { ++exponent; }
        }
        result[b * 17] = exponent + 127;
        for (int i = 0; i < 32; ++i) {
            const int exp = (bits[i] >> 7) & 255;
            const float magnitude = std::ldexp(std::fabs(ggml_bf16_to_fp32({bits[i]})), -exponent);
            int code = exp == 255 ? 7 : 0;
            if (exp > 0 && exp < 255) {
                while (code < 7 && magnitude >= values[code + 1]) { ++code; }
            }
            code |= (bits[i] >> 12) & 8;
            result[b * 17 + 1 + i % 16] |= code << (i / 16 * 4);
        }
    }
    return result;
}

static std::vector<float> values(size_t count, int seed) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = std::ldexp(std::sin(float(i * 7 + seed) * .17f) * .7f, (int) ((i / 32 + seed) % 11) - 5);
    }
    return result;
}

static void layout_test() {
    const auto mm = DayanMxfp4Tile::make(1024, 3584, 1024, true, 1, false);
    const auto converted = DayanMxfp4Tile::make(1024, 3584, 1024, true, 1, true);
    const auto ffn = DayanMxfp4Tile::make(1024, 3584, 1024, true, 2, true);
    require(mm.m == 512 && mm.n == 512 && mm.k == 640, "FP4 demo MatMul tile");
    require(converted.m == 512 && converted.n == 512 && converted.k == 512, "aligned converted MatMul tile");
    require(ffn.m == 256 && ffn.n == 512 && ffn.k == 512, "large FFN tile");
    require(mm.bytes() > DayanMxfp4Tile::uram_bytes * .99 && converted.bytes() > DayanMxfp4Tile::uram_bytes * .97 && ffn.bytes() > DayanMxfp4Tile::uram_bytes * .95, "URAM utilization");
    for (int rows : {1, 9, 127, 128, 257, 513, 1024}) {
        for (int columns : {16, 256, 544, 1024, 3584, 6144, 65536}) {
            for (bool mx : {false, true}) {
                for (int products : {1, 2}) {
                    const auto tile = DayanMxfp4Tile::make(rows, columns, 3584, mx, products, true);
                    require(tile.m > 0 && tile.m % 32 == 0 && tile.m <= (products == 2 ? 256 : 512) && tile.n % 256 == 0 && tile.k >= 512 && tile.k % 128 == 0, "tile alignment");
                    require(tile.bytes() <= DayanMxfp4Tile::uram_bytes, "tile fits URAM");
                    require(tile.accumulator_offset() % 1024 == 0 && tile.scratch_offset() % 32 == 0 && tile.a_scale_stride() % 64 == 0 && tile.bytes() % 32 == 0, "buffer alignment");
                }
            }
        }
    }
    for (int rows : {1, 7, 8, 9, 65}) {
        for (int cols : {32, 128, 160, 672}) {
            const auto layout = DayanMxfp4Layout::make(rows, cols);
            require(layout.row_stride % 64 == 0 && layout.scale_stride % 32 == 0 && layout.scale_offset % 32 == 0 && layout.bytes % 32 == 0, "device payload and scale alignment");
            std::vector<uint8_t> original(rows * cols / 32 * 17), device(layout.bytes, 0), actual(original.size());
            for (size_t i = 0; i < original.size(); ++i) { original[i] = i * 113 + 7; }
            dayan_mxfp4_pack(layout, original.data(), device.data());
            for (int r = 0; r < rows; ++r) {
                for (int b = 0; b < cols / 32; ++b) {
                    const size_t z = (b / 4 * ((rows + 7) / 8) + r / 8) * 32 + r % 8 * 4 + b % 4;
                    require(device[layout.scale_offset + z] == original[(r * cols / 32 + b) * 17], "scale Z layout");
                }
            }
            dayan_mxfp4_unpack(layout, device.data(), actual.data());
            require(actual == original, "layout round trip");
        }
    }
    std::puts("MXFP4 host layout PASS");
}

static ggml_context_ptr context() {
    return ggml_context_ptr(ggml_init({ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false), nullptr, true}));
}

static std::vector<uint8_t> get_bytes(ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

static void conversion_test(ggml_backend_t backend, ggml_type type, int rows, int cols) {
    auto ctx = context();
    auto * input = ggml_new_tensor_2d(ctx.get(), type, cols, rows);
    auto * output = ggml_cast(ctx.get(), input, GGML_TYPE_MXFP4);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(buffer != nullptr, "allocate conversion");
    for (int repeat = 0; repeat < 2; ++repeat) {
        auto data = values(rows * cols, repeat + 9);
        if (type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(input, data.data(), 0, data.size() * 4);
        } else {
            std::vector<uint16_t> packed(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                packed[i] = type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(data[i]).bits : ggml_fp32_to_fp16(data[i]);
                data[i] = type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32({packed[i]}) : ggml_fp16_to_fp32(packed[i]);
            }
            ggml_backend_tensor_set(input, packed.data(), 0, packed.size() * 2);
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "convert execution");
        const auto expected = quantize(data), actual = get_bytes(output);
        if (actual != expected) {
            for (size_t i = 0; i < actual.size(); ++i) {
                if (actual[i] != expected[i]) {
                    std::fprintf(stderr, "convert %s %dx%d byte %zu: %02x != %02x\n", ggml_type_name(type), rows, cols, i, actual[i], expected[i]); break;
                }
            }
            require(false, "VEC quantization reference");
        }
        auto modified = actual;
        modified[19] ^= 0x39;
        ggml_backend_tensor_set(output, modified.data() + 19, 19, 1);
        require(get_bytes(output) == modified, "partial upload");
        uint8_t partial[13];
        ggml_backend_tensor_get(output, partial, 11, sizeof(partial));
        require(std::memcmp(partial, modified.data() + 11, sizeof(partial)) == 0, "partial readback");
        auto copy_ctx = context();
        auto * copy = ggml_new_tensor_2d(copy_ctx.get(), GGML_TYPE_MXFP4, cols, rows);
        ggml_backend_buffer_ptr copy_buffer(ggml_backend_alloc_ctx_tensors(copy_ctx.get(), backend));
        require(copy_buffer != nullptr, "allocate copy");
        ggml_backend_tensor_copy(output, copy);
        require(get_bytes(copy) == modified, "device copy includes scales");
        ggml_backend_tensor_memset(copy, 13, 5, 7);
        std::fill(modified.begin() + 5, modified.begin() + 12, 13);
        require(get_bytes(copy) == modified, "partial memset");
    }
    std::printf("MXFP4 VEC %s %dx%d PASS\n", ggml_type_name(type), rows, cols);
}

static std::vector<float> matmul(const std::vector<float> & a, const std::vector<float> & b, int m, int n, int k) {
    std::vector<float> result(m * n);
    for (int r = 0; r < m; ++r) {
        for (int c = 0; c < n; ++c) {
            double sum = 0;
            for (int i = 0; i < k; ++i) { sum += (double) a[r * k + i] * b[c * k + i]; }
            result[r * n + c] = sum;
        }
    }
    return result;
}

static void matmul_test(ggml_backend_t backend, int m, int n, int k, bool ffn,
                       ggml_type weight_type = GGML_TYPE_MXFP4, ggml_type output_type = GGML_TYPE_F32, bool rewrite_input = true, int activation = 0) {
    auto ctx = context();
    auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    auto * b = ggml_new_tensor_2d(ctx.get(), weight_type, k, n);
    auto * gate = ggml_mul_mat(ctx.get(), b, a);
    auto * up_weight = ffn ? ggml_new_tensor_2d(ctx.get(), weight_type, k, n) : nullptr;
    auto * up = ffn ? ggml_mul_mat(ctx.get(), up_weight, a) : nullptr;
    auto * intermediate = ffn ? (activation == 3 ? ggml_geglu_split(ctx.get(), gate, up) : ggml_swiglu_split(ctx.get(), gate, up)) : activation == 1 ? ggml_silu(ctx.get(), gate) : gate;
    if (ffn) { output_type = GGML_TYPE_MXFP4; }
    auto * output = output_type != GGML_TYPE_F32 ? ggml_cast(ctx.get(), intermediate, output_type) : intermediate;
    ggml_set_input(a);
    ggml_set_input(b);
    if (up_weight) { ggml_set_input(up_weight); }
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, output);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using rewrite_fn = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);
    auto rewrite = (rewrite_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    if (rewrite_input) {
        require(rewrite && rewrite(backend, ctx.get(), graph), "graph rewrite");
        require(gate->src[1]->type == weight_type, "activation cast inserted");
    }
    if (ffn) { require(up->src[1] == gate->src[1], "shared activation cast"); }
    ggml_gallocr_ptr allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    require(ggml_gallocr_alloc_graph(allocator.get(), graph), "allocate matmul with reuse");
    auto encode = [weight_type](const std::vector<float> & input) {
        if (weight_type == GGML_TYPE_MXFP4) { return quantize(input); }
        std::vector<uint8_t> bytes(input.size() * 2);
        for (size_t i = 0; i < input.size(); ++i) {
            uint16_t value = weight_type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(input[i]).bits : ggml_fp32_to_fp16(input[i]);
            std::memcpy(bytes.data() + i * 2, &value, 2);
        }
        return bytes;
    };
    auto unpack = [weight_type](const std::vector<uint8_t> & input) {
        if (weight_type == GGML_TYPE_MXFP4) { return decode(input); }
        std::vector<float> result(input.size() / 2);
        for (size_t i = 0; i < result.size(); ++i) {
            uint16_t value; std::memcpy(&value, input.data() + i * 2, 2);
            // MAT's FP16 input path discards the low three fraction bits.
            result[i] = weight_type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32({value}) : ggml_fp16_to_fp32(value & 0xfff8u);
        }
        return result;
    };
    const auto weights = encode(values(n * k, 23)), ups = encode(values(n * k, 71));
    for (int repeat = 0; repeat < 2; ++repeat) {
        // Graph allocation can reuse an input's storage after its last consumer.
        ggml_backend_tensor_set(b, weights.data(), 0, weights.size());
        require(get_bytes(b) == weights, "weight round trip");
        if (ffn) { ggml_backend_tensor_set(up_weight, ups.data(), 0, ups.size()); }
        const auto data = values(m * k, 31 + repeat);
        ggml_backend_tensor_set(a, data.data(), 0, data.size() * 4);
        auto reference = matmul(unpack(encode(data)), unpack(weights), m, n, k);
        if (ffn) {
            const auto u = matmul(unpack(encode(data)), unpack(ups), m, n, k);
            for (size_t i = 0; i < reference.size(); ++i) {
                const float x = reference[i];
                reference[i] = activation == 3 ? .5f * x * (1 + std::tanh(.7978845608028654f * x * (1 + .044715f * x * x))) * u[i] : x / (1 + std::exp(-x)) * u[i];
            }
        } else if (activation == 1) {
            for (float & x : reference) { x /= 1 + std::exp(-x); }
        }
        if (output_type == GGML_TYPE_MXFP4) { reference = decode(quantize(reference)); }
        if (output_type == GGML_TYPE_BF16 || output_type == GGML_TYPE_F16) {
            for (float & value : reference) {
                value = output_type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(value)) : ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
            }
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "matmul execution");
        std::vector<float> actual(reference.size());
        if (output_type == GGML_TYPE_MXFP4) { actual = decode(get_bytes(output)); }
        else if (output_type == GGML_TYPE_F32) { ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * 4); }
        else {
            std::vector<uint16_t> bytes(actual.size());
            ggml_backend_tensor_get(output, bytes.data(), 0, bytes.size() * 2);
            for (size_t i = 0; i < actual.size(); ++i) {
                actual[i] = output_type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32({bytes[i]}) : ggml_fp16_to_fp32(bytes[i]);
            }
        }
        double error = 0, norm = 0;
        for (size_t i = 0; i < actual.size(); ++i) {
            require(std::isfinite(actual[i]), "finite output");
            error += std::pow(actual[i] - reference[i], 2); norm += std::pow(reference[i], 2);
        }
        std::printf("MXFP4 %s %s -> %s %dx%dx%d repeat=%d relative L2=%g\n", ffn ? (activation == 3 ? "GeGLU" : "SwiGLU") : activation == 1 ? "SiLU" : "MatMul", ggml_type_name(weight_type), ggml_type_name(output_type), m, n, k, repeat, std::sqrt(error / norm));
        if (error > norm * (ffn || activation ? 0.0001 : 1e-10)) {
            int printed = 0;
            for (size_t i = 0; i < actual.size() && printed < 16; ++i) {
                if (std::fabs(actual[i] - reference[i]) > .001f * (1 + std::fabs(reference[i]))) {
                    std::fprintf(stderr, "  row=%zu col=%zu expected=%g actual=%g\n", i / n, i % n, reference[i], actual[i]);
                    ++printed;
                }
            }
        }
        require(error <= norm * (ffn || activation ? 0.0001 : 1e-10), "matmul numerical reference");
    }
}

static void get_rows_rewrite_test(ggml_backend_t backend, ggml_type type) {
    auto ctx = context();
    auto embedding = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_MXFP4, 160, 19);
    auto indices = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 9);
    auto selected = ggml_get_rows(ctx.get(), embedding, indices);
    auto scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 160);
    auto norm = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), selected, 1e-6f), scale);
    auto weight = ggml_new_tensor_2d(ctx.get(), type, 160, 32);
    auto output = ggml_mul_mat(ctx.get(), weight, norm);
    auto graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, output);
    auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using rewrite_fn = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);
    auto rewrite = (rewrite_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    require(rewrite && rewrite(backend, ctx.get(), graph), "embedding graph rewrite");
    require(selected->type == type && ggml_is_contiguous(selected), "MXFP4 embedding keeps norm storage type");
    require(embedding->type == GGML_TYPE_MXFP4 && ggml_backend_supports_op(backend, selected), "MXFP4 embedding weight stays quantized");
}

static void get_rows_test(ggml_backend_t backend, int width, ggml_type output_type, bool wide_indices) {
    auto ctx = context();
    auto source = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_MXFP4, width, 19, 2);
    auto indices = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 9, 2);
    auto output = ggml_get_rows(ctx.get(), source, indices);
    output->type = output_type;
    output->nb[0] = ggml_type_size(output_type);
    for (int i = 1; i < 4; ++i) { output->nb[i] = output->nb[i - 1] * output->ne[i - 1]; }
    if (wide_indices) {
        indices->type = GGML_TYPE_I64;
        indices->nb[0] = sizeof(int64_t);
        for (int i = 1; i < 4; ++i) { indices->nb[i] = indices->nb[i - 1] * indices->ne[i - 1]; }
    }
    ggml_set_output(output);
    require(ggml_backend_supports_op(backend, output), "MXFP4 GET_ROWS support");
    auto graph = ggml_new_graph_custom(ctx.get(), 128, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(buffer != nullptr, "GET_ROWS buffer");
    auto data = quantize(values((size_t) width * 38, 41));
    const auto decoded = decode(data);
    ggml_backend_tensor_set(source, data.data(), 0, data.size());
    for (int round = 0; round < 2; ++round) {
        std::vector<int32_t> ids(18);
        std::vector<int64_t> ids64(18);
        for (size_t i = 0; i < ids.size(); ++i) { ids64[i] = ids[i] = (i * 7 + round * 11) % 19; }
        ggml_backend_tensor_set(indices, wide_indices ? (void *) ids64.data() : (void *) ids.data(), 0, ggml_nbytes(indices));
        ggml_backend_tensor_memset(output, 255, 0, ggml_nbytes(output));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "GET_ROWS graph");
        const auto bytes = get_bytes(output);
        for (size_t i = 0; i < (size_t) width * ids.size(); ++i) {
            const size_t row = i / width;
            float expected = decoded[((row / 9) * 19 + ids[row]) * width + i % width];
            float actual;
            if (output_type == GGML_TYPE_F32) { std::memcpy(&actual, bytes.data() + i * 4, 4); }
            else {
                uint16_t value;
                std::memcpy(&value, bytes.data() + i * 2, 2);
                actual = output_type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32({value}) : ggml_fp16_to_fp32(value);
                expected = output_type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(ggml_fp32_to_bf16(expected)) : ggml_fp16_to_fp32(ggml_fp32_to_fp16(expected));
            }
            if (actual != expected) { std::fprintf(stderr, "GET_ROWS width=%d type=%s index=%zu actual=%g expected=%g\n", width, ggml_type_name(output_type), i, actual, expected); }
            require(actual == expected, "GET_ROWS exact dequantization");
        }
    }
    std::printf("MXFP4 GET_ROWS width=%d type=%s indices=%s elements=%d PASS\n", width, ggml_type_name(output_type), wide_indices ? "i64" : "i32", width * 18);
}

int main(int argc, char ** argv) {
    layout_test();
    if (argc > 1 && std::strcmp(argv[1], "--host-only") == 0) { return 0; }
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name("DAYAN", nullptr));
    require(backend != nullptr, "initialize backend");
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16}) {
        for (int width : {32, 160, 1024, 1056}) { get_rows_test(backend.get(), width, type, width == 1056); }
    }
    get_rows_rewrite_test(backend.get(), GGML_TYPE_BF16);
    get_rows_rewrite_test(backend.get(), GGML_TYPE_F16);
    if (argc > 1 && std::strcmp(argv[1], "--get-rows") == 0) { return 0; }
    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_BF16, GGML_TYPE_F16}) {
        conversion_test(backend.get(), type, 1, 160);
        conversion_test(backend.get(), type, 9, 672);
        conversion_test(backend.get(), type, 65, 160);
        conversion_test(backend.get(), type, 129, 160);
        conversion_test(backend.get(), type, 257, 1056);
    }
    matmul_test(backend.get(), 3, 144, 672, false);
    matmul_test(backend.get(), 129, 384, 160, false);
    matmul_test(backend.get(), 3, 160, 672, true);
    matmul_test(backend.get(), 513, 1040, 672, false);
    matmul_test(backend.get(), 257, 1056, 544, true);
    matmul_test(backend.get(), 3, 16, 32, false, GGML_TYPE_MXFP4, GGML_TYPE_F32, false);
    matmul_test(backend.get(), 9, 160, 128, false, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4);
    matmul_test(backend.get(), 513, 544, 544, false, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4);
    matmul_test(backend.get(), 9, 160, 128, false, GGML_TYPE_MXFP4, GGML_TYPE_BF16);
    matmul_test(backend.get(), 9, 160, 128, false, GGML_TYPE_MXFP4, GGML_TYPE_F16);
    matmul_test(backend.get(), 65, 544, 672, false, GGML_TYPE_MXFP4, GGML_TYPE_BF16);
    matmul_test(backend.get(), 65, 544, 672, false, GGML_TYPE_MXFP4, GGML_TYPE_F16);
    matmul_test(backend.get(), 1, 6144, 160, false, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4);
    matmul_test(backend.get(), 257, 1056, 160, true, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, true, 3);
    matmul_test(backend.get(), 65, 544, 160, false, GGML_TYPE_MXFP4, GGML_TYPE_MXFP4, true, 1);
    matmul_test(backend.get(), 9, 144, 128, false, GGML_TYPE_MXFP4, GGML_TYPE_BF16);
    matmul_test(backend.get(), 9, 144, 128, false, GGML_TYPE_MXFP4, GGML_TYPE_F16);
    for (ggml_type type : {GGML_TYPE_BF16, GGML_TYPE_F16}) {
        matmul_test(backend.get(), 9, 160, 128, false, type, GGML_TYPE_MXFP4);
        matmul_test(backend.get(), 3, 160, 128, true, type);
        matmul_test(backend.get(), 65, 544, 544, true, type);
    }
    std::puts("MXFP4 PASS");
}

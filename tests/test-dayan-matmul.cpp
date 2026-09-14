#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml-dayan.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-matmul: %s\n", message);
    return 1;
}

static bool parse_dimension(const char * text, int64_t * value) {
    char * end = nullptr;

    errno             = 0;
    const long long v = std::strtoll(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || v <= 0) {
        return false;
    }
    *value = v;
    return true;
}

static uint16_t encode_lowp(enum ggml_type type, float value) {
    return type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value);
}

static float decode_lowp(enum ggml_type type, uint16_t value) {
    return type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(ggml_bf16_t{ value }) : ggml_fp16_to_fp32(value);
}

struct trace_state {
    bool found = false;
};

static void trace_callback(const ggml_backend_dayan_slot_trace_event * event, void * user_data) {
    trace_state * state = (trace_state *) user_data;
    if (event->executed && std::strcmp(event->kind, "mul_mat_silu") == 0) {
        state->found = true;
    }
}

static ggml_backend_dayan_set_slot_trace_callback_t trace_setter(ggml_backend_t backend) {
    ggml_backend_dev_t device = ggml_backend_get_device(backend);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    return (ggml_backend_dayan_set_slot_trace_callback_t) ggml_backend_reg_get_proc_address(
        reg, "ggml_backend_dayan_set_slot_trace_callback");
}

static int run_case(ggml_backend_t dayan,
                    enum ggml_type type,
                    int64_t        m,
                    int64_t        n,
                    int64_t        k,
                    int64_t        weight_ne2,
                    int64_t        weight_ne3,
                    int64_t        batch_ne2,
                    int64_t        batch_ne3,
                    bool           silu) {
    constexpr int    graph_size = 8;
    ggml_init_params params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the ggml context");
    }

    ggml_tensor * weight     = ggml_new_tensor_4d(ctx.get(), type, k, n, weight_ne2, weight_ne3);
    ggml_tensor * activation = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, m, batch_ne2, batch_ne3);
    ggml_tensor * mul_mat    = ggml_mul_mat(ctx.get(), weight, activation);
    ggml_tensor * output     = silu ? ggml_silu(ctx.get(), mul_mat) : mul_mat;
    ggml_set_name(weight, "weight");
    ggml_set_name(activation, "activation");
    ggml_set_name(mul_mat, "mul_mat");
    ggml_set_name(output, "output");
    ggml_set_input(weight);
    ggml_set_input(activation);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(dayan, mul_mat) || !ggml_backend_supports_op(dayan, output)) {
        return fail("DAYAN does not support the requested matrix multiplication");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate graph tensors");
    }

    std::vector<uint16_t> weight_data((size_t) (n * k * weight_ne2 * weight_ne3));
    std::vector<float>    activation_data((size_t) (m * k * batch_ne2 * batch_ne3));
    for (size_t index = 0; index < weight_data.size(); ++index) {
        weight_data[index] = encode_lowp(type, ((int) (index % 17) - 8) / 64.0f);
    }
    for (size_t index = 0; index < activation_data.size(); ++index) {
        activation_data[index] = ((int) (index % 13) - 6) / 64.0f;
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(activation, activation_data.data(), 0, ggml_nbytes(activation));

    trace_state                                  trace;
    ggml_backend_dayan_set_slot_trace_callback_t setter = trace_setter(dayan);
    if (silu && setter == nullptr) {
        return fail("cannot install the Dayan slot trace callback");
    }
    if (setter != nullptr) {
        setter(trace_callback, &trace);
    }
    const enum ggml_status status = ggml_backend_graph_compute(dayan, graph);
    if (setter != nullptr) {
        setter(nullptr, nullptr);
    }
    if (status != GGML_STATUS_SUCCESS) {
        return fail("DAYAN matrix multiplication failed");
    }
    if (silu && !trace.found) {
        return fail("MUL_MAT and SiLU were not fused");
    }

    std::vector<float> result((size_t) (m * n * batch_ne2 * batch_ne3));
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));

    float   max_abs        = 0.0f;
    float   max_rel        = 0.0f;
    int64_t worst_row      = 0;
    int64_t worst_column   = 0;
    int64_t worst_batch2   = 0;
    int64_t worst_batch3   = 0;
    float   worst_actual   = 0.0f;
    float   worst_expected = 0.0f;
    bool    close           = true;
    for (int64_t i3 = 0; i3 < batch_ne3; ++i3) {
        const int64_t weight_i3 = i3 / (batch_ne3 / weight_ne3);
        for (int64_t i2 = 0; i2 < batch_ne2; ++i2) {
            const int64_t weight_i2 = i2 / (batch_ne2 / weight_ne2);
            for (int64_t row = 0; row < m; ++row) {
                for (int64_t column = 0; column < n; ++column) {
                    float expected = 0.0f;
                    for (int64_t inner = 0; inner < k; ++inner) {
                        const size_t weight_index =
                            (size_t) ((((weight_i3 * weight_ne2 + weight_i2) * n + column) * k) + inner);
                        const size_t activation_index =
                            (size_t) ((((i3 * batch_ne2 + i2) * m + row) * k) + inner);
                        const float lhs = decode_lowp(type, weight_data[weight_index]);
                        const float rhs = decode_lowp(type, encode_lowp(type, activation_data[activation_index]));
                        expected += lhs * rhs;
                    }
                    if (silu) {
                        expected /= 1.0f + std::exp(-expected);
                    }
                    const size_t result_index =
                        (size_t) ((((i3 * batch_ne2 + i2) * m + row) * n) + column);
                    const float actual    = result[result_index];
                    const float abs_error = std::abs(actual - expected);
                    const float rel_error = abs_error / std::max(std::abs(expected), 1.0e-6f);
                    const float tolerance = silu ? 1.0e-5f + 1.0e-4f * std::abs(expected) : 1.0e-5f;
                    close                 = close && std::isfinite(actual) && abs_error <= tolerance;
                    max_rel               = std::max(max_rel, rel_error);
                    if (abs_error > max_abs) {
                        max_abs        = abs_error;
                        worst_row      = row;
                        worst_column   = column;
                        worst_batch2   = i2;
                        worst_batch3   = i3;
                        worst_actual   = actual;
                        worst_expected = expected;
                    }
                }
            }
        }
    }

    std::printf(
        "type=%s%s M=%lld N=%lld K=%lld weight_batch=[%lld,%lld] batch=[%lld,%lld] elements=%zu max_abs=%g "
        "max_rel=%g worst=C[%lld,%lld,%lld,%lld] actual=%g expected=%g\n",
        ggml_type_name(type), silu ? " SiLU" : "", (long long) m, (long long) n, (long long) k,
        (long long) weight_ne2, (long long) weight_ne3, (long long) batch_ne2, (long long) batch_ne3, result.size(),
        max_abs, max_rel, (long long) worst_column, (long long) worst_row, (long long) worst_batch2,
        (long long) worst_batch3, worst_actual, worst_expected);
    if (!close) {
        return fail("numerical comparison failed");
    }

    return 0;
}

int main(int argc, char ** argv) {
    int64_t m = 1;
    int64_t n = 8064;
    int64_t k = 1024;

    if (argc != 1 && argc != 4) {
        std::fprintf(stderr, "usage: %s [M N K]\n", argv[0]);
        return 2;
    }
    if (argc == 4 &&
        (!parse_dimension(argv[1], &m) || !parse_dimension(argv[2], &n) || !parse_dimension(argv[3], &k))) {
        return fail("matrix dimensions must be positive integers");
    }

    ggml_backend_load_all();
    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    if (!dayan) {
        return fail("cannot initialize the DAYAN backend");
    }

    if (run_case(dayan.get(), GGML_TYPE_BF16, m, n, k, 1, 1, 1, 1, false) != 0) {
        return 1;
    }
    if (argc == 1 &&
        (run_case(dayan.get(), GGML_TYPE_BF16, 1, 1, 2048, 1, 1, 1, 1, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 8, 1, 2048, 1, 1, 1, 1, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_F16, 2, 97, 300, 1, 1, 1, 1, true) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 2, 64, 64, 1, 1, 2, 2, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 2, 64, 64, 2, 1, 4, 2, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 2, 64, 64, 1, 2, 4, 4, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 2, 384, 300, 1, 1, 1, 1, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_F16, 64, 16, 64, 2, 2, 2, 2, false) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 2, 64, 64, 1, 1, 2, 2, true) != 0 ||
         run_case(dayan.get(), GGML_TYPE_BF16, 2, 64, 64, 2, 1, 4, 2, true) != 0 ||
         run_case(dayan.get(), GGML_TYPE_F16, 64, 16, 64, 1, 1, 1, 1, true) != 0)) {
        return 1;
    }

    if (argc == 1) {
        for (int64_t rows : {1, 3, 7, 8, 15, 16, 31, 32, 511, 512, 513}) {
            if (run_case(dayan.get(), GGML_TYPE_BF16, rows, 1541, 257, 1, 1, 1, 1, false) != 0) {
                return 1;
            }
        }
        if (run_case(dayan.get(), GGML_TYPE_BF16, 1024, 1024, 257, 1, 1, 1, 1, false) != 0 ||
            run_case(dayan.get(), GGML_TYPE_F16, 1025, 1025, 257, 2, 1, 2, 1, false) != 0 ||
            run_case(dayan.get(), GGML_TYPE_BF16, 1025, 1537, 257, 1, 1, 1, 1, false) != 0 ||
            run_case(dayan.get(), GGML_TYPE_BF16, 31, 6149, 257, 1, 1, 1, 1, false) != 0 ||
            run_case(dayan.get(), GGML_TYPE_F16, 257, 33, 65, 1, 1, 2, 2, false) != 0) {
            return 1;
        }
    }

    std::puts("test-dayan-matmul: PASS");
    return 0;
}

#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

    constexpr int    graph_size = 4;
    ggml_init_params params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the ggml context");
    }

    ggml_tensor * weight     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, k, n);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * output     = ggml_mul_mat(ctx.get(), weight, activation);
    ggml_set_name(weight, "weight");
    ggml_set_name(activation, "activation");
    ggml_set_name(output, "output");
    ggml_set_input(weight);
    ggml_set_input(activation);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_supports_op(dayan.get(), output)) {
        return fail("DAYAN does not support the requested matrix multiplication");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan.get()));
    if (!buffer) {
        return fail("cannot allocate graph tensors");
    }

    std::vector<ggml_bf16_t> weight_data((size_t) (n * k));
    std::vector<float>       activation_data((size_t) (m * k));
    for (size_t index = 0; index < weight_data.size(); ++index) {
        weight_data[index] = ggml_fp32_to_bf16(((int) (index % 17) - 8) / 64.0f);
    }
    for (size_t index = 0; index < activation_data.size(); ++index) {
        activation_data[index] = ((int) (index % 13) - 6) / 64.0f;
    }
    ggml_backend_tensor_set(weight, weight_data.data(), 0, ggml_nbytes(weight));
    ggml_backend_tensor_set(activation, activation_data.data(), 0, ggml_nbytes(activation));

    if (ggml_backend_graph_compute(dayan.get(), graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN matrix multiplication failed");
    }

    std::vector<float> result((size_t) (m * n));
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));

    float   max_abs        = 0.0f;
    float   max_rel        = 0.0f;
    int64_t worst_row      = 0;
    int64_t worst_column   = 0;
    float   worst_actual   = 0.0f;
    float   worst_expected = 0.0f;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t column = 0; column < n; ++column) {
            float expected = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                const float lhs = ggml_bf16_to_fp32(weight_data[(size_t) (column * k + inner)]);
                const float rhs = ggml_bf16_to_fp32(ggml_fp32_to_bf16(activation_data[(size_t) (row * k + inner)]));
                expected += lhs * rhs;
            }
            const float actual    = result[(size_t) (row * n + column)];
            const float abs_error = std::abs(actual - expected);
            const float rel_error = abs_error / std::max(std::abs(expected), 1.0e-6f);
            max_rel               = std::max(max_rel, rel_error);
            if (abs_error > max_abs) {
                max_abs        = abs_error;
                worst_row      = row;
                worst_column   = column;
                worst_actual   = actual;
                worst_expected = expected;
            }
        }
    }

    std::printf("M=%lld N=%lld K=%lld elements=%lld max_abs=%g max_rel=%g worst=C[%lld,%lld] actual=%g expected=%g\n",
                (long long) m, (long long) n, (long long) k, (long long) (m * n), max_abs, max_rel,
                (long long) worst_row, (long long) worst_column, worst_actual, worst_expected);
    if (max_abs > 1.0e-5f) {
        return fail("numerical comparison failed");
    }

    std::puts("test-dayan-matmul: PASS");
    return 0;
}

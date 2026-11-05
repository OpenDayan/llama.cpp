#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

struct gdn_graph {
    ggml_context_ptr ctx;
    ggml_cgraph *    graph;
    ggml_tensor *    q;
    ggml_tensor *    k;
    ggml_tensor *    v;
    ggml_tensor *    g;
    ggml_tensor *    beta;
    ggml_tensor *    state;
    ggml_tensor *    output;
};

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-gated-delta-net: %s\n", message);
    return 1;
}

static gdn_graph build_graph(int64_t heads, int64_t tokens, int64_t sequences, bool kda, int64_t qk_heads, int64_t qk_sequences, int snapshots) {
    constexpr int64_t head_dim   = 128;
    constexpr int     graph_size = 16;
    ggml_init_params  params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    gdn_graph result;
    result.ctx.reset(ggml_init(params));
    if (!result.ctx) {
        return result;
    }

    result.q     = ggml_new_tensor_4d(result.ctx.get(), GGML_TYPE_F32, head_dim, qk_heads, tokens, qk_sequences);
    result.k     = ggml_new_tensor_4d(result.ctx.get(), GGML_TYPE_F32, head_dim, qk_heads, tokens, qk_sequences);
    result.v     = ggml_new_tensor_4d(result.ctx.get(), GGML_TYPE_F32, head_dim, heads, tokens, sequences);
    result.g     = ggml_new_tensor_4d(result.ctx.get(), GGML_TYPE_F32, kda ? head_dim : 1, heads, tokens, sequences);
    result.beta  = ggml_new_tensor_4d(result.ctx.get(), GGML_TYPE_F32, 1, heads, tokens, sequences);
    result.state = ggml_new_tensor_4d(result.ctx.get(), GGML_TYPE_F32, head_dim, head_dim, heads, sequences);
    ggml_set_input(result.q);
    ggml_set_input(result.k);
    ggml_set_input(result.v);
    ggml_set_input(result.g);
    ggml_set_input(result.beta);
    ggml_set_input(result.state);
    result.output =
        ggml_gated_delta_net(result.ctx.get(), result.q, result.k, result.v, result.g, result.beta, result.state, snapshots);
    ggml_set_output(result.output);
    result.graph = ggml_new_graph_custom(result.ctx.get(), graph_size, false);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static void normalize_rows(std::vector<float> * values, int64_t rows) {
    constexpr int64_t head_dim = 128;
    for (int64_t row = 0; row < rows; ++row) {
        float sum = 0.0f;
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            const float value = (*values)[(size_t) row * head_dim + dim];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum + 1.0e-6f);
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            (*values)[(size_t) row * head_dim + dim] *= scale;
        }
    }
}

static int run_case(ggml_backend_t dayan,
                    ggml_backend_t cpu,
                    int64_t        heads,
                    int64_t        tokens,
                    int64_t        sequences,
                    bool           kda,
                    uint32_t       seed,
                    float          gate_min = -2.0f,
                    float          gate_max = -0.05f,
                    int64_t        qk_heads = 0,
                    int64_t        qk_sequences = 0,
                    int            snapshots = 1) {
    qk_heads = qk_heads ? qk_heads : heads;
    qk_sequences = qk_sequences ? qk_sequences : sequences;
    gdn_graph reference = build_graph(heads, tokens, sequences, kda, qk_heads, qk_sequences, snapshots);
    gdn_graph actual    = build_graph(heads, tokens, sequences, kda, qk_heads, qk_sequences, snapshots);
    if (!reference.ctx || !actual.ctx) {
        return fail("cannot create graph contexts");
    }

    ggml_backend_buffer_ptr reference_buffer(ggml_backend_alloc_ctx_tensors(reference.ctx.get(), cpu));
    ggml_backend_buffer_ptr actual_buffer(ggml_backend_alloc_ctx_tensors(actual.ctx.get(), dayan));
    if (!reference_buffer || !actual_buffer) {
        return fail("cannot allocate graph tensors");
    }

    std::mt19937                          generator(seed);
    std::uniform_real_distribution<float> qk_distribution(-1.0f, 1.0f);
    std::uniform_real_distribution<float> v_distribution(-0.25f, 0.25f);
    std::uniform_real_distribution<float> gate_distribution(gate_min, gate_max);
    std::uniform_real_distribution<float> beta_distribution(0.1f, 0.9f);
    std::uniform_real_distribution<float> state_distribution(-0.02f, 0.02f);
    std::vector<float>                    q((size_t) ggml_nelements(reference.q));
    std::vector<float>                    k((size_t) ggml_nelements(reference.k));
    std::vector<float>                    v((size_t) ggml_nelements(reference.v));
    std::vector<float>                    g((size_t) ggml_nelements(reference.g));
    std::vector<float>                    beta((size_t) ggml_nelements(reference.beta));
    std::vector<float>                    state((size_t) ggml_nelements(reference.state));
    for (float & value : q) {
        value = qk_distribution(generator);
    }
    for (float & value : k) {
        value = qk_distribution(generator);
    }
    normalize_rows(&q, qk_heads * tokens * qk_sequences);
    normalize_rows(&k, qk_heads * tokens * qk_sequences);
    for (float & value : v) {
        value = v_distribution(generator);
    }
    for (float & value : g) {
        value = gate_distribution(generator);
    }
    for (float & value : beta) {
        value = beta_distribution(generator);
    }
    for (float & value : state) {
        value = state_distribution(generator);
    }

    for (gdn_graph * graph : { &reference, &actual }) {
        ggml_backend_tensor_set(graph->q, q.data(), 0, ggml_nbytes(graph->q));
        ggml_backend_tensor_set(graph->k, k.data(), 0, ggml_nbytes(graph->k));
        ggml_backend_tensor_set(graph->v, v.data(), 0, ggml_nbytes(graph->v));
        ggml_backend_tensor_set(graph->g, g.data(), 0, ggml_nbytes(graph->g));
        ggml_backend_tensor_set(graph->beta, beta.data(), 0, ggml_nbytes(graph->beta));
        ggml_backend_tensor_set(graph->state, state.data(), 0, ggml_nbytes(graph->state));
    }

    if (ggml_backend_graph_compute(cpu, reference.graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU reference graph failed");
    }
    if (ggml_backend_graph_compute(dayan, actual.graph) != GGML_STATUS_SUCCESS) {
        return fail("Dayan graph failed");
    }

    std::vector<float> expected((size_t) ggml_nelements(reference.output));
    std::vector<float> observed(expected.size());
    ggml_backend_tensor_get(reference.output, expected.data(), 0, ggml_nbytes(reference.output));
    ggml_backend_tensor_get(actual.output, observed.data(), 0, ggml_nbytes(actual.output));

    float  max_abs       = 0.0f;
    float  max_rel       = 0.0f;
    size_t max_abs_index = 0;
    bool   close         = true;
    double squared_error[2] = {};
    double squared_expected[2] = {};
    const size_t output_elements = (size_t) heads * tokens * sequences * 128;
    const bool chunked = !kda && snapshots == 1 && tokens >= 16;
    // MAT truncates its TF32 inputs; the recurrent VEC path uses FP32.
    const float rtol = chunked ? 3.0e-3f : 1.0e-4f;
    for (size_t index = 0; index < expected.size(); ++index) {
        const float abs_error = std::abs(observed[index] - expected[index]);
        const float rel_error = abs_error / std::max(std::abs(expected[index]), 1.0e-6f);
        const int part = index < output_elements ? 0 : 1;
        const float atol = chunked ? (part == 0 ? 2.0e-5f : 2.0e-4f) : 1.0e-6f;
        close = close && std::isfinite(observed[index]) && abs_error <= atol + rtol * std::abs(expected[index]);
        squared_error[part] += (double) abs_error * abs_error;
        squared_expected[part] += (double) expected[index] * expected[index];
        if (abs_error > max_abs) {
            max_abs       = abs_error;
            max_abs_index = index;
        }
        max_rel = std::max(max_rel, rel_error);
    }

    const double output_l2 = std::sqrt(squared_error[0] / std::max(squared_expected[0], 1.0e-30));
    const double state_l2 = std::sqrt(squared_error[1] / std::max(squared_expected[1], 1.0e-30));
    close = close && output_l2 <= rtol && state_l2 <= rtol;
    std::printf("heads=%lld tokens=%lld sequences=%lld qk_heads=%lld qk_sequences=%lld snapshots=%d kda=%d path=%s elements=%zu max_abs=%g max_rel=%g output_l2=%g state_l2=%g index=%zu\n",
                (long long) heads, (long long) tokens, (long long) sequences, (long long) qk_heads,
                (long long) qk_sequences, snapshots, kda ? 1 : 0, chunked ? "chunk" : "recurrent", expected.size(), max_abs,
                max_rel, output_l2, state_l2, max_abs_index);
    std::fflush(stdout);
    if (!close) {
        std::fprintf(stderr, "expected=%g observed=%g\n", expected[max_abs_index], observed[max_abs_index]);
        return fail("numerical comparison failed");
    }
    return 0;
}

int main() {
    ggml_backend_load_all();
    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!dayan || !cpu) {
        return fail("cannot initialize backends");
    }

    if (run_case(dayan.get(), cpu.get(), 2, 1, 1, false, 0x35d17U) != 0 ||
        run_case(dayan.get(), cpu.get(), 2, 3, 1, false, 0x35d18U) != 0 ||
        run_case(dayan.get(), cpu.get(), 1, 2, 1, true, 0x35d19U) != 0 ||
        run_case(dayan.get(), cpu.get(), 16, 1, 1, false, 0x35d1aU) != 0 ||
        run_case(dayan.get(), cpu.get(), 32, 1, 1, false, 0x35d28U, -2.0f, -0.05f, 16, 1) != 0 ||
        run_case(dayan.get(), cpu.get(), 16, 16, 1, false, 0x35d1bU) != 0 ||
        run_case(dayan.get(), cpu.get(), 5, 15, 1, false, 0x35d1cU) != 0 ||
        run_case(dayan.get(), cpu.get(), 1, 17, 1, false, 0x35d1dU) != 0 ||
        run_case(dayan.get(), cpu.get(), 4, 64, 1, false, 0x35d1eU) != 0 ||
        run_case(dayan.get(), cpu.get(), 5, 65, 1, false, 0x35d1fU, -0.02f, -0.001f) != 0 ||
        run_case(dayan.get(), cpu.get(), 4, 128, 1, false, 0x35d20U) != 0 ||
        run_case(dayan.get(), cpu.get(), 16, 1024, 1, false, 0x35d21U, -0.02f, -0.001f) != 0 ||
        run_case(dayan.get(), cpu.get(), 1, 257, 1, false, 0x35d26U, -0.02f, -0.001f) != 0 ||
        run_case(dayan.get(), cpu.get(), 4, 129, 5, false, 0x35d27U, -0.02f, -0.001f, 2, 1) != 0 ||
        run_case(dayan.get(), cpu.get(), 8, 65, 3, false, 0x35d22U, -2.0f, -0.05f, 2, 1) != 0 ||
        run_case(dayan.get(), cpu.get(), 4, 17, 1, false, 0x35d23U, -2.0f, -0.05f, 0, 0, 2) != 0 ||
        run_case(dayan.get(), cpu.get(), 4, 16, 1, true, 0x35d24U) != 0 ||
        run_case(dayan.get(), cpu.get(), 1, 64, 1, false, 0x35d25U, -4.0f, -3.0f) != 0) {
        return 1;
    }

    std::puts("test-dayan-gated-delta-net: PASS");
    return 0;
}

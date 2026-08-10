#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using ggml_backend_graph_rewrite_t = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);

static constexpr int64_t kEmbedding   = 64;
static constexpr int64_t kHidden      = 32;
static constexpr int64_t kExperts     = 512;
static constexpr int64_t kExpertsUsed = 4;
static constexpr int64_t kTokens      = 2;
static constexpr int     kGraphSize   = 128;

struct moe_graph {
    ggml_context_ptr ctx;
    ggml_cgraph *    graph         = nullptr;
    ggml_tensor *    input         = nullptr;
    ggml_tensor *    router_weight = nullptr;
    ggml_tensor *    gate_weight   = nullptr;
    ggml_tensor *    up_weight     = nullptr;
    ggml_tensor *    down_weight   = nullptr;
    ggml_tensor *    down_bias     = nullptr;
    ggml_tensor *    argsort       = nullptr;
    ggml_tensor *    selected      = nullptr;
    ggml_tensor *    top_k         = nullptr;
    ggml_tensor *    gate          = nullptr;
    ggml_tensor *    up            = nullptr;
    ggml_tensor *    fused         = nullptr;
    ggml_tensor *    fused_lowp    = nullptr;
    ggml_tensor *    down          = nullptr;
    ggml_tensor *    biased        = nullptr;
    ggml_tensor *    output        = nullptr;
};

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-moe: %s\n", message);
    return 1;
}

static bool rewrite_graph(ggml_backend_t dayan, ggml_context * ctx, ggml_cgraph * graph) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    auto rewrite = (ggml_backend_graph_rewrite_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    return rewrite != nullptr && rewrite(dayan, ctx, graph);
}

static ggml_tensor * new_input(ggml_context *  ctx,
                               enum ggml_type  type,
                               int             dimensions,
                               const int64_t * shape,
                               const char *    name) {
    ggml_tensor * tensor = ggml_new_tensor(ctx, type, dimensions, shape);
    ggml_set_name(tensor, name);
    ggml_set_input(tensor);
    return tensor;
}

static moe_graph build_graph() {
    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * kGraphSize + ggml_graph_overhead_custom(kGraphSize, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    moe_graph result;
    result.ctx.reset(ggml_init(params));
    if (!result.ctx) {
        return result;
    }

    const int64_t input_shape[]  = { kEmbedding, kTokens };
    const int64_t router_shape[] = { kEmbedding, kExperts };
    const int64_t up_shape[]     = { kEmbedding, kHidden, kExperts };
    const int64_t down_shape[]   = { kHidden, kEmbedding, kExperts };
    const int64_t bias_shape[]   = { kEmbedding, kExperts };
    result.input                 = new_input(result.ctx.get(), GGML_TYPE_F32, 2, input_shape, "input");
    result.router_weight         = new_input(result.ctx.get(), GGML_TYPE_BF16, 2, router_shape, "router_weight");
    result.gate_weight           = new_input(result.ctx.get(), GGML_TYPE_BF16, 3, up_shape, "gate_weight");
    result.up_weight             = new_input(result.ctx.get(), GGML_TYPE_BF16, 3, up_shape, "up_weight");
    result.down_weight           = new_input(result.ctx.get(), GGML_TYPE_BF16, 3, down_shape, "down_weight");
    result.down_bias             = new_input(result.ctx.get(), GGML_TYPE_F32, 2, bias_shape, "down_bias");

    ggml_tensor * logits = ggml_mul_mat(result.ctx.get(), result.router_weight, result.input);
    ggml_tensor * probs  = ggml_soft_max(result.ctx.get(), logits);
    result.argsort       = ggml_argsort(result.ctx.get(), probs, GGML_SORT_ORDER_DESC);
    result.selected = ggml_view_2d(result.ctx.get(), result.argsort, kExpertsUsed, kTokens, result.argsort->nb[1], 0);
    result.top_k    = ggml_top_k(result.ctx.get(), probs, kExpertsUsed);

    ggml_tensor * expert_input = ggml_reshape_3d(result.ctx.get(), result.input, kEmbedding, 1, kTokens);
    result.gate                = ggml_mul_mat_id(result.ctx.get(), result.gate_weight, expert_input, result.selected);
    result.up                  = ggml_mul_mat_id(result.ctx.get(), result.up_weight, expert_input, result.selected);
    result.fused               = ggml_swiglu_split(result.ctx.get(), result.gate, result.up);
    result.down                = ggml_mul_mat_id(result.ctx.get(), result.down_weight, result.fused, result.selected);
    result.biased              = ggml_add_id(result.ctx.get(), result.down, result.down_bias, result.selected);

    ggml_tensor * probabilities = ggml_reshape_3d(result.ctx.get(), probs, 1, kExperts, kTokens);
    ggml_tensor * weights       = ggml_get_rows(result.ctx.get(), probabilities, result.selected);
    ggml_tensor * weighted      = ggml_mul(result.ctx.get(), result.biased, weights);
    result.output               = ggml_view_2d(result.ctx.get(), weighted, kEmbedding, kTokens, weighted->nb[2], 0);
    for (int64_t used = 1; used < kExpertsUsed; ++used) {
        ggml_tensor * expert = ggml_view_2d(result.ctx.get(), weighted, kEmbedding, kTokens, weighted->nb[2],
                                            (size_t) used * weighted->nb[1]);
        result.output        = ggml_add(result.ctx.get(), result.output, expert);
    }

    ggml_set_name(result.argsort, "argsort");
    ggml_set_name(result.top_k, "top_k");
    ggml_set_name(result.gate, "gate");
    ggml_set_name(result.up, "up");
    ggml_set_name(result.fused, "swiglu");
    ggml_set_name(result.down, "down");
    ggml_set_name(result.biased, "biased");
    ggml_set_name(result.output, "output");

    result.graph = ggml_new_graph_custom(result.ctx.get(), kGraphSize, false);
    ggml_build_forward_expand(result.graph, result.top_k);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static std::vector<ggml_bf16_t> make_router_weight() {
    std::vector<ggml_bf16_t> values((size_t) kEmbedding * kExperts);
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        for (int64_t inner = 0; inner < kEmbedding; ++inner) {
            const int64_t rank                           = (expert * (2 * inner + 1) + 17 * inner) % kExperts;
            values[(size_t) expert * kEmbedding + inner] = ggml_fp32_to_bf16((rank - kExperts / 2) / 64.0f);
        }
    }
    return values;
}

static std::vector<ggml_bf16_t> make_expert_weight(int64_t input, int64_t output, int salt) {
    std::vector<ggml_bf16_t> values((size_t) input * output * kExperts);
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        for (int64_t row = 0; row < output; ++row) {
            for (int64_t inner = 0; inner < input; ++inner) {
                const int value = (int) ((expert * 13 + row * 7 + inner * 3 + salt) % 17) - 8;
                values[((size_t) expert * output + row) * input + inner] = ggml_fp32_to_bf16(value / 128.0f);
            }
        }
    }
    return values;
}

static bool initialize_graph(const moe_graph & graph) {
    std::vector<float> input((size_t) kEmbedding * kTokens, 0.0f);
    for (int64_t token = 0; token < kTokens; ++token) {
        input[(size_t) token * kEmbedding + token] = 1.0f;
    }
    const std::vector<ggml_bf16_t> router = make_router_weight();
    const std::vector<ggml_bf16_t> gate   = make_expert_weight(kEmbedding, kHidden, 1);
    const std::vector<ggml_bf16_t> up     = make_expert_weight(kEmbedding, kHidden, 5);
    const std::vector<ggml_bf16_t> down   = make_expert_weight(kHidden, kEmbedding, 9);
    std::vector<float>             bias((size_t) kEmbedding * kExperts);
    for (int64_t expert = 0; expert < kExperts; ++expert) {
        for (int64_t element = 0; element < kEmbedding; ++element) {
            bias[(size_t) expert * kEmbedding + element] = ((expert * 5 + element * 3) % 11 - 5) / 256.0f;
        }
    }

    ggml_backend_tensor_set(graph.input, input.data(), 0, ggml_nbytes(graph.input));
    ggml_backend_tensor_set(graph.router_weight, router.data(), 0, ggml_nbytes(graph.router_weight));
    ggml_backend_tensor_set(graph.gate_weight, gate.data(), 0, ggml_nbytes(graph.gate_weight));
    ggml_backend_tensor_set(graph.up_weight, up.data(), 0, ggml_nbytes(graph.up_weight));
    ggml_backend_tensor_set(graph.down_weight, down.data(), 0, ggml_nbytes(graph.down_weight));
    ggml_backend_tensor_set(graph.down_bias, bias.data(), 0, ggml_nbytes(graph.down_bias));
    return true;
}

static float compare_f32(const ggml_tensor * expected, const ggml_tensor * actual, bool * close) {
    std::vector<float> reference((size_t) ggml_nelements(expected));
    std::vector<float> observed(reference.size());
    ggml_backend_tensor_get(expected, reference.data(), 0, ggml_nbytes(expected));
    ggml_backend_tensor_get(actual, observed.data(), 0, ggml_nbytes(actual));

    float max_abs = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float error = std::abs(reference[i] - observed[i]);
        max_abs           = std::max(max_abs, error);
        *close            = *close && error <= 0.05f + 0.08f * std::abs(reference[i]);
    }
    return max_abs;
}

static float compare_bf16(const ggml_tensor * expected, const ggml_tensor * actual, bool * close) {
    std::vector<ggml_bf16_t> reference((size_t) ggml_nelements(expected));
    std::vector<ggml_bf16_t> observed(reference.size());
    ggml_backend_tensor_get(expected, reference.data(), 0, ggml_nbytes(expected));
    ggml_backend_tensor_get(actual, observed.data(), 0, ggml_nbytes(actual));

    float max_abs = 0.0f;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float expected_value = ggml_bf16_to_fp32(reference[i]);
        const float actual_value   = ggml_bf16_to_fp32(observed[i]);
        const float error          = std::abs(expected_value - actual_value);
        max_abs                    = std::max(max_abs, error);
        *close                     = *close && error <= 0.02f + 0.08f * std::abs(expected_value);
    }
    return max_abs;
}

static bool compare_i32(const ggml_tensor * expected, const ggml_tensor * actual) {
    std::vector<int32_t> reference((size_t) ggml_nelements(expected));
    std::vector<int32_t> observed(reference.size());
    ggml_backend_tensor_get(expected, reference.data(), 0, ggml_nbytes(expected));
    ggml_backend_tensor_get(actual, observed.data(), 0, ggml_nbytes(actual));
    return reference == observed;
}

int main() {
    ggml_backend_load_all();

    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!dayan || !cpu) {
        return fail("cannot initialize backends");
    }

    moe_graph reference = build_graph();
    moe_graph actual    = build_graph();
    if (!reference.ctx || !actual.ctx || !rewrite_graph(dayan.get(), reference.ctx.get(), reference.graph) ||
        !rewrite_graph(dayan.get(), actual.ctx.get(), actual.graph)) {
        return fail("cannot build or rewrite graphs");
    }

    reference.fused_lowp = reference.down->src[1];
    actual.fused_lowp    = actual.down->src[1];
    if (actual.gate->src[1] == actual.input || actual.gate->src[1] != actual.up->src[1] ||
        actual.gate->src[1]->op != GGML_OP_CPY || actual.gate->src[1]->type != GGML_TYPE_BF16 ||
        actual.fused_lowp == actual.fused || actual.fused_lowp->op != GGML_OP_CPY ||
        actual.fused_lowp->src[0] != actual.fused || actual.fused_lowp->type != GGML_TYPE_BF16) {
        return fail("graph rewrite did not create the expected MoE activation casts");
    }

    ggml_backend_buffer_ptr reference_buffer(ggml_backend_alloc_ctx_tensors(reference.ctx.get(), cpu.get()));
    ggml_backend_buffer_ptr actual_buffer(ggml_backend_alloc_ctx_tensors(actual.ctx.get(), dayan.get()));
    if (!reference_buffer || !actual_buffer || !initialize_graph(reference) || !initialize_graph(actual)) {
        return fail("cannot allocate or initialize tensors");
    }
    if (ggml_backend_graph_compute(cpu.get(), reference.graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU reference graph failed");
    }
    if (ggml_backend_graph_compute(dayan.get(), actual.graph) != GGML_STATUS_SUCCESS) {
        return fail("Dayan MoE graph failed");
    }

    bool        close = compare_i32(reference.argsort, actual.argsort) && compare_i32(reference.top_k, actual.top_k);
    const float fused_max_abs  = compare_bf16(reference.fused_lowp, actual.fused_lowp, &close);
    const float down_max_abs   = compare_f32(reference.biased, actual.biased, &close);
    const float output_max_abs = compare_f32(reference.output, actual.output, &close);
    std::printf(
        "experts=%lld used=%lld tokens=%lld embd=%lld hidden=%lld fused_max_abs=%g down_max_abs=%g "
        "output_max_abs=%g\n",
        (long long) kExperts, (long long) kExpertsUsed, (long long) kTokens, (long long) kEmbedding,
        (long long) kHidden, fused_max_abs, down_max_abs, output_max_abs);
    if (!close) {
        return fail("numerical comparison failed");
    }

    std::puts("test-dayan-moe: PASS");
    return 0;
}

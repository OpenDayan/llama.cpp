#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using ggml_backend_graph_rewrite_t = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);

struct operator_graph {
    ggml_context_ptr           ctx;
    ggml_cgraph *              graph;
    ggml_tensor *              output;
    std::vector<ggml_tensor *> inputs;
};

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-hotspots: %s\n", message);
    return 1;
}

static operator_graph make_graph_context() {
    constexpr int    graph_size = 64;
    ggml_init_params params     = {
        /* .mem_size   = */ 256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    operator_graph result;
    result.ctx.reset(ggml_init(params));
    result.graph  = nullptr;
    result.output = nullptr;
    return result;
}

static ggml_tensor * add_input(operator_graph * graph, enum ggml_type type, int64_t ne0) {
    ggml_tensor * tensor = ggml_new_tensor_1d(graph->ctx.get(), type, ne0);
    ggml_set_input(tensor);
    graph->inputs.push_back(tensor);
    return tensor;
}

static ggml_tensor * add_input(operator_graph * graph, enum ggml_type type, int64_t ne0, int64_t ne1) {
    ggml_tensor * tensor = ggml_new_tensor_2d(graph->ctx.get(), type, ne0, ne1);
    ggml_set_input(tensor);
    graph->inputs.push_back(tensor);
    return tensor;
}

static ggml_tensor * add_input(operator_graph * graph,
                               enum ggml_type type,
                               int64_t        ne0,
                               int64_t        ne1,
                               int64_t        ne2,
                               int64_t        ne3) {
    ggml_tensor * tensor = ggml_new_tensor_4d(graph->ctx.get(), type, ne0, ne1, ne2, ne3);
    ggml_set_input(tensor);
    graph->inputs.push_back(tensor);
    return tensor;
}

static void finish_graph(operator_graph * graph) {
    ggml_set_output(graph->output);
    graph->graph = ggml_new_graph_custom(graph->ctx.get(), 64, false);
    ggml_build_forward_expand(graph->graph, graph->output);
}

static operator_graph build_rope() {
    constexpr int64_t width   = 64;
    constexpr int64_t heads   = 1;
    constexpr int64_t tokens  = 4;
    constexpr int64_t batches = 1;

    operator_graph graph     = make_graph_context();
    ggml_tensor *  input     = add_input(&graph, GGML_TYPE_F32, width, heads, tokens, batches);
    ggml_tensor *  positions = add_input(&graph, GGML_TYPE_I32, 4 * tokens);
    int            sections[GGML_MROPE_SECTIONS] = { 8, 8, 8, 8 };
    graph.output = ggml_rope_multi(graph.ctx.get(), input, positions, nullptr, width, sections, GGML_ROPE_TYPE_MROPE,
                                   tokens, 10000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    finish_graph(&graph);
    return graph;
}

static operator_graph build_ssm_conv() {
    constexpr int64_t kernel   = 4;
    constexpr int64_t channels = 384;
    constexpr int64_t tokens   = 4;

    operator_graph graph = make_graph_context();
    ggml_tensor *  input = add_input(&graph, GGML_TYPE_F32, kernel - 1 + tokens, channels, 1, 1);
    ggml_tensor *  weight = add_input(&graph, GGML_TYPE_F32, kernel, channels);
    graph.output = ggml_silu(graph.ctx.get(), ggml_ssm_conv(graph.ctx.get(), input, weight));
    finish_graph(&graph);
    return graph;
}

static operator_graph build_gdn() {
    constexpr int64_t head_dim = 128;
    constexpr int64_t heads    = 1;
    constexpr int64_t tokens   = 4;

    operator_graph graph = make_graph_context();
    ggml_tensor *  q     = add_input(&graph, GGML_TYPE_F32, head_dim, heads, tokens, 1);
    ggml_tensor *  k     = add_input(&graph, GGML_TYPE_F32, head_dim, heads, tokens, 1);
    ggml_tensor *  v     = add_input(&graph, GGML_TYPE_F32, head_dim, heads, tokens, 1);
    ggml_tensor *  g     = add_input(&graph, GGML_TYPE_F32, 1, heads, tokens, 1);
    ggml_tensor *  beta  = add_input(&graph, GGML_TYPE_F32, 1, heads, tokens, 1);
    ggml_tensor *  state = add_input(&graph, GGML_TYPE_F32, head_dim, head_dim, heads, 1);
    graph.output = ggml_gated_delta_net(graph.ctx.get(), q, k, v, g, beta, state, 1);
    finish_graph(&graph);
    return graph;
}

static operator_graph build_gated_output() {
    constexpr int64_t width  = 64;
    constexpr int64_t heads  = 1;
    constexpr int64_t tokens = 4;

    operator_graph graph      = make_graph_context();
    ggml_tensor *  qg_input   = add_input(&graph, GGML_TYPE_F32, 1, tokens);
    ggml_tensor *  qg_weight  = add_input(&graph, GGML_TYPE_BF16, 1, 2 * width);
    ggml_tensor *  attention  = add_input(&graph, GGML_TYPE_F32, width, tokens);
    ggml_tensor *  out_weight = add_input(&graph, GGML_TYPE_BF16, width, width);
    ggml_tensor *  qg         = ggml_mul_mat(graph.ctx.get(), qg_weight, qg_input);
    ggml_tensor *  gate_view = ggml_view_3d(graph.ctx.get(), qg, width, heads, tokens, 2 * width * sizeof(float),
                                            2 * width * heads * sizeof(float), width * sizeof(float));
    ggml_tensor * gate = ggml_cont_2d(graph.ctx.get(), gate_view, width, tokens);
    ggml_tensor * gated = ggml_mul(graph.ctx.get(), attention, ggml_sigmoid(graph.ctx.get(), gate));
    graph.output = ggml_mul_mat(graph.ctx.get(), out_weight, gated);
    finish_graph(&graph);
    return graph;
}

static bool rewrite_graph(ggml_backend_t dayan, operator_graph * graph) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    auto rewrite = (ggml_backend_graph_rewrite_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    return rewrite != nullptr && rewrite(dayan, graph->ctx.get(), graph->graph);
}

static void initialize_inputs(const operator_graph & graph) {
    for (size_t tensor_index = 0; tensor_index < graph.inputs.size(); ++tensor_index) {
        ggml_tensor * tensor = graph.inputs[tensor_index];
        const size_t  count  = (size_t) ggml_nelements(tensor);
        if (tensor->type == GGML_TYPE_F32) {
            std::vector<float> values(count);
            for (size_t index = 0; index < count; ++index) {
                values[index] = (float) ((int) ((index + 17 * tensor_index) % 31) - 15) / 32.0f;
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        } else if (tensor->type == GGML_TYPE_BF16) {
            std::vector<ggml_bf16_t> values(count);
            for (size_t index = 0; index < count; ++index) {
                const float value = (float) ((int) ((index + 13 * tensor_index) % 29) - 14) / 64.0f;
                values[index]     = ggml_fp32_to_bf16(value);
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        } else if (tensor->type == GGML_TYPE_I32) {
            std::vector<int32_t> values(count);
            for (size_t index = 0; index < count; ++index) {
                values[index] = (int32_t) (index % 4);
            }
            ggml_backend_tensor_set(tensor, values.data(), 0, ggml_nbytes(tensor));
        }
    }
}

static operator_graph build_operator(const std::string & name) {
    if (name == "rope") {
        return build_rope();
    }
    if (name == "ssm-conv") {
        return build_ssm_conv();
    }
    if (name == "gdn") {
        return build_gdn();
    }
    if (name == "gated-output") {
        return build_gated_output();
    }
    return {};
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s rope|ssm-conv|gdn|gated-output\n", argv[0]);
        return 2;
    }

    ggml_backend_load_all();
    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    if (!dayan) {
        return fail("cannot initialize Dayan backend");
    }

    operator_graph graph = build_operator(argv[1]);
    if (!graph.ctx || graph.graph == nullptr || graph.output == nullptr) {
        return fail("unknown operator or graph construction failed");
    }
    if (!rewrite_graph(dayan.get(), &graph)) {
        return fail("graph rewrite failed");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(graph.ctx.get(), dayan.get()));
    if (!buffer) {
        return fail("cannot allocate graph tensors");
    }
    initialize_inputs(graph);
    if (ggml_backend_graph_compute(dayan.get(), graph.graph) != GGML_STATUS_SUCCESS) {
        return fail("Dayan graph failed");
    }

    std::vector<uint8_t> output(ggml_nbytes(graph.output));
    ggml_backend_tensor_get(graph.output, output.data(), 0, output.size());
    std::printf("operator=%s output_bytes=%zu checksum=%u\n", argv[1], output.size(),
                output.empty() ? 0U : (unsigned) output[0]);
    return 0;
}

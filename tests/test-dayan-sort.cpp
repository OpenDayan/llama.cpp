#include "ggml-cpp.h"
#include "layer-compare.h"

#include <limits>
#include <numeric>
#include <string>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct sort_graph {
    ggml_context_ptr ctx;
    ggml_tensor * input;
    ggml_tensor * sorted;
    ggml_tensor * output;
    ggml_cgraph * graph;
};

static sort_graph make_graph(int width, int rows, int used, bool native, bool ascending = false) {
    ggml_init_params params{ggml_tensor_overhead() * 16 + ggml_graph_overhead_custom(16, false), nullptr, true};
    sort_graph g;
    g.ctx.reset(ggml_init(params));
    require(bool(g.ctx), "cannot create sort context");
    g.input = ggml_new_tensor_2d(g.ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_set_input(g.input);
    g.sorted = native ? ggml_top_k(g.ctx.get(), g.input, used) :
                        ggml_argsort(g.ctx.get(), g.input, ascending ? GGML_SORT_ORDER_ASC : GGML_SORT_ORDER_DESC);
    g.output = native || used == width ? g.sorted : ggml_view_2d(g.ctx.get(), g.sorted, used, rows, g.sorted->nb[1], 0);
    ggml_set_name(g.sorted, native ? "top_k" : used == width ? "argsort" : "argsort_prefix");
    auto * sink = native || used == width ? g.output : ggml_get_rows(g.ctx.get(), ggml_reshape_3d(g.ctx.get(), g.input, 1, width, rows), g.output);
    ggml_set_output(sink);
    g.graph = ggml_new_graph_custom(g.ctx.get(), 16, false);
    ggml_build_forward_expand(g.graph, sink);
    return g;
}

static void compare_regressions() {
    for (bool native : {false, true}) {
        for (bool ties : {false, true}) {
            auto g = make_graph(16, 2, 3, native);
            layer_operator_compare compare;
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(g.ctx.get(), compare.cpu.get()));
            require(bool(buffer), "cannot allocate CPU sort tensors");
            std::vector<float> input(32);
            for (size_t i = 0; i < input.size(); ++i) { input[i] = ties ? 1.0f : float(i); }
            ggml_backend_tensor_set(g.input, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_dayan_slot_trace_event event{};
            event.version = 4;
            event.kind = native ? "top_k" : "argsort";
            event.nodes[0] = event.outputs[0] = g.sorted;
            event.node_count = event.output_count = 1;
            event.sort_k = 3;
            event.executed = true;
            for (int defect = 0; defect < 4; ++defect) {
                layer_operator_compare::before(&event, &compare);
                std::vector<int32_t> indices(ggml_nelements(g.sorted), -1);
                for (int row = 0; row < 2; ++row) {
                    for (int i = 0; i < 3; ++i) { indices[row * g.sorted->ne[0] + i] = 15 - i; }
                    if (native) { std::swap(indices[row * 3], indices[row * 3 + 2]); }
                }
                if (defect == 1) { indices[1] = indices[0]; }
                if (defect == 2) { indices[0] = 16; }
                if (defect == 3) { indices[0] = ties ? -1 : 0; }
                ggml_backend_tensor_set(g.sorted, indices.data(), 0, indices.size() * sizeof(int32_t));
                const size_t failures = compare.failures;
                layer_operator_compare::after(&event, &compare);
                require(compare.failures == failures + (defect != 0), "sort comparator accepted an invalid result or rejected a valid result");
                require(compare.nodes.back()["checked_elements"] == 6, "sort comparator checked an unused suffix");
            }
        }
    }
    std::puts("sort comparator regressions: PASS");
}

static void device_case(ggml_backend_t dayan, int width, int rows, int used, bool native, bool ascending, bool ties) {
    auto g = make_graph(width, rows, used, native, ascending);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(g.ctx.get(), dayan));
    require(bool(buffer), "cannot allocate Dayan sort tensors");
    layer_operator_compare compare;
    compare.attach(ggml_backend_get_device(dayan));
    for (uint32_t repeat = 0; repeat < 2; ++repeat) {
        std::vector<float> input((size_t) width * rows);
        uint32_t state = 35 + repeat;
        for (auto & value : input) {
            state = state * 1664525U + 1013904223U;
            value = ties ? float(int(state % 17) - 8) : float(state >> 8) / 8388608 - 1;
        }
        if (input.size() >= 4) {
            input[0] = 0.0f;
            input[1] = -0.0f;
            input[2] = std::numeric_limits<float>::infinity();
            input[3] = -std::numeric_limits<float>::infinity();
        }
        ggml_backend_tensor_set(g.input, input.data(), 0, input.size() * sizeof(float));
        require(ggml_backend_graph_compute(dayan, g.graph) == GGML_STATUS_SUCCESS, "Dayan sorting failed");
        require(compare.failures == 0, "Dayan sorting differs from CPU");
        const auto actual = layer_operator_compare::values<int32_t>(g.output);
        std::vector<int32_t> expected(width);
        for (int row = 0; row < rows; ++row) {
            std::iota(expected.begin(), expected.end(), 0);
            std::stable_sort(expected.begin(), expected.end(), [&](int a, int b) {
                return ascending ? input[row * width + a] < input[row * width + b] : input[row * width + a] > input[row * width + b];
            });
            for (int i = 0; i < used; ++i) {
                require(actual[row * used + i] == expected[i], "Dayan sort indices or tie order differ from the stable reference");
            }
        }
    }
    std::printf("sort width=%d rows=%d used=%d native=%d asc=%d ties=%d: PASS\n", width, rows, used, native, ascending, ties);
}

int main(int argc, char ** argv) {
    try {
        compare_regressions();
        if (argc == 2 && std::string(argv[1]) == "--cpu-only") { return 0; }
        require(argc == 1, "usage: test-dayan-sort [--cpu-only]");
        ggml_backend_load_all();
        ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
        require(bool(dayan), "cannot initialize Dayan");
        device_case(dayan.get(), 1, 1, 1, false, true, false);
        device_case(dayan.get(), 63, 3, 63, false, false, true);
        device_case(dayan.get(), 65, 5, 8, false, false, false);
        device_case(dayan.get(), 65, 5, 8, false, true, false);
        device_case(dayan.get(), 129, 3, 129, false, true, true);
        device_case(dayan.get(), 256, 1, 8, true, false, true);
        device_case(dayan.get(), 256, 1025, 8, false, false, true);
        device_case(dayan.get(), 4096, 1, 8, true, false, false);
        device_case(dayan.get(), 4096, 3, 4096, false, false, false);
        device_case(dayan.get(), 256, 9, 33, false, false, false);
        device_case(dayan.get(), 65, 7, 1, true, false, true);
        device_case(dayan.get(), 512, 2, 4, true, false, false);
        device_case(dayan.get(), 129, 5, 2, true, false, true);
        device_case(dayan.get(), 256, 4, 7, false, false, false);
        device_case(dayan.get(), 256, 5, 32, true, false, false);
        device_case(dayan.get(), 256, 2, 64, false, false, true);
        std::puts("test-dayan-sort: PASS");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-dayan-sort: %s\n", error.what());
        return 1;
    }
}

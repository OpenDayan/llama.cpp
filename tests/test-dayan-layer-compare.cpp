#include "layer-compare.h"

#include <limits>

static void require(bool condition, const char * message) {
    if (!condition) { throw std::runtime_error(message); }
}

static layer_logit_compare compare(std::initializer_list<float> observed, std::initializer_list<float> reference) {
    layer_logit_compare result;
    auto expected = reference.begin();
    for (float value : observed) { result.add(value, *expected++, 0.03, 0.03); }
    return result;
}

static void compare_gated_projection() {
    ggml_init_params params{ggml_tensor_overhead() * 16, nullptr, true};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    auto * qg = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 8, 1);
    auto * gate_view = ggml_view_3d(ctx.get(), qg, 2, 2, 1, 4 * sizeof(float), 8 * sizeof(float), 2 * sizeof(float));
    auto * gate_cont = ggml_cont_2d(ctx.get(), gate_view, 4, 1);
    auto * sigmoid = ggml_sigmoid(ctx.get(), gate_cont);
    auto * attention = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 1);
    auto * gated = ggml_mul(ctx.get(), attention, sigmoid);
    auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 2);
    auto * output = ggml_mul_mat(ctx.get(), weight, gated);
    layer_operator_compare comparison;
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), comparison.cpu.get()), ggml_backend_buffer_free);
    require(buffer != nullptr, "cannot allocate gated projection test");
    const float qg_data[] = {10, 20, 0, 0, 30, 40, 0, 0};
    const float attention_data[] = {2, 4, 6, 8};
    const float weight_data[] = {1, 0, 0, 0, 0, 1, 0, 0};
    const float unused_gate[] = {-100, -100, -100, -100};
    ggml_backend_tensor_set(qg, qg_data, 0, sizeof(qg_data));
    ggml_backend_tensor_set(attention, attention_data, 0, sizeof(attention_data));
    ggml_backend_tensor_set(weight, weight_data, 0, sizeof(weight_data));
    ggml_backend_tensor_set(gate_cont, unused_gate, 0, sizeof(unused_gate));
    ggml_backend_dayan_slot_trace_event event{};
    event.kind = "gated_output_projection";
    event.nodes[0] = sigmoid;
    event.nodes[1] = gated;
    event.nodes[2] = output;
    event.nodes[3] = qg;
    event.outputs[0] = output;
    event.output_count = 1;
    comparison.reference(&event);
    require(comparison.references.at(output) == std::vector<float>({1, 2}), "gated projection read an unused intermediate");
}

int main() {
    try {
        compare_gated_projection();
        const auto equal = compare({1, -2, 0}, {1, -2, 0});
        require(equal.passes(false, 0.35, 0.93) && equal.passes(true, 0.35, 0.93), "equal logits failed");

        const auto quantized = compare({0.8f, -1.6f, 0}, {1, -2, 0});
        require(!quantized.passes(false, 0.35, 0.93), "strict check ignored elementwise mismatches");
        require(quantized.passes(true, 0.35, 0.93), "MXFP4 check rejected bounded quantization error");
        require(quantized.report().at("mismatched_elements") == 2, "MXFP4 check lost diagnostic mismatches");

        const auto scaled = compare({0.6f, -1.2f, 0}, {1, -2, 0});
        require(!scaled.passes(true, 0.35, 0.93), "high cosine concealed excessive L2 error");
        require(scaled.passes(true, 0.5, 0.93), "L2 override ignored");
        const auto rotated = compare({0, 1}, {1, 0});
        require(!rotated.passes(true, 2, 0.93), "L2 override disabled cosine check");
        require(rotated.passes(true, 2, 0), "cosine override ignored");

        require(compare({0, 0}, {0, 0}).passes(true, 0, 1), "equal zero logits failed");
        require(!compare({0, 0}, {1, 2}).passes(true, 0.35, 0.93), "zero output passed");
        require(!compare({1, 2}, {0, 0}).passes(true, 0.35, 0.93), "nonzero output passed zero reference");
        require(!compare({}, {}).passes(true, 0.35, 0.93), "empty comparison passed");

        for (float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            for (bool invalid_reference : {false, true}) {
                bool rejected = false;
                try { compare({invalid_reference ? 0 : invalid}, {invalid_reference ? invalid : 0}); }
                catch (const std::runtime_error &) { rejected = true; }
                require(rejected, "nonfinite logits accepted");
            }
        }
        std::puts("PASS: strict and MXFP4 logit comparison");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}

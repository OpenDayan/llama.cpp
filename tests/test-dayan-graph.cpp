#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using ggml_backend_graph_rewrite_t = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-graph: %s\n", message);
    return 1;
}

static uint16_t encode_lowp(enum ggml_type type, float value) {
    if (type == GGML_TYPE_BF16) {
        return ggml_fp32_to_bf16(value).bits;
    }
    return ggml_fp32_to_fp16(value);
}

static float decode_lowp(enum ggml_type type, uint16_t value) {
    if (type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(ggml_bf16_t{ value });
    }
    return ggml_fp16_to_fp32(value);
}

static bool rewrite_graph(ggml_backend_t dayan, ggml_context * ctx, ggml_cgraph * graph) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    auto rewrite = (ggml_backend_graph_rewrite_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    return rewrite != nullptr && rewrite(dayan, ctx, graph);
}

static int run_fused_case(ggml_backend_t   dayan,
                          ggml_backend_t   cpu,
                          enum ggml_type   type,
                          enum ggml_glu_op glu_op,
                          int64_t          m) {
    constexpr int64_t n          = 16;
    constexpr int64_t k          = 256;
    constexpr int64_t n_down     = 16;
    constexpr int     graph_size = 32;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the ggml context");
    }

    ggml_tensor * activation  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx.get(), type, k, n);
    ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx.get(), type, k, n);
    ggml_tensor * down_weight = ggml_new_tensor_2d(ctx.get(), type, n, n_down);
    ggml_set_name(activation, "activation");
    ggml_set_name(gate_weight, "gate_weight");
    ggml_set_name(up_weight, "up_weight");
    ggml_set_name(down_weight, "down_weight");
    ggml_set_input(activation);
    ggml_set_input(gate_weight);
    ggml_set_input(up_weight);
    ggml_set_input(down_weight);

    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_weight, activation);
    ggml_tensor * up   = ggml_mul_mat(ctx.get(), up_weight, activation);
    ggml_tensor * fused =
        glu_op == GGML_GLU_OP_SWIGLU ? ggml_swiglu_split(ctx.get(), gate, up) : ggml_geglu_split(ctx.get(), gate, up);
    ggml_tensor * down = ggml_mul_mat(ctx.get(), down_weight, fused);
    ggml_set_name(gate, "gate");
    ggml_set_name(up, "up");
    ggml_set_name(fused, "fused");
    ggml_set_name(down, "down");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, down);
    if (!rewrite_graph(dayan, ctx.get(), graph)) {
        return fail("DAYAN graph rewrite failed");
    }

    ggml_tensor * activation_lowp = gate->src[1];
    ggml_tensor * fused_lowp      = down->src[1];
    if (activation_lowp == activation || activation_lowp != up->src[1] || activation_lowp->op != GGML_OP_CPY ||
        activation_lowp->type != type || fused_lowp == fused || fused_lowp->op != GGML_OP_CPY ||
        fused_lowp->src[0] != fused || fused_lowp->type != type) {
        return fail("the graph does not contain the expected shared input and fused output casts");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate graph tensors");
    }

    std::vector<float>    activation_data((size_t) (k * m));
    std::vector<uint16_t> gate_weight_data((size_t) (k * n));
    std::vector<uint16_t> up_weight_data((size_t) (k * n));
    std::vector<uint16_t> down_weight_data((size_t) (n * n_down));

    for (size_t i = 0; i < activation_data.size(); ++i) {
        activation_data[i] = ((int) (i % 19) - 9) / 32.0f;
    }
    for (size_t i = 0; i < gate_weight_data.size(); ++i) {
        gate_weight_data[i] = encode_lowp(type, ((int) (i % 13) - 6) / 32.0f);
        up_weight_data[i]   = encode_lowp(type, ((int) (i % 11) - 5) / 32.0f);
    }
    for (size_t i = 0; i < down_weight_data.size(); ++i) {
        down_weight_data[i] = encode_lowp(type, ((int) (i % 7) - 3) / 32.0f);
    }

    ggml_backend_tensor_set(activation, activation_data.data(), 0, ggml_nbytes(activation));
    ggml_backend_tensor_set(gate_weight, gate_weight_data.data(), 0, ggml_nbytes(gate_weight));
    ggml_backend_tensor_set(up_weight, up_weight_data.data(), 0, ggml_nbytes(up_weight));
    ggml_backend_tensor_set(down_weight, down_weight_data.data(), 0, ggml_nbytes(down_weight));

    ggml_graph_clear(graph);
    ggml_graph_add_node(graph, activation_lowp);
    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU activation cast failed");
    }

    ggml_graph_clear(graph);
    ggml_graph_add_node(graph, gate);
    ggml_graph_add_node(graph, up);
    ggml_graph_add_node(graph, fused);
    ggml_graph_add_node(graph, fused_lowp);
    ggml_graph_add_node(graph, down);
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN fused matrix multiplication graph failed");
    }

    std::vector<uint16_t> fused_result((size_t) (m * n));
    std::vector<float>    down_result((size_t) (m * n_down));
    ggml_backend_tensor_get(fused_lowp, fused_result.data(), 0, ggml_nbytes(fused_lowp));
    ggml_backend_tensor_get(down, down_result.data(), 0, ggml_nbytes(down));

    float max_abs        = 0.0f;
    float max_rel        = 0.0f;
    float worst_actual   = 0.0f;
    float worst_expected = 0.0f;
    bool  close          = true;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            float gate_ref = 0.0f;
            float up_ref   = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                const float act = decode_lowp(type, encode_lowp(type, activation_data[(size_t) (row * k + inner)]));
                gate_ref += decode_lowp(type, gate_weight_data[(size_t) (col * k + inner)]) * act;
                up_ref += decode_lowp(type, up_weight_data[(size_t) (col * k + inner)]) * act;
            }

            float fused_ref;
            if (glu_op == GGML_GLU_OP_SWIGLU) {
                fused_ref = gate_ref / (1.0f + std::exp(-gate_ref)) * up_ref;
            } else {
                constexpr float gelu_a         = 0.044715f;
                constexpr float sqrt_2_over_pi = 0.7978845608028654f;
                fused_ref = 0.5f * gate_ref *
                            (1.0f + std::tanh(sqrt_2_over_pi * gate_ref * (1.0f + gelu_a * gate_ref * gate_ref))) *
                            up_ref;
            }
            fused_ref = decode_lowp(type, encode_lowp(type, fused_ref));

            const float actual    = decode_lowp(type, fused_result[(size_t) (row * n + col)]);
            const float abs_error = std::abs(actual - fused_ref);
            const float rel_error = abs_error / std::max(std::abs(fused_ref), 1.0e-6f);
            close                 = close && abs_error <= 0.02f + 0.08f * std::abs(fused_ref);
            if (abs_error > max_abs) {
                max_abs        = abs_error;
                worst_actual   = actual;
                worst_expected = fused_ref;
            }
            max_rel = std::max(max_rel, rel_error);
        }
    }

    std::printf("type=%s op=%s M=%lld N=%lld K=%lld elements=%lld max_abs=%g max_rel=%g worst=%g expected=%g\n",
                ggml_type_name(type), ggml_glu_op_name(glu_op), (long long) m, (long long) n, (long long) k,
                (long long) (m * n), max_abs, max_rel, worst_actual, worst_expected);
    if (!close) {
        return fail("numerical comparison failed");
    }

    float down_max_abs = 0.0f;
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n_down; ++col) {
            float expected = 0.0f;
            for (int64_t inner = 0; inner < n; ++inner) {
                expected += decode_lowp(type, down_weight_data[(size_t) (col * n + inner)]) *
                            decode_lowp(type, fused_result[(size_t) (row * n + inner)]);
            }
            down_max_abs = std::max(down_max_abs, std::abs(down_result[(size_t) (row * n_down + col)] - expected));
        }
    }
    std::printf("type=%s downstream max_abs=%g\n", ggml_type_name(type), down_max_abs);
    if (down_max_abs > 0.001f) {
        return fail("downstream matrix multiplication comparison failed");
    }
    return 0;
}

static int run_gated_output_case(ggml_backend_t dayan, ggml_backend_t cpu, enum ggml_type type) {
    constexpr int64_t tokens     = 3;
    constexpr int64_t embd       = 64;
    constexpr int64_t head_dim   = 64;
    constexpr int64_t heads      = 2;
    constexpr int64_t width      = head_dim * heads;
    constexpr int64_t output_dim = 64;
    constexpr int     graph_size = 32;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the gated output context");
    }

    ggml_tensor * activation    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, embd, tokens);
    ggml_tensor * qg_weight     = ggml_new_tensor_2d(ctx.get(), type, embd, 2 * width);
    ggml_tensor * attention     = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
    ggml_tensor * output_weight = ggml_new_tensor_2d(ctx.get(), type, width, output_dim);
    ggml_set_input(activation);
    ggml_set_input(qg_weight);
    ggml_set_input(attention);
    ggml_set_input(output_weight);

    ggml_tensor * qg           = ggml_mul_mat(ctx.get(), qg_weight, activation);
    ggml_tensor * gate_view    = ggml_view_3d(ctx.get(), qg, head_dim, heads, tokens, 2 * head_dim * sizeof(float),
                                              2 * width * sizeof(float), head_dim * sizeof(float));
    ggml_tensor * gate         = ggml_cont_2d(ctx.get(), gate_view, width, tokens);
    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx.get(), gate);
    ggml_tensor * gated        = ggml_mul(ctx.get(), attention, gate_sigmoid);
    ggml_tensor * output       = ggml_mul_mat(ctx.get(), output_weight, gated);
    ggml_set_name(qg, "qg");
    ggml_set_name(gate, "gate");
    ggml_set_name(gate_sigmoid, "gate_sigmoid");
    ggml_set_name(gated, "gated_attention");
    ggml_set_name(output, "gated_output");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    if (!rewrite_graph(dayan, ctx.get(), graph)) {
        return fail("DAYAN gated output graph rewrite failed");
    }
    ggml_tensor * activation_lowp = qg->src[1];
    if (activation_lowp == activation || activation_lowp->op != GGML_OP_CPY || activation_lowp->type != type ||
        output->src[1] != gated) {
        return fail("unexpected gated output activation rewrite");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate gated output tensors");
    }

    std::vector<float>    activation_data((size_t) embd * tokens);
    std::vector<float>    attention_data((size_t) width * tokens);
    std::vector<uint16_t> qg_weight_data((size_t) embd * 2 * width);
    std::vector<uint16_t> output_weight_data((size_t) width * output_dim);
    for (size_t index = 0; index < activation_data.size(); ++index) {
        activation_data[index] = ((int) (index % 23) - 11) / 64.0f;
    }
    for (size_t index = 0; index < attention_data.size(); ++index) {
        attention_data[index] = ((int) (index % 29) - 14) / 64.0f;
    }
    for (size_t index = 0; index < qg_weight_data.size(); ++index) {
        qg_weight_data[index] = encode_lowp(type, ((int) (index % 17) - 8) / 128.0f);
    }
    for (size_t index = 0; index < output_weight_data.size(); ++index) {
        output_weight_data[index] = encode_lowp(type, ((int) (index % 13) - 6) / 128.0f);
    }
    ggml_backend_tensor_set(activation, activation_data.data(), 0, ggml_nbytes(activation));
    ggml_backend_tensor_set(attention, attention_data.data(), 0, ggml_nbytes(attention));
    ggml_backend_tensor_set(qg_weight, qg_weight_data.data(), 0, ggml_nbytes(qg_weight));
    ggml_backend_tensor_set(output_weight, output_weight_data.data(), 0, ggml_nbytes(output_weight));

    ggml_graph_clear(graph);
    ggml_graph_add_node(graph, activation_lowp);
    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU gated output activation cast failed");
    }
    ggml_graph_clear(graph);
    ggml_graph_add_node(graph, qg);
    ggml_graph_add_node(graph, gate_view);
    ggml_graph_add_node(graph, gate);
    ggml_graph_add_node(graph, gate_sigmoid);
    ggml_graph_add_node(graph, gated);
    ggml_graph_add_node(graph, output);
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN gated output graph failed");
    }

    std::vector<float> observed((size_t) tokens * output_dim);
    ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));
    std::vector<float> qg_reference((size_t) tokens * 2 * width);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t column = 0; column < 2 * width; ++column) {
            float sum = 0.0f;
            for (int64_t inner = 0; inner < embd; ++inner) {
                const float input =
                    decode_lowp(type, encode_lowp(type, activation_data[(size_t) token * embd + inner]));
                sum += input * decode_lowp(type, qg_weight_data[(size_t) column * embd + inner]);
            }
            qg_reference[(size_t) token * 2 * width + column] = sum;
        }
    }
    std::vector<uint16_t> gated_lowp((size_t) tokens * width);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t head = 0; head < heads; ++head) {
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const size_t index      = (size_t) token * width + head * head_dim + dim;
                const size_t gate_index = (size_t) token * 2 * width + head * 2 * head_dim + head_dim + dim;
                const float  gate_value = decode_lowp(type, encode_lowp(type, qg_reference[gate_index]));
                gated_lowp[index]       = encode_lowp(type, attention_data[index] / (1.0f + std::exp(-gate_value)));
            }
        }
    }

    float max_abs = 0.0f;
    bool  close   = true;
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t column = 0; column < output_dim; ++column) {
            float expected = 0.0f;
            for (int64_t inner = 0; inner < width; ++inner) {
                expected += decode_lowp(type, gated_lowp[(size_t) token * width + inner]) *
                            decode_lowp(type, output_weight_data[(size_t) column * width + inner]);
            }
            const float error = std::abs(observed[(size_t) token * output_dim + column] - expected);
            max_abs           = std::max(max_abs, error);
            close             = close && error <= 0.03f + 0.03f * std::abs(expected);
        }
    }
    std::printf("type=%s gated output tokens=%lld width=%lld output=%lld max_abs=%g\n", ggml_type_name(type),
                (long long) tokens, (long long) width, (long long) output_dim, max_abs);
    return close ? 0 : fail("gated output numerical comparison failed");
}

static int run_ssm_conv_case(ggml_backend_t dayan,
                             int64_t        kernel_size,
                             int64_t        tokens,
                             int64_t        channels,
                             int64_t        sequences,
                             bool           silu) {
    const int64_t input_length = kernel_size - 1 + tokens;
    constexpr int graph_size   = 8;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the SSM convolution context");
    }
    ggml_tensor * input  = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input_length, channels, sequences);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, kernel_size, channels);
    ggml_set_input(input);
    ggml_set_input(weight);
    ggml_tensor * conv   = ggml_ssm_conv(ctx.get(), input, weight);
    ggml_tensor * output = silu ? ggml_silu(ctx.get(), conv) : conv;
    ggml_cgraph * graph  = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate SSM convolution tensors");
    }
    std::vector<float> input_data((size_t) input_length * channels * sequences);
    std::vector<float> weight_data((size_t) kernel_size * channels);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = ((int) (index % 31) - 15) / 32.0f;
    }
    for (size_t index = 0; index < weight_data.size(); ++index) {
        weight_data[index] = ((int) (index % 11) - 5) / 32.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, ggml_nbytes(weight));
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN SSM convolution graph failed");
    }

    std::vector<float> observed((size_t) channels * tokens * sequences);
    ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));
    float  max_abs        = 0.0f;
    float  worst_observed = 0.0f;
    float  worst_expected = 0.0f;
    size_t worst_index    = 0;
    bool   close          = true;
    for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        for (int64_t token = 0; token < tokens; ++token) {
            for (int64_t channel = 0; channel < channels; ++channel) {
                float        sum         = 0.0f;
                const size_t input_base  = ((size_t) sequence * channels + channel) * input_length + token;
                const size_t weight_base = (size_t) channel * kernel_size;
                for (int64_t tap = 0; tap < kernel_size; ++tap) {
                    sum += input_data[input_base + tap] * weight_data[weight_base + tap];
                }
                const float  expected = silu ? sum / (1.0f + std::exp(-sum)) : sum;
                const size_t index    = ((size_t) sequence * tokens + token) * channels + channel;
                const float  error    = std::abs(observed[index] - expected);
                if (error > max_abs) {
                    max_abs        = error;
                    worst_observed = observed[index];
                    worst_expected = expected;
                    worst_index    = index;
                }
                close = close && error <= 0.002f + 0.002f * std::abs(expected);
            }
        }
    }
    std::printf(
        "SSM conv%s kernel=%lld tokens=%lld channels=%lld sequences=%lld max_abs=%g worst=%zu observed=%g "
        "expected=%g\n",
        silu ? " SiLU" : "", (long long) kernel_size, (long long) tokens, (long long) channels, (long long) sequences,
        max_abs, worst_index, worst_observed, worst_expected);
    return close ? 0 : fail("SSM convolution numerical comparison failed");
}

int main() {
    ggml_backend_load_all();

    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    if (!dayan) {
        return fail("cannot initialize the DAYAN backend");
    }

    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cpu) {
        return fail("cannot initialize the CPU backend");
    }

    if (run_fused_case(dayan.get(), cpu.get(), GGML_TYPE_BF16, GGML_GLU_OP_SWIGLU, 1) != 0 ||
        run_fused_case(dayan.get(), cpu.get(), GGML_TYPE_F16, GGML_GLU_OP_SWIGLU, 16) != 0 ||
        run_fused_case(dayan.get(), cpu.get(), GGML_TYPE_BF16, GGML_GLU_OP_GEGLU, 1) != 0 ||
        run_fused_case(dayan.get(), cpu.get(), GGML_TYPE_F16, GGML_GLU_OP_GEGLU, 16) != 0 ||
        run_gated_output_case(dayan.get(), cpu.get(), GGML_TYPE_BF16) != 0 ||
        run_gated_output_case(dayan.get(), cpu.get(), GGML_TYPE_F16) != 0 ||
        run_ssm_conv_case(dayan.get(), 1, 5, 128, 1, false) != 0 ||
        run_ssm_conv_case(dayan.get(), 4, 5, 130, 2, true) != 0 ||
        run_ssm_conv_case(dayan.get(), 4, 69, 384, 1, true) != 0) {
        return 1;
    }

    std::puts("test-dayan-graph: PASS");
    return 0;
}

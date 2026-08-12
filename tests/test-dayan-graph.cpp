#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

static bool has_dependency(const ggml_tensor * tensor, const ggml_tensor * dependency) {
    for (int index = 2; index < GGML_MAX_SRC; ++index) {
        if (tensor->src[index] == dependency) {
            return true;
        }
    }
    return false;
}

static int run_fused_case(ggml_backend_t   dayan,
                          enum ggml_type   type,
                          enum ggml_glu_op glu_op,
                          int64_t          m,
                          int64_t          n,
                          int64_t          k) {
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
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN activation cast failed");
    }

    ggml_graph_clear(graph);
    ggml_build_forward_expand(graph, down);
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

static int run_fusion_fanout_case(ggml_backend_t dayan) {
    constexpr int64_t width      = 16;
    constexpr int     graph_size = 32;
    ggml_init_params  params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the fusion fanout context");
    }

    ggml_tensor * activation  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
    ggml_tensor * gate_weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, width, width);
    ggml_tensor * up_weight   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, width, width);
    ggml_tensor * down_weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, width, width);
    ggml_tensor * gate        = ggml_mul_mat(ctx.get(), gate_weight, activation);
    ggml_tensor * up          = ggml_mul_mat(ctx.get(), up_weight, activation);
    ggml_tensor * fused       = ggml_swiglu_split(ctx.get(), gate, up);
    ggml_tensor * down        = ggml_mul_mat(ctx.get(), down_weight, fused);
    ggml_tensor * gate_sum    = ggml_sum(ctx.get(), gate);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, down);
    ggml_build_forward_expand(graph, gate_sum);
    if (!rewrite_graph(dayan, ctx.get(), graph)) {
        return fail("DAYAN fusion fanout rewrite failed");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer || ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_FAILED) {
        return fail("DAYAN fused a matrix result with another consumer");
    }
    return 0;
}

static int run_layout_cast_hoist_case(ggml_backend_t dayan, enum ggml_type type) {
    constexpr int64_t m          = 2;
    constexpr int64_t k          = 16;
    constexpr int64_t n          = 32;
    constexpr int64_t output_dim = 16;
    constexpr int     graph_size = 32;
    ggml_init_params  params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the layout cast hoist context");
    }

    ggml_tensor * activation      = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, m);
    ggml_tensor * producer_weight = ggml_new_tensor_2d(ctx.get(), type, k, n);
    ggml_tensor * output_weight   = ggml_new_tensor_2d(ctx.get(), type, n, output_dim);
    ggml_set_input(activation);
    ggml_set_input(producer_weight);
    ggml_set_input(output_weight);

    ggml_tensor * producer = ggml_mul_mat(ctx.get(), producer_weight, activation);
    ggml_tensor * view     = ggml_view_2d(ctx.get(), producer, n, m, producer->nb[1], 0);
    ggml_tensor * reshaped = ggml_reshape_4d(ctx.get(), view, n, m, 1, 1);
    ggml_tensor * permuted = ggml_permute(ctx.get(), reshaped, 0, 1, 3, 2);
    ggml_tensor * output   = ggml_mul_mat(ctx.get(), output_weight, permuted);
    ggml_set_name(producer, "layout producer");
    ggml_set_name(output, "layout consumer");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    if (!rewrite_graph(dayan, ctx.get(), graph)) {
        return fail("DAYAN layout cast hoist rewrite failed");
    }

    ggml_tensor * permuted_lowp = output->src[1];
    ggml_tensor * reshaped_lowp = permuted_lowp->src[0];
    ggml_tensor * view_lowp     = reshaped_lowp->src[0];
    ggml_tensor * producer_lowp = view_lowp->src[0];
    if (producer_lowp == producer || producer_lowp->op != GGML_OP_CPY || producer_lowp->src[0] != producer ||
        producer_lowp->type != type || view_lowp->type != type || reshaped_lowp->type != type ||
        permuted_lowp->type != type || view_lowp->view_src != producer_lowp ||
        reshaped_lowp->view_src != producer_lowp || permuted_lowp->view_src != producer_lowp ||
        view_lowp->op != GGML_OP_VIEW || reshaped_lowp->op != GGML_OP_RESHAPE || permuted_lowp->op != GGML_OP_PERMUTE ||
        view->type != GGML_TYPE_F32 || reshaped->type != GGML_TYPE_F32 || permuted->type != GGML_TYPE_F32) {
        return fail("DAYAN did not move the cast before the layout chain");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate layout cast hoist tensors");
    }

    std::vector<float>    activation_data((size_t) k * m);
    std::vector<uint16_t> producer_weight_data((size_t) k * n);
    std::vector<uint16_t> output_weight_data((size_t) n * output_dim);
    for (size_t index = 0; index < activation_data.size(); ++index) {
        activation_data[index] = ((int) (index % 17) - 8) / 32.0f;
    }
    for (size_t index = 0; index < producer_weight_data.size(); ++index) {
        producer_weight_data[index] = encode_lowp(type, ((int) (index % 13) - 6) / 32.0f);
    }
    for (size_t index = 0; index < output_weight_data.size(); ++index) {
        output_weight_data[index] = encode_lowp(type, ((int) (index % 11) - 5) / 32.0f);
    }
    ggml_backend_tensor_set(activation, activation_data.data(), 0, ggml_nbytes(activation));
    ggml_backend_tensor_set(producer_weight, producer_weight_data.data(), 0, ggml_nbytes(producer_weight));
    ggml_backend_tensor_set(output_weight, output_weight_data.data(), 0, ggml_nbytes(output_weight));
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS ||
        ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN layout cast hoist graph failed");
    }

    std::vector<float> observed((size_t) m * output_dim);
    ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));
    float max_abs = 0.0f;
    bool  close   = true;
    for (int64_t row = 0; row < m; ++row) {
        std::vector<float> intermediate((size_t) n);
        for (int64_t column = 0; column < n; ++column) {
            float sum = 0.0f;
            for (int64_t inner = 0; inner < k; ++inner) {
                const float value = decode_lowp(type, encode_lowp(type, activation_data[(size_t) row * k + inner]));
                sum += value * decode_lowp(type, producer_weight_data[(size_t) column * k + inner]);
            }
            intermediate[(size_t) column] = decode_lowp(type, encode_lowp(type, sum));
        }
        for (int64_t column = 0; column < output_dim; ++column) {
            float expected = 0.0f;
            for (int64_t inner = 0; inner < n; ++inner) {
                expected +=
                    intermediate[(size_t) inner] * decode_lowp(type, output_weight_data[(size_t) column * n + inner]);
            }
            const float error = std::abs(observed[(size_t) row * output_dim + column] - expected);
            max_abs           = std::max(max_abs, error);
            close             = close && error <= 0.01f + 0.01f * std::abs(expected);
        }
    }
    std::printf("type=%s layout cast hoist max_abs=%g\n", ggml_type_name(type), max_abs);
    return close ? 0 : fail("layout cast hoist numerical comparison failed");
}

static int run_norm_matmul_case(ggml_backend_t dayan, enum ggml_type type, int64_t width, bool has_residual) {
    constexpr int64_t rows       = 3;
    constexpr int64_t output_dim = 16;
    constexpr float   epsilon    = 1.0e-6f;
    constexpr int     graph_size = 16;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the RMSNorm matrix multiplication context");
    }

    ggml_tensor * left   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_tensor * right  = has_residual ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows) : nullptr;
    ggml_tensor * scale  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), type, width, output_dim);
    ggml_set_input(left);
    if (right != nullptr) {
        ggml_set_input(right);
    }
    ggml_set_input(scale);
    ggml_set_input(weight);

    ggml_tensor * residual = has_residual ? ggml_add(ctx.get(), left, right) : left;
    ggml_tensor * rms      = ggml_rms_norm(ctx.get(), residual, epsilon);
    ggml_tensor * scaled   = ggml_mul(ctx.get(), rms, scale);
    ggml_tensor * output   = ggml_mul_mat(ctx.get(), weight, scaled);
    ggml_set_name(residual, has_residual ? "residual" : "input");
    ggml_set_name(rms, "rms");
    ggml_set_name(scaled, "scaled");
    ggml_set_name(output, "norm_matmul");
    if (has_residual) {
        ggml_set_output(residual);
    }
    ggml_set_output(rms);
    ggml_set_output(scaled);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    if (!rewrite_graph(dayan, ctx.get(), graph)) {
        return fail("DAYAN RMSNorm matrix multiplication rewrite failed");
    }

    ggml_tensor * lowp = output->src[1];
    if (lowp == scaled || lowp->op != GGML_OP_CPY || lowp->src[0] != scaled || lowp->type != type ||
        !has_dependency(lowp, scale) || !has_dependency(lowp, left) || (has_residual && !has_dependency(lowp, right))) {
        return fail("unexpected RMSNorm matrix multiplication activation rewrite");
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate RMSNorm matrix multiplication tensors");
    }

    std::vector<float>    left_data((size_t) width * rows);
    std::vector<float>    right_data((size_t) width * rows);
    std::vector<float>    scale_data((size_t) width);
    std::vector<uint16_t> weight_data((size_t) width * output_dim);
    for (size_t index = 0; index < left_data.size(); ++index) {
        left_data[index]  = ((int) (index % 29) - 14) / 64.0f;
        right_data[index] = ((int) (index % 17) - 8) / 96.0f;
    }
    for (size_t index = 0; index < scale_data.size(); ++index) {
        scale_data[index] = 0.75f + (float) (index % 11) / 32.0f;
    }
    for (size_t index = 0; index < weight_data.size(); ++index) {
        weight_data[index] = encode_lowp(type, ((int) (index % 23) - 11) / 128.0f);
    }
    ggml_backend_tensor_set(left, left_data.data(), 0, ggml_nbytes(left));
    if (right != nullptr) {
        ggml_backend_tensor_set(right, right_data.data(), 0, ggml_nbytes(right));
    }
    ggml_backend_tensor_set(scale, scale_data.data(), 0, ggml_nbytes(scale));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, ggml_nbytes(weight));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN RMSNorm matrix multiplication graph failed");
    }

    std::vector<float>    residual_result((size_t) width * rows);
    std::vector<float>    rms_result((size_t) width * rows);
    std::vector<float>    scaled_result((size_t) width * rows);
    std::vector<uint16_t> lowp_result((size_t) width * rows);
    std::vector<float>    output_result((size_t) output_dim * rows);
    if (has_residual) {
        ggml_backend_tensor_get(residual, residual_result.data(), 0, ggml_nbytes(residual));
    }
    ggml_backend_tensor_get(rms, rms_result.data(), 0, ggml_nbytes(rms));
    ggml_backend_tensor_get(scaled, scaled_result.data(), 0, ggml_nbytes(scaled));
    ggml_backend_tensor_get(lowp, lowp_result.data(), 0, ggml_nbytes(lowp));
    ggml_backend_tensor_get(output, output_result.data(), 0, ggml_nbytes(output));

    std::vector<uint16_t> lowp_reference((size_t) width * rows);
    float                 residual_max_abs = 0.0f;
    float                 rms_max_abs      = 0.0f;
    float                 scaled_max_abs   = 0.0f;
    float                 lowp_max_abs     = 0.0f;
    bool                  close            = true;
    for (int64_t row = 0; row < rows; ++row) {
        float square_sum = 0.0f;
        for (int64_t col = 0; col < width; ++col) {
            const size_t index = (size_t) row * width + col;
            const float  value = left_data[index] + (has_residual ? right_data[index] : 0.0f);
            square_sum += value * value;
            if (has_residual) {
                residual_max_abs = std::max(residual_max_abs, std::abs(residual_result[index] - value));
            }
        }
        const float inverse = 1.0f / std::sqrt(square_sum / (float) width + epsilon);
        for (int64_t col = 0; col < width; ++col) {
            const size_t index     = (size_t) row * width + col;
            const float  value     = left_data[index] + (has_residual ? right_data[index] : 0.0f);
            const float  rms_ref   = value * inverse;
            const float  scale_ref = rms_ref * scale_data[(size_t) col];
            const float  lowp_ref  = decode_lowp(type, encode_lowp(type, scale_ref));
            lowp_reference[index]  = encode_lowp(type, scale_ref);
            rms_max_abs            = std::max(rms_max_abs, std::abs(rms_result[index] - rms_ref));
            scaled_max_abs         = std::max(scaled_max_abs, std::abs(scaled_result[index] - scale_ref));
            lowp_max_abs           = std::max(lowp_max_abs, std::abs(decode_lowp(type, lowp_result[index]) - lowp_ref));
        }
    }

    float output_max_abs = 0.0f;
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < output_dim; ++col) {
            float expected = 0.0f;
            for (int64_t inner = 0; inner < width; ++inner) {
                expected += decode_lowp(type, lowp_reference[(size_t) row * width + inner]) *
                            decode_lowp(type, weight_data[(size_t) col * width + inner]);
            }
            const float error = std::abs(output_result[(size_t) row * output_dim + col] - expected);
            output_max_abs    = std::max(output_max_abs, error);
            close             = close && error <= 0.02f + 0.02f * std::abs(expected);
        }
    }
    close = close && residual_max_abs <= 1.0e-6f && rms_max_abs <= 0.005f && scaled_max_abs <= 0.006f &&
            lowp_max_abs <= 0.008f;
    std::printf("type=%s RMSNorm%s width=%lld rows=%lld residual=%g rms=%g scaled=%g lowp=%g matmul=%g\n",
                ggml_type_name(type), has_residual ? "+residual" : "", (long long) width, (long long) rows,
                residual_max_abs, rms_max_abs, scaled_max_abs, lowp_max_abs, output_max_abs);
    return close ? 0 : fail("RMSNorm matrix multiplication numerical comparison failed");
}

static int run_elementwise_case(ggml_backend_t dayan) {
    constexpr int64_t width      = 130;
    constexpr int64_t rows       = 3;
    constexpr float   l2_epsilon = 1.0e-6f;
    constexpr int     graph_size = 16;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the elementwise context");
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_tensor * bias  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    ggml_tensor * gain  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    ggml_set_input(input);
    ggml_set_input(bias);
    ggml_set_input(gain);

    ggml_tensor * added      = ggml_add(ctx.get(), input, bias);
    ggml_tensor * sigmoid    = ggml_sigmoid(ctx.get(), added);
    ggml_tensor * multiplied = ggml_mul(ctx.get(), sigmoid, gain);
    ggml_tensor * scaled     = ggml_scale_bias(ctx.get(), multiplied, 1.25f, -0.125f);
    ggml_tensor * softplus   = ggml_softplus(ctx.get(), scaled);
    ggml_tensor * normalized = ggml_l2_norm(ctx.get(), softplus, l2_epsilon);
    ggml_tensor * output     = ggml_silu(ctx.get(), normalized);
    ggml_set_name(added, "elementwise_add");
    ggml_set_name(sigmoid, "elementwise_sigmoid");
    ggml_set_name(multiplied, "elementwise_mul");
    ggml_set_name(scaled, "elementwise_scale");
    ggml_set_name(softplus, "elementwise_softplus");
    ggml_set_name(normalized, "elementwise_l2_norm");
    ggml_set_name(output, "elementwise_silu");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate elementwise tensors");
    }

    std::vector<float> input_data((size_t) width * rows);
    std::vector<float> bias_data((size_t) width);
    std::vector<float> gain_data((size_t) width);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = ((int) (index % 37) - 18) / 16.0f;
    }
    for (size_t index = 0; index < bias_data.size(); ++index) {
        bias_data[index] = ((int) (index % 13) - 6) / 32.0f;
        gain_data[index] = 0.5f + (float) (index % 11) / 16.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(bias, bias_data.data(), 0, ggml_nbytes(bias));
    ggml_backend_tensor_set(gain, gain_data.data(), 0, ggml_nbytes(gain));
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN elementwise graph failed");
    }

    std::vector<float> observed((size_t) width * rows);
    ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));

    float max_abs = 0.0f;
    bool  close   = true;
    for (int64_t row = 0; row < rows; ++row) {
        std::vector<float> normalized_reference((size_t) width);
        float              square_sum = 0.0f;
        for (int64_t column = 0; column < width; ++column) {
            const size_t index                    = (size_t) row * width + column;
            const float  added_ref                = input_data[index] + bias_data[(size_t) column];
            const float  sigmoid_ref              = 1.0f / (1.0f + std::exp(-added_ref));
            const float  mul_ref                  = sigmoid_ref * gain_data[(size_t) column];
            const float  scale_ref                = mul_ref * 1.25f - 0.125f;
            const float  soft_ref                 = std::log1p(std::exp(scale_ref));
            normalized_reference[(size_t) column] = soft_ref;
            square_sum += soft_ref * soft_ref;
        }
        const float inverse = 1.0f / std::max(std::sqrt(square_sum), l2_epsilon);
        for (int64_t column = 0; column < width; ++column) {
            const size_t index    = (size_t) row * width + column;
            const float  norm_ref = normalized_reference[(size_t) column] * inverse;
            const float  expected = norm_ref / (1.0f + std::exp(-norm_ref));
            const float  error    = std::abs(observed[index] - expected);
            max_abs               = std::max(max_abs, error);
            close                 = close && error <= 0.003f + 0.003f * std::abs(expected);
        }
    }
    std::printf("elementwise width=%lld rows=%lld max_abs=%g\n", (long long) width, (long long) rows, max_abs);
    return close ? 0 : fail("elementwise numerical comparison failed");
}

static int run_empty_scale_case(ggml_backend_t dayan) {
    constexpr int    graph_size = 4;
    ggml_init_params params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the empty Scale context");
    }

    ggml_tensor * storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 16);
    ggml_tensor * empty   = ggml_view_1d(ctx.get(), storage, 0, 0);
    ggml_tensor * scaled  = ggml_scale_inplace(ctx.get(), empty, 0.0f);
    ggml_set_input(storage);
    if (!ggml_backend_supports_op(dayan, scaled)) {
        return fail("DAYAN rejected empty Scale");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, scaled);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer || ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN empty Scale graph failed");
    }
    return 0;
}

static int run_offset_scale_case(ggml_backend_t dayan) {
    constexpr int64_t view_elements    = 18432;
    constexpr int64_t storage_elements = 64 + view_elements;
    constexpr size_t  view_offset      = 64 * sizeof(float);
    constexpr int     graph_size       = 4;
    ggml_init_params  params           = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the offset Scale context");
    }

    ggml_tensor * storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, storage_elements);
    ggml_tensor * view    = ggml_view_1d(ctx.get(), storage, view_elements, view_offset);
    ggml_tensor * scaled  = ggml_scale_inplace(ctx.get(), view, 0.0f);
    ggml_set_input(storage);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, scaled);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer || storage->extra == nullptr || view->extra == nullptr || scaled->extra == nullptr) {
        return fail("DAYAN did not initialize offset Scale buffers");
    }

    std::vector<float> values(storage_elements);
    for (int64_t index = 0; index < storage_elements; ++index) {
        values[(size_t) index] = (float) index + 1.0f;
    }
    ggml_backend_tensor_set(storage, values.data(), 0, ggml_nbytes(storage));
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN offset Scale graph failed");
    }

    std::vector<float> observed(storage_elements);
    ggml_backend_tensor_get(storage, observed.data(), 0, ggml_nbytes(storage));
    for (int64_t index = 0; index < storage_elements; ++index) {
        const bool  in_view  = index >= 64 && index < 64 + view_elements;
        const float expected = in_view ? 0.0f : values[(size_t) index];
        if (observed[(size_t) index] != expected) {
            return fail("DAYAN offset Scale result mismatch");
        }
    }
    return 0;
}

static int run_math_case(ggml_backend_t dayan) {
    constexpr int64_t width        = 130;
    constexpr int64_t rows         = 3;
    constexpr int     output_count = 14;
    constexpr int     graph_size   = 48;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the math operator context");
    }

    ggml_tensor * input    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_tensor * positive = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, rows);
    ggml_tensor * divisor  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    ggml_set_input(input);
    ggml_set_input(positive);
    ggml_set_input(divisor);

    ggml_tensor * outputs[output_count] = {
        ggml_sub(ctx.get(), input, divisor), ggml_div(ctx.get(), input, divisor), ggml_sqr(ctx.get(), input),
        ggml_sqrt(ctx.get(), positive),      ggml_log(ctx.get(), positive),       ggml_sin(ctx.get(), input),
        ggml_cos(ctx.get(), input),          ggml_abs(ctx.get(), input),          ggml_neg(ctx.get(), input),
        ggml_tanh(ctx.get(), input),         ggml_relu(ctx.get(), input),         ggml_gelu(ctx.get(), input),
        ggml_gelu_quick(ctx.get(), input),   ggml_exp(ctx.get(), input),
    };
    const char * names[output_count] = {
        "sub", "div", "sqr", "sqrt", "log", "sin", "cos", "abs", "neg", "tanh", "relu", "gelu", "gelu_quick", "exp",
    };

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    for (int index = 0; index < output_count; ++index) {
        ggml_set_name(outputs[index], names[index]);
        if (!ggml_backend_supports_op(dayan, outputs[index])) {
            return fail("DAYAN rejected a math operator");
        }
        ggml_build_forward_expand(graph, outputs[index]);
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate math operator tensors");
    }

    std::vector<float> input_data((size_t) width * rows);
    std::vector<float> positive_data((size_t) width * rows);
    std::vector<float> divisor_data((size_t) width);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index]    = ((int) (index % 49) - 24) / 16.0f;
        positive_data[index] = 0.25f + (float) (index % 37) / 16.0f;
    }
    for (size_t index = 0; index < divisor_data.size(); ++index) {
        divisor_data[index] = 0.5f + (float) (index % 17) / 16.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(positive, positive_data.data(), 0, ggml_nbytes(positive));
    ggml_backend_tensor_set(divisor, divisor_data.data(), 0, ggml_nbytes(divisor));
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN math operator graph failed");
    }

    bool close = true;
    for (int output_index = 0; output_index < output_count; ++output_index) {
        std::vector<float> observed(input_data.size());
        ggml_backend_tensor_get(outputs[output_index], observed.data(), 0, ggml_nbytes(outputs[output_index]));
        float max_abs = 0.0f;
        for (size_t index = 0; index < observed.size(); ++index) {
            const float value = input_data[index];
            float       expected;
            switch (output_index) {
                case 0:
                    expected = value - divisor_data[index % width];
                    break;
                case 1:
                    expected = value / divisor_data[index % width];
                    break;
                case 2:
                    expected = value * value;
                    break;
                case 3:
                    expected = std::sqrt(positive_data[index]);
                    break;
                case 4:
                    expected = std::log(positive_data[index]);
                    break;
                case 5:
                    expected = std::sin(value);
                    break;
                case 6:
                    expected = std::cos(value);
                    break;
                case 7:
                    expected = std::abs(value);
                    break;
                case 8:
                    expected = -value;
                    break;
                case 9:
                    expected = std::tanh(value);
                    break;
                case 10:
                    expected = std::max(0.0f, value);
                    break;
                case 11:
                    expected = 0.5f * value *
                               (1.0f + std::tanh(0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
                    break;
                case 12:
                    expected = value / (1.0f + std::exp(-1.702f * value));
                    break;
                case 13:
                    expected = std::exp(value);
                    break;
                default:
                    return fail("invalid math operator test index");
            }
            const float error = std::abs(observed[index] - expected);
            max_abs           = std::max(max_abs, error);
            close             = close && error <= 0.006f + 0.006f * std::abs(expected);
        }
        std::printf("math op=%s max_abs=%g\n", names[output_index], max_abs);
    }
    return close ? 0 : fail("math operator numerical comparison failed");
}

static int run_reduction_norm_case(ggml_backend_t dayan) {
    constexpr int64_t width      = 130;
    constexpr int64_t height     = 3;
    constexpr int64_t channels   = 5;
    constexpr int64_t batches    = 2;
    constexpr int     groups     = 2;
    constexpr float   epsilon    = 1.0e-5f;
    constexpr int     graph_size = 16;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the reduction and normalization context");
    }

    ggml_tensor * input      = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, height, channels, batches);
    ggml_tensor * total      = ggml_sum(ctx.get(), input);
    ggml_tensor * row_sums   = ggml_sum_rows(ctx.get(), input);
    ggml_tensor * row_means  = ggml_mean(ctx.get(), input);
    ggml_tensor * normalized = ggml_norm(ctx.get(), input, epsilon);
    ggml_tensor * grouped    = ggml_group_norm(ctx.get(), input, groups, epsilon);
    ggml_set_input(input);

    ggml_tensor * outputs[] = { total, row_sums, row_means, normalized, grouped };
    ggml_cgraph * graph     = ggml_new_graph_custom(ctx.get(), graph_size, false);
    for (ggml_tensor * output : outputs) {
        if (!ggml_backend_supports_op(dayan, output)) {
            return fail("DAYAN rejected a reduction or normalization operator");
        }
        ggml_build_forward_expand(graph, output);
    }

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate reduction and normalization tensors");
    }

    std::vector<float> input_data((size_t) width * height * channels * batches);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = ((int) (index % 61) - 30) / 24.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN reduction and normalization graph failed");
    }

    float total_result;
    ggml_backend_tensor_get(total, &total_result, 0, sizeof(total_result));
    std::vector<float> row_sum_result((size_t) height * channels * batches);
    std::vector<float> row_mean_result(row_sum_result.size());
    std::vector<float> norm_result(input_data.size());
    std::vector<float> group_result(input_data.size());
    ggml_backend_tensor_get(row_sums, row_sum_result.data(), 0, ggml_nbytes(row_sums));
    ggml_backend_tensor_get(row_means, row_mean_result.data(), 0, ggml_nbytes(row_means));
    ggml_backend_tensor_get(normalized, norm_result.data(), 0, ggml_nbytes(normalized));
    ggml_backend_tensor_get(grouped, group_result.data(), 0, ggml_nbytes(grouped));

    float total_reference = 0.0f;
    for (float value : input_data) {
        total_reference += value;
    }
    float total_error = std::abs(total_result - total_reference);
    float row_error   = 0.0f;
    float norm_error  = 0.0f;
    float group_error = 0.0f;
    bool  close       = total_error <= 0.01f + 0.001f * std::abs(total_reference);
    for (size_t row = 0; row < row_sum_result.size(); ++row) {
        float sum = 0.0f;
        for (int64_t column = 0; column < width; ++column) {
            sum += input_data[row * width + (size_t) column];
        }
        row_error = std::max(row_error, std::abs(row_sum_result[row] - sum));
        row_error = std::max(row_error, std::abs(row_mean_result[row] - sum / (float) width));

        float       variance = 0.0f;
        const float mean     = sum / (float) width;
        for (int64_t column = 0; column < width; ++column) {
            const float centered = input_data[row * width + (size_t) column] - mean;
            variance += centered * centered;
        }
        const float inverse = 1.0f / std::sqrt(variance / (float) width + epsilon);
        for (int64_t column = 0; column < width; ++column) {
            const size_t index    = row * width + (size_t) column;
            const float  expected = (input_data[index] - mean) * inverse;
            norm_error            = std::max(norm_error, std::abs(norm_result[index] - expected));
        }
    }

    const int64_t channels_per_group = (channels + groups - 1) / groups;
    const int64_t channel_elements   = width * height;
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int group = 0; group < groups; ++group) {
            const int64_t channel_start = group * channels_per_group;
            const int64_t channel_end   = std::min<int64_t>(channels, channel_start + channels_per_group);
            const size_t  offset        = (size_t) (batch * channels + channel_start) * channel_elements;
            const size_t  elements      = (size_t) (channel_end - channel_start) * channel_elements;
            float         mean          = 0.0f;
            for (size_t index = 0; index < elements; ++index) {
                mean += input_data[offset + index];
            }
            mean /= (float) elements;
            float variance = 0.0f;
            for (size_t index = 0; index < elements; ++index) {
                const float centered = input_data[offset + index] - mean;
                variance += centered * centered;
            }
            const float inverse = 1.0f / std::sqrt(variance / (float) elements + epsilon);
            for (size_t index = 0; index < elements; ++index) {
                const float expected = (input_data[offset + index] - mean) * inverse;
                group_error          = std::max(group_error, std::abs(group_result[offset + index] - expected));
            }
        }
    }
    close = close && row_error <= 0.005f && norm_error <= 0.005f && group_error <= 0.005f;
    std::printf("reduction total=%g rows=%g norm=%g group=%g\n", total_error, row_error, norm_error, group_error);
    return close ? 0 : fail("reduction or normalization numerical comparison failed");
}

static int run_strided_rms_case(ggml_backend_t dayan) {
    constexpr int64_t width      = 130;
    constexpr int64_t heads      = 2;
    constexpr int64_t tokens     = 3;
    constexpr int64_t full_width = 2 * width * heads;
    constexpr float   epsilon    = 1.0e-6f;
    constexpr int     graph_size = 8;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the strided RMSNorm context");
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, full_width, tokens);
    ggml_tensor * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
    ggml_tensor * view =
        ggml_view_3d(ctx.get(), input, width, heads, tokens, 2 * width * sizeof(float), full_width * sizeof(float), 0);
    ggml_tensor * norm   = ggml_rms_norm(ctx.get(), view, epsilon);
    ggml_tensor * output = ggml_mul(ctx.get(), norm, scale);
    ggml_set_input(input);
    ggml_set_input(scale);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate strided RMSNorm tensors");
    }

    std::vector<float> input_data((size_t) full_width * tokens);
    std::vector<float> scale_data((size_t) width);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = ((int) (index % 43) - 21) / 32.0f;
    }
    for (size_t index = 0; index < scale_data.size(); ++index) {
        scale_data[index] = 0.75f + (float) (index % 9) / 16.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(scale, scale_data.data(), 0, ggml_nbytes(scale));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN strided RMSNorm graph failed");
    }

    std::vector<float> observed((size_t) width * heads * tokens);
    ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));
    float max_abs = 0.0f;
    bool  close   = true;
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t head = 0; head < heads; ++head) {
            float square_sum = 0.0f;
            for (int64_t column = 0; column < width; ++column) {
                const size_t source = (size_t) token * full_width + head * 2 * width + column;
                square_sum += input_data[source] * input_data[source];
            }
            const float inverse = 1.0f / std::sqrt(square_sum / (float) width + epsilon);
            for (int64_t column = 0; column < width; ++column) {
                const size_t source   = (size_t) token * full_width + head * 2 * width + column;
                const size_t target   = ((size_t) token * heads + head) * width + column;
                const float  expected = input_data[source] * inverse * scale_data[(size_t) column];
                const float  error    = std::abs(observed[target] - expected);
                max_abs               = std::max(max_abs, error);
                close                 = close && error <= 0.002f + 0.002f * std::abs(expected);
            }
        }
    }
    std::printf("strided RMSNorm width=%lld heads=%lld tokens=%lld max_abs=%g\n", (long long) width, (long long) heads,
                (long long) tokens, max_abs);
    return close ? 0 : fail("strided RMSNorm numerical comparison failed");
}

static int run_rope_case(ggml_backend_t dayan, ggml_backend_t cpu, int mode) {
    constexpr int64_t width       = 130;
    constexpr int64_t heads       = 2;
    constexpr int64_t tokens      = 3;
    constexpr int     n_dims      = 64;
    constexpr int     graph_size  = 8;
    int               sections[4] = { 11, 11, 10, 0 };
    const int         position_axes = (mode & GGML_ROPE_TYPE_MROPE) != 0 ? 4 : 1;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the RoPE context");
    }

    ggml_tensor * input     = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, heads, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens * position_axes);
    ggml_tensor * output = ggml_rope_multi(ctx.get(), input, positions, nullptr, n_dims, sections, mode, 262144,
                                           10000000.0f, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    ggml_set_input(input);
    ggml_set_input(positions);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate RoPE tensors");
    }

    std::vector<float>   input_data((size_t) width * heads * tokens);
    std::vector<int32_t> position_data((size_t) tokens * position_axes);
    for (size_t index = 0; index < input_data.size(); ++index) {
        input_data[index] = ((int) (index % 47) - 23) / 32.0f;
    }
    for (int axis = 0; axis < position_axes; ++axis) {
        for (int token = 0; token < tokens; ++token) {
            position_data[(size_t) axis * tokens + token] = token + 3 * axis;
        }
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(positions, position_data.data(), 0, ggml_nbytes(positions));

    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU RoPE reference failed");
    }
    std::vector<float> reference((size_t) ggml_nelements(output));
    std::memcpy(reference.data(), output->data, ggml_nbytes(output));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN RoPE graph failed");
    }
    std::vector<float> observed((size_t) ggml_nelements(output));
    ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));

    float max_abs = 0.0f;
    bool  close   = true;
    for (size_t index = 0; index < observed.size(); ++index) {
        const float error = std::abs(observed[index] - reference[index]);
        max_abs           = std::max(max_abs, error);
        close             = close && error <= 0.002f + 0.002f * std::abs(reference[index]);
    }
    std::printf("RoPE mode=%d width=%lld heads=%lld tokens=%lld n_dims=%d max_abs=%g\n", mode, (long long) width,
                (long long) heads, (long long) tokens, n_dims, max_abs);
    return close ? 0 : fail("RoPE numerical comparison failed");
}

static int run_rope_cases(ggml_backend_t dayan, ggml_backend_t cpu) {
    return run_rope_case(dayan, cpu, GGML_ROPE_TYPE_NORMAL) != 0 ||
                   run_rope_case(dayan, cpu, GGML_ROPE_TYPE_NEOX) != 0 ||
                   run_rope_case(dayan, cpu, GGML_ROPE_TYPE_MROPE) != 0 ||
                   run_rope_case(dayan, cpu, GGML_ROPE_TYPE_IMROPE) != 0 ?
               1 :
               0;
}

static int run_memory_case(ggml_backend_t dayan) {
    constexpr int64_t width         = 130;
    constexpr int64_t source_rows   = 7;
    constexpr int64_t selected_rows = 3;
    constexpr int64_t slices        = 2;
    constexpr int64_t set_rows      = 2;
    constexpr int64_t target_rows   = 6;
    constexpr int64_t concat_rows   = 4;
    constexpr int64_t left_width    = 5;
    constexpr int64_t right_width   = 7;
    constexpr int     graph_size    = 24;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the memory operator context");
    }

    ggml_tensor * embedding = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, width, source_rows, slices);
    ggml_tensor * row_ids   = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, selected_rows, slices);
    ggml_tensor * selected  = ggml_get_rows(ctx.get(), embedding, row_ids);
    ggml_tensor * copy_dst  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 65, 12);
    ggml_tensor * copied    = ggml_cpy(ctx.get(), selected, copy_dst);

    ggml_tensor * set_dst = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, width, target_rows, slices);
    ggml_tensor * set_src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, set_rows, slices);
    ggml_tensor * set_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, set_rows, slices);
    ggml_tensor * set_out = ggml_set_rows(ctx.get(), set_dst, set_src, set_ids);

    ggml_tensor * left         = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, left_width, concat_rows);
    ggml_tensor * right        = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, right_width, concat_rows);
    ggml_tensor * concatenated = ggml_concat(ctx.get(), left, right, 0);

    ggml_set_input(embedding);
    ggml_set_input(row_ids);
    ggml_set_input(copy_dst);
    ggml_set_input(set_dst);
    ggml_set_input(set_src);
    ggml_set_input(set_ids);
    ggml_set_input(left);
    ggml_set_input(right);
    ggml_set_name(selected, "get_rows");
    ggml_set_name(copied, "copy");
    ggml_set_name(set_out, "set_rows");
    ggml_set_name(concatenated, "concat");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, copied);
    ggml_build_forward_expand(graph, set_out);
    ggml_build_forward_expand(graph, concatenated);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate memory operator tensors");
    }

    std::vector<uint16_t> embedding_data((size_t) width * source_rows * slices);
    std::vector<int32_t>  row_id_data = { 4, 1, 6, 0, 5, 2 };
    std::vector<uint16_t> set_dst_data((size_t) width * target_rows * slices);
    std::vector<float>    set_src_data((size_t) width * set_rows * slices);
    std::vector<int32_t>  set_id_data = { 4, 1, 0, 5 };
    std::vector<float>    left_data((size_t) left_width * concat_rows);
    std::vector<float>    right_data((size_t) right_width * concat_rows);
    for (size_t index = 0; index < embedding_data.size(); ++index) {
        embedding_data[index] = encode_lowp(GGML_TYPE_BF16, ((int) (index % 41) - 20) / 32.0f);
    }
    for (size_t index = 0; index < set_dst_data.size(); ++index) {
        set_dst_data[index] = encode_lowp(GGML_TYPE_BF16, ((int) (index % 17) - 8) / 16.0f);
    }
    for (size_t index = 0; index < set_src_data.size(); ++index) {
        set_src_data[index] = ((int) (index % 31) - 15) / 24.0f;
    }
    for (size_t index = 0; index < left_data.size(); ++index) {
        left_data[index] = ((int) (index % 13) - 6) / 8.0f;
    }
    for (size_t index = 0; index < right_data.size(); ++index) {
        right_data[index] = ((int) (index % 19) - 9) / 12.0f;
    }

    ggml_backend_tensor_set(embedding, embedding_data.data(), 0, ggml_nbytes(embedding));
    ggml_backend_tensor_set(row_ids, row_id_data.data(), 0, ggml_nbytes(row_ids));
    ggml_backend_tensor_set(set_dst, set_dst_data.data(), 0, ggml_nbytes(set_dst));
    ggml_backend_tensor_set(set_src, set_src_data.data(), 0, ggml_nbytes(set_src));
    ggml_backend_tensor_set(set_ids, set_id_data.data(), 0, ggml_nbytes(set_ids));
    ggml_backend_tensor_set(left, left_data.data(), 0, ggml_nbytes(left));
    ggml_backend_tensor_set(right, right_data.data(), 0, ggml_nbytes(right));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN memory operator graph failed");
    }

    std::vector<float>    selected_result((size_t) width * selected_rows * slices);
    std::vector<uint16_t> copied_result((size_t) width * selected_rows * slices);
    std::vector<uint16_t> set_result((size_t) width * target_rows * slices);
    std::vector<float>    concat_result((size_t) (left_width + right_width) * concat_rows);
    ggml_backend_tensor_get(selected, selected_result.data(), 0, ggml_nbytes(selected));
    ggml_backend_tensor_get(copied, copied_result.data(), 0, ggml_nbytes(copied));
    ggml_backend_tensor_get(set_out, set_result.data(), 0, ggml_nbytes(set_out));
    ggml_backend_tensor_get(concatenated, concat_result.data(), 0, ggml_nbytes(concatenated));

    float get_max_abs    = 0.0f;
    float copy_max_abs   = 0.0f;
    float set_max_abs    = 0.0f;
    float concat_max_abs = 0.0f;
    bool  close          = true;
    for (int64_t slice = 0; slice < slices; ++slice) {
        for (int64_t row = 0; row < selected_rows; ++row) {
            const int64_t source_row = row_id_data[(size_t) slice * selected_rows + row];
            for (int64_t column = 0; column < width; ++column) {
                const size_t output_index  = ((size_t) slice * selected_rows + row) * width + column;
                const size_t source_index  = ((size_t) slice * source_rows + source_row) * width + column;
                const float  expected      = decode_lowp(GGML_TYPE_BF16, embedding_data[source_index]);
                const float  copy_expected = decode_lowp(GGML_TYPE_F16, encode_lowp(GGML_TYPE_F16, expected));
                get_max_abs                = std::max(get_max_abs, std::abs(selected_result[output_index] - expected));
                copy_max_abs               = std::max(
                    copy_max_abs, std::abs(decode_lowp(GGML_TYPE_F16, copied_result[output_index]) - copy_expected));
            }
        }
    }

    std::vector<uint16_t> set_reference = set_dst_data;
    for (int64_t slice = 0; slice < slices; ++slice) {
        for (int64_t row = 0; row < set_rows; ++row) {
            const int64_t target_row = set_id_data[(size_t) slice * set_rows + row];
            for (int64_t column = 0; column < width; ++column) {
                const size_t source_index   = ((size_t) slice * set_rows + row) * width + column;
                const size_t target_index   = ((size_t) slice * target_rows + target_row) * width + column;
                set_reference[target_index] = encode_lowp(GGML_TYPE_BF16, set_src_data[source_index]);
            }
        }
    }
    for (size_t index = 0; index < set_result.size(); ++index) {
        set_max_abs = std::max(set_max_abs, std::abs(decode_lowp(GGML_TYPE_BF16, set_result[index]) -
                                                     decode_lowp(GGML_TYPE_BF16, set_reference[index])));
    }

    for (int64_t row = 0; row < concat_rows; ++row) {
        for (int64_t column = 0; column < left_width + right_width; ++column) {
            const float  expected = column < left_width ? left_data[(size_t) row * left_width + column] :
                                                          right_data[(size_t) row * right_width + column - left_width];
            const size_t index    = (size_t) row * (left_width + right_width) + column;
            concat_max_abs        = std::max(concat_max_abs, std::abs(concat_result[index] - expected));
        }
    }
    close = get_max_abs == 0.0f && copy_max_abs == 0.0f && set_max_abs == 0.0f && concat_max_abs == 0.0f;
    std::printf("memory get=%g copy=%g set=%g concat=%g\n", get_max_abs, copy_max_abs, set_max_abs, concat_max_abs);
    return close ? 0 : fail("memory operator numerical comparison failed");
}

static int run_empty_memory_case(ggml_backend_t dayan) {
    constexpr int64_t width      = 130;
    constexpr int64_t state_rows = 4;
    constexpr int     graph_size = 8;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the empty memory operator context");
    }

    ggml_tensor * states        = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, state_rows);
    ggml_tensor * index_storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * empty_indices = ggml_view_1d(ctx.get(), index_storage, 0, 0);
    ggml_tensor * selected      = ggml_get_rows(ctx.get(), states, empty_indices);
    ggml_tensor * destination   = ggml_view_2d(ctx.get(), states, width, 0, states->nb[1], 0);
    ggml_tensor * copied        = ggml_cpy(ctx.get(), selected, destination);
    ggml_set_input(states);
    ggml_set_input(index_storage);

    if (!ggml_backend_supports_op(dayan, selected) || !ggml_backend_supports_op(dayan, copied)) {
        return fail("DAYAN rejected an empty memory operator");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, copied);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate empty memory operator tensors");
    }
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN empty memory operator graph failed");
    }

    std::puts("empty memory get_rows=PASS copy=PASS");
    return 0;
}

static int run_copy_fast_path_case(ggml_backend_t dayan) {
    constexpr int64_t contiguous_width  = 128;
    constexpr int64_t contiguous_rows   = 128;
    constexpr int64_t contiguous_slices = 16;
    constexpr int64_t narrow_width      = 3;
    constexpr int64_t narrow_rows       = 6144;
    constexpr int64_t narrow_stride     = 4;
    constexpr int     graph_size        = 16;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the copy fast path context");
    }

    ggml_tensor * contiguous_source =
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, contiguous_width, contiguous_rows, contiguous_slices);
    ggml_tensor * contiguous_destination =
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, contiguous_width, contiguous_rows, contiguous_slices);
    ggml_tensor * contiguous_copy = ggml_cpy(ctx.get(), contiguous_source, contiguous_destination);

    ggml_tensor * narrow_storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, narrow_stride * narrow_rows);
    ggml_tensor * narrow_source =
        ggml_view_2d(ctx.get(), narrow_storage, narrow_width, narrow_rows, narrow_stride * sizeof(float), 0);
    ggml_tensor * narrow_destination = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, narrow_width, narrow_rows);
    ggml_tensor * narrow_copy        = ggml_cpy(ctx.get(), narrow_source, narrow_destination);

    ggml_set_input(contiguous_source);
    ggml_set_input(contiguous_destination);
    ggml_set_input(narrow_storage);
    ggml_set_input(narrow_destination);
    ggml_set_name(contiguous_copy, "copy_contiguous");
    ggml_set_name(narrow_copy, "copy_narrow_padded");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, contiguous_copy);
    ggml_build_forward_expand(graph, narrow_copy);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate copy fast path tensors");
    }

    std::vector<float> contiguous_data((size_t) ggml_nelements(contiguous_source));
    std::vector<float> narrow_data((size_t) ggml_nelements(narrow_storage));
    for (size_t index = 0; index < contiguous_data.size(); ++index) {
        contiguous_data[index] = ((int) (index % 257) - 128) / 64.0f;
    }
    for (size_t index = 0; index < narrow_data.size(); ++index) {
        narrow_data[index] = ((int) (index % 127) - 63) / 32.0f;
    }
    ggml_backend_tensor_set(contiguous_source, contiguous_data.data(), 0, ggml_nbytes(contiguous_source));
    ggml_backend_tensor_set(narrow_storage, narrow_data.data(), 0, ggml_nbytes(narrow_storage));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN copy fast path graph failed");
    }

    std::vector<float> contiguous_result(contiguous_data.size());
    std::vector<float> narrow_result((size_t) ggml_nelements(narrow_copy));
    ggml_backend_tensor_get(contiguous_copy, contiguous_result.data(), 0, ggml_nbytes(contiguous_copy));
    ggml_backend_tensor_get(narrow_copy, narrow_result.data(), 0, ggml_nbytes(narrow_copy));
    if (contiguous_result != contiguous_data) {
        return fail("contiguous D2D copy comparison failed");
    }
    for (int64_t row = 0; row < narrow_rows; ++row) {
        for (int64_t column = 0; column < narrow_width; ++column) {
            const float expected = narrow_data[(size_t) row * narrow_stride + column];
            const float actual   = narrow_result[(size_t) row * narrow_width + column];
            if (actual != expected) {
                return fail("narrow padded D2D copy comparison failed");
            }
        }
    }

    std::puts("copy contiguous=PASS narrow-padded=PASS");
    return 0;
}

static int run_get_rows_fast_path_case(ggml_backend_t dayan) {
    constexpr int64_t width       = 262144;
    constexpr int64_t source_rows = 3;
    constexpr int     graph_size  = 8;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the GET_ROWS fast path context");
    }

    ggml_tensor * source  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, source_rows);
    ggml_tensor * row_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * output  = ggml_get_rows(ctx.get(), source, row_ids);
    ggml_set_input(source);
    ggml_set_input(row_ids);
    ggml_set_name(output, "get_rows_f32_wide");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate GET_ROWS fast path tensors");
    }

    std::vector<float> source_data((size_t) ggml_nelements(source));
    for (size_t index = 0; index < source_data.size(); ++index) {
        source_data[index] = (float) ((int) (index % 1021) - 510) / 128.0f;
    }
    const int32_t selected_row = 2;
    ggml_backend_tensor_set(source, source_data.data(), 0, ggml_nbytes(source));
    ggml_backend_tensor_set(row_ids, &selected_row, 0, sizeof(selected_row));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN GET_ROWS fast path graph failed");
    }

    std::vector<float> result((size_t) width);
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    const float * expected = source_data.data() + selected_row * width;
    if (!std::equal(result.begin(), result.end(), expected)) {
        return fail("GET_ROWS F32 D2D comparison failed");
    }

    std::puts("get_rows f32-wide=PASS");
    return 0;
}

static int run_gated_output_case(ggml_backend_t dayan, enum ggml_type type) {
    constexpr int64_t tokens     = 3;
    constexpr int64_t embd       = 64;
    constexpr int64_t head_dim   = 128;
    constexpr int64_t heads      = 4;
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
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN gated output activation cast failed");
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

static int run_layout_case(ggml_backend_t dayan) {
    constexpr int64_t base_width = 6;
    constexpr int64_t view_width = 3;
    constexpr int64_t view_rows  = 4;
    constexpr int64_t dup_width  = 5;
    constexpr int64_t dup_rows   = 3;
    constexpr int64_t repeat_ne0 = 3;
    constexpr int64_t repeat_ne1 = 2;
    constexpr int64_t repeat_ne2 = 2;
    constexpr int     graph_size = 32;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the layout context");
    }

    ggml_tensor * base         = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, base_width, view_rows);
    ggml_tensor * view         = ggml_view_2d(ctx.get(), base, view_width, view_rows, base->nb[1], sizeof(float));
    ggml_tensor * contiguous   = ggml_cont(ctx.get(), view);
    ggml_tensor * dup_input    = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, dup_width, dup_rows);
    ggml_tensor * duplicated   = ggml_dup(ctx.get(), dup_input);
    ggml_tensor * repeat_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, repeat_ne0, repeat_ne1, repeat_ne2);
    ggml_tensor * repeated = ggml_repeat_4d(ctx.get(), repeat_input, 2 * repeat_ne0, 2 * repeat_ne1, 2 * repeat_ne2, 2);
    ggml_tensor * converted = ggml_cast(ctx.get(), repeat_input, GGML_TYPE_F16);
    ggml_set_input(base);
    ggml_set_input(dup_input);
    ggml_set_input(repeat_input);

    if (!ggml_backend_supports_op(dayan, contiguous) || !ggml_backend_supports_op(dayan, duplicated) ||
        !ggml_backend_supports_op(dayan, repeated) || !ggml_backend_supports_op(dayan, converted)) {
        return fail("DAYAN rejected a layout operator");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, contiguous);
    ggml_build_forward_expand(graph, duplicated);
    ggml_build_forward_expand(graph, repeated);
    ggml_build_forward_expand(graph, converted);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate layout tensors");
    }

    std::vector<float>    base_data((size_t) base_width * view_rows);
    std::vector<uint16_t> dup_data((size_t) dup_width * dup_rows);
    std::vector<uint16_t> repeat_data((size_t) repeat_ne0 * repeat_ne1 * repeat_ne2);
    for (size_t index = 0; index < base_data.size(); ++index) {
        base_data[index] = ((int) index - 11) / 16.0f;
    }
    for (size_t index = 0; index < dup_data.size(); ++index) {
        dup_data[index] = encode_lowp(GGML_TYPE_F16, ((int) index - 7) / 13.0f);
    }
    for (size_t index = 0; index < repeat_data.size(); ++index) {
        repeat_data[index] = encode_lowp(GGML_TYPE_BF16, ((int) index - 5) / 9.0f);
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, ggml_nbytes(base));
    ggml_backend_tensor_set(dup_input, dup_data.data(), 0, ggml_nbytes(dup_input));
    ggml_backend_tensor_set(repeat_input, repeat_data.data(), 0, ggml_nbytes(repeat_input));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN layout graph failed");
    }

    std::vector<float>    contiguous_result((size_t) view_width * view_rows);
    std::vector<uint16_t> duplicated_result(dup_data.size());
    std::vector<uint16_t> repeated_result((size_t) ggml_nelements(repeated));
    std::vector<uint16_t> converted_result(repeat_data.size());
    ggml_backend_tensor_get(contiguous, contiguous_result.data(), 0, ggml_nbytes(contiguous));
    ggml_backend_tensor_get(duplicated, duplicated_result.data(), 0, ggml_nbytes(duplicated));
    ggml_backend_tensor_get(repeated, repeated_result.data(), 0, ggml_nbytes(repeated));
    ggml_backend_tensor_get(converted, converted_result.data(), 0, ggml_nbytes(converted));

    float cont_max_abs = 0.0f;
    float cast_max_abs = 0.0f;
    bool  exact        = duplicated_result == dup_data;
    for (int64_t row = 0; row < view_rows; ++row) {
        for (int64_t column = 0; column < view_width; ++column) {
            const float expected = base_data[(size_t) row * base_width + column + 1];
            cont_max_abs =
                std::max(cont_max_abs, std::abs(contiguous_result[(size_t) row * view_width + column] - expected));
        }
    }
    for (int64_t index3 = 0; index3 < repeated->ne[3]; ++index3) {
        for (int64_t index2 = 0; index2 < repeated->ne[2]; ++index2) {
            for (int64_t index1 = 0; index1 < repeated->ne[1]; ++index1) {
                for (int64_t index0 = 0; index0 < repeated->ne[0]; ++index0) {
                    const size_t output_index =
                        (((size_t) index3 * repeated->ne[2] + index2) * repeated->ne[1] + index1) * repeated->ne[0] +
                        index0;
                    const size_t source_index = (((size_t) (index3 % repeat_input->ne[3]) * repeat_input->ne[2] +
                                                  index2 % repeat_input->ne[2]) *
                                                     repeat_input->ne[1] +
                                                 index1 % repeat_input->ne[1]) *
                                                    repeat_input->ne[0] +
                                                index0 % repeat_input->ne[0];
                    exact                     = exact && repeated_result[output_index] == repeat_data[source_index];
                }
            }
        }
    }
    for (size_t index = 0; index < repeat_data.size(); ++index) {
        const float expected =
            decode_lowp(GGML_TYPE_F16, encode_lowp(GGML_TYPE_F16, decode_lowp(GGML_TYPE_BF16, repeat_data[index])));
        const float observed = decode_lowp(GGML_TYPE_F16, converted_result[index]);
        cast_max_abs         = std::max(cast_max_abs, std::abs(observed - expected));
    }

    std::printf("layout cont=%g cast=%g duplicate=%s repeat=%s\n", cont_max_abs, cast_max_abs,
                duplicated_result == dup_data ? "PASS" : "FAIL", exact ? "PASS" : "FAIL");
    return cont_max_abs == 0.0f && cast_max_abs == 0.0f && exact ? 0 : fail("layout numerical comparison failed");
}

static int run_indexed_layout_case(ggml_backend_t dayan, ggml_backend_t cpu) {
    constexpr int    graph_size = 64;
    ggml_init_params params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the indexed layout context");
    }

    ggml_tensor * set_base             = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 6, 4);
    ggml_tensor * set_source           = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_tensor * set_output           = ggml_set_2d(ctx.get(), set_base, set_source, set_base->nb[1], sizeof(float));
    ggml_tensor * diagonal_input       = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 7);
    ggml_tensor * diagonal             = ggml_diag(ctx.get(), diagonal_input);
    ggml_tensor * mask_input           = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 9, 4, 2);
    ggml_tensor * mask_inf             = ggml_diag_mask_inf(ctx.get(), mask_input, 2);
    ggml_tensor * mask_zero            = ggml_diag_mask_zero(ctx.get(), mask_input, 1);
    ggml_tensor * pad_input            = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * padded               = ggml_pad_ext(ctx.get(), pad_input, 1, 2, 1, 1, 0, 0, 0, 0);
    ggml_tensor * padded_circular      = ggml_pad_ext_circular(ctx.get(), pad_input, 2, 1, 1, 2, 0, 0, 0, 0);
    ggml_tensor * reflected            = ggml_pad_reflect_1d(ctx.get(), pad_input, 2, 1);
    ggml_tensor * roll_input           = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 5, 4, 3);
    ggml_tensor * rolled               = ggml_roll(ctx.get(), roll_input, 2, -1, 1, 0);
    ggml_tensor * concat_left          = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 3, 2, 2);
    ggml_tensor * concat_right         = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 3, 1, 2);
    ggml_tensor * concatenated         = ggml_concat(ctx.get(), concat_left, concat_right, 1);
    std::vector<ggml_tensor *> inputs  = { set_base,  set_source, diagonal_input, mask_input,
                                           pad_input, roll_input, concat_left,    concat_right };
    std::vector<ggml_tensor *> outputs = { set_output,      diagonal,  mask_inf, mask_zero,   padded,
                                           padded_circular, reflected, rolled,   concatenated };
    for (ggml_tensor * input : inputs) {
        ggml_set_input(input);
    }
    for (ggml_tensor * output : outputs) {
        if (!ggml_backend_supports_op(dayan, output)) {
            return fail("DAYAN rejected an indexed layout operator");
        }
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    for (ggml_tensor * output : outputs) {
        ggml_build_forward_expand(graph, output);
    }
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate indexed layout tensors");
    }

    std::vector<std::vector<float>> input_values;
    input_values.reserve(inputs.size());
    for (size_t tensor_index = 0; tensor_index < inputs.size(); ++tensor_index) {
        input_values.emplace_back((size_t) ggml_nelements(inputs[tensor_index]));
        for (size_t element = 0; element < input_values.back().size(); ++element) {
            input_values.back()[element] = ((int) ((element + 7 * tensor_index) % 31) - 15) / 13.0f;
        }
        ggml_backend_tensor_set(inputs[tensor_index], input_values.back().data(), 0, ggml_nbytes(inputs[tensor_index]));
    }

    if (ggml_backend_graph_compute(cpu, graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU indexed layout reference failed");
    }
    std::vector<std::vector<float>> references;
    references.reserve(outputs.size());
    for (ggml_tensor * output : outputs) {
        references.emplace_back((size_t) ggml_nelements(output));
        std::memcpy(references.back().data(), output->data, ggml_nbytes(output));
    }

    for (size_t tensor_index = 0; tensor_index < inputs.size(); ++tensor_index) {
        ggml_backend_tensor_set(inputs[tensor_index], input_values[tensor_index].data(), 0,
                                ggml_nbytes(inputs[tensor_index]));
    }
    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN indexed layout graph failed");
    }

    float max_abs = 0.0f;
    bool  close   = true;
    for (size_t tensor_index = 0; tensor_index < outputs.size(); ++tensor_index) {
        std::vector<float> observed((size_t) ggml_nelements(outputs[tensor_index]));
        ggml_backend_tensor_get(outputs[tensor_index], observed.data(), 0, ggml_nbytes(outputs[tensor_index]));
        for (size_t element = 0; element < observed.size(); ++element) {
            if (std::isinf(observed[element]) || std::isinf(references[tensor_index][element])) {
                close = close && observed[element] == references[tensor_index][element];
            } else {
                const float error = std::abs(observed[element] - references[tensor_index][element]);
                max_abs           = std::max(max_abs, error);
                close             = close && error == 0.0f;
            }
        }
    }
    std::printf("indexed layout ops=%zu max_abs=%g\n", outputs.size(), max_abs);
    return close ? 0 : fail("indexed layout numerical comparison failed");
}

static float softmax_slope_reference(int64_t head, int64_t heads, float max_bias) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    int64_t head_log2 = 1;
    while (head_log2 <= heads / 2) {
        head_log2 *= 2;
    }
    return head < head_log2 ? std::exp2(-max_bias * (float) (head + 1) / (float) head_log2) :
                              std::exp2(-0.5f * max_bias * (float) (2 * (head - head_log2) + 1) / (float) head_log2);
}

static int run_softmax_clamp_case(ggml_backend_t dayan) {
    constexpr int64_t width      = 130;
    constexpr int64_t rows       = 3;
    constexpr int64_t heads      = 3;
    constexpr int64_t batches    = 2;
    constexpr int64_t mask_rows  = 4;
    constexpr float   scale      = 0.75f;
    constexpr float   max_bias   = 2.0f;
    constexpr float   clamp_min  = -0.75f;
    constexpr float   clamp_max  = 0.625f;
    constexpr int     graph_size = 24;

    ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context_ptr ctx(ggml_init(params));
    if (!ctx) {
        return fail("cannot create the softmax context");
    }

    ggml_tensor * plain_input  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, rows, heads, batches);
    ggml_tensor * masked_input = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, width, rows, heads, batches);
    ggml_tensor * mask         = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, width, mask_rows, 1, 1);
    ggml_tensor * sinks        = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, heads);
    ggml_tensor * clamp_input  = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 2);
    ggml_set_input(plain_input);
    ggml_set_input(masked_input);
    ggml_set_input(mask);
    ggml_set_input(sinks);
    ggml_set_input(clamp_input);
    ggml_tensor * plain  = ggml_soft_max(ctx.get(), plain_input);
    ggml_tensor * masked = ggml_soft_max_ext(ctx.get(), masked_input, mask, scale, max_bias);
    ggml_soft_max_add_sinks(masked, sinks);
    ggml_tensor * clamped = ggml_clamp(ctx.get(), clamp_input, clamp_min, clamp_max);

    if (!ggml_backend_supports_op(dayan, plain) || !ggml_backend_supports_op(dayan, masked) ||
        !ggml_backend_supports_op(dayan, clamped)) {
        return fail("DAYAN rejected Softmax or Clamp");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), graph_size, false);
    ggml_build_forward_expand(graph, plain);
    ggml_build_forward_expand(graph, masked);
    ggml_build_forward_expand(graph, clamped);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), dayan));
    if (!buffer) {
        return fail("cannot allocate Softmax tensors");
    }

    std::vector<float>    plain_data((size_t) ggml_nelements(plain_input));
    std::vector<float>    masked_data((size_t) ggml_nelements(masked_input));
    std::vector<uint16_t> mask_data((size_t) ggml_nelements(mask));
    std::vector<float>    sink_data = { -0.2f, 0.1f, -0.4f };
    std::vector<float>    clamp_data((size_t) ggml_nelements(clamp_input));
    for (size_t index = 0; index < plain_data.size(); ++index) {
        plain_data[index]  = ((int) (index % 37) - 18) / 17.0f;
        masked_data[index] = ((int) (index % 43) - 21) / 19.0f;
    }
    for (int64_t row = 0; row < mask_rows; ++row) {
        for (int64_t column = 0; column < width; ++column) {
            const float value                        = column < 91 + row * 7 ? ((int) (column % 5) - 2) / 16.0f : -8.0f;
            mask_data[(size_t) row * width + column] = encode_lowp(GGML_TYPE_F16, value);
        }
    }
    for (size_t index = 0; index < clamp_data.size(); ++index) {
        clamp_data[index] = ((int) (index % 29) - 14) / 10.0f;
    }
    ggml_backend_tensor_set(plain_input, plain_data.data(), 0, ggml_nbytes(plain_input));
    ggml_backend_tensor_set(masked_input, masked_data.data(), 0, ggml_nbytes(masked_input));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, ggml_nbytes(mask));
    ggml_backend_tensor_set(sinks, sink_data.data(), 0, ggml_nbytes(sinks));
    ggml_backend_tensor_set(clamp_input, clamp_data.data(), 0, ggml_nbytes(clamp_input));

    if (ggml_backend_graph_compute(dayan, graph) != GGML_STATUS_SUCCESS) {
        return fail("DAYAN Softmax graph failed");
    }

    std::vector<float> plain_result(plain_data.size());
    std::vector<float> masked_result(masked_data.size());
    std::vector<float> clamp_result(clamp_data.size());
    ggml_backend_tensor_get(plain, plain_result.data(), 0, ggml_nbytes(plain));
    ggml_backend_tensor_get(masked, masked_result.data(), 0, ggml_nbytes(masked));
    ggml_backend_tensor_get(clamped, clamp_result.data(), 0, ggml_nbytes(clamped));

    float plain_max_abs  = 0.0f;
    float masked_max_abs = 0.0f;
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t head = 0; head < heads; ++head) {
            const float slope = softmax_slope_reference(head, heads, max_bias);
            for (int64_t row = 0; row < rows; ++row) {
                const size_t base       = (size_t) (((batch * heads + head) * rows + row) * width);
                float        plain_max  = -INFINITY;
                float        masked_max = sink_data[(size_t) head];
                for (int64_t column = 0; column < width; ++column) {
                    plain_max              = std::max(plain_max, plain_data[base + column]);
                    const float mask_value = decode_lowp(GGML_TYPE_F16, mask_data[(size_t) row * width + column]);
                    masked_max = std::max(masked_max, scale * masked_data[base + column] + slope * mask_value);
                }
                float plain_sum  = 0.0f;
                float masked_sum = std::exp(sink_data[(size_t) head] - masked_max);
                for (int64_t column = 0; column < width; ++column) {
                    plain_sum += std::exp(plain_data[base + column] - plain_max);
                    const float mask_value = decode_lowp(GGML_TYPE_F16, mask_data[(size_t) row * width + column]);
                    masked_sum += std::exp(scale * masked_data[base + column] + slope * mask_value - masked_max);
                }
                for (int64_t column = 0; column < width; ++column) {
                    const float plain_expected = std::exp(plain_data[base + column] - plain_max) / plain_sum;
                    const float mask_value     = decode_lowp(GGML_TYPE_F16, mask_data[(size_t) row * width + column]);
                    const float masked_expected =
                        std::exp(scale * masked_data[base + column] + slope * mask_value - masked_max) / masked_sum;
                    plain_max_abs  = std::max(plain_max_abs, std::abs(plain_result[base + column] - plain_expected));
                    masked_max_abs = std::max(masked_max_abs, std::abs(masked_result[base + column] - masked_expected));
                }
            }
        }
    }

    float clamp_max_abs = 0.0f;
    for (size_t index = 0; index < clamp_data.size(); ++index) {
        const float expected = std::max(clamp_min, std::min(clamp_max, clamp_data[index]));
        clamp_max_abs        = std::max(clamp_max_abs, std::abs(clamp_result[index] - expected));
    }
    std::printf("softmax plain=%g masked=%g clamp=%g\n", plain_max_abs, masked_max_abs, clamp_max_abs);
    if (plain_max_abs > 2.0e-5f || masked_max_abs > 2.0e-5f || clamp_max_abs != 0.0f) {
        float plain_sum  = 0.0f;
        float masked_sum = 0.0f;
        for (int64_t column = 0; column < width; ++column) {
            plain_sum += plain_result[(size_t) column];
            masked_sum += masked_result[(size_t) column];
        }
        std::printf("softmax observed sums plain=%g masked=%g\n", plain_sum, masked_sum);
        for (int index = 0; index < 6; ++index) {
            std::printf("softmax[%d] plain=%g masked=%g clamp=%g input=%g\n", index, plain_result[(size_t) index],
                        masked_result[(size_t) index], clamp_result[(size_t) index], clamp_data[(size_t) index]);
        }
    }
    return plain_max_abs <= 2.0e-5f && masked_max_abs <= 2.0e-5f && clamp_max_abs == 0.0f ?
               0 :
               fail("Softmax comparison failed");
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    if (!dayan) {
        return fail("cannot initialize the DAYAN backend");
    }

    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!cpu) {
        return fail("cannot initialize the CPU backend");
    }

    if (argc == 2 && std::strcmp(argv[1], "rope") == 0) {
        return run_rope_cases(dayan.get(), cpu.get());
    }
    if (argc == 2 && std::strcmp(argv[1], "ssm-conv") == 0) {
        return run_ssm_conv_case(dayan.get(), 1, 5, 128, 1, false) != 0 ||
                       run_ssm_conv_case(dayan.get(), 4, 5, 130, 2, true) != 0 ||
                       run_ssm_conv_case(dayan.get(), 4, 69, 384, 1, true) != 0 ?
                   1 :
                   0;
    }
    if (argc == 2 && std::strcmp(argv[1], "gated-output") == 0) {
        return run_gated_output_case(dayan.get(), GGML_TYPE_BF16) != 0 ||
                       run_gated_output_case(dayan.get(), GGML_TYPE_F16) != 0 ?
                   1 :
                   0;
    }
    if (argc == 2 && std::strcmp(argv[1], "copy") == 0) {
        return run_copy_fast_path_case(dayan.get());
    }
    if (argc == 2 && std::strcmp(argv[1], "get-rows") == 0) {
        return run_get_rows_fast_path_case(dayan.get());
    }
    if (argc == 2 && std::strcmp(argv[1], "memory") == 0) {
        return run_memory_case(dayan.get());
    }
    if (argc != 1) {
        return fail("usage: test-dayan-graph [rope|ssm-conv|gated-output|copy|get-rows|memory]");
    }

    if (run_norm_matmul_case(dayan.get(), GGML_TYPE_BF16, 256, false) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_BF16, 256, true) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_F16, 128, false) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_F16, 128, true) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_BF16, 130, false) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_BF16, 130, true) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_F16, 130, false) != 0 ||
        run_norm_matmul_case(dayan.get(), GGML_TYPE_F16, 130, true) != 0 || run_fusion_fanout_case(dayan.get()) != 0 ||
        run_layout_cast_hoist_case(dayan.get(), GGML_TYPE_BF16) != 0 ||
        run_layout_cast_hoist_case(dayan.get(), GGML_TYPE_F16) != 0 || run_elementwise_case(dayan.get()) != 0 ||
        run_empty_scale_case(dayan.get()) != 0 || run_offset_scale_case(dayan.get()) != 0 ||
        run_math_case(dayan.get()) != 0 || run_reduction_norm_case(dayan.get()) != 0 ||
        run_strided_rms_case(dayan.get()) != 0 || run_rope_cases(dayan.get(), cpu.get()) != 0 ||
        run_memory_case(dayan.get()) != 0 || run_empty_memory_case(dayan.get()) != 0 ||
        run_copy_fast_path_case(dayan.get()) != 0 || run_get_rows_fast_path_case(dayan.get()) != 0 ||
        run_layout_case(dayan.get()) != 0 || run_indexed_layout_case(dayan.get(), cpu.get()) != 0 ||
        run_softmax_clamp_case(dayan.get()) != 0 ||
        run_fused_case(dayan.get(), GGML_TYPE_BF16, GGML_GLU_OP_SWIGLU, 1, 16, 256) != 0 ||
        run_fused_case(dayan.get(), GGML_TYPE_F16, GGML_GLU_OP_SWIGLU, 16, 16, 256) != 0 ||
        run_fused_case(dayan.get(), GGML_TYPE_BF16, GGML_GLU_OP_GEGLU, 1, 16, 256) != 0 ||
        run_fused_case(dayan.get(), GGML_TYPE_F16, GGML_GLU_OP_GEGLU, 16, 16, 256) != 0 ||
        run_fused_case(dayan.get(), GGML_TYPE_BF16, GGML_GLU_OP_SWIGLU, 8, 3584, 1024) != 0 ||
        run_gated_output_case(dayan.get(), GGML_TYPE_BF16) != 0 ||
        run_gated_output_case(dayan.get(), GGML_TYPE_F16) != 0 ||
        run_ssm_conv_case(dayan.get(), 1, 5, 128, 1, false) != 0 ||
        run_ssm_conv_case(dayan.get(), 4, 5, 130, 2, true) != 0 ||
        run_ssm_conv_case(dayan.get(), 4, 69, 384, 1, true) != 0) {
        return 1;
    }

    std::puts("test-dayan-graph: PASS");
    return 0;
}

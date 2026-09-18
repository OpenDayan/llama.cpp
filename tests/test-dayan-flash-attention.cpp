#include "ggml-backend.h"
#include "ggml-cpp.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using ggml_backend_graph_rewrite_t = bool (*)(ggml_backend_t, ggml_context *, ggml_cgraph *);

struct attention_graph {
    ggml_context_ptr ctx;
    ggml_cgraph *    graph;
    ggml_tensor *    q_input;
    ggml_tensor *    k_input;
    ggml_tensor *    v_input;
    ggml_tensor *    q;
    ggml_tensor *    k;
    ggml_tensor *    v;
    ggml_tensor *    mask;
    ggml_tensor *    output;
};

static int fail(const char * message) {
    std::fprintf(stderr, "test-dayan-flash-attention: %s\n", message);
    return 1;
}

static bool rewrite_graph(ggml_backend_t dayan, ggml_context * ctx, ggml_cgraph * graph) {
    ggml_backend_dev_t device = ggml_backend_get_device(dayan);
    ggml_backend_reg_t reg    = ggml_backend_dev_backend_reg(device);
    auto rewrite = (ggml_backend_graph_rewrite_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_graph_rewrite");
    return rewrite != nullptr && rewrite(dayan, ctx, graph);
}

static attention_graph build_graph(enum ggml_type q_type,
                                   enum ggml_type kv_type,
                                   int64_t        head_dim,
                                   int64_t        q_len,
                                   int64_t        kv_len,
                                   int64_t        q_heads,
                                   int64_t        kv_heads,
                                   int64_t        batch,
                                   bool           causal,
                                   bool           permuted) {
    constexpr int    graph_size = 24;
    ggml_init_params params     = {
        /* .mem_size   = */ ggml_tensor_overhead() * graph_size + ggml_graph_overhead_custom(graph_size, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };

    attention_graph result;
    result.ctx.reset(ggml_init(params));
    if (!result.ctx) {
        return result;
    }

    if (permuted) {
        result.q_input = ggml_new_tensor_4d(result.ctx.get(), q_type, head_dim, q_heads, q_len, batch);
        result.k_input = ggml_new_tensor_4d(result.ctx.get(), kv_type, head_dim, kv_heads, kv_len, batch);
        result.v_input = ggml_new_tensor_4d(result.ctx.get(), kv_type, head_dim, kv_heads, kv_len, batch);
        result.q       = ggml_permute(result.ctx.get(), result.q_input, 0, 2, 1, 3);
        result.k       = ggml_permute(result.ctx.get(), result.k_input, 0, 2, 1, 3);
        result.v       = ggml_permute(result.ctx.get(), result.v_input, 0, 2, 1, 3);
    } else {
        result.q_input = result.q = ggml_new_tensor_4d(result.ctx.get(), q_type, head_dim, q_len, q_heads, batch);
        result.k_input = result.k = ggml_new_tensor_4d(result.ctx.get(), kv_type, head_dim, kv_len, kv_heads, batch);
        result.v_input = result.v = ggml_new_tensor_4d(result.ctx.get(), kv_type, head_dim, kv_len, kv_heads, batch);
    }
    ggml_set_name(result.q_input, "q input");
    ggml_set_name(result.k_input, "k input");
    ggml_set_name(result.v_input, "v input");
    ggml_set_name(result.q, "q");
    ggml_set_name(result.k, "k");
    ggml_set_name(result.v, "v");
    ggml_set_input(result.q_input);
    ggml_set_input(result.k_input);
    ggml_set_input(result.v_input);

    result.mask = causal ? ggml_new_tensor_2d(result.ctx.get(), GGML_TYPE_F16, kv_len, q_len) : nullptr;
    if (result.mask != nullptr) {
        ggml_set_name(result.mask, "mask");
        ggml_set_input(result.mask);
    }

    result.output = ggml_flash_attn_ext(result.ctx.get(), result.q, result.k, result.v, result.mask,
                                        1.0f / std::sqrt((float) head_dim), 0.0f, 0.0f);
    ggml_set_name(result.output, "attention");
    ggml_set_output(result.output);
    result.graph = ggml_new_graph_custom(result.ctx.get(), graph_size, false);
    ggml_build_forward_expand(result.graph, result.output);
    return result;
}

static bool compare_outputs(ggml_tensor * expected_tensor,
                            ggml_tensor * observed_tensor,
                            float *       max_abs,
                            float *       max_rel) {
    std::vector<float> expected((size_t) ggml_nelements(expected_tensor));
    std::vector<float> observed(expected.size());
    ggml_backend_tensor_get(expected_tensor, expected.data(), 0, ggml_nbytes(expected_tensor));
    ggml_backend_tensor_get(observed_tensor, observed.data(), 0, ggml_nbytes(observed_tensor));

    bool close = true;
    *max_abs   = 0.0f;
    *max_rel   = 0.0f;
    for (size_t index = 0; index < expected.size(); ++index) {
        const float abs_error = std::abs(observed[index] - expected[index]);
        const float rel_error = abs_error / std::max(std::abs(expected[index]), 1.0e-6f);
        close                 = close && abs_error <= 0.025f + 0.025f * std::abs(expected[index]);
        *max_abs              = std::max(*max_abs, abs_error);
        *max_rel              = std::max(*max_rel, rel_error);
    }
    return close;
}

static int test_permuted_cache_layout(ggml_backend_t dayan,
                                      ggml_backend_t cpu,
                                      int64_t        q_len,
                                      int64_t        physical_kv_len,
                                      int64_t        effective_kv_len,
                                      uint32_t       seed,
                                      ggml_type      type = GGML_TYPE_BF16) {
    constexpr int64_t head_dim = 256;
    constexpr int64_t q_heads  = 8;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t batch    = 1;

    attention_graph reference = build_graph(GGML_TYPE_F32, type, head_dim, q_len, physical_kv_len, q_heads,
                                            kv_heads, batch, true, true);
    attention_graph actual    = build_graph(type, type, head_dim, q_len, physical_kv_len, q_heads,
                                            kv_heads, batch, true, true);
    if (!reference.ctx || !actual.ctx) {
        return fail("cannot create permuted graph contexts");
    }

    ggml_backend_buffer_ptr reference_buffer(ggml_backend_alloc_ctx_tensors(reference.ctx.get(), cpu));
    ggml_backend_buffer_ptr actual_buffer(ggml_backend_alloc_ctx_tensors(actual.ctx.get(), dayan));
    if (!reference_buffer || !actual_buffer) {
        return fail("cannot allocate permuted graph tensors");
    }

    std::mt19937                          generator(seed);
    std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
    std::vector<float>                    q_f32((size_t) ggml_nelements(reference.q_input));
    std::vector<uint16_t>                 q_lowp(q_f32.size());
    std::vector<uint16_t>                 k((size_t) ggml_nelements(reference.k_input));
    std::vector<uint16_t>                 v((size_t) ggml_nelements(reference.v_input));
    std::vector<ggml_fp16_t>              mask((size_t) q_len * physical_kv_len);
    for (size_t index = 0; index < q_f32.size(); ++index) {
        const float value = distribution(generator);
        q_lowp[index] = type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(value).bits : ggml_fp32_to_fp16(value);
        q_f32[index] = type == GGML_TYPE_BF16 ? ggml_bf16_to_fp32(ggml_bf16_t{q_lowp[index]}) : ggml_fp16_to_fp32(q_lowp[index]);
    }
    for (uint16_t & value : k) {
        const float input = distribution(generator);
        value = type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(input).bits : ggml_fp32_to_fp16(input);
    }
    for (uint16_t & value : v) {
        const float input = distribution(generator);
        value = type == GGML_TYPE_BF16 ? ggml_fp32_to_bf16(input).bits : ggml_fp32_to_fp16(input);
    }
    const int64_t causal_offset = effective_kv_len - q_len;
    for (int64_t query = 0; query < q_len; ++query) {
        for (int64_t key = 0; key < physical_kv_len; ++key) {
            mask[(size_t) query * physical_kv_len + key] =
                ggml_fp32_to_fp16(key <= causal_offset + query ? 0.0f : -INFINITY);
        }
    }

    ggml_backend_tensor_set(reference.q_input, q_f32.data(), 0, ggml_nbytes(reference.q_input));
    ggml_backend_tensor_set(actual.q_input, q_lowp.data(), 0, ggml_nbytes(actual.q_input));
    for (attention_graph * graph : { &reference, &actual }) {
        ggml_backend_tensor_set(graph->k_input, k.data(), 0, ggml_nbytes(graph->k_input));
        ggml_backend_tensor_set(graph->v_input, v.data(), 0, ggml_nbytes(graph->v_input));
        ggml_backend_tensor_set(graph->mask, mask.data(), 0, ggml_nbytes(graph->mask));
    }

    if (ggml_backend_graph_compute(cpu, reference.graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU permuted reference graph failed");
    }
    if (ggml_backend_graph_compute(dayan, actual.graph) != GGML_STATUS_SUCCESS) {
        return fail("Dayan permuted graph failed");
    }

    float max_abs;
    float max_rel;
    if (!compare_outputs(reference.output, actual.output, &max_abs, &max_rel)) {
        return fail("permuted numerical comparison failed");
    }

    std::printf(
        "permuted type=%s q_len=%lld physical_kv_len=%lld effective_kv_len=%lld head_dim=%lld "
        "max_abs=%g max_rel=%g\n",
        ggml_type_name(type), (long long) q_len, (long long) physical_kv_len, (long long) effective_kv_len, (long long) head_dim, max_abs,
        max_rel);
    return 0;
}

int main() {
    constexpr int64_t head_dim = 64;
    constexpr int64_t q_len    = 17;
    constexpr int64_t kv_len   = 23;
    constexpr int64_t q_heads  = 4;
    constexpr int64_t kv_heads = 2;
    constexpr int64_t batch    = 2;
    constexpr bool    causal   = true;

    ggml_backend_load_all();
    ggml_backend_ptr dayan(ggml_backend_init_by_name("DAYAN", nullptr));
    ggml_backend_ptr cpu(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
    if (!dayan || !cpu) {
        return fail("cannot initialize backends");
    }

    attention_graph reference =
        build_graph(GGML_TYPE_F32, GGML_TYPE_F16, head_dim, q_len, kv_len, q_heads, kv_heads, batch, causal, false);
    attention_graph actual =
        build_graph(GGML_TYPE_F32, GGML_TYPE_F16, head_dim, q_len, kv_len, q_heads, kv_heads, batch, causal, false);
    if (!reference.ctx || !actual.ctx) {
        return fail("cannot create graph contexts");
    }
    if (ggml_backend_supports_op(dayan.get(), actual.output)) {
        return fail("mixed-precision FlashAttention unexpectedly supported before rewrite");
    }
    if (!rewrite_graph(dayan.get(), actual.ctx.get(), actual.graph)) {
        return fail("FlashAttention graph rewrite failed");
    }
    if (actual.output->src[0]->type != GGML_TYPE_F16 || actual.output->src[1]->type != GGML_TYPE_F16 ||
        actual.output->src[2]->type != GGML_TYPE_F16 || !ggml_backend_supports_op(dayan.get(), actual.output)) {
        return fail("FlashAttention graph rewrite did not preserve FP16 inputs");
    }

    ggml_backend_buffer_ptr reference_buffer(ggml_backend_alloc_ctx_tensors(reference.ctx.get(), cpu.get()));
    ggml_backend_buffer_ptr actual_buffer(ggml_backend_alloc_ctx_tensors(actual.ctx.get(), dayan.get()));
    if (!reference_buffer || !actual_buffer) {
        return fail("cannot allocate graph tensors");
    }

    std::mt19937                          generator(0x351faU);
    std::uniform_real_distribution<float> distribution(-0.25f, 0.25f);
    std::vector<ggml_fp16_t>              q_f16((size_t) ggml_nelements(actual.q));
    std::vector<ggml_fp16_t>              k_f16((size_t) ggml_nelements(actual.k));
    std::vector<ggml_fp16_t>              v_f16((size_t) ggml_nelements(actual.v));
    std::vector<float>                    q_f32(q_f16.size());
    std::vector<ggml_fp16_t>              mask((size_t) q_len * kv_len);
    for (size_t index = 0; index < q_f16.size(); ++index) {
        q_f16[index] = ggml_fp32_to_fp16(distribution(generator));
        q_f32[index] = ggml_fp16_to_fp32(q_f16[index]);
    }
    for (ggml_fp16_t & value : k_f16) {
        value = ggml_fp32_to_fp16(distribution(generator));
    }
    for (ggml_fp16_t & value : v_f16) {
        value = ggml_fp32_to_fp16(distribution(generator));
    }
    const int64_t causal_offset = kv_len - q_len;
    for (int64_t query = 0; query < q_len; ++query) {
        for (int64_t key = 0; key < kv_len; ++key) {
            mask[(size_t) query * kv_len + key] = ggml_fp32_to_fp16(key <= causal_offset + query ? 0.0f : -INFINITY);
        }
    }

    ggml_backend_tensor_set(reference.q_input, q_f32.data(), 0, ggml_nbytes(reference.q_input));
    ggml_backend_tensor_set(reference.k_input, k_f16.data(), 0, ggml_nbytes(reference.k_input));
    ggml_backend_tensor_set(reference.v_input, v_f16.data(), 0, ggml_nbytes(reference.v_input));
    ggml_backend_tensor_set(actual.q_input, q_f32.data(), 0, ggml_nbytes(actual.q_input));
    ggml_backend_tensor_set(actual.k_input, k_f16.data(), 0, ggml_nbytes(actual.k_input));
    ggml_backend_tensor_set(actual.v_input, v_f16.data(), 0, ggml_nbytes(actual.v_input));
    ggml_backend_tensor_set(reference.mask, mask.data(), 0, ggml_nbytes(reference.mask));
    ggml_backend_tensor_set(actual.mask, mask.data(), 0, ggml_nbytes(actual.mask));

    if (ggml_backend_graph_compute(cpu.get(), reference.graph) != GGML_STATUS_SUCCESS) {
        return fail("CPU reference graph failed");
    }
    if (ggml_backend_graph_compute(dayan.get(), actual.graph) != GGML_STATUS_SUCCESS) {
        return fail("Dayan graph failed");
    }

    float      max_abs;
    float      max_rel;
    const bool close = compare_outputs(reference.output, actual.output, &max_abs, &max_rel);

    std::printf(
        "batch=%lld q_heads=%lld kv_heads=%lld q_len=%lld kv_len=%lld head_dim=%lld causal_offset=%lld max_abs=%g "
        "max_rel=%g\n",
        (long long) batch, (long long) q_heads, (long long) kv_heads, (long long) q_len, (long long) kv_len,
        (long long) head_dim, (long long) causal_offset, max_abs, max_rel);
    if (!close) {
        return fail("numerical comparison failed");
    }
    if (test_permuted_cache_layout(dayan.get(), cpu.get(), 3, 64, 9, 0x35f17U) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 4, 4, 4, 0x35f26U) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 1, 256, 1, 0x35f18U) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 1, 2048, 1025, 0x35f20U) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 1, 1024, 513, 0x35f21U, GGML_TYPE_F16) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 128, 512, 128, 0x35f24U, GGML_TYPE_F16) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 1, 512, 129, 0x35f25U, GGML_TYPE_F16) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 1, 1024, 512, 0x35f22U) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 1, 1024, 511, 0x35f23U) != 0 ||
        test_permuted_cache_layout(dayan.get(), cpu.get(), 65, 256, 151, 0x35f19U) != 0) {
        return 1;
    }

    std::puts("test-dayan-flash-attention: PASS");
    return 0;
}

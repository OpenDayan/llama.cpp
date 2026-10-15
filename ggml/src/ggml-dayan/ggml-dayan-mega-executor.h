#pragma once

#include "ggml-dayan-kernels.h"

#include <stdint.h>

#if defined(__DAYAN_DEVICE__)
#    include <dayan_sys.h>
#endif

enum : uint16_t {
    kDayanMegaInvalid = 0,
#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) Id,
#define MEGA_STEP(...)
#define MEGA_GRAPH_END()
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END
};

enum : uint32_t {
    kDayanMegaMaxParameters = 256,
    kDayanMegaRootCount     = 16,
};

struct alignas(8) DayanMegaConfig {
    uint32_t total_bytes;
    uint16_t graph;
    uint16_t parameter_count;
    uint32_t workspace_offset = 0;
};

struct DayanMegaParameter {
    uint16_t source;
    uint16_t reserved;
    uint64_t offset;
};

namespace ggml_dayan_mega_detail {

struct GraphKind {
    bool moe;
    bool linear;
};

constexpr GraphKind graph_kind(uint16_t graph) {
    switch (graph) {
#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) case Id: return { Moe, Linear };
#define MEGA_STEP(...)
#define MEGA_GRAPH_END()
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END
        default: return { false, false };
    }
}

int norm_pipeline(const DayanNormPipelineConfig * config, const void * input, const void * right, const float * scale, void * residual, float * rms, float * scaled, uint16_t * lowp);
int residual_add(const DayanElementwiseConfig * config, const void * left, const void * right, void * output);
int residual_add_bf16(const DayanElementwiseConfig * config, const void * left, const void * right, void * output);
int rms_norm(const DayanNormConfig * config, const void * input, float * output);
int scale(const DayanScaleConfig * config, const float * input, float * output);
int get_rows(const DayanRowsConfig * config, const void * input, const void * indices, void * output);
int quantize_mxfp4(const DayanMxfp4Config * config, const void * input, void * output, void * scales);
int matmul_mxfp4(const DayanMxfp4Config * config, const void * input, const void * input_scales, const void * weight, const void * weight_scales, const void * up, const void * up_scales, void * output, void * output_scales);
int concat(const DayanConcatConfig * config, const void * left, const void * right, void * output);
int copy(const DayanCopyConfig * config, const void * input, void * output);
int ssm_conv(const DayanSSMConvConfig * config, const float * input, const float * weight, float * output);
int ssm_conv_split(const DayanSSMConvConfig * config, const float * input, const float * weight, float * output, const float * current, float * state);
int l2_norm(const DayanNormConfig * config, const float * input, float * output);
int add(const DayanBinaryConfig * config, const float * left, const float * right, float * output);
int multiply(const DayanBinaryConfig * config, const float * left, const float * right, float * output);
int softplus(const DayanElementwiseConfig * config, const float * input, float * output);
int sigmoid(const DayanElementwiseConfig * config, const float * input, float * output);
int silu(const DayanElementwiseConfig * config, const float * input, float * output);
int gated_delta_net(const DayanGatedDeltaNetConfig * config, const float * q, const float * k, const float * v, const float * g, const float * beta, const float * state, float * output);
int rope(const DayanRopeConfig * config, const float * input, const int32_t * positions, const float * factors, float * output);
int set_rows(const DayanRowsConfig * config, const float * input, const void * indices, void * output);
int flash_attention_fp16(const DayanFlashAttentionConfig * config, const uint16_t * q, const uint16_t * k, const uint16_t * v, float * output);
int matmul_bf16(const DayanMatmulConfig * config, const uint16_t * input, const uint16_t * weight, float * output);
int matmul_pair_bf16(const DayanMatmulPairConfig * config, const uint16_t * input, const uint16_t * first, const uint16_t * second, float * first_output, float * second_output);
int matmul_swiglu_bf16(const DayanMatmulConfig * config, const uint16_t * input, const uint16_t * gate, const uint16_t * up, uint16_t * output);
int matmul_quantized_bf16(const DayanMatmulConfig * config, const uint16_t * input, const uint16_t * weight, uint16_t * output);
int gated_output_projection_bf16(const DayanGatedProjectionConfig * config, const float * input, const float * gate, const uint16_t * weight, float * output, uint16_t * scratch);
int matmul_tf32(const DayanMatmulConfig * config, const float * input, const float * weight, float * output);
int moe_sort(const DayanMoeSortConfig * config, const float * probabilities, int32_t * selected, void * workspace);
int matmul_id_mxfp4(const DayanMxfp4MoeConfig * config, const void * input, const void * input_scales, const void * weight, const void * weight_scales, const void * up, const void * up_scales, const int32_t * selected, void * output, void * output_scales, void * workspace);
int softmax(const DayanSoftmaxConfig * config, const float * input, float * output);
int sum_rows(const DayanReduceConfig * config, const float * input, float * output);
int clamp(const DayanClampConfig * config, const float * input, float * output);
int divide(const DayanBinaryConfig * config, const float * left, const float * right, float * output);
int noop();

template <typename Function> struct Signature;

template <typename... Args>
struct Signature<int (*)(Args...)> {
    static constexpr uint16_t count = sizeof...(Args);
};

template <typename Config, typename... Args>
struct Signature<int (*)(const Config *, Args...)> {
    using config_type = Config;
    static constexpr uint16_t count = 1 + sizeof...(Args);
};

template <unsigned Count>
constexpr uint16_t parameter_count(const uint16_t (&counts)[Count]) {
    uint16_t total = 0;
    for (uint16_t count : counts) {
        total += count;
    }
    return total;
}

#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) namespace Name { enum Step {
#define MEGA_STEP(Name, Kind, Function, Barriers, Mode, ...) Name,
#define MEGA_GRAPH_END() step_count }; }
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END

#define MEGA_GRAPH_BEGIN(Name, Id, Moe, Linear) namespace Name { static constexpr uint16_t counts[] = {
#define MEGA_STEP(Name, Kind, Function, Barriers, Mode, ...) Signature<decltype(&ggml_dayan_mega_detail::Function)>::count,
#define MEGA_GRAPH_END() \
    }; \
    static constexpr uint16_t parameter_count = ggml_dayan_mega_detail::parameter_count(counts); \
    static_assert(parameter_count <= kDayanMegaMaxParameters, "Mega parameter table is too large"); \
    }
#include "mega-graph/qwen35.inc"
#undef MEGA_GRAPH_BEGIN
#undef MEGA_STEP
#undef MEGA_GRAPH_END

#if defined(__DAYAN_DEVICE__)

struct Operation {
    int (*function)(const uintptr_t *, uint64_t);
    uint16_t argument_count;
    uint16_t barriers;
    char name[DAYAN_THREAD_NAME_SIZE];
};

template <unsigned... Indices> struct Sequence {};
template <unsigned Count, unsigned... Indices>
struct MakeSequence : MakeSequence<Count - 1, Count - 1, Indices...> {};
template <unsigned... Indices>
struct MakeSequence<0, Indices...> { using type = Sequence<Indices...>; };

template <typename Function, Function Entry> struct Binding;

template <typename... Args, int (*Entry)(Args...)>
struct Binding<int (*)(Args...), Entry> {
    template <unsigned... Indices>
    static int invoke(const uintptr_t * parameters, Sequence<Indices...>) {
        return Entry(reinterpret_cast<Args>(parameters[Indices])...);
    }

    static int call(const uintptr_t * parameters, uint64_t) {
        return invoke(parameters, typename MakeSequence<sizeof...(Args)>::type{});
    }
};

template <typename Config> struct ClusterConfig {
    static constexpr bool enabled = false;
};

template <> struct ClusterConfig<DayanMatmulConfig> {
    static constexpr bool enabled = true;
    static void bind(DayanMatmulConfig & config, uint64_t workspace) { config.cluster_workspace = workspace; }
};

template <> struct ClusterConfig<DayanMxfp4Config> {
    static constexpr bool enabled = true;
    static void bind(DayanMxfp4Config & config, uint64_t workspace) { config.cluster_workspace = workspace; }
};

template <> struct ClusterConfig<DayanFlashAttentionConfig> {
    static constexpr bool enabled = true;
    static void bind(DayanFlashAttentionConfig & config, uint64_t workspace) { config.cluster_workspace = workspace; }
};

template <> struct ClusterConfig<DayanMatmulPairConfig> {
    static constexpr bool enabled = true;
    static void bind(DayanMatmulPairConfig & config, uint64_t workspace) {
        config.first.cluster_workspace = config.second.cluster_workspace = workspace;
    }
};

template <> struct ClusterConfig<DayanGatedProjectionConfig> {
    static constexpr bool enabled = true;
    static void bind(DayanGatedProjectionConfig & config, uint64_t workspace) { config.matmul.cluster_workspace = workspace; }
};

template <> struct ClusterConfig<DayanMxfp4MoeConfig> {
    static constexpr bool enabled = true;
    static void bind(DayanMxfp4MoeConfig & config, uint64_t workspace) { config.matmul.cluster_workspace = workspace; }
};

template <typename Config, typename... Args, int (*Entry)(const Config *, Args...)>
struct Binding<int (*)(const Config *, Args...), Entry> {
    template <unsigned... Indices>
    static int invoke(const uintptr_t * parameters, uint64_t workspace, Sequence<Indices...>) {
        const auto * original = reinterpret_cast<const Config *>(parameters[0]);
        if constexpr (ClusterConfig<Config>::enabled) {
            Config config = *original;
            ClusterConfig<Config>::bind(config, workspace);
            return Entry(&config, reinterpret_cast<Args>(parameters[Indices + 1])...);
        } else {
            return Entry(original, reinterpret_cast<Args>(parameters[Indices + 1])...);
        }
    }

    static int call(const uintptr_t * parameters, uint64_t workspace) {
        return invoke(parameters, workspace, typename MakeSequence<sizeof...(Args)>::type{});
    }
};

#endif

} // namespace ggml_dayan_mega_detail

#if !defined(__DAYAN_DEVICE__)

int qwen35_dense_mega(
    npu_buffer_t * config,
    npu_buffer_t * root0,
    npu_buffer_t * root1,
    npu_buffer_t * root2,
    npu_buffer_t * root3,
    npu_buffer_t * root4,
    npu_buffer_t * root5,
    npu_buffer_t * root6,
    npu_buffer_t * root7,
    npu_buffer_t * root8,
    npu_buffer_t * root9,
    npu_buffer_t * root10,
    npu_buffer_t * root11,
    npu_buffer_t * root12,
    npu_buffer_t * root13,
    npu_buffer_t * root14,
    npu_buffer_t * root15,
    const npu_kernel_launch_attrs_t * attrs);

int qwen35_moe_mega(
    npu_buffer_t * config,
    npu_buffer_t * root0,
    npu_buffer_t * root1,
    npu_buffer_t * root2,
    npu_buffer_t * root3,
    npu_buffer_t * root4,
    npu_buffer_t * root5,
    npu_buffer_t * root6,
    npu_buffer_t * root7,
    npu_buffer_t * root8,
    npu_buffer_t * root9,
    npu_buffer_t * root10,
    npu_buffer_t * root11,
    npu_buffer_t * root12,
    npu_buffer_t * root13,
    npu_buffer_t * root14,
    npu_buffer_t * root15,
    const npu_kernel_launch_attrs_t * attrs);

#endif

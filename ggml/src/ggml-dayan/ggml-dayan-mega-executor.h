#pragma once

#include "ggml-dayan-kernels.h"

#include <stdint.h>

enum : uint16_t {
    kDayanMegaQwen35DenseMxfp4Linear = 1,
    kDayanMegaQwen35DenseMxfp4Full   = 2,
    kDayanMegaQwen35DenseBf16Linear  = 3,
    kDayanMegaQwen35DenseBf16Full    = 4,
};

enum : uint32_t {
    kDayanMegaMaxParameters                    = 256,
    kDayanMegaRootCount                        = 16,
    kDayanMegaQwen35DenseMxfp4LinearParameters = 166,
    kDayanMegaQwen35DenseMxfp4FullParameters   = 137,
    kDayanMegaQwen35DenseBf16LinearParameters  = 129,
    kDayanMegaQwen35DenseBf16FullParameters    = 89,
};

struct DayanMegaConfig {
    uint32_t total_bytes;
    uint16_t graph;
    uint16_t parameter_count;
};

struct DayanMegaParameter {
    uint16_t source;
    uint16_t reserved;
    uint32_t offset;
};

#if defined(__DAYAN_DEVICE__)

namespace ggml_dayan_mega_detail {

using Function = int (*)(const uintptr_t *, const uint16_t *);

struct Operation {
    Function         function;
    const uint16_t * arguments;
};

template <uint16_t Base, uint16_t Count>
struct NextParameter {
    static constexpr uint16_t value = Base + Count;
};

template <uint16_t... Indices>
struct Arguments {
    static const uint16_t values[sizeof...(Indices)];
};

template <uint16_t... Indices>
const uint16_t Arguments<Indices...>::values[sizeof...(Indices)] = { Indices... };

int norm_pipeline(const uintptr_t *, const uint16_t *);
int residual_add(const uintptr_t *, const uint16_t *);
int residual_add_bf16(const uintptr_t *, const uint16_t *);
int rms_norm(const uintptr_t *, const uint16_t *);
int scale(const uintptr_t *, const uint16_t *);
int get_rows(const uintptr_t *, const uint16_t *);
int quantize_mxfp4(const uintptr_t *, const uint16_t *);
int matmul_mxfp4(const uintptr_t *, const uint16_t *);
int concat(const uintptr_t *, const uint16_t *);
int copy(const uintptr_t *, const uint16_t *);
int repeat(const uintptr_t *, const uint16_t *);
int ssm_conv(const uintptr_t *, const uint16_t *);
int l2_norm(const uintptr_t *, const uint16_t *);
int add(const uintptr_t *, const uint16_t *);
int multiply(const uintptr_t *, const uint16_t *);
int softplus(const uintptr_t *, const uint16_t *);
int sigmoid(const uintptr_t *, const uint16_t *);
int silu(const uintptr_t *, const uint16_t *);
int gated_delta_net(const uintptr_t *, const uint16_t *);
int rope(const uintptr_t *, const uint16_t *);
int set_rows(const uintptr_t *, const uint16_t *);
int flash_attention_fp16(const uintptr_t *, const uint16_t *);
int matmul_bf16(const uintptr_t *, const uint16_t *);
int matmul_pair_bf16(const uintptr_t *, const uint16_t *);
int matmul_swiglu_bf16(const uintptr_t *, const uint16_t *);
int matmul_quantized_bf16(const uintptr_t *, const uint16_t *);
int gated_output_projection_bf16(const uintptr_t *, const uint16_t *);

} // namespace ggml_dayan_mega_detail

#else

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

#endif

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../ggml-impl.h"
#include "./common.hpp"
#include "./ops.hpp"
#include "./utils.hpp"
#include "ggml.h"

#include "NPUDriver.h"
#include "NPURuntime.h"
struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

// reserve 128MB for cpu buffer
static char src0_cpu_buffer[1024L * 1024L * 512];
static char src1_cpu_buffer[1024L * 1024L * 512];
static char dst_cpu_buffer[1024L * 1024L * 512];
static void copy_binary_op_data_from_dev_to_cpu(
                    ggml_tensor * dst, void ** src0_data_cpu,
                    void ** src1_data_cpu, void ** dst_data_cpu) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    size_t src0_size = ggml_nbytes(src0);
    size_t src1_size = ggml_nbytes(src1);
    size_t dst_size = ggml_nbytes(dst);
    npuMemcpy(src0_cpu_buffer, src0->data, src0_size, npuMemcpyDeviceToHost);
    npuMemcpy(src1_cpu_buffer, src1->data, src1_size, npuMemcpyDeviceToHost);
    if(dst->view_src == src0) {
        *dst_data_cpu = (void *)src0_cpu_buffer;
    } else if(dst->view_src == src1) {
        *dst_data_cpu = (void *)src1_cpu_buffer;
    } else {
        *dst_data_cpu = (void *)dst_cpu_buffer;
        npuMemcpy(dst_cpu_buffer, dst->data, dst_size, npuMemcpyDeviceToHost);
    }
    *src0_data_cpu = (void *)src0_cpu_buffer;
    *src1_data_cpu = (void *)src1_cpu_buffer;

}

static void copy_binary_op_data_from_cpu_to_dev(
                    ggml_tensor * dst, void * src0_data_cpu,
                    void * src1_data_cpu, void * dst_data_cpu) {
    GGML_UNUSED(src0_data_cpu);
    GGML_UNUSED(src1_data_cpu);
    npuMemcpy(dst->data, dst_data_cpu, ggml_nbytes(dst), npuMemcpyHostToDevice);
}

static void copy_unary_op_data_from_dev_to_cpu(
                ggml_tensor * dst, void ** src0_data_cpu, void ** dst_data_cpu) {
    const ggml_tensor * src0 = dst->src[0];
    size_t src0_size = ggml_nbytes(src0);
    size_t dst_size = ggml_nbytes(dst);
    if (dst->view_src == src0) {
        *dst_data_cpu = (void *)src0_cpu_buffer;
    } else {
        *dst_data_cpu = (void *)dst_cpu_buffer;
        npuMemcpy(dst_cpu_buffer, dst->data, dst_size, npuMemcpyDeviceToHost);
    }
    npuMemcpy(src0_cpu_buffer, src0->data, src0_size, npuMemcpyDeviceToHost);
    *src0_data_cpu = (void *)src0_cpu_buffer;
}

static void copy_unary_op_data_from_cpu_to_dev(
                ggml_tensor * dst, void * src0_data_cpu, void * dst_data_cpu) {
    GGML_UNUSED(src0_data_cpu);
    npuMemcpy(dst->data, dst_data_cpu, ggml_nbytes(dst), npuMemcpyHostToDevice);
}

static void ggml_rnpu_get_rows_bf16_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    // source tensor
    int bs0 = src0->ne[3];
    int h0 = src0->ne[2];
    int n0 = src0->ne[1];
    int m0 = src0->ne[0];

    // index tensor
    int bs1 = src1->ne[3];
    int h1 = src1->ne[2];
    int n1 = src1->ne[1];
    int m1 = src1->ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    uint16_t * a = (uint16_t *)src0_data_cpu;
    int * b = (int *)src1_data_cpu;
    float * res = (float *)dst_data_cpu;
    for (int j = 0; j < h1; ++j)
        for (int k = 0; k < n1; ++k)
            for (int p = 0; p < m1; ++p) {
                // row_idx = b[0][j][k][p]
                int row_idx = b[j*n1*m1 + k*m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    GGML_ABORT( "[ERROR] invalid row_idx value: %d\n", row_idx);
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    uint32_t val_u32 = (uint32_t)a[j*h0*n0*m0 + k*n0*m0 + row_idx*m0 + l];
                    val_u32 <<= 16;
                    float val_f32;
                    std::memcpy(&val_f32, &val_u32, sizeof(float));
                    res[j*n1*m1*m0 + k*n1*m1 + p*m0 + l] = val_f32;
                }

            }
    GGML_UNUSED(ctx);

    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_get_rows_bf16_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_get_rows_f32_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    // source tensor
    int bs0 = src0->ne[3];
    int h0 = src0->ne[2];
    int n0 = src0->ne[1];
    int m0 = src0->ne[0];

    // index tensor
    int bs1 = src1->ne[3];
    int h1 = src1->ne[2];
    int n1 = src1->ne[1];
    int m1 = src1->ne[0];

    assert((bs0 == h1) && (h0 == n1));
    assert(bs1 == 1);

    float * a = (float *)src0_data_cpu;
    int * b = (int *)src1_data_cpu;
    float * res = (float *)dst_data_cpu;
    for (int j = 0; j < h1; ++j)
        for (int k = 0; k < n1; ++k)
            for (int p = 0; p < m1; ++p) {
                // row_idx = b[0][j][k][p]
                int row_idx = b[j*n1*m1 + k*m1 + p];
                if (row_idx < 0 || row_idx > n0) {
                    GGML_ABORT("[ERROR] invalid row_idx value: %d\n", row_idx);
                    return;
                }
                // res[j][k][p][:] = a[j][k][row_idx][:]
                for (int l = 0; l < m0; l++) {
                    res[j*n1*m1*m0 + k*m1*m0 + p*m0 + l] = a[j*h0*n0*m0 + k*n0*m0 + row_idx*m0 + l];
                }
            }
    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_get_rows_f32_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_get_rows_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    // TODO: LAUNCH YOUR KERNEL HERE
    GGML_UNUSED(ctx);
    // GGML_UNUSED(dst);
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    int bs0 = src0->ne[3];
    int h0 = src0->ne[2];
    int n0 = src0->ne[1];
    int m0 = src0->ne[0];

    int bs1 = src1->ne[3];
    int h1 = src1->ne[2];
    int n1 = src1->ne[1];
    int m1 = src1->ne[0];

    GGML_ASSERT((bs0 == h1) && (h0 == n1));
    GGML_ASSERT(bs1 == 1);

    static struct NPUKernelDesc* get_rows_f32 = NULL;
    if (get_rows_f32 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "get_rows_f32_i32_f32", &get_rows_f32);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%ld\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded get_rows_f32_i32_f32 kernel desc successfully.\n");
    }
    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, get_rows_f32, &kernel_ret, 8,
                             src0->data, src1->data, dst->data,
                             (uint64_t)h1, (uint64_t)n1, (uint64_t)m1, (uint64_t)m0, (uint64_t)n0);
    RECORD_OP_END("get_rows_f32_i32_f32", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_get_rows(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_BF16:
            // ggml_rnpu_get_rows_bf16_cpu(ctx, dst);
            GGML_ABORT("get_rows_bf16_cpu should not be called in R-NPU backend (always in CPU)\n");
            break;
        case GGML_TYPE_F32:
            // ggml_rnpu_get_rows_f32_cpu(ctx, dst);
            ggml_rnpu_get_rows_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op GET_ROWS in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static void ggml_rnpu_op_add_f32_cpu(ggml_backend_rnpu_context & ctx, ggml_tensor * dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TODO: Only support f32 add
    if (src0->type != GGML_TYPE_F32) {
        GGML_ABORT("FATAL ERROR: type %s not supported for op ADD in R-NPU\n",
                    ggml_type_name(src0->type));
    }

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    // src0 and src1 must be contiguous
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    for (int i = 0; i < bs; ++i) {
        int i1 = src1->ne[3] == 1 ? 0 : i;
        for (int j = 0; j < h; ++j) {
            int j1 = src1->ne[2] == 1 ? 0 : j;
            for (int k = 0; k < n; ++k) {
                int k1 = src1->ne[1] == 1 ? 0 : k;
                for (int p = 0; p < m; ++p) {
                    int p1 = src1->ne[0] == 1 ? 0 : p;
                    ((float *)dst_data_cpu)[i * h * n * m + j * n * m + k * m + p] =
                        ((float *)src0_data_cpu)[i * h * n * m + j * n * m + k * m + p] +
                        ((float* )src1_data_cpu)[i1* h * n * m + j1 * n * m + k1 * m + p1];
                }
            }
        }
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_op_add_f32_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_op_add_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    // TODO: LAUNCH YOUR KERNEL HERE
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];


    static struct NPUKernelDesc* kernel_add_fp32_desc = NULL;
    if (kernel_add_fp32_desc == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "kernel_add_fp32", &kernel_add_fp32_desc);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%ld\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded kernel_add_fp32 kernel desc successfully.\n");
    }

    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, kernel_add_fp32_desc, &kernel_ret, 11,
                            src0->data, src1->data, dst->data, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);

    GGML_UNUSED(ctx);
    // GGML_UNUSED(dst);
    RECORD_OP_END("kernel_add_fp32", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_op_add(ggml_backend_rnpu_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            // ggml_rnpu_op_add_f32_cpu(ctx, dst);
            ggml_rnpu_op_add_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op ADD in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static void ggml_rnpu_op_mul_f32_cpu(ggml_backend_rnpu_context & ctx, ggml_tensor * dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TODO: Only support f32 mul
    if (src0->type != GGML_TYPE_F32) {
        GGML_ABORT("FATAL ERROR: type %s not supported for op MUL in R-NPU\n",
                    ggml_type_name(src0->type));
    }

    GGML_ASSERT(ggml_can_repeat(src1, src0) && ggml_are_same_shape(src0, dst));

    GGML_TENSOR_BINARY_OP_LOCALS

    // src0 and src1 must be contiguous
    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    for (int i = 0; i < bs; ++i) {
        int i1 = src1->ne[3] == 1 ? 0 : i;
        for (int j = 0; j < h; ++j) {
            int j1 = src1->ne[2] == 1 ? 0 : j;
            for (int k = 0; k < n; ++k) {
                int k1 = src1->ne[1] == 1 ? 0 : k;
                for (int p = 0; p < m; ++p) {
                    int p1 = src1->ne[0] == 1 ? 0 : p;
                    ((float *)dst_data_cpu)[i * h * n * m + j * n * m + k * m + p] =
                        ((float *)src0_data_cpu)[i * h * n * m + j * n * m + k * m + p] *
                        ((float* )src1_data_cpu)[i1* h * n * m + j1 * n * m + k1 * m + p1];
                }
            }
        }
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_op_mul_f32_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_op_mul_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    // TODO: LAUNCH YOUR KERNEL HERE
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];


    static struct NPUKernelDesc* kernel_mul_fp32_desc = NULL;
    if (kernel_mul_fp32_desc == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "kernel_mul_fp32", &kernel_mul_fp32_desc);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%ld\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded kernel_mul_fp32 kernel desc successfully.\n");
    }

    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, kernel_mul_fp32_desc, &kernel_ret, 11,
                            src0->data, src1->data, dst->data, src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);

    GGML_UNUSED(ctx);
    // GGML_UNUSED(dst);
    RECORD_OP_END("kernel_mul_fp32", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_op_mul(ggml_backend_rnpu_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            // ggml_rnpu_op_mul_f32_cpu(ctx, dst);
            ggml_rnpu_op_mul_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op MUL in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static void rope_cache_init(std::vector<float>& cache,
                     float theta_base, float theta_scale,
                     float freq_scale,
                     int m,
                     //  float * corr_dims = NULL, // useless in qwen2
                     //  float * freq_factor = NULL, // freq_factor is always NULL in qwen2
                     //  float ext_factor = 0.0f, // // ext_factor is always 0.0f in qwen2
                     float mscale = 1.0f, // mscale == attn_factor is always 1.0f in qwen2
                     float sin_sign = 1.0f // sin_sign is always 1.0f in forward computing
                     ) {
    float theta = theta_base;
    for (size_t i = 0; i < (size_t)m; i += 2) {
        // const float ff = freq_factors ? freq_factors[i/2] : 1.0f;
        const float ff = 1.0f;

        // YaRN
        // float theta_extrap = theta / ff;
        float theta_interp = freq_scale * (theta / ff);
        // if (ext_factor != 0.0f) ...
        cache[i] = std::cos(theta_interp) * mscale;
        cache[i + 1] = std::sin(theta_interp) * mscale;

        cache[i + 1] *= sin_sign;
        theta *= theta_scale;
    }
}

static void ggml_rnpu_rope_f32_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TODO: Only support f32 rope
    if (src0->type != GGML_TYPE_F32) {
        GGML_ABORT("FATAL ERROR: type %s not supported for op RoPE in R-NPU\n",
                    ggml_type_name(src0->type));
    }

    int sections[4];
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);

    // TODO: Only support specified params
    GGML_ASSERT(mode == GGML_ROPE_TYPE_NEOX);
    GGML_ASSERT(attn_factor == 1.0f);
    GGML_ASSERT(ext_factor == 0.0f);
    GGML_UNUSED(n_ctx_orig);
    GGML_UNUSED(sections);

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_UNARY_OP_LOCALS

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);

    std::vector<float> cache;
    cache.resize(src0->ne[0]);

    for (int i = 0; i < bs; ++i) { // batch
        for (int j = 0; j < h; ++j) { // seq-len
            int64_t p = ((int32_t *)src1_data_cpu)[j];
            rope_cache_init(cache, p, theta_scale,
                            static_cast<float>(freq_scale), dst->ne[0]);
            for (int k = 0; k < n; ++k) { // attn-heads
                // fill res[i][j][k][:prope.n_dims]
                for (int p = 0; p < n_dims; p += 2) {
                    const int64_t ic = p / 2;
                    const float cos_theta = cache[p];
                    const float sin_theta = cache[p + 1];

                    const float x0 = ((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + ic];
                    const float x1 = ((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + ic + n_dims / 2];

                    ((float *)dst_data_cpu)[i*h*n*m + j*n*m + k*m + ic] =
                                                    (x0*cos_theta) - (x1*sin_theta);
                    ((float *)dst_data_cpu)[i*h*n*m + j*n*m + k*m + ic + n_dims / 2] =
                                                    (x0*sin_theta) + (x1*cos_theta);
                }

                // fill remain res[i][j][k][prope.n_dims:]
                for (size_t p = n_dims; p < (size_t)m; p += 2) {
                    ((float *)dst_data_cpu)[i*h*n*m + j*n*m + k*m + p] =
                            ((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + p];
                    ((float *)dst_data_cpu)[i*h*n*m + j*n*m + k*m + p + 1] =
                            ((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + p + 1];
                }
            }
        }
    }
    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_rope_f32_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_rope_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TODO: Only support f32 rope
    if (src0->type != GGML_TYPE_F32) {
        GGML_ABORT("FATAL ERROR: type %s not supported for op RoPE in R-NPU\n",
                    ggml_type_name(src0->type));
    }

    int sections[4];
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));
    memcpy(&sections,    (int32_t *) dst->op_params + 11, sizeof(int)*4);

    // TODO: Only support specified params
    GGML_ASSERT(mode == GGML_ROPE_TYPE_NEOX);
    GGML_ASSERT(attn_factor == 1.0f);
    GGML_ASSERT(ext_factor == 0.0f);
    GGML_UNUSED(n_ctx_orig);
    GGML_UNUSED(sections);

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
    int mdst = dst->ne[0];

    static struct NPUKernelDesc* rope_f32_i32_f32 = NULL;
    if (rope_f32_i32_f32 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "rope_f32_i32_f32", &rope_f32_i32_f32);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded rope_f32_i32_f32 kernel desc successfully.\n");
    }

    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, rope_f32_i32_f32, &kernel_ret, 11,
                             src0->data, src1->data, dst->data,
                             (uint64_t)bs, (uint64_t)h, (uint64_t)n, (uint64_t)m, (uint64_t)n_dims,
                             float_as_u32(theta_scale), float_as_u32(freq_scale), mdst);
    GGML_UNUSED(ctx);
    RECORD_OP_END("rope_f32_i32_f32", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_rope(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_rnpu_rope_f32(ctx, dst);
            // ggml_rnpu_rope_f32_cpu(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op RoPE in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static uint16_t f32_to_bf16(float h) {
    uint16_t res;
    union {
        float value;
        uint32_t bits;
    } fp32;
    fp32.value = h;
    if ((fp32.bits & 0x7fffffff) > 0x7f800000) { /* nan */
        res = (fp32.bits >> 16) | 64; /* force to quiet */
        return res;
    }
    res = (fp32.bits + (0x7fff + ((fp32.bits >> 16) & 1))) >> 16;
    return res;
}

static void ggml_rnpu_dup_f32_bf16_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    GGML_TENSOR_UNARY_OP_LOCALS;

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu = src0->data;
    void * dst_data_cpu = dst->data;
    copy_unary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &dst_data_cpu);

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];

    if (ggml_is_contiguous(src0)) {
        if (ggml_is_contiguous(dst)) {
            // dup_cont_f32_cont_bf16
            for (int i = 0; i < bs; ++i)
                for (int j = 0; j < h; ++j)
                    for (int k = 0; k < n; ++k)
                        for (int p = 0; p < m; ++p)
                            ((uint16_t *)dst_data_cpu)[i*h*n*m + j*n*m + k*m + p] =
                                f32_to_bf16(((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + p]);
        } else {
            // dup_cont_f32_no_cont_bf16
            for (int i = 0; i < bs; ++i)
                for (int j = 0; j < h; ++j)
                    for (int k = 0; k < n; ++k)
                        for (int p = 0; p < m; ++p)
                            ((uint16_t *)dst_data_cpu)[i*dst->nb[3]/sizeof(uint16_t) +
                                                    j*dst->nb[2]/sizeof(uint16_t) +
                                                    k*dst->nb[1]/sizeof(uint16_t) +
                                                    p*dst->nb[0]/sizeof(uint16_t)] =
                                    f32_to_bf16(((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + p]);
        }
    } else {
        // TODO: Not support dup_no_cont_f32_cont_bf16
        if (ggml_is_contiguous(dst)) {
            GGML_ABORT("FATAL ERROR: Non-Contiguous BF16 destination is not supported for"
                       "op DUP in R-NPU\n");
        } else {
            // dup_no_cont_f32_no_cont_bf16
            for (int i = 0; i < bs; ++i)
                for (int j = 0; j < h; ++j)
                    for (int k = 0; k < n; ++k)
                        for (int p = 0; p < m; ++p)
                            ((uint16_t *)dst_data_cpu)[i*dst->nb[3]/sizeof(uint16_t) +
                                                    j*dst->nb[2]/sizeof(uint16_t) +
                                                    k*dst->nb[1]/sizeof(uint16_t) +
                                                    p*dst->nb[0]/sizeof(uint16_t)] =
                                    f32_to_bf16(((float *)src0_data_cpu)[i*src0->nb[3]/sizeof(float) +
                                                                        j*src0->nb[2]/sizeof(float) +
                                                                        k*src0->nb[1]/sizeof(float) +
                                                                        p*src0->nb[0]/sizeof(float)]);
        }
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_unary_op_data_from_cpu_to_dev(dst, src0_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_dup_f32_bf16_cpu", src0->ne, nullptr, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_dup_f32_f32_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_TENSOR_UNARY_OP_LOCALS;

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * dst_data_cpu;
    copy_unary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &dst_data_cpu);

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    ((float *)dst_data_cpu)[i*h*n*m + j*n*m + k*m + p] =
                        ((float *)src0_data_cpu)[i*h*n*m + j*n*m + k*m + p];
                }
            }
        }
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_unary_op_data_from_cpu_to_dev(dst, src0_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_dup_f32_f32_cpu", src0->ne, nullptr, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_dup_f32_bf16(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    // TODO: LAUNCH YOUR KERNEL HERE
    GGML_UNUSED(ctx);
    // GGML_UNUSED(dst);

    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    int src0_bs = src0->ne[3];
    int src0_h = src0->ne[2];
    int src0_n = src0->ne[1];
    int src0_m = src0->ne[0];

    int dst_bs = dst->ne[3];
    int dst_h = dst->ne[2];
    int dst_n = dst->ne[1];
    int dst_m = dst->ne[0];

    int src0_nb3 = src0->nb[3];
    int src0_nb2 = src0->nb[2];
    int src0_nb1 = src0->nb[1];
    int src0_nb0 = src0->nb[0];

    int dst_nb3 = dst->nb[3];
    int dst_nb2 = dst->nb[2];
    int dst_nb1 = dst->nb[1];
    int dst_nb0 = dst->nb[0];

    static struct NPUKernelDesc* dup_cont_f32_cont_bf16 = NULL;
    static struct NPUKernelDesc* dup_cont_f32_no_cont_bf16 = NULL;
    static struct NPUKernelDesc* dup_no_cont_f32_no_cont_bf16 = NULL;
    if (ggml_is_contiguous(src0)) {
        if (ggml_is_contiguous(dst)) {
            // dup_cont_f32_cont_bf16
            if (dup_cont_f32_cont_bf16 == NULL) {
                ssize_t err = npu_runtime_get_kernel_desc(&runtime, "dup_cont_f32_cont_bf16", &dup_cont_f32_cont_bf16);
                if (err != NPU_SUCCESS) {
                    npu_runtime_deinit(&runtime);
                    GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
                }
                GGML_LOG_INFO("[INFO] Loaded dup_cont_f32_cont_bf16 kernel desc successfully.\n");
            }
            int64_t start_time = RECORD_OP_START();
            int kernel_ret;
            npu_runtime_kernel_start(&runtime, dup_cont_f32_cont_bf16, &kernel_ret, 10,
                                    src0->data, dst->data, src0_bs, src0_h, src0_n, src0_m, dst_bs, dst_h, dst_n, dst_m);
            RECORD_OP_END("dup_cont_f32_cont_bf16", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
        } else {
            // dup_cont_f32_no_cont_bf16
            if (dup_cont_f32_no_cont_bf16 == NULL) {
                ssize_t err = npu_runtime_get_kernel_desc(&runtime, "dup_cont_f32_no_cont_bf16", &dup_cont_f32_no_cont_bf16);
                if (err != NPU_SUCCESS) {
                    npu_runtime_deinit(&runtime);
                    GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
                }
                GGML_LOG_INFO("[INFO] Loaded dup_cont_f32_no_cont_bf16 kernel desc successfully.\n");
            }
            int64_t start_time = RECORD_OP_START();
            int kernel_ret;
            // npu_runtime_kernel_start(&runtime, dup_cont_f32_no_cont_bf16, &kernel_ret, 10,
            //                         src0->data, dst->data, src0_bs, src0_h, src0_n, src0_m, dst_bs, dst_h, dst_n, dst_m);

            npu_runtime_kernel_start(&runtime, dup_cont_f32_no_cont_bf16, &kernel_ret, 18,
                                src0->data, dst->data, src0_bs, src0_h, src0_n, src0_m, dst_bs, dst_h, dst_n, dst_m,
                                src0_nb3, src0_nb2, src0_nb1, src0_nb0, dst_nb3, dst_nb2, dst_nb1, dst_nb0);
            RECORD_OP_END("dup_cont_f32_no_cont_bf16", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
        }
    } else {
        // TODO: Not support dup_no_cont_f32_cont_bf16
        if (ggml_is_contiguous(dst)) {
            GGML_ABORT("FATAL ERROR: Non-Contiguous BF16 destination is not supported for"
                       "op DUP in R-NPU\n");
        } else {
            // dup_no_cont_f32_no_cont_bf16
            if (dup_no_cont_f32_no_cont_bf16 == NULL) {
                ssize_t err = npu_runtime_get_kernel_desc(&runtime, "dup_no_cont_f32_no_cont_bf16", &dup_no_cont_f32_no_cont_bf16);
                if (err != NPU_SUCCESS) {
                    npu_runtime_deinit(&runtime);
                    GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
                }
                GGML_LOG_INFO("[INFO] Loaded dup_no_cont_f32_no_cont_bf16 kernel desc successfully.\n");
            }
            int64_t start_time = RECORD_OP_START();
            int kernel_ret;
            npu_runtime_kernel_start(&runtime, dup_no_cont_f32_no_cont_bf16, &kernel_ret, 18,
                                src0->data, dst->data, src0_bs, src0_h, src0_n, src0_m, dst_bs, dst_h, dst_n, dst_m,
                                src0_nb3, src0_nb2, src0_nb1, src0_nb0, dst_nb3, dst_nb2, dst_nb1, dst_nb0);
            RECORD_OP_END("dup_no_cont_f32_no_cont_bf16", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
        }
    }

}

static void ggml_rnpu_dup_f32_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    // TODO: LAUNCH YOUR KERNEL HERE
    GGML_UNUSED(ctx);
    // GGML_UNUSED(dst);
    const ggml_tensor * src0 = dst->src[0];
    int src0_bs = src0->ne[3];
    int src0_h = src0->ne[2];
    int src0_n = src0->ne[1];
    int src0_m = src0->ne[0];

    int dst_bs = dst->ne[3];
    int dst_h = dst->ne[2];
    int dst_n = dst->ne[1];
    int dst_m = dst->ne[0];

    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    static struct NPUKernelDesc* dup_cont_f32_cont_f32 = NULL;
    if (dup_cont_f32_cont_f32 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "dup_cont_f32_cont_f32", &dup_cont_f32_cont_f32);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded dup_cont_f32_cont_f32 kernel desc successfully.\n");
    }
    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, dup_cont_f32_cont_f32, &kernel_ret, 10,
                            src0->data, dst->data, src0_bs, src0_h, src0_n, src0_m, dst_bs, dst_h, dst_n, dst_m);
    RECORD_OP_END("dup_cont_f32_cont_f32", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_dup(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    // TODO: Only support copy from fp32 tensor.
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    switch (dst->type) {
        case GGML_TYPE_BF16:
            // ggml_rnpu_dup_f32_bf16_cpu(ctx, dst);
            ggml_rnpu_dup_f32_bf16(ctx, dst);
            break;
        case GGML_TYPE_F32:
            // ggml_rnpu_dup_f32_f32_cpu(ctx, dst);
            ggml_rnpu_dup_f32_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op DUP in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

void ggml_rnpu_cpy(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    ggml_rnpu_dup(ctx, dst);
}

void ggml_rnpu_cont(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));
    ggml_rnpu_dup(ctx, dst);
}

void ggml_rnpu_reshape(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    // Empty op
}

void ggml_rnpu_view(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    // Empty op
}

static void ggml_rnpu_permute_f32_2_1_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const int axis0 = ((int32_t *) dst->op_params)[0];
    const int axis1 = ((int32_t *) dst->op_params)[1];
    const int axis2 = ((int32_t *) dst->op_params)[2];
    const int axis3 = ((int32_t *) dst->op_params)[3];

    // TODO: Only support fp32 permute
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    // TODO: Only support permute(1,2)
    GGML_ASSERT(axis0 == 0 && axis1 == 2 && axis2 == 1 && axis3 == 3);

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * dst_data_cpu;
    copy_unary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &dst_data_cpu);

    int bs = src0->ne[3];
    int n = src0->ne[2];
    int h = src0->ne[1];
    int w = src0->ne[0];

    int row_size = w * sizeof(float);

    char * a = (char *) src0_data_cpu;
    char * res = (char *) dst_data_cpu;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < h; ++k) {
                // Copy a[i][j][k][:] to res[i][k][j][:]
                const char * a_ptr = (char *)a + k*src0->nb[1] + j*src0->nb[2] + i*src0->nb[3];
                size_t id = (k*n + j) * row_size;
                std::memcpy((char *)res + id, a_ptr, row_size);
            }
        }
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_unary_op_data_from_cpu_to_dev(dst, src0_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_permute_f32_2_1_cpu", src0->ne, nullptr, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_permute_f32_2_1(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];

    // Kernel lazy load
    static struct NPUKernelDesc* permute_f32_2_1 = NULL;
    if (permute_f32_2_1 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "permute_f32_2_1", &permute_f32_2_1);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded permute_f32_2_1 kernel desc successfully.\n");
    }

    // Launch kernel
    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, permute_f32_2_1, &kernel_ret, 6,
                            src0->data, dst->data, bs, h, n, m);
    GGML_UNUSED(ctx);
    RECORD_OP_END("permute_f32_2_1", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_permute(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            // ggml_rnpu_permute_f32_2_1_cpu(ctx, dst);
            ggml_rnpu_permute_f32_2_1(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op PERMUTE in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

void ggml_rnpu_transpose(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    // Empty op
}

static void ggml_rnpu_rms_norm_f32_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];

    // TODO: Only support fp32 RMS_NORM
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * dst_data_cpu;
    copy_unary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &dst_data_cpu);

    int b = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    float * res = (float *)dst_data_cpu;
    float * a = (float *)src0_data_cpu;

    for (int i = 0; i < (b * h * n); ++i) {
        double sum = 0.0;
        for (int j = 0; j < m; ++j) {
            sum += (double)(a[i * m + j] * a[i * m + j]);
        }
        float mean = (float)(sum / m);
        for (int j = 0; j < m; ++j) {
            res[i * m + j] = a[i * m + j];
            res[i * m + j] *= 1.0f/std::sqrt(mean + eps);
        }
    }
    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_unary_op_data_from_cpu_to_dev(dst, src0_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_rms_norm_f32_cpu", src0->ne, nullptr, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_rms_norm_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    // Kernel lazy load
    static struct NPUKernelDesc* rms_norm_f32 = NULL;
    if (rms_norm_f32 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "rms_norm_f32", &rms_norm_f32);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded rms_norm_f32 kernel desc successfully.\n");
    }

    // Launch kernel
    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, rms_norm_f32, &kernel_ret, 7,
                            src0->data, dst->data, bs, h, n, m, float_as_u32(eps));
    GGML_UNUSED(ctx);
    RECORD_OP_END("rms_norm_f32", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_rms_norm(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            // ggml_rnpu_rms_norm_f32_cpu(ctx, dst);
            ggml_rnpu_rms_norm_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op RMS_NORM in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static void ggml_rnpu_softmax_f32_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TODO: Only support fp32 Softmax
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    const int bs          = src0->ne[3];
    const uint32_t h      = src0->ne[2];
    const int n           = src0->ne[1];
    const int m           = src0->ne[0];
    const uint32_t h_log2 = 1u << (uint32_t) floor(log2(h));

    const float m0 = powf(2.0f, -(max_bias       ) / h_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / h_log2);

    float * a = (float *)src0_data_cpu;
    float * b = (float *)src1_data_cpu;
    float * res = (float *)dst_data_cpu;
    for (int i = 0; i < bs; ++i) { // batch
        for (int j = 0; j < (h * n); ++j) { // attn-heads * seq_len
            const uint32_t hidx = (j / n); // current head idx
            // const float slope = (max_bias > 0.0f) ? h < h_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - h_log2) + 1) : 1.0f;
            const float slope = 1.0f; // only support max_bias == 0.0f in qwen2

            // scale
            // a[i][j][k][:] *= scale
            for (int p = 0; p < m; ++p) {
                a[i*h*n*m + j*m + p] = a[i*h*n*m + j*m + p] * scale;
            }

            // mask
            // a[i][j][k][:] += slope * b[i][0][k][:]
            for (int p = 0; p < m; ++p) {
                a[i*h*n*m + j*m + p] += slope * b[i*h*n*m + (j%n)*m + p];
            }

            // get max value for dim[-1]
            // maxval = max(x(i))
            float maxval = -INFINITY;

            for (int p = 0; p < m; ++p) {
                if (a[i*h*n*m + j*m + p] > maxval)
                    maxval = a[i*h*n*m + j*m + p];
            }

            // a[i][j][k][:] -= max
            // x(i) = e^(x(i) - max)
            // sum = 1 / (sum of e^(x(i) - max))
            double sum = 0.0f;
            for (int p = 0; p < m; ++p) {
                float val = std::exp(a[i*h*n*m + j*m + p] - maxval);
                sum += (double)val;
                res[i*h*n*m + j*m + p] = val;
            }
            sum = 1.0 / sum;

            // res[i][j][k][:] *= 1/sum
            // x(i) = e^(x(i) - max) / (sum of e^(x(i) - max))
            for (int p = 0; p < m; ++p) {
                res[i*h*n*m + j*m + p] *= (float)sum;
            }
        }
    }
    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_softmax_f32_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_softmax_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    // TODO: Only support fp32 Softmax
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    float scale    = 1.0f;
    float max_bias = 0.0f;
    memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    const uint32_t h_log2 = 1u << (uint32_t) floor(log2(h));

    const float m0 = powf(2.0f, -(max_bias       ) / h_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / h_log2);
    static struct NPUKernelDesc* softmax_f32_f32_f32 = NULL;
    if (softmax_f32_f32_f32 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "softmax_f32_f32_f32", &softmax_f32_f32_f32);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded softmax_f32_f32_f32 kernel desc successfully.\n");
    }

    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, softmax_f32_f32_f32, &kernel_ret, 8,
                             src0->data, src1->data, dst->data, float_as_u32(scale),
                             (uint64_t)bs, (uint64_t)h, (uint64_t)n, (uint64_t)m);
    GGML_UNUSED(ctx);
    RECORD_OP_END("softmax_f32_f32_f32", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_softmax(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_rnpu_softmax_f32(ctx, dst);
            // ggml_rnpu_softmax_f32_cpu(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op SOFTMAX in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static void ggml_rnpu_silu_f32_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const ggml_tensor * src0 = dst->src[0];

    // TODO: Only support fp32 SiLU
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * dst_data_cpu;
    copy_unary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &dst_data_cpu);

    size_t src0_size = ggml_nelements(src0);

    for (size_t i = 0; i < src0_size; ++i) {
        float x = ((float *)src0_data_cpu)[i];
        ((float *)dst_data_cpu)[i] = x / (1.0f + expf(-x));
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_unary_op_data_from_cpu_to_dev(dst, src0_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_silu_f32_cpu", src0->ne, nullptr, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_silu_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];

    // Kernel lazy load
    static struct NPUKernelDesc* silu_f32 = NULL;
    if (silu_f32 == NULL) {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "silu_f32", &silu_f32);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded silu_f32 kernel desc successfully.\n");
    }

    // Launch kernel
    int64_t start_time = RECORD_OP_START();
    int kernel_ret;
    // GGML_LOG_WARN("silu_f32 %p %p %d %d %d %d\n", src0->data, dst->data, bs, h, n, m);
    npu_runtime_kernel_start(&runtime, silu_f32, &kernel_ret, 6,
                            src0->data, dst->data, bs, h, n, m);
    GGML_UNUSED(ctx);
    RECORD_OP_END("silu_f32", src0->ne, nullptr, dst->ne, start_time, "R-NPU");
}

void ggml_rnpu_silu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_F32:
            // ggml_rnpu_silu_f32_cpu(ctx, dst);
            ggml_rnpu_silu_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op SiLU in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

static const int QK = 256;

struct block_e4m3_q{
    float scale;
    uint8_t data[QK];
};

static uint16_t fp32_to_bf16(float f) {
    uint16_t res;
    union {
        float value;
        uint32_t bits;
    } fp32;
    fp32.value = f;
    if ((fp32.bits & 0x7fffffff) > 0x7f800000) { /* nan */
        res = (fp32.bits >> 16) | 64; /* force to quiet */
        return res;
    }
    res = (fp32.bits + (0x7fff + ((fp32.bits >> 16) & 1))) >> 16;
    return res;
}

static float bf16_to_fp32(uint16_t h) {
    union {
        float value;
        uint32_t bits;
    } fp32;
    fp32.bits = (static_cast<uint32_t>(h) << 16);
    return fp32.value;
}

static uint8_t fp32_to_fp8_e4m3_round(float f) {
    uint32_t x = 0;
    memcpy(&x, &f, sizeof(float));
    uint8_t sign = (x >> 31) & 0x1;
    int32_t exponent = ((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x007FFFFF;

    if ((x & 0x7F800000) == 0x7F800000) {
      return sign ? 0xFF : 0x7F;
    }
    if (exponent < -127) {
      return sign << 7;
    }

    int32_t e4m3_exponent = exponent + 7;
    uint8_t e4m3_mantissa = (mantissa >> 20) & 0x7;

    if (e4m3_exponent <= 0) {
      return sign << 7;
    }

    if (e4m3_exponent >= 15) {
      return (sign << 7) | 0x7F;
    }

    uint32_t round_bit = (mantissa >> 19) & 0x1;
    uint32_t sticky_bits = (mantissa & 0x0007FFFF) ? 1 : 0;

    uint8_t result = (sign << 7) | (e4m3_exponent << 3) | e4m3_mantissa;

    if (round_bit && (sticky_bits || (e4m3_mantissa & 1))) {
        result += 1;
    }

    if ((result & 0x7F) == 0x7F) {
      result = (sign << 7) | 0x7F;
    }

    return result;
}

static float fp8_e4m3_to_fp32(uint8_t x) {
    int sign = (x & 0x80) ? -1 : 1;
    int exponent = (x >> 3) & 0x0F;
    float mantissa = (x & 0x07) / 8.0f;
    if (exponent == 0) {
      if (mantissa == 0.0f) {
        return 0.0f * sign;
      }
      return sign * mantissa * 0.015625f;
    }

    mantissa += 1.0f;
    float exp_value = 1.0f;
    int exp = exponent - 7;

    if (exp > 0) {
      for (int i = 0; i < exp; i++) {
        exp_value *= 2.0f;
      }
    } else {
      for (int i = 0; i < -exp; i++) {
        exp_value /= 2.0f;
      }
    }

    return sign * mantissa * exp_value;
}

union fp32_int32 {
    float f;
    uint32_t bits;
};

// (2 - std::exp2(-M + 1)) * std::exp2(std::exp2(E - 1)) = (2 - 2**(-2)) *(2**(2**(3))) = (2 - 2**(-2)) * 2**(8)
static const float e4m3q_max = (2 - std::exp2(-2)) * (std::exp2(8));

// (std::exp2(-M)) * (std::exp2(2 - std::exp2(E - 1))) = (2**(-3)) * (2**(2 - 2**3)) = 2**(-9)
static const float e4m3q_min = std::exp2(-3) * std::exp2(-6);
static uint8_t float_to_e4m3q(float h) {
    uint8_t out;
    fp32_int32 in = {h};
    out = (in.bits >> 24) & 0x80;
    in.bits &= 0x7fffffff;
    if (in.f >= e4m3q_max) {
        out |= 0x7E;
    } else if (in.f < e4m3q_min) { // => 0.
    } else {
        in.f *= std::exp2(7 - 127); // E==4 in e4m3q, so E_BIAS = 2^(E-1) - 1 = 7
        uint32_t eps = (0x3fffff>>3) + ((in.bits >> (23-3)) & 0x1);
        in.bits += eps;
        out |= (in.bits >> (23-3)) & 0x7F;
    }
    return out;
}

static uint8_t float_to_e4m3q_nemu(float h) {
    uint8_t out;
    fp32_int32 in = {h};
    int32_t exponent = ((in.bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = in.bits & 0x007FFFFF;

    out = (in.bits >> 24) & 0x80;

    in.bits &= 0x7fffffff;

    if ((exponent > 8) || ((exponent == 8) && (mantissa >= 0x600000))) {
        out |= 0x7E;
    } else if (exponent < -7) { // => 0.
        // out |= 0x1;
    } else {
        in.bits = (in.bits & 0x807fffff) | (((exponent + 7) & 0xff) << 23);
        uint32_t eps = (0x3fffff>>3) + ((in.bits >> (23-3)) & 0x1);
        in.bits += eps;
        out |= (in.bits >> (23-3)) & 0x7F ;
    }
    return out;
}

static float e4m3q_to_float(uint8_t h) {
    fp32_int32 out = {0};
    out.bits = h & 0x80;
    out.bits <<= 24;
    uint32_t h_bits = h & 0x7F;
    h_bits <<= (23 - 3); // 23 - FP8::M (M==3 in e4m3q)
    out.bits |= h_bits;
    // 2^(127 - FP8::E_BIAS)
    out.f *= std::pow(2, 127 - 7); // E==4 in e4m3q, so E_BIAS = 2^(E-1) - 1 = 7
    return out.f;
}

static void block_fp32_to_e4m3q(float * b_f32, block_e4m3_q * b_fp8, int block_size) {
    float m = 0;
    for (int i = 0; i < block_size; ++i) {
        m = std::max(std::abs(b_f32[i]),m);
    }

    const float D = e4m3q_max / m;

    // set dequantize scale
    b_fp8->scale = m / e4m3q_max;

    for (int i = 0; i < block_size; ++i) {
        // b_fp8->data[i] = fp32_to_fp8_e4m3_round(b_f32[i] * D);
        // b_fp8->data[i] = float_to_e4m3q(b_f32[i] * D);
        b_fp8->data[i] = float_to_e4m3q_nemu(b_f32[i] * D);
    }
}

static void ggml_rnpu_matmul_e4m3q_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    int bs = src1->ne[3];
    int h = src1->ne[2];

    assert(src1->ne[3] == src0->ne[3]);
    assert(src1->ne[2] == src0->ne[2]);

    int n = src0->ne[1];
    int m = src1->ne[1];
    int k = src1->ne[0];
    assert(src1->ne[0] == src0->ne[0]);
    assert(k % QK == 0);
    // rowsize = sizeof block scales + sizeof block data
    int row_size = (k/QK)*sizeof(float) + k;

    uint8_t * a = (uint8_t *) src0_data_cpu;
    float * b = (float *) src1_data_cpu;
    float * res = (float *) dst_data_cpu;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    float dot_sum = 0.0; // DOT SUM

                    // vector-dot loop
                    for (int ib = 0; ib < k/QK; ++ib) {
                        float block_sum = 0.0f; // BLOCK_SUM

                        // get A block (fp8)
                        block_e4m3_q * a_block_fp8 = (block_e4m3_q *)(a + i*h*n*row_size + j*n*row_size + l*row_size + ib*sizeof(block_e4m3_q));

                        // get B block (fp32)
                        float * b_block_f32 = b + i*h*m*k + j*m*k + p*k + (ib*QK);

                        // transfer B block (to fp8)
                        static struct block_e4m3_q b_block_fp8;
                        block_fp32_to_e4m3q(b_block_f32, &b_block_fp8, QK);

                        // in-block loop
                        for (int x = 0; x < QK; ++x) {
                            float ea = e4m3q_to_float(a_block_fp8->data[x]);
                            float eb = e4m3q_to_float(b_block_fp8.data[x]);
                            // float ea = fp8_e4m3_to_fp32(a_block_fp8->data[x]);
                            // float eb = fp8_e4m3_to_fp32(b_block_fp8.data[x]);
                            block_sum += ea * eb;
                        }

                        // add block sum to dot sum (with scaling)
                        float scale_a = a_block_fp8->scale;
                        float scale_b = b_block_fp8.scale;
                        dot_sum += scale_a * scale_b * block_sum;
                    }

                    // vector-dot result
                    res[i*h*m*n + j*m*n + p*n + l] = dot_sum;
                }
            }
        }
    }
    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_matmul_e4m3q_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_matmul_e4m3q_cpu_reordered(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    int bs = src1->ne[3];
    int h = src1->ne[2];
    int n = src0->ne[1];
    int m = src1->ne[1];
    int k = src1->ne[0];

    assert(src1->ne[3] == src0->ne[3]);
    assert(src1->ne[2] == src0->ne[2]);
    assert(src1->ne[0] == src0->ne[0]);
    assert(k % QK == 0);
    // rowsize = sizeof block scales + sizeof block data
    int row_size = (k/QK)*sizeof(float) + k;

    uint8_t * a_base = (uint8_t *) src0_data_cpu;
    float * b = (float *) src1_data_cpu;
    float * res = (float *) dst_data_cpu;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    float dot_sum = 0.0; // DOT SUM

                    uint8_t * row_start = a_base + (i*h*n + j*n + l) * row_size;
                    
                    uint8_t * row_data_ptr = row_start;
                    float   * row_scale_ptr = (float *)(row_start + k);

                    for (int ib = 0; ib < k/QK; ++ib) {
                        float block_sum = 0.0f; // BLOCK_SUM

                        // get A block (fp8)
                        uint8_t * a_qs_ptr = row_data_ptr + (ib * QK);

                        // get B block (fp32)
                        float * b_block_f32 = b + i*h*m*k + j*m*k + p*k + (ib*QK);

                        // transfer B block (to fp8)
                        static struct block_e4m3_q b_block_fp8;
                        block_fp32_to_e4m3q(b_block_f32, &b_block_fp8, QK);

                        for (int x = 0; x < QK; ++x) {
                            float ea = e4m3q_to_float(a_qs_ptr[x]);
                            float eb = e4m3q_to_float(b_block_fp8.data[x]);
                            block_sum += ea * eb;
                        }

                        // add block sum to dot sum (with scaling)
                        float scale_a = row_scale_ptr[ib];
                        float scale_b = b_block_fp8.scale;
                        dot_sum += scale_a * scale_b * block_sum;
                    }

                    // vector-dot result
                    res[i*h*m*n + j*m*n + p*n + l] = dot_sum;
                }
            }
        }
    }

    GGML_UNUSED(ctx);
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    RECORD_OP_END("ggml_rnpu_matmul_e4m3q_cpu_reordered", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static void ggml_rnpu_matmul_bf16_cpu(ggml_backend_rnpu_context& ctx, struct ggml_tensor * dst) {
    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    int bs = src1->ne[3];
    int h = src1->ne[2];

    int rbs = bs / src0->ne[3];
    int rj = h / src0->ne[2];
    assert(bs >= src0->ne[3]);
    assert(h >= src0->ne[2]);

    int n = src0->ne[1];
    int m = src1->ne[1];
    int k = src1->ne[0];
    assert(src1->ne[0] == src0->ne[0]);

    uint16_t * a = (uint16_t *) src0_data_cpu;
    float * b = (float *) src1_data_cpu;
    float * res = (float *) dst_data_cpu;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            // broadcast in dim(0)
            int i0 = i / rbs;
            // broadcase in dim(1)
            int j0 = j / rj;

            // b[i][j][:][:] * a[i0][j0][:][:]^T
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    float val = 0.0;
                    // res[i][j][p][l] = b[i][j][p][:] * a[i0][j0][l][:]^T
                    for (int x = 0; x < k; ++x) {
                        uint16_t ea = a[(i0*src0->nb[3] + j0*src0->nb[2] + l*src0->nb[1] + x*src0->nb[0]) / sizeof(uint16_t)];
                        float eb = b[(i*src1->nb[3] + j*src1->nb[2] + p*src1->nb[1] + x*src1->nb[0]) / sizeof(float)];
                        val += bf16_to_fp32(ea) * bf16_to_fp32(fp32_to_bf16(eb));
                    }
                    res[i*h*m*n + j*m*n + p*n + l] = val;
                }
            }
        }
    }
    // FIXME: Copy result to device memory
    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    GGML_UNUSED(ctx);
    RECORD_OP_END("ggml_rnpu_matmul_bf16_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

static int gmma_e4m3q_fp32_AT_QK256(
                        void *a, void *b, void *c,
                        int m, int n, int k,
                        int lda, int ldb, int ldc) {
    static struct NPUKernelDesc *gmma_e4m3q_fp32_AT_QK256_desc = NULL;
    if (gmma_e4m3q_fp32_AT_QK256_desc == NULL)
    {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "gmma_e4m3q_fp32_AT_QK256", &gmma_e4m3q_fp32_AT_QK256_desc);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("[ERROR] %s: Failed to get kernel description: err=%zu\n", __func__, err);
        }
        GGML_LOG_INFO("[INFO] Loaded gmma_e4m3q_fp32_AT_QK256 kernel desc successfully.\n");
    }
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, gmma_e4m3q_fp32_AT_QK256_desc, &kernel_ret, 9,
                             a, b, c,
                             (uint64_t)m, (uint64_t)n, (uint64_t)k,
                             (uint64_t)lda, (uint64_t)ldb, (uint64_t)ldc);
    return kernel_ret;
}

static void ggml_rnpu_matmul_e4m3q(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    int bs = src1->ne[3];
    int h = src1->ne[2];
    int h0 = src0->ne[2];

    // dim(0) and dim(1) only broadcast from src0 to src1
    int rbs = bs / src0->ne[3];
    int rj = h / src0->ne[2];
    assert(bs >= src0->ne[3]);
    assert(h >= src0->ne[2]);
    assert(bs == 1);
    assert(h == 1);

    assert(dst->nb[0] == sizeof(float));

    // src0 shape: (bs0, h0, m, k)
    // src1 shape: (bs, h, n, k)
    // dst shape: (bs, h, n, m)
    int m = src0->ne[1];
    int n = src1->ne[1];
    int k = src1->ne[0];
    assert(src1->ne[0] == src0->ne[0]);

    // Call Kernel
    gmma_e4m3q_fp32_AT_QK256(
        src0->data, src1->data, dst->data,
        m, n, k,
        src1->ne[0] * (QK + sizeof(float)) / QK, src1->ne[0], dst->nb[1] / sizeof(float)
    );

    GGML_UNUSED(ctx);
    RECORD_OP_END("ggml_rnpu_matmul_e4m3q", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");
}

static int gmma_bf16_fp32_AT_QK256(
    void *a, void *b, void *c,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    int g, int h,
    int stride_g, int stride_h, int a_is_transposed)
{
    // GGML_LOG_INFO("\nm:%d, n:%d, k:%d, g:%d, h:%d\n", m, n, k, g, h);
    // GGML_LOG_INFO("lda:%d, ldb:%d, ldc:%d, stride_g:%d, stride_h:%d\n", lda, ldb, ldc, stride_g, stride_h);
    static struct NPUKernelDesc *gmma_bf16_fp32_AT_QK256_desc = NULL;
    if (gmma_bf16_fp32_AT_QK256_desc == NULL)
    {
        ssize_t err = npu_runtime_get_kernel_desc(&runtime, "gmma_bf16_fp32_AT_QK256", &gmma_bf16_fp32_AT_QK256_desc);
        if (err != NPU_SUCCESS) {
            npu_runtime_deinit(&runtime);
            GGML_ABORT("Failed to get kernel description: err=%zu\n", err);
        }
        GGML_LOG_INFO("[INFO] Loaded gmma_bf16_fp32_AT_QK256 kernel desc successfully.\n");
    }
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, gmma_bf16_fp32_AT_QK256_desc, &kernel_ret, 14,
                             a, b, c,
                             (uint64_t)m, (uint64_t)n, (uint64_t)k,
                             (uint64_t)lda, (uint64_t)ldb, (uint64_t)ldc,
                             (uint64_t)g, (uint64_t)h,
                             (uint64_t)stride_g, (uint64_t)stride_h, (uint64_t)a_is_transposed);
    return kernel_ret;
}

static void * debug_matmul_bf16_cpu(ggml_backend_rnpu_context& ctx, struct ggml_tensor * dst) {
    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    // FIXME: Copy from device to cpu memory
    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    int bs = src1->ne[3];
    int h = src1->ne[2];

    int rbs = bs / src0->ne[3];
    int rj = h / src0->ne[2];
    assert(bs >= src0->ne[3]);
    assert(h >= src0->ne[2]);

    int n = src0->ne[1];
    int m = src1->ne[1];
    int k = src1->ne[0];
    assert(src1->ne[0] == src0->ne[0]);

    uint16_t * a = (uint16_t *) src0_data_cpu;
    float * b = (float *) src1_data_cpu;
    float * res = (float *) dst_data_cpu;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            // broadcast in dim(0)
            int i0 = i / rbs;
            // broadcase in dim(1)
            int j0 = j / rj;

            // b[i][j][:][:] * a[i0][j0][:][:]^T
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    float val = 0.0;
                    // res[i][j][p][l] = b[i][j][p][:] * a[i0][j0][l][:]^T
                    for (int x = 0; x < k; ++x) {
                        uint16_t ea = a[(i0*src0->nb[3] + j0*src0->nb[2] + l*src0->nb[1] + x*src0->nb[0]) / sizeof(uint16_t)];
                        float eb = b[(i*src1->nb[3] + j*src1->nb[2] + p*src1->nb[1] + x*src1->nb[0]) / sizeof(float)];
                        val += bf16_to_fp32(ea) * bf16_to_fp32(fp32_to_bf16(eb));
                    }
                    res[i*h*m*n + j*m*n + p*n + l] = val;
                }
            }
        }
    }
    // FIXME: Copy result to device memory
    // copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu); // 结果不用写回 dst，暂存在 dst_data_cpu 上
    GGML_UNUSED(ctx);
    RECORD_OP_END("ggml_rnpu_matmul_bf16_cpu", src0->ne, src1->ne, dst->ne, start_time, "CPU");
    return dst_data_cpu; // 返回暂存结果的 buffer 指针 dst_data_cpu
}

static void debug_compare_matmul_bf16(ggml_tensor* dst, void * dst_data_cpu) {
    static int call_count = 0;
    size_t dst_size = ggml_nbytes(dst);
    assert((uint64_t)dst_data_cpu != (uint64_t)src0_cpu_buffer);
    // 把 NPU 运算结果拷贝到 src0_cpu_buffer 上
    npuMemcpy(src0_cpu_buffer, dst->data, dst_size, npuMemcpyDeviceToHost);


    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    GGML_TENSOR_BINARY_OP_LOCALS;
    int bs = src1->ne[3];
    int h = src1->ne[2];
    int n = src0->ne[1];
    int m = src1->ne[1];

    // 逐个对比结果
    float * npu_res = (float *) src0_cpu_buffer;
    float * cpu_res = (float *) dst_data_cpu;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    float cpu_val = cpu_res[i*h*m*n + j*m*n + p*n + l];
                    float npu_val = npu_res[i*h*m*n + j*m*n + p*n + l];
                    if (std::abs(cpu_val - npu_val) > 1e-4) {
                        GGML_LOG_ERROR("[ERRRO] Matmul comparing failed!!!\n");
                        GGML_LOG_ERROR("src0->name=%s@0x%09lx, ne=[%ld, %ld, %ld, %ld], nb=[%ld, %ld, %ld, %ld]\n", src0->name, src0->data,
                            src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
                        GGML_LOG_ERROR("src1->name=%s@0x%09lx, ne=[%ld, %ld, %ld, %ld], nb=[%ld, %ld, %ld, %ld]\n", src1->name, src1->data,
                            src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
                        GGML_LOG_ERROR("dst->name=%s@0x%09lx, ne=[%ld, %ld, %ld, %ld], nb=[%ld, %ld, %ld, %ld]\n", dst->name, dst->data,
                            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
                        GGML_ABORT("[ERROR] Matmul comparing failed at [%d][%d][%d][%d] = (%f, %f)\n", i, j, p, l, cpu_val, npu_val);
                    }
                }
            }
        }
    }

    // 每两层打印一次
    if(call_count % 2 == 1) {
        GGML_LOG_INFO("[INFO] Matmul bf16 comparing call count at %d\n", call_count + 1);
    }
    ++call_count;
}

static void ggml_rnpu_matmul_bf16(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    // FIXME: 先用 cpu 版本计算，把 cpu 计算结果存储到 buffer 中
    // void * dst_data_cpu = debug_matmul_bf16_cpu(ctx, dst);

    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    int bs = src1->ne[3];
    int h = src1->ne[2];
    int h0 = src0->ne[2];

    assert(bs >= src0->ne[3]);
    assert(h >= src0->ne[2]);
    assert(src1->ne[0] == src0->ne[0]);

    // src0 shape: (bs0, h0, m, k)
    // src1 shape: (bs, h, n, k)
    // dst shape: (bs, h, n, m)
    int m = src0->ne[1];
    int n = src1->ne[1];
    int k = src1->ne[0];
    assert(src1->ne[0] == src0->ne[0]);
    // GGML_LOG_INFO("name=%s, src0->data: 0x%p, src1->data: 0x%p, dst->data: 0x%p\n", dst->name, src0->data, src1->data, dst->data);

    int a_is_transposed = (src0->nb[0] > src0->nb[1]);
    // Call Kernel
    // GGML_LOG_WARN("gmma_bf16_fp32_AT_QK256 %p %p %p %d %d %d %zu %zu %zu %zu %zu %zu %zu %d\n", src0->data, src1->data, dst->data,
    //     m, n, k,
    //     (a_is_transposed ? src0->nb[0] : src0->nb[1]) / sizeof(uint16_t),
    //     src1->nb[1] / sizeof(float),
    //     dst->nb[1] / sizeof(float),
    //     src0->ne[2], src1->ne[2],
    //     src0->nb[2] / sizeof(uint16_t), src1->nb[2] / sizeof(float),
    //     a_is_transposed);
    gmma_bf16_fp32_AT_QK256(
        src0->data, src1->data, dst->data,
        m, n, k,
        (a_is_transposed ? src0->nb[0] : src0->nb[1]) / sizeof(uint16_t),
        src1->nb[1] / sizeof(float),
        dst->nb[1] / sizeof(float),
        src0->ne[2], src1->ne[2],
        src0->nb[2] / sizeof(uint16_t), src1->nb[2] / sizeof(float),
        a_is_transposed
    );
    GGML_UNUSED(ctx);
    RECORD_OP_END("ggml_rnpu_matmul_bf16", src0->ne, src1->ne, dst->ne, start_time, "R-NPU");

    // FIXME：对比 CPU 和 NPU 运算结果
    // debug_compare_matmul_bf16(dst, dst_data_cpu);
}

static void ggml_rnpu_matmul_mxfp8_cpu_reordered(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    int64_t start_time = RECORD_OP_START();
    const struct ggml_tensor * src0 = dst->src[0]; // Weights (MXFP8 Reordered)
    const struct ggml_tensor * src1 = dst->src[1]; // Activations (BF16 or F32)

    void * src0_data_cpu;
    void * src1_data_cpu;
    void * dst_data_cpu;
    
    copy_binary_op_data_from_dev_to_cpu(dst, &src0_data_cpu, &src1_data_cpu, &dst_data_cpu);

    GGML_TENSOR_BINARY_OP_LOCALS;

    int bs = src1->ne[3];
    int h = src1->ne[2];
    int n = src0->ne[1];
    int m = src1->ne[1];
    int k = src1->ne[0];
    
    assert(src1->ne[0] == src0->ne[0]);
    assert(k % 32 == 0);

    const int QK_MXFP8 = 32;
    const int nb_k = k / QK_MXFP8;

    const int row_size_in_bytes = nb_k * 33;

    uint8_t * a_base = (uint8_t *) src0_data_cpu;
    float   * res    = (float *)   dst_data_cpu;

    const bool src1_is_bf16 = (src1->type == GGML_TYPE_BF16);
    float       * b_base_f32  = src1_is_bf16 ? NULL : (float *) src1_data_cpu;
    ggml_bf16_t * b_base_bf16 = src1_is_bf16 ? (ggml_bf16_t *) src1_data_cpu : NULL;

    int rbs = bs / src0->ne[3];
    int rj  = h  / src0->ne[2];

    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            
            int i0 = i / rbs;
            int j0 = j / rj;

            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    
                    float dot_sum = 0.0f;

                    uint8_t * a_row_start = a_base + (i0 * src0->ne[2] * n * row_size_in_bytes)
                                                   + (j0 * n * row_size_in_bytes)
                                                   + (l * row_size_in_bytes);

                    uint8_t * row_scales = a_row_start;
                    uint8_t * row_payloads = a_row_start + nb_k;

                    int64_t b_offset_base = (i * h * m * k) + (j * m * k) + (p * k);

                    for (int ib = 0; ib < nb_k; ++ib) {
                        
                        uint8_t e = row_scales[ib];
                        
                        uint8_t * qs = row_payloads + (ib * QK_MXFP8);

                        union { uint32_t i; float f; } u_scale;
                        u_scale.i = (uint32_t)e << 23; 
                        float scale = u_scale.f * 0.00390625f;

                        for (int x = 0; x < QK_MXFP8; ++x) {
                            float w = ggml_compute_e4m3_to_fp32(qs[x]);
                            
                            float act;
                            int64_t b_idx = b_offset_base + ib * QK_MXFP8 + x;
                            
                            if (src1_is_bf16) {
                                act = ggml_bf16_to_fp32(b_base_bf16[b_idx]);
                            } else {
                                act = b_base_f32[b_idx];
                            }
                            
                            dot_sum += (w * scale) * act;
                        }
                    }

                    res[i*h*m*n + j*m*n + p*n + l] = dot_sum;
                }
            }
        }
    }

    copy_binary_op_data_from_cpu_to_dev(dst, src0_data_cpu, src1_data_cpu, dst_data_cpu);
    
    GGML_UNUSED(ctx);
    RECORD_OP_END("ggml_rnpu_matmul_mxfp8_cpu_reordered", src0->ne, src1->ne, dst->ne, start_time, "CPU");
}

void ggml_rnpu_mul_mat(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_BF16:
            // ggml_rnpu_matmul_bf16_cpu(ctx, dst);
            ggml_rnpu_matmul_bf16(ctx, dst);
            break;
        case GGML_TYPE_E4M3_Q:
            if (src0->buffer && ggml_backend_buffer_is_rnpu(src0->buffer)) {
                ggml_rnpu_matmul_e4m3q_cpu_reordered(ctx, dst);
            } else {
                ggml_rnpu_matmul_e4m3q_cpu(ctx, dst);
                // ggml_rnpu_matmul_e4m3q(ctx, dst);
            }
            break;
        case GGML_TYPE_MXFP8: 
            ggml_rnpu_matmul_mxfp8_cpu_reordered(ctx, dst); 
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op MATMUL in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}


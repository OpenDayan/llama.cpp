#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../ggml-impl.h"
#include "./common.hpp"
#include "./ops.hpp"
#include "ggml.h"

#include "NPUDriver.h"
#include "NPURuntime.h"
static struct NPURuntimeState runtime;
extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static void ggml_rnpu_get_rows_bf16(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

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

    uint16_t * a = (uint16_t *)src0->data;
    int * b = (int *)src1->data;
    float * res = (float *)dst->data;
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
}

static void ggml_rnpu_get_rows_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));

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

    float * a = (float *)src0->data;
    int * b = (int *)src1->data;
    float * res = (float *)dst->data;
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
}

void ggml_rnpu_get_rows(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_BF16:
            ggml_rnpu_get_rows_bf16(ctx, dst);
            break;
        case GGML_TYPE_F32:
            ggml_rnpu_get_rows_f32(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op GET_ROWS in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}

void ggml_rnpu_op_add(ggml_backend_rnpu_context & ctx, ggml_tensor * dst) {
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
                    ((float *)dst->data)[i * h * n * m + j * n * m + k * m + p] =
                        ((float *)src0->data)[i * h * n * m + j * n * m + k * m + p] +
                        ((float* )src1->data)[i1* h * n * m + j1 * n * m + k1 * m + p1];
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

void ggml_rnpu_op_mul(ggml_backend_rnpu_context & ctx, ggml_tensor * dst) {
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

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    for (int i = 0; i < bs; ++i) {
        int i1 = src1->ne[3] == 1 ? 0 : i;
        for (int j = 0; j < h; ++j) {
            int j1 = src1->ne[2] == 1 ? 0 : i;
            for (int k = 0; k < n; ++k) {
                int k1 = src1->ne[1] == 1 ? 0 : k;
                for (int p = 0; p < m; ++p) {
                    int p1 = src1->ne[0] == 1 ? 0 : p;
                    ((float *)dst->data)[i * h * n * m + j * n * m + k * m + p] =
                        ((float *)src0->data)[i * h * n * m + j * n * m + k * m + p] *
                        ((float* )src1->data)[i1* h * n * m + j1 * n * m + k1 * m + p1];
                }
            }
        }
    }

    GGML_UNUSED(ctx);
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

void ggml_rnpu_rope(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
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
            int64_t p = ((int32_t *)src1->data)[j];
            rope_cache_init(cache, p, theta_scale,
                            static_cast<float>(freq_scale), dst->ne[0]);
            for (int k = 0; k < n; ++k) { // attn-heads
                // fill res[i][j][k][:prope.n_dims]
                for (int p = 0; p < n_dims; p += 2) {
                    const int64_t ic = p / 2;
                    const float cos_theta = cache[p];
                    const float sin_theta = cache[p + 1];

                    const float x0 = ((float *)src0->data)[i*h*n*m + j*n*m + k*m + ic];
                    const float x1 = ((float *)src0->data)[i*h*n*m + j*n*m + k*m + ic + n_dims / 2];

                    ((float *)dst->data)[i*h*n*m + j*n*m + k*m + ic] =
                                                    (x0*cos_theta) - (x1*sin_theta);
                    ((float *)dst->data)[i*h*n*m + j*n*m + k*m + ic + n_dims / 2] =
                                                    (x0*sin_theta) + (x1*cos_theta);
                }

                // fill remain res[i][j][k][prope.n_dims:]
                for (size_t p = n_dims; p < (size_t)m; p += 2) {
                    ((float *)dst->data)[i*h*n*m + j*n*m + k*m + p] =
                            ((float *)src0->data)[i*h*n*m + j*n*m + k*m + p];
                    ((float *)dst->data)[i*h*n*m + j*n*m + k*m + p + 1] =
                            ((float *)src0->data)[i*h*n*m + j*n*m + k*m + p + 1];
                }
            }
        }
    }
    GGML_UNUSED(ctx);
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

static void ggml_rnpu_dup_f32_bf16(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    GGML_TENSOR_UNARY_OP_LOCALS;

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
                            ((uint16_t *)dst->data)[i*h*n*m + j*n*m + k*m + p] =
                                f32_to_bf16(((float *)src0->data)[i*h*n*m + j*n*m + k*m + p]);
        } else {
            // dup_cont_f32_no_cont_bf16
            for (int i = 0; i < bs; ++i)
                for (int j = 0; j < h; ++j)
                    for (int k = 0; k < n; ++k)
                        for (int p = 0; p < m; ++p)
                            ((uint16_t *)dst->data)[i*dst->nb[3]/sizeof(uint16_t) +
                                                    j*dst->nb[2]/sizeof(uint16_t) +
                                                    k*dst->nb[1]/sizeof(uint16_t) +
                                                    p*dst->nb[0]/sizeof(uint16_t)] =
                                    f32_to_bf16(((float *)src0->data)[i*h*n*m + j*n*m + k*m + p]);
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
                            ((uint16_t *)dst->data)[i*dst->nb[3]/sizeof(uint16_t) +
                                                    j*dst->nb[2]/sizeof(uint16_t) +
                                                    k*dst->nb[1]/sizeof(uint16_t) +
                                                    p*dst->nb[0]/sizeof(uint16_t)] =
                                    f32_to_bf16(((float *)src0->data)[i*src0->nb[3]/sizeof(float) +
                                                                        j*src0->nb[2]/sizeof(float) +
                                                                        k*src0->nb[1]/sizeof(float) +
                                                                        p*src0->nb[0]/sizeof(float)]);
        }
    }

    GGML_UNUSED(ctx);
}

static void ggml_rnpu_dup_f32_f32(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));
    GGML_ASSERT(ggml_is_contiguous(dst) && ggml_is_contiguous(src0));
    GGML_TENSOR_UNARY_OP_LOCALS;

    int bs = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int p = 0; p < m; ++p) {
                    ((float *)dst->data)[i*h*n*m + j*n*m + k*m + p] =
                        ((float *)src0->data)[i*h*n*m + j*n*m + k*m + p];
                }
            }
        }
    }

    GGML_UNUSED(ctx);
}

void ggml_rnpu_dup(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    // TODO: Only support copy from fp32 tensor.
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    switch (dst->type) {
        case GGML_TYPE_BF16:
            ggml_rnpu_dup_f32_bf16(ctx, dst);
            break;
        case GGML_TYPE_F32:
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

void ggml_rnpu_permute(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int axis0 = ((int32_t *) dst->op_params)[0];
    const int axis1 = ((int32_t *) dst->op_params)[1];
    const int axis2 = ((int32_t *) dst->op_params)[2];
    const int axis3 = ((int32_t *) dst->op_params)[3];

    // TODO: Only support fp32 permute
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    // TODO: Only support permute(1,2)
    GGML_ASSERT(axis0 == 0 && axis1 == 2 && axis2 == 1 && axis3 == 3);

    int bs = src0->ne[3];
    int n = src0->ne[2];
    int h = src0->ne[1];
    int w = src0->ne[0];

    int row_size = w * sizeof(float);

    char * a = (char *) src0->data;
    char * res = (char *) dst->data;
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
}

void ggml_rnpu_transpose(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    // Empty op
}

void ggml_rnpu_rms_norm(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];

    // TODO: Only support fp32 RMS_NORM
    GGML_ASSERT(src0->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    int b = src0->ne[3];
    int h = src0->ne[2];
    int n = src0->ne[1];
    int m = src0->ne[0];
    float * res = (float *)dst->data;
    float * a = (float *)src0->data;

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
}

void ggml_rnpu_softmax(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
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

    const int bs          = src0->ne[3];
    const uint32_t h      = src0->ne[2];
    const int n           = src0->ne[1];
    const int m           = src0->ne[0];
    const uint32_t h_log2 = 1u << (uint32_t) floor(log2(h));

    const float m0 = powf(2.0f, -(max_bias       ) / h_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / h_log2);

    float * a = (float *)src0->data;
    float * b = (float *)src1->data;
    float * res = (float *)dst->data;
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
}

void ggml_rnpu_silu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];

    // TODO: Only support fp32 SiLU
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(ggml_is_contiguous_1(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    size_t src0_size = ggml_nelements(src0);

    for (size_t i = 0; i < src0_size; ++i) {
        float x = ((float *)src0->data)[i];
        ((float *)dst->data)[i] = x / (1.0f + expf(-x));
    }

    GGML_UNUSED(ctx);
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

static void ggml_rnpu_matmul_bf16(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    // GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

    GGML_TENSOR_BINARY_OP_LOCALS;

    int bs = src1->ne[3];
    int h = src1->ne[2];
    int h0 = src0->ne[2];

    int rbs = bs / src0->ne[3];
    int rj = h / src0->ne[2];
    assert(bs >= src0->ne[3]);
    assert(h >= src0->ne[2]);

    int n = src0->ne[1];
    int m = src1->ne[1];
    int k = src1->ne[0];
    assert(src1->ne[0] == src0->ne[0]);

    uint16_t * a = (uint16_t *) src0->data;
    float * b = (float *) src1->data;
    float * res = (float *) dst->data;
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j < h; ++j) {
            // broadcast in dim(0)
            int i0 = i / rbs;
            // broadcase in dim(1)
            int j0 = j / rj;

            // b[i][j][:][:] * a[i0][j0][:][:]^T
            for (int p = 0; p < m; ++p) {
                for (int l = 0; l < n; ++l) {
                    double val = 0.0;
                    // res[i][j][p][l] = b[i][j][p][:] * a[i0][j0][l][:]^T
                    for (int x = 0; x < k; ++x) {
                        uint16_t ea = a[(i0*src0->nb[3] + j0*src0->nb[2] + l*src0->nb[1] + x*src0->nb[0]) / sizeof(uint16_t)];
                        float eb = b[(i*src1->nb[3] + j*src1->nb[2] + p*src1->nb[1] + x*src1->nb[0]) / sizeof(float)];
                        // b[...] is converted to bf16 then converted to fp32
                        val += (double)(bf16_to_fp32(ea) * bf16_to_fp32(fp32_to_bf16(eb)));
                    }
                    // double to float cast
                    res[i*h*m*n + j*m*n + p*n + l] = val;
                }
            }
        }
    }
    GGML_UNUSED(ctx);
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

static void block_float_to_e4m3q(float * b_f32, block_e4m3_q * b_fp8) {
    float m = 0;
    for (int i = 0; i < QK; ++i) {
        m = std::max(std::abs(b_f32[i]),m);
    }

    const float D = e4m3q_max / m;

    // set dequantize scale
    b_fp8->scale = m / e4m3q_max;

    for (int i = 0; i < QK; ++i) {
        b_fp8->data[i] = float_to_e4m3q(b_f32[i] * D);
    }
}

static void ggml_rnpu_matmul_e4m3q_cpu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(ggml_is_contiguous(src0) && ggml_is_contiguous(src1));

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

    uint8_t * a = (uint8_t *) src0->data;
    float * b = (float *) src1->data;
    float * res = (float *) dst->data;
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
                        block_float_to_e4m3q(b_block_f32, &b_block_fp8);

                        // in-block loop
                        for (int x = 0; x < QK; ++x) {
                            float ea = e4m3q_to_float(a_block_fp8->data[x]);
                            float eb = e4m3q_to_float(b_block_fp8.data[x]);
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
}

template<int N>
constexpr int exp_i2() {
    return 1 << N;
}

template<int N>
constexpr float exp_f2() {
    if constexpr (N>0) return exp_f2<N-1>()*2;
    if constexpr (N<0) return exp_f2<N+1>()/2;
    if constexpr (N==0) return 1.;
}

template<int _E> //, int M=7-E>  1.7 bits!
struct FP8 {
    uint8_t bits;
    using type = FP8<_E>;
    static constexpr int E      = _E;
    static constexpr int M      = (7-_E);
    static constexpr int E_BIAS = exp_i2<E-1>()-1;
    static constexpr float MAX  = (2-exp_f2<-M+1>())*exp_f2<exp_i2<E-1>()>();
    static constexpr float MIN  = exp_f2<-M>()*exp_f2<2-exp_i2<E-1>()>();
};

template <int E>
struct bloc_fp8 {
    float d;
    FP8<E> qs[QK];
};

template<int E>
inline FP8<E> float_to_fp8(float value) {
    FP8<E> out;
    fp32_int32 in = {value};
    // the sign
    out.bits = (in.bits >> 24) & 0x80;
    // value without sign
    in.bits &= 0x7fffffff;
    //GGML_ASSERT(in.bits < 0x7f800000); // +/- infinity or NAN
    if (in.f >= FP8<E>::MAX) {
        out.bits |= 0x7E;
    } else if (in.f < FP8<E>::MIN) { // => 0.
        // OK: S.0000000
    } else {
        in.f *= exp_f2<FP8<E>::E_BIAS-127>();
        // - trunc
        //uint32_t eps = 0;
        // - rounding half away from zero
        //uint32_t eps = 0x400000>>FP8<E>::M;
        // - rounding half toward zero
        //uint32_t eps = 0x3fffff>>FP8<E>::M;
        // - rounding to nearest even
        uint32_t eps = (0x3fffff>>FP8<E>::M) + ((in.bits >> (23-FP8<E>::M)) & 0x1);
        // shift mantissa.
        in.bits += eps;
        out.bits |= (in.bits >> (23-FP8<E>::M)) & 0x7F;
    }
    return out;
}

template <int E>
static inline void float_to_e4m3q(const float* x, bloc_fp8<E>* y, int64_t size) {
    const auto qk_size = size / QK;
    for (int64_t q=0; q<qk_size; ++q) {
        float m = 0;
        for (int64_t i=0; i<QK; i++) {
            m = std::max(std::abs(x[q*QK+i]),m);
        }
        const float D = FP8<E>::MAX/m;
        y[q].d = m/FP8<E>::MAX;
        for (int64_t i=0; i<QK; i++) {
            y[q].qs[i] = float_to_fp8<E>(x[q*QK+i]*D);
        }
    }
}

static int gmma_e4m3q_fp32_AT_QK256(
                        void *a, void *b, void *c,
                        int m, int n, int k,
                        int lda, int ldb, int ldc) {
    static struct NPUKernelDesc *gmma_e4m3q_fp32_AT_QK256_desc = NULL;
    if (gmma_e4m3q_fp32_AT_QK256_desc == NULL)
    {
        ssize_t err = 0;
        err = npu_runtime_init(&runtime);
        if (err != NPU_SUCCESS) {
          printf("[ERROR] %s: Failed to initialize Runtime: err=%d\n", __func__, err);
          exit(-1);
        }
        err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
        if (err != NPU_SUCCESS) {
          printf("[ERROR] %s: Failed to load kernel set: err=%d\n", __func__, err);
          npu_runtime_deinit(&runtime);
          exit(-1);
        }
        err = npu_runtime_get_kernel_desc(&runtime, "gmma_e4m3q_fp32_AT_QK256", &gmma_e4m3q_fp32_AT_QK256_desc);
        if (err != NPU_SUCCESS) {
          printf("[ERROR] %s: Failed to get kernel description: err=%d\n", __func__, err);
          npu_runtime_deinit(&runtime);
          exit(-1);
        }
        printf("[INFO] Loaded gmma_e4m3q_fp32_AT_QK256 kernel desc successfully.\n");
    }
    // static address allocation
    uint64_t a_dev = 0x81000000;
    uint64_t b_dev = 0x83000000;
    uint64_t c_dev = 0x85000000;
    // Copy data to device
    ssize_t ret = 0;
    ret = npu_memcpy_to_dev(&runtime.dev_state, a_dev, a, m * lda * sizeof(uint8_t)); // a
    if (ret < 0)
        printf("[ERROR] Failed to write a to device at 0x%lx: err=%d\n", a_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, b_dev, b, n * ldb * sizeof(uint8_t)); // b
    if (ret < 0)
        printf("[ERROR] Failed to write b to device at 0x%lx: err=%d\n", b_dev, ret);
    ret = npu_memcpy_to_dev(&runtime.dev_state, c_dev, c, n * ldc * sizeof(float));   // c
    if (ret < 0)
        printf("[ERROR] Failed to write c to device at 0x%lx: err=%d\n", c_dev, ret);
    int kernel_ret;
    npu_runtime_kernel_start(&runtime, gmma_e4m3q_fp32_AT_QK256_desc, &kernel_ret, 9,
                             a_dev, b_dev, c_dev,
                             (uint64_t)m, (uint64_t)n, (uint64_t)k,
                             (uint64_t)lda, (uint64_t)ldb, (uint64_t)ldc);
    npu_memcpy_from_dev(&runtime.dev_state, c, c_dev, n * ldc * sizeof(float)); // c
    return kernel_ret;
}

static void ggml_rnpu_matmul_e4m3q(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
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
    // Convert src1 to e4m3q
    float * b = (float *)(src1->data);
    std::vector<uint8_t> b_e4m3q(bs * h * m * k * (QK + sizeof(float)) / QK);
    for (int64_t i13 = 0; i13 < bs; ++i13) {
        for (int64_t i12 = 0; i12 < h; ++i12) {
            for (int64_t i11 = 0; i11 < n; ++i11) {
                float_to_e4m3q(b + (
                                i13 * src1->nb[3]
                              + i12 * src1->nb[2]
                              + i11 * src1->nb[1]
                            )/sizeof(float),
                            (bloc_fp8<4>*)(b_e4m3q.data() + (
                                i13 * (src1->nb[3] / sizeof(float)) * (QK + sizeof(float)) / QK
                              + i12 * (src1->nb[2] / sizeof(float)) * (QK + sizeof(float)) / QK
                              + i11 * (src1->nb[1] / sizeof(float)) * (QK + sizeof(float)) / QK
                            )/sizeof(uint8_t)),
                            k);
            }
        }
    }

    // Call Kernel
    gmma_e4m3q_fp32_AT_QK256(
        src0->data, b_e4m3q.data(), dst->data,
        m, n, k,
        src1->ne[0] * (QK + sizeof(float)) / QK, src0->ne[0] * (QK + sizeof(float)) / QK, dst->nb[1] / sizeof(float)
    );

    GGML_UNUSED(ctx);
}

void ggml_rnpu_mul_mat(ggml_backend_rnpu_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    switch (src0->type) {
        case GGML_TYPE_BF16:
            ggml_rnpu_matmul_bf16(ctx, dst);
            break;
        case GGML_TYPE_E4M3_Q:
            ggml_rnpu_matmul_e4m3q(ctx, dst);
            break;
        default:
            GGML_ABORT("FATAL ERROR: type %s not supported for op MATMUL in R-NPU\n",
                       ggml_type_name(src0->type));
    }
}
#pragma once

#include "./common.hpp"

#ifdef  __cplusplus
extern "C" {
#endif

/**
 * @brief Extracts specific rows from a tensor based on indices.
 *
 * @details This function retrieves rows from a source tensor src0 according to
 *          the indices provided in another tensor src1 and stores the result in
 *          a destination tensor. It supports different data types F32 and BF16.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor where the extracted rows will be stored.
 *            dst->op is `GGML_OP_GET_ROWS`.
 */
void ggml_rnpu_get_rows(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Writes source rows into destination rows according to index tensor.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor. dst->op is `GGML_OP_SET_ROWS`.
 */
void ggml_rnpu_set_rows(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Adds two tensors element-wise.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor where the element-wise add result
 *            will be stored. dst->op is `GGML_OP_Add`.
 */
void ggml_rnpu_op_add(ggml_backend_rnpu_context & ctx, ggml_tensor * dst);

/**
 * @brief Multiplies two tensors element-wise.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor where the element-wise multiplication result
 *            will be stored. dst->op is `GGML_OP_MUL`.
 */
void ggml_rnpu_op_mul(ggml_backend_rnpu_context & ctx, ggml_tensor * dst);

/**
 * @brief Applies Rotary Positional Embedding (RoPE) to the input tensor.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor where the RoPE-transformed data will be
 *            stored. dst->op is `GGML_OP_ROPE`.
 *
 * @note The function currently does not support cases where the n_dims is less
 *       than the input tensor's first dimension.
 * @note The function currently does not support cases where the freq_factors is
 *       not NULL.
 * @note The function currently does not support cases where the ext_factor is
 *       not equal 0.
 * @note The function currently does not support cases where the freq_scale is
 *       not equal 1.
 * @note The function currently only support cases where the input data type is FP32.
 * @note The function currently only support cases where the RoPE type is NEOX.
 */
void ggml_rnpu_rope(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Duplicates a ggml tensor using the R-NPU backend.
 *
 * @details This function duplicates the contents of the source tensor `src` to
 *          the destination tensor `dst`. The function supports various tensor
 *          types and configurations, including handling of extra data, type
 *          conversions, and special cases for contiguous and non-contiguous
 *          tensors.
 *
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_DUP`.
 *
 * @note Only support FP32 duplicates to BF16/FP32.
 * @note Not support when src is no-contiguous FP32 and dst is contiguous BF16.
 * @note Not support when src and dst are both FP32 and src or dst is no-contiguous.
 * @note Not support when src and dst have different shape.
 */
void ggml_rnpu_dup(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

// @see ggml_rnpu_dup.
void ggml_rnpu_cpy(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

// @see ggml_rnpu_dup.
void ggml_rnpu_cont(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Reshapes a ggml tensor using the R-NPU backend.
 *
 * This function reshapes a ggml tensor to a specified shape.
 *
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_RESHAPE`.
 * @attention This function does not have data movement operation. The real operation
 *            to change the shape is done after building the compute graph.
 */
void ggml_rnpu_reshape(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Views a ggml tensor using the R-NPU backend.
 *
 * This function views a ggml tensor to a specified shape.
 *
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_VIEW`.
 * @attention This function does not have data movement operation. The real operation
 *            to change the shape is done after building the compute graph.
 */
void ggml_rnpu_view(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Permutes a ggml tensor using the R-NPU backend.
 *
 * This function permute a ggml tensor shape to a specified shape.
 *
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_PERMUTE`.
 * @attention This function has real data movement operation to make the destination
 *            data conitguous for the following mul_mat.
 */
void ggml_rnpu_permute(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Transposes a ggml tensor using the R-NPU backend.
 *
 * This function reshapes a ggml tensor to a specified shape. This function
 * does not have data movement operation.
 *
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the duplicated data will be stored.
 *            dst->op is `GGML_OP_transpose`.
 */
void ggml_rnpu_transpose(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Computes the Root Mean Square (RMS) normalization of a ggml tensor
 *        using the R-NPU backend.
 *
 * @details This function applies RMS normalization to the input tensor `src`
 *          and stores the result in the destination tensor `dst`. RMS
 *          normalization involves computing the root mean square of the input
 *          tensor along a specified dimension and then dividing each element of
 *          the tensor by this value, adjusted by a small epsilon value to
 *          prevent division by zero.
 *          The operation is defined as:
 *          \f[
 *               \text{RmsNorm}\left(x_i\right)=\frac{x_i}{\text{Rms}(\mathbf{x})} g_i,
 *               \quad \text { where } \text{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+e p s}
 *          \f]
 *          `eps` is in dst->op_params.
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the normalized values will be stored.
 *            dst->op is `GGML_OP_RMS_NORM`.
 * @note Only support FP32 src and FP32 dst.
 * @note Only support do operation in the last dimension.
 */
void ggml_rnpu_rms_norm(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Computes the softmax activation with optional masking.
 *
 * @details This function computes the softmax activation over the input tensor,
 *          optionally applying a mask and scaling factor. It can handle masking
 *          by broadcasting the mask across rows if necessary.
 *          The function performs the following steps:
 *          1. Multiplies the input tensor by a scale factor.
 *          2. Broadcasts the mask tensor if its dimensions do not match the
 *             input tensor's dimensions.
 *          3. Adds the mask to the scaled input tensor.
 *          4. Applies the softmax activation function along the specified
 *             dimension.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor where the result will be stored. dst->op is
 *            `GGML_OP_SOFTMAX`.
 * @note Only support FP32 src and FP32 dst.
 */
void ggml_rnpu_softmax(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief Applies the SiLU (Sigmoid Linear Unit) activation to a ggml tensor
 *        using the R-NPU backend.
 *
 * @param ctx The R-NPU context used for operations.
 * @param dst The destination tensor where the SiLU-activated result will be stored.
 *            dst->op is expected to be `GGML_OP_SiLU`.
 * @note Only support FP32 src and FP32 dst.
 */
void ggml_rnpu_silu(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

/**
 * @brief   Executes matrix multiplication for the given tensor.
 *
 * @details This function performs matrix multiplication on the source tensors
 *          associated with the destination tensor. It supports matrix
 *          multiplication F32, F16, and E4M3_Q.
 *
 * @param ctx The backend R-NPU context for executing operations.
 * @param dst The destination tensor for storing the result of the matrix
 *            multiplication. dst->op is `GGML_OP_MUL_MAT`.
 * @note Only support when src0 is FP32, src1 is FP16, dst is FP32, the src1 and
 *       dst are required to be contiguous.
 * @note Only support when src0 and src1 are E4M3_Q, dst is FP32, the src0, src1
 *       and dst are required to be contiguous.
 */
void ggml_rnpu_mul_mat(ggml_backend_rnpu_context& ctx, ggml_tensor* dst);

#ifdef  __cplusplus
}
#endif

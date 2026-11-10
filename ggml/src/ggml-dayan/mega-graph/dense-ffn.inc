/*
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */
#if !defined(MEGA_MXFP4)
MEGA_STEP(ffn_copy, copy, copy, 1, required,
          pack(copy, at(ffn_copy)))
#endif
MEGA_STEP(ffn_norm, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(ffn_norm), true))
#if defined(MEGA_MXFP4)
MEGA_STEP(ffn_quant, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(ffn_quant)))
#ifdef MEGA_FFN_FUSED
MEGA_STEP(ffn, mul_mat_swiglu, ffn_decode, 1, required,
          pack(ffn, at(ffn), at(down), nullptr, nullptr))
MEGA_STEP(down, mul_mat, noop, 0, required,
          pack(noop, 0))
#else
MEGA_STEP(ffn, mul_mat_swiglu, matmul_mxfp4, 1, required,
          pack(matmul, at(ffn).nodes[0], at(ffn).nodes[1], at(ffn).nodes[3], 2))
MEGA_STEP(down, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(down).nodes[0], nullptr, at(down).nodes[0], 0))
#endif
MEGA_STEP(residual, norm_pipeline, residual_add, 0, borrow,
          pack(residual, at(residual), GGML_TYPE_F32))
#else
#ifdef MEGA_FFN_FUSED
MEGA_STEP(ffn, mul_mat_swiglu, ffn_decode, 1, required,
          pack(ffn, at(ffn), at(down), nullptr, nullptr))
MEGA_STEP(down, mul_mat_quantized, noop, 0, required,
          pack(noop, 0))
#else
MEGA_STEP(ffn, mul_mat_swiglu, matmul_swiglu_bf16, 1, required,
          pack(bf16_swiglu, at(ffn)))
MEGA_STEP(down, mul_mat_quantized, matmul_quantized_bf16, 1, required,
          pack(bf16_quantized, at(down)))
#endif
MEGA_STEP(residual, norm_pipeline, residual_add_bf16, 0, borrow,
          pack(residual, at(residual), GGML_TYPE_BF16))
#endif

#if !defined(MEGA_MXFP4)
MEGA_STEP(ffn_copy, copy, copy, 1, required,
          pack(copy, at(ffn_copy)))
#endif
MEGA_STEP(ffn_norm, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(ffn_norm), true))
#if defined(MEGA_MXFP4)
MEGA_STEP(ffn_quant, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(ffn_quant)))
MEGA_STEP(ffn, mul_mat_swiglu, matmul_mxfp4, 1, required,
          pack(matmul, at(ffn).nodes[0], at(ffn).nodes[1], at(ffn).nodes[3], 2))
MEGA_STEP(down, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(down).nodes[0], nullptr, at(down).nodes[0], 0))
MEGA_STEP(residual, norm_pipeline, residual_add, 0, borrow,
          pack(residual, at(residual), GGML_TYPE_F32))
#else
MEGA_STEP(ffn, mul_mat_swiglu, matmul_swiglu_bf16, 1, required,
          pack(bf16_swiglu, at(ffn)))
MEGA_STEP(down, mul_mat_quantized, matmul_quantized_bf16, 1, required,
          pack(bf16_quantized, at(down)))
MEGA_STEP(residual, norm_pipeline, residual_add_bf16, 0, borrow,
          pack(residual, at(residual), GGML_TYPE_BF16))
#endif

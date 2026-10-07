MEGA_STEP(norm0, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(norm0), false))
#if defined(MEGA_MXFP4)
MEGA_STEP(input_copy, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(input_copy)))
#endif
#if defined(MEGA_MXFP4)
MEGA_STEP(qg, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(qg).nodes[0], nullptr, at(qg).nodes[0], 0))
#else
MEGA_STEP(qg, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(qg).nodes[0], at(qg).nodes[0]))
#endif
MEGA_STEP(q_norm, rms_norm, rms_norm, 1, required,
          pack(rms_norm, at(q_norm)))
MEGA_STEP(q_scale, elementwise_mul, multiply, 1, required,
          pack(binary, at(q_scale)))
MEGA_STEP(q_rope, rope, rope, 1, required,
          pack(rope, at(q_rope)))
#if defined(MEGA_MXFP4)
MEGA_STEP(k, mul_mat_pair, matmul_mxfp4, 0, required,
          pack(matmul, at(k).nodes[0], nullptr, at(k).nodes[0], 0))
MEGA_STEP(v, mul_mat_pair, matmul_mxfp4, 1, reuse,
          pack(matmul, at(v).nodes[1], nullptr, at(v).nodes[1], 0))
#else
MEGA_STEP(kv, mul_mat_pair, matmul_pair_bf16, 1, required,
          pack(bf16_matmul_pair, at(kv)))
#endif
MEGA_STEP(k_norm, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(k_norm), false))
MEGA_STEP(k_rope, rope, rope, 1, required,
          pack(rope, at(k_rope)))
MEGA_STEP(set_k, set_rows, set_rows, 0, required,
          pack(rows, at(set_k), true))
MEGA_STEP(set_v, set_rows, set_rows, 1, required,
          pack(rows, at(set_v), true))
MEGA_STEP(q_copy, copy, copy, 1, required,
          pack(copy, at(q_copy)))
MEGA_STEP(attention, flash_attention, flash_attention_fp16, 1, required,
          pack(flash_attention, at(attention)))
#if defined(MEGA_MXFP4)
MEGA_STEP(cont, copy, copy, 1, required,
          pack(copy, at(cont)))
MEGA_STEP(gate_sigmoid, sigmoid, sigmoid, 1, required,
          pack(unary, at(gate_sigmoid)))
MEGA_STEP(gate_mul, elementwise_mul, multiply, 1, required,
          pack(binary, at(gate_mul)))
MEGA_STEP(output_copy, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(output_copy)))
MEGA_STEP(out, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(out).nodes[0], nullptr, at(out).nodes[0], 0))
#else
MEGA_STEP(gated, gated_output_projection, gated_output_projection_bf16, 1, required,
          pack(bf16_gated_projection, at(gated), at(q_copy).nodes[0]))
#endif

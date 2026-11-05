MEGA_STEP(norm0, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(norm0), false))
MEGA_STEP(scale0, scale, scale, 1, optional,
          ctx.missing(scale0) ? pack(noop_scale, at(rows0).nodes[0]) : pack(scale, at(scale0)))
MEGA_STEP(rows0, get_rows, get_rows, 1, required,
          pack(rows, at(rows0), false))
#if defined(MEGA_MXFP4)
MEGA_STEP(input_copy, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(input_copy)))
#else
MEGA_STEP(input_copy, copy, copy, 1, required,
          pack(copy, at(input_copy)))
#endif
#if defined(MEGA_MXFP4)
MEGA_STEP(qkv, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(qkv).nodes[0], nullptr, at(qkv).nodes[0], 0))
#else
MEGA_STEP(qkv, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(qkv).nodes[0], at(qkv).nodes[0]))
#endif
MEGA_STEP(scale1, scale, scale, 1, optional,
          ctx.missing(scale1) ? pack(noop_scale, at(rows1).nodes[0]) : pack(scale, at(scale1)))
MEGA_STEP(rows1, get_rows, get_rows, 1, required,
          pack(rows, at(rows1), false))
MEGA_STEP(conv, ssm_conv_silu, ssm_conv_split, 1, required,
          pack(ssm_conv_split, at(conv)))
MEGA_STEP(l2q, l2_norm, l2_norm, 0, required,
          pack(l2_norm, at(l2q)))
MEGA_STEP(l2k, l2_norm, l2_norm, 1, required,
          pack(l2_norm, at(l2k)))
#if defined(MEGA_MXFP4)
MEGA_STEP(alpha, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(alpha).nodes[0], nullptr, at(alpha).nodes[0], 0))
#else
MEGA_STEP(alpha, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(alpha).nodes[0], at(alpha).nodes[0]))
#endif
MEGA_STEP(alpha_add, elementwise_add, add, 1, required,
          pack(binary, at(alpha_add)))
MEGA_STEP(alpha_softplus, softplus, softplus, 1, required,
          pack(unary, at(alpha_softplus)))
MEGA_STEP(alpha_mul, elementwise_mul, multiply, 1, required,
          pack(binary, at(alpha_mul)))
#if defined(MEGA_MXFP4)
MEGA_STEP(beta, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(beta).nodes[0], nullptr, at(beta).nodes[0], 0))
#else
MEGA_STEP(beta, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(beta).nodes[0], at(beta).nodes[0]))
#endif
MEGA_STEP(beta_sigmoid, sigmoid, sigmoid, 1, required,
          pack(unary, at(beta_sigmoid)))
#if defined(MEGA_GDN_Z_GATE)
MEGA_STEP(gdn_z_gate, gated_delta_net, gated_delta_net_z_bf16, 1, fused,
          pack(gdn_z_gate, ctx.slots, ctx.indices[gdn_z_gate]))
#else
MEGA_STEP(gdn, gated_delta_net, gated_delta_net, 1, required,
          pack(gdn, at(gdn)))
MEGA_STEP(state_copy, copy, copy, 1, required,
          pack(copy, at(state_copy)))
MEGA_STEP(gdn_norm, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(gdn_norm), false))
#if defined(MEGA_MXFP4)
MEGA_STEP(z, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(z).nodes[0], nullptr, at(z).nodes[0], 0))
#else
MEGA_STEP(z, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(z).nodes[0], at(z).nodes[0]))
#endif
MEGA_STEP(z_silu, silu, silu, 1, required,
          pack(unary, at(z_silu)))
MEGA_STEP(gate_mul, elementwise_mul, multiply, 1, required,
          pack(binary, at(gate_mul)))
#endif
#if defined(MEGA_MXFP4)
MEGA_STEP(output_copy, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(output_copy)))
#else
MEGA_STEP(output_copy, copy, copy, 1, required,
          pack(copy, at(output_copy)))
#endif
#if defined(MEGA_MXFP4)
MEGA_STEP(out, mul_mat, matmul_mxfp4, 1, required,
          pack(matmul, at(out).nodes[0], nullptr, at(out).nodes[0], 0))
#else
MEGA_STEP(out, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(out).nodes[0], at(out).nodes[0]))
#endif

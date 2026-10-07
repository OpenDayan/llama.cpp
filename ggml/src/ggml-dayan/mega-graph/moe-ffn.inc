MEGA_STEP(ffn_norm, norm_pipeline, norm_pipeline, 1, required,
          pack(norm, at(ffn_norm), true))
MEGA_STEP(router, mul_mat, router_indexer, 1, required,
          pack(router_indexer, at(router), at(experts).nodes[0]->src[2], ctx.workspace, ctx.workspace_bytes))
MEGA_STEP(probs, softmax, softmax, 1, required,
          pack(routing, at(probs)))
MEGA_STEP(sort, argsort, noop, 2, required,
          at(sort).nodes[0] != nullptr && ggml_get_op_params_i32(at(sort).nodes[0], 0) == GGML_SORT_ORDER_DESC)
MEGA_STEP(weights, get_rows, get_rows, 1, required,
          pack(rows, at(weights), false))
MEGA_STEP(sum, sum_rows, sum_rows, 1, required,
          pack(routing, at(sum)))
MEGA_STEP(clamp_weights, clamp, clamp, 1, required,
          pack(routing, at(clamp_weights)))
MEGA_STEP(divide_weights, elementwise_div, divide, 1, required,
          pack(binary, at(divide_weights)))
MEGA_STEP(ffn_quant, copy, quantize_mxfp4, 1, required,
          pack(quantize, at(ffn_quant)))
MEGA_STEP(experts, mul_mat_id_swiglu, matmul_id_mxfp4, 1, required,
          pack(matmul, at(experts).nodes[0], at(experts).nodes[1], at(experts).nodes[3], 2, ctx.workspace, ctx.workspace_bytes))
MEGA_STEP(experts_down, mul_mat_id, matmul_id_mxfp4, 1, required,
          pack(matmul, at(experts_down).nodes[0], nullptr, at(experts_down).nodes[0], 0, ctx.workspace, ctx.workspace_bytes))
MEGA_STEP(weighted, elementwise_mul, multiply, 1, required,
          pack(binary, at(weighted)))
MEGA_STEP(expert_add0, elementwise_add, add, 1, required,
          pack(binary, at(expert_add0)))
MEGA_STEP(expert_add1, elementwise_add, add, 1, required,
          pack(binary, at(expert_add1)))
MEGA_STEP(expert_add2, elementwise_add, add, 1, required,
          pack(binary, at(expert_add2)))
MEGA_STEP(expert_add3, elementwise_add, add, 1, required,
          pack(binary, at(expert_add3)))
MEGA_STEP(expert_add4, elementwise_add, add, 1, required,
          pack(binary, at(expert_add4)))
MEGA_STEP(expert_add5, elementwise_add, add, 1, required,
          pack(binary, at(expert_add5)))
MEGA_STEP(expert_add6, elementwise_add, add, 1, required,
          pack(binary, at(expert_add6)))
MEGA_STEP(shared_copy, copy, copy, 1, required,
          pack(copy, at(shared_copy)))
MEGA_STEP(shared_ffn, mul_mat_swiglu, matmul_swiglu_bf16, 1, required,
          pack(bf16_swiglu, at(shared_ffn)))
MEGA_STEP(shared_down, mul_mat, matmul_bf16, 1, required,
          pack(bf16_matmul, at(shared_down).nodes[0], at(shared_down).nodes[0]))
MEGA_STEP(shared_router, mul_mat, matmul_tf32, 1, required,
          pack(tf32, at(shared_router)))
MEGA_STEP(shared_sigmoid, sigmoid, sigmoid, 1, required,
          pack(unary, at(shared_sigmoid)))
MEGA_STEP(shared_gate, elementwise_mul, multiply, 1, required,
          pack(binary, at(shared_gate)))
MEGA_STEP(combine, residual_add, add, 1, required,
          pack(binary, at(combine)))
MEGA_STEP(residual, norm_pipeline, residual_add, 0, borrow,
          pack(residual, at(residual), GGML_TYPE_F32))

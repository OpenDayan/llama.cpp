# Graph optimization

## Two separate passes

Dayan graph handling has two conceptually different passes.

`ggml_backend_dayan_graph_rewrite()` changes the GGML graph before scheduling. It inserts low-precision casts, moves eligible casts across metadata-only layout chains, rewires consumers, rebuilds the graph, and adds lifetime dependencies.

`ggml_backend_dayan_compile_graph()` runs inside Dayan graph execution. It does not create GGML tensors. It scans supported compute nodes and lowers them into `ggml_backend_dayan_graph_slot` records, including fused slots.

Do not mix their responsibilities. A type or dependency change belongs in rewrite. A launch grouping with unchanged externally visible tensors belongs in slot compilation.

## Support predicates are contracts

Every `ggml_backend_dayan_supports_*()` predicate must fully describe what the Host launcher and Device kernel accept. Check all of the following where applicable:

- Exact GGML op and source presence.
- Input and output types.
- Shape relationships and broadcast rules.
- Nonzero dimensions and narrowing limits for config fields.
- Overflow-safe byte spans.
- Contiguous, row-contiguous, or explicitly strided layout.
- View offsets and alias restrictions.
- Operation parameters such as epsilon, scale, mask, order, and mode.
- Kernel-specific alignment, tile, and tail restrictions.

`ggml_backend_dayan_device_supports_op()` must route the GGML op to the same predicate used by the compiler. A permissive device switch with a stricter compiler causes scheduler assignment followed by execution failure. A predicate that accepts more than the kernel supports is a correctness bug.

## Low-precision rewrite

MatMul weights are BF16 or FP16. An F32 activation is cast to the weight storage type. The rewrite deduplicates conversions using `(activation tensor, target type)`, so consumers sharing both source and target type share one cast.

If one activation feeds weights with conflicting storage types, it cannot use one shared hoist. Keep separate conversions at the safe locations.

FlashAttention requires Q, K, and V to share one low-precision type. The current target selection prefers an existing K storage type, then V, then Q, and otherwise BF16. Inputs not already in the target type receive shared casts.

The rewrite clears the cast's secondary source used by generic cast machinery so scheduler assignment follows the actual source. Preserve this behavior unless the scheduler contract changes.

## Cast hoisting through metadata

Hoisting a cast before a layout chain can remove a large F32 intermediate, but it is valid only when all of these are proven:

- Every traversed node is an eligible metadata-only layout operation.
- The chain has one exclusive path to the activation being replaced.
- No side consumer needs an F32 value that would disappear.
- Shape and view offset remain representable after element size changes.
- Every byte stride and offset is scaled consistently for the target type.
- Rebuilt aliases preserve the same logical indexing.

Read `ggml_backend_dayan_find_cast_hoist()`, `ggml_backend_dayan_layout_can_change_type()`, and `ggml_backend_dayan_apply_cast_hoist()` together. Do not add a metadata op to the allowlist based only on its name.

## Slot compilation

The compiler skips nodes without `GGML_TENSOR_FLAG_COMPUTE` and recognized metadata nodes. It then emits slots in graph order. A slot has a kind, up to five component node pointers, and a populated node count.

Current fusion families include:

- Two MatMuls with one activation.
- MatMul pair plus SWIGLU or GEGLU plus low-precision copy.
- Equivalent `MUL_MAT_ID` MoE patterns.
- MatMul plus low-precision output copy.
- Residual add plus RMSNorm plus scale plus optional low-precision output.
- SSM convolution plus SiLU.
- Attention output multiplied by sigmoid gate plus output projection.

Some nodes can appear adjacent after metadata nodes are skipped. Adjacency is necessary for the current scanner but is never sufficient for fusion.

The current fused component positions are:

| Slot kind | `nodes[]` contract |
| --- | --- |
| `mul_mat_pair` | `[0]` first MatMul, `[1]` second MatMul |
| `mul_mat_quantized` | `[0]` MatMul, `[1]` low-precision copy |
| `mul_mat_swiglu` / `mul_mat_geglu` | `[0]` and `[1]` MatMuls in GLU operand order, `[2]` GLU, `[3]` low-precision copy |
| `mul_mat_id_pair` | `[0]` first expert MatMul, `[1]` second expert MatMul |
| `mul_mat_id_swiglu` / `mul_mat_id_geglu` | `[0]` and `[1]` expert MatMuls in GLU operand order, `[2]` GLU, `[3]` low-precision copy |
| `norm_pipeline` | `[0]` optional residual add, `[1]` RMSNorm, `[2]` scale multiply, `[3]` optional low-precision copy |
| `gated_output_projection` | `[0]` gate sigmoid, `[1]` attention/gate multiply, `[2]` output projection, `[3]` backing gate source needed by the launcher |
| `ssm_conv_silu` | `[0]` SSM convolution, `[1]` SiLU |

This table is a current implementation note. Recheck compiler and Host launch helpers before relying on it.

## Fusion proof checklist

Before eliminating an intermediate, prove:

1. Each component has the expected op, type, shape, and layout.
2. Producer and consumer pointers match the intended chain.
3. Paired MatMuls use the same activation and compatible weight type.
4. Operand order is handled when a commutative op can present either input first.
5. Every eliminated producer and intermediate has exactly the expected number of uses.
6. No output flag, graph output, side consumer, or callback requires the intermediate.
7. All early inputs remain live through the final logical node.
8. The fused Device kernel reproduces each externally visible output that remains in the graph.

`ggml_backend_dayan_can_elide_glu_intermediates()` and the negative fan-out case in `test-dayan-graph` are the model for use-count protection. If a tensor has another consumer, fall back to unfused slots or reject the unsupported pattern. Never silently discard the side consumer.

## Adding or changing a slot kind

Update all of these together:

- `ggml_backend_dayan_slot_kind` in `ggml-dayan-graph.h`.
- Component placement and `node_count` in `ggml_backend_dayan_compile_graph()`.
- Slot output selection in `ggml_backend_dayan_slot_outputs()`.
- Human-readable kind name in `ggml_backend_dayan_slot_kind_name()`.
- Host dispatch in `ggml_backend_dayan_graph_compute()`.
- Slot counters and lowering log if the family is reported.
- Trace signature expectations if component meaning changes.
- Focused numerical and negative fan-out tests.

Slot component positions form an internal ABI between compiler, dispatcher, tracer, and tests. Document the meaning near the Host launch helper when it is not obvious.

## Failure behavior

If any non-metadata node is unsupported, `ggml_backend_dayan_compile_graph()` returns the node through `unsupported` and graph execution fails. This is intentional for the all-Dayan Qwen3.5 path. Avoid a hidden CPU fallback because it changes buffer ownership and can conceal a missing Device contract.

## Review questions

- Can the same optimization be expressed with an existing slot and a config flag?
- Is graph mutation required, or can slot compilation group the existing nodes?
- Does the support predicate cover decode views and nonzero offsets, not only contiguous prefill tensors?
- Is the fusion based on capabilities and tensor relationships rather than a model name?
- Does the change preserve graph outputs and side consumers?
- Does trace output still identify the correct final tensor or tensors?

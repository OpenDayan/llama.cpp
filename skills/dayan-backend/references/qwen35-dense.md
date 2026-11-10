# Qwen3.5 dense graph knowledge

## Why the model graph matters

Dayan optimizations should match GGML capabilities and tensor relationships, but Qwen3.5 defines the concrete production patterns that exercise them. Read `src/models/qwen35.cpp` before changing a fusion, attention layout, recurrent state path, or low-precision boundary.

The main graph alternates recurrent Gated Delta Net layers and full-attention layers according to model hyperparameters. Each layer has attention RMSNorm, attention or recurrent computation, a residual add, post-attention RMSNorm, dense gated FFN, and a second residual add. The model output has RMSNorm followed by the output projection.

Do not hardcode a layer number in the backend. Match operator capabilities and tensor structure.

## Full-attention layer

`build_layer_attn()` creates this conceptual path:

```text
input
  -> joint Q and gate projection
     -> Q view -> RMSNorm -> MRoPE --------------------+
     -> gate view -> CONT -> sigmoid --------+         |
  -> K projection -> reshape -> RMSNorm -> MRoPE       |
  -> V projection -> reshape                           |
  -> KV cache update and FlashAttention <--------------+
  -> attention * sigmoid(gate)
  -> output projection
```

Important layout facts:

- The Q projection stores Q and gate interleaved in the projected row layout.
- Q is a strided view of the first half of each head record.
- Gate is a strided view of the second half and is made contiguous before sigmoid.
- Q and K receive per-head RMSNorm and MRoPE.
- V is reshaped but not normalized or rotated.
- FlashAttention output is gated before the output projection.

Dayan has dedicated projection kernels that can fuse MatMul output conversion with Q/K normalization, RoPE, and layout conversion. K/V cache variants write indexed cache rows directly. The gated output projection slot fuses attention, sigmoid gate, low-precision conversion, and the output MatMul.

When changing one stage, verify both the standalone GGML node support and the fused production chain. A view stride or gate offset error can pass a contiguous synthetic test and fail the model.

## Recurrent layer

`build_layer_attn_linear()` creates this conceptual path:

```text
input
  -> mixed QKV projection -----------------> recurrent convolution state -> SSM_CONV -> SiLU -> Q/K/V views
  -> z projection ------------------------------------------------------------------------------+
  -> beta projection -> sigmoid ----------------------------------------------------------------|
  -> alpha projection -> bias -> softplus -> multiply by A -------------------------------------|
                                                                                                 v
Q/K -> L2 norm -------------------------------------------------------------> GATED_DELTA_NET -> RMSNorm
                                                                                                 |
z reshape -> SiLU ---------------------------------------------------------------------------> multiply
                                                                                                 |
                                                                                         output projection
```

Important facts:

- Convolution output is one packed F32 tensor. Q, K, and V are views with explicit offsets and shared outer strides.
- Q and K receive L2 normalization before Gated Delta Net.
- Head mapping can require logical repetition, but fused Gated Delta Net can consume the original grouped-head strides directly.
- The recurrent state tensor is part of operator correctness and must remain Device-resident.
- The Gated Delta Net output contains the attention result and state snapshots according to the GGML op contract.
- Final recurrent gating is RMSNorm output multiplied by `SiLU(z)`, followed by output projection.

Current Dayan slot fusions include SSM convolution plus SiLU and the normalization/gate/output-projection pattern where supported. Keep convolution and the Gated Delta Net core as separate semantic operators unless a design proves state, lifetime, and test coverage for a larger fusion.

## Dense FFN

Qwen3.5 dense uses parallel gate and up projections, SWIGLU, low-precision conversion for the next MatMul, and the down projection. Dayan can pair the two MatMuls and fuse the GLU epilogue when gate, up, GLU, and cast intermediates have exclusive consumers.

The production Qwen3.5 0.8B shape used by focused tests includes `M=8`, `N=3584`, and `K=1024`. Do not optimize only this shape; keep the support predicate aligned with all shapes the kernel truly handles.

## Low-precision boundaries

Weights are BF16 or FP16. GGML MatMul normally produces F32. Dayan graph rewrite inserts storage-type activation casts and can fuse an F32 epilogue plus low-precision writeback when the next Device MatMul consumes that value.

CPU reference tests intentionally reproduce the same low-precision boundaries. Compare against that reference rather than an all-F32 mathematical graph when evaluating backend error.

## Model-level no-fallback rule

When the model uses Dayan, every non-metadata model graph node must be assigned to Dayan. This includes small layout, activation, normalization, state, and copy nodes around the expensive kernels. Sampling stays outside the model graph on CPU.

After a backend change, a successful isolated kernel test is not sufficient. Confirm scheduler assignment and run the dense model test. A newly unsupported `CONT`, `CPY`, view-derived stride, or recurrent state op can invalidate the whole graph.

## Source anchors

- Model graph: `llama_model_qwen35::graph::graph()`.
- Full attention: `llama_model_qwen35::graph::build_layer_attn()`.
- Recurrent attention: `llama_model_qwen35::graph::build_layer_attn_linear()`.
- Dense FFN: `llama_model_qwen35::graph::build_layer_ffn()`.
- Attention projection Device helpers: `ggml-dayan-attention.h`.
- Recurrent convolution: `ggml-dayan-conv.h`.
- Gated Delta Net: `ggml-dayan-gated-delta-net.h`.
- Model-level reference: `tests/test-dayan-qwen35-dense.cpp`.

# FlashAttention

## Framework contract

The Dayan node is `GGML_OP_FLASH_ATTN_EXT` with:

- `src[0]`: Q.
- `src[1]`: K.
- `src[2]`: V.
- `src[3]`: optional F16 mask.
- `src[4]`: unsupported attention sinks and therefore required to be null.
- Output: F32.

The logical input layout is `[head_dim, sequence, heads, batch]` in GGML `ne[]`. Rows along `head_dim` must be contiguous, but row, head, and batch strides are carried separately because Q/K/V may be permuted views or KV-cache views.

The output layout is `[head_dim, q_heads, q_len, batch]`. It is contiguous in the current support contract.

Current supported limits:

- Q, K, and V share BF16 or FP16 storage.
- `head_dim` is a nonzero multiple of 64 and no more than 256.
- K and V have identical sequence, head, and batch extents.
- Q and K have the same head dimension and batch count.
- GQA is supported when `q_heads` is an integer multiple of `kv_heads`.
- Scale is finite and nonnegative.
- ALiBi bias is zero.
- Logit soft cap is zero.
- Attention sinks are absent.

Read `ggml_backend_dayan_supports_flash_attention()` before changing this list.

## Type rewrite

`ggml_backend_dayan_graph_rewrite()` makes Q, K, and V share one low-precision type. It prefers an already low-precision K type, then V, then Q, and otherwise BF16. A conversion is shared with any other consumer using the same `(tensor, target type)` key.

The Device kernel has separate BF16 and FP16 entries but shares one templated implementation. MAT data type and probability conversion must follow the selected format through both QK and PV. Do not silently route FP16 through a BF16 intermediate.

## Causal mask reduction

The Device kernel does not consume the GGML mask tensor. The Host scans a supported mask and reduces it to:

- `causal_offset`: the absolute key position corresponding to query row zero.
- `kv_len`: the effective key prefix required by the query tile.
- A causal flag.

For every query row, valid keys must be one zero-valued prefix followed only by masked values. The valid prefix length must equal `causal_offset + query + 1`. Accepted masked values are negative infinity or a finite value no greater than the current large-negative threshold.

This is a right-aligned causal mask. Arbitrary sparse, additive, sliding-window, or biased masks are not supported by this reduction.

Physical K/V storage can be longer than effective `kv_len`, especially during decode. Keep the original physical row/head/batch strides while traversing only the effective prefix. Never replace a physical cache length with an effective length when computing outer addresses.

## Current tiled algorithm

The current implementation uses:

- Q row tile `Br = 64`.
- KV row tile `Bc = 64`.
- One K/V producer.
- Three fixed Q-tile consumers.
- Two K/V URAM buffers.
- Independent LSU and MAT groups per consumer.
- F32 online-softmax state `m`, `l`, and `O`.

For each Q tile and KV tile, a consumer performs:

1. Load Q and initialize `m = -FLT_MAX`, `l = 0`, and `O = 0`.
2. Compute QK scores with MAT in F32 accumulation.
3. Prepare the causal threshold for the tile.
4. Update online softmax in VEC.
5. Convert probabilities to the input low-precision format.
6. Compute PV with MAT.
7. Release K/V after PV no longer reads it.
8. Rescale and update F32 `O`.
9. Normalize `O` by `l` and store F32 output.

Scores and probabilities are URAM tiles. They are not materialized in DRAM.

## Online softmax invariant

For old state `(m_old, l_old, O_old)` and a new score tile:

```text
m_new = max(m_old, row_max(scores))
alpha = exp(m_old - m_new)
P     = exp(scores - m_new)
l_new = alpha * l_old + row_sum(P)
O_new = alpha * O_old + P * V
```

Masked values must not change row max or sum. A partial final KV tile and a partially valid causal tile require exact masking before exponentiation. Keep `m`, `l`, and `O` in F32 even when Q/K/V and P are low precision.

## Producer and consumer invariant

The producer walks a monotonic generation sequence over batch, query head, Q super-group, and KV tile. Each double-buffered K/V object records a full generation, not only parity.

For a generation:

- Producer count is one.
- Consumer count is the number of Q slices that actually use that KV tile.
- The buffer remains READY until every such consumer completes PV.
- A fast consumer must not mistake an older READY parity slot for its expected future generation.
- Every fixed consumer advances its local generation base across the producer's complete range, including causal suffix tiles it did not consume and an inactive final Q slice.

These rules prevent stale K/V use and double-buffer overwrite. Preserve them even if the current scheduler classes are replaced.

## URAM structure

Each consumer owns disjoint Q, score, P, PV, O, row-state, constants, and mask scratch regions. K and transposed V are shared through two producer buffers. Layout sizes use maximum `head_dim`, not the runtime dimension.

When changing a tile or consumer count, recalculate:

- Per-consumer context bytes.
- K and transposed-V buffer sizes.
- TCSR group allocation.
- VEC in-flight block count and completion FIFO pressure.
- Descriptor URAM bytes.
- Producer consumer-count logic.

A tile-size change is not local to one loop.

## Dependency cleanup direction

The current file expresses the pipeline through the firmware public proc and resource API:

1. Keep contract validation independent of scheduling.
2. Keep a named compile-time URAM layout.
3. Keep issue helpers for load/transpose, QK, mask, softmax, P conversion, PV, O update, and store.
4. Express producer/consumer generations with public Dayan proc/resource APIs.
5. Keep traversal state explicit and monotonic.

Do not copy private scheduler types or instruction encodings into a new local compatibility layer.

## Required tests after a change

- `test-dayan-flash-attention` for GQA, right-aligned causal prefill, permuted cache layout, physical/effective KV separation, and one-token decode.
- `test-dayan-attention-projection` for Q/K/V projection, cache writes, FlashAttention, gate, and output projection in BF16 and FP16.
- `test-dayan-qwen35-dense` for the four-layer hybrid graph and final logits.
- Decode trace replay with `--check` for signature/order stability and exact cold/warm hashes.

Add a new test only when the changed contract is not exercised by these cases.

## Source anchors

- Host support and mask reduction: `ggml_backend_dayan_supports_flash_attention()` and `ggml_backend_dayan_flash_attention_lengths()`.
- Host config and launch: `ggml_backend_dayan_flash_attention()`.
- Config ABI: `DayanFlashAttentionConfig`.
- Kernel entries: `flash_attention_bf16_1c()` and `flash_attention_fp16_1c()`.
- Device implementation: `ggml-dayan-flash-attention.h`.

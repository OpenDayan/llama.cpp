# Quick Dayan context

This page is the minimum context for most Dayan work.

## End-to-end path

```text
Qwen3.5 model builder
  -> GGML graph
  -> backend graph rewrite
  -> GGML scheduler assignment and arena allocation
  -> Dayan slot compiler
  -> Host slot dispatch and config upload
  -> dycc-generated wrapper
  -> Device kernel
```

The important boundaries are:

- `src/models/qwen35.cpp` defines the model graph and therefore the real operator patterns, tensor views, and residual paths.
- `ggml_backend_dayan_graph_rewrite()` mutates graph types and dependencies before scheduler allocation.
- `ggml_backend_dayan_device_supports_op()` decides whether a node can be assigned to Dayan.
- `ggml_backend_dayan_compile_graph()` lowers supported compute nodes to fused slots.
- `ggml_backend_dayan_graph_compute()` dispatches each slot.
- `ggml-dayan-kernels.h` is the Host/Device configuration ABI.
- `ggml-dayan-kernels.cpp` contains external kernel entries and resource descriptors.
- Operator headers contain Device implementations.

## Tensor facts

- `ne[]` is the logical extent and `nb[]` is the physical byte stride.
- Do not infer contiguity from shape. Check the exact `ggml_is_contiguous*()` predicate required by the operator.
- A metadata op can change shape, strides, or view offset without executing a kernel.
- Dayan stores one Device allocation per GGML arena and creates an `npu_buffer_t` view for each tensor. The view is stored in `tensor->extra`.
- Passing only `ggml_nbytes()` is insufficient for a strided view. A kernel must receive and honor every stride that it uses.
- Strides, DMA lengths, offsets, and MAT row strides are bytes unless an API explicitly states otherwise.
- Check products and address spans before multiplication. Shapes and strides can overflow a narrower config field or address calculation.

## Graph optimization facts

- A support predicate is part of correctness. It must reject every type, shape, stride, mask, parameter, and alias pattern the kernel cannot execute.
- The rewrite inserts shared BF16 or FP16 activation casts for MatMul and FlashAttention. Cast identity is `(source tensor, target type)`.
- A cast can move across `VIEW`, `RESHAPE`, `PERMUTE`, or `TRANSPOSE` only when the metadata chain has compatible type-scaled strides and no side consumer that needs the old value.
- Fusion is valid only when all tensor relationships match and every eliminated intermediate has no consumer outside the fused chain.
- A fused kernel may read an early input at the time of the final logical node. Add dependencies to the final node so GGML's allocator keeps those inputs live.
- Metadata nodes are skipped by the slot compiler, but a `CONT` that performs a real copy is not metadata.
- Adding a slot kind requires synchronized changes to the slot enum, compiler, Host dispatch, output reporting, counters if applicable, and tests.
- Qwen3.5 Dayan execution rejects CPU fallback for non-metadata nodes. Supporting an isolated op is not enough if another node in the model pattern remains unsupported.

## Host execution facts

- The backend shares one Runtime, Context, and Stream through a reference-counted device state.
- Backend capabilities currently report synchronous graph execution and no backend events.
- Config, performance, and workspace buffers are cached by slot index and grown only when needed.
- Config upload is skipped when the bytes are unchanged.
- Temporary converted activations are cached per graph execution by `(tensor, storage type)`.
- Generated wrappers are the kernel launch API. Do not load modules manually in an operator path.
- Preserve the first negative status while still releasing owned buffers and events.

## Device kernel facts

- Put stable Host/Device POD configs in `ggml-dayan-kernels.h`.
- External `__kernel__` entries live in `ggml-dayan-kernels.cpp`; reusable Device functions live in focused headers.
- Every entry needs exactly one matching `DAYAN_KERNEL_DESC` or `DAYAN_KERNEL_TASK_DESC`.
- Prove URAM layout at compile time and validate `dayan_uram_size()` at runtime.
- Never encode `count - 1` into LSU or MAT configuration until `count > 0` is proven.
- Wait for LSU, MAT, VEC, or MACRO completion before reading, reusing, publishing, or returning from the affected buffer.
- Resource producer and consumer counts must equal the actual users for every generation.
- Use `dayan_resource_t` only within one Hart/proc family. Use `dayan_shared_resource_t` and a cluster barrier for cross-Core communication.
- The public firmware Device API is authoritative. The Dayan backend does not depend on `npu-kernels` headers.

## FlashAttention facts

- Dayan FlashAttention consumes Q, K, and V in one shared BF16 or FP16 storage type and produces F32.
- It supports head dimensions that are nonzero multiples of 64 up to 256 and GQA where `q_heads % kv_heads == 0`.
- It supports no ALiBi bias, attention sinks, or logit soft cap in the current implementation.
- A supported causal mask is a right-aligned prefix mask with one constant `causal_offset` across query rows.
- Physical KV cache length and effective causal KV length are different. Keep physical strides while limiting traversal to the effective prefix.
- Online softmax state `m`, `l`, and `O` remains F32. Scores are tiled and not materialized in DRAM.
- The current pipeline uses 64-row Q tiles, 64-row KV tiles, one KV producer, three fixed consumers, and double-buffered K/V in URAM.

## Validation facts

- Use an existing focused CPU-reference test first.
- Run `test-dayan-qwen35-dense` after changing a graph pattern or a Qwen3.5 operator boundary.
- Run FlashAttention layout, GQA, prefill, and decode cases after attention changes.
- Use `llama-dayan-trace --replay ... --check` to compare fused signature order/counts and exact cold/warm output hashes.
- Trace replay skips duplicate execution and can feed synthetic or stale values to later unique slots. It is a shape/performance replay, not a CPU numerical reference.
- Compare the warm unique replay and per-signature `latency_times_count_us` under the same binary, firmware, model, trace, Hart count, and memory size.

## Stop conditions

Stop and read more source when any of these is true:

- The proposed fusion removes a tensor with an uncounted side consumer.
- A view or permute is present and the kernel config does not carry its strides.
- Prefill passes but decode changes physical and effective cache lengths.
- A Device buffer is reused before an explicit unit wait and resource transition.
- A new kernel needs instruction encodings not exposed by the public firmware API.
- The focused test has no independent reference for the behavior being changed.

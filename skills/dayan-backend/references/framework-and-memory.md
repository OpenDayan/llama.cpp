# Framework and memory model

## Where the backend rewrite runs

`llama_model::build_graph()` builds the architecture graph, sets graph outputs, and then calls `llama_model_apply_backend_graph_rewrites()`. That helper asks every scheduled backend registry for `ggml_backend_graph_rewrite` and calls it before scheduler allocation.

Dayan exports `ggml_backend_dayan_graph_rewrite()` through `ggml_backend_dayan_reg_get_proc_address()`. A rewrite therefore operates on a complete GGML graph but before its final scheduler arena layout. This timing permits inserting casts and dependencies, but the rewritten graph must still satisfy scheduler support and allocation rules.

For a Dayan model, `llama_graph_assign_dayan()` assigns each supported non-metadata node to Dayan. `llama_graph_is_dayan_only()` then requires every non-metadata compute node to remain on Dayan during reserve and execution. FlashAttention and fused Gated Delta Net probes also treat fallback from a Dayan layer as an error.

Consequence: a backend change is complete only when the whole model graph remains supported. A locally correct operator can still break model execution by changing a producer type, view layout, or downstream support predicate.

## GGML tensor representation

A `ggml_tensor` combines logical and physical information:

- `type` defines element storage.
- `ne[0..3]` defines logical extents.
- `nb[0..3]` defines byte strides.
- `view_src` and `view_offs` describe aliasing into another tensor.
- `src[]` defines graph dependencies, not ownership.
- `data` is the address used by GGML allocation and view arithmetic.
- `extra` is backend-private state. Dayan stores the tensor's `npu_buffer_t` view there.

Metadata operations such as `VIEW`, `RESHAPE`, `PERMUTE`, and `TRANSPOSE` can be zero-work graph nodes while producing a layout that a Device operator must honor. `CONT`, `CPY`, `DUP`, and related operations can require real data movement.

Never summarize a tensor as only a shape. Record type, extents, byte strides, offset, contiguity class, and aliasing.

## Dayan arena mapping

`ggml_backend_dayan_buffer_type_alloc()` creates:

1. A Host virtual mapping used as the GGML arena base.
2. One Device `npu_buffer_t` allocation for the complete arena.
3. A buffer context that owns both and tracks tensor views.

`ggml_backend_dayan_buffer_init_tensor()` computes the tensor offset from the arena base, creates an `npu_buffer_t` view into the Device allocation, and stores the view in `tensor->extra`.

This has several consequences:

- Two GGML views can refer to overlapping regions of one Device allocation.
- A kernel argument must use the tensor view from `tensor->extra`, not reinterpret `tensor->data` as a Device address.
- Host pointer range lookup is used to recognize copies whose source is already in a Dayan arena and turn them into Device-to-Device copies.
- Arena lifetime owns all tensor Device views. A tensor does not destroy its own view.

`set_tensor`, `get_tensor`, memset, and Device-to-Device copy use Runtime copy APIs and wait for their events. Non-weight Host arena bytes are kept coherent where the current backend needs them, but Device data is authoritative for kernel chaining.

## Backend ownership

The current backend has one reference-counted `ggml_backend_dayan_device_state` containing Runtime, Context, and Stream. Backend contexts and buffer contexts acquire this shared state and release it when destroyed.

The graph executor owns per-slot resources:

- A config buffer and a Host byte copy used to skip unchanged uploads.
- An optional performance/result buffer.
- An optional reusable workspace buffer.

Resources are indexed by compiled slot order. They grow when capacity is insufficient and are reused across graph executions. A shape change is safe only because each launch rebuilds the Host config and compares its bytes before reuse.

Temporary low-precision activations are owned for one `graph_compute()` call. `ggml_backend_dayan_prepare_activation()` caches them by `(source tensor, target type)` so MatMuls sharing an activation share one conversion. An activation already stored in the requested type uses its tensor Device view directly.

## Synchronization boundary

The backend currently advertises no asynchronous backend capability and no backend Event interface. Runtime operations may use asynchronous copy APIs internally, but the helper waits and destroys the returned Event before proceeding. Generated kernel wrappers are used synchronously by the graph executor.

Do not infer that Runtime objects are disposable per operator. The framework owns the shared Runtime, Context, Stream, tensor buffers, and graph sequencing. An operator owns only its cached config/workspace/perf resources and temporary buffers.

## Lifetime and fusion

GGML's allocator uses graph dependencies to decide when a buffer can be reused. If a fused slot executes several logical nodes at once, it may read a source later than the original graph implied.

The normalization pipeline demonstrates the required fix: `ggml_backend_dayan_preserve_norm_pipeline_inputs()` attaches the residual inputs, normalization input, and scale tensor as extra sources of the final fused logical operation. This keeps them live until the fused slot executes.

Rules:

- Do not rely on C++ pointers in the slot plan to extend allocator lifetime.
- Attach every early source read by a later fused slot to the final logical node.
- Do not overwrite an existing semantic source. Use a free `src[]` position and fail rewrite if none exists.
- Rebuild the graph after inserting nodes or replacing dependencies so topological order and reachability are recomputed.
- Add a test that would fail when the allocator aliases a still-live input, not only a numerical happy path with independent allocations.

## Source anchors

- `src/llama-model.cpp`: `llama_model_apply_backend_graph_rewrites()` and `llama_model::build_graph()`.
- `src/llama-context.cpp`: `llama_graph_assign_dayan()` and `llama_graph_is_dayan_only()`.
- `ggml/src/ggml-dayan/ggml-dayan.cpp`: device state, buffer type, tensor view initialization, slot resources, and graph compute.
- `ggml/src/ggml-dayan/ggml-dayan-graph.cpp`: rewrite and lifetime dependency insertion.

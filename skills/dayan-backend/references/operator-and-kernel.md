# Operator and kernel implementation

## File roles

- `ggml-dayan-kernels.h` contains POD configuration structs, flags, type codes, Host wrapper declarations, and constants shared by Host and Device.
- `ggml-dayan-kernels.cpp` contains external non-template `__kernel__` entries and their resource descriptors.
- Focused headers such as `ggml-dayan-matmul.h`, `ggml-dayan-memory.h`, and `ggml-dayan-normalization.h` contain Device implementations.
- `ggml-dayan.cpp` converts GGML tensor metadata into configs, obtains Device views, caches launch resources, selects the typed wrapper, and reports status.

Keep this separation. Do not put GGML framework logic in Device headers or Runtime ownership in a kernel.

## Define the tensor contract first

For each argument, record:

- Semantic axes and how they map to `ne[]`.
- Storage type and output accumulation type.
- Byte strides used by the kernel.
- Required alignment and contiguous dimensions.
- Supported empty dimensions, tails, broadcasting, and batches.
- Whether input/output ranges can overlap.
- Numerical tolerance and exceptional-value policy.

Build the Host support predicate from this contract. Build the Host config from the same fields. Validate the same contract on Device before issuing work.

## Host configuration ABI

Configs cross the Host/Device boundary as Device buffers. Keep them standard-layout POD values with fixed-width integer fields. Store floating parameters as their exact 32-bit bit patterns when the config must remain unambiguous across compilation targets.

Use byte units for all strides and offsets. Validate before narrowing an `int64_t` GGML dimension to `uint32_t`. Check size multiplication before calculating allocation or address ranges.

The Host launcher should:

1. Reconfirm the tensor relationship assumed by its support predicate.
2. Fill every config field, including outer strides for views.
3. Obtain `npu_buffer_t` arguments through `tensor->extra`.
4. Reuse per-slot config, perf, and workspace buffers.
5. Initialize launch attributes with the framework Context and Stream.
6. Select task dimensions and mode that match the descriptor and Device validation.
7. Call the generated wrapper and propagate its status.
8. Release only temporary resources it owns.

## dycc and kernel entries

An external entry is a named, non-template `__kernel__ void` with no more than eight machine-word arguments. Structured configuration is passed by pointer. dycc turns Device pointer parameters into typed Host wrapper buffer arguments and appends launch attributes.

Every entry needs one descriptor:

```cpp
DAYAN_KERNEL_DESC(kernel, max_child_procs, shared_bytes, uram_bytes);
DAYAN_KERNEL_TASK_DESC(kernel, max_tasks_per_core, max_child_procs, shared_bytes, uram_bytes);
```

Declare the maximum simultaneously live child procs, not the total ever created. URAM size is rounded to the firmware allocation granule. Coherent shared bytes include barriers and cross-Core resources.

## Public Device API boundary

Treat `${DAYAN_ROOT}/device/include/api` as the source of truth. Use `dayan_sys.h` for task, proc, resource, URAM, wait, and barrier APIs. Use `dayan_intrinsics.h` for LSU, MAT, VEC, and related instruction builders.

The Dayan backend uses only this public Device API for instruction and scheduler support. Do not add `npu-kernels` headers, copy instruction encodings into llama.cpp, or depend on private scheduler types.

## Device validation

Validate before the first instruction issue:

- Non-null required pointers.
- Nonzero dimensions used as `count - 1`.
- Type and flag enum ranges.
- Shape and stride relationships.
- Address span overflow and forbidden overlap.
- Task mode, task dimensions, and task ID.
- URAM base, runtime URAM size, and static layout bound.
- Alignment required by an instruction path.

Return a negative errno-like Device status with `dayan_proc_exit()` for invalid arguments or resource failure.

`DAYAN_PARAMS_IF` checks are currently controlled by `ENABLE_PARAMS_CHECK` and can compile out. This makes precise Host predicates essential, but new code should still keep Device validation structurally correct. Do not use disabled checks as permission to issue undefined hardware work.

## URAM layout

Use named compile-time offsets and a final `static_assert(total <= declared_uram)`. Include:

- Alignment padding.
- Double-buffer copies.
- Maximum tile dimensions, not only the default shape.
- Separate regions for simultaneous LSU, MAT, and VEC work.
- Per-task offsets when multiple logical tasks share the allocation contract.

Do not reuse a region until both conditions hold:

1. The hardware unit that reads or writes it has completed.
2. Every resource consumer for that generation has released it.

## LSU, MAT, and VEC

LSU addresses and strides are bytes. `d2u`, `u2d`, `u2u`, and `d2d` name source-to-destination direction while intrinsic arguments are destination first. A final partial transaction must use the exact valid byte count.

Describe each continuous or fixed-stride region with the largest LSU transaction that fits the three hardware dimensions. Fold contiguous dimensions, keep repeated source dimensions at zero input stride, and loop in Device code only for dimensions or indexed addresses that LSU cannot encode.

MAT dimensions are commonly encoded as `real_dimension - 1`. Update M, N, and K for edge tiles. Use MM for the first logical K tile and MMA for later accumulation tiles. Preserve the padded URAM row layout expected by MAT.

VEC tensor dimensions often operate on 64 lanes. Pad only the URAM scratch representation; limit the final LSU store to valid elements. A broad synchronization can destroy intended overlap, so wait on the narrowest unit boundary that proves the next use is safe.

## Task and resource design

Use logical task APIs, not physical Core IDs. Prefer static partitioning when shapes are known.

Resource lifecycle is:

```text
init -> wait FREE -> allocate(producers, consumers) -> issue and wait producer work
     -> produce READY -> wait READY -> consume data -> consume/release
```

The counts must match actual users for every generation, including inactive tail workers that still advance a shared generation sequence. Add a generation tag when double-buffer parity alone cannot distinguish an old READY buffer from a future generation.

Use `dayan_resource_t` for one Hart/proc family. Use cache-line-separated `dayan_shared_resource_t` in coherent shared storage for cross-Core state. Use `dayan_cluster_barrier_t` for publication and rendezvous across a four-Core group.

## Optimization priorities

Prefer these changes in order:

1. Remove Host round trips and redundant Device copies.
2. Use correct strided Device views directly.
3. Fuse an epilogue when it removes a materialized intermediate and has a proven exclusive consumer chain.
4. Replace scalar Device loops with LSU, MAT, or VEC only when the hardware operation matches the layout.
5. Add overlap and multiple workers after the sequential path is numerically correct.

An elegant kernel has a small explicit contract, a compile-time memory layout, monotonic ownership transitions, separate issue helpers, and one readable traversal. It does not hide correctness in scheduler side effects.

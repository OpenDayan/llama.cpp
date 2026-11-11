# Mega graph definitions

`qwen35.inc` registers graphs and composes the shared attention and FFN fragments. C++ includes each fragment to generate graph IDs, launch selection, host slot matching and packing, parameter counts, and device operations. CMake tracks these files for both host and device builds. There is no generated header to update or separate generation command.

Each step declares its name, ggml slot kind, typed device function, number of following cluster barriers, slot matching mode, and host packing expression:

```cpp
MEGA_STEP(q_norm, rms_norm, rms_norm, 1, required,
          pack(rms_norm, at(q_norm)))
```

`at(name)` refers to a named matched slot. Use these names for scratch buffers and other cross-step references. Parameter positions and total counts come from the function signatures in `ggml-dayan-mega-executor.h`; `Binding` constructs the device call. Host `parameters->add<mega::function>(config, ...)` checks the config type and argument count at compile time. Each packed step must also use the declared device function and write exactly its argument count before launch.

The slot modes are:

- `required`: match and advance one slot.
- `optional`: leave the slot absent when its kind does not match. Its packing expression must handle `ctx.missing(name)`. Decode uses this for the omitted state scales.
- `reuse`: use the preceding matched slot without advancing. MXFP4 K/V projections use two device calls for one paired slot.
- `borrow`: match the current slot without consuming it. The final residual uses the next layer's norm slot, which must remain available after mega finishes.

The parameter packet ends with a cache-line-aligned `DayanClusterState` in ordinary DRAM. `Binding` supplies this address to matrix and attention configs without adding graph parameters. Per-core records and explicit cache maintenance keep peer metadata separate from the outer mega barrier. Internal rendezvous uses Shared Resources 20..23; each call completes two generations to return those tokens to phase zero. Only the outer step barrier uses Atomic Cache (16 bytes before Runtime alignment).

Keep synchronization explicit. A zero barrier count is valid only when the following operation can run independently on each core or the graph ends. MoE's `sort` step runs VEC TopK on the softmax probabilities and prepares grouped dispatch tables. Its final barrier publishes the selections and tables to following steps; the router step only computes logits.

Each device step calls `dayan_set_proc_name` with its declared name before execution. The following barriers stay in that trace segment. Names must fit in `DAYAN_THREAD_NAME_SIZE` including NUL; the operation table's fixed name array enforces this at compile time.

To add or remove an existing operation, edit its fragment. No parameter offsets, graph counts, decode indices, or dense-prefix lengths need editing. A new operation also needs a typed device function and a host packer that validates its tensors. Dayan-emu tests should assert that mega launched, so a packing failure cannot pass through standalone fallback.

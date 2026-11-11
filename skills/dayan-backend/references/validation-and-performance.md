# Validation and performance

## Validation layers

Use the narrowest existing test that has an independent reference, then expand according to blast radius.

1. Build the Dayan kernel object and affected test target.
2. Run the focused operator numerical test.
3. Run `test-dayan-graph` when support predicates, rewrite, fusion, memory layout, or common native helpers change.
4. Run an operator pipeline test when projections or fused neighbors change.
5. Run `test-dayan-qwen35-dense` when a Qwen3.5 production path changes.
6. Replay the saved full-model trace for graph shape and relative performance changes.

Do not add a test when an existing case exercises the changed contract. Add one only for a missing shape, layout, mask, tail, alias, or failure mode.

## Existing test map

| Test | Primary coverage |
| --- | --- |
| `test-dayan-matmul` | Standalone BF16 MatMul and CPU reference |
| `test-dayan-graph` | Rewrite, fusion, fan-out rejection, normalization, elementwise, reduction, layout, memory, RoPE, SSM convolution, and gated output |
| `test-dayan-flash-attention` | GQA, BF16/FP16 paths, causal masks, permuted K/V cache layout, physical/effective KV length, and decode |
| `test-dayan-attention-projection` | QG/K/V projection, cache writes, FlashAttention, gate, and output projection data flow |
| `test-dayan-gated-delta-net` | Head mapping, token cases, state, and numerical comparison |
| `test-dayan-qwen35-dense` | Four-layer hybrid dense graph, low-precision boundaries, intermediate stages, final hidden state, and logits |
| `test-dayan-moe` | Routing, fused expert projections, down projection, and weighted aggregation |
| `test-dayan-hotspots` | Focused diagnostic workloads for profiling, not an independent full correctness suite |

`test-dayan-graph` has command selectors for `rope`, `ssm-conv`, `gated-output`, `copy`, `get-rows`, and `memory`.

The graph suite's fan-out case intentionally produces an unsupported GLU message and then finishes with PASS. This proves a fusion did not eliminate an intermediate that has another consumer.

## Dayan-emu lifecycle

Use the existing `npu-firmware/run_dayan_emu_server.sh` lifecycle for functional work. Confirm no unrelated process owns the fixed simulator, FUSE, shared-memory, or socket resources before starting it. Use four Harts for the full Qwen3.5 path unless a focused operator explicitly uses another configuration.

Typical client commands are:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-flash-attention
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-qwen35-dense
```

Judge success by process status and numerical/reference output. Clean up only resources created by the current run.

## Trace capture and replay

`llama-dayan-trace` observes slots after graph rewrite and slot fusion. A signature contains:

- Slot kind and component operations.
- Tensor types, extents, byte strides, views, and source layouts.
- Raw operation parameters.
- Fused output layouts.

Names and addresses are excluded, so repeated layer shapes share a signature. The full slot sequence is retained as signature references.

Each run has cold and warm rounds. The filter executes only the first occurrence of each signature in a round and reports duplicates without Device execution. Occurrence counts scale the sampled mean to estimate full operator time.

For decode, the tool first builds a saved deterministic prefix with all Device slots skipped. This advances memory metadata to the real decode position without executing the prefix operators. `setup_us` is separate from the one-token decode measurement.

Use:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-dayan-trace \
  --replay dayan-trace/dayan-decode-128-trace.json \
  --trace /tmp/dayan-decode-after.json \
  --check
```

Require:

- `validation_pass=true`.
- Signature counts match.
- Slot order matches.
- Every required unique signature executes.
- Cold and warm output hashes match when `--check` is used.

`--check` is deterministic byte-hash comparison between replay rounds. It is not comparison with a CPU implementation. Skipped producers can leave synthetic or stale inputs for later unique slots.

## Timing interpretation

The slot timer includes Host preparation, config cache/upload handling, wrapper launch, and Device wait. Output downloads for `--check` are outside the slot timer.

Use these fields:

- `unique_replay_wall_us`: elapsed wall time for all unique signatures in one round.
- `measured_slot_total_us`: sum of measured unique slot intervals.
- `estimated_operator_total_us`: each unique mean multiplied by its full occurrence count.
- Per-signature `latency_times_count_us`: the best isolation for a changed operator.
- `unaccounted_us`: graph/scheduler overhead outside measured slots.

Compare performance only when all of these remain fixed:

- llama.cpp commit and build type except for the change under test.
- Kernel object and firmware build.
- Dayan-emu and QEMU versions.
- Hart count and memory size.
- Model file and saved trace.
- `--check` setting.
- Cold/warm interpretation.

Dayan-emu has run-to-run variation. Prefer repeated samples and a median for a focused operator. Use per-signature deltas to isolate a kernel change; the full estimated total includes variation from every operator.

Never present Dayan-emu wall time or `mcycle` as RTL or hardware performance. Use RTL simulation or hardware measurements for cycle-accurate claims, contention behavior, and final throughput.

## Numerical reporting

For an operator test, report shape, type, element count, maximum absolute error, maximum relative error where meaningful, worst index, and exit status. Reject NaN or infinity unless the operation contract explicitly expects it.

Preserve the established tolerance for unchanged low-precision boundaries. A tighter or looser tolerance is a separate behavioral change and needs justification.

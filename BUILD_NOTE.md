# Dayan Minimal Build, Test, and Run

The minimal direct inference executable in this checkout is `llama-completion`. It avoids the server and UI dependencies of `llama-cli`.

## Build

Run from the llama.cpp repository root:

```sh
cmake -S . -B build-dayan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_DAYAN=ON \
  -DDAYAN_ROOT="$HOME/dayan-base" \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=OFF

cmake --build build-dayan -j4 --target \
  llama-completion \
  test-dayan-matmul \
  test-dayan-graph \
  test-dayan-flash-attention \
  test-dayan-gated-delta-net \
  test-dayan-hotspots \
  test-dayan-qwen35-dense \
  test-dayan-moe
```

## Start the Dayan-emu server

Keep this command running in terminal 1:

```sh
cd "$HOME/dayan-base"
DAYAN_EMU_HARTS=4 \
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./run_dayan_emu_server.sh
```

The launcher configures 4096 MiB for both QEMU DRAM and the HAL Normal-memory region by default. Set `DAYAN_EMU_MEMORY_MIB` to override both values together.

Wait for `Dayan-emu server is ready` before starting llama.cpp. Press Ctrl-C in terminal 1 after the client exits.

Do not start another Dayan-emu instance while `/tmp/virtual_npu`, `/tmp/dayan-server.sock`, `dayan-emu`, `dl-riscv64`, or `dayan_server` belongs to a live run. The Dayan-emu frontend uses the fixed `/tmp/virtual_npu` path.

## Device-resident tensor storage

The Dayan buffer type allocates one NPU storage buffer for each ggml arena. `init_tensor` creates an NPU view at the tensor's planned arena offset, so weights, activations, recurrent state, KV cache, and outputs keep stable Device addresses for the lifetime of the arena.

Model loading and explicit input updates use `set_tensor` to upload data. Graph execution passes the preallocated Device views directly between kernels. It does not download an operator result before launching its consumer. Device-to-Host transfer exists only in explicit `get_tensor`, such as output inspection, CPU sampling, or a scheduler boundary. Dayan reports only the Dayan buffer type as directly consumable; the ggml scheduler creates explicit copies when a graph crosses to another backend.

The current transfer audit is:

```sh
rg -n 'npu_memcpy_(d2h|h2d)_async|ggml_backend_dayan_(pack|unpack|append|download)' \
  ggml/src/ggml-dayan/ggml-dayan.cpp
```

- D2H appears only in `ggml_backend_dayan_buffer_get_tensor`.
- H2D appears in tensor initialization, memset/clear, and small launch configuration uploads.
- Same-backend tensor copies use Device-to-Device copy.
- No Host pack/unpack path remains in graph execution.

## Graph rewrite and fusion

Before scheduling, the graph rewrite inserts one BF16 or FP16 activation cast for each unique F32 matrix input and weight type. Matrix multiplication therefore reads activations in the same storage type as its weights while retaining F32 accumulation and F32 ggml outputs.

At each FlashAttention boundary, the rewrite selects the common K/V storage type and converts Q to match it. Existing BF16 or FP16 inputs remain in that type. If K/V are both F32, the rewrite follows an existing low-precision Q type or defaults to BF16. Casts are shared by source tensor and destination type, so a projection result used by more than one consumer has one Device-resident low-precision copy.

When an F32 MatMul output reaches its next MatMul only through `VIEW`, `RESHAPE`, `PERMUTE`, or `TRANSPOSE`, the rewrite can move the BF16/FP16 cast before that metadata-only chain and rebuild the aliases in the low-precision type. It does this only when the chain has no F32 side consumer and is not a graph output. The compiler matches the MatMul producer and its low-precision output by dependency, so the cast remains eligible for the existing MatMul epilogue slot even when metadata nodes separate them in the original graph.

The normalization combinations are represented by one `norm_pipeline` slot with optional residual and low-precision outputs:

```text
RMS_NORM -> MUL(scale)
RMS_NORM -> MUL(scale) -> CPY(BF16/FP16)
ADD -> RMS_NORM -> MUL(scale)
ADD -> RMS_NORM -> MUL(scale) -> CPY(BF16/FP16)
```

MatMul remains a separate slot and consumes the Device-resident low-precision result. This removes separate kernel variants for every Norm/residual/cast/MatMul combination while preserving a MatMul fusion slot for future work.

The fused normalization kernel spans several logical ggml nodes, so the graph rewrite adds allocator dependencies from the final logical node to the original ADD, RMSNorm, and scale inputs. These dependencies prevent a later CPY output from reusing an input range before both native Tasks have finished reading it. Without this lifetime extension, the scheduler can legally overlap the BF16 output with a copied scale tensor under sequential ggml semantics, which becomes a cross-Task data race after fusion. This is an allocation lifetime hazard, not a MatMul pipeline deadlock.

`ADD -> RMS_NORM` cannot move after MatMul. RMSNorm depends on the complete residual vector and is nonlinear, so in general `W * RMSNorm(x + r)` is not equivalent to an ADD or RMSNorm applied to `W * x`. The useful simplification is one parameterized normalization slot followed by an independent MatMul slot.

The MatMul template still provides a compile-time epilogue function pointer. The SWIGLU and GEGLU slots compute both matrix results, apply the VEC epilogue to URAM data, and perform one final low-precision writeback. Ordinary and MoE matrix operations remain separate kernels.

## Native Task execution

Non-MatMul elementwise, layout, reduction, normalization, routing, RoPE, concat, and copy kernels use native Task dimensions with two Tasks per core. Each Task executes a local load, wait, compute, wait, and writeback sequence. Independent Task coordinates interleave these serial sequences to hide Device latency without a Producer/Consumer kernel pipeline.

The current per-Task URAM reservations are:

- Normal native Task kernels: 8 KiB.
- Gated Delta Net: 100 KiB.
- MatMul family, FlashAttention, gated output projection, and SSM convolution: 1 MiB.

Gated Delta Net also uses two native Tasks. The MatMul family, FlashAttention, and gated output projection retain their dedicated single-Task MAT-oriented launch paths. SSM convolution also uses one dedicated Task, but its tensor computation uses LSU and VEC only.

VEC control issue snapshots the complete VEC TCSR state. Software can update the configuration registers immediately after the control write; each issued block continues with its own snapshot. The Dayan-emu implements this copy in `vec_t::handle_vfork` and preserves it when the executor creates each subtask. Do not add a software VEC lock around configuration and issue. Gated Delta Net relies on this behavior to run two Tasks concurrently with separate URAM regions.

## Shared VEC activation and scalar-loop removal

`ggml-dayan-activation.h` contains the common VEC implementations of SiLU, SWIGLU, GEGLU, sigmoid-multiply, and FP32-to-FP16/BF16 packing. MatMul epilogues, normalization, memory conversion, SSM convolution, FlashAttention, and gated output projection call these implementations instead of maintaining operator-local copies.

The FP32-to-FP16/BF16 path uses the packed Convert semantics. Convert places each adjacent pair of 16-bit results in an even 32-bit lane, then `tx.vec.shuffle.compress` with `amount=1` gathers the 32 valid lanes into the low half of the vector. Thus conversion and gathering require two VEC data instructions. The VEC block also loads FP32 input and performs one masked low-32-lane URAM store; two scalar instructions initialize that fixed store mask once per VEC task. Input iterators advance by 256 bytes and output iterators by 128 bytes. Rows use their actual padded byte stride, so tail blocks do not create a second FP32 output or write undefined high lanes.

Device parameter checks use `DAYAN_PARAMS_IF` and `DAYAN_PARAMS_ERROR` from `ggml-dayan-common.h`. Uncomment `ENABLE_PARAMS_CHECK` in that header to enable the checks. The default build leaves it commented out. Both configurations compile, and the focused operator tests pass with the checks enabled.

The four audited scalar tensor paths now use accelerator instructions:

- RoPE loads positions, factors, inputs, and outputs with LSU. VEC selects MRoPE axes, converts positions, computes reciprocal and sine/cosine values, and rotates each pair. Static axis and angle multiplier tables are prepared in the launch config.
- SSM convolution gathers FP32 input and depthwise weights into tap-major URAM tiles with multi-Dim LSU. VEC multiplies and accumulates the taps in FP32, applies the shared VEC SiLU when fused, and LSU writes the strided output once.
- Gated Delta Net uses LSU transpose/gather for reductions and VEC for reduction combine, delta, and output scale. Its two native Tasks use disjoint 100 KiB URAM regions without a VEC lock.
- Gated output projection uses LSU, VEC sigmoid-multiply, and VEC conversion and packing for its tensor path.

Gated output preparation is now the MatMul A-tile loader. The loader uses LSU multi-Dim access to load one `128 x 256` attention/gate tile, expands packed BF16/FP16 gate values to the 32-bit lane layout in URAM, and runs one VEC block per phase. VEC `Dim0` and `Dim1` stay at zero, `Dim2` is the real count of 64-lane slices, and `Dim3` is the token row count. VEC Convert plus Compress writes the gated FP32 tile directly into the low-precision shared A buffer consumed by MAT.

The gated activation is not allocated or written in Device DRAM. The common MatMul template has a compile-time input-loader function slot and an optional URAM scratch size. The ordinary loader still copies low-precision A from Device memory with MAT swizzle. The gated loader writes an unswizzled low-precision A tile in shared URAM, marks the existing producer resource ready, and MAT consumes it without another LSU load. The graph compiler restricts this prototype to head dimensions divisible by 64 that do not cross a 256-element K tile.

The multi-Dim path exposed an iterator bug in the shared sigmoid-multiply implementation. Its value operand configured the v1 iterator but selected v0 in the load instruction. This was invisible when `rows=1` because both outer offsets were zero. Multi-row execution then applied the gate row stride to the attention input. The load now selects v1, so gate, attention, and output can use independent multi-Dim strides.

`test-dayan-attention-projection` passes for BF16 and FP16 after full fusion. Final gated output errors are 1.66707e-05 for BF16 and 4.46122e-05 for FP16. The graph regression uses `head_dim=128`, four heads, and width 512. Each head covers two real VEC Dim2 slices, and the projection covers two MatMul K tiles. BF16 is exact and FP16 has maximum absolute error 3.32892e-05.

After switching all FP32-to-FP16/BF16 operator paths from MACRO to VEC Convert plus Compress, the Dayan-emu execution test validates the exact packed lane layout for both types. `test-dayan-matmul 3 160 64`, `test-dayan-attention-projection`, `test-dayan-flash-attention`, and the complete `test-dayan-graph` pass. The graph suite includes widths 128, 130, and 256, F32-to-F16 copy, SWIGLU/GEGLU, gated output, and three SSM convolution shapes. The focused non-64-aligned MatMul is exact; attention projection reports BF16 attention `max_abs=6.14524e-05`, FP16 attention `max_abs=1.35154e-05`, and FP16 gated output `max_abs=6.20633e-06`.

The Dayan-emu `dyperf` comparison uses the same `test-dayan-hotspots` shapes before and after these changes. The values below are the full lifetime span of the named Device task, in Dayan-emu cycles:

| Operator task | Baseline | VEC/LSU version | Change |
| --- | ---: | ---: | ---: |
| `rope` | 243,067,652 | 545,127,776 | +124.3% |
| `ssm-conv` | 252,370,076 | 242,271,000 | -4.0% |
| `ggml-gdn` | 566,505,012 | 843,881,256 | +49.0% |
| `gate-out` | 96,600,312 | 699,274,428 | +623.9% |

Only SSM convolution is faster in this Dayan-emu measurement. The result must not be presented as RTL performance: Dayan-emu is used here for functional and host/device validation, and its VEC execution cost is not a hardware timing result. The regressions are still recorded because the requested before/after comparison must remain visible.

## Dayan-emu convolution execution

Gated Delta Net and SSM convolution are separate Device operators. In a Qwen3.5 recurrent layer, SSM convolution produces the Q/K/V input consumed by Gated Delta Net, but the `ggml-gdn` kernel does not execute convolution internally.

The Dayan-emu MAT convolution path now decodes input and kernel tensors once into contiguous FP32 buffers. The compute loop then reuses those buffers and performs each Ci dot product over contiguous memory. Padding, stride, dilation, tensor strides, swizzle, accumulation, and every previously supported non-scaled MAT data type remain generic. A function-local GCC optimization attribute enables fast-math vectorization only for this dot product; the rest of MAT and VEC retains the repository `-O2` behavior. On the test host, the main BF16 loop compiles to 256-bit `vfmadd231ps` instructions.

A host microbenchmark uses the SSM channel-tile shape `Hi=Ho=1, Wi=Wo=64, Ci=Co=256, Hk=Wk=1`. The previous loop, which decoded both operands inside every MAC, takes about 9.48 ms per convolution. The decoded and SIMD loop takes about 0.79 ms, or about 12.0x less Host time.

Three `dyperf` runs of `test-dayan-hotspots ssm-conv` report task spans of 248,745,328, 247,095,904, and 247,037,516 Dayan-emu cycles. The median is 247,095,904, down 55.2% from the 551,892,724 pre-optimization capture. The corresponding `RUNNING` total remains about 31.7 million cycles; most of the span change comes from completing the Host-side MAT model sooner and unblocking the Device task. These Dayan-emu cycle counts are not RTL performance estimates.

`make test -j4`, `test-dayan-graph ssm-conv`, `test-dayan-gated-delta-net`, and both BF16 and FP16 cases in `test-dayan-qwen35-dense` pass with the optimized Dayan-emu. The largest SSM convolution absolute error is `1.49012e-08` in the `kernel=4, tokens=69, channels=384` case.

## Pure VEC SSM convolution

The SSM depthwise convolution no longer uses the MAT path described in the historical measurements above. LSU gathers channel-major F32 input and weights into tap-major URAM tiles. VEC performs one elementwise multiplication for each tap, accumulates the products in F32, and optionally calls the common SiLU implementation in-place. LSU writes the result once after the optional activation. The kernel does not convert to BF16, build diagonal weight matrices, or use `tx.mat.conv2d` and `tx.mat.conv2da`. It supports kernel widths up to 63 because one VEC block can encode at most 255 instructions and the convolution uses four instructions per tap.

The focused Dayan-emu graph test passes all three SSM cases. Kernel 1 with 5 tokens and 128 channels is exact. Kernel 4 with fused SiLU has maximum absolute error `1.49012e-08` for 5 tokens, 130 channels, and 2 sequences, and for 69 tokens and 384 channels.

For the Decode shape `kernel=4`, `tokens=1`, and `channels=6144`, three `dyperf` samples measure the task from its first state until it enters `WAIT_EXIT`. The raw convolution spans are 72,746, 77,194, and 78,886 Dayan-emu cycles. The fused SiLU spans are 80,404, 82,454, and 80,136 cycles. The medians are 77,194 and 80,404 cycles, so fused SiLU adds about 4.2% to this task interval. The median `RUNNING` totals are 14,484 and 21,053 cycles, respectively. SiLU adds about 45.4% to the instruction-running portion, while LSU and accelerator wait states reduce its share of the complete task interval.

The Decode trace replay records a warm fused SSM latency of 25.044 ms for one unique shape, compared with 3.334431 s in the preceding GET_ROWS-optimized trace. This is about 133.1x faster in Dayan-emu. With 18 occurrences, its estimated contribution changes from 60.019758 s to 0.450792 s. The complete warm Decode estimate in the new capture is 59.425190 s. Changes in the complete estimate include Dayan-emu run-to-run variation, so the per-signature SSM values isolate this kernel change more directly. These are Dayan-emu software simulation measurements, not RTL or hardware timing results.

FlashAttention consumes strided BF16 or FP16 Device views directly. Both kernel entries use the same Task pipeline and FP32 online-softmax state. Their MAT data type and FP32 probability conversion are selected at compile time, so FP16 Q/K/V use `DT_FP16` for both QK and PV without an intermediate BF16 tensor. The config carries row, head, and batch byte strides for Q, K, and V, and separates the physical KV-cache length from the effective causal prefix. Concat and Gated Delta Net likewise use full tensor byte strides rather than Host-packed contiguous buffers.

## Run the focused tests

Run these commands from the llama.cpp repository root in terminal 2:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-matmul
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-flash-attention
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-gated-delta-net
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph rope
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph ssm-conv
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph gated-output
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-attention-projection
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-qwen35-dense
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-moe
```

The 2026-08-11 Dayan-emu results are:

- BF16 MatMul, `M=1 N=8064 K=1024`: `max_abs=0`, `max_rel=0`.
- Graph suite: PASS for BF16/FP16 normalization, residual, activation cast, MatMul, SWIGLU/GEGLU, elementwise, layout, reduction, RoPE, gated output, SSM convolution, a 73,728-byte nonzero-offset in-place Scale view, and BF16/FP16 cast hoisting across metadata-only layout chains.
- RoPE modes 0, 2, and 8 are exact; mode 40 has `max_abs=1.49012e-07`.
- SSM convolution kernel 1 is exact; kernel 4 with SiLU has `max_abs=1.49012e-08` for both 130- and 384-channel cases.
- Gated output projection is exact for BF16 and has `max_abs=2.36332e-05` for FP16.
- Native FP16 FlashAttention, `q_len=17`, KV 23, 4 query heads, and 2 KV heads: `max_abs=0.000161469`.
- FlashAttention permuted BF16, `q_len=3`, physical KV 64, effective KV 9: `max_abs=0.000228759`.
- FlashAttention decode, `q_len=1`, physical KV 256, effective KV 1: exact.
- FP16 projection through native FP16 FlashAttention: attention `max_abs=0.00012207`.
- Gated Delta Net, 16 heads and 16 tokens: `max_abs=3.72529e-09`.
- Four-layer Qwen3.5 dense BF16 graph: `max_abs=0.00158247`, logits `max_abs=0.00138266`.
- Four-layer Qwen3.5 dense FP16 graph: `max_abs=0.00390625`, FlashAttention `max_abs=0.000976562`, logits `max_abs=0.000968814`.
- MoE with 512 experts, top-4, and two tokens: fused `max_abs=0.00225067`, down `max_abs=0.0321879`, final output `max_abs=0.000781553`.

`test-dayan-graph` contains a negative fan-out case. Its expected `unsupported op SWIGLU` message proves that the graph compiler did not fuse a GLU intermediate with another consumer. The test must continue and finish with `test-dayan-graph: PASS`.

## Run Qwen3.5 on Dayan

Device-resident full-model inference requires the Dayan HAL Normal-memory region and the Dayan-emu shared-memory backing to exceed the model arena plus graph, KV-cache, firmware, and kernel allocations. Qwen3.5-0.8B BF16 currently requests a 1,505,783,040-byte model arena (`1436.03 MiB`) before graph buffers.

On 2026-08-11, `/dev/shm` has a 32 GiB limit and `run_dayan_emu_server.sh` defaults both memory limits to 4096 MiB. The full-model smoke allocated and uploaded the `1436.03 MiB` DAYAN model buffer successfully. Before native FP16 FlashAttention was added, the graph rewrite processed 181 matrix multiplications and 18 FlashAttention inputs with 109 shared low-precision casts, and graph reservation verified all 972 compute nodes on DAYAN.

After native FP16 FlashAttention was added, a context-only reserve with the default F16 K/V cache processed 181 matrix multiplications and 6 FlashAttention inputs with 97 shared casts. It verified all 960 compute nodes in that reserve graph on DAYAN. Only Q needed conversion; K/V stayed F16. The full Qwen3.5 reserve graph has no eligible metadata-only chain, so it reports `hoisted=0` and retains all 97 casts. Synthetic BF16 and FP16 graph cases each report `hoisted=1` and exact output.

The full-model rewrite takes about 0.43-1.52 ms per reserve graph, while the complete scheduler reserve takes about 23-57 ms. A multi-minute pause after `warming up the model with an empty run` or after prompt tokenization is Dayan-emu Device execution, not graph optimization. Use an empty prompt with `--no-warmup` for a reserve-only diagnostic:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-completion \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  -dev DAYAN \
  -ngl all \
  -fit off \
  --no-warmup \
  -fa on \
  -c 512 \
  -n 0 \
  -p "" \
  -v
```

This command returns 255 after printing `input is empty`; the completed reserve and Dayan-only verification are the intended result. A nonempty prompt still starts Device prompt evaluation even with `-n 0`. With `-fit off`, `-dev DAYAN` is required. Without the explicit device list, no accelerator is selected, layers stay on CPU, and Dayan tensor buffer overrides conflict with CPU scheduling.

The first full execution exposed a kernel cache issue after graph reservation. Each generated Host wrapper packages the same complete 268 KiB Device object, but the server previously keyed loaded kernel modules only by the transient object pointer inside each package. Resolving a different entry attempted to allocate and relocate the identical image again and the Scale launch received `-EINVAL`. The server now compares object contents, attaches every equivalent package identity to one loaded image, and releases the Device image only after the final package identity is forgotten. This was a kernel resolution failure, not a Scale view or native Task deadlock.

With that fix, the full prompt smoke completed 1484 nodes as 707 Dayan slots with no fallback. The 4-hart Dayan-emu processed the six-token prompt in 679734.28 ms. This is a functional simulator result, not a performance result.

A later run failed while launching `CPY final_output-12` because its 224-byte Host-owned launch descriptor was at `0xdc6f9ec0`. BL1 incorrectly used `CONFIG_RAM_BASE + CONFIG_RAM_SIZE` as the upper bound for launch descriptors, kernel entries, dcache flush ranges, and profile buffers. `CONFIG_RAM_SIZE` is the 64 MiB BL1 linker region, not the dynamic Host-managed Normal-memory arena. The Dayan-emu launcher now keeps the default 64 MiB BL1 link region while configuring only QEMU DRAM and the HAL Normal-memory arena to 4096 MiB. BL1 retains alignment, Normal-memory lower-bound, and overflow checks for Host-owned ranges, while the Server HAL memory map supplies the dynamic upper bound. Fixed coherent and per-Hart URAM windows retain their bounded checks.

After correcting that boundary, the same six-token prompt completed 1472 nodes as 695 Dayan slots with no fallback or negative launch status. Prompt evaluation took 577940.11 ms in the 4-hart Dayan-emu. The earlier `CPY final_output-12` failure did not recur.

## Fused slot trace and replay

`llama-dayan-trace` records the final `ggml_backend_dayan_graph_plan::slots` after graph fusion. It does not record the original GGML node list. A deduplication signature contains the fused slot kind, populated component-node indices, GGML operations, tensor types, dimensions, byte strides, view layouts, source layouts, raw operation parameters, and fused output layouts. Tensor names and addresses are excluded, so identical layer shapes share one entry. The full slot order is stored as references to the deduplicated entries.

Capture and replay use deterministic saved tokens, `n_batch=n_ubatch=128`, native FlashAttention, and all model layers on DAYAN. Each invocation builds the complete graph twice but executes only the first occurrence of every unique fused signature in each round. Duplicate slots are reported without Device execution. Their occurrence counts scale the measured unique-slot latency to estimate full-model operator time. The cold round includes first kernel/module resolution; the warm round reuses the same model, context, Device buffers, slot resources, and module cache. `--replay` checks the fused signatures, counts, and order against the input trace. `--check` additionally downloads each executed unique-slot output after its timer and compares exact cold/warm byte hashes. Skipped producers can leave later slots with synthetic input values, so this is a deterministic shape replay check, not an independent CPU numerical reference.

`--decode` records one token after a deterministic prefix of `--prefill` tokens. Each round clears model memory and builds the prefix with all Device slots filtered out. This advances the memory metadata required to form the real decode graph without running the prefix operators. The JSON reports graph setup cost as `setup_us`; decode slot and end-to-end times cover only the final token.

The 2026-08-11 Qwen3.5-0.8B BF16 capture used a 4-hart Dayan-emu with 4096 MiB Normal memory:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-dayan-trace \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  --trace dayan-trace/dayan-prefill-128-trace.json \
  --prefill 128
```

The original 2026-08-11 trace executed all 695 slots in both rounds. It produced 50 unique signatures but did not use them to reduce execution. The cold end-to-end time was 6374.88 s and the warm time was 6137.72 s. Timed slots accounted for 6374.78 s and 6137.62 s respectively, leaving about 101 ms and 95 ms of graph, scheduler, and output overhead.

The largest warm contributions grouped by fused kind were:

| Fused kind | Calls | Warm contribution |
| --- | ---: | ---: |
| `gated_output_projection` | 6 | 1991.19 s |
| `gated_delta_net` | 18 | 1899.58 s |
| `norm_pipeline` | 73 | 560.14 s |
| `l2_norm` | 36 | 461.53 s |
| `copy` | 115 | 261.16 s |
| `rope` | 12 | 244.99 s |
| `mul_mat` | 121 | 227.50 s |
| `mul_mat_swiglu` | 24 | 124.48 s |

The trace validates that Gated Delta Net and gated output projection dominate this Dayan-emu run. It also exposes 115 independent copy slots totaling 261.16 s in the warm round, so copy elimination remains measurable work after the existing MatMul and normalization fusions.

The 2026-08-12 replay after the packed VEC Convert change executed only the 50 unique Prefill-128 signatures per round and skipped the other 645 slots. Counts and order matched the original trace. The warm unique replay took 339.25 s; multiplying each unique latency by its occurrence count estimated 4744.02 s of full-model operator time. The largest warm estimates were Gated Delta Net 1979.21 s, Norm Pipeline 739.28 s, L2 Norm 562.77 s, Copy 360.75 s, RoPE 284.05 s, and MatMul 230.42 s.

`dayan-trace/dayan-decode-128-trace.json` records one token at position 128. Its 695 fused slots deduplicate to 49 signatures. Each round executed 49 and skipped 646; counts and order matched, and an independent replay of the saved file passed. The capture warm unique replay took 35.30 s and estimated 306.69 s for all Decode slots. The independent replay estimated 283.48 s. Dayan-emu software execution has noticeable run-to-run variation, so the JSON preserves the individual samples and these values are not RTL or hardware performance estimates.

COPY now has two same-type fast paths. A fully contiguous source and destination split contiguous 64-byte transactions across the native Tasks. Each Task uses one multidimensional LSU D2D instruction, and the final Task handles a possible partial transaction. A two-dimensional source with a contiguous `dim0`, one `dim2` and `dim3` slice, and at most 64 bytes per row also uses one multidimensional LSU D2D instruction per native Task. Each Task copies interleaved rows directly from DRAM to DRAM with partial-byte transactions. Neither path stages data through URAM or Host memory. Type conversion and other irregular layouts retain the URAM and VEC Convert path.

The focused Dayan-emu test covers the Decode shapes `ne=[128,128,16,1]` with contiguous strides and `ne=[3,6144,1,1]`, `nb=[4,16,98304,98304]`. Both are exact. The existing F32-to-FP16 memory test also remains exact. Replaying `dayan-trace/dayan-decode-128-trace.json` after this change produced the independent `dayan-trace/dayan-decode-128-copy-lsu-opt.json` trace and passed signature, count, order, and execution validation. Warm COPY results were:

| Shape | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| F32 `ne=[3,6144,1,1]`, padded rows | 4.634545 s | 9.689 ms | 478.3x |
| F32 `ne=[128,128,16,1]`, contiguous | 2.704588 s | 10.693 ms | 252.9x |
| All unique COPY signatures | 7.522693 s | 0.253150 s | 29.7x |
| COPY latency times occurrence count | 134.793081 s | 3.861267 s | 34.9x |

The complete warm Decode operator estimate changed from 306.685110 s to 243.706186 s in these two Dayan-emu captures. This total includes run-to-run variation in every operator; the per-signature COPY numbers isolate the implemented optimization more directly.

GET_ROWS now has an F32-to-F32 fast path. The selected row stays in Device memory and each native Task copies a contiguous range of 64-byte transactions with one multidimensional LSU D2D instruction. The final Task issues one partial transaction when needed. All selected rows are issued before one LSU barrier. BF16-to-F32 conversion and overlapping source and output ranges retain the existing URAM path. COPY and GET_ROWS share the same D2D LSU configuration helper.

The focused Dayan-emu test selects a nonzero row from an F32 tensor with width 262144 and compares all 262144 output elements exactly. The existing BF16 GET_ROWS memory test remains exact, and the complete `test-dayan-graph` suite passes. Replaying `dayan-trace/dayan-decode-128-copy-lsu-opt.json` produced the independent ignored trace `dayan-trace/dayan-decode-128-get-rows-opt.json`; all 695 slots, 49 unique signatures, occurrence counts, order, and executions validated. Warm GET_ROWS results were:

| Shape | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| F32 `ne=[18432,1,1,1]` | 225.637 ms | 10.587 ms | 21.3x |
| F32 `ne=[262144,1,1,1]` | 3.230481 s | 10.669 ms | 302.8x |
| All unique GET_ROWS signatures | 3.496033 s | 51.502 ms | 67.9x |
| GET_ROWS latency times occurrence count | 62.250039 s | 412.854 ms | 150.8x |

The complete warm Decode operator estimate changed from 243.706186 s to 181.910332 s in these captures. The current largest estimated contributions are SSM convolution with SiLU at 60.019758 s, MatMul at 41.278257 s, Gated Delta Net at 18.885006 s, fused MatMul with SWIGLU at 16.293000 s, Norm Pipeline at 10.653345 s, and FlashAttention at 10.141086 s. These are Dayan-emu software simulation timings, not RTL or hardware performance estimates.

Run the full-model smoke with:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-completion \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  -dev DAYAN \
  -ngl all \
  -fit off \
  --no-warmup \
  -no-cnv \
  -v \
  --log-colors off \
  -c 256 \
  -b 8 \
  -ub 8 \
  -s 1234 \
  -n 1 \
  -p "Hi, Who are you?" \
  > /tmp/dayan-qwen35.log 2>&1
```

`-dev DAYAN` selects the Dayan device explicitly. `-ngl all` places all model layer slots on Dayan. `-fit off` remains required because the prototype reports zero available Device memory to llama.cpp automatic fitting.

Validate the log with:

```sh
rg 'using device DAYAN|assigned to device DAYAN|verified .* DAYAN|lowered .* slots|unsupported|fallback|status=-|trap|timeout|eval time' \
  /tmp/dayan-qwen35.log
```

A valid run must assign all 26 model layer slots to DAYAN, print the Dayan-only graph verification, complete every lowering summary, and contain no unsupported operation, fallback, negative status, trap, or timeout. Fixed slot counts from the former Host-streaming implementation are no longer a valid acceptance criterion.

## Expected Host usage

This is not a CPU-free Host process. Model files and explicit inputs originate on the Host, CPU sampling reads the requested logits output, and the Host submits Device commands and small configs. The Dayan-emu also executes `dayan-emu` and RISC-V QEMU on the Host CPU, so high CPU usage from `dayan-emu` or `dl-riscv64` represents simulated Dayan execution.

Pure Dayan graph execution does not pack or download intermediate tensors. Every non-metadata model graph node must be assigned to Dayan; context creation or graph execution fails before compute if the scheduler assigns one to another backend. Sampling remains on CPU and is not inserted into the Dayan model graph.

Use `GGML_SCHED_DEBUG=2` for scheduler diagnostics. Use this command during a run to separate simulator and Host usage:

```sh
pidstat -u -p "$(pgrep -d, -f 'llama-completion|dayan-emu|dl-riscv64|dayan_server')" 1
```

## Current limitations

- The matrix kernel uses one consumer slice. Validate MAT issue ordering and VEC/store ownership before increasing `MATMUL_DAYAN_1C_SLICES`; VEC TCSR reconfiguration does not require software serialization because the control issue snapshots the configuration.
- FlashAttention requires Q/K/V to use one common BF16 or FP16 storage type. Mixed BF16/FP16 K/V inputs are not supported.
- The Qwen3.5-0.8B model is dense, so use `test-dayan-moe` to validate the independent MoE and fused expert paths.
- Full-model Dayan-emu execution is slow. The six-token functional prompt smoke takes about 11 minutes and must not be used as an RTL performance result.

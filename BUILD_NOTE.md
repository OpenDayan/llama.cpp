# Dayan Minimal Build, Test, and Run

The minimal direct inference executable in this checkout is `llama-completion`. It avoids the server and UI dependencies of `llama-cli`.

## Build

Run from the llama.cpp repository root:

```sh
cmake -S . -B build-dayan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_DAYAN=ON \
  -DDAYAN_ROOT="$HOME/npu-firmware" \
  -DDAYAN_NPU_KERNELS_DIR="$HOME/npu-kernels" \
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

## Start the cmodel server

Keep this command running in terminal 1:

```sh
cd "$HOME/npu-firmware"
NPU_CMODEL_HARTS=4 \
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./run_cmodel_server.sh
```

The launcher configures 4096 MiB for both QEMU DRAM and the HAL Normal-memory region by default. Set `NPU_CMODEL_MEMORY_MIB` to override both values together.

Wait for `Dayan cmodel server is ready` before starting llama.cpp. Press Ctrl-C in terminal 1 after the client exits.

Do not start another cmodel instance while `/tmp/virtual_npu`, `/tmp/dayan-server.sock`, `rvt_emulator`, `dl-riscv64`, or `dayan_server` belongs to a live run. The cmodel frontend uses the fixed `/tmp/virtual_npu` path.

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

Gated Delta Net also uses two native Tasks. The MatMul family, FlashAttention, gated output projection, and SSM convolution retain their dedicated single-Task MAT-oriented launch paths.

VEC control issue snapshots the complete VEC TCSR state. Software can update the configuration registers immediately after the control write; each issued block continues with its own snapshot. The cmodel implements this copy in `vec_t::handle_vfork` and preserves it when the executor creates each subtask. Do not add a software VEC lock around configuration and issue. Gated Delta Net relies on this behavior to run two Tasks concurrently with separate URAM regions.

## Shared VEC activation and scalar-loop removal

`ggml-dayan-activation.h` contains the common VEC implementations of SiLU, SWIGLU, GEGLU, and sigmoid-multiply. MatMul epilogues, SSM convolution, and gated output projection call these implementations instead of maintaining operator-local copies.

Device parameter checks use `DAYAN_PARAMS_IF` and `DAYAN_PARAMS_ERROR` from `ggml-dayan-common.h`. Uncomment `ENABLE_PARAMS_CHECK` in that header to enable the checks. The default build leaves it commented out. Both configurations compile, and the focused operator tests pass with the checks enabled.

The four audited scalar tensor paths now use accelerator instructions:

- RoPE loads positions, factors, inputs, and outputs with LSU. VEC selects MRoPE axes, converts positions, computes reciprocal and sine/cosine values, and rotates each pair. Static axis and angle multiplier tables are prepared in the launch config.
- SSM convolution gathers FP32 input and weight data with multi-Dim LSU, converts it with MACRO, scatters weights into the diagonal MAT layout with LSU, applies the shared VEC SiLU, and writes strided output with LSU.
- Gated Delta Net uses LSU transpose/gather for reductions and VEC for reduction combine, delta, and output scale. Its two native Tasks use disjoint 100 KiB URAM regions without a VEC lock.
- Gated output projection uses LSU, VEC sigmoid-multiply, and MACRO conversion for its tensor path.

The cmodel `dyperf` comparison uses the same `test-dayan-hotspots` shapes before and after these changes. The values below are the full lifetime span of the named Device task, in cmodel cycles:

| Operator task | Baseline | VEC/LSU version | Change |
| --- | ---: | ---: | ---: |
| `rope` | 243,067,652 | 545,127,776 | +124.3% |
| `ssm-conv` | 252,370,076 | 242,271,000 | -4.0% |
| `ggml-gdn` | 566,505,012 | 843,881,256 | +49.0% |
| `gate-out` | 96,600,312 | 699,274,428 | +623.9% |

Only SSM convolution is faster in this cmodel measurement. The result must not be presented as RTL performance: cmodel is used here for functional and host/device validation, and its VEC execution cost is not a hardware timing result. The regressions are still recorded because the requested before/after comparison must remain visible.

## Cmodel convolution execution

Gated Delta Net and SSM convolution are separate Device operators. In a Qwen3.5 recurrent layer, SSM convolution produces the Q/K/V input consumed by Gated Delta Net, but the `ggml-gdn` kernel does not execute convolution internally.

The cmodel MAT convolution path now decodes input and kernel tensors once into contiguous FP32 buffers. The compute loop then reuses those buffers and performs each Ci dot product over contiguous memory. Padding, stride, dilation, tensor strides, swizzle, accumulation, and every previously supported non-scaled MAT data type remain generic. A function-local GCC optimization attribute enables fast-math vectorization only for this dot product; the rest of MAT and VEC retains the repository `-O2` behavior. On the test host, the main BF16 loop compiles to 256-bit `vfmadd231ps` instructions.

A host microbenchmark uses the SSM channel-tile shape `Hi=Ho=1, Wi=Wo=64, Ci=Co=256, Hk=Wk=1`. The previous loop, which decoded both operands inside every MAC, takes about 9.48 ms per convolution. The decoded and SIMD loop takes about 0.79 ms, or about 12.0x less Host time.

Three `dyperf` runs of `test-dayan-hotspots ssm-conv` report task spans of 248,745,328, 247,095,904, and 247,037,516 cmodel cycles. The median is 247,095,904, down 55.2% from the 551,892,724 pre-optimization capture. The corresponding `RUNNING` total remains about 31.7 million cycles; most of the span change comes from completing the Host-side MAT model sooner and unblocking the Device task. These cmodel cycle counts are not RTL performance estimates.

`make test -j4`, `test-dayan-graph ssm-conv`, `test-dayan-gated-delta-net`, and both BF16 and FP16 cases in `test-dayan-qwen35-dense` pass with the optimized cmodel. The largest SSM convolution absolute error is `1.49012e-08` in the `kernel=4, tokens=69, channels=384` case.

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

The 2026-08-11 cmodel results are:

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

Device-resident full-model inference requires the Dayan HAL Normal-memory region and the cmodel shared-memory backing to exceed the model arena plus graph, KV-cache, firmware, and kernel allocations. Qwen3.5-0.8B BF16 currently requests a 1,505,783,040-byte model arena (`1436.03 MiB`) before graph buffers.

On 2026-08-11, `/dev/shm` has a 32 GiB limit and `run_cmodel_server.sh` defaults both memory limits to 4096 MiB. The full-model smoke allocated and uploaded the `1436.03 MiB` DAYAN model buffer successfully. Before native FP16 FlashAttention was added, the graph rewrite processed 181 matrix multiplications and 18 FlashAttention inputs with 109 shared low-precision casts, and graph reservation verified all 972 compute nodes on DAYAN.

After native FP16 FlashAttention was added, a context-only reserve with the default F16 K/V cache processed 181 matrix multiplications and 6 FlashAttention inputs with 97 shared casts. It verified all 960 compute nodes in that reserve graph on DAYAN. Only Q needed conversion; K/V stayed F16. The full Qwen3.5 reserve graph has no eligible metadata-only chain, so it reports `hoisted=0` and retains all 97 casts. Synthetic BF16 and FP16 graph cases each report `hoisted=1` and exact output.

The full-model rewrite takes about 0.43-1.52 ms per reserve graph, while the complete scheduler reserve takes about 23-57 ms. A multi-minute pause after `warming up the model with an empty run` or after prompt tokenization is cmodel Device execution, not graph optimization. Use an empty prompt with `--no-warmup` for a reserve-only diagnostic:

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

With that fix, the full prompt smoke completed 1484 nodes as 707 Dayan slots with no fallback. The 4-hart cmodel processed the six-token prompt in 679734.28 ms. This is a functional simulator result, not a performance result.

A later run failed while launching `CPY final_output-12` because its 224-byte Host-owned launch descriptor was at `0xdc6f9ec0`. BL1 incorrectly used `CONFIG_RAM_BASE + CONFIG_RAM_SIZE` as the upper bound for launch descriptors, kernel entries, dcache flush ranges, and profile buffers. `CONFIG_RAM_SIZE` is the 64 MiB BL1 linker region, not the dynamic Host-managed Normal-memory arena. The cmodel launcher now keeps the default 64 MiB BL1 link region while configuring only QEMU DRAM and the HAL Normal-memory arena to 4096 MiB. BL1 retains alignment, Normal-memory lower-bound, and overflow checks for Host-owned ranges, while the Server HAL memory map supplies the dynamic upper bound. Fixed coherent and per-Hart URAM windows retain their bounded checks.

After correcting that boundary, the same six-token prompt completed 1472 nodes as 695 Dayan slots with no fallback or negative launch status. Prompt evaluation took 577940.11 ms in the 4-hart cmodel. The earlier `CPY final_output-12` failure did not recur.

## Fused slot trace and replay

`llama-dayan-trace` records the final `ggml_backend_dayan_graph_plan::slots` after graph fusion. It does not record the original GGML node list. A deduplication signature contains the fused slot kind, populated component-node indices, GGML operations, tensor types, dimensions, byte strides, view layouts, source layouts, raw operation parameters, and fused output layouts. Tensor names and addresses are excluded, so identical layer shapes share one entry. The full slot order is stored as references to the deduplicated entries.

Capture and replay use deterministic saved tokens, `n_batch=n_ubatch=128`, native FlashAttention, and all model layers on DAYAN. Each invocation executes two rounds. The cold round includes first kernel/module resolution; the warm round reuses the same model, context, Device buffers, slot resources, and module cache. `--replay` reconstructs the model graph and checks the fused signatures, counts, and order against the input trace. `--check` additionally downloads each fused slot output after its timer and compares exact cold/warm byte hashes. This is a deterministic replay check, not an independent CPU numerical reference, and the downloads perturb end-to-end timing.

The 2026-08-11 Qwen3.5-0.8B BF16 capture used a 4-hart cmodel with 4096 MiB Normal memory:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-dayan-trace \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  --trace dayan-prefill-128-trace.json \
  --prefill 128
```

The 1472 lowered nodes produced 695 fused slots and 50 unique signatures. Both rounds observed all 695 slots with identical counts and order. The cold end-to-end time was 6374.88 s and the warm time was 6137.72 s. Timed slots accounted for 6374.78 s and 6137.62 s respectively, leaving about 101 ms and 95 ms of graph, scheduler, and output overhead. These are cmodel software execution times, not RTL or hardware performance estimates.

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

The trace validates that Gated Delta Net and gated output projection dominate this cmodel run. It also exposes 115 independent copy slots totaling 261.16 s in the warm round, so copy elimination remains measurable work after the existing MatMul and normalization fusions.

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

This is not a CPU-free Host process. Model files and explicit inputs originate on the Host, CPU sampling reads the requested logits output, and the Host submits Device commands and small configs. The cmodel also executes `rvt_emulator` and RISC-V QEMU on the Host CPU, so high CPU usage from `rvt_emulator` or `dl-riscv64` represents simulated Dayan execution.

Pure Dayan graph execution does not pack or download intermediate tensors. Every non-metadata model graph node must be assigned to Dayan; context creation or graph execution fails before compute if the scheduler assigns one to another backend. Sampling remains on CPU and is not inserted into the Dayan model graph.

Use `GGML_SCHED_DEBUG=2` for scheduler diagnostics. Use this command during a run to separate simulator and Host usage:

```sh
pidstat -u -p "$(pgrep -d, -f 'llama-completion|rvt_emulator|dl-riscv64|dayan_server')" 1
```

## Current limitations

- The matrix kernel uses one consumer slice. Validate MAT issue ordering and VEC/store ownership before increasing `MATMUL_DAYAN_1C_SLICES`; VEC TCSR reconfiguration does not require software serialization because the control issue snapshots the configuration.
- FlashAttention requires Q/K/V to use one common BF16 or FP16 storage type. Mixed BF16/FP16 K/V inputs are not supported.
- The Qwen3.5-0.8B model is dense, so use `test-dayan-moe` to validate the independent MoE and fused expert paths.
- Full-model cmodel execution is slow. The six-token functional prompt smoke takes about 11 minutes and must not be used as an RTL performance result.

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
  test-dayan-qwen35-dense \
  test-dayan-moe
```

## Start the cmodel server

Keep this command running in terminal 1:

```sh
cd "$HOME/npu-firmware"
NPU_CMODEL_HARTS=1 \
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./run_cmodel_server.sh
```

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

FlashAttention consumes strided BF16 Device views directly. Its config carries row, head, and batch byte strides for Q, K, and V, and separates the physical KV-cache length from the effective causal prefix. Concat and Gated Delta Net likewise use full tensor byte strides rather than Host-packed contiguous buffers.

## Run the focused tests

Run these commands from the llama.cpp repository root in terminal 2:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-matmul
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-graph
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-flash-attention
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-gated-delta-net
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-qwen35-dense
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./build-dayan/bin/test-dayan-moe
```

The 2026-08-11 cmodel results are:

- BF16 MatMul, `M=1 N=8064 K=1024`: `max_abs=0`, `max_rel=0`.
- Graph suite: PASS for BF16/FP16 normalization, residual, activation cast, MatMul, SWIGLU/GEGLU, elementwise, layout, reduction, RoPE, gated output, and SSM convolution cases.
- FlashAttention permuted BF16, `q_len=3`, physical KV 64, effective KV 9: `max_abs=0.000228759`.
- FlashAttention decode, `q_len=1`, physical KV 256, effective KV 1: exact.
- Gated Delta Net, 16 heads and 16 tokens: `max_abs=3.72529e-09`.
- Four-layer Qwen3.5 dense BF16 graph: `max_abs=0.00158247`, logits `max_abs=0.00138266`.
- Four-layer Qwen3.5 dense FP16 graph: `max_abs=0.0078125`, logits `max_abs=0.00141311`.
- MoE with 512 experts, top-4, and two tokens: fused `max_abs=0.00225067`, down `max_abs=0.0321879`, final output `max_abs=0.000781553`.

`test-dayan-graph` contains a negative fan-out case. Its expected `unsupported op SWIGLU` message proves that the graph compiler did not fuse a GLU intermediate with another consumer. The test must continue and finish with `test-dayan-graph: PASS`.

## Run Qwen3.5 on Dayan

Device-resident full-model inference requires the Dayan HAL Normal-memory region and the cmodel shared-memory backing to exceed the model arena plus graph, KV-cache, firmware, and kernel allocations. Qwen3.5-0.8B BF16 currently requests a 1,505,783,040-byte model arena (`1436.03 MiB`) before graph buffers.

The stock cmodel server exposes only a 64 MiB HAL Normal-memory arena, and this container mounts `/dev/shm` with a 64 MiB limit. The full-model command therefore cannot run in this environment after Device residency is enabled. With the stock HAL it fails the model arena allocation with `-ENOMEM`. A diagnostic 4 GiB HAL window allows the arena allocation, but the emulator then exits during the first persistent weight upload because its 4 GiB sparse DRAM file has only 64 MiB of tmpfs backing.

Configure both limits before running the full model. Do not restore Host streaming to make the model fit. After providing sufficient Device memory, run:

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
  -n 2 \
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

- The matrix kernel uses one consumer slice. Three fused consumer slices can deadlock because they configure shared VEC TCSRs. Keep `MATMUL_DAYAN_1C_SLICES=1` until MAT issue ordering and VEC/store ownership are separated.
- Full Qwen3.5-0.8B BF16 cmodel inference needs more than the stock 64 MiB HAL Normal-memory window and 64 MiB `/dev/shm` mount used by this container.
- FlashAttention consumes BF16 Q/K/V. FP16 projection flows convert Q/K/V to BF16 at the FlashAttention boundary.
- The Qwen3.5-0.8B model is dense, so use `test-dayan-moe` to validate the independent MoE and fused expert paths.

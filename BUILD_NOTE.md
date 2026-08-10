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

## Run the focused tests

Run these commands from the llama.cpp repository root in terminal 2:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/test-dayan-matmul

DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/test-dayan-graph

DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/test-dayan-moe
```

The current cmodel results are:

- BF16 matrix multiplication, `M=1 N=8064 K=1024`: `max_abs=0`, `max_rel=0`.
- Qwen3.5 BF16 SWIGLU, `M=8 N=3584 K=1024`: `max_abs=0`, `max_rel=0`.
- MoE with 512 experts, top-4, and two tokens: fused and down errors are zero; final aggregation `max_abs=1.74623e-10`.

`test-dayan-graph` contains a negative fan-out case. Its expected `unsupported op SWIGLU` message proves that the graph compiler did not fuse a GLU intermediate with another consumer. The test must continue and finish with `test-dayan-graph: PASS`.

## Run Qwen3.5 on Dayan

Run from the llama.cpp repository root in terminal 2:

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

This command was verified with the Qwen3.5 0.8B BF16 model. It runs two eval calls and produces two sampled tokens. With seed 1234, the verified output is `A simplest`.

`-dev DAYAN` selects the Dayan device explicitly. `-ngl all` places all model layer slots on Dayan. `-fit off` is required because the prototype reports zero device memory. With automatic fit enabled, current llama.cpp can select CPU even though `--list-devices` reports DAYAN. A fast run with no `ggml_backend_dayan_*` messages is a CPU run and is not a valid backend test.

The backend requires Flash Attention and Gated Delta Net without CPU fallback. Model weights remain in their original BF16 layout; matrix inputs are converted to the weight type by the graph rewrite.

## Validate the full run

The command must exit with status zero. Audit the log with:

```sh
rg 'using device DAYAN|assigned to device DAYAN|verified .* DAYAN|lowered .* slots|unsupported|fallback|status=-|trap|timeout|eval time' \
  /tmp/dayan-qwen35.log
```

The current two-eval result has these required markers:

- All 26 model layer slots are assigned to DAYAN.
- Each checked graph prints `llama_graph_is_dayan_only: verified 954 compute nodes on DAYAN`.
- Each eval prints `lowered 1466 nodes to 683 slots`.
- Each eval reports `paired=6`, `fused=78`, `attention=6`, `delta=18`, `gated=6`, and `activations=91`.
- The final performance summary reports `2 runs`.
- The log contains no `unsupported`, fallback, negative status, trap, or timeout message.

The lowering summary is printed only after every execution slot returns, so two summaries prove that both 683-slot graphs completed.

## Expected CPU usage

This is not a CPU-free host process. The cmodel executes `rvt_emulator` and RISC-V QEMU on the host CPU, so high CPU usage from `rvt_emulator` or `dl-riscv64` is expected and represents simulated Dayan execution.

The backend also uses host CPU buffers, packs tensors on the host, and transfers operator inputs and outputs. These tasks are runtime orchestration rather than ggml numerical operators. Every non-metadata model graph node must be assigned to Dayan; context creation or graph execution fails before compute if the scheduler assigns one to another backend. Zero-sized recurrent-state operations are assigned to Dayan and execute as no-ops.

Sampling remains on the CPU. Backend sampling is disabled for a Dayan model so its operations are not inserted into the Dayan model graph. Contiguous F32 residual adds and RMSNorm paths run on Dayan. The graph compiler also fuses `RMSNorm + scale + BF16/FP16 quantization + MatMul`, including the residual-add variant, into one Dayan kernel.

Use `GGML_SCHED_DEBUG=2` for additional scheduler diagnostics. A valid run prints `llama_graph_is_dayan_only: verified ... compute nodes on DAYAN` and has no CPU split:

```sh
GGML_SCHED_DEBUG=2 \
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-completion \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  -dev DAYAN -ngl all -fit off \
  --no-warmup -no-cnv -v --log-colors off \
  -c 256 -b 8 -ub 8 -s 1234 -n 2 -p "A"
```

Use this command during a run to separate simulator and host usage:

```sh
pidstat -u -p "$(pgrep -d, -f 'llama-completion|rvt_emulator|dl-riscv64|dayan_server')" 1
```

Dayan execution is also visible in the client log through messages such as `ggml_backend_dayan_mul_mat` and `ggml_backend_dayan_graph_compute`.

## Current limitations

The matrix kernel currently uses one consumer slice. Three concurrent slice consumers can deadlock non-deterministically when they configure the shared VEC TCSRs for a fused SWIGLU epilogue. Keep `MATMUL_DAYAN_1C_SLICES=1` until MAT issue ordering and VEC/store ownership are separated into ordered MAT workers and one epilogue/store worker.

The Qwen3.5 0.8B model is dense, so its full inference log reports `moe=0`. Use `test-dayan-moe` to validate the independent MoE kernels and fused expert path.

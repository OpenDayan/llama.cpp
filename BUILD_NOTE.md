# Dayan Minimal Build and Run

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

cmake --build build-dayan -j4 --target llama-completion
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

## Run Qwen3.5 on Dayan

Run from the llama.cpp repository root in terminal 2:

```sh
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-completion \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  -dev DAYAN \
  -ngl 99 \
  -fit off \
  -fa on \
  --no-repack \
  --no-perf \
  --no-warmup \
  --simple-io \
  --no-conversation \
  -lv 1 \
  -c 256 \
  -b 2 \
  -ub 2 \
  -n 1 \
  -p "Hi"
```

This is the minimum cmodel smoke command verified with the Qwen3.5 0.8B BF16 model. The `Hi` prompt is two tokens for this model, so the logical and physical batch sizes are both set to 2. Increase `-n` to generate more tokens after the smoke test passes.

`-dev DAYAN` selects the Dayan device explicitly. `-ngl 99` places all model layer slots on Dayan. `-fa on` requires Flash Attention, and the backend also requires Gated Delta Net without CPU fallback. `--no-repack` keeps the original BF16 weight layout expected by the current Dayan kernels.

The generic argument parser may print `no usable GPU found` because Dayan is registered as an accelerator instead of a GPU. The explicit `-dev DAYAN` selection still assigns the model layers to Dayan.

## Expected CPU usage

This is not a CPU-free host process. The cmodel executes `rvt_emulator` and RISC-V QEMU on the host CPU, so high CPU usage from `rvt_emulator` or `dl-riscv64` is expected and represents simulated Dayan execution.

The backend also uses host CPU buffers, packs tensors on the host, and transfers operator inputs and outputs. These tasks are runtime orchestration rather than ggml numerical operators. Every non-metadata model graph node must be assigned to Dayan; context creation or graph execution fails before compute if the scheduler assigns one to another backend. Zero-sized recurrent-state operations are assigned to Dayan and execute as no-ops.

Sampling remains on the CPU. Backend sampling is disabled for a Dayan model so its operations are not inserted into the Dayan model graph. Contiguous F32 residual adds and RMSNorm paths run on Dayan. The graph compiler also fuses `RMSNorm + scale + BF16/FP16 quantization + MatMul`, including the residual-add variant, into one Dayan kernel.

Use `-v` together with `GGML_SCHED_DEBUG=2` to audit placement. A valid run prints `llama_graph_is_dayan_only: verified ... compute nodes on DAYAN` and has no CPU split:

```sh
GGML_SCHED_DEBUG=2 \
DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock \
./build-dayan/bin/llama-completion \
  -m /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  -dev DAYAN -ngl 99 -fit off -fa on --no-repack \
  --no-perf --no-warmup --simple-io --no-conversation \
  -v -c 256 -b 2 -ub 2 -n 1 -p "Hi"
```

Use this command during a run to separate simulator and host usage:

```sh
pidstat -u -p "$(pgrep -d, -f 'llama-completion|rvt_emulator|dl-riscv64|dayan_server')" 1
```

Dayan execution is also visible in the client log through messages such as `ggml_backend_dayan_mul_mat` and `ggml_backend_dayan_graph_compute`.

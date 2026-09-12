# Limited operator profiling

`limited-perf.py` measures a small set of existing Dayan kernels and estimates Qwen3.5-0.8B BF16 inference cost. It connects to an existing `dayan_server` and reads the JSON output of an existing firmware `dyperf`. It does not start or stop TestCluster, cmodel, the server, or a profiler. No kernel contains custom performance instrumentation.

Build the host client from the repository root:

```sh
cmake -S . -B build-dayan -DGGML_DAYAN=ON -DDAYAN_ROOT="$HOME/npu-firmware"
cmake --build build-dayan --target llama-dayan-operator-bench -j4
python3 tools/dayan-trace/limited-perf.py plan
```

The default plan has 16 cases: eight operators for prefill with M=1024 and the same eight for decode with M=1. Each runs once, with no hidden warmup or shape sweep. A four-core server with core mask `0xf` is required. Run one operator client at a time; leave TestCluster advancing continuously and waveform recording disabled.

| Sample | Shape or parameters | Why it is measured |
| --- | --- | --- |
| Fused FFN gate/up | M x 3584 x 1024, BF16 | Dedicated combined gate/up B tiles, one MAT per K tile, SwiGLU, and BF16 conversion |
| FFN down | M x 1024 x 3584, BF16 inputs, F32 output | No GLU activation; subsequent residual/norm processing is sampled separately |
| Input projection | M x 6144 x 1024 | Samples the large linear-attention projection; supplies an N-scaled proxy for the other input projections and vocabulary output |
| Output projection | M x 1024 x 2048 | Samples the other recurring projection shape |
| Gated DeltaNet | 16 heads, head dimension 128, M tokens | Stateful linear attention, including one final state snapshot |
| SSM Conv + SiLU | 6144 channels, kernel width 4, M tokens | The linear-attention convolution slot |
| FlashAttention | 8 query heads, 2 KV heads, head dimension 256 | Causal prefill or decode with a populated KV cache |
| Residual + RMSNorm + scale + BF16 | M x 1024 | Repeated memory/vector work outside MatMul |

The cases follow the backend dispatch: M<=512 uses the four-core inference entry and M>512 uses the general entry. MatMul uses N sharding for inference and a 2x2 layout for larger M. Both FFN entries use the dedicated pipeline with N sharding, combined gate/up B tiles, and exact M-row A tiles when M<32. Set `--prefill` to 1..1024 to sample another M; a single run has only one prefill size. Decode uses one token with a prefix of `--context` tokens (default 1024), hence KV length 1025. Inputs use deterministic constants with analytical references; every output element is checked, including the DeltaNet state. This is a measurement smoke check; existing numerical operator tests cover varied data and layouts.

## Use the existing server and profiler

Pass the actual file selected by the already-running `dyperf -o`. It must be capturing before the first operator launch. Proc state tracing is sufficient; `--pcu` is unnecessary.

Configure the server with `--normal-memory-size-mib 256` or more for the default plan, within the simulator's DRAM capacity. The server's default Normal DRAM budget is 64 MiB; firmware reservations, tensor buffers, and dyperf's default 16 MiB buffer share this budget when the MMU is disabled. The current four-core startup reserves about 8.63 MiB for firmware, 16 MiB for kernel text, and 4 MiB for Atomic Cache backing. Prefill FFN needs another 23 MiB of tensor data plus its config. Together with dyperf this exceeds 64 MiB, so buffer allocation fails with `-12` (`ENOMEM`) before any kernel launch. This is separate from the 1.5 MiB URAM budget per core.

When restarting the server after a firmware update, rebuild the Device firmware, API ELF, host driver, and profiler together, then relink the benchmark:

```sh
make -C "$HOME/npu-firmware" -j4 device host-driver dycc dyperf
cmake --build build-dayan --target llama-dayan-operator-bench -j4
# After the simulator is ready, use these arguments when starting the server.
"$HOME/npu-firmware/run_server.sh" --normal-memory-size-mib 256
```

The run script connects to the existing socket and does not change the server's memory budget or the active profiler. Errors include the failing API and transfer/allocation size in the case log and reports. An empty `samples` array means setup failed before calling the kernel; a failed launch has a sample with its returned status.

```sh
python3 tools/dayan-trace/limited-perf.py run \
    --socket /tmp/dayan-server.sock \
    --perf-trace /tmp/dayan-perf.json \
    --output build-dayan/rtl-perf
```

The command writes progress and the final report to stdout. The output directory contains:

- `report.md`, `report.json`, and `operators.csv`: per-shape measurements and predictions, with formulas and limitations.
- `manifest.json`: selected cases, dimensions, repetition counts, executable/backend hashes, simulator label, and socket.
- `*.case.json`, `*.result.json`, and `*.log`: inputs, raw wall measurements, unique profiler task names, reference errors, commands, and client output. Results are saved after each case.

Firmware dyperf buffers records on Device and in its output stream. It can keep a short run's final records until collection ends. A live profiler therefore does not guarantee that all cycle records are readable. The script leaves it running, reports missing cycles as `N/A`, and exits with code 2. Once the existing collection has ended and dyperf has written its final records, regenerate the report without executing any operators:

```sh
python3 tools/dayan-trace/limited-perf.py report --output build-dayan/rtl-perf
```

Exit status 0 means all selected cases passed and have complete cycle records; 1 means a run/configuration/reference failure; 2 means successful operators with incomplete cycle data. A subset can exit 0 even if it lacks the samples required for end-to-end prediction: the report lists those dependencies and prints `N/A` for that prediction.

## Limit expensive RTL work

Inspect a proposed change without contacting the server:

```sh
python3 tools/dayan-trace/limited-perf.py plan --prefill 128 --context 512
```

Run just decode FFN and down projection:

```sh
python3 tools/dayan-trace/limited-perf.py run \
    --perf-trace /tmp/dayan-perf.json --output build-dayan/rtl-ffn \
    --case decode-ffn --case decode-down
```

Use `--warmup 1 --repeats 1` for one unmeasured warmup and one measured launch per case. The profiler still records the warmup, but the report excludes it by task name. With multiple measured repetitions, the report gives the median and retains each sample in JSON. Defaults favor minimum RTL runtime; without warmup, the wall measurement includes first-use module loading.

Each case has a wall deadline of 24 hours, covering setup, uploads, execution, and teardown; change it with `--timeout SECONDS`. A failure stops further launches. After a timeout or interruption, check the device/server state before resuming: terminating a host client does not prove the Device has stopped executing it.

Resume with the original options plus `--resume`. It skips completed cases and rejects changed shapes, repetition counts, socket, trace path, or executable/backend hashes. Keep the same trace file: restarting dyperf with `-o` on that file truncates earlier cycle records. Use `report` rather than `run --resume` when only the report needs updating. Retries get new task names, so incomplete old launches cannot be mistaken for a new result.

## Measurement and prediction meanings

The launch wall time is a real host `steady_clock` duration around the synchronous DAPI kernel call. It includes lazy module loading, submission, Device execution, profiler overhead, and completion waiting. Input/config/weight upload and output validation are outside this interval. A separate process wall time includes all host-client setup and data transfer. Neither measurement includes simulator or server startup.

Cycles come only from firmware dyperf proc-state records. For each unique launch name the script requires four complete root lifetimes, then computes:

```text
launch_cycles = max(root WAIT_EXIT begin) - min(root first RUNNING begin)
```

It uses raw 48-bit counters and handles wraparound for launches shorter than half the counter period. All cores must have a shared counter origin. This is the Device execution span including scheduler gaps and child work, after the first root starts and before the final root's exit RPC. It does not add overlapping core times, child times, or idle time until the profiler closes. Incomplete launches and non-positive counters never become zero-cycle measurements.

The prediction assumes 24 inference blocks (18 linear-attention and 6 full-attention blocks), hidden size 1024, FFN width 3584, vocabulary 248320, batch 1, resident weights, and only the last token's logits. The GGUF's extra MTP block is excluded. The report exposes every multiplicity and proxy factor. In particular, the huge vocabulary projection is extrapolated from M=1, N=6144, K=1024 by `248320 / 6144`; it is not simulated at full size. Other input projections share an N-scaled proxy. Pair fusion and attention epilogues can change that estimate.

For wall time and cycles separately, the report sums `sample * factor` and displays both that subtotal and a scenario with `--overhead-percent` added (default 20%). This is an explicit, uncalibrated allowance for omitted casts/layouts, RoPE, small gates, KV/state copies, host scheduling, and sampling; it is not a measured overhead or confidence interval. The timing of those operations can differ substantially from the allowance. Cold module-loading cost is repeated by this formula unless warmup is enabled. The preset is a coarse estimate, not a full-model benchmark or a general model profiler. Model loading, tokenization, and simulator startup are excluded.

`report --overhead-percent 0` shows the sampled subtotal without allowance. `--clock-hz HZ` optionally adds ideal Device seconds in JSON from predicted cycles; it does not infer cycles from wall time. TestCluster host wall predictions estimate RTL simulation runtime on the same host and settings, not hardware inference latency. cmodel measurements validate this workflow and describe cmodel runtime only; its counters are not RTL performance evidence.

## Validate with cmodel

With a four-core cmodel server and dyperf already ready, use the same command with `--simulator cmodel`, the cmodel socket, and a separate output directory. The simulator option labels the results; it does not select or start a simulator.

```sh
python3 tools/dayan-trace/limited-perf.py run \
    --simulator cmodel --socket /tmp/dayan-cmodel.sock \
    --perf-trace /tmp/dayan-cmodel-perf.json \
    --output build-dayan/cmodel-perf
python3 tools/dayan-trace/limited-perf.py report --output build-dayan/cmodel-perf
python3 -m unittest discover -s tools/dayan-trace -p 'test_*.py'
```

The parser tests cover overlapping cores, counter wrap, partial files, missing exits, zero counters, child exclusion, and prediction dependencies. No TestCluster run is needed for these checks.

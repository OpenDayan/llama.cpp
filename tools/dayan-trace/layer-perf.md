# Single-layer inference profiling

`layer-perf.py` runs one complete Qwen3.5 dense or MoE layer with mega enabled. It replaces the operator-by-operator performance workflow for this use case. `limited-perf.py` remains available for operator correctness checks.

`prepare` copies the selected block from a GGUF, renames it to block 0, and retains its weights, dimensions and quantization. It also keeps the tokenizer, embedding and output head needed by `llama_decode`. It supports BF16/MXFP4 dense layers and MXFP4 MoE experts with BF16 attention. Merge split GGUFs before use. The resulting one-layer model receives embeddings directly; its hidden states and generated text do not reproduce the source model's intermediate layer inputs or language quality.

`run` enables mega for the layer body and checks exactly one mega launch per inference step. Embedding and the output head still launch separate kernels. In the server log, `[kernel] lp-prefill started` is the layer mega; `lp-setup` is decode's prompt setup and `lp-d...` identifies decode steps. A preceding `get-rows` and following output normalization, row selection, conversion or matrix kernels are outside the layer mega. Four `proc-create` RPCs under one launch create its four core tasks. After capture and CPU comparison finish, `run` automatically invokes the independent report script. The offline layer timing covers only the mega body. `cmp-ops` explicitly disables mega for per-slot CPU comparison.

Build the client and extract a layer once:

```sh
cmake -S . -B build-dayan-public -DGGML_DAYAN=ON -DDAYAN_ROOT="$HOME/npu-firmware"
cmake --build build-dayan-public --target llama-dayan-layer-bench -j8

python3 tools/dayan-trace/layer-perf.py prepare \
    --model /share/models/Qwen3.5-0.8B-MXFP4/Qwen3.5-0.8B-MXFP4.gguf \
    --layer 0 --output build/dayan-layer/linear
```

`--layer` is a zero-based source layer index. Select a full-attention block, such as layer 3 for the standard Qwen3.5-0.8B cycle, to profile FlashAttention instead of gated delta net. The extractor detects the layer type from its tensors. MoE uses the same commands with its source GGUF. Extraction uses the local `gguf-py` package and NumPy; reporting uses only Python's standard library.

## MoE layers

Qwen3.5/3.6 MoE GGUFs use architecture `qwen35moe`. Prepare and run their linear and full-attention layers together:

```sh
python3 tools/dayan-trace/layer-perf.py prepare \
    --model /share/models/Qwen3.6-35B-A3B-MXFP4_MOE_BF16.gguf \
    --layer 0 3 --output build-dayan/layers-moe

python3 tools/dayan-trace/layer-perf.py run \
    --prepared build-dayan/layers-moe --mode prefill decode \
    --input-len 1024 --output-len 32 --cmp \
    --output build-dayan/perf-moe

python3 tools/dayan-trace/layer-perf-report.py \
    --run build-dayan/perf-moe --mode prefill decode
```

The MoE mega graph includes attention, router, softmax, TopK, routed experts, shared experts and the final residual. Each slot has its own cycles and layer percentage in the report. The report also identifies the graph, weight precision, expert count and experts selected per token. Weights and expert metadata are retained during extraction, including the F32 routers, BF16 attention/shared experts and MXFP4 routed experts. This path requires MXFP4 expert weights; it does not select the dense FFN graph. `run --cmp` uses the CPU baseline, not standalone Dayan.

For a cmodel check, use `--input-len 4 --output-len 2` and omit `--pcu`. The random two-layer MoE fixture in [MoE validation](qwen35-moe.md#cmodel-and-fixture) can be used with `--layer 0 1`. It exercises both attention kinds with a 512-token vocabulary; it does not measure trained-model quality. The server remains externally managed. Do not start cmodel while another test owns its device endpoints.

## Dense layers and batch modes

Pass multiple indices to prepare independent single-layer models in one command, for MXFP4 or BF16:

```sh
python3 tools/dayan-trace/layer-perf.py prepare \
    --model /share/models/Qwen3.5-0.8B-MXFP4/Qwen3.5-0.8B-MXFP4.gguf \
    --layer 0 3 --output build/dayan-layer/selected-mxfp4

python3 tools/dayan-trace/layer-perf.py prepare \
    --model /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
    --layer 0 3 --output build/dayan-layer/selected-bf16
```

Each output directory contains `layer-0/` and `layer-3/`, each with its own `layer.gguf` and `layer.json`. Each model still contains one block. With a single index, files stay directly under `--output`; omitting `prepare --layer` selects layer 0. Duplicate or out-of-range indices and existing prepared outputs are rejected before extraction.

Start a four-core `dayan_server` connected to the desired hardware, TestCluster or cmodel instance. Set `DAYAN_SERVER_SOCKET` if its socket differs from `/tmp/dayan-server.sock`. Then run selected layers together:

```sh
python3 tools/dayan-trace/layer-perf.py run \
    --prepared build/dayan-layer/selected-mxfp4 --layer 0 3 --input-len 1024 --cmp \
    --output build/dayan-layer/batch-mxfp4

python3 tools/dayan-trace/layer-perf.py run \
    --prepared build/dayan-layer/selected-bf16 --layer 0 3 --input-len 1024 --cmp \
    --output build/dayan-layer/batch-bf16

python3 tools/dayan-trace/layer-perf-report.py --run build/dayan-layer/batch-mxfp4
python3 tools/dayan-trace/layer-perf-report.py --run build/dayan-layer/batch-bf16
```

Prefill is the default mode. Omit `run --layer` to select every prepared layer in numeric order. Explicit indices run in the given order. All selections are checked before starting the profiler. Layers run sequentially, each with its own `layer-N/` output directory and CPU comparison. The batch records every case's status in `batch.json` and returns nonzero if any case or report failed. CPU comparison failures allow the next case to run. Inference or profiler failures stop the batch with remaining cases marked `not-run`, since device work may still be active. Reports are generated after the batch finishes, including available timing for failed CPU comparisons. The report accepts the batch directory and returns nonzero for failed or incomplete batches.

Measure both modes for every selected layer in one invocation:

```sh
python3 tools/dayan-trace/layer-perf.py run \
    --prepared build-dayan/layers-mxfp4 --layer 0 3 \
    --mode prefill decode --input-len 1024 --output-len 32 \
    --cmp --pcu --output build-dayan/perf-mxfp4-both
```

Each layer runs its selected modes in argument order, with independent client processes, traces and CPU comparisons. Decode rebuilds its prompt state; it does not reuse the measured prefill process. `--output-len` applies only to decode and defaults to one step; prefill always uses zero output steps. Omit `--pcu` for cmodel. The same command works with the BF16 prepared directory.

Multiple modes use this layout, even when only one source layer is selected:

```text
perf-mxfp4-both/
  batch.json
  report.md
  report.json
  summary.csv
  layer-0/
    prefill/  # perf.json, run.json, report.md, report.json, slots.csv, ...
    decode/
  layer-3/
    prefill/
    decode/
```

The top-level report has one row per layer and mode, including step count, total layer cycles, median/min/max cycles per step and CPU comparison status. Detailed reports retain slot cycles and percentages for that mode. Decode totals exclude prompt setup. Single-mode runs retain their existing directory layout, and old records remain readable.

`run` connects to the external server without a `--simulator` argument. The client defaults to `DAYAN_LAYER_CLIENT` or `build-dayan-public/bin/llama-dayan-layer-bench`; override it with `--client`. The profiler defaults to `$DAYAN_ROOT/build/host/dyperf/dyperf` (`DAYAN_ROOT` defaults to `~/npu-firmware`). Pass `--dyperf /path/to/dyperf` to choose another profiler; it is launched and stopped automatically. `--socket` overrides `DAYAN_SERVER_SOCKET`.

PCU sampling is off by default. Add `--pcu` to `run` on a backend that supports PCU; this passes `--pcu-interval 0` to dyperf for continuous sampling. cmodel does not support PCU, so omit `--pcu` there. Proc-state tracing and offline slot timing work without PCU.

The backend waits indefinitely by default. This applies to kernel and backend transfer waits and is logged as `4294967295 ms (infinite)` when the backend opens. Set `GGML_DAYAN_TIMEOUT_MS` to a positive number of milliseconds before launching the client to request a finite timeout; unset it or use `4294967295` for unlimited waits. After status `-110`, check the external server: the timed-out task may still be running and holding resources, so finish or reset it before retrying. The runner does not reset the simulator or server.

For a single prepared model, run prefill:

```sh
python3 tools/dayan-trace/layer-perf.py run \
    --prepared build/dayan-layer/linear --mode prefill --input-len 1024 \
    --output build/dayan-layer/prefill-1024

python3 tools/dayan-trace/layer-perf-report.py --run build/dayan-layer/prefill-1024
```

Or build the prefix and measure consecutive decode steps:

```sh
python3 tools/dayan-trace/layer-perf.py run \
    --prepared build/dayan-layer/linear --mode decode \
    --input-len 1024 --output-len 32 \
    --output build/dayan-layer/decode-1024-32

python3 tools/dayan-trace/layer-perf-report.py \
    --run build/dayan-layer/decode-1024-32 --clock-hz 1000000000
```

- `input-len` is the synthetic prompt token count. The prompt is executed normally, including KV and recurrent-state updates. It is excluded from the decode report.
- `output-len` is the number of measured single-token decode calls after the prompt. Each consumes the previous call's greedy output token; EOS does not stop the requested run. Prefill requires zero output steps, which is its default. Decode defaults to one step.
- Each inference call uses one microbatch and must launch mega exactly once. Splitting or standalone fallback fails the run. Use an input length that fits the selected layer and available device memory.
- `run` owns only its `dyperf --follow` process and client. It uses an existing server and waits for the profiler to close `perf.json`. It does not parse timings during inference or install per-slot host callbacks. With `--pcu`, PCU metrics are also included in `perf.json`, with no delay between sampling rounds. `--client` and `--dyperf` override executable paths.
- Each run needs an empty output directory. `run.log`, `case.json`, `run.json`, `command.json` and `perf.json` preserve the inputs and execution record. No warmup is performed; module loading and other cold host costs remain in the separate host wall time.

## CPU comparison

Add `--cmp` to `run` for an independent GGML CPU baseline. After dyperf closes the trace, a separate process keeps only the CPU backend registered and loads the same single-layer GGUF with zero GPU layers, KV offload disabled and operation offload disabled. Removing the other backends also prevents implicit ACCEL buffer selection. It replays the same prompt and the tokens selected by the Dayan run, so different greedy predictions cannot change subsequent inputs. The reference uses the original stored weights and the CPU backend's own arithmetic and activation quantization.

```sh
python3 tools/dayan-trace/layer-perf.py run \
    --prepared build/dayan-layer/linear --mode decode \
    --input-len 4 --output-len 2 --cmp \
    --output build/dayan-layer/decode-cmp

python3 tools/dayan-trace/layer-perf-report.py --run build/dayan-layer/decode-cmp
```

Comparison covers every vocabulary logit for the prompt's final token and every decode step. Decode prompt setup is checked even though it is outside the timing summary. It does not download or compare intermediate slots. `--cmp-profile auto` selects the following defaults:

| Weights | Profile | Required for each step |
| --- | --- | --- |
| BF16 | `strict` | Every logit satisfies `abs(Dayan - CPU) <= 0.03 + 0.03 * abs(CPU)` |
| MXFP4 dense or MoE | `mxfp4` | Relative L2 error <= 0.35 and cosine similarity >= 0.93 |

Relative L2 is `norm(Dayan - CPU) / norm(CPU)`. The MXFP4 profile allows activation quantization error while retaining every elementwise mismatch, maximum error and greedy token in the report. Its mismatch count is diagnostic; pass/fail uses both aggregate limits separately for every step, including decode setup. Nonfinite logits fail in either profile. This is an end-to-end quantization check, not a replacement for operator correctness or language-quality evaluation. Large deviations can still fail the relaxed profile.

Override the MXFP4 limits with `--cmp-max-rel-l2` and `--cmp-min-cosine`. Use `--cmp-profile strict` to require elementwise agreement, with `--cmp-atol` and `--cmp-rtol` overriding the default 0.03 tolerances. Those two tolerances only control diagnostic mismatch counts under the `mxfp4` profile. Resolved profiles and thresholds are saved in `case.json` and the CPU report. Existing prepared MoE/MXFP4 graph metadata also selects the MXFP4 profile, without preparing again. Older run results retain their recorded criteria when reported offline. BF16 defaults and `cmp-ops` thresholds are unchanged.

`cpu-reference.json` and `cpu-reference.log` store the CPU result. `run.json` and the offline report include per-step and overall maximum absolute/relative error, relative L2 error, checked logit count and mismatch count. `run.json.logits.f32` stores the Dayan outputs as native float32 rows for this local comparison. The CPU process and comparisons run after capture; output saving occurs outside each inference timer. A failed comparison returns nonzero from both `run` and the report command, while preserving the trace and slot timings for diagnosis. Without `--cmp`, the run is an execution/finite-output check only.

## CPU operator comparison

`cmp-ops` runs a separate diagnostic pass over the prepared model. It supports the same `--prepared`, multi-layer selection, `--mode prefill decode`, lengths, socket and client options as `run`. It does not use dyperf or PCU and does not generate performance reports.

```sh
python3 tools/dayan-trace/layer-perf.py cmp-ops \
    --prepared build/dayan-layer/selected-mxfp4 --layer 0 3 \
    --mode prefill --input-len 128 --output build/dayan-layer/operators-prefill

python3 tools/dayan-trace/layer-perf.py cmp-ops \
    --prepared build/dayan-layer/selected-mxfp4 --layer 0 3 \
    --mode decode --input-len 4 --output-len 2 --output build/dayan-layer/operators-decode
```

The backend's slot callbacks capture actual stored inputs before each slot runs, including inputs of in-place operations. The CPU backend evaluates floating-point arithmetic in F32, with quantized inputs dequantized from their stored values. Integer indices retain their original type. Output conversion uses GGML CPU BF16/FP16 rounding or its MXFP4 reference quantizer. SET_ROWS also snapshots unmodified destination rows. Each slot starts from Dayan inputs, so upstream numerical differences do not accumulate in its local comparison.

Mega is disabled in this pass. Necessary operator fusions are retained, so fused SwiGLU and other fused groups are checked at their materialized outputs. `fused_nodes` lists the covered graph nodes; elided intermediates are not reported as independently checked. This pass covers executable slots in a single extracted layer, embedding and output head. It does not establish full-model quality or validate every mega intermediate lifetime.

`operators.json` records slot kind, output name/type/shape, mismatch count, maximum absolute/relative error, relative L2 error and first mismatch. `operators.log` prints a result for each output. Both prefill setup and requested decode steps are checked. The default per-value tolerance is `1e-4 + 1e-3 * abs(CPU)`, configurable with `--op-atol` and `--op-rtol`; integer outputs require exact equality. Matching infinities are allowed for masks, while NaNs or differing infinities fail. Comparison failures are preserved and return nonzero; missing outputs or unsupported references fail instead of being skipped.

Dayan's current BF16-to-MXFP4 instruction uses fixed truncation and a scale that avoids finite overflow. GGML CPU MXFP4 quantization uses nearest values and a different scale rule. The diagnostic uses the CPU rule and reports this difference as a failure; it does not silently substitute the hardware rule or increase tolerances. A successful matrix comparison after an MXFP4 cast only validates matrix arithmetic on the stored quantized inputs. MAT also discards the low three FP16 fraction bits, and the attention PV and chunked GDN products use TF32 operands. Their differences from full F32 arithmetic remain visible under the same strict thresholds.

## Offline report

The independent report script needs no device or model. It reads the completed trace and the slot schema saved by the client, which is compiled directly from the mega flow declarations. Adding or removing a mega step requires no analyzer table changes. `run` calls it automatically after capture; invoke it again to change report options without running inference.

All recorded modes are included by default. Use `--mode prefill decode` explicitly, or select only one mode:

```sh
python3 tools/dayan-trace/layer-perf-report.py \
    --run build-dayan/perf-mxfp4-both --mode prefill decode

python3 tools/dayan-trace/layer-perf-report.py \
    --run build-dayan/perf-mxfp4-both --mode decode \
    --output build-dayan/reports-mxfp4-decode
```

The filter applies to every source layer, and the return status reflects only the selected cases. A requested mode absent from the run is an error. To analyze one completed case while other cases are running, point `--run` at its `layer-N/prefill` or `layer-N/decode` directory and use a separate `--output`.

When captured with `--pcu`, open `perf.json` in Perfetto to inspect each core's PCU tracks: MAT utilization, VEC group/total IPC, Scalar IPC, LSU read/write bandwidth, and URAM read/write mean, variance and standard deviation. `PCU snapshot` events also retain the 112 raw counters. The trace metadata records `pcu_enabled: true` and `pcu_interval_ms: 0`. Slot reports use proc-state events and do not require PCU counters.

`report.md` shows the complete Transformer layer span per step, total measured layer cycles, and each slot's median/min/max cycles and percentage of layer cycles. `slots.csv` contains every slot for every decode step, with the growing context length and `percent_of_layer`. `report.json` also records each core's span and root RUNNING/wait-state durations. Incomplete traces, missing cores, missing slots and missing root exits fail analysis.

Percentages use cycles, not host time: each slot's total cycles divided by the total measured layer cycles. For several decode steps this is a ratio of sums, not a mean of percentages or a ratio of medians. Embedding, output normalization and lm_head are excluded from both the default report and its layer denominator.

A slot here is one named mega device step, including its following barriers. Optional no-op steps remain visible, and two device steps that reuse one host slot remain separate. Timing uses raw 48-bit counters: earliest step start to latest step end across four cores. Child-thread execution and concurrent cores are not added together. Slot spans can overlap, so their sum is not the layer duration. Root scheduling states do not measure MAT/VEC/LSU utilization.

The layer span covers the first named step through the final root's `WAIT_EXIT`, excluding embedding and output norm/projection. The launch span also includes root startup before its first rename. The host wall time covers the whole synchronous `llama_decode`, including those surrounding operations and profiling overhead. No full-model extrapolation is made.

Add `--include-outer` to also list independent embedding/output kernels and the mega launch in execution order, with `kernels.csv` and per-step `kernels` entries in `report.json`. This optional section uses a separate full-inference span, from the first embedding root `RUNNING` to the last output projection root entering `WAIT_EXIT`, including inter-kernel gaps. It shows each kernel's cycles and percentage of that span, plus the total launch cycles and gap cycles/percentage. It does not change Transformer layer totals or slot percentages. Child threads are excluded using their enclosing root spans; the layer client's launches are serial. Four-core launch sequences and the embedding/output boundaries must match. Older traces containing only mega steps show full inference as unavailable. Decode setup is excluded from measured totals.

Use `--output` to write reports outside the run directory, preserving the batch's layer/mode subdirectories. This leaves the source run and any existing reports untouched. The script never connects to the device. For example, while another layer is still running, analyze only a completed layer:

```sh
python3 tools/dayan-trace/layer-perf-report.py \
    --run build-dayan/perf-mxfp4-1006/layer-0 \
    --output build-dayan/reports-mxfp4-1006/layer-0
```

The original command without `--output` still writes reports into the run directory. Traces must be complete before analysis; no new inference or performance capture is needed.

For a cmodel smoke test, use its socket, for example with input length 4 and output length 2. New runs do not record a simulator label, and dyperf currently does not identify the counter source. Reports therefore show raw counter ticks by default and leave `performance_valid` false. For hardware or cycle-accurate RTL, pass the actual frequency to the report with `--clock-hz`, for example `--clock-hz 1000000000` for 1 GHz, to enable time conversion. Do not use this conversion for cmodel: its counters and wall times are not hardware performance measurements. Older records that explicitly identify cmodel still suppress time conversion.

```sh
python3 -m unittest discover -s tools/dayan-trace -p 'test_layer_perf.py'
```

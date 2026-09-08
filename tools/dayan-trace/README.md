# llama-dayan-trace

`llama-dayan-trace` records and replays fused Dayan graph slots for model prefill or single-token decode. It builds the complete model graph but executes each unique fused signature only once per round. The first round includes cold kernel loading and the second round measures the warm path. Duplicate slots are skipped and their recorded occurrence counts scale the measured latency to estimate full-model time.

```sh
mkdir -p dayan-trace
llama-dayan-trace -m model.gguf --trace dayan-trace/prefill.json --prefill 128
llama-dayan-trace -m model.gguf --trace dayan-trace/decode.json --prefill 128 --decode
llama-dayan-trace -m model.gguf --replay dayan-trace/prefill.json
```

Keep generated traces in the repository-root `dayan-trace/` directory. The directory is ignored by Git.

Decode mode clears model memory and builds the saved prefix graph before each measured single-token round. All prefix slots are skipped: this advances the model memory metadata needed for real decode shapes without running the prefix on Device. Prefix setup time is reported separately as `setup_us` and is excluded from decode slot and end-to-end times.

Use `--check` to download each fused slot output and compare exact output hashes between the two rounds. These downloads are outside the slot timers but add overhead between slots. This check validates deterministic replay; it is not an independent CPU numerical reference.

Skipped producers can leave later unique operators with synthetic or stale input data. Replay is intended for shape-driven timing. It does not run model inference and `--check` is not a numerical comparison against a CPU implementation.

The JSON stores each unique fused slot signature once. A signature includes the fused kind, component GGML operations, tensor types, shapes, strides, view layouts, source layouts, operation parameters, and fused outputs. The occurrence count is used with the measured mean latency to estimate the full traced phase operator time.

## Random Qwen3.5 MoE fixture

`create-qwen35-moe-fixture.py` creates a deterministic BF16 GGUF from the official Qwen3.5-35B-A3B configuration and tokenizer metadata from an existing GGUF. Four layers preserve one complete hybrid cycle: three linear-attention layers and one full-attention layer. Hidden widths, head dimensions, vocabulary, 256 experts, and Top-8 routing retain the source dimensions. The parameters are random; this fixture checks execution and numerical agreement, not language quality.

```sh
curl -L https://huggingface.co/Qwen/Qwen3.5-35B-A3B/resolve/main/config.json -o dayan-trace/qwen35-a3b-config.json
python3 tools/dayan-trace/create-qwen35-moe-fixture.py \
  --config dayan-trace/qwen35-a3b-config.json \
  --tokenizer-model /share/models/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf \
  --output dayan-trace/qwen35-a3b-random-4l-bf16.gguf --layers 4 --seed 1234
```

The four-layer GGUF is about 8.8 GB. Start cmodel with `NPU_CMODEL_MEMORY_MIB=16384` before loading it. The cmodel shared DRAM allocation and both scalar and tensor MMU bounds must support that size; increasing only the Runtime memory window is insufficient. Confirm that the frontend startup log reports 17179869184 shared DRAM bytes. Use `llama-graph-export` and `tools/graph-export/graph-export-html.py` for the complete compiled physical slot plan. A trace replay skips duplicate signatures and does not replace that graph or a full inference check.

## Top-8 benchmark

[topk-bench.cpp](topk-bench.cpp) compares Device RISC-V selection from DRAM with VEC radix selection and the URAM insertion baseline. See [topk-bench.md](topk-bench.md) for build commands, input limits, cache handling, and the Device cycle timer scope.

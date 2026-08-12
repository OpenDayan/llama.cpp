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

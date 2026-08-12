# llama-dayan-trace

`llama-dayan-trace` records and replays fused Dayan graph slots for one model prefill. It runs the same input twice. The first round includes cold kernel loading and the second round measures the warm path.

```sh
llama-dayan-trace -m model.gguf --trace trace.json --prefill 128
llama-dayan-trace -m model.gguf --replay trace.json
```

Use `--check` to download each fused slot output and compare exact output hashes between the two rounds. These downloads are outside the slot timers but add overhead between slots. This check validates deterministic replay; it is not an independent CPU numerical reference.

The JSON stores each unique fused slot signature once. A signature includes the fused kind, component GGML operations, tensor types, shapes, strides, view layouts, source layouts, operation parameters, and fused outputs. The occurrence count is used with the measured mean latency to estimate the full prefill operator time.

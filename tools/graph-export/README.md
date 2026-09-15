# llama-graph-export

`llama-graph-export` builds representative prefill and single-token decode graphs without running inference. It exports tensor names, operations, types, shapes, strides, buffer placement, flags, views, and source edges as JSON. For Dayan, it also compiles the graph into physical execution slots.

Each tensor includes numeric `op_id`, `type_id`, and the raw `op_params` integer array for reconstructing operator tests. These IDs and parameter layouts belong to the exporting build; use a matching build of `test-backend-ops` when converting them to its `--test-file` format.

```sh
llama-graph-export -m model.gguf -p "Hello" -o graph.json
python3 tools/graph-export/graph-export-html.py graph.json graph.html --graph decode
```

Open `graph.html` in a browser. By default, the generator selects one complete linear-attention Transformer Block and one complete full-attention Transformer Block and displays them in separate interactive panels. Each panel includes the full block range, including normalization, attention or state-space operators, output projection, FFN, and residual paths. Each main node is one Dayan physical slot, not one logical GGML operation. A slot shows its physical operator kind, host executor, launch name, device kernel, task mode, task dimensions, and fused logical members. Tensor type, shape, role, and backend are shown on edges. Weight and block boundary tensors remain separate nodes. The viewer supports node dragging, pan, zoom, search, layout reset, and tensor inspection.

Use `--blocks 0,3` to select two Transformer Blocks explicitly, or `--block 0` to display one block. `--layers` and `--layer` remain supported as aliases.

Use `--serve` to generate the page and start a local HTTP server:

```sh
python3 tools/graph-export/graph-export-html.py graph.json graph.html --blocks 0,3 --serve --port 8000
```

The server listens on `127.0.0.1` by default and serves only the generated page. Use `--host 0.0.0.0` to allow remote access, or `--port 0` to select a free port.

The HTML generator requires Python 3. The generated page loads Cytoscape.js and its Dagre layout plugin from jsDelivr. The prompt controls the prefill token count. If no prompt is given, the prefill graph uses the smaller of the context and micro-batch sizes. Normal model and backend options such as `-ngl`, `-ot`, `-c`, and `-ub` control graph construction and placement.

The logical graph comes from the staging graph reserve API. It reflects graph rewrites and scheduler backend assignments performed during graph splitting. The Dayan backend then runs its normal graph compiler and exports the resulting physical slot plan. The plan is compile-time data, so slot timing and runtime-only staging work are not included.

For a Dayan graph, start the cmodel server first and select the device explicitly:

```sh
cd ${DAYAN_ROOT}
NPU_CMODEL_HARTS=1 DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock ./run_cmodel_server.sh

DAYAN_SERVER_SOCKET=/tmp/dayan-server.sock llama-graph-export -dev DAYAN -m model.gguf -p "Hello" -o graph.json
```

The exporter uses allocation simulation, but Dayan backend initialization still requires a running Runtime server. `-dev DAYAN` is required so the model, graph scheduler, and physical slot compiler use the same Dayan device. The HTML generator rejects JSON without a Dayan physical plan.

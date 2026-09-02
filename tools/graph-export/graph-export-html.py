#!/usr/bin/env python3

import argparse
import http.server
import json
import pathlib
import re
import urllib.parse


BLOCK_NAME = re.compile(r"^blk\.(\d+)\.")


def graph_by_name(data, name):
    for graph in data["graphs"]:
        if graph["name"] == name:
            return graph
    names = ", ".join(graph["name"] for graph in data["graphs"])
    raise ValueError("graph not found: " + name + "; available: " + names)


def block_consumers(graph):
    nodes = {node["id"]: node for node in graph["nodes"]}
    result = {}
    for edge in graph["edges"]:
        source = nodes[edge["source"]]
        target = nodes[edge["target"]]
        match = BLOCK_NAME.match(source["name"])
        if match is None or target["order"] is None:
            continue
        layer = int(match.group(1))
        result.setdefault(layer, set()).add(target["order"])
    return {layer: sorted(orders) for layer, orders in result.items()}


def block_start(layer, consumers, operations):
    first_weight_order = consumers[layer][0]
    previous_layers = [candidate for candidate in consumers if candidate < layer]
    lower_bound = consumers[max(previous_layers)][-1] + 1 if previous_layers else 0
    norm_orders = [
        order for order, node in operations.items()
        if lower_bound <= order <= first_weight_order and "NORM" in node["op"]
    ]
    return max(norm_orders) if norm_orders else first_weight_order


def block_range(graph, layer):
    consumers = block_consumers(graph)
    if layer not in consumers:
        layers = ", ".join(str(value) for value in sorted(consumers))
        raise ValueError("layer not found: " + str(layer) + "; available: " + layers)

    operations = {
        node["order"]: node for node in graph["nodes"]
        if node["order"] is not None
    }
    start = block_start(layer, consumers, operations)
    next_layers = [candidate for candidate in consumers if candidate > layer]
    if next_layers:
        end = block_start(min(next_layers), consumers, operations) - 1
    else:
        last_weight_order = consumers[layer][-1]
        following_norms = [
            order for order, node in operations.items()
            if order > last_weight_order and "NORM" in node["op"]
        ]
        end = min(following_norms) - 1 if following_norms else max(operations)
    return start, end


def tensor_shape(tensor):
    values = [str(value) for value in tensor["shape"] if value != 1]
    return "x".join(values) if values else "1"


def tensor_backend(tensor):
    if tensor["execution_backend"]:
        return tensor["execution_backend"]
    if tensor["buffer"]:
        return tensor["buffer"].get("device") or tensor["buffer"].get("type") or ""
    return ""


def tensor_title(tensor, prefix):
    name = tensor["name"]
    if name.startswith(prefix):
        name = name[len(prefix):]
    name = name.removesuffix(".weight").removesuffix(".bias")
    if not name or name.startswith("node_"):
        name = tensor["op"]
    return name


def edge_data(edge_id, source, target, tensor, role, port):
    backend = tensor_backend(tensor)
    name = tensor["name"] or tensor["op"]
    label = port + " " + role + " | " + name + " | " + tensor["type"] + " [" + tensor_shape(tensor) + "]"
    if backend:
        label += " @ " + backend
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "role": role,
        "port": port,
        "label": label,
        "tensor_id": tensor["id"],
        "type": tensor["type"],
        "shape": tensor_shape(tensor),
        "backend": backend,
        "quantized": tensor["quantized"],
    }


def extract_physical_block(graph, layer):
    plan = graph.get("physical_plan")
    if not plan or plan.get("backend") != "DAYAN":
        raise ValueError("graph has no DAYAN physical slot plan; export it with -dev DAYAN")

    start, end = block_range(graph, layer)
    tensors = {node["id"]: node for node in graph["nodes"]}
    slots = plan["slots"]
    slots_by_id = {slot["id"]: slot for slot in slots}
    selected = {
        slot["id"] for slot in slots
        if any(node["order"] is not None and start <= node["order"] <= end for node in slot["logical_nodes"])
    }
    if not selected:
        raise ValueError("no DAYAN physical slots found for layer " + str(layer))

    producer = {}
    consumers = {}
    for slot in slots:
        for output in slot["outputs"]:
            producer[output["tensor"]] = slot["id"]
        for slot_input in slot["inputs"]:
            consumers.setdefault(slot_input["tensor"], []).append(slot["id"])

    nodes = []
    edges = []
    tensor_nodes = {}
    used_tensors = set()
    edge_index = 0
    prefix = "blk." + str(layer) + "."

    for slot in slots:
        if slot["id"] not in selected:
            continue
        item = dict(slot)
        item["kind"] = "physical_slot"
        item["slot_kind"] = slot["kind"]
        item["title"] = "slot " + str(slot["index"]) + ": " + slot["kind"]
        nodes.append(item)

    def add_tensor_node(tensor_id, direction):
        used_tensors.add(tensor_id)
        visual_id = "t-" + tensor_id
        if visual_id in tensor_nodes:
            tensor_nodes[visual_id]["directions"] = sorted(set(tensor_nodes[visual_id]["directions"] + [direction]))
            return visual_id
        tensor = tensors[tensor_id]
        if tensor["kind"] == "weight":
            visual_kind = "weight"
        elif direction == "output":
            visual_kind = "block_output"
        else:
            visual_kind = "block_input"
        item = dict(tensor)
        item["id"] = visual_id
        item["tensor_id"] = tensor_id
        item["kind"] = visual_kind
        item["directions"] = [direction]
        item["title"] = tensor_title(tensor, prefix)
        tensor_nodes[visual_id] = item
        return visual_id

    for slot in slots:
        slot_id = slot["id"]
        if slot_id not in selected:
            continue
        for slot_input in slot["inputs"]:
            tensor_id = slot_input["tensor"]
            tensor = tensors[tensor_id]
            producer_id = producer.get(tensor_id)
            source = producer_id if producer_id in selected else add_tensor_node(tensor_id, "input")
            port = "in" + str(slot_input["input_index"])
            edges.append(edge_data("e" + str(edge_index), source, slot_id, tensor, slot_input["role"], port))
            used_tensors.add(tensor_id)
            edge_index += 1

        for output in slot["outputs"]:
            tensor_id = output["tensor"]
            tensor_consumers = consumers.get(tensor_id, [])
            if tensor_consumers and all(consumer in selected for consumer in tensor_consumers):
                continue
            tensor = tensors[tensor_id]
            target = add_tensor_node(tensor_id, "output")
            port = "out" + str(output["output_index"])
            edges.append(edge_data("e" + str(edge_index), slot_id, target, tensor, "output", port))
            edge_index += 1

    nodes.extend(tensor_nodes.values())
    logical_operations = {
        logical["tensor"]
        for slot_id in selected
        for logical in slots_by_id[slot_id]["logical_nodes"]
    }
    fused_slots = sum(len(slots_by_id[slot_id]["logical_nodes"]) > 1 for slot_id in selected)
    slot_kinds = {
        slots_by_id[slot_id]["kind"]
        for slot_id in selected
    }
    if "flash_attention" in slot_kinds:
        attention_type = "Full attention"
    elif "gated_delta_net" in slot_kinds:
        attention_type = "Linear attention"
    else:
        attention_type = "Other"
    return {
        "name": graph["name"] + " block " + str(layer),
        "layer": layer,
        "attention_type": attention_type,
        "order_start": start,
        "order_end": end,
        "slot_count": len(selected),
        "fused_slot_count": fused_slots,
        "logical_operation_count": len(logical_operations),
        "tensor_count": len(used_tensors),
        "nodes": nodes,
        "edges": edges,
        "tensors": {tensor_id: tensors[tensor_id] for tensor_id in used_tensors},
    }


def default_blocks(graph):
    layers = sorted(block_consumers(graph))
    linear = None
    full = None
    blocks = {}
    for layer in layers:
        block = extract_physical_block(graph, layer)
        blocks[layer] = block
        if block["attention_type"] == "Linear attention" and linear is None:
            linear = layer
        elif block["attention_type"] == "Full attention" and full is None:
            full = layer
        if linear is not None and full is not None:
            return [linear, full], blocks

    selected = []
    for layer in (linear, full):
        if layer is not None and layer not in selected:
            selected.append(layer)
    for layer in layers:
        if layer not in selected:
            selected.append(layer)
        if len(selected) == 2:
            break
    return selected, blocks


def parse_blocks(value):
    try:
        blocks = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("blocks must be comma-separated integers") from error
    blocks = list(dict.fromkeys(blocks))
    if not blocks or len(blocks) > 2:
        raise argparse.ArgumentTypeError("--blocks requires one or two Transformer Block indices")
    return blocks


def html_document(data, blocks):
    model = data.get("model", {}).get("description", "llama.cpp model")
    devices = data.get("model", {}).get("devices", [])
    blocks_data = json.dumps(blocks, ensure_ascii=True).replace("</", "<\\/")
    model_data = json.dumps(model, ensure_ascii=True).replace("</", "<\\/")
    devices_data = json.dumps(devices, ensure_ascii=True).replace("</", "<\\/")
    panels = "".join(
        '<section class="graph-panel"><div class="graph-title" id="graph-title-' + str(index) +
        '"></div><div class="cy" id="cy-' + str(index) + '"></div></section>'
        for index in range(len(blocks))
    )
    return """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dayan Transformer Block Physical Slots</title>
<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.34.2/dist/cytoscape.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@4.0.1/dist/cytoscape-dagre.js"></script>
<style>
:root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
* { box-sizing: border-box; }
body { margin: 0; background: #11131d; color: #cad3f5; overflow: hidden; }
#app { display: grid; grid-template-columns: 350px 1fr; height: 100vh; }
aside { padding: 18px; border-right: 1px solid #363a4f; background: #1e2030; overflow: auto; }
h1 { margin: 0; font-size: 19px; }
#subtitle { margin: 6px 0 16px; color: #a5adcb; font-size: 12px; line-height: 1.4; }
label { display: block; margin: 14px 0 6px; color: #a5adcb; font-size: 12px; }
input, button { width: 100%; border: 1px solid #494d64; border-radius: 6px; background: #24273a; color: #cad3f5; padding: 9px; }
button { cursor: pointer; margin-top: 8px; }
.check { display: flex; align-items: center; gap: 8px; margin-top: 12px; color: #b8c0e0; font-size: 13px; }
.check input { width: auto; }
#stats { margin-top: 14px; color: #a5adcb; font-size: 13px; line-height: 1.5; white-space: pre-line; }
#details { white-space: pre-wrap; overflow-wrap: anywhere; font: 11px/1.45 ui-monospace, monospace; color: #b8c0e0; }
#legend { display: grid; grid-template-columns: 12px 1fr; gap: 6px 8px; margin-top: 14px; color: #a5adcb; font-size: 12px; }
.swatch { width: 12px; height: 12px; border-radius: 3px; }
main { display: grid; min-width: 0; min-height: 0; gap: 1px; background: #363a4f; overflow: hidden; }
.graph-panel { position: relative; min-height: 0; background: #11131d; }
.graph-title { position: absolute; z-index: 2; top: 9px; left: 11px; padding: 6px 9px; border: 1px solid #494d64; border-radius: 6px; background: rgba(30, 32, 48, 0.94); color: #cad3f5; font: 12px/1.3 ui-monospace, monospace; pointer-events: none; }
.cy { position: absolute; inset: 0; }
@media (max-width: 800px) { #app { grid-template-columns: 270px 1fr; } }
</style>
</head>
<body>
<div id="app">
<aside>
<h1>Dayan Transformer Blocks</h1>
<div id="subtitle"></div>
<label for="search">Search slot kind, kernel, tensor, type, or shape</label>
<input id="search" type="search" placeholder="flash_attention, swiglu, bf16...">
<label class="check"><input id="logical-members" type="checkbox" checked>Show fused logical members</label>
<button id="fit">Fit both blocks</button>
<button id="relayout">Reset both block layouts</button>
<div id="stats"></div>
<div id="legend">
<span class="swatch" style="background:#ef9f76"></span><span>Matmul physical slot</span>
<span class="swatch" style="background:#8aadf4"></span><span>Attention or state slot</span>
<span class="swatch" style="background:#91d7e3"></span><span>Normalization slot</span>
<span class="swatch" style="background:#f4c95d"></span><span>Weight tensor boundary</span>
<span class="swatch" style="background:#c6a0f6"></span><span>Activation boundary</span>
<span class="swatch" style="height:3px;margin-top:5px;background:#8aadf4"></span><span>F32 tensor edge</span>
<span class="swatch" style="height:3px;margin-top:5px;background:#ef9f76"></span><span>BF16 tensor edge</span>
<span class="swatch" style="height:3px;margin-top:5px;background:#a6da95"></span><span>F16 tensor edge</span>
</div>
<label>Selected slot or tensor</label>
<pre id="details">Click a node or edge to inspect it.</pre>
</aside>
<main id="graphs" style="grid-template-rows:repeat(""" + str(len(blocks)) + """,minmax(0,1fr))">""" + panels + """</main>
</div>
<script>
const MODEL = """ + model_data + """;
const DEVICES = """ + devices_data + """;
const BLOCKS = """ + blocks_data + """;

function slotLabel(node, showMembers) {
  const implementation = node.implementation;
  const lines = [node.title, 'DAYAN PHYSICAL SLOT'];
  lines.push('host: ' + implementation.executor);
  lines.push('kernel: ' + implementation.kernel);
  lines.push('launch: ' + implementation.launch_name);
  lines.push('mode: ' + implementation.task_mode + ' x' + implementation.task_dim_x);
  lines.push('I/O: ' + node.inputs.length + ' inputs / ' + node.outputs.length + ' outputs');
  if (showMembers) {
    lines.push('--- fused logical ops ---');
    node.logical_nodes.forEach(logical => {
      const name = logical.name && !logical.name.startsWith('node_') ? ' ' + logical.name : '';
      lines.push('n' + logical.slot_node_index + ': ' + logical.op + name);
    });
  }
  return lines.join('\\n');
}

function tensorLabel(node) {
  const kind = node.kind === 'weight' ? 'WEIGHT TENSOR' : node.kind === 'block_output' ? 'BLOCK OUTPUT' : 'BLOCK INPUT';
  return node.title + '\\n' + kind;
}

function graphElements(block, showMembers) {
  const elements = block.nodes.map(node => ({
    data: Object.assign({}, node, {
      label: node.kind === 'physical_slot' ? slotLabel(node, showMembers) : tensorLabel(node)
    })
  }));
  block.edges.forEach(edge => elements.push({ data: edge }));
  return elements;
}

const GRAPH_STYLE = [
  { selector: 'node', style: {
    'label': 'data(label)', 'shape': 'round-rectangle', 'background-color': '#cad3f5', 'border-color': '#494d64', 'border-width': 1,
    'color': '#181926', 'font-size': 9, 'font-family': 'ui-monospace, monospace', 'text-wrap': 'wrap', 'text-max-width': 290,
    'text-justification': 'left', 'width': 290, 'height': 'label', 'padding': 12, 'text-valign': 'center', 'text-halign': 'center'
  } },
  { selector: 'node[slot_kind *= "mul_mat"], node[slot_kind = "gated_output_projection"]', style: { 'background-color': '#ef9f76' } },
  { selector: 'node[slot_kind *= "norm"]', style: { 'background-color': '#91d7e3' } },
  { selector: 'node[slot_kind *= "attention"], node[slot_kind *= "gated"], node[slot_kind *= "ssm"]', style: { 'background-color': '#8aadf4' } },
  { selector: 'node[kind = "weight"]', style: { 'background-color': '#f4c95d', 'shape': 'round-tag', 'width': 190 } },
  { selector: 'node[kind = "block_input"], node[kind = "block_output"]', style: { 'background-color': '#c6a0f6', 'shape': 'hexagon', 'width': 190 } },
  { selector: 'edge', style: {
    'label': 'data(label)', 'font-size': 8, 'font-family': 'ui-monospace, monospace', 'color': '#cad3f5',
    'text-background-color': '#11131d', 'text-background-opacity': 0.92, 'text-background-padding': 2,
    'width': 1.8, 'line-color': '#6e738d', 'target-arrow-color': '#6e738d', 'target-arrow-shape': 'triangle',
    'curve-style': 'bezier', 'arrow-scale': 0.8, 'opacity': 0.92
  } },
  { selector: 'edge[type = "f32"]', style: { 'line-color': '#8aadf4', 'target-arrow-color': '#8aadf4' } },
  { selector: 'edge[type = "bf16"]', style: { 'line-color': '#ef9f76', 'target-arrow-color': '#ef9f76' } },
  { selector: 'edge[type = "f16"]', style: { 'line-color': '#a6da95', 'target-arrow-color': '#a6da95' } },
  { selector: 'edge[quantized = true]', style: { 'line-color': '#eed49f', 'target-arrow-color': '#eed49f' } },
  { selector: ':selected', style: { 'border-color': '#ed8796', 'border-width': 4, 'line-color': '#ed8796', 'target-arrow-color': '#ed8796' } },
  { selector: '.dim', style: { 'opacity': 0.08 } },
  { selector: '.match', style: { 'border-color': '#ed8796', 'border-width': 4 } }
];

function runLayout(view) {
  view.cy.layout({ name: 'dagre', rankDir: 'LR', nodeSep: 34, rankSep: 120, edgeSep: 16, ranker: 'network-simplex', animate: false, fit: true, padding: 55 }).run();
}

function bindEvents(view) {
  view.cy.nodes().off('tap');
  view.cy.edges().off('tap');
  view.cy.nodes().on('tap', event => {
    document.getElementById('details').textContent = JSON.stringify(view.nodeById.get(event.target.id()), null, 2);
  });
  view.cy.edges().on('tap', event => {
    const edge = event.target.data();
    document.getElementById('details').textContent = JSON.stringify({
      transformer_block: view.block.layer,
      attention_type: view.block.attention_type,
      port: edge.port,
      role: edge.role,
      tensor: view.tensorById.get(edge.tensor_id),
      producer: view.nodeById.get(edge.source),
      consumer: view.nodeById.get(edge.target)
    }, null, 2);
  });
}

function createView(block, index) {
  const view = {
    block,
    nodeById: new Map(block.nodes.map(node => [node.id, node])),
    tensorById: new Map(Object.entries(block.tensors)),
    cy: cytoscape({
      container: document.getElementById('cy-' + index),
      elements: graphElements(block, true),
      wheelSensitivity: 1.2,
      minZoom: 0.08,
      maxZoom: 4,
      style: GRAPH_STYLE
    })
  };
  document.getElementById('graph-title-' + index).textContent =
    'Transformer Block ' + block.layer + ' | ' + block.attention_type + ' | ' +
    block.slot_count + ' slots, ' + block.fused_slot_count + ' fused';
  bindEvents(view);
  runLayout(view);
  return view;
}

const views = BLOCKS.map(createView);

function rebuild() {
  const showMembers = document.getElementById('logical-members').checked;
  views.forEach(view => {
    view.cy.elements().remove();
    view.cy.add(graphElements(view.block, showMembers));
    bindEvents(view);
    runLayout(view);
  });
  applySearch();
}

function applySearch() {
  const query = document.getElementById('search').value.trim().toLowerCase();
  views.forEach(view => {
    view.cy.elements().removeClass('dim match');
    if (!query) return;
    view.cy.elements().forEach(element => {
      const data = element.data();
      const tensor = data.tensor_id ? view.tensorById.get(data.tensor_id) : null;
      const matched = (JSON.stringify(data) + JSON.stringify(tensor)).toLowerCase().includes(query);
      element.addClass(matched ? 'match' : 'dim');
    });
  });
}

const deviceSummary = DEVICES.length ? DEVICES.join(', ') : 'CPU';
const blockSummary = BLOCKS.map(block => 'Block ' + block.layer + ' ' + block.attention_type).join(' | ');
document.getElementById('subtitle').textContent = MODEL + ' | backend: ' + deviceSummary + ' | ' + blockSummary;
document.getElementById('stats').textContent = BLOCKS.map(block =>
  'Transformer Block ' + block.layer + ': ' + block.slot_count + ' slots (' + block.fused_slot_count + ' fused), ' +
  block.logical_operation_count + ' logical ops, ' + block.tensor_count + ' edge tensors'
).join('\\n');
document.getElementById('search').addEventListener('input', applySearch);
document.getElementById('logical-members').addEventListener('change', rebuild);
document.getElementById('fit').addEventListener('click', () => views.forEach(view => view.cy.fit(undefined, 55)));
document.getElementById('relayout').addEventListener('click', () => views.forEach(runLayout));
window.addEventListener('resize', () => views.forEach(view => view.cy.resize()));
</script>
</body>
</html>
"""


def serve_html(path, host, port):
    content = path.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def send_html(self, include_body):
            if urllib.parse.urlsplit(self.path).path != "/":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if include_body:
                self.wfile.write(content)

        def do_GET(self):
            self.send_html(True)

        def do_HEAD(self):
            self.send_html(False)

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    address = server.server_address[0]
    if address == "0.0.0.0":
        address = "127.0.0.1"
    elif ":" in address:
        address = "[" + address + "]"
    print("serving " + str(path) + " at http://" + address + ":" + str(server.server_port) + "/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Render DAYAN physical slots for one or two complete Transformer Blocks")
    parser.add_argument("input", type=pathlib.Path, help="JSON file from llama-graph-export")
    parser.add_argument("output", type=pathlib.Path, nargs="?", help="output HTML file")
    parser.add_argument("--graph", default="decode", help="graph name to render")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--block", "--layer", dest="block", type=int,
                           help="render one complete Transformer Block")
    selection.add_argument("--blocks", "--layers", dest="blocks", type=parse_blocks,
                           help="render one or two comma-separated Transformer Block indices")
    parser.add_argument("--serve", action="store_true", help="serve the generated HTML until interrupted")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP server bind address")
    parser.add_argument("--port", type=int, default=8000, help="HTTP server port; use 0 to select a free port")
    args = parser.parse_args()

    output = args.output or args.input.with_suffix(".html")
    with args.input.open(encoding="utf-8") as source:
        data = json.load(source)

    graph = graph_by_name(data, args.graph)
    if args.block is not None:
        selected_blocks = [args.block]
        cached_blocks = {}
    elif args.blocks is not None:
        selected_blocks = args.blocks
        cached_blocks = {}
    else:
        selected_blocks, cached_blocks = default_blocks(graph)
    blocks = [
        cached_blocks.get(block) or extract_physical_block(graph, block)
        for block in selected_blocks
    ]
    output.write_text(html_document(data, blocks), encoding="utf-8")
    print("wrote " + str(output))
    if args.serve:
        serve_html(output, args.host, args.port)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare Qwen3.5 layers for single-layer inference with firmware dyperf."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


class InferenceError(ValueError):
    pass


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare_cpu(client, case, result):
    run = json.loads(result.read_text())
    reference = result.parent / "cpu-reference.json"
    command = [str(client.resolve()), str(case), str(reference), "--cpu-reference", str(result)]
    write_json(result.parent / "cpu-command.json", command)
    run["inference_status"] = run.get("inference_status", run["status"])
    run["comparison"] = dict(reference="CPU", status="not-run")
    write_json(result, run)
    with (result.parent / "cpu-reference.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    cpu = json.loads(reference.read_text()) if reference.exists() else {}
    comparison = cpu.get("comparison", dict(reference="CPU", status="error", error=cpu.get("error", "CPU reference did not complete")))
    passed = completed.returncode == 0 and cpu.get("backend") == "CPU" and cpu.get("case") == run["case"] and cpu.get("status") == "ok" and comparison.get("reference") == "CPU" and comparison.get("status") == "pass"
    if not passed and comparison.get("status") == "pass":
        comparison = dict(reference="CPU", status="error", error="CPU reference process or metadata validation failed")
    run["comparison"] = comparison
    run["cpu_reference"] = str(reference)
    run["status"] = "ok" if passed else "failed"
    write_json(result, run)
    return passed


def prepare(model, selected_layers, output):
    sys.path.insert(0, str(ROOT / "gguf-py"))
    import gguf

    reader = gguf.GGUFReader(model)
    arch = reader.get_field("general.architecture").contents()
    if arch not in ("qwen35", "qwen35moe"):
        raise ValueError("only qwen35 and qwen35moe mega graphs are supported")
    split = reader.get_field("split.count")
    if split and split.contents() != 1:
        raise ValueError("merge the split GGUF before extracting a layer")
    layers = reader.get_field(arch + ".block_count").contents()
    nextn = reader.get_field(arch + ".nextn_predict_layers")
    if not selected_layers or len(set(selected_layers)) != len(selected_layers):
        raise ValueError("select one or more unique layer indices")
    outputs = [output if len(selected_layers) == 1 else output / f"layer-{layer}" for layer in selected_layers]
    for layer, directory in zip(selected_layers, outputs):
        if not 0 <= layer < layers - (nextn.contents() if nextn else 0):
            raise ValueError(f"layer {layer} must select an inference block, excluding MTP")
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"output is not a directory: {directory}")
        if (directory / "layer.gguf").exists() or (directory / "layer.json").exists():
            raise ValueError(f"prepared model already exists: {directory}; choose a new output directory")
    for layer, directory in zip(selected_layers, outputs):
        prepare_layer(model, reader, layer, directory)


def prepare_layer(model, reader, layer, output):
    import gguf

    arch = reader.get_field("general.architecture").contents()
    layers = reader.get_field(arch + ".block_count").contents()
    prefix = f"blk.{layer}."
    tensors = [t for t in reader.tensors if t.name.startswith(prefix)]
    linear = any(t.name == prefix + "ssm_conv1d.weight" for t in tensors)
    weight_name = "ffn_gate_exps.weight" if arch == "qwen35moe" else "ffn_gate.weight"
    weight = next((t for t in tensors if t.name == prefix + weight_name), None)
    if weight is None or weight.tensor_type not in (gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.MXFP4):
        raise ValueError("the selected layer needs BF16 or MXFP4 weights")
    if arch == "qwen35moe" and weight.tensor_type != gguf.GGMLQuantizationType.MXFP4:
        raise ValueError("the MoE mega graph requires MXFP4 expert weights")
    dtype = "Mxfp4" if weight.tensor_type == gguf.GGMLQuantizationType.MXFP4 else "Bf16"
    graph = "Qwen35" + ("Moe" if arch == "qwen35moe" else "Dense" + dtype) + ("Linear" if linear else "Full")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "layer.gguf"
    temporary = output / "layer.gguf.tmp"
    writer = gguf.GGUFWriter(temporary, arch, endianess=reader.endianess)
    replaced = {"general.architecture", "general.alignment", arch + ".block_count", arch + ".full_attention_interval",
                arch + ".attention.recurrent_layers", arch + ".nextn_predict_layers"}
    for key, field in reader.fields.items():
        if key.startswith("GGUF.") or key.startswith("split.") or key in replaced:
            continue
        value = field.contents()
        if key in {arch + ".attention.head_count", arch + ".attention.head_count_kv", arch + ".feed_forward_length"} and field.types[0] == gguf.GGUFValueType.ARRAY:
            if len(value) != layers:
                raise ValueError(f"invalid per-layer metadata: {key}")
            value = [value[layer]]
        writer.add_key_value(key, value, field.types[0], field.types[-1])
    writer.add_custom_alignment(int(reader.alignment))
    writer.add_block_count(1)
    writer.add_full_attention_interval(2 if linear else 1)
    writer.add_nextn_predict_layers(0)
    kept = []
    for tensor in reader.tensors:
        if tensor.name.startswith(prefix):
            name = "blk.0." + tensor.name[len(prefix):]
        elif tensor.name in ("token_embd.weight", "output_norm.weight", "output.weight"):
            name = tensor.name
        else:
            continue
        writer.add_tensor(name, tensor.data, raw_dtype=tensor.tensor_type, tensor_endianess=reader.endianess)
        kept.append(name)
    try:
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
    finally:
        writer.close()
    temporary.replace(target)
    metadata = dict(schema=1, model=str(target.resolve()), source_model=str(model.resolve()), source_layer=layer,
                    architecture=arch, attention="linear" if linear else "full", graph=graph, tensors=kept)
    write_json(output / "layer.json", metadata)
    print(f"Prepared {graph}, source layer {layer}: {target}")


def selected_models(prepared, layers):
    batch = not (prepared / "layer.json").is_file()
    if batch and layers is None:
        layers = sorted(int(path.name[6:]) for path in prepared.iterdir()
                        if path.is_dir() and path.name.startswith("layer-") and path.name[6:].isdigit())
    if layers is not None and (not layers or len(set(layers)) != len(layers) or min(layers) < 0):
        raise ValueError("select one or more unique nonnegative layer indices")
    directories = [prepared / f"layer-{layer}" for layer in layers] if batch else [prepared]
    models = []
    for directory in directories:
        metadata = json.loads((directory / "layer.json").read_text())
        if metadata.get("schema") != 1 or not Path(metadata["model"]).is_file():
            raise ValueError(f"invalid prepared model: {directory}")
        layer = metadata["source_layer"]
        if batch and directory.name != f"layer-{layer}":
            raise ValueError(f"source layer does not match directory: {directory}")
        if not batch and layers is not None and layers != [layer]:
            raise ValueError(f"{prepared} contains only source layer {layer}; use the batch directory")
        models.append(metadata)
    return batch, models


def run_layer(args, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(metadata, mode=args.mode, input_len=args.input_len, output_len=args.output_len,
                seed=args.seed, cmp=args.cmp, cmp_atol=args.cmp_atol, cmp_rtol=args.cmp_rtol)
    case = output.resolve() / "case.json"
    result = output.resolve() / "run.json"
    trace = output.resolve() / "perf.json"
    write_json(case, spec)
    command = [str(args.dyperf.resolve()), "--socket", args.socket, "--follow"]
    if args.pcu:
        command += ["--pcu-interval", "0"]
    command += ["-o", str(trace), "--", str(args.client.resolve()), str(case), str(result)]
    write_json(output / "command.json", command)
    with (output / "run.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if completed.returncode:
        raise InferenceError(f"inference or profiler failed ({completed.returncode}); see {output / 'run.log'}; device work may still be active")
    if json.loads(result.read_text()).get("status") != "ok":
        raise InferenceError(f"inference did not pass: {result}; device work may still be active")
    print(f"Recorded {trace}")
    if args.cmp:
        if not compare_cpu(args.client, case, result):
            raise ValueError(f"CPU comparison failed; see {output / 'cpu-reference.json'} and cpu-reference.log")
        print(f"Layer {metadata['source_layer']} CPU comparison: PASS")


def compare_operators(args, metadata, output):
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(metadata, mode=args.mode, input_len=args.input_len, output_len=args.output_len,
                seed=args.seed, op_atol=args.op_atol, op_rtol=args.op_rtol)
    case, result = output.resolve() / "case.json", output.resolve() / "operators.json"
    write_json(case, spec)
    command = [str(args.client.resolve()), str(case), str(result), "--cmp-ops"]
    write_json(output / "command.json", command)
    environment = dict(os.environ, DAYAN_SERVER_SOCKET=args.socket)
    with (output / "operators.log").open("w") as log:
        completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=False)
    run = json.loads(result.read_text()) if result.exists() else {}
    comparison = run.get("operator_comparison", {})
    if run.get("inference_status") != "ok":
        raise InferenceError(f"operator comparison did not complete ({completed.returncode}); see {output / 'operators.log'}")
    if completed.returncode or comparison.get("status") != "pass":
        raise ValueError(f"CPU operator comparison failed; see {result}")
    print(f"Layer {metadata['source_layer']}: {comparison['checked_outputs']} outputs passed CPU comparison: {result}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="copy selected blocks to separate single-layer GGUFs")
    prepare_parser.add_argument("--model", type=Path, required=True)
    prepare_parser.add_argument("--layer", type=int, nargs="+", default=[0], help="zero-based source block indices (default: 0)")
    prepare_parser.add_argument("--output", type=Path, required=True, help="output directory; multiple layers use layer-N subdirectories")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--prepared", type=Path, required=True, help="single-layer or batch directory produced by prepare")
    common.add_argument("--layer", type=int, nargs="+", help="source layer indices (default: all prepared layers)")
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--mode", choices=("prefill", "decode"), nargs="+", default=["prefill"], help="inference modes in execution order (default: prefill)")
    common.add_argument("--input-len", type=int, default=128, help="prompt length; executed before decode")
    common.add_argument("--output-len", type=int, help="number of single-token decode calls (default: 1); prefill always uses 0")
    common.add_argument("--seed", type=int, default=1)
    common.add_argument("--socket", default=os.environ.get("DAYAN_SERVER_SOCKET", "/tmp/dayan-server.sock"))
    common.add_argument("--client", type=Path, default=Path(os.environ.get("DAYAN_LAYER_CLIENT", ROOT / "build-dayan-public/bin/llama-dayan-layer-bench")), help="client executable (default: DAYAN_LAYER_CLIENT or build-dayan-public/bin/llama-dayan-layer-bench)")
    run_parser = commands.add_parser("run", parents=[common], help="run inference, record dyperf and generate reports")
    run_parser.add_argument("--pcu", action="store_true", help="include continuous PCU sampling (off by default; unsupported by cmodel)")
    run_parser.add_argument("--cmp", action="store_true", help="compare every step's final-token logits with a CPU replay after profiling")
    run_parser.add_argument("--cmp-atol", type=float, default=0.03, help="absolute logit tolerance (default: 0.03)")
    run_parser.add_argument("--cmp-rtol", type=float, default=0.03, help="relative logit tolerance (default: 0.03)")
    run_parser.add_argument("--dyperf", type=Path, default=Path(os.environ.get("DAYAN_ROOT", Path.home() / "npu-firmware")) / "build/host/dyperf/dyperf", help="profiler executable; launched automatically (default: $DAYAN_ROOT/build/host/dyperf/dyperf)")
    compare_parser = commands.add_parser("cmp-ops", parents=[common], help="compare each inference operator against CPU using its actual inputs")
    compare_parser.add_argument("--op-atol", type=float, default=1e-4)
    compare_parser.add_argument("--op-rtol", type=float, default=1e-3)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.model, args.layer, args.output)
            return 0
        if len(set(args.mode)) != len(args.mode):
            raise ValueError("select unique inference modes")
        if args.output_len is None:
            args.output_len = int("decode" in args.mode)
        if args.input_len < 1 or args.output_len < 0 or args.input_len + args.output_len > 2**31 - 257:
            raise ValueError("invalid input/output lengths")
        if (args.mode == ["prefill"] and args.output_len != 0) or ("decode" in args.mode and args.output_len == 0):
            raise ValueError("prefill requires output-len=0; decode requires output-len>0")
        if not 0 <= args.seed < 2**32:
            raise ValueError("seed must fit uint32")
        tolerances = (args.cmp_atol, args.cmp_rtol) if args.command == "run" else (args.op_atol, args.op_rtol)
        if any(not math.isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("comparison tolerances must be finite and nonnegative")
        batch, models = selected_models(args.prepared, args.layer)
        execute = run_layer if args.command == "run" else compare_operators
        for path in ([args.client, args.dyperf] if args.command == "run" else [args.client]):
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f"executable not found: {path}")
        if not Path(args.socket).is_socket():
            raise ValueError(f"start a four-core Dayan server first: {args.socket}")
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise ValueError("run output directory must be empty")
        multiple_modes = len(args.mode) > 1
        entries = [dict(source_layer=model["source_layer"], mode=mode,
                        directory=f"layer-{model['source_layer']}/{mode}" if multiple_modes else f"layer-{model['source_layer']}" if batch else ".",
                        status="not-run") for model in models for mode in args.mode]
        manifest = None
        if batch or multiple_modes:
            manifest = dict(schema=2, status="running", modes=args.mode, runs=entries) if multiple_modes else dict(schema=1, status="running", layers=entries)
            write_json(args.output / "batch.json", manifest)
        status = 0
        for index, entry in enumerate(entries):
            case_args = argparse.Namespace(**vars(args))
            case_args.mode = entry["mode"]
            case_args.output_len = args.output_len if entry["mode"] == "decode" else 0
            try:
                execute(case_args, models[index // len(args.mode)], args.output / entry["directory"])
                entry["status"] = "ok"
            except (OSError, ValueError, KeyError) as error:
                entry.update(status="failed", error=str(error))
                print(f"layer-perf: layer {entry['source_layer']} {entry['mode']}: {error}", file=sys.stderr)
                status = 1
                if isinstance(error, InferenceError):
                    print("layer-perf: batch stopped; check the external server before running another case", file=sys.stderr)
                    break
            if manifest:
                write_json(args.output / "batch.json", manifest)
        if manifest:
            manifest["status"] = "failed" if status else "ok"
            write_json(args.output / "batch.json", manifest)
        if args.command == "run":
            command = [sys.executable, str(Path(__file__).with_name("layer-perf-report.py")), "--run", str(args.output)]
            if subprocess.run(command, check=False).returncode:
                status = 1
        return status
    except (OSError, ValueError, KeyError, StopIteration) as error:
        print(f"layer-perf: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Measure a small Dayan operator set using an existing server and dyperf trace."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
COUNTER_PERIOD = 1 << 48
MODEL_CONFIG = Path(__file__).with_name("limited-perf-models.json")
DEFAULT_MODEL = "qwen35-0.8B-bf16"


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_model(name=DEFAULT_MODEL):
    configuration = json.loads(MODEL_CONFIG.read_text())
    if configuration.get("schema") != 1:
        raise ValueError("unsupported model configuration schema")
    models = configuration["models"]
    if name not in models:
        raise ValueError(f"unknown model {name!r}; choose from: " + ", ".join(models))
    model = models[name]
    names = set()
    allowed = {"name", "op", "m", "n", "k", "dtype", "output_dtype", "phases", "channels", "kernel",
               "heads", "q_heads", "kv_heads", "head_dim", "snapshots", "mode", "layers", "experts", "experts_used", "decode_steps"}
    for case in model["cases"]:
        if not re.fullmatch(r"[a-z0-9-]+", case["name"]) or case["name"] in names or set(case) - allowed:
            raise ValueError(f"invalid or duplicate case in model {name}: {case}")
        if case["op"] not in ("matmul", "ffn", "moe_ffn", "moe_down", "sort", "quantize", "gdn", "ssm_conv", "attention", "norm", "mega"):
            raise ValueError(f"unsupported operator in model {name}: {case['op']}")
        if case["op"] == "mega":
            dense = case.get("mode") in ("linear", "full") and case.get("layers") == 1 and case.get("dtype", "bf16") == "bf16"
            moe = case.get("mode") == "hybrid" and case.get("layers") == 2 and case.get("dtype") == "mxfp4" and 2 <= case.get("decode_steps", 2) <= 16
            if not (dense or moe) or not 1 <= case.get("m", 1) <= 512 or case.get("output_dtype", "f32") != "f32":
                raise ValueError(f"invalid dense or MoE mega case in model {name}: {case['name']}")
        names.add(case["name"])
        phases = case.get("phases", ["prefill", "decode"])
        if not phases or any(phase not in ("prefill", "decode") for phase in phases):
            raise ValueError(f"invalid phases in model {name}: {case['name']}")
    if not names or not model["prediction_terms"] or not isinstance(model["description"], str):
        raise ValueError(f"empty cases, prediction terms, or invalid description in model {name}")
    ids = {case["id"] for case in make_cases(1024, 1024, model=model)}
    for phase in ("prefill", "decode"):
        for case, factor, reason in prediction_terms(phase, model):
            if case not in ids or not math.isfinite(factor) or factor <= 0 or not isinstance(reason, str):
                raise ValueError(f"invalid prediction term in model {name}: {case}")
    return model


def make_cases(prefill, context, repeats=1, warmup=0, timeout_ms=86400000, model=None, sort_rows=4):
    model = load_model() if model is None else model
    cases = []
    for phase, m in (("prefill", prefill), ("decode", 1)):
        for template in model["cases"]:
            shape = dict(template)
            name = shape.pop("name")
            if phase not in shape.pop("phases", ["prefill", "decode"]):
                continue
            if shape["op"] == "attention":
                shape["kv_len"] = m if phase == "prefill" else context + 1
            case_m = shape.pop("m", m)
            if phase == "prefill" and shape["op"] == "sort" and sort_rows and case_m > sort_rows:
                shape["target_m"] = case_m
                case_m = sort_rows
            cases.append(dict(id=f"{phase}-{name}", phase=phase, m=case_m, repeats=repeats, warmup=warmup,
                              timeout_ms=timeout_ms, **shape))
    return cases


def prediction_terms(phase, model=None, cases=()):
    model = load_model() if model is None else model
    sampled = {case["id"]: case for case in cases if "target_m" in case}
    terms = []
    for term in model["prediction_terms"]:
        name = term["case"].replace("{phase}", phase)
        factor, reason = term["factor"] / term.get("divisor", 1), term["reason"]
        if name in sampled:
            case = sampled[name]
            factor *= case["target_m"] / case["m"]
            reason += f"; estimated row scaling {case['target_m']}/{case['m']}"
        terms.append((name, factor, reason))
    return terms


def case_types(case):
    dtype = case.get("dtype", "bf16" if case["op"] in ("matmul", "ffn", "attention", "mega") else "f32")
    output = case.get("output_dtype", "bf16" if case["op"] in ("ffn", "norm") else "mxfp4" if case["op"] == "quantize" else "f32")
    return dtype, output


def read_trace(path, wanted):
    """Read complete events from a possibly unfinished dyperf Chrome trace."""
    roots = {}
    names = {}
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buffer = stream.read(65536)
        prefix = buffer.find('"traceEvents"')
        start = buffer.find("[", prefix) if prefix >= 0 else -1
        if start < 0:
            return roots, "trace header is not available"
        buffer = buffer[start + 1:]
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                return roots, None
            try:
                event, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                more = stream.read(65536)
                if not more:
                    return roots, "trace is still open or truncated; only complete events were read"
                buffer += more
                if len(buffer) > 1024 * 1024:
                    raise ValueError("malformed dyperf event (larger than 1 MiB)")
                continue
            buffer = buffer[end:]
            key = (event.get("pid"), event.get("tid"))
            if event.get("name") == "thread_name":
                name = event.get("args", {}).get("name")
                if name in wanted:
                    names[key] = name
                    roots.setdefault(name, {})[key] = {}
            elif key in names and event.get("cat") == "proc_state" and event.get("ph") == "B":
                state = event.get("name")
                if state in ("RUNNING", "WAIT_EXIT"):
                    counter = event.get("args", {}).get("counter")
                    if not isinstance(counter, int) or not 0 <= counter < COUNTER_PERIOD:
                        raise ValueError("missing or invalid dyperf raw cycle counter")
                    roots[names[key]][key].setdefault(state, counter)


def cycle_span(roots, tag, tasks):
    entries = list(roots.get(tag, {}).values())
    if len(entries) != tasks or any("RUNNING" not in item or "WAIT_EXIT" not in item for item in entries):
        return None, f"need {tasks} complete root lifetimes for {tag}; found {len(entries)} roots"
    anchor = entries[0]["RUNNING"]

    def relative(counter):
        return (counter - anchor + COUNTER_PERIOD // 2) % COUNTER_PERIOD - COUNTER_PERIOD // 2

    starts = [relative(item["RUNNING"]) for item in entries]
    ends = [relative(item["WAIT_EXIT"]) for item in entries]
    if any(end <= start for start, end in zip(starts, ends)):
        return None, "non-positive root duration; counter source is not suitable for performance measurement"
    span = max(ends) - min(starts)
    if span >= COUNTER_PERIOD // 2:
        return None, "ambiguous 48-bit counter wrap"
    return span, None


def shape_text(case):
    dtype, output = case_types(case)
    shape = " ".join(f"{key}={case[key]}" for key in ("m", "n", "k", "experts", "experts_used", "channels", "kernel", "heads", "q_heads", "kv_heads", "head_dim", "kv_len", "mode", "layers", "decode_steps") if key in case)
    target = f" target_m={case['target_m']} row_scale={case['target_m'] / case['m']:g}" if "target_m" in case else ""
    return f"{shape} dtype={dtype} output={output}{target}"


def make_report(directory, manifest, trace_path, overhead_percent, clock_hz):
    model_name = manifest.get("model", DEFAULT_MODEL)
    model = manifest.get("model_config")
    if model is None:
        model = load_model(model_name)
    results = {}
    for case in manifest["cases"]:
        path = directory / (case["id"] + ".result.json")
        if path.exists():
            results[case["id"]] = json.loads(path.read_text())
    wanted = {sample["tag"] for result in results.values() for sample in result.get("samples", [])}
    roots, trace_note = ({}, "dyperf trace file is not available")
    if trace_path.exists():
        roots, trace_note = read_trace(trace_path, wanted)
    rows = []
    for case in manifest["cases"]:
        result = results.get(case["id"], {})
        dtype, output_dtype = case_types(case)
        row = dict(id=case["id"], op=case["op"], shape=shape_text(case), dtype=dtype, output_dtype=output_dtype,
                   status=result.get("status", "not-run"), wall_s=None, cycles=None,
                   process_wall_s=result.get("process_wall_s"), max_abs_error=result.get("max_abs_error"))
        comparison = result.get("comparison", {})
        row.update(comparison=comparison, validation=result.get("validation"),
                   cmp_status=comparison.get("status", "not-run") if case.get("cmp") else "smoke",
                   max_rel_error=comparison.get("max_rel_error"), relative_l2_error=comparison.get("relative_l2_error"),
                   checked_elements=result.get("checked_elements"), reference_wall_s=result.get("reference_wall_s"))
        if case.get("cmp") and row["status"] == "ok" and row["cmp_status"] != "pass":
            row["status"] = "failed"
            row["error"] = "requested comparison did not pass"
        samples = []
        for sample in result.get("samples", []):
            if sample["warmup"] or sample["status"] != 0 or not math.isfinite(sample["wall_s"]) or sample["wall_s"] <= 0:
                continue
            cycles, note = cycle_span(roots, sample["tag"], sample["tasks"])
            samples.append(dict(sample, cycles=cycles, cycle_note=note))
        row["samples"] = samples
        if row["status"] == "ok" and len(samples) == case["repeats"]:
            row["wall_s"] = statistics.median(sample["wall_s"] for sample in samples)
            if all(sample["cycles"] is not None for sample in samples):
                row["cycles"] = statistics.median(sample["cycles"] for sample in samples)
        if result.get("error"):
            row["error"] = result["error"]
        rows.append(row)
    by_id = {row["id"]: row for row in rows}
    predictions = []
    for phase, tokens in (("prefill", manifest["prefill"]), ("decode", 1)):
        terms = prediction_terms(phase, model, manifest["cases"])
        missing = [name for name, _, _ in terms if by_id.get(name, {}).get("wall_s") is None]
        prediction = dict(phase=phase, tokens=tokens, context=0 if phase == "prefill" else model.get("decode_context", manifest["context"]),
                          terms=[dict(case=name, factor=factor, reason=reason) for name, factor, reason in terms],
                          missing_cases=sorted(set(missing)), subtotal_wall_s=None, subtotal_cycles=None,
                          predicted_wall_s=None, predicted_cycles=None, predicted_tokens_per_s=None)
        if not missing:
            wall = sum(by_id[name]["wall_s"] * factor for name, factor, _ in terms)
            prediction["subtotal_wall_s"] = wall
            prediction["predicted_wall_s"] = wall * (1 + overhead_percent / 100)
            prediction["predicted_tokens_per_s"] = tokens / prediction["predicted_wall_s"]
            if all(by_id[name]["cycles"] is not None for name, _, _ in terms):
                cycles = sum(by_id[name]["cycles"] * factor for name, factor, _ in terms)
                prediction["subtotal_cycles"] = cycles
                prediction["predicted_cycles"] = cycles * (1 + overhead_percent / 100)
                if clock_hz:
                    prediction["ideal_device_s_at_clock"] = prediction["predicted_cycles"] / clock_hz
        predictions.append(prediction)
    notes = [
        "Wall time is host steady_clock around a synchronous kernel launch, including module loading and device wait; it excludes input/config uploads, validation, and runtime setup.",
        "Process wall time includes setup, uploads, launches, output download, validation, and teardown. Both wall measurements include the cost of the active profiler.",
        "Cycles are max(root WAIT_EXIT begin) - min(root first RUNNING begin) across the launched root tasks, using dyperf 48-bit counters. Child threads and overlapping cores are not summed. Host/device submission and the final RPC reply are outside this span.",
        "Cross-core counters must share a time origin. A launch must last less than half the 48-bit counter period. No PCU counters or kernel-side instrumentation are required.",
        f"Model preset: {model_name}. {model['description']} Empty-cache prefill and one decode token at the specified prior context.",
        "Predictions are a serial weighted sum of representative kernels, not an executed model. N scaling assumes linear cost, including the large vocabulary projection; pair/projection fusion and final-layer token pruning can change that cost.",
        "Unmeasured work includes RoPE and projection epilogues, extra layouts/casts, small gates/elementwise ops, KV/state copies, host graph scheduling, and sampling. Weights are assumed resident. Model loading, tokenization, and simulator/server startup are excluded.",
        f"The end-to-end scenario adds {overhead_percent:g}% to BOTH sampled subtotals as an explicit uncalibrated allowance, not a measured overhead or a confidence interval. Inspect and adjust the factors before using a different model.",
        "With no warmup, the wall prediction repeats each sample's cold module-loading cost; this can overestimate a resident model. Use --warmup 1 for steady-state measurements at the cost of an extra launch per case.",
        "TestCluster wall predictions apply only to the same simulator/host/profiler setup and are not hardware latency. Dayan-emu wall times and counters do not establish RTL performance.",
        "With --cmp, deterministic varied inputs are shared with a GGML CPU F32 reference. Stored BF16/MXFP4 inputs and output conversion are included in the reference. Every launch is checked outside its timing interval. These are synthetic operator inputs, not tensors captured from model inference.",
    ]
    if model.get("comparison_reference") == "dayan-standalone":
        notes = [
            "Wall time measures one complete llama_decode call on the two-layer hybrid fixture, including host graph work, embedding, and final projection. Setup, reference runs, and extra decode validation steps are excluded.",
            "Every initial and subsequent decode step compares logits with standalone Dayan and requires one mega launch per layer. This checks fusion; independent operator cases provide the CPU reference checks.",
            "The cycle span covers both four-core mega launches and their intervening host gap. It excludes embedding and final projection. Dayan-emu counters do not establish hardware or RTL performance.",
            "The reported subtotal covers this fixture only. It is not a 40-layer model estimate. Decode uses two cached tokens. Repeats=1 and warmup=0 are required.",
            f"The additional {overhead_percent:g}% is an uncalibrated allowance. Use --overhead-percent 0 for the measured subtotal.",
        ]
    if trace_note:
        notes.append(trace_note)
    for case in manifest["cases"]:
        if "target_m" in case:
            notes.append(f"{case['id']} measures {case['m']} rows and estimates {case['target_m']} rows using a {case['target_m'] / case['m']:g}x factor. The measured row remains unscaled. This assumes linear row cost; launch overhead and task occupancy can bias the estimate, especially when the sample does not fill all cores.")
    report = dict(schema=2, model=model_name, model_config=model, simulator=manifest["simulator"], run_id=manifest["run_id"],
                  perf_trace=str(trace_path), overhead_percent=overhead_percent, clock_hz=clock_hz,
                  cases=rows, predictions=predictions, notes=notes)
    write_json(directory / "report.json", report)
    with (directory / "operators.csv").open("w", newline="") as stream:
        fields = ["id", "op", "dtype", "output_dtype", "shape", "status", "wall_s", "cycles", "process_wall_s", "cmp_status", "checked_elements", "max_abs_error", "max_rel_error", "relative_l2_error", "reference_wall_s"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    def number(value, digits=6):
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = [f"Dayan limited operator measurements ({model_name}, {manifest['simulator']}, run {manifest['run_id']})", "",
             f"Measured repeats per case: {manifest['cases'][0]['repeats']}; warmups per case: {manifest['cases'][0]['warmup']}. Cycles on Dayan-emu are not RTL cycles.", "",
             "| Case | Shape | Status | Launch wall (s) | Cycles | Process wall (s) | CPU cmp | Max abs error | Relative L2 |",
             "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['shape']} | {row['status']} | {number(row['wall_s'])} | {number(row['cycles'], 0)} | {number(row['process_wall_s'])} | {row['cmp_status']} | {number(row['max_abs_error'], 9)} | {number(row['relative_l2_error'], 9)} |")
    lines += ["", "End-to-end projection (not measured model execution)", "",
              "| Phase | Tokens | Prior context | Sampled subtotal (s / cycles) | With allowance (s / cycles) | Tokens/s |",
              "| --- | ---: | ---: | --- | --- | ---: |"]
    for p in predictions:
        lines.append(f"| {p['phase']} | {p['tokens']} | {p['context']} | {number(p['subtotal_wall_s'])} / {number(p['subtotal_cycles'], 0)} | {number(p['predicted_wall_s'])} / {number(p['predicted_cycles'], 0)} | {number(p['predicted_tokens_per_s'])} |")
    lines += ["", "Projection factors (also saved in report.json):", ""]
    for phase in ("prefill", "decode"):
        lines.append(f"- {phase}: " + "; ".join(f"{name} * {factor:g}" for name, factor, _ in prediction_terms(phase, model, manifest["cases"])))
    lines += ["", *[f"- {note}" for note in notes], ""]
    for row in rows:
        if row.get("error"):
            lines.append(f"- {row['id']}: {row['error']}")
        for sample in row["samples"]:
            if sample["cycle_note"]:
                lines.append(f"- {row['id']}: {sample['cycle_note']}")
    for p in predictions:
        if p["missing_cases"]:
            lines.append(f"- {p['phase']} prediction needs: {', '.join(p['missing_cases'])}")
    rendered = "\n".join(lines) + "\n"
    (directory / "report.md").write_text(rendered)
    print(rendered, flush=True)
    if any(row["status"] != "ok" for row in rows):
        return 1
    return 2 if any(row["cycles"] is None for row in rows) else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--model", help=f"model preset from {MODEL_CONFIG.name} (default: {DEFAULT_MODEL}, or the saved run for report)")
    parser.add_argument("--output", type=Path, default=ROOT / "build-dayan/limited-perf")
    parser.add_argument("--client", type=Path, default=ROOT / "build-dayan/bin/llama-dayan-operator-bench")
    parser.add_argument("--perf-trace", type=Path, help="JSON output of the already-running dyperf (required for run)")
    parser.add_argument("--socket", default=os.environ.get("DAYAN_SERVER_SOCKET", "/tmp/dayan-server.sock"))
    parser.add_argument("--simulator", choices=("testcluster", "dayan_emu"), default="testcluster")
    parser.add_argument("--prefill", type=int, default=1024, help="prefill tokens (default: 1024)")
    parser.add_argument("--sort-rows", type=int, default=4, help="maximum measured prefill rows for sorting; estimate the requested prefill by row scaling (default: 4; 0: full size)")
    parser.add_argument("--context", type=int, default=1024, help="decode prefix tokens, excluding the new token (default: 1024)")
    parser.add_argument("--case", dest="cases", action="append", help="select one case; repeat to select several")
    parser.add_argument("--cases", action="extend", nargs="+", help="select a group of cases; defaults to all cases in the model preset")
    parser.add_argument("--case-from", help="start at this case, inclusive, in plan order after case selection")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--cmp", action="store_true", help="compare against CPU operators, or standalone Dayan for the MoE mega preset")
    parser.add_argument("--cmp-seed", type=int, default=1, help="32-bit input seed for --cmp (default: 1)")
    parser.add_argument("--timeout", type=int, default=86400, help="per-case wall deadline in seconds (default: 24 hours)")
    parser.add_argument("--overhead-percent", type=float, help="uncalibrated allowance for unmeasured work (default: 20%%, or saved report setting)")
    parser.add_argument("--clock-hz", type=float, help="optional hardware clock for an additional ideal-device estimate; never used to infer cycles")
    parser.add_argument("--resume", action="store_true", help="skip completed cases; require the same executable and run parameters")
    args = parser.parse_args()
    if not 0 <= args.cmp_seed <= 0xffffffff:
        parser.error("--cmp-seed must be an unsigned 32-bit integer")
    if args.cmp_seed != 1 and not args.cmp:
        parser.error("--cmp-seed requires --cmp")
    if not (1 <= args.prefill <= 1024 and 0 <= args.sort_rows <= 1024 and 0 <= args.context < 65536 and 1 <= args.repeats <= 100 and
            0 <= args.warmup <= 99 and args.repeats + args.warmup <= 100 and 1 <= args.timeout <= 4294967):
        parser.error("invalid shape, repetitions, or timeout")
    if (args.overhead_percent is not None and (not math.isfinite(args.overhead_percent) or args.overhead_percent < 0)) or (
            args.clock_hz is not None and (not math.isfinite(args.clock_hz) or args.clock_hz <= 0)):
        parser.error("invalid overhead or clock")
    directory = args.output.resolve()
    manifest_path = directory / "manifest.json"
    if args.command == "report":
        manifest = json.loads(manifest_path.read_text())
        if args.model is not None and args.model != manifest.get("model", DEFAULT_MODEL):
            parser.error("--model does not match the saved run")
        cases = manifest["cases"]
        if args.cmp and not all(case.get("cmp") for case in cases):
            parser.error("the saved run has no CPU comparison; run with --cmp in a new output directory")
    else:
        model_name = args.model or DEFAULT_MODEL
        try:
            model = load_model(model_name)
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
            parser.error(str(error))
        cases = make_cases(args.prefill, args.context, args.repeats, args.warmup, args.timeout * 1000, model, args.sort_rows)
        if args.cmp:
            for case in cases:
                case.update(cmp=True, seed=args.cmp_seed)
    if any(case["op"] == "mega" for case in cases) and args.command == "run" and args.simulator != "dayan_emu":
        parser.error("mega cases are restricted to --simulator Dayan-emu")
    if any(case["op"] == "mega" for case in cases) and args.command == "run" and (args.repeats != 1 or args.warmup != 0):
        parser.error("mega cases currently require --repeats 1 --warmup 0")
    if args.cases:
        missing = set(args.cases) - {case["id"] for case in cases}
        if missing:
            parser.error("cases are not in this model plan or saved run: " + ", ".join(sorted(missing)))
        cases = [case for case in cases if case["id"] in args.cases]
    if args.case_from:
        selected_ids = [case["id"] for case in cases]
        if args.case_from not in selected_ids:
            parser.error("--case-from must be in the selected cases: " + args.case_from)
        cases = cases[selected_ids.index(args.case_from):]
    if args.command == "plan":
        for case in cases:
            reference = model.get("comparison_reference", "cpu")
            print(f"{case['id']:20} {case['op']:10} {shape_text(case)}" + (f" cmp={reference} seed={case['seed']}" if case.get("cmp") else ""))
        unit = "measured inference calls" if model.get("comparison_reference") == "dayan-standalone" else "kernel launches"
        print(f"{len(cases)} cases, {len(cases) * (args.repeats + args.warmup)} {unit}; model={model_name}; no server access.")
        return 0
    if args.command == "report":
        manifest["cases"] = cases
        trace = (args.perf_trace or Path(manifest["perf_trace"])).resolve()
        overhead = args.overhead_percent if args.overhead_percent is not None else manifest.get("overhead_percent", 20)
        return make_report(directory, manifest, trace, overhead, args.clock_hz or manifest.get("clock_hz"))
    if args.perf_trace is None:
        parser.error("run requires --perf-trace from the existing dyperf")
    trace = args.perf_trace.resolve()
    client = args.client.resolve()
    if args.cmp or model_name != DEFAULT_MODEL or any(case.get("dtype") == "mxfp4" or case.get("output_dtype") == "mxfp4" or
           (case["op"] == "norm" and case.get("output_dtype") == "f32") for case in cases):
        probe = subprocess.run([str(client), "--capabilities"], capture_output=True, text=True, timeout=10)
        try:
            capabilities = json.loads(probe.stdout)
        except ValueError:
            capabilities = {}
        if probe.returncode != 0 or capabilities.get("case_schema", 0) < 2:
            parser.error("this model needs an updated llama-dayan-operator-bench; rebuild it and pass --client")
        if args.cmp and capabilities.get("cmp") is not True:
            parser.error("--cmp needs an updated llama-dayan-operator-bench; rebuild it and pass --client")
        if any(case["op"] == "mega" for case in cases) and "mega" not in capabilities.get("ops", []):
            parser.error("mega cases need --client build-dayan/bin/test-dayan-qwen35-dense")
    fingerprint = hashlib.sha256(client.read_bytes()).hexdigest()
    configuration = dict(model=model_name, model_config=model, simulator=args.simulator, prefill=args.prefill, sort_rows=args.sort_rows, context=args.context,
                         cases=cases, client=str(client), client_sha256=fingerprint,
                         backend_sha256=hashlib.sha256((client.parent / "libggml-dayan.so").read_bytes()).hexdigest()
                         if (client.parent / "libggml-dayan.so").exists() else None,
                         perf_trace=str(trace), socket=args.socket)
    if manifest_path.exists():
        if not args.resume:
            parser.error("output already contains a run; use --resume or a new directory")
        manifest = json.loads(manifest_path.read_text())
        saved = dict(manifest)
        saved.setdefault("model", DEFAULT_MODEL)
        if "model_config" not in saved:
            saved["model_config"] = load_model(saved["model"])
        if any(saved.get(key) != value for key, value in configuration.items()):
            parser.error("resume parameters or executable changed; use the original options or a new directory")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema=2, run_id=uuid.uuid4().hex[:7], created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        overhead_percent=20 if args.overhead_percent is None else args.overhead_percent,
                        clock_hz=args.clock_hz, **configuration)
        write_json(manifest_path, manifest)
    env = dict(os.environ, DAYAN_SERVER_SOCKET=args.socket)
    if any(case["op"] == "mega" for case in cases):
        env["GGML_DAYAN_MEGA"] = "1"
    for index, case in enumerate(cases):
        result_path = directory / (case["id"] + ".result.json")
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get("status") == "ok" and (case["op"] != "mega" or prior.get("mega_launched") is True):
                print(f"Skip completed {case['id']}", flush=True)
                continue
        spec_path = directory / (case["id"] + ".case.json")
        write_json(spec_path, case)
        # A new tag on retry prevents an old partial launch from matching the new one.
        tag = "d" + uuid.uuid4().hex[:10]
        print(f"[{index + 1}/{len(cases)}] {case['id']}: {shape_text(case)} (log: {directory / (case['id'] + '.log')})", flush=True)
        command = [str(client), str(spec_path), str(result_path), tag]
        start = time.monotonic()
        interrupted = False
        with (directory / (case["id"] + ".log")).open("w") as log:
            try:
                completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                result = json.loads(result_path.read_text()) if result_path.exists() else dict(status="failed", error="client did not write a result")
                result["exit_code"] = completed.returncode
                if completed.returncode:
                    result["status"] = "failed"
                if case["op"] == "mega" and result.get("mega_launched") is not True:
                    result["status"] = "failed"
                    result["error"] = "client graph did not launch a Dayan mega executor"
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
                result = dict(status="failed", error=type(error).__name__ + ": verify device/server state before resuming")
                interrupted = True
        result["process_wall_s"] = time.monotonic() - start
        result["command"] = command
        write_json(result_path, result)
        print(f"  {result['status']}, process wall {result['process_wall_s']:.3f} s", flush=True)
        if result["status"] != "ok" or interrupted:
            break
    overhead = args.overhead_percent if args.overhead_percent is not None else manifest.get("overhead_percent", 20)
    return make_report(directory, manifest, trace, overhead, args.clock_hz or manifest.get("clock_hz"))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print(f"limited-perf: {error}", file=sys.stderr)
        sys.exit(1)

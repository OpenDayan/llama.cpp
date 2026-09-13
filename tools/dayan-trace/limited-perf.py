#!/usr/bin/env python3
"""Measure a small Dayan operator set using an existing server and dyperf trace."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[2]
COUNTER_PERIOD = 1 << 48


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_cases(prefill, context, repeats=1, warmup=0, timeout_ms=86400000):
    cases = []
    for phase, m in (("prefill", prefill), ("decode", 1)):
        shapes = [
            ("ffn", dict(op="ffn", m=m, n=3584, k=1024)),
            ("down", dict(op="matmul", m=m, n=1024, k=3584)),
            ("input", dict(op="matmul", m=m, n=6144, k=1024)),
            ("output", dict(op="matmul", m=m, n=1024, k=2048)),
            ("gdn", dict(op="gdn", m=m, heads=16, head_dim=128, snapshots=1)),
            ("conv", dict(op="ssm_conv", m=m, channels=6144, kernel=4)),
            ("attention", dict(op="attention", m=m, q_heads=8, kv_heads=2, head_dim=256,
                               kv_len=m if phase == "prefill" else context + 1)),
            ("norm", dict(op="norm", m=m, n=1024)),
        ]
        for name, shape in shapes:
            cases.append(dict(id=f"{phase}-{name}", phase=phase, repeats=repeats, warmup=warmup,
                              timeout_ms=timeout_ms, **shape))
    return cases


def prediction_terms(phase):
    # The 25th GGUF block is the MTP block and is not part of ordinary inference.
    return [
        (f"{phase}-ffn", 24, "24 fused gate/up + SwiGLU + BF16 slots"),
        (f"{phase}-down", 24, "24 FFN down projections"),
        (f"{phase}-input", (18 * (6144 + 2048 + 32) + 6 * (4096 + 512 + 512)) / 6144,
         "input projection proxy: sum of N / 6144 at K=1024; excludes fused epilogues"),
        (f"{phase}-output", 24, "24 output projections; excludes the full-attention gate epilogue"),
        (f"{phase}-gdn", 18, "18 linear-attention recurrences"),
        (f"{phase}-conv", 18, "18 depthwise convolutions + SiLU"),
        (f"{phase}-attention", 6, "6 full-attention layers"),
        (f"{phase}-norm", 48, "two residual/norm/scale/cast proxies per block"),
        ("decode-norm", 1, "one final-token norm proxy"),
        ("decode-input", 248320 / 6144, "one final-token vocabulary projection, N scaled from 6144 to 248320"),
    ]


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
    return " ".join(f"{key}={case[key]}" for key in ("m", "n", "k", "channels", "kernel", "heads", "q_heads", "kv_heads", "head_dim", "kv_len") if key in case)


def make_report(directory, manifest, trace_path, overhead_percent, clock_hz):
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
        row = dict(id=case["id"], op=case["op"], shape=shape_text(case),
                   status=result.get("status", "not-run"), wall_s=None, cycles=None,
                   process_wall_s=result.get("process_wall_s"), max_abs_error=result.get("max_abs_error"))
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
        terms = prediction_terms(phase)
        missing = [name for name, _, _ in terms if by_id.get(name, {}).get("wall_s") is None]
        prediction = dict(phase=phase, tokens=tokens, context=0 if phase == "prefill" else manifest["context"],
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
        "Model preset: Qwen3.5-0.8B BF16, 24 inference blocks (18 linear, 6 full attention), hidden=1024, FFN=3584, vocab=248320; excludes the MTP block. Batch=1, one final-token logit row, empty-cache prefill and one decode token at the specified prior context.",
        "Predictions are a serial weighted sum of representative kernels, not an executed model. N scaling assumes linear cost, including the large vocabulary projection; pair/projection fusion and final-layer token pruning can change that cost.",
        "Unmeasured work includes RoPE and projection epilogues, extra layouts/casts, small gates/elementwise ops, KV/state copies, host graph scheduling, and sampling. Weights are assumed resident. Model loading, tokenization, and simulator/server startup are excluded.",
        f"The end-to-end scenario adds {overhead_percent:g}% to BOTH sampled subtotals as an explicit uncalibrated allowance, not a measured overhead or a confidence interval. Inspect and adjust the factors before using a different model.",
        "With no warmup, the wall prediction repeats each sample's cold module-loading cost; this can overestimate a resident model. Use --warmup 1 for steady-state measurements at the cost of an extra launch per case.",
        "TestCluster wall predictions apply only to the same simulator/host/profiler setup and are not hardware latency. cmodel wall times and counters do not establish RTL performance.",
    ]
    if trace_note:
        notes.append(trace_note)
    report = dict(schema=1, simulator=manifest["simulator"], run_id=manifest["run_id"],
                  perf_trace=str(trace_path), overhead_percent=overhead_percent, clock_hz=clock_hz,
                  cases=rows, predictions=predictions, notes=notes)
    write_json(directory / "report.json", report)
    with (directory / "operators.csv").open("w", newline="") as stream:
        fields = ["id", "op", "shape", "status", "wall_s", "cycles", "process_wall_s", "max_abs_error"]
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    def number(value, digits=6):
        return "N/A" if value is None else f"{value:.{digits}f}"

    lines = [f"Dayan limited operator measurements ({manifest['simulator']}, run {manifest['run_id']})", "",
             f"Measured repeats per case: {manifest['cases'][0]['repeats']}; warmups per case: {manifest['cases'][0]['warmup']}. Cycles on cmodel are not RTL cycles.", "",
             "| Case | Shape | Status | Launch wall (s) | Cycles | Process wall (s) |",
             "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['id']} | {row['shape']} | {row['status']} | {number(row['wall_s'])} | {number(row['cycles'], 0)} | {number(row['process_wall_s'])} |")
    lines += ["", "End-to-end projection (not measured model execution)", "",
              "| Phase | Tokens | Prior context | Sampled subtotal (s / cycles) | With allowance (s / cycles) | Tokens/s |",
              "| --- | ---: | ---: | --- | --- | ---: |"]
    for p in predictions:
        lines.append(f"| {p['phase']} | {p['tokens']} | {p['context']} | {number(p['subtotal_wall_s'])} / {number(p['subtotal_cycles'], 0)} | {number(p['predicted_wall_s'])} / {number(p['predicted_cycles'], 0)} | {number(p['predicted_tokens_per_s'])} |")
    lines += ["", "Projection factors (also saved in report.json):", ""]
    for phase in ("prefill", "decode"):
        lines.append(f"- {phase}: " + "; ".join(f"{name} * {factor:g}" for name, factor, _ in prediction_terms(phase)))
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
    parser.add_argument("--output", type=Path, default=ROOT / "build-dayan/limited-perf")
    parser.add_argument("--client", type=Path, default=ROOT / "build-dayan/bin/llama-dayan-operator-bench")
    parser.add_argument("--perf-trace", type=Path, help="JSON output of the already-running dyperf (required for run)")
    parser.add_argument("--socket", default=os.environ.get("DAYAN_SERVER_SOCKET", "/tmp/dayan-server.sock"))
    parser.add_argument("--simulator", choices=("testcluster", "cmodel"), default="testcluster")
    parser.add_argument("--prefill", type=int, default=1024, help="prefill tokens (default: 1024)")
    parser.add_argument("--context", type=int, default=1024, help="decode prefix tokens, excluding the new token (default: 1024)")
    parser.add_argument("--case", action="append", help="select a case ID from plan; repeat to select several")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=86400, help="per-case wall deadline in seconds (default: 24 hours)")
    parser.add_argument("--overhead-percent", type=float, help="uncalibrated allowance for unmeasured work (default: 20%%, or saved report setting)")
    parser.add_argument("--clock-hz", type=float, help="optional hardware clock for an additional ideal-device estimate; never used to infer cycles")
    parser.add_argument("--resume", action="store_true", help="skip completed cases; require the same executable and run parameters")
    args = parser.parse_args()
    if not (1 <= args.prefill <= 1024 and 0 <= args.context < 65536 and 1 <= args.repeats <= 100 and
            0 <= args.warmup <= 99 and args.repeats + args.warmup <= 100 and 1 <= args.timeout <= 4294967):
        parser.error("invalid shape, repetitions, or timeout")
    if (args.overhead_percent is not None and (not math.isfinite(args.overhead_percent) or args.overhead_percent < 0)) or (
            args.clock_hz is not None and (not math.isfinite(args.clock_hz) or args.clock_hz <= 0)):
        parser.error("invalid overhead or clock")
    cases = make_cases(args.prefill, args.context, args.repeats, args.warmup, args.timeout * 1000)
    if args.case:
        if set(args.case) - {case["id"] for case in cases}:
            parser.error("unknown --case; see plan")
        cases = [case for case in cases if case["id"] in args.case]
    if args.command == "plan":
        for case in cases:
            print(f"{case['id']:20} {case['op']:10} {shape_text(case)}")
        print(f"{len(cases)} cases, {len(cases) * (args.repeats + args.warmup)} kernel launches; no server access.")
        return 0
    directory = args.output.resolve()
    manifest_path = directory / "manifest.json"
    if args.command == "report":
        manifest = json.loads(manifest_path.read_text())
        trace = (args.perf_trace or Path(manifest["perf_trace"])).resolve()
        overhead = args.overhead_percent if args.overhead_percent is not None else manifest.get("overhead_percent", 20)
        return make_report(directory, manifest, trace, overhead, args.clock_hz or manifest.get("clock_hz"))
    if args.perf_trace is None:
        parser.error("run requires --perf-trace from the existing dyperf")
    trace = args.perf_trace.resolve()
    client = args.client.resolve()
    fingerprint = hashlib.sha256(client.read_bytes()).hexdigest()
    configuration = dict(simulator=args.simulator, prefill=args.prefill, context=args.context,
                         cases=cases, client=str(client), client_sha256=fingerprint,
                         backend_sha256=hashlib.sha256((client.parent / "libggml-dayan.so").read_bytes()).hexdigest()
                         if (client.parent / "libggml-dayan.so").exists() else None,
                         perf_trace=str(trace), socket=args.socket)
    if manifest_path.exists():
        if not args.resume:
            parser.error("output already contains a run; use --resume or a new directory")
        manifest = json.loads(manifest_path.read_text())
        if any(manifest[key] != value for key, value in configuration.items()):
            parser.error("resume parameters or executable changed; use the original options or a new directory")
    else:
        directory.mkdir(parents=True, exist_ok=True)
        manifest = dict(schema=1, run_id=uuid.uuid4().hex[:7], created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        overhead_percent=20 if args.overhead_percent is None else args.overhead_percent,
                        clock_hz=args.clock_hz, **configuration)
        write_json(manifest_path, manifest)
    env = dict(os.environ, DAYAN_SERVER_SOCKET=args.socket)
    for index, case in enumerate(cases):
        result_path = directory / (case["id"] + ".result.json")
        if result_path.exists():
            prior = json.loads(result_path.read_text())
            if prior.get("status") == "ok":
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

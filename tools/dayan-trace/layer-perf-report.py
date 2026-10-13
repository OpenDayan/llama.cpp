#!/usr/bin/env python3
"""Analyze completed single-layer dyperf traces without connecting to a device."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys

PERIOD = 1 << 48


def trace_events(path, metadata):
    """Stream events, but require a complete dyperf JSON document."""
    decoder = json.JSONDecoder()
    with path.open() as stream:
        buffer = stream.read(65536)
        header = re.match(r'\s*\{\s*"traceEvents"\s*:\s*\[', buffer)
        if not header:
            raise ValueError("expected a dyperf traceEvents array")
        buffer = buffer[header.end():]
        first = True
        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                tail = json.loads('{"traceEvents":[]' + buffer[1:] + stream.read())
                metadata.update(tail.get("metadata", {}))
                return
            if not first:
                if not buffer.startswith(","):
                    if buffer:
                        raise ValueError("missing comma between trace events")
                    more = stream.read(65536)
                    if not more:
                        raise ValueError("trace is incomplete; wait for dyperf to exit")
                    buffer += more
                    continue
                buffer = buffer[1:].lstrip()
            while True:
                buffer = buffer.lstrip()
                try:
                    event, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    more = stream.read(65536)
                    if not more:
                        raise ValueError("trace is incomplete or malformed") from None
                    buffer += more
                    if len(buffer) > 1024 * 1024:
                        raise ValueError("trace event exceeds 1 MiB") from None
            if not isinstance(event, dict):
                raise ValueError("invalid trace event")
            yield event
            buffer = buffer[end:]
            first = False


def relative(counter, anchor):
    return (counter - anchor + PERIOD // 2) % PERIOD - PERIOD // 2


def read_segments(path, samples, slots, include_outer=False):
    tags = {sample["tag"] for sample in samples}
    if len(tags) != len(samples):
        raise ValueError("sample tags must be unique")
    names = [slot["name"] for slot in slots]
    if not names or len(set(names)) != len(names):
        raise ValueError("slot names must be nonempty and unique")
    groups = {tag: {} for tag in tags}
    if include_outer:
        groups.setdefault("lp-setup", {})
    active = {}
    tracked = {}
    cores = {}
    metadata = {}
    threads = {}
    for event in trace_events(path, metadata):
        key = (event.get("pid"), event.get("tid"))
        pid = key[0]
        if event.get("ph") == "M":
            if event.get("name") == "process_name":
                cores[pid] = event.get("args", {}).get("name", str(pid))
            if event.get("name") != "thread_name":
                continue
            name = event.get("args", {}).get("name")
            threads[key] = dict(name=name, start=None, end=None)
            if name in groups:
                if pid in groups[name]:
                    raise ValueError(f"duplicate launch tag {name} on core {pid}")
                group = {"tag": name, "segments": [], "launch_start": None}
                groups[name][pid] = active[pid] = group
                tracked[key] = (group, None)
            elif pid in active:
                group = active[pid]
                index = len(group["segments"])
                if index < len(names) and name == names[index]:
                    segment = dict(name=name, start=None, end=None, pending=None, states=Counter(), exited=False)
                    group["segments"].append(segment)
                    tracked[key] = (group, segment)
            continue
        if key not in threads or event.get("cat") != "proc_state":
            continue
        counter = event.get("args", {}).get("counter")
        if type(counter) is not int or not 0 <= counter < PERIOD:
            raise ValueError("missing or invalid raw 48-bit cycle counter")
        state, phase = event.get("name"), event.get("ph")
        thread = threads[key]
        if state == "RUNNING" and phase == "B" and thread["start"] is None:
            thread["start"] = counter
        if state == "WAIT_EXIT" and phase == "B":
            thread["end"] = counter
        if key not in tracked:
            continue
        group, segment = tracked[key]
        if segment is None:
            if state == "RUNNING" and phase == "B" and group["launch_start"] is None:
                group["launch_start"] = counter
            continue
        if segment["exited"]:
            continue
        if state == "WAIT_EXIT" and phase == "B":
            if segment["pending"] is not None:
                raise ValueError("unclosed state at root exit")
            segment["end"] = counter
            segment["exited"] = True
            if segment is group["segments"][-1] and segment["name"] == names[-1]:
                active.pop(pid, None)
            continue
        if phase == "B":
            if segment["pending"] is not None:
                raise ValueError("overlapping states on a root proc")
            if segment["start"] is None:
                if state != "RUNNING":
                    raise ValueError("renamed mega step must start in RUNNING")
                segment["start"] = counter
            segment["pending"] = (state, counter)
        elif phase == "E":
            pending = segment["pending"]
            if pending is None or pending[0] != state:
                raise ValueError("unmatched root state end")
            duration = (counter - pending[1]) % PERIOD
            if duration >= PERIOD // 2:
                raise ValueError("ambiguous counter wrap or reversed timestamps")
            segment["states"][state] += duration
            segment["end"] = counter
            segment["pending"] = None
    for tag, group in groups.items():
        if not group and tag not in tags:
            continue
        if len(group) != 4:
            raise ValueError(f"{tag}: expected four cores, found {len(group)}")
        for pid, core in group.items():
            segments = core["segments"]
            if core["launch_start"] is None or len(segments) != len(names) or not segments[-1]["exited"]:
                raise ValueError(f"{tag}, core {pid}: missing launch, slot or final root exit")
            previous_end = core["launch_start"]
            for index, segment in enumerate(segments):
                if segment["start"] is None or segment["end"] is None or segment["pending"] is not None:
                    raise ValueError(f"{tag}, core {pid}: incomplete slot {segment['name']}")
                duration = (segment["end"] - segment["start"]) % PERIOD
                if duration >= PERIOD // 2 or sum(segment["states"].values()) != duration:
                    raise ValueError(f"{tag}, core {pid}: inconsistent slot state durations")
                if relative(segment["start"], previous_end) < 0 or (segment["exited"] and index != len(names) - 1):
                    raise ValueError(f"{tag}, core {pid}: invalid step order or early root exit")
                previous_end = segment["end"]
    threads = {key: value for key, value in threads.items() if key not in tracked}
    return groups, cores, metadata, threads


def kernel_spans(groups, cores, threads):
    services = {"scheduler", "PCU snapshots", "rpc-receive", "rpc-send", "profile-flush"}
    first = next(group for group in groups.values() if group)
    anchor = next(iter(first.values()))["launch_start"]
    per_core = {pid: [] for pid in first}
    for tag, group in groups.items():
        for pid, root in group.items():
            per_core[pid].append(dict(name=tag, kind="mega", start=root["launch_start"], end=root["segments"][-1]["end"]))
    for (pid, _), thread in threads.items():
        if pid in per_core and thread["name"] not in services and thread["start"] is not None:
            per_core[pid].append(dict(thread, kind="standalone"))
    for pid, candidates in per_core.items():
        for root in candidates:
            root["start"] = relative(root["start"], anchor)
            if root["end"] is not None:
                root["end"] = relative(root["end"], anchor)
        candidates.sort(key=lambda root: (root["start"], -root["end"] if root["end"] is not None else 0))
        roots = []
        for root in candidates:
            # Layer launches are serial. Child threads run inside their root span.
            if roots and root["start"] < roots[-1]["end"]:
                if root["end"] is not None and root["end"] > roots[-1]["end"]:
                    raise ValueError(f"core {pid}: overlapping root or child spans")
                continue
            if root["end"] is None or not 0 <= root["end"] - root["start"] < PERIOD // 2:
                raise ValueError(f"core {pid}: missing exit or invalid span for {root['name']}")
            roots.append(root)
        per_core[pid] = roots
    sequence = [(root["name"], root["kind"]) for root in next(iter(per_core.values()))]
    if any([(root["name"], root["kind"]) for root in roots] != sequence for roots in per_core.values()):
        raise ValueError("kernel launch sequence differs across cores; trace may be incomplete")
    launches = []
    for index, (name, kind) in enumerate(sequence):
        tasks = [dict(core=cores[pid], **roots[index]) for pid, roots in per_core.items()]
        start, end = min(task["start"] for task in tasks), max(task["end"] for task in tasks)
        if launches and start < launches[-1]["end"]:
            raise ValueError("kernel launches overlap across cores")
        launches.append(dict(name=name, kind=kind, start=start, end=end, cycles=end - start, cores=tasks))
    return launches


def inference_kernels(launches):
    mega = [index for index, launch in enumerate(launches) if launch["kind"] == "mega"]
    if len(mega) == len(launches):
        return {}
    starts = [index - 1 for index in mega]
    if not starts or starts[0] != 0 or any(index < 0 or launches[index]["name"] != "get-rows" for index in starts):
        raise ValueError("missing embedding kernel before layer mega")
    result = {}
    for i, index in enumerate(mega):
        end = starts[i + 1] if i + 1 < len(starts) else len(launches)
        kernels = launches[starts[i]:end]
        if kernels[-1]["name"] != "ggml-mul-mat":
            raise ValueError("missing output projection after layer mega")
        result[launches[index]["name"]] = kernels
    return result


def analyze(run, trace, clock_hz=None, include_outer=False):
    if run.get("inference_status", run.get("status")) != "ok" or not run.get("samples"):
        raise ValueError("inference did not complete successfully")
    spec = run["case"]
    comparison = run.get("comparison", {"reference": "CPU", "status": "not-run"})
    comparison_ok = not spec.get("cmp", False) or (comparison.get("reference") == "CPU" and comparison.get("status") == "pass")
    samples = run["samples"]
    expected = 1 if spec["mode"] == "prefill" else spec["output_len"]
    if len(samples) != expected or any(sample.get("mega_launches") != 1 for sample in samples):
        raise ValueError("sample count or mega launch count does not match the run")
    groups, cores, metadata, threads = read_segments(trace, samples, run["slots"], include_outer)
    inference = inference_kernels(kernel_spans(groups, cores, threads)) if include_outer else {}
    timed = spec.get("simulator") in ("hardware", "testcluster") or clock_hz is not None
    frequency = clock_hz if clock_hz is not None else metadata.get("clock_hz") if timed else None
    if frequency is not None and (not math.isfinite(frequency) or frequency <= 0):
        raise ValueError("clock frequency must be positive and finite")
    functional = spec.get("simulator") == "cmodel"

    def microseconds(cycles):
        return cycles * 1e6 / frequency if frequency and not functional else None

    measured = []
    for sample in samples:
        group = groups[sample["tag"]]
        anchor = next(iter(group.values()))["segments"][0]["start"]
        start = min(relative(core["segments"][0]["start"], anchor) for core in group.values())
        end = max(relative(core["segments"][-1]["end"], anchor) for core in group.values())
        launch_start = min(relative(core["launch_start"], anchor) for core in group.values())
        if not 0 <= end - launch_start < PERIOD // 2:
            raise ValueError("launch spans an ambiguous counter wrap")
        rows = []
        for index, slot in enumerate(run["slots"]):
            per_core = []
            for pid, core in group.items():
                segment = core["segments"][index]
                first, last = relative(segment["start"], anchor), relative(segment["end"], anchor)
                if last < first or first < start or last > end:
                    raise ValueError("inconsistent cross-core timestamps")
                per_core.append(dict(core=cores[pid], start=first - start, end=last - start, cycles=last - first,
                                     states=dict(segment["states"])))
            span = max(core["end"] for core in per_core) - min(core["start"] for core in per_core)
            rows.append(dict(slot, index=index, cycles=span, device_us=microseconds(span),
                             percent_of_layer=100 * span / (end - start) if end > start else 0, cores=per_core))
        kernels = [dict(kernel, index=i, device_us=microseconds(kernel["cycles"]))
                   for i, kernel in enumerate(inference.get(sample["tag"], []))]
        full = kernels[-1]["end"] - kernels[0]["start"] if kernels else None
        if full is not None and not 0 <= full < PERIOD // 2:
            raise ValueError("inference spans an ambiguous counter wrap")
        kernel_cycles = sum(kernel["cycles"] for kernel in kernels) if kernels else None
        for kernel in kernels:
            kernel["percent_of_inference"] = 100 * kernel["cycles"] / full if full else 0
        measured.append(dict(sample, layer_cycles=end - start, layer_device_us=microseconds(end - start),
                             launch_cycles=end - launch_start, slots=rows, kernels=kernels,
                             inference_cycles=full, inference_device_us=microseconds(full) if full is not None else None,
                             kernel_cycles=kernel_cycles, inter_kernel_cycles=full - kernel_cycles if kernels else None))
    total = sum(sample["layer_cycles"] for sample in measured)
    aggregate = []
    for index, slot in enumerate(run["slots"]):
        cycles = [sample["slots"][index]["cycles"] for sample in measured]
        median = statistics.median(cycles)
        aggregate.append(dict(slot, index=index, count=len(cycles), median_cycles=median, min_cycles=min(cycles),
                              max_cycles=max(cycles), median_device_us=microseconds(median),
                              percent_of_layer=100 * sum(cycles) / total if total else 0))
    full_total = sum(sample["inference_cycles"] for sample in measured) if all(sample["kernels"] for sample in measured) else None
    return dict(schema=1, status="ok" if comparison_ok else "failed", case=spec, perf_trace=str(trace), trace_metadata=metadata, clock_hz=frequency,
                comparison=comparison, performance_valid=timed and not functional and comparison_ok, samples=measured, slots=aggregate,
                total_layer_cycles=total, total_layer_device_us=microseconds(total), include_outer=include_outer,
                total_inference_cycles=full_total, total_inference_device_us=microseconds(full_total) if full_total is not None else None,
                notes=[
                    "A slot is a named mega device step, including its following barriers. Paired/reused and optional/no-op steps remain separate.",
                    "Slot cycles = latest core end - earliest core start. Core durations and child threads are not added to this span.",
                    "Layer cycles run from the first named step to final root WAIT_EXIT. Launch cycles also include root startup before the first rename.",
                    "Slot spans can overlap across cores; their sum is not the layer duration. Per-core states describe root scheduling, not MAT/VEC/LSU utilization.",
                    "Slot percentages use raw cycles: sum of that slot's spans / sum of complete Transformer layer spans. They can sum above 100% when slot spans overlap. Embedding and output kernels are excluded from this denominator.",
                    "Counters must share a time origin across cores. Each launch must fit within half the 48-bit counter period.",
                    "Layer timing excludes embedding, output norm/projection, host submission and reply. Host inference wall time includes those operations and active profiling.",
                    *([
                        "Full inference spans the first embedding root RUNNING to the last output projection root WAIT_EXIT entry, including gaps between launches. Host preparation before that first root and output readback after that last root are excluded.",
                        "Kernel percentages use that full inference span, separately from layer slot percentages. Child threads are excluded; mega slots are a breakdown, not additional launches. kernel_cycles sums launches; inter_kernel_cycles records the intervening gaps.",
                        "Full inference is unavailable for older traces containing only mega steps. Trace boundaries require embedding and output projection on all four cores; decode setup is excluded from measured totals.",
                    ] if include_outer else []),
                    "Decode prompt setup is executed but excluded. Every decode step has its own context length and slot timings.",
                    "cmodel counters validate trace analysis only; device microseconds are suppressed." if functional else
                    "Device microseconds use the selected clock frequency and require hardware or a cycle-accurate simulator. Host wall time is not hardware latency." if timed else
                    "Counter source is controlled externally. Raw counter ticks are reported; use --clock-hz for time conversion only with hardware or a cycle-accurate simulator.",
                ])


def report_layer(directory, clock_hz, output=None, include_outer=False, trace=None, modes=None):
    run = json.loads((directory / "run.json").read_text())
    if modes is not None and set(modes) != {run["case"]["mode"]}:
        raise ValueError("requested modes do not match this single-mode run")
    report = analyze(run, trace or directory / "perf.json", clock_hz, include_outer)
    output = output or directory
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with (output / "slots.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tag", "step", "context_before", "index", "name", "kind", "cycles", "device_us", "percent_of_layer"])
        writer.writeheader()
        for sample in report["samples"]:
            for slot in sample["slots"]:
                writer.writerow({key: (slot if key in slot else sample)[key] for key in writer.fieldnames})
    lines = ["# Transformer layer timing", "", f"Mode: {report['case']['mode']}; input: {report['case']['input_len']}; output: {report['case']['output_len']}; counter source: {report['case'].get('simulator', 'external')}", "",
             "Complete layer: first mega slot through final residual, including following barriers. Embedding and output norm/head are excluded. Host inference time includes them.", "",
             f"Total measured layer cycles: {report['total_layer_cycles']:,}.", "",
             "| Step | Context before | Layer cycles | Device us | Host inference s |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for sample in report["samples"]:
        time = "n/a" if sample["layer_device_us"] is None else f"{sample['layer_device_us']:.3f}"
        lines.append(f"| {sample['tag']} | {sample['context_before']} | {sample['layer_cycles']:,} | {time} | {sample['host_inference_wall_s']:.6f} |")
    lines += ["", "| Slot | Kind | Median cycles | Min | Max | Layer cycles (%) | Median device us |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for slot in report["slots"]:
        time = "n/a" if slot["median_device_us"] is None else f"{slot['median_device_us']:.3f}"
        lines.append(f"| {slot['name']} | {slot['kind']} | {slot['median_cycles']:,} | {slot['min_cycles']:,} | {slot['max_cycles']:,} | {slot['percent_of_layer']:.2f}% | {time} |")
    if include_outer:
        with (output / "kernels.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["tag", "step", "context_before", "index", "name", "kind", "cycles", "device_us", "percent_of_inference"])
            writer.writeheader()
            for sample in report["samples"]:
                for kernel in sample["kernels"]:
                    writer.writerow({key: (kernel if key in kernel else sample)[key] for key in writer.fieldnames})
        lines += ["", "Full inference (including embedding and output head; separate from layer totals):", "",
                  "| Step | Full cycles | Kernel cycles | Between kernels | Gap (%) | Device us |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for sample in report["samples"]:
            time = "n/a" if sample["inference_device_us"] is None else f"{sample['inference_device_us']:.3f}"
            counts = [f"{sample[key]:,}" if sample[key] is not None else "n/a" for key in ("inference_cycles", "kernel_cycles", "inter_kernel_cycles")]
            gap = f"{100 * sample['inter_kernel_cycles'] / sample['inference_cycles']:.2f}%" if sample["inference_cycles"] else "n/a"
            lines.append(f"| {sample['tag']} | {' | '.join(counts)} | {gap} | {time} |")
        lines += ["", "| Step | Index | Kernel | Cycles | Full inference cycles (%) | Device us |",
                  "| --- | ---: | --- | ---: | ---: | ---: |"]
        for sample in report["samples"]:
            for kernel in sample["kernels"]:
                time = "n/a" if kernel["device_us"] is None else f"{kernel['device_us']:.3f}"
                lines.append(f"| {sample['tag']} | {kernel['index']} | {kernel['name']} | {kernel['cycles']:,} | {kernel['percent_of_inference']:.2f}% | {time} |")
    if report["case"].get("cmp"):
        cmp = report["comparison"]
        lines += ["", f"CPU comparison: {cmp.get('status', 'not-run')}. Criterion: abs(Dayan - CPU) <= {cmp.get('atol', 'n/a')} + {cmp.get('rtol', 'n/a')} * abs(CPU).", "",
                  "| Step | Phase | Status | Checked logits | Mismatches | Max abs | Max rel | Relative L2 |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for step in cmp.get("steps", []):
            lines.append(f"| {step['step']} | {step['phase']} | {step['status']} | {step['checked_elements']} | {step['mismatched_elements']} | {step['max_abs_error']:.6g} | {step['max_rel_error']:.6g} | {step['relative_l2_error']:.6g} |")
        if cmp.get("error"):
            lines += ["", cmp["error"]]
        lines += ["", "CPU replay runs after perf capture with GPU layers, KV offload and operation offload disabled. It uses the same stored weights and Dayan token sequence. Decode prompt logits are also checked. CPU and Dayan activation quantization can differ; these differences remain in the reported errors."]
    lines += ["", *["- " + note for note in report["notes"]], ""]
    (output / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return report


def report_batch(directory, clock_hz, output=None, include_outer=False, modes=None):
    output = output or directory
    batch = json.loads((directory / "batch.json").read_text())
    key = "runs" if batch.get("schema") == 2 else "layers"
    if batch.get("schema") not in (1, 2) or not batch.get(key):
        raise ValueError("invalid batch manifest")
    entries = []
    for original in batch[key]:
        entry = dict(original)
        child = entry["directory"]
        expected = f"layer-{entry['source_layer']}"
        if key == "runs":
            if entry.get("mode") not in ("prefill", "decode"):
                raise ValueError("invalid batch mode")
            expected += f"/{entry['mode']}"
        if child != expected:
            raise ValueError("invalid batch layer directory")
        if modes is not None and "mode" not in entry:
            case = directory / child / "case.json"
            run = directory / child / "run.json"
            if case.is_file():
                entry["mode"] = json.loads(case.read_text())["mode"]
            elif run.is_file():
                entry["mode"] = json.loads(run.read_text())["case"]["mode"]
        entries.append(entry)
    if modes is not None:
        available = {entry.get("mode") for entry in entries}
        if not set(modes) <= available:
            raise ValueError("requested modes are not present in this batch")
        entries = [entry for entry in entries if entry.get("mode") in modes or "mode" not in entry]
    results = []
    for entry in entries:
        child = entry["directory"]
        result = dict(source_layer=entry["source_layer"], status=entry["status"], report=None)
        if "mode" in entry:
            result["mode"] = entry["mode"]
        results.append(result)
        if entry["status"] == "not-run":
            continue
        print(f"\nSource layer {entry['source_layer']} {entry.get('mode', '')}:")
        try:
            measured = report_layer(directory / child, clock_hz, output / child, include_outer,
                                    modes=[entry["mode"]] if "mode" in entry else None)
            cycles = [sample["layer_cycles"] for sample in measured["samples"]]
            result.update(status="ok" if measured["status"] == entry["status"] == "ok" else "failed",
                          mode=measured["case"]["mode"], report=f"{child}/report.md", steps=len(cycles),
                          total_layer_cycles=measured["total_layer_cycles"], median_layer_cycles=statistics.median(cycles),
                          min_layer_cycles=min(cycles), max_layer_cycles=max(cycles),
                          cpu_comparison=measured["comparison"].get("status", "not-run"))
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="failed", error=str(error))
            print(f"layer-perf-report: layer {entry['source_layer']} {entry.get('mode', '')}: {error}", file=sys.stderr)
    passed = (modes is not None or batch["status"] == "ok") and all(result["status"] == "ok" for result in results)
    summary = dict(schema=batch["schema"], status="ok" if passed else "failed", **{key: results})
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    fields = ["source_layer", "mode", "status", "steps", "total_layer_cycles", "median_layer_cycles", "min_layer_cycles", "max_layer_cycles", "cpu_comparison", "report"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    lines = ["# Layer batch", "", f"Status: {summary['status']}", "",
             "Prefill and decode are separate measurements. Decode totals exclude prompt setup; medians are per step. Cycles from different modes or source layers are not added together.", "",
             "| Source layer | Mode | Status | Steps | Total layer cycles | Median cycles/step | Min | Max | CPU comparison | Report |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for result in results:
        counts = [f"{result[field]:,}" if field in result else "-" for field in fields[3:8]]
        link = f"[Details]({result['report']})" if result["report"] else "-"
        lines.append(f"| {result['source_layer']} | {result.get('mode', '-')} | {result['status']} | {' | '.join(counts)} | {result.get('cpu_comparison', '-')} | {link} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="completed single-layer or batch run directory")
    parser.add_argument("--mode", nargs="+", choices=("prefill", "decode"), help="report only these modes (default: all recorded modes)")
    parser.add_argument("--trace", type=Path, help="override perf.json location")
    parser.add_argument("--output", type=Path, help="report destination (default: run directory); leaves source files untouched when separate")
    parser.add_argument("--include-outer", action="store_true", help="also report embedding/output kernels and the full inference span, separately from Transformer layer timing")
    parser.add_argument("--clock-hz", type=float, help="enable time conversion at this frequency for hardware or cycle-accurate simulation")
    args = parser.parse_args(argv)
    try:
        if args.mode is not None and len(set(args.mode)) != len(args.mode):
            raise ValueError("select unique report modes")
        if (args.run / "batch.json").is_file():
            if args.trace:
                raise ValueError("--trace requires a single-layer run directory")
            return report_batch(args.run, args.clock_hz, args.output, args.include_outer, args.mode)
        result = report_layer(args.run, args.clock_hz, args.output, args.include_outer, args.trace, args.mode)
        return 0 if result["status"] == "ok" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"layer-perf-report: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

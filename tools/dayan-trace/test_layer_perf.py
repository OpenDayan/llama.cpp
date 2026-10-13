import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("layer_report", Path(__file__).with_name("layer-perf-report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
spec = importlib.util.spec_from_file_location("layer_runner", Path(__file__).with_name("layer-perf.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(tags=("lp-prefill",), anchor=100, stride=100):
    events = []
    slots = [dict(name="norm0", kind="norm_pipeline"), dict(name="residual", kind="norm_pipeline")]
    run = dict(status="ok", case=dict(mode="prefill" if len(tags) == 1 else "decode", simulator="hardware",
                                    input_len=4, output_len=0 if len(tags) == 1 else len(tags)),
               slots=slots, samples=[dict(tag=tag, step=i, context_before=i + 4, mega_launches=1,
                                         host_inference_wall_s=0.5) for i, tag in enumerate(tags)])
    for core in range(4):
        events.append(dict(name="process_name", ph="M", pid=core, tid=0, args=dict(name=f"cluster 0 hart {core}")))
        for step, tag in enumerate(tags):
            base = anchor + core + stride * step
            tid = 10 * step

            def name(value):
                events.append(dict(name="thread_name", ph="M", pid=core, tid=tid, args=dict(name=value)))

            def state(value, phase, tick):
                events.append(dict(name=value, ph=phase, cat="proc_state", pid=core, tid=tid,
                                   args=dict(counter=(base + tick) % report.PERIOD)))

            name(tag)
            state("RUNNING", "B", 0)
            state("RUNNING", "E", 10)
            tid += 1
            name("norm0")
            state("RUNNING", "B", 10)
            state("RUNNING", "E", 14)
            state("WAIT_LSU", "B", 14)
            state("WAIT_LSU", "E", 20)
            tid += 1
            name("residual")
            state("RUNNING", "B", 21)
            state("RUNNING", "E", 41)
            state("WAIT_EXIT", "B", 41)
            state("WAIT_EXIT", "E", 90)
            tid += 1
            name("post")
            state("RUNNING", "B", 22)
            state("RUNNING", "E", 40)
    return run, dict(traceEvents=events, metadata=dict(clock_hz=500000000))


def full_fixture(tags=("lp-prefill",), anchor=100):
    run, trace = fixture(tags, anchor, stride=200)
    for core in range(4):
        for step, _ in enumerate(tags):
            for index, (name, start, end) in enumerate([
                ("get-rows", -20, -10), ("worker-0", -18, -12),
                ("rms-scale", 50, 60), ("get-rows", 65, 68),
                ("copy", 72, 74), ("ggml-mul-mat", 78, 88), ("load-a", 80, 85),
            ]):
                tid = 1000 + step * 10 + index
                trace["traceEvents"].append(dict(name="thread_name", ph="M", pid=core, tid=tid, args=dict(name=name)))
                for state, phase, tick in [("RUNNING", "B", start), ("RUNNING", "E", end), ("WAIT_EXIT", "B", end), ("WAIT_EXIT", "E", end + 1)]:
                    trace["traceEvents"].append(dict(name=state, ph=phase, cat="proc_state", pid=core, tid=tid,
                                                     args=dict(counter=(anchor + core + 200 * step + tick) % report.PERIOD)))
    return run, trace


class LayerPerfTests(unittest.TestCase):
    def analyze(self, run, trace, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "perf.json"
            path.write_text(trace if isinstance(trace, str) else json.dumps(trace))
            return report.analyze(run, path, **kwargs)

    def test_four_core_span_and_wait_exit_boundary(self):
        result = self.analyze(*fixture())
        sample = result["samples"][0]
        self.assertEqual(sample["layer_cycles"], 34)
        self.assertEqual(sample["launch_cycles"], 44)
        self.assertEqual([s["cycles"] for s in sample["slots"]], [13, 23])
        self.assertEqual(sample["slots"][0]["cores"][0]["states"], dict(RUNNING=4, WAIT_LSU=6))
        self.assertAlmostEqual(sample["slots"][0]["device_us"], 0.026)
        self.assertAlmostEqual(sample["slots"][0]["percent_of_layer"], 100 * 13 / 34)
        self.assertAlmostEqual(result["slots"][1]["percent_of_layer"], 100 * 23 / 34)
        self.assertGreater(sum(slot["percent_of_layer"] for slot in result["slots"]), 100)

    def test_outer_kernels_are_optional_and_do_not_change_layer_totals(self):
        run, trace = full_fixture()
        result = self.analyze(run, trace)
        self.assertEqual(result["total_layer_cycles"], 34)
        self.assertIsNone(result["total_inference_cycles"])
        self.assertEqual(result["samples"][0]["kernels"], [])
        included = self.analyze(run, trace, include_outer=True)
        sample = included["samples"][0]
        self.assertEqual(included["slots"], result["slots"])
        self.assertEqual(sample["inference_cycles"], 111)
        self.assertEqual(sample["kernel_cycles"], 94)
        self.assertEqual(sample["inter_kernel_cycles"], 17)
        self.assertEqual([kernel["name"] for kernel in sample["kernels"]],
                         ["get-rows", "lp-prefill", "rms-scale", "get-rows", "copy", "ggml-mul-mat"])
        self.assertEqual([kernel["cycles"] for kernel in sample["kernels"]], [13, 44, 13, 6, 5, 13])
        self.assertAlmostEqual(sample["kernels"][0]["percent_of_inference"], 100 * 13 / 111)

    def test_complete_decode_spans_exclude_setup_and_handle_counter_wrap(self):
        run, trace = full_fixture(("lp-setup", "lp-d0000000001", "lp-d0000000002"), report.PERIOD - 10)
        run["setup"] = run["samples"].pop(0)
        run["case"]["output_len"] = 2
        result = self.analyze(run, trace, include_outer=True)
        self.assertEqual(result["total_layer_cycles"], 68)
        self.assertEqual(result["total_inference_cycles"], 222)
        self.assertEqual([sample["inference_cycles"] for sample in result["samples"]], [111, 111])
        self.assertEqual([sample["kernels"][1]["name"] for sample in result["samples"]], ["lp-d0000000001", "lp-d0000000002"])
        self.assertAlmostEqual(result["slots"][0]["percent_of_layer"], 100 * 26 / 68)

    def test_outer_missing_core_and_exit_are_rejected(self):
        run, trace = full_fixture()
        for keep in (lambda event: not (event["pid"] == 0 and event["tid"] == 1000),
                     lambda event: not (event["pid"] == 0 and event["tid"] == 1005 and event["name"] == "WAIT_EXIT")):
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                self.analyze(run, dict(trace, traceEvents=[event for event in trace["traceEvents"] if keep(event)]), include_outer=True)

    def test_outer_requires_embedding_and_output_boundaries(self):
        run, trace = full_fixture()
        for tid in (1000, 1005):
            with self.subTest(tid=tid), self.assertRaises(ValueError):
                self.analyze(run, dict(trace, traceEvents=[event for event in trace["traceEvents"] if event["tid"] != tid]), include_outer=True)

    def test_separate_output_preserves_source_and_exports_percentages(self):
        run, trace = full_fixture()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            source, output = Path(directory) / "source", Path(directory) / "report"
            source.mkdir()
            (source / "run.json").write_text(json.dumps(run))
            (source / "perf.json").write_text(json.dumps(trace))
            (source / "report.md").write_text("keep existing report")
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            self.assertEqual(report.main(["--run", str(source), "--output", str(output), "--include-outer"]), 0)
            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, before)
            self.assertIn("Total measured layer cycles: 34.", (output / "report.md").read_text())
            self.assertIn("38.24%", (output / "report.md").read_text())
            self.assertIn("percent_of_layer", (output / "slots.csv").read_text())
            self.assertIn("percent_of_inference", (output / "kernels.csv").read_text())
            self.assertEqual(len((output / "kernels.csv").read_text().splitlines()), 7)

    def test_mega_only_trace_does_not_claim_complete_inference(self):
        result = self.analyze(*fixture(), include_outer=True)
        self.assertIsNone(result["total_inference_cycles"])

    def test_counter_wrap_and_multiple_steps(self):
        result = self.analyze(*fixture(("lp-d0000000001", "lp-d0000000002"), report.PERIOD - 20))
        self.assertEqual(result["total_layer_cycles"], 68)
        self.assertEqual([s["count"] for s in result["slots"]], [2, 2])

    def test_percentage_uses_total_cycles_across_different_decode_lengths(self):
        run, trace = fixture(("lp-d0000000001", "lp-d0000000002"), stride=300)
        for event in trace["traceEvents"]:
            if event["tid"] == 12 and ((event["name"] == "RUNNING" and event["ph"] == "E") or
                                        (event["name"] == "WAIT_EXIT" and event["ph"] == "B")):
                event["args"]["counter"] += 40
        result = self.analyze(run, trace)
        self.assertEqual([sample["layer_cycles"] for sample in result["samples"]], [34, 74])
        self.assertAlmostEqual(result["slots"][0]["percent_of_layer"], 100 * 26 / 108)

    def test_cmodel_has_no_device_latency(self):
        run, trace = fixture()
        run["case"]["simulator"] = "cmodel"
        result = self.analyze(run, trace)
        self.assertFalse(result["performance_valid"])
        self.assertIsNone(result["total_layer_device_us"])

    def test_external_counter_requires_explicit_clock(self):
        run, trace = fixture()
        del run["case"]["simulator"]
        result = self.analyze(run, trace)
        self.assertFalse(result["performance_valid"])
        self.assertIsNone(result["total_layer_device_us"])
        result = self.analyze(run, trace, clock_hz=1000000000)
        self.assertTrue(result["performance_valid"])
        self.assertAlmostEqual(result["total_layer_device_us"], 0.034)

    def test_unmeasured_prefix_is_ignored(self):
        run, trace = fixture(("lp-setup", "lp-d0000000001"))
        run["samples"] = run["samples"][1:]
        run["case"]["output_len"] = 1
        self.assertEqual(self.analyze(run, trace)["total_layer_cycles"], 34)

    def test_missing_core_slot_and_exit_are_rejected(self):
        run, trace = fixture()
        selectors = [lambda e: e["pid"] != 3,
                     lambda e: not (e["pid"] == 0 and e["tid"] == 1),
                     lambda e: not (e["pid"] == 0 and e["name"] == "WAIT_EXIT")]
        for keep in selectors:
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                self.analyze(run, dict(trace, traceEvents=[e for e in trace["traceEvents"] if keep(e)]))

    def test_truncated_trace_is_rejected(self):
        run, trace = fixture()
        text = json.dumps(trace)
        for end in (50, len(text) // 2, len(text) - 1):
            with self.subTest(end=end), self.assertRaises(ValueError):
                self.analyze(run, text[:end])

    def test_missing_counter_and_unmatched_state_are_rejected(self):
        run, trace = fixture()
        for change in ("counter", "state"):
            broken = copy.deepcopy(trace)
            event = next(e for e in broken["traceEvents"] if e["pid"] == 0 and e["tid"] == 1 and e["ph"] == "E")
            if change == "counter":
                event["args"] = {}
            else:
                event["name"] = "WAIT_VEC"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.analyze(run, broken)

    def test_large_trace_streaming(self):
        run, trace = fixture()
        trace["traceEvents"][:0] = [dict(name="ignored", ph="M", args=dict(name="x" * 63))] * 2000
        self.assertEqual(self.analyze(run, trace)["total_layer_cycles"], 34)

    def test_bad_run_and_clock_are_rejected(self):
        run, trace = fixture()
        for frequency in (0, -1, float("nan")):
            with self.subTest(clock=frequency), self.assertRaises(ValueError):
                self.analyze(run, trace, clock_hz=frequency)
        run["samples"][0]["mega_launches"] = 0
        with self.assertRaises(ValueError):
            self.analyze(run, trace)

    def test_requested_cpu_comparison_must_pass(self):
        run, trace = fixture()
        run["case"]["cmp"] = True
        for comparison in ({}, {"reference": "CPU", "status": "fail"}, {"reference": "DAYAN", "status": "pass"}):
            with self.subTest(comparison=comparison):
                run["comparison"] = comparison
                result = self.analyze(run, trace)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["performance_valid"])
        run["comparison"] = {"reference": "CPU", "status": "pass"}
        self.assertEqual(self.analyze(run, trace)["status"], "ok")

    def test_failed_comparison_keeps_timing_report_and_fails_cli(self):
        run, trace = fixture()
        run.update(status="failed", inference_status="ok", comparison=dict(reference="CPU", status="fail"))
        run["case"]["cmp"] = True
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            path = Path(directory)
            (path / "run.json").write_text(json.dumps(run))
            (path / "perf.json").write_text(json.dumps(trace))
            self.assertEqual(report.main(["--run", directory]), 1)
            self.assertIn("CPU comparison: fail", (path / "report.md").read_text())
            self.assertTrue((path / "slots.csv").is_file())

    def test_cpu_process_failure_cannot_pass_comparison(self):
        for process_status in (0, 1):
            with self.subTest(process_status=process_status), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                case = dict(cmp=True)
                run = dict(status="ok", case=case)
                result = path / "run.json"
                runner.write_json(result, run)
                runner.write_json(path / "cpu-reference.json", dict(status="ok", backend="CPU", case=case,
                                                                     comparison=dict(reference="CPU", status="pass")))
                with patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=process_status)) as execute:
                    self.assertEqual(runner.compare_cpu(path / "client", path / "case.json", result), process_status == 0)
                    self.assertIn("--cpu-reference", execute.call_args.args[0])
                saved = json.loads(result.read_text())
                self.assertEqual(saved["inference_status"], "ok")
                self.assertEqual(saved["status"], "ok" if process_status == 0 else "failed")


class LayerBatchTests(unittest.TestCase):
    def prepare(self, directory):
        for layer in (0, 3):
            path = directory / f"layer-{layer}"
            path.mkdir(parents=True)
            (path / "layer.gguf").touch()
            runner.write_json(path / "layer.json", dict(schema=1, source_layer=layer, model=str(path / "layer.gguf")))

    def test_selection_and_single_layer_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.prepare(path)
            for selected, expected in ((None, [0, 3]), ([3, 0], [3, 0]), ([3], [3])):
                batch, models = runner.selected_models(path, selected)
                self.assertTrue(batch)
                self.assertEqual([m["source_layer"] for m in models], expected)
            for selected in (None, [3]):
                batch, models = runner.selected_models(path / "layer-3", selected)
                self.assertFalse(batch)
                self.assertEqual(models[0]["source_layer"], 3)
            for selected in ([0, 0], [-1], [0, 2]):
                with self.subTest(selected=selected), self.assertRaises((ValueError, OSError)):
                    runner.selected_models(path, selected)
            with self.assertRaises(ValueError):
                runner.selected_models(path / "layer-3", [0, 3])

    def test_batch_report_supports_separate_output_and_outer_kernels(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            source, output = Path(directory) / "source", Path(directory) / "report"
            self.prepare(source)
            batch = dict(schema=1, status="ok", layers=[])
            for layer in (0, 3):
                run, trace = full_fixture()
                child = source / f"layer-{layer}"
                runner.write_json(child / "run.json", run)
                runner.write_json(child / "perf.json", trace)
                batch["layers"].append(dict(source_layer=layer, directory=child.name, status="ok"))
            runner.write_json(source / "batch.json", batch)
            before = {str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}
            self.assertEqual(report.main(["--run", str(source), "--output", str(output), "--include-outer"]), 0)
            self.assertEqual({str(path): path.read_bytes() for path in source.rglob("*") if path.is_file()}, before)
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "ok")
            for layer in (0, 3):
                self.assertTrue((output / f"layer-{layer}/kernels.csv").is_file())

    def test_missing_layer_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
            path = Path(directory)
            self.prepare(path / "prepared")
            with patch.object(runner.subprocess, "run") as execute:
                status = runner.main(["run", "--prepared", str(path / "prepared"), "--layer", "0", "2", "--output", str(path / "output")])
            self.assertEqual(status, 1)
            execute.assert_not_called()
            self.assertFalse((path / "output").exists())

    def test_batch_profiles_then_compares_each_layer_and_reports_all_results(self):
        for failed_layer, pcu in ((None, False), (None, True), (0, False), (0, True)):
            with self.subTest(failed_layer=failed_layer, pcu=pcu), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                prepared, output = path / "prepared", path / "output"
                self.prepare(prepared)
                profiler = path / "build/host/dyperf/dyperf"
                profiler.parent.mkdir(parents=True)
                profiler.symlink_to("/bin/true")
                calls = []

                def execute(command, **kwargs):
                    if command[0] == runner.sys.executable:
                        return SimpleNamespace(returncode=report.main(command[2:]))
                    profiling = command[1] == "--socket"
                    case = Path(command[-2] if profiling else command[1])
                    spec = json.loads(case.read_text())
                    layer = spec["source_layer"]
                    self.assertNotIn("simulator", spec)
                    calls.append(("perf" if profiling else "cpu", layer))
                    if profiling:
                        self.assertEqual(command[0], str(Path("/bin/true").resolve()))
                        if pcu:
                            self.assertEqual(command[command.index("--pcu-interval") + 1], "0")
                        else:
                            self.assertNotIn("--pcu", command)
                            self.assertNotIn("--pcu-interval", command)
                        run, trace = fixture()
                        run["case"] = spec
                        runner.write_json(Path(command[-1]), run)
                        runner.write_json(Path(command[command.index("-o") + 1]), trace)
                    else:
                        self.assertTrue((case.parent / "perf.json").is_file())
                        passed = layer != failed_layer
                        runner.write_json(Path(command[2]), dict(case=spec, backend="CPU", status="ok" if passed else "failed",
                                                                 comparison=dict(reference="CPU", status="pass" if passed else "fail")))
                    return SimpleNamespace(returncode=0)

                with socket.socket(socket.AF_UNIX) as server:
                    server.bind(str(path / "server.sock"))
                    env = dict(DAYAN_SERVER_SOCKET=str(path / "server.sock"), DAYAN_ROOT=str(path), DAYAN_LAYER_CLIENT="/bin/true")
                    with patch.dict(runner.os.environ, env), patch.object(runner.subprocess, "run", side_effect=execute), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        status = runner.main(["run", "--prepared", str(prepared), "--input-len", "4", "--cmp", "--output", str(output)] + (["--pcu"] if pcu else []))
                self.assertEqual(calls, [("perf", 0), ("cpu", 0), ("perf", 3), ("cpu", 3)])
                self.assertEqual(status, int(failed_layer is not None))
                batch = json.loads((output / "batch.json").read_text())
                self.assertEqual(batch["status"], "ok" if status == 0 else "failed")
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(report.main(["--run", str(output)]), status)
                for layer in (0, 3):
                    self.assertTrue((output / f"layer-{layer}/slots.csv").is_file())
                self.assertEqual(json.loads((output / "report.json").read_text())["status"], batch["status"])

    def test_operator_comparison_without_profiler(self):
        for outcome in ("pass", "fail", "incomplete"):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(path / "prepared")
                calls = []

                def execute(command, **kwargs):
                    self.assertEqual(command[-1], "--cmp-ops")
                    self.assertEqual(kwargs["env"]["DAYAN_SERVER_SOCKET"], str(path / "server.sock"))
                    spec = json.loads(Path(command[1]).read_text())
                    self.assertEqual(spec["op_atol"], 1e-4)
                    self.assertNotIn("cmp", spec)
                    calls.append(spec["source_layer"])
                    good = outcome == "pass" or spec["source_layer"] == 3
                    result = dict(status="ok" if good else "failed", operator_comparison=dict(
                        status="pass" if good else "fail", checked_nodes=30, checked_outputs=30))
                    if outcome != "incomplete":
                        result["inference_status"] = "ok"
                    runner.write_json(Path(command[2]), result)
                    return SimpleNamespace(returncode=0 if good else 1)

                with socket.socket(socket.AF_UNIX) as server:
                    server.bind(str(path / "server.sock"))
                    with patch.object(runner.subprocess, "run", side_effect=execute), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        status = runner.main(["cmp-ops", "--prepared", str(path / "prepared"), "--mode", "decode",
                                              "--input-len", "4", "--output-len", "2", "--socket", str(path / "server.sock"),
                                              "--client", "/bin/true", "--output", str(path / "output")])
                self.assertEqual(status, int(outcome != "pass"))
                self.assertEqual(calls, [0] if outcome == "incomplete" else [0, 3])
                self.assertFalse((path / "output/layer-0/perf.json").exists())

    def test_multiple_modes_generate_independent_reports_and_filter_results(self):
        for failure in (None, "comparison", "inference", "trace"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(path / "prepared")
                output = path / "output"
                calls = []

                def execute(command, **kwargs):
                    if command[0] == runner.sys.executable:
                        calls.append(("report",))
                        return SimpleNamespace(returncode=report.main(command[2:]))
                    profiling = command[1] == "--socket"
                    case = Path(command[-2] if profiling else command[1])
                    spec = json.loads(case.read_text())
                    layer, mode = spec["source_layer"], spec["mode"]
                    calls.append(("perf" if profiling else "cpu", layer, mode))
                    self.assertEqual(case.parent, output / f"layer-{layer}" / mode)
                    self.assertEqual(spec["input_len"], 4)
                    self.assertEqual(spec["output_len"], 0 if mode == "prefill" else 2)
                    failing = (layer, mode) == (0, "decode")
                    if profiling:
                        if failing and failure == "inference":
                            return SimpleNamespace(returncode=1)
                        tags = ("lp-prefill",) if mode == "prefill" else ("lp-setup", "lp-d0000000001", "lp-d0000000002")
                        run, trace = full_fixture(tags)
                        run["case"] = spec
                        if mode == "decode":
                            run["samples"] = run["samples"][1:]
                        runner.write_json(Path(command[-1]), run)
                        if not (failing and failure == "trace"):
                            runner.write_json(Path(command[command.index("-o") + 1]), trace)
                    else:
                        passed = not (failing and failure == "comparison")
                        runner.write_json(Path(command[2]), dict(case=spec, backend="CPU", status="ok" if passed else "failed",
                                                                 comparison=dict(reference="CPU", status="pass" if passed else "fail")))
                    return SimpleNamespace(returncode=0)

                with socket.socket(socket.AF_UNIX) as server:
                    server.bind(str(path / "server.sock"))
                    with patch.object(runner.subprocess, "run", side_effect=execute), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        status = runner.main(["run", "--prepared", str(path / "prepared"), "--layer", "0", "3",
                                              "--mode", "prefill", "decode", "--input-len", "4", "--output-len", "2", "--cmp",
                                              "--socket", str(path / "server.sock"), "--client", "/bin/true", "--dyperf", "/bin/true",
                                              "--output", str(output)])
                self.assertEqual(status, int(failure is not None))
                expected = [(stage, layer, mode) for layer in (0, 3) for mode in ("prefill", "decode") for stage in ("perf", "cpu")]
                self.assertEqual(calls, (expected[:3] if failure == "inference" else expected) + [("report",)])
                batch = json.loads((output / "batch.json").read_text())
                self.assertEqual(batch["schema"], 2)
                summary = json.loads((output / "report.json").read_text())
                self.assertEqual(summary["status"], "failed" if status else "ok")
                self.assertEqual(len(summary["runs"]), 4)
                self.assertEqual(summary["runs"][0]["total_layer_cycles"], 34)
                if failure not in ("inference", "trace"):
                    self.assertEqual(summary["runs"][1]["total_layer_cycles"], 68)
                    self.assertEqual(summary["runs"][1]["median_layer_cycles"], 34)
                if failure == "inference":
                    self.assertEqual([entry["status"] for entry in batch["runs"]], ["ok", "failed", "not-run", "not-run"])
                    self.assertFalse((output / "layer-3").exists())
                else:
                    before = {str(file): file.read_bytes() for file in output.rglob("*") if file.is_file()}
                    with contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(report.main(["--run", str(output), "--mode", "prefill", "--include-outer", "--output", str(path / "filtered")]), 0)
                    filtered = json.loads((path / "filtered/report.json").read_text())
                    self.assertEqual([entry["mode"] for entry in filtered["runs"]], ["prefill", "prefill"])
                    self.assertEqual({str(file): file.read_bytes() for file in output.rglob("*") if file.is_file()}, before)
                    self.assertTrue((path / "filtered/layer-3/prefill/kernels.csv").is_file())
                self.assertIn("Median cycles/step", (output / "report.md").read_text())
                self.assertEqual(len((output / "summary.csv").read_text().splitlines()), 5)

    def test_single_prepared_layer_supports_mode_order_and_defaults(self):
        for command in ("run", "cmp-ops"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(path / "prepared")
                seen = []

                def execute(args, metadata, output):
                    seen.append((metadata["source_layer"], args.mode, args.output_len, output.relative_to(path / "output")))

                with socket.socket(socket.AF_UNIX) as server:
                    server.bind(str(path / "server.sock"))
                    with patch.object(runner, "run_layer", side_effect=execute), patch.object(runner, "compare_operators", side_effect=execute), patch.object(runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as process, contextlib.redirect_stdout(io.StringIO()):
                        status = runner.main([command, "--prepared", str(path / "prepared/layer-3"), "--mode", "decode", "prefill",
                                              "--client", "/bin/true", "--socket", str(path / "server.sock"), "--output", str(path / "output")] +
                                             (["--dyperf", "/bin/true"] if command == "run" else []))
                self.assertEqual(status, 0)
                self.assertEqual(seen, [(3, "decode", 1, Path("layer-3/decode")), (3, "prefill", 0, Path("layer-3/prefill"))])
                self.assertEqual(process.call_count, int(command == "run"))

    def test_invalid_modes_and_lengths_fail_before_launch(self):
        options = [("--mode", "prefill", "prefill"), ("--mode", "decode", "prefill", "--output-len", "0"),
                   ("--mode", "prefill", "--output-len", "2"), ("--mode", "prefill", "decode", "--output-len", "-1")]
        for args in options:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory, contextlib.redirect_stderr(io.StringIO()):
                path = Path(directory)
                with patch.object(runner.subprocess, "run") as execute:
                    self.assertEqual(runner.main(["run", "--prepared", str(path / "missing"), "--output", str(path / "output"), *args]), 1)
                execute.assert_not_called()
                self.assertFalse((path / "output").exists())

    def test_report_mode_selection_preserves_legacy_runs(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            path = Path(directory)
            self.prepare(path)
            for layer in (0, 3):
                run, trace = fixture()
                runner.write_json(path / f"layer-{layer}/run.json", run)
                runner.write_json(path / f"layer-{layer}/perf.json", trace)
            runner.write_json(path / "batch.json", dict(schema=1, status="ok", layers=[
                dict(source_layer=layer, directory=f"layer-{layer}", status="ok") for layer in (0, 3)]))
            self.assertEqual(report.main(["--run", directory, "--mode", "prefill"]), 0)
            self.assertEqual(report.main(["--run", directory, "--mode", "decode"]), 1)
            self.assertEqual(report.main(["--run", directory, "--mode", "prefill", "prefill"]), 1)
            self.assertEqual(report.main(["--run", str(path / "layer-0"), "--mode", "prefill"]), 0)
            self.assertEqual(report.main(["--run", str(path / "layer-0"), "--mode", "prefill", "decode"]), 1)

    def test_inference_failure_stops_batch_without_cpu_comparison(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                self.prepare(path / "prepared")

                def execute(command, **kwargs):
                    if command[0] == runner.sys.executable:
                        return SimpleNamespace(returncode=report.main(command[2:]))
                    runner.write_json(Path(command[-1]), dict(status="failed", error="model inference failed at step 0"))
                    return SimpleNamespace(returncode=exit_code)

                with socket.socket(socket.AF_UNIX) as server:
                    server.bind(str(path / "server.sock"))
                    with patch.object(runner.subprocess, "run", side_effect=execute) as process, patch.object(runner, "compare_cpu") as compare, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        status = runner.main(["run", "--prepared", str(path / "prepared"), "--cmp",
                                              "--socket", str(path / "server.sock"), "--client", "/bin/true", "--dyperf", "/bin/true",
                                              "--output", str(path / "output")])
                    self.assertEqual(status, 1)
                    self.assertEqual(process.call_count, 2)
                    compare.assert_not_called()
                batch = json.loads((path / "output/batch.json").read_text())
                self.assertEqual(batch["status"], "failed")
                self.assertEqual([entry["status"] for entry in batch["layers"]], ["failed", "not-run"])
                self.assertFalse((path / "output/layer-3").exists())
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(report.main(["--run", str(path / "output")]), 1)
                summary = json.loads((path / "output/report.json").read_text())
                self.assertEqual(summary["layers"][1], dict(source_layer=3, mode="prefill", status="not-run", report=None))


if __name__ == "__main__":
    unittest.main()

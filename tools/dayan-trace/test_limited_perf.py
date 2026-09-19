import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("limited_perf", Path(__file__).with_name("limited-perf.py"))
perf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(perf)


def events(tag, intervals):
    result = []
    for hart, (start, end) in enumerate(intervals):
        key = dict(pid=65536 + hart, tid=1)
        result.append(dict(name="thread_name", ph="M", args=dict(name=tag), **key))
        for state, counter in (("RUNNING", start), ("WAIT_EXIT", end)):
            result.append(dict(name=state, cat="proc_state", ph="B", args=dict(counter=counter), **key))
        result.append(dict(name="WAIT_EXIT", cat="proc_state", ph="E", args=dict(counter=end + 10000), **key))
    return result


class PerfReportTests(unittest.TestCase):
    def test_moe_model_plan_keeps_selected_experts_inside_each_case(self):
        model = perf.load_model("qwen36-35B-A3B-mxfp4-moe-bf16")
        cases = {case["id"]: case for case in perf.make_cases(5, 16, model=model)}
        self.assertEqual((cases["prefill-router"]["dtype"], cases["prefill-router"]["n"]), ("f32", 256))
        expert = cases["prefill-expert-ffn"]
        self.assertEqual((expert["m"], expert["experts"], expert["experts_used"]), (5, 256, 8))
        terms = {name: factor for name, factor, _ in perf.prediction_terms("prefill", model)}
        self.assertEqual(terms["prefill-expert-ffn"], 40)
        self.assertEqual(cases["decode-vocab"]["m"], 1)

    def test_moe_mega_plan_records_decode_regression_and_reference(self):
        argv = ["limited-perf.py", "plan", "--model", "qwen36-35B-A3B-moe-mega", "--prefill", "5", "--cmp"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(perf.main(), 0)
        self.assertIn("decode_steps=2", output.getvalue())
        self.assertIn("cmp=dayan-standalone", output.getvalue())

    def test_cmp_plan_and_old_client_rejection(self):
        with patch("sys.argv", ["limited-perf.py", "plan", "--cmp", "--cmp-seed", "42"]), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(perf.main(), 0)
        self.assertIn("cmp=cpu seed=42", output.getvalue())
        argv = ["limited-perf.py", "run", "--cmp", "--perf-trace", "/tmp/unused.json"]
        old_client = perf.subprocess.CompletedProcess([], 0, '{"case_schema":2}', "")
        with patch("sys.argv", argv), patch.object(perf.subprocess, "run", return_value=old_client), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                perf.main()
            self.assertEqual(error.exception.code, 2)

    def test_cmp_report_requires_a_pass_and_preserves_failure_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = perf.make_cases(8, 8)[0]
            case.update(cmp=True, seed=1)
            manifest = dict(cases=[case], simulator="cmodel", run_id="cmp", prefill=8, context=8)
            result = dict(status="ok", samples=[dict(tag="cmp", warmup=False, wall_s=1, tasks=4, status=0)])
            trace = root / "trace.json"
            perf.write_json(trace, dict(traceEvents=events("cmp", [(100, 200)] * 4)))
            for comparison, expected in ((None, 1), (dict(status="fail", nonfinite_elements=1), 1),
                                         (dict(status="pass", max_abs_error=0.001, relative_l2_error=0.0001), 0)):
                if comparison is not None:
                    result["comparison"] = comparison
                perf.write_json(root / (case["id"] + ".result.json"), result)
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(perf.make_report(root, manifest, trace, 0, None), expected)
                report = json.loads((root / "report.json").read_text())
                self.assertEqual(report["cases"][0]["comparison"], comparison or {})
                if expected:
                    self.assertIsNone(report["cases"][0]["wall_s"])

    def test_cmp_cannot_be_added_to_an_existing_smoke_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            perf.write_json(root / "manifest.json", dict(cases=perf.make_cases(8, 8)))
            with patch("sys.argv", ["limited-perf.py", "report", "--cmp", "--output", directory]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    perf.main()
                self.assertEqual(error.exception.code, 2)

    def test_quantized_vocabulary_has_its_own_preset(self):
        model = perf.load_model("qwen35-0.8B-mxfp4-vocab")
        cases = {case["id"]: case for case in perf.make_cases(64, 1024, model=model)}
        self.assertEqual(perf.case_types(cases["decode-vocab"]), ("mxfp4", "f32"))
        self.assertEqual(perf.case_types(cases["decode-attention"]), ("bf16", "f32"))

    def test_model_presets_keep_matrix_and_attention_types_separate(self):
        bf16 = perf.make_cases(128, 512)
        self.assertEqual(len(bf16), 16)
        self.assertNotIn("dtype", bf16[0])
        model = perf.load_model("qwen35-0.8B-mxfp4")
        cases = {case["id"]: case for case in perf.make_cases(128, 512, model=model)}
        self.assertEqual(len(cases), 21)
        self.assertEqual(perf.case_types(cases["prefill-ffn"]), ("mxfp4", "mxfp4"))
        self.assertEqual(perf.case_types(cases["decode-input"]), ("mxfp4", "f32"))
        self.assertEqual(perf.case_types(cases["decode-vocab"]), ("bf16", "f32"))
        self.assertEqual(perf.case_types(cases["decode-attention"]), ("bf16", "f32"))
        self.assertEqual(perf.case_types(cases["decode-norm"]), ("f32", "f32"))
        self.assertEqual(cases["decode-attention"]["kv_len"], 513)
        terms = dict((name, factor) for name, factor, _ in perf.prediction_terms("prefill", model))
        self.assertEqual(terms["decode-vocab"], 248320 / 6144)
        self.assertNotIn("decode-input", terms)

    def test_model_selects_additional_case_ids(self):
        output = io.StringIO()
        argv = ["limited-perf.py", "plan", "--model", "qwen35-0.8B-mxfp4", "--cases", "decode-vocab", "decode-quantize-input"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(output):
            self.assertEqual(perf.main(), 0)
        self.assertIn("dtype=f32 output=mxfp4", output.getvalue())
        for arguments in (["--model", "unknown"], ["--cases", "decode-vocab"]):
            with patch("sys.argv", ["limited-perf.py", "plan", *arguments]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    perf.main()
                self.assertEqual(error.exception.code, 2)

    def test_saved_model_controls_report_without_live_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = perf.load_model("qwen35-0.8B-mxfp4")
            cases = perf.make_cases(8, 8, model=model)
            model["prediction_terms"] = [{"case": "decode-vocab", "factor": 7, "reason": "saved factor"}]
            manifest = dict(model="qwen35-0.8B-mxfp4", model_config=model, cases=cases, simulator="cmodel", run_id="unit", prefill=8, context=8, perf_trace=str(root / "trace.json"))
            perf.write_json(root / "manifest.json", manifest)
            perf.write_json(root / "decode-vocab.result.json", dict(status="ok", samples=[
                dict(tag="vocab", warmup=False, wall_s=2, tasks=4, status=0)]))
            perf.write_json(root / "trace.json", dict(traceEvents=events("vocab", [(100, 200)] * 4)))
            argv = ["limited-perf.py", "report", "--output", directory, "--cases", "decode-vocab"]
            with patch("sys.argv", argv), patch.object(perf, "load_model", side_effect=AssertionError("must use snapshot")), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(perf.main(), 0)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["model"], manifest["model"])
            self.assertEqual(report["predictions"][0]["subtotal_wall_s"], 14)
            with patch("sys.argv", argv + ["--model", perf.DEFAULT_MODEL]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    perf.main()
                self.assertEqual(error.exception.code, 2)

    def test_old_client_cannot_silently_run_mxfp4_as_bf16(self):
        argv = ["limited-perf.py", "run", "--model", "qwen35-0.8B-mxfp4", "--perf-trace", "/tmp/unused.json"]
        with patch("sys.argv", argv), patch.object(perf.subprocess, "run", return_value=perf.subprocess.CompletedProcess([], 1, "", "")), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                perf.main()
            self.assertEqual(error.exception.code, 2)

    def test_resume_rejects_changed_model_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client = root / "client"
            client.write_text("test client")
            model = perf.load_model()
            cases = perf.make_cases(1024, 1024)
            model["prediction_terms"][0]["factor"] = 99
            manifest = dict(model=perf.DEFAULT_MODEL, model_config=model, simulator="testcluster", prefill=1024, context=1024,
                            cases=cases, client=str(client), client_sha256=perf.hashlib.sha256(client.read_bytes()).hexdigest(),
                            backend_sha256=None, perf_trace=str(root / "trace.json"), socket="/tmp/unused.sock")
            perf.write_json(root / "manifest.json", manifest)
            argv = ["limited-perf.py", "run", "--resume", "--output", directory, "--client", str(client), "--perf-trace", str(root / "trace.json"), "--socket", "/tmp/unused.sock"]
            with patch("sys.argv", argv), patch.object(perf.subprocess, "run", side_effect=AssertionError("must not launch")), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    perf.main()
                self.assertEqual(error.exception.code, 2)

    def test_case_from_starts_at_selected_case(self):
        output = io.StringIO()
        argv = ["limited-perf.py", "plan", "--cases", "prefill-conv", "decode-conv", "--case-from", "decode-conv"]
        with patch("sys.argv", argv), contextlib.redirect_stdout(output):
            self.assertEqual(perf.main(), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual([line.split()[0] for line in lines[:-1]], ["decode-conv"])
        self.assertIn("1 cases, 1 kernel launches", lines[-1])

    def test_case_from_requires_selected_case(self):
        with patch("sys.argv", ["limited-perf.py", "plan", "--cases", "decode-conv", "--case-from", "prefill-conv"]), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                perf.main()
            self.assertEqual(error.exception.code, 2)

    def test_cases_selection_combines_options_without_duplicate_launches(self):
        output = io.StringIO()
        with patch("sys.argv", ["limited-perf.py", "plan", "--cases", "decode-conv", "prefill-conv", "--case", "decode-conv"]), contextlib.redirect_stdout(output):
            self.assertEqual(perf.main(), 0)
        lines = output.getvalue().splitlines()
        self.assertEqual([line.split()[0] for line in lines[:-1]], ["prefill-conv", "decode-conv"])
        self.assertIn("2 cases, 2 kernel launches", lines[-1])

    def test_cases_reject_unknown_or_empty_selection(self):
        for selection in ([], ["invalid"]):
            with self.subTest(selection=selection), patch("sys.argv", ["limited-perf.py", "plan", "--cases", *selection]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    perf.main()
                self.assertEqual(error.exception.code, 2)

    def test_report_selection_preserves_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = perf.make_cases(1024, 1024)
            manifest = dict(cases=cases, simulator="cmodel", run_id="unit", prefill=1024, context=1024, perf_trace=str(root / "trace.json"))
            perf.write_json(root / "manifest.json", manifest)
            perf.write_json(root / "decode-conv.result.json", dict(status="ok", samples=[
                dict(tag="conv", warmup=False, wall_s=1, tasks=4, status=0)]))
            perf.write_json(root / "trace.json", dict(traceEvents=events("conv", [(100, 200)] * 4)))
            with patch("sys.argv", ["limited-perf.py", "report", "--output", directory, "--cases", "decode-conv"]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(perf.main(), 0)
            report = json.loads((root / "report.json").read_text())
            self.assertEqual([case["id"] for case in report["cases"]], ["decode-conv"])
            self.assertIsNone(report["predictions"][1]["predicted_wall_s"])
            self.assertEqual(json.loads((root / "manifest.json").read_text()), manifest)

    def read(self, content, wanted=("sample",)):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            trace.write_text(content)
            return perf.read_trace(trace, set(wanted))

    def test_multicore_span_excludes_children_and_deferred_exit_end(self):
        data = events("sample", [(100, 200), (110, 250), (90, 190), (130, 240)])
        data += [dict(name="thread_name", ph="M", pid=65536, tid=2, args=dict(name="post")),
                 dict(name="RUNNING", cat="proc_state", ph="B", pid=65536, tid=2, args=dict(counter=0))]
        roots, note = self.read(json.dumps(dict(traceEvents=data)))
        self.assertIsNone(note)
        self.assertEqual(perf.cycle_span(roots, "sample", 4), (160, None))

    def test_wrap_across_cores(self):
        period = perf.COUNTER_PERIOD
        data = events("sample", [(period - 10, 20), (2, 30), (period - 20, 10), (5, 40)])
        roots, _ = self.read(json.dumps(dict(traceEvents=data)))
        self.assertEqual(perf.cycle_span(roots, "sample", 4), (60, None))

    def test_partial_file_retains_complete_launches(self):
        data = events("sample", [(100, 200)] * 4)
        partial = '{"traceEvents":[' + ",".join(json.dumps(item) for item in data) + ',{"name":'
        roots, note = self.read(partial)
        self.assertIsNotNone(note)
        self.assertEqual(perf.cycle_span(roots, "sample", 4), (100, None))

    def test_missing_root_or_zero_counter_is_not_a_measurement(self):
        roots, _ = self.read(json.dumps(dict(traceEvents=events("sample", [(0, 0)] * 4))))
        self.assertIsNone(perf.cycle_span(roots, "sample", 4)[0])
        self.assertIsNone(perf.cycle_span(roots, "sample", 5)[0])
        self.assertIsNone(perf.cycle_span(roots, "different-tag", 4)[0])

    def test_stream_crosses_chunk_boundary(self):
        data = [dict(name="ignored", args=dict(text="x" * 40000))] * 3
        data += events("sample", [(10, 20)] * 4)
        roots, _ = self.read(json.dumps(dict(traceEvents=data)))
        self.assertEqual(perf.cycle_span(roots, "sample", 4), (10, None))

    def test_missing_exit_is_pending(self):
        data = events("sample", [(100, 200)] * 4)
        data = [item for item in data if not (item.get("pid") == 65539 and item["name"] == "WAIT_EXIT")]
        roots, _ = self.read(json.dumps(dict(traceEvents=data)))
        self.assertIsNone(perf.cycle_span(roots, "sample", 4)[0])

    def test_no_projection_from_failed_or_unmeasured_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = perf.make_cases(1024, 1024)
            manifest = dict(cases=cases, simulator="cmodel", run_id="unit", prefill=1024, context=1024)
            trace_events = []
            for index, case in enumerate(cases):
                tag = f"case{index}"
                perf.write_json(root / (case["id"] + ".result.json"), dict(status="ok", samples=[
                    dict(tag=tag, warmup=False, wall_s=1, tasks=4, status=0),
                    dict(tag=tag + "warm", warmup=True, wall_s=100, tasks=4, status=0)]))
                trace_events += events(tag, [(100, 200)] * 4)
                # The profiler uses a new TID for each slot generation.
                for item in trace_events[-16:]:
                    item["tid"] = index + 1
            trace = root / "trace.json"
            perf.write_json(trace, dict(traceEvents=trace_events))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(perf.make_report(root, manifest, trace, 20, None), 0)
            report = json.loads((root / "report.json").read_text())
            expected = sum(factor for _, factor, _ in perf.prediction_terms("prefill"))
            self.assertAlmostEqual(report["predictions"][0]["predicted_wall_s"], expected * 1.2)
            self.assertAlmostEqual(report["predictions"][0]["predicted_cycles"], expected * 120)
            perf.write_json(root / "prefill-ffn.result.json", dict(status="failed", error="bad result"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(perf.make_report(root, manifest, trace, 20, None), 1)
            report = json.loads((root / "report.json").read_text())
            self.assertIsNone(report["predictions"][0]["predicted_wall_s"])
            self.assertIsNotNone(report["predictions"][1]["predicted_wall_s"])


if __name__ == "__main__":
    unittest.main()

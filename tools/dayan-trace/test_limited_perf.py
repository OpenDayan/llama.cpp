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

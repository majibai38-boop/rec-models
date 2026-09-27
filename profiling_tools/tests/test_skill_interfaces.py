import contextlib
import copy
import csv
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from profiling_tools.__main__ import main
from profiling_tools.build_bundle import build_bundle
from profiling_tools.compare import CompareConfig, InputSelectionError, compare_profiles
from profiling_tools.inspection import inspect_inputs
from profiling_tools.result_api import query_results, summarize_results, validate_results, file_identity
from profiling_tools.provenance import write_json
from profiling_tools.tests.test_compare import complete
from profiling_tools.version import __version__


def fixture(root, *, extra_gpu=False, multiple=False):
    root = Path(root)
    for side, pid in (("gpu", 1), ("npu", 3)):
        events = [complete("ProfilerStep#1", "user_annotation", 0, 100, pid, pid),
                  complete("aten::add", "cpu_op", 10, 5, pid, pid,
                           {"External id": 1, "Input Dims": [[4], [4]], "Input type": ["float", "float"]}),
                  complete("add_kernel", "kernel", 20, 2 if side == "gpu" else 4, pid + 1, 7,
                           {"External id": 1, "stream": 7})]
        if extra_gpu and side == "gpu":
            events += [complete("aten::unknown", "cpu_op", 40, 5, pid, pid, {"External id": 2}),
                       complete("unknown_kernel", "kernel", 60, 3, pid + 1, 7, {"External id": 2})]
        if multiple:
            events += [complete("ProfilerStep#2", "user_annotation", 100, 100, pid, pid),
                       complete("aten::add", "cpu_op", 110, 5, pid, pid),
                       complete("aten::other_rank", "cpu_op", 10, 5, pid + 10, pid + 10)]
        (root / f"{side}.json").write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
    (root / "details.csv").write_text("Name,Type,Duration(us),Start Time(us),Stream ID,Device_id,Step ID\nadd_kernel,Add,4,20,7,0,1\n", encoding="utf-8")
    return CompareConfig(str(root / "gpu.json"), str(root / "npu.json"), str(root / "details.csv"), str(root / "out"),
                         gpu_step="1", npu_step="1", require_explicit_scope=True)


class SkillInterfacesTest(unittest.TestCase):
    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            code = main(args)
        return code, output.getvalue()

    def args(self, config):
        return ["--gpu-trace", config.gpu_trace, "--npu-trace", config.npu_trace,
                "--npu-kernel-details", config.npu_kernel_details]

    def test_preflight_inventory_and_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            before = {p.name: file_identity(p)["sha256"] for p in Path(tmp).iterdir()}
            report = inspect_inputs(config)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["profiles"]["gpu"]["window_kernel_candidate_count"], 1)
            self.assertEqual(report["profiles"]["npu"]["dtype_present_count"], 1)
            self.assertEqual(before, {p.name: file_identity(p)["sha256"] for p in Path(tmp).iterdir()})

    def test_ambiguous_pid_and_step_block_without_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(fixture(tmp, multiple=True), gpu_step="", npu_step="")
            result = inspect_inputs(config)
            self.assertEqual(result["status"], "needs_selection")
            self.assertEqual({r["code"] for r in result["selections_required"]}, {"ambiguous_pid", "step_scope_unspecified"})
            with self.assertRaises(InputSelectionError):
                compare_profiles(config)
            self.assertFalse(Path(config.output_dir).exists())
            selected = replace(config, gpu_pid="1", npu_pid="3", gpu_step="1", npu_step="1")
            self.assertEqual(inspect_inputs(selected)["status"], "ready")

    def test_all_steps_is_explicit_and_cannot_be_mixed(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            self.assertEqual(inspect_inputs(replace(config, all_steps=True))["status"], "invalid")
            self.assertEqual(inspect_inputs(replace(config, all_steps=True, gpu_step="", npu_step=""))["status"], "ready")

    def test_bad_inputs_and_empty_detail_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            for broken in (replace(config, gpu_pid="99"), replace(config, npu_step="42"),
                           replace(config, npu_kernel_details=str(Path(tmp) / "absent.csv"))):
                self.assertEqual(inspect_inputs(broken)["status"], "invalid")
            Path(config.npu_kernel_details).write_text("Name,Type\n", encoding="utf-8")
            self.assertEqual(inspect_inputs(config)["status"], "invalid")
            Path(config.gpu_trace).write_text('{"traceEvents": {}}', encoding="utf-8")
            self.assertIn("invalid_trace", {e["code"] for e in inspect_inputs(config)["errors"]})

    def test_duplicate_step_does_not_select_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            data = json.loads(Path(config.gpu_trace).read_text())
            data["traceEvents"].append(copy.deepcopy(data["traceEvents"][0]))
            Path(config.gpu_trace).write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(inspect_inputs(config)["status"], "invalid")

    def test_invalid_raw_trace_timing_does_not_become_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            original = json.loads(Path(config.gpu_trace).read_text())
            for field, value in (("ts", "NaN"), ("dur", "Infinity"), ("dur", "bad"), ("dur", None), ("dur", -1)):
                data = copy.deepcopy(original)
                data["traceEvents"][2][field] = value
                Path(config.gpu_trace).write_text(json.dumps(data), encoding="utf-8")
                self.assertEqual(inspect_inputs(config)["status"], "invalid", (field, value))
            self.assertFalse(Path(config.output_dir).exists())

    def test_invalid_detail_timing_and_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            for value in ("NaN", "Infinity", "bad", "-1"):
                Path(config.npu_kernel_details).write_text(f"Name,Type,Duration(us)\nadd_kernel,Add,{value}\n", encoding="utf-8")
                self.assertEqual(inspect_inputs(config)["status"], "invalid", value)
            Path(config.npu_kernel_details).write_text("Name,Type,Duration(us)\nadd_kernel,Add,\n", encoding="utf-8")
            self.assertEqual(inspect_inputs(config)["status"], "ready")
            for broken in (replace(config, flow_tolerance_us=float("nan")), replace(config, diagnostic_coverage_threshold=101),
                           replace(config, op_map=str(Path(tmp) / "absent.json"))):
                self.assertEqual(inspect_inputs(broken)["status"], "invalid")

    def test_missing_metadata_warns_not_invents(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            data = json.loads(Path(config.gpu_trace).read_text())
            data["traceEvents"][1]["args"] = {}
            Path(config.gpu_trace).write_text(json.dumps(data), encoding="utf-8")
            report = inspect_inputs(config)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["profiles"]["gpu"]["shape_present_count"], 0)
            self.assertIn("partial_cpu_metadata", {w["code"] for w in report["warnings"]})

    def test_cli_exit_codes_and_structured_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            code, text = self.invoke(["inspect", *self.args(config)])
            self.assertEqual(code, 3)
            self.assertEqual(json.loads(text)["status"], "needs_selection")
            code, text = self.invoke(["compare", *self.args(config), "--gpu-step", "1", "--npu-step", "1",
                                      "--output-dir", config.output_dir, "--json"])
            self.assertEqual(code, 0, text)
            self.assertEqual(json.loads(text)["status"], "complete")
            self.assertEqual(self.invoke(["validate", "--result-dir", config.output_dir])[0], 0)
            self.assertEqual(self.invoke(["query", "--result-dir", config.output_dir])[0], 2)

    def test_manifest_and_fresh_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            summary = compare_profiles(config)
            manifest = json.loads((Path(config.output_dir) / "run_manifest.json").read_text())
            self.assertEqual(manifest["run_id"], summary["run_id"])
            self.assertEqual(manifest["producer"]["version"], __version__)
            self.assertEqual(manifest["inputs"]["gpu_trace"]["sha256"], file_identity(config.gpu_trace)["sha256"])
            self.assertEqual(validate_results(config.output_dir)["status"], "valid")
            with self.assertRaisesRegex(ValueError, "new or empty"):
                compare_profiles(config)

    @unittest.skipUnless(os.name == "nt", "Windows-specific file locking")
    def test_atomic_json_retries_transient_windows_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.json"
            write_json(output, {"old": True})
            original = Path.replace
            attempts = []

            def replace_file(source, target):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError("simulated sharing violation")
                return original(source, target)

            with patch.object(Path, "replace", replace_file), patch("profiling_tools.provenance.time.sleep"):
                write_json(output, {"new": True})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(output.read_text()), {"new": True})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    @unittest.skipUnless(os.name == "nt", "Windows-specific file locking")
    def test_atomic_json_persistent_lock_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "summary.json"
            write_json(output, {"old": True})
            with patch.object(Path, "replace", side_effect=PermissionError("locked")) as replace_file, patch("profiling_tools.provenance.time.sleep"):
                with self.assertRaises(PermissionError):
                    write_json(output, {"new": True})
            self.assertEqual(replace_file.call_count, 6)
            self.assertEqual(json.loads(output.read_text()), {"old": True})
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_reuse_and_queries_never_parse_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, extra_gpu=True)
            summary = compare_profiles(config)
            self.assertEqual(summary["status"], "complete_with_gaps")
            with patch("profiling_tools.parser._load_trace_events", side_effect=AssertionError("must not parse")):
                result = summarize_results(config.output_dir)
                self.assertEqual(result["gpu_accounting"]["kernel"]["unmapped_count"], 1)
                self.assertEqual(query_results(config.output_dir, op_type="Add")["mappings"]["total"], 1)
                self.assertEqual(query_results(config.output_dir, npu_kernel_index=0)["mappings"]["total"], 1)
                self.assertEqual(query_results(config.output_dir, gpu_kernel_index=1)["gpu_unmapped"]["total"], 1)
                self.assertEqual(query_results(config.output_dir, cpu_op_index=0, side="gpu")["mappings"]["total"], 1)
                self.assertEqual(query_results(config.output_dir, cpu_op_index=1, side="gpu")["gpu_unmapped"]["total"], 1)
                self.assertEqual(validate_results(config.output_dir)["status"], "valid")

    def test_query_bounds_and_no_false_positive_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            compare_profiles(config)
            q = query_results(config.output_dir, op_type="Add", limit=1, offset=1)
            self.assertEqual(q["mappings"]["total"], 1)
            self.assertEqual(q["mappings"]["items"], [])
            self.assertEqual(query_results(config.output_dir, op_type="Unknown")["mappings"]["total"], 0)
            for kwargs in ({}, {"op_type": "Add", "limit": 0}, {"npu_kernel_index": -1}):
                with self.assertRaises(ValueError):
                    query_results(config.output_dir, **kwargs)

    def test_query_and_filters_do_not_leak_gpu_remainder(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, extra_gpu=True)
            compare_profiles(config)
            for kwargs in ({"op_type": "Add"}, {"npu_kernel_index": 0}, {"device_id": "0"},
                           {"core_type": "AI_CORE"}, {"cpu_op_index": 1, "side": "npu"}):
                result = query_results(config.output_dir, gpu_kernel_index=1, **kwargs)
                self.assertEqual(result["gpu_unmapped"]["total"], 0, kwargs)

    def test_query_and_summary_expose_typed_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, extra_gpu=True)
            compare_profiles(config)
            total = summarize_results(config.output_dir)["overall_statistics"][0]
            self.assertEqual(total["comparable_ratio_gpu_npu"], 0.5)
            self.assertEqual(total["total_time_ratio_gpu_npu"], 1.25)
            query = query_results(config.output_dir, op_type="Add")
            row = query["type_statistics"]["items"][0]
            self.assertIsInstance(row["npu_kernel_count"], int)
            self.assertEqual(row["coverage"]["comparable"]["time_pct"], 100)
            self.assertEqual(query["reason_groups"]["total"], 0)
            unknown = query_results(config.output_dir, op_type="Unknown")
            self.assertEqual(unknown["type_statistics"]["total"], 0)

    def test_json_cli_is_utf8_even_with_legacy_pipe_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            for file in (config.gpu_trace, config.npu_trace, config.npu_kernel_details):
                path = Path(file)
                path.write_text(path.read_text(encoding="utf-8").replace("add_kernel", "add_kernel_中文"), encoding="utf-8")
            compare_profiles(config)
            run = subprocess.run([sys.executable, "-m", "profiling_tools", "query", "--result-dir", config.output_dir,
                                  "--op-type", "Add"], cwd=Path(__file__).resolve().parents[2],
                                 env={**os.environ, "PYTHONIOENCODING": "gbk"}, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            payload = run.stdout.decode("utf-8")
            self.assertEqual(json.loads(payload)["mappings"]["total"], 1)
            self.assertIn("add_kernel_中文", payload)

    def test_validation_rejects_tampering_and_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            compare_profiles(config)
            file = Path(config.output_dir) / "npu_gpu_op_statistic.csv"
            text = file.read_text(encoding="utf-8-sig")
            file.write_text(text.replace("0.5", "0.7"), encoding="utf-8-sig")
            self.assertEqual(validate_results(config.output_dir)["status"], "invalid")
            self.assertEqual(self.invoke(["validate", "--result-dir", config.output_dir])[0], 4)
            file.unlink()
            self.assertEqual(validate_results(config.output_dir)["status"], "invalid")

    def test_scope_warnings_are_not_arithmetic_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            data = json.loads(Path(config.npu_trace).read_text())
            data["traceEvents"].append(complete("trace_only_copy", "kernel", 50, 2, 4, 8))
            Path(config.npu_trace).write_text(json.dumps(data), encoding="utf-8")
            summary = compare_profiles(config)
            self.assertEqual(summary["validation"]["status"], "valid")
            self.assertIn("npu_trace_only_excluded", {w["code"] for w in summary["warnings"]})

    def test_standalone_bundle_from_unrelated_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixture(root)
            compare_profiles(config)
            bundle = build_bundle(root / "bundle")
            script = Path(bundle["skill"]) / "scripts" / "run_compare.py"
            other = root / "unrelated cwd"
            other.mkdir()
            run = subprocess.run([sys.executable, str(script), "validate", "--result-dir", config.output_dir],
                                 cwd=other, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["status"], "valid")
            run = subprocess.run([sys.executable, bundle["application"], "--version"], cwd=other, capture_output=True, text=True)
            self.assertEqual(run.stdout.strip(), __version__)
            portable_output = str(root / "portable result")
            run = subprocess.run([sys.executable, str(script), "compare", *self.args(config),
                                  "--gpu-step", "1", "--npu-step", "1", "--output-dir", portable_output, "--json"],
                                 cwd=other, capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads(run.stdout)["status"], "complete")
            self.assertEqual(validate_results(portable_output)["status"], "valid")
            source_manifest = json.loads((Path(config.output_dir) / "run_manifest.json").read_text())
            bundle_manifest = json.loads((Path(portable_output) / "run_manifest.json").read_text())
            self.assertEqual(source_manifest["producer"]["source_sha256"], bundle_manifest["producer"]["source_sha256"])
            with zipfile.ZipFile(bundle["application"]) as archive:
                self.assertTrue(all(name.endswith(".py") for name in archive.namelist()))
                self.assertFalse(any("example/" in name or "tests/" in name for name in archive.namelist()))
            with self.assertRaises(ValueError):
                build_bundle(root / "bundle")


if __name__ == "__main__":
    unittest.main()

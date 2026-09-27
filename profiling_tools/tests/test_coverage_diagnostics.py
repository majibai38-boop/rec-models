import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from profiling_tools.compare import CompareConfig, compare_profiles
from profiling_tools.coverage_diagnostics import build_coverage_diagnostics, write_coverage_diagnostics
from profiling_tools.model import OpMatch
from profiling_tools.tests.test_logical_calls import op, profile


def case(types=("UnseenKernel",)):
    gpu = profile("gpu", [op("aten::new_operator", 0)], [(0, "new_cuda_kernel")])
    npu = profile("npu", [op("aten::new_operator", 0)], [(0, name) for name in types])
    for kernel in gpu.kernels + npu.kernels:
        kernel.link_method = "external_id"
        kernel.stream_id = "7"
    for kernel in npu.kernels:
        d = kernel.detail
        d.input_shapes = d.output_shapes = '"4"'
        d.input_dtypes = d.output_dtypes = "FLOAT"
        d.device_id, d.stream_id = "0", "7"
        d.name = "aclnnArbitrary_" + d.op_type
    rows = [{"Device_id": "0", "OP Type": name, "Core Type": "AI_CORE", "NPU Kernel Index": i,
             "NPU Duration(us)": 2.0, "Comparable": False, "Accounted": False,
             "Mapping Mode": "new_unmapped_mode", "GPU Kernel Indices": "[]"}
            for i, name in enumerate(types)]
    return gpu, npu, [OpMatch(1, 0, 0, 1, "high")], rows


class CoverageDiagnosticsTest(unittest.TestCase):
    def diagnose(self, fixture=None, **kwargs):
        return build_coverage_diagnostics(*(fixture or case()), **kwargs)

    def reason(self, fixture, **kwargs):
        return self.diagnose(fixture, **kwargs)["kernels"][0]["reason_id"]

    def test_unseen_kernel_and_unknown_reason_are_retained(self):
        report = self.diagnose()
        self.assertEqual(report["types"][0]["op_type"], "UnseenKernel")
        self.assertEqual(report["kernels"][0]["reason_id"], "unclassified")
        self.assertEqual(report["summary"]["analyzed_kernel_count"], 1)
        self.assertTrue(report["kernels"][0]["next_steps"])

    def test_threshold_boundaries_count_or_time(self):
        fixture = case(("UnseenKernel",) * 10)
        for row in fixture[3][:9]:
            row["Comparable"] = True
        # Exactly 90% on both dimensions is not below 90.
        self.assertEqual(self.diagnose(fixture)["summary"]["selected_type_count"], 0)
        self.assertEqual(self.diagnose(fixture, threshold=100)["summary"]["analyzed_kernel_count"], 1)
        # Count coverage 90, time coverage lower than 90.
        fixture[3][-1]["NPU Duration(us)"] = 20
        result = self.diagnose(fixture)["types"][0]
        self.assertEqual(result["comparable_count_coverage_pct"], 90)
        self.assertLess(result["comparable_time_coverage_pct"], 90)
        # High time coverage does not mask low count coverage.
        fixture[3][-2]["Comparable"] = False
        fixture[3][-2]["NPU Duration(us)"] = fixture[3][-1]["NPU Duration(us)"] = .01
        result = self.diagnose(fixture)["types"][0]
        self.assertEqual(result["comparable_count_coverage_pct"], 80)
        self.assertGreater(result["comparable_time_coverage_pct"], 99)

    def test_all_mapped_empty_zero_duration_and_csv_booleans(self):
        fixture = case()
        fixture[3][0]["Comparable"] = "True"
        self.assertEqual(self.diagnose(fixture, threshold=100)["summary"]["selected_type_count"], 0)
        fixture[3][0]["Comparable"] = "False"
        fixture[3][0]["NPU Duration(us)"] = 0
        item = self.diagnose(fixture)["types"][0]
        self.assertIsNone(item["comparable_time_coverage_pct"])
        self.assertEqual(item["uncomparable_kernel_count"], 1)
        self.assertEqual(self.diagnose(fixture, threshold=0)["summary"]["selected_type_count"], 0)
        self.assertEqual(self.diagnose(case(()))["summary"]["scanned_type_count"], 0)

    def test_invalid_threshold_fails_before_parsing(self):
        for threshold in (-1, 101, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                self.diagnose(threshold=threshold)
            config = CompareConfig("missing", "missing", "missing", "unused",
                                   diagnostic_coverage_threshold=threshold)
            with patch("profiling_tools.compare.parse_trace") as parse:
                with self.assertRaises(ValueError):
                    compare_profiles(config)
                parse.assert_not_called()

    def test_duplicate_indices_and_bad_duration_rejected(self):
        fixture = case()
        fixture[3].append(copy.deepcopy(fixture[3][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.diagnose(fixture)
        for duration in (-1, float("nan"), float("inf")):
            fixture = case()
            fixture[3][0]["NPU Duration(us)"] = duration
            with self.assertRaisesRegex(ValueError, "duration"):
                self.diagnose(fixture)

    def test_device_and_core_are_separate_groups(self):
        fixture = case(("UnseenKernel",) * 3)
        fixture[3][1]["Device_id"] = "1"
        fixture[3][2]["Core Type"] = "AI_VECTOR_CORE"
        report = self.diagnose(fixture)
        self.assertEqual(report["summary"]["selected_type_count"], 3)
        self.assertEqual(report["summary"]["analyzed_kernel_count"], 3)

    def test_owner_missing_weak_and_api_conflict(self):
        fixture = case()
        fixture[1].kernels[0].owner_op_index = None
        self.assertEqual(self.reason(fixture), "npu_owner_missing")
        fixture[1].kernels[0].owner_op_index = 0
        fixture[1].kernels[0].link_method = "flow"
        self.assertEqual(self.reason(fixture), "npu_owner_weak")
        fixture[3][0]["Mapping Mode"] = "api_semantic_conflict_unmapped"
        fixture[3][0]["API Expected CPU OPs"] = '["aten::different"]'
        self.assertEqual(self.reason(fixture), "api_cpu_conflict")

    def test_untrusted_and_ambiguous_cpu_matches(self):
        fixture = case()
        fixture[2][0].confidence = "low"
        self.assertEqual(self.reason(fixture), "cpu_match_untrusted")
        fixture[2].append(OpMatch(2, 0, 0, 1, "high"))
        self.assertEqual(self.reason(fixture), "cpu_match_ambiguous")

    def test_cpu_metadata_conflict_is_not_overridden_by_helper(self):
        fixture = case()
        fixture[3][0]["Mapping Mode"] = "auxiliary_unmapped"
        fixture[0].ops[0].input_dtypes = '["int64","scalar"]'
        self.assertEqual(self.reason(fixture), "cpu_metadata_conflict")
        e = self.diagnose(fixture)["kernels"][0]["evidence"]
        self.assertEqual(e["cpu_metadata_differences"], ["input_dtypes"])

    def test_cpu_aliases_and_explicit_op_map(self):
        fixture = case()
        fixture[0].ops[0].input_dtypes = '["c10::Half","scalar"]'
        fixture[1].ops[0].input_dtypes = '["float16","scalar"]'
        self.assertNotEqual(self.reason(fixture), "cpu_metadata_conflict")
        fixture[0].ops[0].canonical_name = "aten::gpu_alternative"
        self.assertEqual(self.reason(fixture), "cpu_metadata_conflict")
        self.assertNotEqual(self.reason(fixture, op_map={"aten::gpu_alternative": "aten::new_operator"}),
                            "cpu_metadata_conflict")

    def test_same_module_metadata_candidates_not_auto_matched(self):
        fixture = case()
        fixture[2].clear()
        report = self.diagnose(fixture)
        self.assertEqual(report["kernels"][0]["reason_id"], "cpu_alignment_gap")
        self.assertEqual(report["kernels"][0]["evidence"]["same_module_metadata_candidate_count"], 1)
        self.assertEqual(report["kernels"][0]["evidence"]["gpu_scope_kernel_indices"], [])
        self.assertFalse(fixture[3][0]["Comparable"])
        fixture[1].ops[0].module_key = "different_module"
        self.assertEqual(self.reason(fixture), "cpu_match_missing")
        fixture[1].ops[0].module_key = ""
        self.assertEqual(self.reason(fixture), "cpu_match_missing")

    def test_matched_ancestor_remains_a_hint(self):
        fixture = case()
        fixture[1].ops.append(op("aten::child", 1, parent=0))
        fixture[1].kernels[0].owner_op_index = 1
        report = self.diagnose(fixture)
        e = report["kernels"][0]["evidence"]
        self.assertEqual(report["kernels"][0]["reason_id"], "cpu_match_missing")
        self.assertEqual(e["nearest_matched_npu_ancestor"], 0)
        self.assertEqual(e["nearest_matched_gpu_ancestor"], 0)
        self.assertEqual(e["gpu_scope_kernel_indices"], [])

    def test_candidate_samples_are_bounded_and_not_silent(self):
        fixture = case()
        fixture[2].clear()
        for i in range(1, 9):
            fixture[0].ops.append(op("aten::new_operator", i))
        e = self.diagnose(fixture)["kernels"][0]["evidence"]
        self.assertEqual(e["same_module_metadata_candidate_count"], 9)
        self.assertEqual(len(e["gpu_cpu_candidate_sample"]), 5)
        self.assertTrue(e["candidate_sample_truncated"])

    def test_memory_only_and_absent_gpu_work(self):
        fixture = case()
        fixture[0].memory_activities = fixture[0].kernels
        fixture[0].kernels = []
        fixture[0].ops[0].kernel_indices = []
        self.assertEqual(self.reason(fixture), "gpu_memory_only")
        fixture[0].memory_activities = []
        self.assertEqual(self.reason(fixture), "gpu_work_absent")

    def test_generic_auxiliary_metadata_not_kernel_name(self):
        fixture = case()
        fixture[3][0]["Mapping Mode"] = "api_helper_unmapped"
        self.assertEqual(self.reason(fixture), "auxiliary_work")
        detail = fixture[1].kernels[0].detail
        detail.input_dtypes = "FLOAT16"
        self.assertEqual(self.reason(fixture), "auxiliary_dtype")
        detail.input_dtypes = "FLOAT"
        detail.output_shapes = '"2,2"'
        self.assertEqual(self.reason(fixture), "auxiliary_layout")
        detail.output_shapes = detail.input_shapes
        detail.raw = {"Input Formats": "ND", "Output Formats": "NZ"}
        self.assertEqual(self.reason(fixture), "auxiliary_layout")

    def test_missing_metadata_is_preserved(self):
        fixture = case()
        fixture[1].kernels[0].detail.input_dtypes = "N/A"
        fixture[1].kernels[0].detail.output_dtypes = "N/A"
        e = self.diagnose(fixture)["kernels"][0]["evidence"]
        self.assertIn("incomplete_kernel_metadata", e["metadata_effects"])
        self.assertNotIn("dtypes_change", e["metadata_effects"])

    def test_index_and_axis_inputs_are_not_casts_or_format_changes(self):
        fixture = case()  # Type name is deliberately not Slice or Transpose.
        fixture[3][0]["Mapping Mode"] = "auxiliary_unmapped"
        d = fixture[1].kernels[0].detail
        d.input_dtypes, d.output_dtypes = "FLOAT16;INT64;INT64", "FLOAT16"
        d.input_shapes, d.output_shapes = '"61,41006";"2";"2"', '"61,8"'
        d.raw = {"Input Formats": "ND;ND;ND", "Output Formats": "ND"}
        report = self.diagnose(fixture)
        e = report["kernels"][0]["evidence"]
        self.assertEqual(report["kernels"][0]["reason_id"], "auxiliary_layout")
        self.assertNotIn("dtypes_change", e["metadata_effects"])
        self.assertNotIn("formats_change", e["metadata_effects"])
        self.assertIn("mixed_input_dtypes", e["metadata_effects"])
        self.assertEqual(e["kernel_metadata"]["shape_format_comparison_basis"], "first_input_output_candidate")

    def test_multi_input_mixed_precision_is_not_proven_conversion(self):
        fixture = case()
        fixture[3][0]["Mapping Mode"] = "auxiliary_unmapped"
        d = fixture[1].kernels[0].detail
        d.input_dtypes, d.output_dtypes = "FLOAT16;FLOAT", "FLOAT"
        d.input_shapes, d.output_shapes = '"4";"4"', '"4"'
        self.assertNotEqual(self.reason(fixture), "auxiliary_dtype")

    def test_gpu_child_work_and_already_used_scope(self):
        fixture = case(("UnseenKernel", "MainKernel"))
        fixture[0].ops.append(op("aten::child", 1, parent=0))
        fixture[0].ops[0].kernel_indices = []
        fixture[0].ops[1].kernel_indices = [0]
        fixture[0].kernels[0].owner_op_index = 1
        self.assertEqual(self.reason(fixture), "gpu_child_work")
        fixture[3][1]["Comparable"] = True
        fixture[3][1]["GPU Kernel Indices"] = "[0]"
        self.assertEqual(self.reason(fixture), "gpu_scope_overlap")
        e = self.diagnose(fixture)["kernels"][0]["evidence"]
        self.assertEqual(e["gpu_existing_npu_users"], {"0": [1]})

    def test_ambiguous_bundle_is_not_unknown_or_one_to_one(self):
        fixture = case(("NewA", "NewB"))
        self.assertEqual(self.reason(fixture), "bundle_ambiguity")
        self.assertEqual(self.diagnose(fixture)["summary"]["analyzed_kernel_count"], 2)

    def test_optional_whole_call_evidence_does_not_change_coverage(self):
        fixture = case()
        precision = [{"NPU Kernel Indices": "[0]", "Comparable": True, "Precision Path ID": 7,
                      "NPU Path": "arbitrary path", "Pattern": "arbitrary_pattern", "Cause Status": "not_verified"}]
        report = self.diagnose(fixture, precision_rows=precision)
        self.assertEqual(report["kernels"][0]["reason_id"], "whole_call_available")
        self.assertEqual(report["types"][0]["comparable_count_coverage_pct"], 0)
        self.assertFalse(fixture[3][0]["Comparable"])
        precision[0]["Comparable"] = False
        self.assertNotEqual(self.reason(fixture, precision_rows=precision), "whole_call_available")
        bundle = [{"NPU Kernel Indices": "[0]", "CPU Bundle ID": "new:7", "CPU Bundle": "ArbitraryPipeline"}]
        self.assertEqual(self.reason(fixture, cpu_bundle_rows=bundle), "whole_call_available")
        fused = [{"NPU Kernel Indices": "[0]", "Occurrence": 3, "Fused OP": "UnknownFusedOp"}]
        fixture[1].kernels[0].owner_op_index = None
        self.assertEqual(self.reason(fixture, fused_rows=fused), "whole_call_available")

    def test_accounting_and_non_mutation(self):
        fixture = case(("UnknownA", "UnknownB", "UnknownA"))
        fixture[3][0]["Mapping Mode"] = "auxiliary_unmapped"
        fixture[3][2]["Mapping Mode"] = "api_semantic_conflict_unmapped"
        before = copy.deepcopy(fixture)
        report = self.diagnose(fixture)
        self.assertEqual(fixture, before)
        self.assertEqual(sum(report["summary"]["reason_kernel_counts"].values()), 3)
        self.assertEqual(len({r["npu_kernel_index"] for r in report["kernels"]}), 3)
        for t in report["types"]:
            self.assertEqual(sum(d["kernel_count"] for d in t["diagnoses"]), t["uncomparable_kernel_count"])
            self.assertAlmostEqual(sum(d["npu_time_us"] for d in t["diagnoses"]), t["uncomparable_time_us"])

    def test_markdown_and_json_empty_and_populated(self):
        for fixture in (case(), case(())):
            report = self.diagnose(fixture)
            with tempfile.TemporaryDirectory() as tmp:
                write_coverage_diagnostics(tmp, report)
                saved = json.loads((Path(tmp) / "coverage_diagnostics.json").read_text(encoding="utf-8"))
                self.assertEqual(saved, report)
                text = (Path(tmp) / "coverage_diagnostics.md").read_text(encoding="utf-8")
                self.assertIn("< 90%", text)
                self.assertIn("不是可节省时间", text)

    def test_cli_option_and_pipeline_outputs(self):
        from profiling_tools.__main__ import build_parser
        from profiling_tools.report import write_reports
        args = build_parser().parse_args(["--gpu-trace", "g", "--npu-trace", "n", "--npu-kernel-details", "k",
                                         "--output-dir", "o", "--diagnostic-coverage-threshold", "100"])
        self.assertEqual(args.diagnostic_coverage_threshold, 100)
        gpu, npu, matches, _ = case()
        gpu.kernels, gpu.ops[0].kernel_indices = [], []
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_reports(tmp, gpu, npu, matches, [], [], {}, diagnostic_coverage_threshold=100)
            self.assertEqual(summary["coverage_diagnostics"]["selected_type_count"], 1)
            self.assertIn("coverage_diagnostics.md", summary["reports"])
            self.assertTrue((Path(tmp) / "coverage_diagnostics.json").is_file())


if __name__ == "__main__":
    unittest.main()

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from profiling_tools.model import OpMatch
from profiling_tools.precision_analysis import build_precision_rows, summarize_precision_rows
from profiling_tools.report import write_reports
from profiling_tools.tests.test_logical_calls import op, profile


def precision_case(main="Muls", before=True, after=True):
    cpu = "aten::sub" if main == "Sub" else "aten::div"
    api = "aclnnSubs_" if main == "Sub" else "aclnnDivs_"
    types = (["Cast"] if before else []) + [main] + (["Cast"] if after else [])
    dtype = '["c10::Half","float64"]' if before else '["float32","float64"]'
    gpu = profile("gpu", [op(cpu, 0, dtype=dtype)], [(0, "vectorized_MulFunctor<float>")])
    npu = profile("npu", [op(cpu, 0, dtype=dtype)], [(0, t) for t in types])
    gpu.kernels[0].link_method = "external_id"
    for i, k in enumerate(npu.kernels):
        k.link_method = "torch_to_npu"
        k.stream_id = "7"
        d = k.detail
        d.name = api + d.op_type
        d.stream_id, d.device_id = "7", "0"
        d.input_shapes = d.output_shapes = '"4"'
        d.input_dtypes = d.output_dtypes = "FLOAT"
        if d.op_type == "Cast":
            d.input_dtypes, d.output_dtypes = ("FLOAT16", "FLOAT") if before and i == 0 else ("FLOAT", "FLOAT16")
    return gpu, npu, [OpMatch(1, 0, 0, 1, "high")]


class PrecisionAnalysisTest(unittest.TestCase):
    def analyze(self, case):
        return build_precision_rows(*case)[0]

    def test_round_trip_muls_realdiv_and_sub(self):
        for main in ("Muls", "RealDiv", "Sub"):
            with self.subTest(main=main):
                case = precision_case(main)
                original = copy.deepcopy(case)
                rows, diagnostics = build_precision_rows(*case)
                self.assertEqual(case, original, "diagnosis must not mutate mapping inputs")
                row = rows[0]
                self.assertEqual(row["Pattern"], "round_trip_promotion")
                self.assertTrue(row["Comparable"])
                self.assertEqual(row["Cast Time(us)"], 4)
                self.assertEqual(row["NPU Total Time(us)"], 6)
                self.assertAlmostEqual(row["Comparable Ratio(GPU/NPU)"], 1 / 3, places=6)
                self.assertEqual(row["Candidate Cause"], "possible_unfused_conversion")
                self.assertEqual(row["Cause Status"], "not_verified")
                self.assertEqual(diagnostics["unique_gpu_kernel_count"], 1)

    def test_pre_cast_only_and_post_cast_only(self):
        self.assertEqual(self.analyze(precision_case(after=False))[0]["Pattern"], "input_promotion")
        row = self.analyze(precision_case(before=False))[0]
        self.assertEqual(row["Pattern"], "output_conversion")
        self.assertEqual(row["Candidate Cause"], "requires_validation")

    def test_native_half_and_float(self):
        case = precision_case("Sub", False, False)
        self.assertEqual(self.analyze(case)[0]["Pattern"], "native_fp32")
        case[1].kernels[0].detail.input_dtypes = "FLOAT16"
        case[1].kernels[0].detail.output_dtypes = "FLOAT16"
        case[0].ops[0].input_dtypes = case[1].ops[0].input_dtypes = '["float16","float64"]'
        row = self.analyze(case)[0]
        self.assertEqual(row["Pattern"], "native_fp16")
        self.assertEqual(row["Cast Time(us)"], 0)
        self.assertEqual(row["Candidate Cause"], "no_external_cast")

    def test_bfloat16_and_cpu_aliases(self):
        gpu, npu, matches = precision_case()
        gpu.ops[0].input_dtypes = '["torch.float16","float64"]'
        self.assertTrue(self.analyze((gpu, npu, matches))[0]["Comparable"])
        npu.kernels[0].detail.input_dtypes = "BFLOAT16"
        npu.kernels[-1].detail.output_dtypes = "BFLOAT16"
        self.assertFalse(self.analyze((gpu, npu, matches))[0]["Comparable"])
        gpu.ops[0].input_dtypes = npu.ops[0].input_dtypes = '["bfloat16","float64"]'
        self.assertEqual(self.analyze((gpu, npu, matches))[0]["Pattern"], "round_trip_promotion")
        self.assertTrue(self.analyze((gpu, npu, matches))[0]["Comparable"])

    def test_missing_dtype_and_shape_are_not_zero_cost_success(self):
        for field in ("input_dtypes", "output_dtypes", "input_shapes", "output_shapes"):
            case = precision_case()
            setattr(case[1].kernels[0].detail, field, "")
            row = self.analyze(case)[0]
            self.assertEqual(row["Pattern"], "incomplete_metadata")
            self.assertFalse(row["Comparable"])
            self.assertEqual(row["GPU Total Time(us)"], "")
            self.assertEqual(summarize_precision_rows([row])[0]["Comparable Ratio(GPU/NPU)"], "")

    def test_shape_mismatch_and_missing_detail(self):
        case = precision_case()
        case[1].kernels[0].detail.input_shapes = case[1].kernels[0].detail.output_shapes = '"8"'
        self.assertEqual(self.analyze(case)[0]["Pattern"], "ambiguous_path")
        case[1].kernels[0].detail = None
        self.assertFalse(self.analyze(case)[0]["Comparable"])

    def test_cross_stream_or_device_is_not_a_chain(self):
        for field in ("stream_id", "device_id"):
            case = precision_case()
            setattr(case[1].kernels[1].detail, field, "9")
            row = self.analyze(case)[0]
            self.assertEqual(row["Pattern"], "ambiguous_path")
            self.assertIn("multiple_or_unknown_streams", row["Pattern Issues"])

    def test_epoch_timestamp_uses_exact_detail_order(self):
        case = precision_case()
        for i, k in enumerate(case[1].kernels):
            k.event.timestamp_us = 1790085129496033.0  # Float timestamp lost order.
            k.event.duration_us = .01
            k.detail.raw = {"Start Time(us)": f"1790085129496033.0{i + 1}0\t", "Duration(us)": "0.005"}
        row = self.analyze(case)[0]
        self.assertEqual(row["Pattern"], "round_trip_promotion")
        self.assertEqual(row["NPU Total Time(us)"], .015)
        self.assertEqual(json.loads(row["NPU Kernel Details"])[0]["start_us"], "1790085129496033.010")

    def test_overlap_and_unrecognized_precision_do_not_compare(self):
        case = precision_case()
        case[1].kernels[1].event.timestamp_us = 100.5
        self.assertIn("overlapping_kernel_intervals", self.analyze(case)[0]["Pattern Issues"])
        case = precision_case()
        case[1].kernels[0].detail.input_dtypes = "INT64"
        self.assertFalse(self.analyze(case)[0]["Comparable"])

    def test_wrong_api_and_explicit_conversion_are_not_div_helpers(self):
        case = precision_case()
        case[1].kernels[0].detail.name = "aclnnInplaceCopy_Cast"
        row = self.analyze(case)[0]
        self.assertIn("mixed_or_missing_api", row["Pattern Issues"])
        self.assertEqual(row["Candidate Cause"], "requires_validation")
        for k in case[1].kernels:
            k.detail.name = "aclnnOther_" + k.detail.op_type
        self.assertEqual(self.analyze(case), [])

    def test_unmatched_and_low_confidence_retained_without_ratio(self):
        gpu, npu, _ = precision_case()
        for matches in ([], [OpMatch(1, 0, 0, .4, "low")],
                        [OpMatch(1, 0, 0, 1, "high"), OpMatch(2, 0, 0, 1, "high")]):
            row = self.analyze((gpu, npu, matches))[0]
            self.assertEqual(row["Pattern"], "round_trip_promotion")
            self.assertFalse(row["Comparable"])
            self.assertEqual(row["Comparable Ratio(GPU/NPU)"], "")

    def test_metadata_conflicts_and_weak_link(self):
        for field, value in (("input_shapes", "[[8],[]]"),
                             ("input_dtypes", '["float32","float64"]'),
                             ("input_dtypes", ""), ("module_key", "other")):
            case = precision_case()
            setattr(case[0].ops[0], field, value)
            self.assertFalse(self.analyze(case)[0]["Comparable"])
        case = precision_case()
        case[1].kernels[0].link_method = "flow"
        self.assertIn("weak_npu_owner_evidence", self.analyze(case)[0]["Comparison Status"])

    def test_host_queue_evidence_and_unknown_shapes(self):
        for method in ("host_to_device_queue", "host_to_device_thread"):
            case = precision_case()
            for k in case[1].kernels:
                k.link_method = method
            self.assertTrue(self.analyze(case)[0]["Comparable"])
        for value in ("N/A", "unknown", "null", "[]"):
            case = precision_case()
            case[1].kernels[1].detail.input_shapes = value
            self.assertFalse(self.analyze(case)[0]["Comparable"])
        case = precision_case()
        case[0].ops[0].input_shapes = case[1].ops[0].input_shapes = "[]"
        self.assertIn("missing_cpu_metadata", self.analyze(case)[0]["Comparison Status"])

    def test_multiple_main_kernels_are_ambiguous(self):
        case = precision_case()
        duplicate = copy.deepcopy(case[1].kernels[1])
        duplicate.event.timestamp_us += 100
        case[1].kernels.append(duplicate)
        row = self.analyze(case)[0]
        self.assertFalse(row["Comparable"])
        self.assertEqual(row["Pattern"], "ambiguous_path")
        self.assertIn("non_unique_main_or_other_kernel", row["Pattern Issues"])

    def test_missing_gpu_work_is_not_zero_time_or_no_cast_evidence(self):
        case = precision_case()
        case[0].kernels = []
        row = self.analyze(case)[0]
        self.assertFalse(row["Comparable"])
        self.assertEqual(row["GPU Cast Evidence"], "unknown")
        self.assertEqual(row["GPU Total Time(us)"], "")

    def test_shared_gpu_memory_is_not_counted_twice(self):
        gpu, npu, matches = precision_case()
        gpu.ops.append(op("aten::div", 1, parent=0, dtype=gpu.ops[0].input_dtypes))
        extra_gpu = copy.deepcopy(gpu.kernels[0])
        extra_gpu.owner_op_index = 1
        gpu.kernels.append(extra_gpu)
        memory = copy.deepcopy(extra_gpu)
        memory.event.category = "gpu_memcpy"
        gpu.memory_activities = [memory]
        npu.ops.append(op("aten::div", 1, dtype=npu.ops[0].input_dtypes))
        for k in copy.deepcopy(npu.kernels):
            k.owner_op_index = 1
            k.event.timestamp_us += 100
            npu.kernels.append(k)
        matches.append(OpMatch(2, 1, 1, 1, "high"))
        rows, info = build_precision_rows(gpu, npu, matches)
        self.assertEqual(info["unique_gpu_memory_activity_count"], 1)
        self.assertFalse(rows[1]["Comparable"])

    def test_gpu_conversion_child_not_hidden(self):
        gpu, npu, matches = precision_case()
        gpu.ops.append(op("aten::_to_copy", 1, parent=0))
        gpu.kernels[0].owner_op_index = 1
        row = self.analyze((gpu, npu, matches))[0]
        self.assertTrue(row["Comparable"])
        self.assertEqual(row["GPU Cast Evidence"], "conversion_child_observed")
        self.assertEqual(row["Candidate Cause"], "requires_validation")

    def test_gpu_name_evidence_does_not_match_broadcast(self):
        case = precision_case()
        case[0].kernels[0].event.name = "broadcast_kernel"
        self.assertEqual(self.analyze(case)[0]["GPU Cast Evidence"], "no_standalone_cast_observed")
        case[0].kernels[0].event.name = "cast_kernel<Half,float>"
        self.assertEqual(self.analyze(case)[0]["GPU Cast Evidence"], "cast_kernel_name_observed")

    def test_memory_time_included_but_runtime_time_not_used(self):
        case = precision_case()
        memory = copy.deepcopy(case[0].kernels[0])
        memory.event.category, memory.event.name, memory.event.duration_us = "gpu_memcpy", "Memcpy HtoD", .5
        case[0].memory_activities = [memory]
        row = self.analyze(case)[0]
        self.assertEqual(row["GPU Total Time(us)"], 2.5)
        self.assertEqual(row["GPU Memory Activity Indices"], "[0]")

    def test_shared_gpu_scope_counted_once(self):
        gpu, npu, matches = precision_case()
        npu.ops.append(op("aten::div", 1, dtype=npu.ops[0].input_dtypes))
        second = copy.deepcopy(npu.kernels)
        for i, k in enumerate(second):
            k.owner_op_index = 1
            k.event.timestamp_us += 100
        npu.kernels.extend(second)
        matches.append(OpMatch(2, 0, 1, 1, "high"))
        rows, info = build_precision_rows(gpu, npu, matches)
        self.assertEqual(len(rows), 2)
        self.assertEqual(sum(r["Comparable"] for r in rows), 1)
        self.assertIn("gpu_scope_already_used", rows[1]["Comparison Status"])
        self.assertEqual(info["unique_gpu_kernel_count"], 1)
        self.assertEqual(sum(r["GPU Total Time(us)"] or 0 for r in rows), 2)

    def test_same_op_nested_wrapper_is_one_complete_call(self):
        gpu, npu, matches = precision_case()
        npu.ops.append(op("aten::div", 1, parent=0, dtype=npu.ops[0].input_dtypes))
        npu.kernels[1].owner_op_index = 1
        rows = self.analyze((gpu, npu, matches))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["NPU Kernel Count"], 3)
        self.assertTrue(rows[0]["Comparable"])

    def test_adjacent_orphan_not_attached_by_time(self):
        case = precision_case()
        case[1].kernels[0].owner_op_index = None
        rows, info = build_precision_rows(*case)
        self.assertEqual(rows[0]["NPU Kernel Count"], 2)
        self.assertEqual(rows[0]["Pattern"], "output_conversion")
        self.assertEqual(info["unscoped_api_kernel_count"], 1)

    def test_summary_ratio_uses_same_population_not_average_ratios(self):
        rows = self.analyze(precision_case())
        row2 = copy.deepcopy(rows[0])
        row2.update({"NPU Total Time(us)": 12, "GPU Total Time(us)": 6, "Cast Time(us)": 8})
        stat = summarize_precision_rows(rows + [row2])[0]
        self.assertEqual(stat["Call Count"], 2)
        self.assertEqual(stat["Cast Time(us)"], 12)
        self.assertAlmostEqual(stat["Comparable Ratio(GPU/NPU)"], 8 / 18, places=6)

    def test_reports_integrate_without_changing_legacy_assignments(self):
        gpu, npu, matches = precision_case()
        from profiling_tools.report import build_npu_kernel_gpu_mapping_rows
        before = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_reports(tmp, gpu, npu, matches, [], [], {})
            self.assertEqual(before, build_npu_kernel_gpu_mapping_rows(gpu, npu, matches))
            self.assertEqual(summary["precision_analysis"]["call_count"], 1)
            self.assertNotIn("precision_path_details.csv", summary["reports"])
            self.assertNotIn("precision_path_statistic.csv", summary["reports"])
            report = json.loads((Path(tmp) / "coverage_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(report["specialized_evidence"]["precision_paths"]), 1)
            self.assertIn("NPU Kernel Details", report["specialized_evidence"]["precision_paths"][0])
            text = (Path(tmp) / "coverage_diagnostics.md").read_text(encoding="utf-8")
            self.assertIn("完整精度路径", text)
            self.assertIn("round_trip_promotion", text)
            self.assertIn("可比完整调用 1 次", text)
            self.assertNotIn("precision_path_details.csv", text)
            ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            with zipfile.ZipFile(Path(tmp) / "npu_gpu_op_statistic.xlsx") as z:
                root = ET.fromstring(z.read("xl/workbook.xml"))
                self.assertEqual([s.attrib["name"] for s in root.findall(".//m:sheet", ns)],
                                 ["总览", "覆盖率诊断", "耗时分布", "整体算子"])

    def test_path_evidence_survives_zero_diagnostic_threshold(self):
        gpu, npu, matches = precision_case()
        with tempfile.TemporaryDirectory() as tmp:
            write_reports(tmp, gpu, npu, matches, [], [], {}, diagnostic_coverage_threshold=0)
            report = json.loads((Path(tmp) / "coverage_diagnostics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["types"], [])
            self.assertEqual(len(report["specialized_evidence"]["precision_paths"]), 1)
            self.assertIn("round_trip_promotion", (Path(tmp) / "coverage_diagnostics.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

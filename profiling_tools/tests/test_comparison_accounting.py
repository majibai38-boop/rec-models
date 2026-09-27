import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from profiling_tools.comparison_accounting import add_coverage_conclusions, complete_gpu_accounting
from profiling_tools.coverage_diagnostics import build_coverage_diagnostics
from profiling_tools.report import build_npu_gpu_op_statistic_rows, write_reports
from profiling_tools.tests.test_logical_calls import op, profile


def mapping(npu_index=0, *, op_type="NewType", gpu_indices=(0,), memory_indices=(), duration=4):
    return {"Device_id": "0", "OP Type": op_type, "Core Type": "AI_CORE", "NPU Kernel Index": npu_index,
            "NPU Duration(us)": duration, "Mapped": bool(gpu_indices or memory_indices),
            "Comparable": bool(gpu_indices or memory_indices), "Accounted": True,
            "Mapping Mode": "npu_gpu_bundle_shared", "CPU Match ID": 1,
            "GPU Kernel Indices": json.dumps(gpu_indices), "GPU Kernel Durations(us)": json.dumps([2] * len(gpu_indices)),
            "GPU Memory Activity Indices": json.dumps(memory_indices),
            "GPU Memory Activity Durations(us)": json.dumps([1] * len(memory_indices))}


class ComparisonAccountingTest(unittest.TestCase):
    def setUp(self):
        self.gpu = profile("gpu", [op("aten::new", 0)], [(0, "k0"), (0, "k1"), (0, "k2")])
        self.gpu.memory_activities = [copy.deepcopy(k) for k in self.gpu.kernels[:2]]
        for record in self.gpu.memory_activities:
            record.event.duration_us = 1

    def run_accounting(self, rows, **kwargs):
        types = build_npu_gpu_op_statistic_rows(rows)
        add_coverage_conclusions(types, {"types": [], "selection": {"threshold_pct": 90}})
        return complete_gpu_accounting(self.gpu, rows, types, **kwargs)

    def test_shared_bundle_dedup_and_namespaces(self):
        source = [mapping(0, memory_indices=(0,)), mapping(1, memory_indices=(0,))]
        original = copy.deepcopy(source)
        rows, evidence, audit = self.run_accounting(source)
        self.assertEqual(source, original)
        self.assertEqual(audit["kernel"]["mapped_count"], 1)
        self.assertEqual(audit["memory"]["mapped_count"], 1)
        self.assertTrue(audit["type_rows_additive"])
        self.assertEqual({(r["GPU Activity Kind"], r["GPU Activity Index"]) for r in evidence},
                         {("kernel", 1), ("kernel", 2), ("memory", 1)})
        self.assertEqual(rows[-1]["GPU Total Time(us)"], 8)
        self.assertEqual(rows[-1]["NPU Total Time(us)"], 8)
        self.assertEqual(rows[-1]["Total Time Ratio(GPU/NPU)"], 1)
        self.assertEqual(rows[-1]["Comparable Ratio(GPU/NPU)"], 3 / 8)
        self.assertEqual(sum(r["GPU Total Time(us)"] for r in rows[:-1]), 8)
        self.assertEqual(sum(r["NPU Total Time(us)"] for r in rows[:-1]), 8)

    def test_whole_call_only_stays_in_type_remainder(self):
        self.gpu.kernels[2].owner_op_index = None
        whole = [{"GPU Kernel Indices": "[1]", "CPU Bundle ID": "Stack:1", "CPU Bundle": "Stack"}]
        rows, evidence, audit = self.run_accounting([mapping()], whole_call_rows=whole)
        self.assertEqual(audit["kernel"]["unmapped_count"], 2)
        self.assertEqual(audit["kernel"]["whole_call_only_count"], 1)
        kernels = {r["GPU Activity Index"]: r for r in evidence if r["GPU Activity Kind"] == "kernel"}
        self.assertEqual(kernels[1]["Status"], "whole_call_only")
        self.assertEqual(kernels[2]["Status"], "gpu_owner_missing")
        self.assertEqual(rows[-1]["GPU Total Time(us)"], 8)
        whole[0]["Comparable"] = False
        self.assertEqual(self.run_accounting([mapping()], whole_call_rows=whole)[2]["kernel"]["whole_call_only_count"], 0)

    def test_cross_type_reuse_warns_and_dedups_overall(self):
        rows, _, audit = self.run_accounting([mapping(), mapping(1, op_type="Other")])
        self.assertFalse(audit["type_rows_additive"])
        self.assertEqual(audit["kernel"]["cross_type_reused_indices"], [0])
        self.assertEqual(rows[-1]["GPU Total Time(us)"], 8)
        self.assertEqual(rows[-1]["Comparable Ratio(GPU/NPU)"], 2 / 8)
        self.assertIn("不可相加", rows[-1]["排查结论"])

    def test_all_mapped_emits_zero_remainder_and_na_ratios(self):
        rows, evidence, audit = self.run_accounting([mapping(gpu_indices=(0, 1, 2), memory_indices=(0, 1))])
        self.assertEqual(evidence, [])
        self.assertEqual(len(rows), 3)
        remainder = rows[-2]
        self.assertEqual(remainder["GPU Kernel Count"], 0)
        self.assertEqual(remainder["GPU Total Time(us)"], 0)
        self.assertEqual(remainder["Comparable Coverage(%)"], "")
        self.assertEqual(remainder["Total Time Ratio(GPU/NPU)"], "")

    def test_empty_npu_and_empty_both_pipeline(self):
        for gpu in (self.gpu, profile("gpu", [])):
            with tempfile.TemporaryDirectory() as tmp:
                write_reports(tmp, gpu, profile("npu", []), [], [], [], {})
                with (Path(tmp) / "npu_gpu_op_statistic.csv").open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(rows[-1]["Row Kind"], "total")
                self.assertEqual(rows[-1]["Comparable Coverage(%)"], "")
                self.assertEqual(rows[-1]["Total Time Ratio(GPU/NPU)"], "")
                self.assertEqual(int(rows[-1]["GPU Kernel Count"]), len(gpu.kernels))

    def test_zero_duration_npu_and_no_gpu(self):
        self.gpu = profile("gpu", [])
        rows, _, _ = self.run_accounting([mapping(gpu_indices=(), duration=0)])
        self.assertEqual(rows[0]["Comparable Time Coverage(%)"], "")
        self.assertEqual(rows[-1]["Total Time Ratio(GPU/NPU)"], "")

    def test_multiple_devices_not_inferred_from_gpu_ids(self):
        a, b = mapping(), mapping(1, gpu_indices=(1,))
        b["Device_id"] = "7"
        rows, _, _ = self.run_accounting([a, b])
        self.assertEqual({r["Device_id"] for r in rows if r["Row Kind"] == "npu_type"}, {"0", "7"})
        self.assertEqual(rows[-1]["Device_id"], "ALL_SELECTED")
        self.assertEqual(rows[-1]["NPU Total Time(us)"], 8)

    def test_bad_index_and_duration_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            self.run_accounting([mapping(gpu_indices=(99,))])
        self.gpu.kernels[0].event.duration_us = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.run_accounting([])

    def test_conclusions_are_generic_and_do_not_raise_coverage(self):
        npu = profile("npu", [op("aten::new", 0)], [(0, "NewType"), (0, "NewType")])
        source = [mapping(), mapping(1, gpu_indices=())]
        report = build_coverage_diagnostics(self.gpu, npu, [], source)
        rows = build_npu_gpu_op_statistic_rows(source)
        original = copy.deepcopy(rows)
        add_coverage_conclusions(rows, report)
        self.assertIn("排查结论", rows[0])
        self.assertIn("排查建议", rows[0])
        self.assertEqual(rows[0]["Comparable Coverage(%)"], original[0]["Comparable Coverage(%)"])
        self.assertNotIn("Div", rows[0]["排查结论"])
        add_coverage_conclusions(rows, {"types": [], "selection": {"threshold_pct": 0}})
        self.assertIn("未触发", rows[0]["排查结论"])
        self.assertNotIn("全部纳入", rows[0]["排查结论"])


if __name__ == "__main__":
    unittest.main()

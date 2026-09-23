from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from profiling_tools.compare import CompareConfig, compare_profiles


def complete(name, category, ts, dur, pid, tid, args=None):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "ts": ts,
        "dur": dur,
        "pid": pid,
        "tid": tid,
        "args": args or {},
    }


class CompareProfilesTest(unittest.TestCase):
    def test_bundle_comparison_and_npu_type_mapping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "kernel_details.csv"
            output = root / "out"

            gpu_events = [
                complete("ProfilerStep#0", "user_annotation", 0, 100, 10, 10),
                complete("model", "user_annotation", 1, 90, 10, 10),
                complete(
                    "aten::linear",
                    "cpu_op",
                    10,
                    10,
                    10,
                    10,
                    {"External id": 1, "Input Dims": [[4, 8], [16, 8]]},
                ),
                complete(
                    "gpu_gemm",
                    "kernel",
                    30,
                    3,
                    0,
                    7,
                    {"External id": 1, "stream": 7},
                ),
                complete(
                    "aten::relu",
                    "cpu_op",
                    40,
                    8,
                    10,
                    10,
                    {"External id": 2, "Input Dims": [[4, 16]]},
                ),
                complete(
                    "gpu_relu_a",
                    "kernel",
                    50,
                    2,
                    0,
                    7,
                    {"External id": 2, "stream": 7},
                ),
                complete(
                    "gpu_relu_b",
                    "kernel",
                    51,
                    2,
                    0,
                    8,
                    {"External id": 2, "stream": 8},
                ),
            ]
            npu_events = [
                complete("ProfilerStep#0", "user_annotation", 0, 100, 20, 20),
                complete("model", "user_annotation", 1, 90, 20, 20),
                complete(
                    "aten::linear",
                    "cpu_op",
                    10,
                    10,
                    20,
                    20,
                    {"External id": 11, "Input Dims": [[4, 8], [16, 8]]},
                ),
                complete(
                    "npu_matmul",
                    "kernel",
                    30,
                    5,
                    1,
                    9,
                    {"External id": 11, "Task ID": 101, "Stream ID": 9},
                ),
                complete(
                    "aten::relu",
                    "cpu_op",
                    40,
                    8,
                    20,
                    20,
                    {"External id": 12, "Input Dims": [[4, 16]]},
                ),
                complete(
                    "npu_relu",
                    "kernel",
                    50,
                    6,
                    1,
                    9,
                    {"External id": 12, "Task ID": 102, "Stream ID": 9},
                ),
            ]
            gpu_trace.write_text(json.dumps({"traceEvents": gpu_events}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": npu_events}), encoding="utf-8")
            with details.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "Step Id",
                        "Task ID",
                        "Stream ID",
                        "Name",
                        "Type",
                        "Accelerator Core",
                        "Start Time(us)",
                        "Duration(us)",
                        "Input Shapes",
                        "Input Data Types",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "Step Id": "0",
                            "Task ID": "101",
                            "Stream ID": "9",
                            "Name": "npu_matmul",
                            "Type": "MatMulV2",
                            "Accelerator Core": "AI_CORE",
                            "Start Time(us)": "30",
                            "Duration(us)": "5",
                            "Input Shapes": "4x8;16x8",
                            "Input Data Types": "FLOAT;FLOAT",
                        },
                        {
                            "Step Id": "0",
                            "Task ID": "102",
                            "Stream ID": "9",
                            "Name": "npu_relu",
                            "Type": "Relu",
                            "Accelerator Core": "AI_VECTOR_CORE",
                            "Start Time(us)": "50",
                            "Duration(us)": "6",
                            "Input Shapes": "4x16",
                            "Input Data Types": "FLOAT",
                        },
                    ]
                )

            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                    gpu_step="0",
                    npu_step="0",
                )
            )

            self.assertEqual(summary["comparison"]["matched_op_count"], 2)
            self.assertEqual(summary["npu"]["kernel_detail_matched_count"], 2)

            with (output / "matches.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["NPU Kernel Type"], "MatMulV2")
            self.assertEqual(rows[1]["GPU Kernel Count"], "2")
            self.assertEqual(rows[1]["NPU Kernel Count"], "1")
            self.assertEqual(rows[1]["GPU Kernel Sum(us)"], "4.0")
            self.assertEqual(rows[1]["GPU Device Busy(us)"], "3.0")

            with (output / "gpu_kernel_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mapping_rows = list(csv.DictReader(handle))
            relu_rows = [row for row in mapping_rows if row["Canonical OP"] == "aten::relu"]
            self.assertEqual({row["Mapping Mode"] for row in relu_rows}, {"bundle"})
            self.assertEqual({row["Mapped NPU Type"] for row in relu_rows}, {"Relu"})

    def test_operator_map(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            op_map = root / "map.json"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::gpu_alias", "cpu_op", 1, 2, 1, 1, {"External id": 1}),
                            complete("g", "kernel", 4, 1, 2, 2, {"External id": 1}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::npu_alias", "cpu_op", 1, 2, 3, 3, {"External id": 2}),
                            complete("n", "kernel", 4, 1, 4, 4, {"External id": 2}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text("Name,Type\nn,NpuAlias\n", encoding="utf-8")
            op_map.write_text(
                json.dumps({"aten::gpu_alias": "aten::npu_alias"}), encoding="utf-8"
            )
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(root / "out"),
                    op_map=str(op_map),
                )
            )
            self.assertEqual(summary["comparison"]["matched_op_count"], 1)

    def test_chrome_flow_links_kernel_without_external_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            flow_start = {
                "name": "cpu_to_gpu",
                "cat": "ac2g",
                "ph": "s",
                "ts": 15,
                "pid": 1,
                "tid": 1,
                "id": 88,
                "args": {},
            }
            flow_finish = {
                "name": "cpu_to_gpu",
                "cat": "ac2g",
                "ph": "f",
                "ts": 30,
                "pid": 2,
                "tid": 7,
                "id": 88,
                "args": {},
            }
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::add", "cpu_op", 10, 10, 1, 1),
                            flow_start,
                            flow_finish,
                            complete("add_kernel", "kernel", 30, 3, 2, 7),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::add", "cpu_op", 10, 10, 3, 3, {"External id": 3}),
                            complete("npu_add", "kernel", 30, 4, 4, 9, {"External id": 3}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text("Name,Type\nnpu_add,Add\n", encoding="utf-8")
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(root / "out"),
                )
            )
            self.assertEqual(summary["gpu"]["kernel_link_methods"], {"flow": 1})


if __name__ == "__main__":
    unittest.main()

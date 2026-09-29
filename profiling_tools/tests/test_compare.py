from __future__ import annotations

import csv
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from profiling_tools.compare import CompareConfig, compare_profiles
from profiling_tools.parser import (
    normalize_input_dtypes,
    normalize_input_shapes,
    normalize_module_key,
)
from profiling_tools.report import build_npu_gpu_op_statistic_rows
from profiling_tools.xlsx_writer import write_merged_table_xlsx


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
    def _write_single_detail(self, path: Path, *, start: int, duration: int) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "Task ID",
                    "Stream ID",
                    "Name",
                    "Type",
                    "Accelerator Core",
                    "Start Time(us)",
                    "Duration(us)",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "Task ID": "101",
                    "Stream ID": "3",
                    "Name": "Slice1",
                    "Type": "Slice",
                    "Accelerator Core": "AI_CORE",
                    "Start Time(us)": str(start),
                    "Duration(us)": str(duration),
                }
            )

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
            self.assertEqual(summary["comparison"]["npu_first_kernel_count"], 2)
            self.assertEqual(summary["comparison"]["npu_first_mapped_kernel_count"], 2)
            self.assertEqual(
                summary["comparison"]["npu_first_duplicate_gpu_assignment_count"], 0
            )

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

            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                npu_first_rows = list(csv.DictReader(handle))
            relu_mapping = next(
                row for row in npu_first_rows if row["OP Type"] == "Relu"
            )
            self.assertEqual(relu_mapping["Mapping Mode"], "npu_to_gpu_bundle")
            self.assertEqual(relu_mapping["GPU Kernel Count"], "2")
            self.assertEqual(relu_mapping["GPU Total Time(us)"], "4.0")

            with (output / "npu_gpu_op_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                npu_first_stats = list(csv.DictReader(handle))
            relu_stat = next(
                row for row in npu_first_stats if row["OP Type"] == "Relu"
            )
            self.assertEqual(relu_stat["NPU Kernel Count"], "1")
            self.assertEqual(relu_stat["GPU Kernel Count"], "2")
            self.assertEqual(relu_stat["NPU Total Time(us)"], "6.0")
            self.assertEqual(relu_stat["GPU Total Time(us)"], "4.0")
            self.assertEqual(relu_stat["Comparable Ratio(GPU/NPU)"], "0.666667")
            self.assertEqual(relu_stat["Total Time Ratio(GPU/NPU)"], "0.666667")
            self.assertEqual(relu_stat["Direct Mapping Coverage(%)"], "0.0")
            self.assertEqual(relu_stat["Comparable Coverage(%)"], "100.0")
            self.assertEqual(relu_stat["Accounted Coverage(%)"], "100.0")

    def test_comparable_and_total_ratios_have_distinct_denominators_and_colors(self):
        statistic = build_npu_gpu_op_statistic_rows(
            [
                {
                    "Device_id": "0",
                    "OP Type": "Partial",
                    "Core Type": "AI_CORE",
                    "NPU Duration(us)": 2.0,
                    "Mapped": True,
                    "GPU Kernel Durations(us)": "[6.0]",
                    "Mapping Mode": "one_to_one",
                    "CPU Match ID": "1",
                },
                {
                    "Device_id": "0",
                    "OP Type": "Partial",
                    "Core Type": "AI_CORE",
                    "NPU Duration(us)": 4.0,
                    "Mapped": False,
                    "GPU Kernel Durations(us)": "[]",
                    "Mapping Mode": "unmatched_cpu_op",
                    "CPU Match ID": "",
                },
            ]
        )[0]
        self.assertEqual(statistic["Comparable Ratio(GPU/NPU)"], 3.0)
        self.assertEqual(statistic["Total Time Ratio(GPU/NPU)"], 1.0)

        ratio_columns = (
            "Comparable Ratio(GPU/NPU)",
            "Total Time Ratio(GPU/NPU)",
        )
        ratio_values = (2.5, 1.5, 0.5, 1.0, 2.0, "")
        with tempfile.TemporaryDirectory() as temp_dir:
            workbook_path = Path(temp_dir) / "ratios.xlsx"
            write_merged_table_xlsx(
                workbook_path,
                [dict.fromkeys(ratio_columns, value) for value in ratio_values],
                ratio_columns,
                sheet_name="Ratios",
                group_key_columns=(),
                merge_columns=(),
            )
            with zipfile.ZipFile(workbook_path) as workbook:
                worksheet = ElementTree.fromstring(
                    workbook.read("xl/worksheets/sheet1.xml")
                )
                styles = ElementTree.fromstring(workbook.read("xl/styles.xml"))
        namespace = {
            "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        }
        for row_number, style_id in enumerate((8, 9, 10, 9, 8, 7), 2):
            for column in "AB":
                cell = worksheet.find(
                    f".//main:c[@r='{column}{row_number}']", namespace
                )
                self.assertEqual(cell.attrib["s"], str(style_id))
        fills = styles.findall(".//main:fills/main:fill", namespace)
        self.assertEqual(
            [fills[index].find("main:patternFill/main:fgColor", namespace).attrib["rgb"]
             for index in (3, 4, 5)],
            ["FF1B5E20", "FFC6E0B4", "FFFFF2CC"],
        )

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

    def test_ascend_hardware_task_without_kernel_category(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            device_start = 1_699_529_623_106_750

            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::slice", "cpu_op", 200, 100, 10, 2, {"External id": 1}),
                            complete("gpu_slice", "kernel", 400, 10, 11, 3, {"External id": 1}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {
                                "ph": "M",
                                "name": "process_name",
                                "pid": 4,
                                "tid": 0,
                                "args": {"name": "Ascend Hardware"},
                            },
                            complete("aten::slice", "cpu_op", 200, 100, 1, 2),
                            {
                                "ph": "s",
                                "name": "link1",
                                "id": 1,
                                "pid": 1,
                                "tid": 3,
                                "ts": 200,
                                "args": {},
                            },
                            {
                                "ph": "f",
                                "name": "link1",
                                "id": 1,
                                "pid": 1,
                                "tid": 3,
                                "ts": device_start,
                                "args": {},
                            },
                            complete(
                                "Slice1",
                                "",
                                device_start,
                                80,
                                4,
                                3,
                                {"Task Type": "AI_CORE"},
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self._write_single_detail(details, start=device_start, duration=80)

            output = root / "out"
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            self.assertEqual(summary["npu"]["device_process_ids"], ["4"])
            self.assertEqual(summary["npu"]["kernel_count"], 1)
            self.assertEqual(summary["npu"]["linked_kernel_count"], 1)
            self.assertEqual(summary["npu"]["kernel_link_methods"], {"flow": 1})
            with (output / "matches.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["NPU Kernel Count"], "1")
            self.assertEqual(row["NPU Kernel Type"], "Slice")

    def test_kernel_details_row_fills_missing_npu_trace_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            device_start = 1_699_529_623_106_750
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::slice", "cpu_op", 200, 100, 10, 2, {"External id": 1}),
                            complete("gpu_slice", "kernel", 400, 10, 11, 3, {"External id": 1}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::slice", "cpu_op", 200, 100, 1, 2),
                            {
                                "ph": "s",
                                "name": "link1",
                                "id": 1,
                                "pid": 1,
                                "tid": 3,
                                "ts": 200,
                                "args": {},
                            },
                            {
                                "ph": "f",
                                "name": "link1",
                                "id": 1,
                                "pid": 1,
                                "tid": 3,
                                "ts": device_start,
                                "args": {},
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self._write_single_detail(details, start=device_start, duration=80)

            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(root / "out"),
                )
            )
            self.assertEqual(summary["npu"]["kernel_count"], 1)
            self.assertEqual(summary["npu"]["linked_kernel_count"], 1)
            self.assertEqual(summary["npu"]["kernel_detail_matched_count"], 0)
            self.assertEqual(summary["npu"]["kernel_detail_synthetic_count"], 1)

    def test_metadata_normalization_and_confidence_gated_statistics(self):
        self.assertEqual(normalize_input_shapes([[2, 3], []]), "[[2,3],[]]")
        self.assertEqual(normalize_input_shapes("2,3;"), "[[2,3],[]]")
        self.assertEqual(normalize_input_dtypes(["float", "long int"]), '["float32","int64"]')
        self.assertEqual(normalize_input_dtypes("FLOAT;INT64"), '["float32","int64"]')
        self.assertEqual(normalize_module_key("nn.Module: Block, callsite: 42"), "block")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            output = root / "out"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("nn.Module: Block, callsite: 42", "user_annotation", 0, 100, 1, 1),
                            complete(
                                "aten::add",
                                "cpu_op",
                                10,
                                10,
                                1,
                                1,
                                {
                                    "External id": 1,
                                    "Input Dims": [[2, 3], [2, 3]],
                                    "Input type": ["float", "float"],
                                },
                            ),
                            complete("gpu_add", "kernel", 30, 2, 2, 2, {"External id": 1}),
                            complete(
                                "aten::view",
                                "cpu_op",
                                40,
                                5,
                                1,
                                1,
                                {"Input Dims": [[2, 3]], "Input type": ["float"]},
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("nn.Module: Block", "user_annotation", 0, 100, 3, 3),
                            complete(
                                "aten::add",
                                "cpu_op",
                                10,
                                10,
                                3,
                                3,
                                {
                                    "External id": 2,
                                    "Input Shapes": "2,3;2,3",
                                    "Input Data Types": "FLOAT;FLOAT",
                                },
                            ),
                            complete(
                                "npu_add",
                                "kernel",
                                30,
                                3,
                                4,
                                4,
                                {"External id": 2, "Task ID": 8, "Stream ID": 4},
                            ),
                            complete(
                                "aten::view",
                                "cpu_op",
                                40,
                                5,
                                3,
                                3,
                                {"Input Shapes": "2,3", "Input Data Types": "FLOAT"},
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "8,4,npu_add,Add,AI_VECTOR_CORE,30,3\n",
                encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            self.assertEqual(summary["comparison"]["confidence_counts"], {"high": 2})
            self.assertEqual(summary["comparison"]["both_kernel_empty_match_count"], 1)
            self.assertEqual(summary["comparison"]["performance_matched_op_count"], 1)
            with (output / "kernel_type_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                type_rows = list(csv.DictReader(handle))
            self.assertEqual([row["Mapped OP Type"] for row in type_rows], ["Add"])
            with (output / "module_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                module_rows = list(csv.DictReader(handle))
            block = next(row for row in module_rows if row["Module Key"] == "block")
            self.assertEqual(block["GPU OP Calls"], "2")
            self.assertEqual(block["NPU OP Calls"], "2")

    def test_unmatched_op_spans_are_reported_as_regions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            output = root / "out"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("nn.Module: Block", "user_annotation", 0, 100, 1, 1),
                            complete("aten::gpu_parts", "cpu_op", 10, 5, 1, 1, {"External id": 1}),
                            complete("gpu_part", "kernel", 20, 4, 2, 2, {"External id": 1}),
                            complete(
                                "aten::relu",
                                "cpu_op",
                                40,
                                5,
                                1,
                                1,
                                {"External id": 2, "Input Dims": [[4]], "Input type": ["float"]},
                            ),
                            complete("gpu_relu", "kernel", 50, 2, 2, 2, {"External id": 2}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("nn.Module: Block", "user_annotation", 0, 100, 3, 3),
                            complete("aten::npu_fused", "cpu_op", 10, 5, 3, 3, {"External id": 3}),
                            complete(
                                "npu_fused",
                                "kernel",
                                20,
                                6,
                                4,
                                4,
                                {"External id": 3, "Task ID": 30, "Stream ID": 4},
                            ),
                            complete(
                                "aten::relu",
                                "cpu_op",
                                40,
                                5,
                                3,
                                3,
                                {
                                    "External id": 4,
                                    "Input Shapes": "4",
                                    "Input Data Types": "FLOAT",
                                },
                            ),
                            complete(
                                "npu_relu",
                                "kernel",
                                50,
                                3,
                                4,
                                4,
                                {"External id": 4, "Task ID": 31, "Stream ID": 4},
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "30,4,npu_fused,FusedOp,AI_CORE,20,6\n"
                "31,4,npu_relu,Relu,AI_VECTOR_CORE,50,3\n",
                encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            self.assertEqual(summary["comparison"]["performance_region_count"], 1)
            with (output / "region_matches.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                region = next(csv.DictReader(handle))
            self.assertEqual(region["Region Type"], "many_to_many")
            self.assertEqual(region["Eligible For Performance"], "True")
            self.assertEqual(region["GPU Kernel Sum(us)"], "4.0")
            self.assertEqual(region["NPU Kernel Sum(us)"], "6.0")

    def test_marker_tasks_are_excluded_from_selected_step(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("ProfilerStep#1", "user_annotation", 0, 100, 1, 1),
                            complete("aten::add", "cpu_op", 10, 5, 1, 1, {"External id": 1}),
                            complete("gpu_add", "kernel", 20, 2, 2, 2, {"External id": 1}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            {"ph": "M", "name": "process_name", "pid": 4, "tid": 0, "args": {"name": "Ascend Hardware"}},
                            complete("ProfilerStep#1", "user_annotation", 0, 100, 3, 3),
                            complete("aten::add", "cpu_op", 10, 5, 3, 3, {"External id": 2}),
                            complete("npu_add", "", 20, 3, 4, 4, {"External id": 2, "Task Type": "AI_CORE"}),
                            complete("DAVID_EVENT_RECORD", "", 30, 1, 4, 4, {"Task Type": "DAVID_EVENT_RECORD"}),
                            complete("PLACE_HOLDER_SQE", "", 120, 1, 4, 4, {"Task Type": "PLACE_HOLDER_SQE"}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text("Name,Type\n", encoding="utf-8")
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(root / "out"),
                    gpu_step="1",
                    npu_step="1",
                )
            )
            self.assertEqual(summary["npu"]["kernel_count"], 1)
            self.assertEqual(summary["npu"]["task_type_counts"], {"AI_CORE": 1})
            self.assertEqual(
                summary["npu"]["excluded_task_type_counts"],
                {"DAVID_EVENT_RECORD": 1},
            )

    def test_kernel_detail_type_order_repairs_nested_flow_owner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            output = root / "out"

            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete(
                                "aten::index",
                                "cpu_op",
                                10,
                                10,
                                1,
                                1,
                                {"External id": 1, "Input Dims": [[4], [2]]},
                            ),
                            complete("gpu_index_1", "kernel", 100, 2, 2, 7, {"External id": 1}),
                            complete(
                                "aten::index",
                                "cpu_op",
                                40,
                                10,
                                1,
                                1,
                                {"External id": 2, "Input Dims": [[4], [2]]},
                            ),
                            complete("gpu_index_2", "kernel", 200, 2, 2, 7, {"External id": 2}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete(
                                "aten::index",
                                "cpu_op",
                                10,
                                20,
                                3,
                                3,
                                {"External id": 11, "Input Shapes": "4;2"},
                            ),
                            complete("aten::ones_like", "cpu_op", 12, 4, 3, 3),
                            {"name": "flow1", "ph": "s", "ts": 13, "pid": 3, "tid": 3, "id": 101, "args": {}},
                            {"name": "flow1", "ph": "f", "ts": 100, "pid": 4, "tid": 7, "id": 101, "args": {}},
                            complete("npu_index_1", "kernel", 100, 3, 4, 7, {"Task ID": 1, "Stream ID": 7}),
                            complete("npu_tile_aux", "kernel", 105, 1, 4, 7, {"External id": 11, "Task ID": 3, "Stream ID": 7}),
                            complete(
                                "aten::index",
                                "cpu_op",
                                40,
                                20,
                                3,
                                3,
                                {"External id": 12, "Input Shapes": "4;2"},
                            ),
                            {"name": "flow2", "ph": "s", "ts": 42, "pid": 3, "tid": 3, "id": 102, "args": {}},
                            {"name": "flow2", "ph": "f", "ts": 200, "pid": 4, "tid": 7, "id": 102, "args": {}},
                            complete("npu_index_2", "kernel", 200, 4, 4, 7, {"Task ID": 2, "Stream ID": 7}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,npu_index_1,Index,AI_CORE,100,3\n"
                "3,7,npu_tile_aux,Tile,AI_VECTOR_CORE,105,1\n"
                "2,7,npu_index_2,Index,AI_CORE,200,4\n",
                encoding="utf-8",
            )

            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            reconciliation = summary["npu"]["kernel_owner_reconciliation"]
            self.assertEqual(reconciliation["groups"]["index"]["moved_kernel_count"], 1)
            self.assertEqual(summary["comparison"]["performance_matched_op_count"], 2)

            with (output / "gpu_kernel_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mapping_rows = list(csv.DictReader(handle))
            self.assertEqual(len(mapping_rows), 2)
            self.assertEqual({row["Mapped NPU Type"] for row in mapping_rows}, {"Index"})
            self.assertEqual(
                {row["NPU Kernel Name"] for row in mapping_rows},
                {"npu_index_1", "npu_index_2"},
            )
            self.assertIn("semantic_primary", {row["Mapping Mode"] for row in mapping_rows})

            with (output / "aten_op_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                index_stat = next(
                    row for row in csv.DictReader(handle) if row["OP Type"] == "aten::index"
                )
            self.assertEqual(index_stat["NPU Same-Type Kernel Count"], "2")
            self.assertEqual(index_stat["NPU Other-Type Kernel Count"], "1")

            with (output / "kernel_link_diagnostic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                link_rows = list(csv.DictReader(handle))
            repaired = next(row for row in link_rows if row["Kernel Name"] == "npu_index_1")
            self.assertEqual(repaired["Owner OP Name"], "aten::index")
            self.assertEqual(repaired["Link Method"], "detail_type_order")
            self.assertEqual(repaired["Original Owner OP Name"], "aten::ones_like")

    def test_kernel_api_family_repairs_alias_owner(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            output = root / "out"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::clamp", "cpu_op", 10, 20, 1, 1, {"External id": 1, "Input Dims": [[4]]}),
                            complete("gpu_clamp_1", "kernel", 100, 2, 2, 7, {"External id": 1}),
                            complete("aten::clamp", "cpu_op", 40, 20, 1, 1, {"External id": 2, "Input Dims": [[4]]}),
                            complete("gpu_clamp_2", "kernel", 200, 2, 2, 7, {"External id": 2}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::clamp", "cpu_op", 10, 20, 3, 3, {"Input Shapes": "4"}),
                            complete("aten::reshape", "cpu_op", 12, 4, 3, 3),
                            {"name": "flow1", "ph": "s", "ts": 13, "pid": 3, "tid": 3, "id": 201, "args": {}},
                            {"name": "flow1", "ph": "f", "ts": 100, "pid": 4, "tid": 7, "id": 201, "args": {}},
                            complete("npu_clip_1", "kernel", 100, 3, 4, 7, {"Task ID": 1, "Stream ID": 7}),
                            complete("aten::clamp", "cpu_op", 40, 20, 3, 3, {"Input Shapes": "4"}),
                            {"name": "flow2", "ph": "s", "ts": 42, "pid": 3, "tid": 3, "id": 202, "args": {}},
                            {"name": "flow2", "ph": "f", "ts": 200, "pid": 4, "tid": 7, "id": 202, "args": {}},
                            complete("npu_clip_2", "kernel", 200, 4, 4, 7, {"Task ID": 2, "Stream ID": 7}),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,aclnnClamp_ClipByValueV2AiCore_ClipByValueV2,ClipByValueV2,AI_VECTOR_CORE,100,3\n"
                "2,7,aclnnClamp_ClipByValueV2AiCore_ClipByValueV2,ClipByValueV2,AI_VECTOR_CORE,200,4\n",
                encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            api_groups = summary["npu"]["kernel_owner_reconciliation"]["api_groups"]
            self.assertEqual(api_groups["aten::clamp"]["moved_kernel_count"], 1)
            with (output / "npu_gpu_op_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                clip = next(
                    row
                    for row in csv.DictReader(handle)
                    if row["OP Type"] == "ClipByValueV2"
                )
            self.assertEqual(clip["Mapped NPU Kernel Count"], "2")
            self.assertEqual(clip["Mapping Coverage(%)"], "100.0")

    def test_flash_attention_bundle_maps_primary_and_auxiliary_kernels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            output = root / "out"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::empty", "cpu_op", 1, 1, 1, 1),
                            complete(
                                "FlashAttentionFunc",
                                "cpu_op",
                                10,
                                20,
                                1,
                                1,
                                {"External id": 10},
                            ),
                            complete(
                                "fwd_kernel",
                                "kernel",
                                40,
                                8,
                                2,
                                7,
                                {"External id": 10},
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("aten::empty", "cpu_op", 1, 1, 3, 3),
                            complete(
                                "gen_fa_mask_kernel",
                                "kernel",
                                30,
                                4,
                                4,
                                7,
                                {"Task ID": 1, "Stream ID": 7},
                            ),
                            complete(
                                "_attn_fwd",
                                "kernel",
                                40,
                                6,
                                4,
                                7,
                                {"Task ID": 2, "Stream ID": 7},
                            ),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,gen_fa_mask_kernel,gen_fa_mask_kernel,AI_VECTOR_CORE,30,4\n"
                "2,7,_attn_fwd,_attn_fwd,AI_CORE,40,6\n",
                encoding="utf-8",
            )

            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            self.assertEqual(
                summary["comparison"]["npu_first_duplicate_gpu_assignment_count"],
                0,
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = {row["OP Type"]: row for row in csv.DictReader(handle)}
            primary = mappings["_attn_fwd"]
            auxiliary = mappings["gen_fa_mask_kernel"]
            self.assertEqual(primary["GPU OP Name"], "FlashAttentionFunc")
            self.assertEqual(primary["GPU Kernel Names"], '["fwd_kernel"]')
            self.assertEqual(primary["Mapping Mode"], "fused_primary_gpu_bundle")
            self.assertEqual(primary["CPU Match Confidence"], "high")
            self.assertEqual(
                primary["CPU Alignment Basis"],
                "explicit_fused_type_equal_count_order",
            )
            self.assertEqual(auxiliary["Mapping Mode"], "fused_bundle_auxiliary")
            self.assertEqual(auxiliary["Mapped"], "False")
            with (output / "npu_gpu_op_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames[:4],
                    [
                        "Device_id",
                        "FUSED_OP_NPU_TYPE",
                        "FUSED_OP_GPU_TYPE",
                        "OP Type",
                    ],
                )
                statistic_rows = list(reader)
            statistics = {row["OP Type"]: row for row in statistic_rows}
            self.assertEqual(statistics["_attn_fwd"]["Mapped NPU Kernel Count"], "1")
            self.assertEqual(statistics["_attn_fwd"]["Mapping Coverage(%)"], "100.0")
            self.assertEqual(
                statistics["gen_fa_mask_kernel"]["Mapped NPU Kernel Count"],
                "0",
            )
            self.assertEqual(
                statistics["gen_fa_mask_kernel"]["Comparable Coverage(%)"], "0.0"
            )
            self.assertEqual(
                statistics["gen_fa_mask_kernel"]["Accounted Coverage(%)"], "100.0"
            )
            for op_type in ("_attn_fwd", "gen_fa_mask_kernel"):
                self.assertEqual(
                    statistics[op_type]["FUSED_OP_NPU_TYPE"],
                    "_attn_fwd + gen_fa_mask_kernel",
                )
                self.assertEqual(
                    statistics[op_type]["FUSED_OP_GPU_TYPE"],
                    "FlashAttentionFunc",
                )
            self.assertEqual(
                [row["OP Type"] for row in statistic_rows if row["Row Kind"] == "npu_type"],
                ["_attn_fwd", "gen_fa_mask_kernel"],
            )
            statistic_xlsx = output / "npu_gpu_op_statistic.xlsx"
            self.assertTrue(statistic_xlsx.is_file())
            with zipfile.ZipFile(statistic_xlsx) as workbook:
                workbook_xml = ElementTree.fromstring(workbook.read("xl/workbook.xml"))
                worksheets = [
                    ElementTree.fromstring(workbook.read(f"xl/worksheets/sheet{index}.xml"))
                    for index in (1, 3, 5)
                ]
            namespace = {
                "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
            }
            self.assertEqual(
                [sheet.attrib["name"] for sheet in workbook_xml.findall(".//main:sheet", namespace)],
                ["类型对比", "整体对比", "覆盖率诊断", "调用明细", "耗时分布"],
            )
            for worksheet in worksheets:
                merged_ranges = {
                    cell.attrib["ref"]
                    for cell in worksheet.findall(".//main:mergeCell", namespace)
                }
                self.assertEqual(merged_ranges, {"A2:A3", "B2:B3"})
                self.assertEqual(
                    len(worksheet.findall(".//main:sheetData/main:row", namespace)),
                    len(statistic_rows) + 1,
                )
            def headers(worksheet):
                return [
                    cell.findtext("main:is/main:t", namespaces=namespace)
                    for cell in worksheet.findall(".//main:sheetData/main:row[@r='1']/main:c", namespace)
                ]
            self.assertEqual(len(headers(worksheets[0])), 18)
            self.assertIn("排查结论", headers(worksheets[0]))
            self.assertIn("排查建议", headers(worksheets[1]))
            self.assertIn("GPU Activity Kind", headers(worksheets[0]))
            self.assertIn("Comparable Ratio(GPU/NPU)", headers(worksheets[0]))
            self.assertIn("Total Time Ratio(GPU/NPU)", headers(worksheets[0]))
            self.assertNotIn("Mapping Mode Counts", headers(worksheets[0]))
            self.assertIn("Mapping Mode Counts", headers(worksheets[1]))
            self.assertIn("GPU Max Time(us)", headers(worksheets[2]))
            comparable_coverage = worksheets[0].find(
                ".//main:sheetData/main:row[@r='2']/main:c[@r='H2']/main:v",
                namespace,
            )
            self.assertIsNotNone(comparable_coverage)
            self.assertAlmostEqual(
                float(comparable_coverage.text),
                float(statistic_rows[0]["Comparable Time Coverage(%)"]) / 100,
            )

            ordinary = build_npu_gpu_op_statistic_rows(
                [
                    {
                        "Device_id": "0",
                        "OP Type": "MatMul",
                        "Core Type": "AI_CORE",
                        "NPU Duration(us)": 3.0,
                        "Mapped": True,
                        "GPU Kernel Durations(us)": "[4.0]",
                        "Mapping Mode": "one_to_one",
                        "CPU Match ID": "1",
                    }
                ]
            )[0]
            self.assertEqual(ordinary["FUSED_OP_NPU_TYPE"], "MatMul")
            self.assertEqual(ordinary["FUSED_OP_GPU_TYPE"], "MatMul")
            with (output / "fused_op_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                fused = next(csv.DictReader(handle))
            self.assertEqual(fused["Fused OP"], "FlashAttentionFunc")
            self.assertEqual(fused["NPU Kernel Types"], '["gen_fa_mask_kernel","_attn_fwd"]')
            self.assertEqual(fused["GPU Kernel Sum(us)"], "8.0")
            self.assertEqual(fused["NPU Kernel Sum(us)"], "10.0")
            self.assertEqual(fused["Performance Ratio(GPU/NPU Bundle)"], "0.8")
            self.assertEqual(summary["comparison"]["fused_op_occurrence_count"], 1)

    def test_same_module_does_not_promote_scalar_tensor_fill_mismatch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace = root / "gpu.json"
            npu_trace = root / "npu.json"
            details = root / "details.csv"
            output = root / "out"
            gpu_trace.write_text(
                json.dumps(
                    {
                        "traceEvents": [
                            complete("nn.Module: Block", "user_annotation", 0, 100, 1, 1),
                            complete(
                                "aten::fill_",
                                "cpu_op",
                                10,
                                5,
                                1,
                                1,
                                {
                                    "External id": 1,
                                    "Input Dims": [[61], []],
                                    "Input type": ["float", "scalar"],
                                },
                            ),
                            complete("gpu_fill", "kernel", 20, 2, 2, 7, {"External id": 1}),
                            complete(
                                "aten::fill_",
                                "cpu_op",
                                30,
                                5,
                                1,
                                1,
                                {"External id": 3},
                            ),
                            complete("gpu_fill", "kernel", 40, 2, 2, 7, {"External id": 3}),
                            complete("aten::relu", "cpu_op", 80, 2, 1, 1),
                        ]
                    }
                ),
                encoding="utf-8",
            )
            npu_events = [complete("nn.Module: Block", "user_annotation", 0, 100, 3, 3)]
            npu_events.extend(
                complete("aten::view", "cpu_op", 1 + index, 1, 3, 3)
                for index in range(5)
            )
            npu_events.extend(
                [
                    complete(
                        "aten::fill_",
                        "cpu_op",
                        50,
                        5,
                        3,
                        3,
                        {
                            "External id": 2,
                            "Input Shapes": ";",
                            "Input Data Types": "BOOL;SCALAR",
                        },
                    ),
                    complete(
                        "npu_fill",
                        "kernel",
                        60,
                        3,
                        4,
                        7,
                        {"External id": 2, "Task ID": 1, "Stream ID": 7},
                    ),
                ]
            )
            npu_trace.write_text(json.dumps({"traceEvents": npu_events}), encoding="utf-8")
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,aclnnInplaceFillScalar_FillAiCore_Fill,Fill,AI_VECTOR_CORE,60,3\n",
                encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(
                    gpu_trace=str(gpu_trace),
                    npu_trace=str(npu_trace),
                    npu_kernel_details=str(details),
                    output_dir=str(output),
                )
            )
            self.assertEqual(summary["comparison"]["performance_matched_op_count"], 0)
            with (output / "matches.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                fills = [row for row in csv.DictReader(handle)
                         if row["Canonical OP"] == "aten::fill_"]
            self.assertEqual(fills, [])


    def test_gpu_parent_uses_unclaimed_child_kernels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            cases = [
                ("relu", "clamp_min", "Relu"),
                ("embedding", "index_select", "GatherV2"),
                ("matmul", "mm", "MatMulV3"),
                ("stack", "cat", "ConcatD"),
                ("type_as", "copy_", "Cast"),
            ]
            gpu_events, npu_events, detail_rows = [], [], []
            for index, (parent, child, npu_type) in enumerate(cases, start=1):
                start = index * 100
                gpu_events.extend(
                    [
                        complete(
                            f"aten::{parent}", "cpu_op", start, 30, 1, 1,
                            {"Input Dims": [[index, 4]]},
                        ),
                        complete(
                            f"aten::{child}", "cpu_op", start + 2, 8, 1, 1,
                            {"External id": index},
                        ),
                        complete(
                            f"gpu_{child}", "kernel", start + 20, 4, 2, 7,
                            {"External id": index},
                        ),
                    ]
                )
                npu_events.extend(
                    [
                        complete(
                            f"aten::{parent}", "cpu_op", start, 30, 3, 3,
                            {"External id": 100 + index, "Input Dims": [[index, 4]]},
                        ),
                        complete(
                            f"npu_{parent}", "kernel", start + 20, 6, 4, 7,
                            {
                                "External id": 100 + index,
                                "Task ID": index,
                                "Stream ID": 7,
                            },
                        ),
                    ]
                )
                detail_rows.append(
                    f"{index},7,npu_{parent},{npu_type},AI_CORE,{start + 20},6"
                )
            gpu_trace.write_text(json.dumps({"traceEvents": gpu_events}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": npu_events}), encoding="utf-8")
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                + "\n".join(detail_rows) + "\n",
                encoding="utf-8",
            )

            summary = compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = {row["OP Type"]: row for row in csv.DictReader(handle)}
            for _, child, npu_type in cases:
                row = mappings[npu_type]
                self.assertEqual(row["Mapping Mode"], "gpu_child_one_to_one")
                self.assertEqual(row["GPU Kernel Source"], "descendant_op")
                self.assertEqual(
                    row["GPU Kernel Owner OP Names"], f'["aten::{child}"]'
                )
                self.assertEqual(row["Direct Kernel Pair"], "True")
            with (output / "npu_gpu_op_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                statistics = list(csv.DictReader(handle))
            type_statistics = [row for row in statistics if row["Row Kind"] == "npu_type"]
            self.assertEqual(len(type_statistics), len(cases))
            for row in type_statistics:
                self.assertEqual(row["Direct Mapping Coverage(%)"], "100.0")
                self.assertEqual(row["Accounted Coverage(%)"], "100.0")
                self.assertEqual(row["Comparable Coverage(%)"], "100.0")
            self.assertEqual(summary["comparison"]["npu_first_direct_kernel_count"], 5)
            self.assertEqual(
                summary["comparison"]["npu_first_duplicate_gpu_assignment_count"], 0
            )

    def test_gpu_child_kernel_is_not_reassigned_to_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_trace.write_text(
                json.dumps({"traceEvents": [
                    complete(
                        "aten::relu", "cpu_op", 10, 30, 1, 1,
                        {"Input Dims": [[4]]},
                    ),
                    complete(
                        "aten::clamp_min", "cpu_op", 12, 8, 1, 1,
                        {"External id": 1, "Input Dims": [[4]]},
                    ),
                    complete("gpu_clamp", "kernel", 30, 4, 2, 7, {"External id": 1}),
                ]}),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps({"traceEvents": [
                    complete(
                        "aten::relu", "cpu_op", 10, 30, 3, 3,
                        {"External id": 11, "Input Dims": [[4]]},
                    ),
                    complete(
                        "aten::clamp_min", "cpu_op", 12, 8, 3, 3,
                        {"External id": 12, "Input Dims": [[4]]},
                    ),
                    complete(
                        "npu_relu", "kernel", 30, 6, 4, 7,
                        {"External id": 11, "Task ID": 1, "Stream ID": 7},
                    ),
                    complete(
                        "npu_clamp", "kernel", 40, 5, 4, 7,
                        {"External id": 12, "Task ID": 2, "Stream ID": 7},
                    ),
                ]}),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,npu_relu,Relu,AI_CORE,30,6\n"
                "2,7,npu_clamp,ClampMin,AI_CORE,40,5\n",
                encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = {row["OP Type"]: row for row in csv.DictReader(handle)}
            self.assertEqual(mappings["ClampMin"]["Mapped"], "True")
            self.assertEqual(mappings["Relu"]["Mapped"], "False")
            self.assertEqual(
                summary["comparison"]["npu_first_duplicate_gpu_assignment_count"], 0
            )

    def test_gpu_child_fallback_requires_npu_parent_type(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_trace.write_text(
                json.dumps({"traceEvents": [
                    complete(
                        "aten::stack", "cpu_op", 10, 30, 1, 1,
                        {"Input Dims": [[4]]},
                    ),
                    complete(
                        "aten::cat", "cpu_op", 12, 8, 1, 1,
                        {"External id": 1},
                    ),
                    complete("gpu_cat", "kernel", 30, 4, 2, 7, {"External id": 1}),
                ]}),
                encoding="utf-8",
            )
            npu_trace.write_text(
                json.dumps({"traceEvents": [
                    complete(
                        "aten::stack", "cpu_op", 10, 30, 3, 3,
                        {"External id": 11, "Input Dims": [[4]]},
                    ),
                    complete(
                        "npu_reduce", "kernel", 30, 6, 4, 7,
                        {"External id": 11, "Task ID": 1, "Stream ID": 7},
                    ),
                ]}),
                encoding="utf-8",
            )
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,npu_reduce,ReduceSum,AI_CORE,30,6\n",
                encoding="utf-8",
            )
            compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mapping = next(csv.DictReader(handle))
            self.assertEqual(mapping["Mapping Mode"], "gpu_op_without_kernel")
            self.assertEqual(mapping["Comparable"], "False")
            self.assertEqual(mapping["Accounted"], "False")

    def test_same_type_cat_kernels_compare_as_one_shared_bundle(self):
        for gpu_durations in ((2, 3), (2, 3, 4)):
            with self.subTest(gpu_durations=gpu_durations):
                self._assert_cat_shared_bundle(gpu_durations)

    def test_stack_shared_bundle_uses_child_cat_without_duplicate_gpu_time(self):
        for gpu_durations in ((2,), (2, 3)):
            with self.subTest(gpu_durations=gpu_durations):
                self._assert_cat_shared_bundle(gpu_durations, parent="stack")

    def _assert_cat_shared_bundle(self, gpu_durations, parent="cat"):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_trace.write_text(json.dumps({"traceEvents": [
                *([complete("aten::stack", "cpu_op", 8, 60, 1, 1,
                            {"Input Dims": [[4, 8]]})] if parent == "stack" else []),
                complete("aten::cat", "cpu_op", 10, 20, 1, 1,
                         {"External id": 1, "Input Dims": [[4, 8]]}),
                *[complete(f"gpu_cat_{i}", "kernel", 30 + i * 10, duration, 2, 7,
                           {"External id": 1})
                  for i, duration in enumerate(gpu_durations)],
            ]}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": [
                complete(f"aten::{parent}", "cpu_op", 10, 20, 3, 3,
                         {"External id": 11, "Input Dims": [[4, 8]]}),
                complete("npu_cat_1", "kernel", 35, 5, 4, 7,
                         {"External id": 11, "Task ID": 1, "Stream ID": 7}),
                complete("npu_cat_2", "kernel", 45, 7, 4, 7,
                         {"External id": 11, "Task ID": 2, "Stream ID": 7}),
            ]}), encoding="utf-8")
            api = "aclnnStack" if parent == "stack" else "aclnnCat"
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                f"1,7,{api}_ConcatD_ConcatD,ConcatD,AI_CORE,35,5\n"
                f"2,7,{api}_ConcatD_ConcatD,ConcatD,AI_CORE,45,7\n",
                encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = list(csv.DictReader(handle))
            self.assertEqual(len(mappings), 2)
            self.assertEqual({row["Mapping Mode"] for row in mappings},
                             {"gpu_child_shared_bundle" if parent == "stack"
                              else "npu_gpu_bundle_shared"})
            self.assertEqual({row["GPU Kernel Source"] for row in mappings},
                             {"descendant_op" if parent == "stack" else "direct_op"})
            self.assertTrue(all(row["Mapped"] == "True" for row in mappings))
            self.assertTrue(all(row["Direct Kernel Pair"] == "False" for row in mappings))
            self.assertTrue(all(row["Performance Ratio(GPU/NPU)"] == "" for row in mappings))
            with (output / "npu_gpu_op_statistic.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                statistic = next(csv.DictReader(handle))
            self.assertEqual(int(statistic["GPU Kernel Count"]), len(gpu_durations))
            self.assertEqual(float(statistic["GPU Total Time(us)"]), sum(gpu_durations))
            self.assertEqual(statistic["Comparable NPU Time(us)"], "12.0")
            self.assertEqual(float(statistic["Comparable Ratio(GPU/NPU)"]),
                             round(sum(gpu_durations) / 12, 6))
            self.assertEqual(summary["comparison"]["npu_first_duplicate_gpu_assignment_count"], 0)
            self.assertEqual(summary["comparison"]["npu_first_shared_gpu_bundle_assignment_count"],
                             len(gpu_durations))

    def test_stack_api_repairs_wrong_flow_owner_and_uses_gpu_child_cat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_events, npu_events, detail_rows = [], [], []
            for index, start in enumerate((10, 110), start=1):
                gpu_events.extend([
                    complete("aten::stack", "cpu_op", start, 30, 1, 1,
                             {"Input Dims": [[index, 4]]}),
                    complete("aten::cat", "cpu_op", start + 2, 8, 1, 1,
                             {"External id": index}),
                    complete(f"gpu_cat_{index}", "kernel", start + 20, 4, 2, 7,
                             {"External id": index}),
                ])
                npu_events.extend([
                    complete("aten::stack", "cpu_op", start, 30, 3, 3,
                             {"Input Dims": [[index, 4]]}),
                    complete("aten::sum", "cpu_op", start + 2, 8, 3, 3),
                    {"name": f"flow{index}", "ph": "s", "ts": start + 3,
                     "pid": 3, "tid": 3, "id": index, "args": {}},
                    {"name": f"flow{index}", "ph": "f", "ts": start + 20,
                     "pid": 4, "tid": 7, "id": index, "args": {}},
                    complete(f"npu_stack_{index}", "kernel", start + 20, 6,
                             4, 7, {"Task ID": index, "Stream ID": 7}),
                ])
                detail_rows.append(
                    f"{index},7,aclnnStack_ConcatD_ConcatD,ConcatD,AI_CORE,{start + 20},6"
                )
            gpu_trace.write_text(json.dumps({"traceEvents": gpu_events}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": npu_events}), encoding="utf-8")
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                + "\n".join(detail_rows) + "\n", encoding="utf-8",
            )
            summary = compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = list(csv.DictReader(handle))
            self.assertEqual(len(mappings), 2)
            self.assertEqual({row["NPU Canonical OP"] for row in mappings}, {"aten::stack"})
            self.assertEqual({row["Mapping Mode"] for row in mappings},
                             {"gpu_child_one_to_one"})
            self.assertEqual(summary["npu"]["kernel_owner_reconciliation"]["api_groups"]
                             ["aten::stack"]["moved_kernel_count"], 2)

    def test_acl_api_semantic_conflicts_do_not_claim_gpu_time(self):
        cases = [
            ("div", "aclnnSubs_SubAiCore_Sub", "Sub"),
            ("mean", "aclnnPowTensorScalar_SquareAiCore_Square", "Square"),
            ("pow", "aclnnInplaceCopy_CastAiCore_Cast", "Cast"),
            ("cumsum", "aclnnReduceSum_ReduceSumOpAiCore_ReduceSum", "ReduceSum"),
            ("cumsum", "aclnnRepeat_TileAiCore_Tile", "Tile"),
            ("sum", "aclnnStack_ConcatD_ConcatD", "ConcatD"),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_events, npu_events, detail_rows = [], [], []
            for index, (op, kernel_name, kernel_type) in enumerate(cases, start=1):
                start = index * 100
                gpu_events.extend([
                    complete(f"aten::{op}", "cpu_op", start, 20, 1, 1,
                             {"External id": index}),
                    complete(f"gpu_{op}", "kernel", start + 30, 4, 2, 7,
                             {"External id": index}),
                ])
                npu_events.extend([
                    complete(f"aten::{op}", "cpu_op", start, 20, 3, 3,
                             {"External id": index + 100}),
                    complete(f"npu_{op}", "kernel", start + 30, 5, 4, 7,
                             {"External id": index + 100, "Task ID": index,
                              "Stream ID": 7}),
                ])
                detail_rows.append(
                    f"{index},7,{kernel_name},{kernel_type},AI_CORE,{start + 30},5"
                )
            gpu_trace.write_text(json.dumps({"traceEvents": gpu_events}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": npu_events}), encoding="utf-8")
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                + "\n".join(detail_rows) + "\n", encoding="utf-8",
            )
            compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = list(csv.DictReader(handle))
            self.assertEqual(len(mappings), len(cases))
            self.assertEqual({row["Mapping Mode"] for row in mappings},
                             {"api_semantic_conflict_unmapped"})
            self.assertTrue(all(row["Mapped"] == "False" for row in mappings))
            self.assertTrue(all(row["API Semantic Conflict"] == "True" for row in mappings))

    def test_cast_helper_does_not_take_primary_gpu_kernel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_trace.write_text(json.dumps({"traceEvents": [
                complete("aten::div", "cpu_op", 10, 20, 1, 1,
                         {"External id": 1, "Input Dims": [[4, 8]]}),
                complete("gpu_div", "kernel", 30, 4, 2, 7,
                         {"External id": 1}),
            ]}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": [
                complete("aten::div", "cpu_op", 10, 20, 3, 3,
                         {"External id": 11, "Input Dims": [[4, 8]]}),
                complete("npu_cast", "kernel", 30, 2, 4, 7,
                         {"External id": 11, "Task ID": 1, "Stream ID": 7}),
                complete("npu_div", "kernel", 35, 5, 4, 7,
                         {"External id": 11, "Task ID": 2, "Stream ID": 7}),
            ]}), encoding="utf-8")
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,aclnnDivs_CastAiCore_Cast,Cast,AI_CORE,30,2\n"
                "2,7,aclnnDivs_RealDivAiCore_RealDiv,RealDiv,AI_CORE,35,5\n",
                encoding="utf-8",
            )
            compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = {row["OP Type"]: row for row in csv.DictReader(handle)}
            self.assertEqual(mappings["Cast"]["Mapping Mode"], "api_helper_unmapped")
            self.assertEqual(mappings["Cast"]["Accounted"], "True")
            self.assertEqual(mappings["RealDiv"]["Mapping Mode"], "one_to_one")
            self.assertEqual(mappings["RealDiv"]["Mapped"], "True")

    def test_cat_slice_helper_does_not_map_to_gpu_cat(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gpu_trace, npu_trace = root / "gpu.json", root / "npu.json"
            details, output = root / "details.csv", root / "out"
            gpu_trace.write_text(json.dumps({"traceEvents": [
                complete("aten::cat", "cpu_op", 10, 20, 1, 1,
                         {"External id": 1, "Input Dims": [[4, 8]]}),
                complete("gpu_cat", "kernel", 30, 4, 2, 7,
                         {"External id": 1}),
            ]}), encoding="utf-8")
            npu_trace.write_text(json.dumps({"traceEvents": [
                complete("aten::cat", "cpu_op", 10, 20, 3, 3,
                         {"External id": 11, "Input Dims": [[4, 8]]}),
                complete("npu_slice", "kernel", 30, 2, 4, 7,
                         {"External id": 11, "Task ID": 1, "Stream ID": 7}),
                complete("npu_cat", "kernel", 35, 5, 4, 7,
                         {"External id": 11, "Task ID": 2, "Stream ID": 7}),
            ]}), encoding="utf-8")
            details.write_text(
                "Task ID,Stream ID,Name,Type,Accelerator Core,Start Time(us),Duration(us)\n"
                "1,7,aclnnCat_SliceAiCore_Slice,Slice,AI_CORE,30,2\n"
                "2,7,aclnnCat_ConcatD_ConcatD,ConcatD,AI_CORE,35,5\n",
                encoding="utf-8",
            )
            compare_profiles(
                CompareConfig(str(gpu_trace), str(npu_trace), str(details), str(output))
            )
            with (output / "npu_kernel_gpu_mapping.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                mappings = {row["OP Type"]: row for row in csv.DictReader(handle)}
            self.assertEqual(mappings["Slice"]["Mapping Mode"], "api_helper_unmapped")
            self.assertEqual(mappings["ConcatD"]["Mapping Mode"], "one_to_one")


if __name__ == "__main__":
    unittest.main()

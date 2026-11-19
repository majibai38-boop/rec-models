import csv
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from profiling_tools.compare import CompareConfig, compare_profiles
from profiling_tools.result_api import read_csv, validate_results
from profiling_tools.tests.test_compare import complete


def fixture(root, *, repetitions=1, anchors=True, repeated_gemm=False, conflict=False):
    gpu = [complete("ProfilerStep#10", "user_annotation", 0, 1000, 1, 1)]
    npu = [complete("ProfilerStep#10", "user_annotation", 0, 1000, 3, 3)]
    details = []
    forward = [[2, 5, 7], [2, 7, 3]]
    grad_a = [[2, 5, 3], [2, 3, 7]]
    grad_b = grad_a if repeated_gemm else [[2, 7, 5], [2, 5, 3]]

    def cpu(name, ts, duration, pid, tid, external, operands=(), **args):
        return complete(name, "cpu_op", ts, duration, pid, tid, {
            "Input Dims": operands, "Input type": ["Float"] * len(operands),
            "External id": external, **args,
        })

    def npu_launch(name, kind, core, ts, duration, owner, operands=(), output=()):
        task = len(details) + 1
        npu.append(complete(name, "kernel", ts, duration, 4, 7, {
            "External id": owner, "Task ID": task, "Stream ID": 7,
        }))
        formatted = lambda dims: '"' + ";".join(",".join(map(str, shape)) for shape in dims) + '"'
        details.append({
            "Task ID": task, "Stream ID": 7, "Name": name, "Type": kind, "Accelerator Core": core,
            "Start Time(us)": ts, "Duration(us)": duration, "Device_id": 0, "Step Id": 10,
            "Input Shapes": formatted(operands) if operands else "",
            "Input Data Types": ";".join(["FLOAT"] * len(operands)),
            "Output Shapes": formatted(output) if output else "",
        })

    for i in range(repetitions):
        t, ext = 10 + i * 100, 100 + i * 20
        gpu_seq, npu_seq = 1000 + i, 2000 + i
        call_gpu = {"Sequence number": gpu_seq} if anchors else {}
        call_npu = {"Sequence number": npu_seq} if anchors else {}
        gpu_backward = {"Sequence number": 1000 + repetitions - 1 - i} if anchors else {}
        gpu.extend([
            cpu("aten::matmul", t, 20, 1, 1, ext, forward),
            cpu("aten::bmm", t+1, 5, 1, 1, ext+1, forward, **call_gpu),
            complete("gpu_fwd", "kernel", t+20, 4, 2, 7, {"External id": ext+1}),
            cpu("autograd::engine::evaluate_function: BmmBackward0", t+30, 25, 1, 2,
                ext+2, **gpu_backward),
            cpu("BmmBackward0", t+31, 22, 1, 2, ext+3, **gpu_backward),
            cpu("aten::bmm", t+33, 2, 1, 2, ext+4, grad_a),
            cpu("aten::bmm", t+36, 2, 1, 2, ext+5, grad_b),
            cpu("aten::clone", t+39, 2, 1, 2, ext+6, [[2, 5, 3]]),
            complete("gpu_grad_a", "kernel", t+60, 5, 2, 7, {"External id": ext+4}),
            complete("gpu_grad_b", "kernel", t+66, 6, 2, 7, {"External id": ext+5}),
            complete("gpu_layout", "kernel", t+73, 2, 2, 7, {"External id": ext+6}),
        ])
        npu.extend([
            cpu("aten::matmul", t, 20, 3, 3, ext, forward, **call_npu),
            cpu("aten::matmul_backward", t+30, 25, 3, 3, ext+2, **call_npu),
        ])
        npu_launch("aclnnMatmul_fwd", "BatchMatMulV3", "AI_CORE", t+20, 6, ext, forward)
        # Device launch order and storage layout differ from the GPU.
        grad_b_npu = [[2, 7, 5], [2, 5, 4]] if conflict else grad_b
        npu_launch("aclnnMatmul_grad_b", "BatchMatMulV3", "MIX_AIC", t+60, 8, ext+2, grad_b_npu)
        npu_launch("aclnnMatmul_layout", "TransData", "AI_CORE", t+69, 3, ext+2)
        npu_launch("aclnnMatmul_grad_a", "BatchMatMulV3", "AI_CORE", t+74, 7, ext+2,
                   [[2, 3, 5], grad_a[1]], [[2, 5, 7]])

    gpu_path, npu_path, detail_path = root/"gpu.json", root/"npu.json", root/"kernel_details.csv"
    gpu_path.write_text(json.dumps({"traceEvents": gpu}), encoding="utf-8")
    npu_path.write_text(json.dumps({"traceEvents": npu}), encoding="utf-8")
    with detail_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=details[0])
        writer.writeheader()
        writer.writerows(details)
    return CompareConfig(str(gpu_path), str(npu_path), str(detail_path), str(root/"out"),
                         gpu_step="10", npu_step="10", gpu_pid="1", npu_pid="3")


class MatmulBackwardTest(unittest.TestCase):
    def test_scope_alignment_preserves_forward_type_accounting_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(root, repetitions=2)
            summary = compare_profiles(config, cache_dir=root/"cache")
            output = Path(config.output_dir)
            self.assertEqual(validate_results(output)["status"], "valid")
            calls = read_csv(output/"whole_call_mapping.csv")
            backward = [r for r in calls if r["Source"] == "matmul_backward_scope"]
            self.assertEqual(len(backward), 2)
            self.assertEqual({r["Alignment Basis"] for r in backward}, {"matmul_backward_forward_anchor"})
            self.assertEqual({r["GPU Scope Name"] for r in backward}, {"BmmBackward0"})
            self.assertGreater(int(backward[0]["GPU Scope Trace Event Index"]), int(backward[1]["GPU Scope Trace Event Index"]))
            self.assertEqual({(float(r["GPU Total Time(us)"]), float(r["NPU Total Time(us)"])) for r in backward},
                             {(13, 18)})
            mappings = read_csv(output/"npu_kernel_gpu_mapping.csv")
            self.assertEqual({r["Mapping Mode"] for r in mappings if r["NPU Kernel Name"] == "aclnnMatmul_fwd"},
                             {"gpu_child_one_to_one"})
            self.assertEqual({r["GPU Kernel Names"] for r in mappings if r["NPU Kernel Name"] == "aclnnMatmul_grad_a"},
                             {'["gpu_grad_a"]'})
            self.assertEqual({r["GPU Kernel Names"] for r in mappings if r["NPU Kernel Name"] == "aclnnMatmul_grad_b"},
                             {'["gpu_grad_b"]'})
            remainder = read_csv(output/"gpu_unmapped_activity.csv")
            self.assertEqual({(r["GPU Name"],r["Status"]) for r in remainder}, {("gpu_layout", "whole_call_only")})
            self.assertEqual(summary["gpu_accounting"]["kernel"]["whole_call_only_count"], 2)
            self.assertEqual(summary["gpu_accounting"]["kernel"]["mapped_time_us"], 30)
            self.assertEqual(summary["gpu_accounting"]["kernel"]["total_time_us"], 34)
            unmatched = read_csv(output/"unmatched_gpu.csv")
            bmm = [r for r in unmatched if r["OP Name"] == "aten::bmm"]
            self.assertEqual(len(bmm), 6)
            self.assertTrue(all(r["GPU Type Assigned Kernel Count"] == "1" for r in bmm))
            second = replace(config, output_dir=str(root/"cached"))
            cached = compare_profiles(second, cache_dir=root/"cache")
            self.assertTrue(cached["analysis_cache"]["hit"])
            self.assertEqual(read_csv(Path(second.output_dir)/"whole_call_mapping.csv"), calls)

    def test_indistinguishable_gradients_remain_whole_call_only(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(Path(directory), anchors=False, repeated_gemm=True)
            summary = compare_profiles(config)
            calls = read_csv(Path(config.output_dir)/"whole_call_mapping.csv")
            self.assertEqual([r["Alignment Basis"] for r in calls if r["Source"] == "matmul_backward_scope"],
                             ["matmul_backward_unique_workload"])
            self.assertEqual(summary["gpu_accounting"]["kernel"]["mapped_count"], 1)
            self.assertEqual(summary["gpu_accounting"]["kernel"]["whole_call_only_count"], 3)

    def test_ambiguous_calls_and_conflicting_dimensions_do_not_recover(self):
        for options in ({"anchors": False, "repetitions": 2}, {"conflict": True}):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                config = fixture(Path(directory), **options)
                summary = compare_profiles(config)
                calls = read_csv(Path(config.output_dir)/"whole_call_mapping.csv")
                self.assertFalse(any(r["Source"] == "matmul_backward_scope" for r in calls))
                self.assertEqual(summary["gpu_accounting"]["kernel"]["whole_call_only_count"], 0)

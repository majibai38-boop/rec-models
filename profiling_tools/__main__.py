from __future__ import annotations

import argparse
import json
import sys

from .compare import CompareConfig, compare_profiles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m profiling_tools",
        description="Align PyTorch GPU/NPU aten operators and compare their device-kernel bundles.",
    )
    parser.add_argument("--gpu-trace", required=True, help="GPU PyTorch profiler trace JSON")
    parser.add_argument("--npu-trace", required=True, help="NPU trace_view.json")
    parser.add_argument(
        "--npu-kernel-details", required=True, help="NPU kernel_details.csv"
    )
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON reports")
    parser.add_argument(
        "--gpu-step", default="", help="GPU ProfilerStep number/name; default compares all steps"
    )
    parser.add_argument(
        "--npu-step", default="", help="NPU ProfilerStep number/name; default compares all steps"
    )
    parser.add_argument("--gpu-pid", default="", help="GPU host PID; auto-detected by default")
    parser.add_argument("--npu-pid", default="", help="NPU host PID; auto-detected by default")
    parser.add_argument(
        "--op-prefix", default="aten::", help="Framework operator prefix (default: aten::)"
    )
    parser.add_argument(
        "--op-map",
        default="",
        help="Optional JSON object mapping GPU operator names to NPU operator names",
    )
    parser.add_argument(
        "--include-communication",
        action="store_true",
        help="Include NCCL/HCCL/communication-category device events",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = CompareConfig(
        gpu_trace=args.gpu_trace,
        npu_trace=args.npu_trace,
        npu_kernel_details=args.npu_kernel_details,
        output_dir=args.output_dir,
        gpu_step=args.gpu_step,
        npu_step=args.npu_step,
        gpu_pid=args.gpu_pid,
        npu_pid=args.npu_pid,
        op_prefix=args.op_prefix,
        op_map=args.op_map,
        include_communication=args.include_communication,
    )
    try:
        summary = compare_profiles(config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"profiling_tools: error: {error}", file=sys.stderr)
        return 2

    comparison = summary["comparison"]
    print(f"Reports written to: {args.output_dir}")
    print(
        "Matched ops: {matched}; unmatched GPU/NPU: {gpu}/{npu}".format(
            matched=comparison["matched_op_count"],
            gpu=comparison["unmatched_gpu_op_count"],
            npu=comparison["unmatched_npu_op_count"],
        )
    )
    print(
        "Matched kernel-duration coverage GPU/NPU: {gpu:.2%}/{npu:.2%}".format(
            gpu=comparison["gpu_matched_kernel_duration_coverage"],
            npu=comparison["npu_matched_kernel_duration_coverage"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

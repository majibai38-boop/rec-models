from __future__ import annotations

import argparse
import json
import sys

from .compare import CompareConfig, InputSelectionError, compare_profiles
from .inspection import inspect_inputs
from .result_api import summarize_results, query_results, validate_results
from .version import __version__


def build_parser(*, inspection=False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m profiling_tools",
        description="Align PyTorch GPU/NPU aten operators and compare their device-kernel bundles.",
    )
    parser.add_argument("--gpu-trace", required=True, help="GPU PyTorch profiler trace JSON")
    parser.add_argument("--npu-trace", required=True, help="NPU trace_view.json")
    parser.add_argument(
        "--npu-kernel-details", required=True, help="NPU kernel_details.csv"
    )
    parser.add_argument("--output-dir", required=not inspection, default="", help="New or empty directory for reports")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--json", action="store_true", help="Print a machine-readable result")
    parser.add_argument("--all-steps", action="store_true", help="Explicitly compare all steps, including unmarked work")
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
    parser.add_argument(
        "--flow-tolerance-us",
        type=float,
        default=1000.0,
        help="Maximum timestamp distance for flow-to-kernel fallback (default: 1000 us)",
    )
    parser.add_argument(
        "--diagnostic-coverage-threshold", type=float, default=90.0,
        help="Diagnose every NPU type with Comparable count OR time coverage below this percent (0-100; default: 90)",
    )
    return parser


def _print_json(value):
    # JSON is a UTF-8 API even when a Windows shell/pipe uses a legacy codepage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def _result_command(command, argv):
    parser = argparse.ArgumentParser(prog=f"python -m profiling_tools {command}")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--json", action="store_true", help="JSON is already the default for read-only commands")
    if command == "query":
        parser.add_argument("--op-type")
        parser.add_argument("--npu-kernel-index", type=int)
        parser.add_argument("--gpu-kernel-index", type=int)
        parser.add_argument("--memory-activity-index", type=int)
        parser.add_argument("--cpu-op-index", type=int)
        parser.add_argument("--side", choices=("gpu", "npu"), default="npu")
        parser.add_argument("--device-id")
        parser.add_argument("--core-type")
        parser.add_argument("--offset", type=int, default=0)
        parser.add_argument("--limit", type=int, default=20)
    args = vars(parser.parse_args(argv))
    output = args.pop("result_dir")
    args.pop("json")
    try:
        result = (query_results(output, **args) if command == "query" else
                  validate_results(output) if command == "validate" else summarize_results(output))
        _print_json(result)
        return 4 if command == "validate" and result["status"] == "invalid" else 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        _print_json({"schema_version": 1, "status": "error", "message": str(error)})
        return 2


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv.pop(0) if argv and argv[0] in {"inspect", "compare", "summary", "query", "validate"} else "legacy"
    if command in {"summary", "query", "validate"}:
        return _result_command(command, argv)
    args = build_parser(inspection=command == "inspect").parse_args(argv)
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
        flow_tolerance_us=args.flow_tolerance_us,
        diagnostic_coverage_threshold=args.diagnostic_coverage_threshold,
        all_steps=args.all_steps,
        require_explicit_scope=command in {"compare", "inspect"},
    )
    try:
        if command == "inspect":
            result = inspect_inputs(config)
            _print_json(result)
            return {"ready": 0, "needs_selection": 3, "invalid": 2}[result["status"]]
        summary = compare_profiles(config)
    except InputSelectionError as error:
        _print_json(error.report)
        return 3
    except (OSError, ValueError, json.JSONDecodeError) as error:
        if args.json:
            _print_json({"schema_version": 1, "status": "error", "message": str(error)})
        else:
            print(f"profiling_tools: error: {error}", file=sys.stderr)
        return 2

    if args.json:
        _print_json(summarize_results(args.output_dir))
        return 4 if summary["status"] == "invalid" else 0

    comparison = summary["comparison"]
    print(f"Reports written to: {args.output_dir}")
    print(f"Result status: {summary['status']}; validation: {summary['validation']['status']}")
    print(
        "Matched ops: {matched}; unmatched GPU/NPU: {gpu}/{npu}".format(
            matched=comparison["matched_op_count"],
            gpu=comparison["unmatched_gpu_op_count"],
            npu=comparison["unmatched_npu_op_count"],
        )
    )
    print(
        "Trusted/performance matches: {trusted}/{performance}; performance regions: {regions}".format(
            trusted=comparison["trusted_matched_op_count"],
            performance=comparison["performance_matched_op_count"],
            regions=comparison["performance_region_count"],
        )
    )
    print(
        "Matched kernel-duration coverage GPU/NPU: {gpu:.2%}/{npu:.2%}".format(
            gpu=comparison["gpu_matched_kernel_duration_coverage"],
            npu=comparison["npu_matched_kernel_duration_coverage"],
        )
    )
    precision = summary["precision_analysis"]
    diagnostic = summary["coverage_diagnostics"]
    print(
        "Low-coverage diagnostics: {types} types, {kernels} kernels, {time:.3f} us; see coverage_diagnostics.md".format(
            types=diagnostic["selected_type_count"], kernels=diagnostic["analyzed_kernel_count"],
            time=diagnostic["uncomparable_time_us"],
        )
    )
    print(
        "Precision paths: {calls}; comparable: {comparable}; observed Cast time: {cast:.3f} us".format(
            calls=precision["call_count"], comparable=precision["comparable_call_count"],
            cast=precision["cast_time_us"],
        )
    )
    return 4 if summary["status"] == "invalid" else 0


if __name__ == "__main__":
    raise SystemExit(main())

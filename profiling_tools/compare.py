from __future__ import annotations

import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .matcher import load_op_map, match_ops
from .parser import attach_kernel_details, parse_trace
from .report import write_reports
from .coverage_diagnostics import validate_threshold
from .inspection import inspect_inputs
from .provenance import begin_run, assert_inputs_unchanged, write_json
from .result_api import file_identity, validate_results


@dataclass(frozen=True)
class CompareConfig:
    gpu_trace: str
    npu_trace: str
    npu_kernel_details: str
    output_dir: str
    gpu_step: str = ""
    npu_step: str = ""
    gpu_pid: str = ""
    npu_pid: str = ""
    op_prefix: str = "aten::"
    op_map: str = ""
    include_communication: bool = False
    flow_tolerance_us: float = 1000.0
    diagnostic_coverage_threshold: float = 90.0
    require_explicit_scope: bool = False
    all_steps: bool = False


class InputSelectionError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__("Input scope is ambiguous; run inspect and select PID/step or --all-steps.")


def compare_profiles(config: CompareConfig, *, progress=None) -> Dict[str, Any]:
    # Optional service adapter hook; no web dependencies in the analysis engine.
    def stage(name):
        if progress is not None:
            progress(name)

    validate_threshold(config.diagnostic_coverage_threshold)
    if not math.isfinite(config.flow_tolerance_us) or config.flow_tolerance_us < 0:
        raise ValueError("flow tolerance must be finite and non-negative")
    output = Path(config.output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output directory must be new or empty; use a new run directory to preserve previous results")
    started_at = time.perf_counter()
    stage("inspection")
    manifest = begin_run(config)
    inspection, loaded, details = inspect_inputs(config, retain_data=True)
    if inspection["errors"]:
        raise ValueError("; ".join(f"{e['side']}: {e['message']}" for e in inspection["errors"]))
    if config.require_explicit_scope and inspection["selections_required"]:
        raise InputSelectionError(inspection)
    inspected_at = time.perf_counter()
    stage("gpu_parse")
    gpu = parse_trace(
        config.gpu_trace,
        "gpu",
        op_prefix=config.op_prefix,
        requested_pid=config.gpu_pid,
        requested_step=config.gpu_step,
        include_communication=config.include_communication,
        flow_tolerance_us=config.flow_tolerance_us,
        trace_data=loaded.pop("gpu"),
    )
    gpu_parsed_at = time.perf_counter()
    stage("npu_parse")
    npu = parse_trace(
        config.npu_trace,
        "npu",
        op_prefix=config.op_prefix,
        requested_pid=config.npu_pid,
        requested_step=config.npu_step,
        include_communication=config.include_communication,
        flow_tolerance_us=config.flow_tolerance_us,
        trace_data=loaded.pop("npu"),
    )
    npu_parsed_at = time.perf_counter()
    stage("detail_link")
    attach_kernel_details(npu, details)
    details_attached_at = time.perf_counter()
    stage("matching")
    op_map = load_op_map(config.op_map)
    matches, unmatched_gpu, unmatched_npu = match_ops(gpu, npu, op_map)
    matched_at = time.perf_counter()
    assert_inputs_unchanged(manifest)
    stage("reports")
    summary = write_reports(
        config.output_dir,
        gpu,
        npu,
        matches,
        unmatched_gpu,
        unmatched_npu,
        op_map,
        diagnostic_coverage_threshold=config.diagnostic_coverage_threshold,
    )
    reports_written_at = time.perf_counter()
    summary["timing_seconds"] = {
        "input_inspection": round(inspected_at - started_at, 6),
        "gpu_parse": round(gpu_parsed_at - inspected_at, 6),
        "npu_parse": round(npu_parsed_at - gpu_parsed_at, 6),
        "kernel_detail_attach": round(details_attached_at - npu_parsed_at, 6),
        "operator_match": round(matched_at - details_attached_at, 6),
        "report_write": round(reports_written_at - matched_at, 6),
        "total": round(reports_written_at - started_at, 6),
    }
    summary.update({key: manifest[key] for key in ("schema_version", "run_id", "created_at", "producer")})
    summary["warnings"] = inspection["warnings"] + inspection["selections_required"]
    summary["status"] = "pending_validation"
    write_json(output / "input_inspection.json", inspection)
    manifest["artifacts"] = {name: {k: v for k, v in file_identity(output / name).items() if k in {"sha256", "size_bytes"}}
                             for name in [*summary["reports"], "input_inspection.json"]}
    write_json(output / "run_manifest.json", manifest)
    summary["reports"] = sorted([*summary["reports"], "input_inspection.json", "run_manifest.json", "validation.json"])
    write_json(output / "summary.json", summary)
    stage("validation")
    validation = validate_results(output)
    summary["validation"] = validation
    summary["warnings"].extend(validation["warnings"])
    has_gaps = (summary["comparison"]["npu_first_comparable_kernel_count"] < summary["comparison"]["npu_first_kernel_count"]
                or any(summary["gpu_accounting"][kind]["unmapped_count"] for kind in ("kernel", "memory")))
    summary["status"] = "invalid" if validation["status"] == "invalid" else "complete_with_gaps" if has_gaps else "complete"
    summary["timing_seconds"]["validation_and_manifest"] = round(time.perf_counter() - reports_written_at, 6)
    summary["timing_seconds"]["total"] = round(time.perf_counter() - started_at, 6)
    write_json(output / "validation.json", validation)
    write_json(output / "summary.json", summary)
    return summary

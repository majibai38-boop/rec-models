from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .model import BundleMetrics, KernelRecord, OpMatch, OpRecord, ParsedProfile, bundle_metrics


def _ratio(numerator: float, denominator: float) -> Any:
    return round(numerator / denominator, 6) if denominator else ""


def _round(value: float) -> float:
    return round(value, 6)


def _json_list(values: Iterable[Any]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _op_kernels(profile: ParsedProfile, op_index: int) -> List[KernelRecord]:
    return [profile.kernels[index] for index in profile.ops[op_index].kernel_indices]


def _type_signature(kernels: Sequence[KernelRecord]) -> str:
    counts = Counter(
        kernel.detail.op_type
        for kernel in kernels
        if kernel.detail is not None and kernel.detail.op_type
    )
    if not counts:
        return "UNRESOLVED"
    if len(counts) == 1:
        return next(iter(counts))
    return "FUSED[" + ",".join(f"{name}x{count}" for name, count in sorted(counts.items())) + "]"


def _core_signature(kernels: Sequence[KernelRecord]) -> str:
    values = sorted(
        {
            kernel.detail.core_type
            for kernel in kernels
            if kernel.detail is not None and kernel.detail.core_type
        }
    )
    return "+".join(values) if values else "UNRESOLVED"


def _detail_shapes(kernels: Sequence[KernelRecord]) -> List[str]:
    return sorted(
        {
            kernel.detail.input_shapes
            for kernel in kernels
            if kernel.detail is not None and kernel.detail.input_shapes
        }
    )


def _metric_fields(prefix: str, metrics: BundleMetrics) -> Dict[str, Any]:
    return {
        f"{prefix} Kernel Count": metrics.kernel_count,
        f"{prefix} Kernel Sum(us)": _round(metrics.sum_us),
        f"{prefix} Device Busy(us)": _round(metrics.busy_us),
        f"{prefix} Bundle Span(us)": _round(metrics.span_us),
    }


def build_match_rows(
    gpu: ParsedProfile, npu: ParsedProfile, matches: Sequence[OpMatch]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in matches:
        gpu_op = gpu.ops[match.gpu_op_index]
        npu_op = npu.ops[match.npu_op_index]
        gpu_kernels = _op_kernels(gpu, match.gpu_op_index)
        npu_kernels = _op_kernels(npu, match.npu_op_index)
        gpu_metrics = bundle_metrics(gpu_kernels)
        npu_metrics = bundle_metrics(npu_kernels)
        row: Dict[str, Any] = {
            "Match ID": match.match_id,
            "Confidence": match.confidence,
            "Match Score": _round(match.score),
            "GPU OP Name": gpu_op.event.name,
            "NPU OP Name": npu_op.event.name,
            "Canonical OP": npu_op.canonical_name,
            "GPU Input Shapes": gpu_op.input_shapes,
            "NPU Input Shapes": npu_op.input_shapes,
            "GPU Module Path": gpu_op.module_path,
            "NPU Module Path": npu_op.module_path,
            "NPU Kernel Type": _type_signature(npu_kernels),
            "NPU Core Type": _core_signature(npu_kernels),
            "NPU Kernel Input Shapes": _json_list(_detail_shapes(npu_kernels)),
            "GPU Kernel Names": _json_list(kernel.event.name for kernel in gpu_kernels),
            "NPU Kernel Names": _json_list(kernel.event.name for kernel in npu_kernels),
            "GPU Link Methods": _json_list(kernel.link_method for kernel in gpu_kernels),
            "NPU Link Methods": _json_list(kernel.link_method for kernel in npu_kernels),
            "Kernel Count Equal": gpu_metrics.kernel_count == npu_metrics.kernel_count,
            "NPU/GPU Sum Ratio": _ratio(npu_metrics.sum_us, gpu_metrics.sum_us),
            "NPU/GPU Busy Ratio": _ratio(npu_metrics.busy_us, gpu_metrics.busy_us),
            "NPU/GPU Span Ratio": _ratio(npu_metrics.span_us, gpu_metrics.span_us),
            "Sum Diff(us)": _round(npu_metrics.sum_us - gpu_metrics.sum_us),
            "Busy Diff(us)": _round(npu_metrics.busy_us - gpu_metrics.busy_us),
            "Span Diff(us)": _round(npu_metrics.span_us - gpu_metrics.span_us),
        }
        row.update(_metric_fields("GPU", gpu_metrics))
        row.update(_metric_fields("NPU", npu_metrics))
        rows.append(row)
    return rows


def build_gpu_kernel_mapping_rows(
    gpu: ParsedProfile, npu: ParsedProfile, matches: Sequence[OpMatch]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in matches:
        gpu_op = gpu.ops[match.gpu_op_index]
        npu_op = npu.ops[match.npu_op_index]
        gpu_kernels = sorted(
            _op_kernels(gpu, match.gpu_op_index), key=lambda kernel: kernel.event.timestamp_us
        )
        npu_kernels = sorted(
            _op_kernels(npu, match.npu_op_index), key=lambda kernel: kernel.event.timestamp_us
        )
        type_signature = _type_signature(npu_kernels)
        core_signature = _core_signature(npu_kernels)
        if len(gpu_kernels) == len(npu_kernels) == 1:
            mode = "one_to_one"
        elif gpu_kernels and len(gpu_kernels) == len(npu_kernels):
            mode = "ordered_equal_count"
        else:
            mode = "bundle"

        for position, gpu_kernel in enumerate(gpu_kernels):
            paired_npu = npu_kernels[position] if mode != "bundle" else None
            detail = paired_npu.detail if paired_npu is not None else None
            rows.append(
                {
                    "Match ID": match.match_id,
                    "Confidence": match.confidence,
                    "Mapping Mode": mode,
                    "Canonical OP": npu_op.canonical_name,
                    "GPU OP Name": gpu_op.event.name,
                    "NPU OP Name": npu_op.event.name,
                    "GPU Kernel Position": position,
                    "GPU Kernel Name": gpu_kernel.event.name,
                    "GPU Kernel Duration(us)": _round(gpu_kernel.event.duration_us),
                    "GPU Stream ID": gpu_kernel.stream_id,
                    "NPU Kernel Name": paired_npu.event.name if paired_npu else "",
                    "Mapped NPU Type": detail.op_type if detail and detail.op_type else type_signature,
                    "Mapped NPU Core Type": detail.core_type if detail and detail.core_type else core_signature,
                    "Mapped Input Shapes": detail.input_shapes if detail else _json_list(_detail_shapes(npu_kernels)),
                    "Mapped Input Data Types": detail.input_dtypes if detail else "",
                    "NPU Bundle Kernel Count": len(npu_kernels),
                }
            )
    return rows


def _duration_values(profile: ParsedProfile, op_indices: Sequence[int]) -> List[float]:
    return [bundle_metrics(_op_kernels(profile, index)).sum_us for index in op_indices]


def build_aten_statistic_rows(
    gpu: ParsedProfile,
    npu: ParsedProfile,
    matches: Sequence[OpMatch],
    op_map: Mapping[str, str],
) -> List[Dict[str, Any]]:
    gpu_groups: Dict[str, List[int]] = defaultdict(list)
    npu_groups: Dict[str, List[int]] = defaultdict(list)
    matched_counts: Counter[str] = Counter()
    for index, op in enumerate(gpu.ops):
        gpu_groups[op_map.get(op.canonical_name, op.canonical_name)].append(index)
    for index, op in enumerate(npu.ops):
        npu_groups[op.canonical_name].append(index)
    for match in matches:
        matched_counts[npu.ops[match.npu_op_index].canonical_name] += 1

    rows: List[Dict[str, Any]] = []
    for op_type in sorted(set(gpu_groups) | set(npu_groups)):
        gpu_indices = gpu_groups.get(op_type, [])
        npu_indices = npu_groups.get(op_type, [])
        gpu_values = _duration_values(gpu, gpu_indices)
        npu_values = _duration_values(npu, npu_indices)
        gpu_total = sum(gpu_values)
        npu_total = sum(npu_values)
        rows.append(
            {
                "OP Type": op_type,
                "GPU Calls": len(gpu_indices),
                "NPU Calls": len(npu_indices),
                "Matched Calls": matched_counts[op_type],
                "GPU Raw Kernel Count": sum(len(gpu.ops[index].kernel_indices) for index in gpu_indices),
                "NPU Raw Kernel Count": sum(len(npu.ops[index].kernel_indices) for index in npu_indices),
                "GPU Total Time(us)": _round(gpu_total),
                "NPU Total Time(us)": _round(npu_total),
                "GPU Avg Time(us)": _round(mean(gpu_values)) if gpu_values else 0.0,
                "NPU Avg Time(us)": _round(mean(npu_values)) if npu_values else 0.0,
                "GPU Max Time(us)": _round(max(gpu_values)) if gpu_values else 0.0,
                "NPU Max Time(us)": _round(max(npu_values)) if npu_values else 0.0,
                "GPU Min Time(us)": _round(min(gpu_values)) if gpu_values else 0.0,
                "NPU Min Time(us)": _round(min(npu_values)) if npu_values else 0.0,
                "NPU/GPU Ratio": _ratio(npu_total, gpu_total),
                "Diff Time(us)": _round(npu_total - gpu_total),
            }
        )
    return rows


def build_kernel_type_statistic_rows(
    gpu: ParsedProfile, npu: ParsedProfile, matches: Sequence[OpMatch]
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[OpMatch]] = defaultdict(list)
    for match in matches:
        npu_kernels = _op_kernels(npu, match.npu_op_index)
        groups[(_type_signature(npu_kernels), _core_signature(npu_kernels))].append(match)

    aggregates: List[Tuple[str, str, List[OpMatch], List[BundleMetrics], List[BundleMetrics]]] = []
    for (op_type, core_type), group_matches in sorted(groups.items()):
        gpu_metrics = [
            bundle_metrics(_op_kernels(gpu, match.gpu_op_index)) for match in group_matches
        ]
        npu_metrics = [
            bundle_metrics(_op_kernels(npu, match.npu_op_index)) for match in group_matches
        ]
        aggregates.append((op_type, core_type, group_matches, gpu_metrics, npu_metrics))

    all_gpu_time = sum(metric.sum_us for _, _, _, metrics, _ in aggregates for metric in metrics)
    all_npu_time = sum(metric.sum_us for _, _, _, _, metrics in aggregates for metric in metrics)
    rows: List[Dict[str, Any]] = []
    for op_type, core_type, group_matches, gpu_metrics, npu_metrics in aggregates:
        gpu_total = sum(metric.sum_us for metric in gpu_metrics)
        npu_total = sum(metric.sum_us for metric in npu_metrics)
        rows.append(
            {
                "Mapped OP Type": op_type,
                "Core Type": core_type,
                "Count": len(group_matches),
                "Logical Calls": len(group_matches),
                "GPU Raw Kernel Count": sum(metric.kernel_count for metric in gpu_metrics),
                "NPU Raw Kernel Count": sum(metric.kernel_count for metric in npu_metrics),
                "GPU Total Time(us)": _round(gpu_total),
                "NPU Total Time(us)": _round(npu_total),
                "GPU Avg Time(us)": _round(gpu_total / len(group_matches)),
                "NPU Avg Time(us)": _round(npu_total / len(group_matches)),
                "GPU Max Time(us)": _round(max(metric.sum_us for metric in gpu_metrics)),
                "NPU Max Time(us)": _round(max(metric.sum_us for metric in npu_metrics)),
                "GPU Min Time(us)": _round(min(metric.sum_us for metric in gpu_metrics)),
                "NPU Min Time(us)": _round(min(metric.sum_us for metric in npu_metrics)),
                "GPU Ratio(%)": _round(gpu_total * 100.0 / all_gpu_time) if all_gpu_time else 0.0,
                "NPU Ratio(%)": _round(npu_total * 100.0 / all_npu_time) if all_npu_time else 0.0,
                "NPU/GPU Ratio": _ratio(npu_total, gpu_total),
                "Diff Time(us)": _round(npu_total - gpu_total),
            }
        )
    return rows


def build_module_statistic_rows(
    gpu: ParsedProfile, npu: ParsedProfile
) -> List[Dict[str, Any]]:
    def aggregate(profile: ParsedProfile) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"calls": 0.0, "kernels": 0.0, "sum": 0.0, "busy": 0.0, "span": 0.0}
        )
        for index, op in enumerate(profile.ops):
            key = op.module_path or "[NO MODULE]"
            metrics = bundle_metrics(_op_kernels(profile, index))
            result[key]["calls"] += 1
            result[key]["kernels"] += metrics.kernel_count
            result[key]["sum"] += metrics.sum_us
            result[key]["busy"] += metrics.busy_us
            result[key]["span"] += metrics.span_us
        return result

    gpu_values = aggregate(gpu)
    npu_values = aggregate(npu)
    rows: List[Dict[str, Any]] = []
    for module in sorted(set(gpu_values) | set(npu_values)):
        gpu_row = gpu_values.get(module, {})
        npu_row = npu_values.get(module, {})
        gpu_total = float(gpu_row.get("sum", 0.0))
        npu_total = float(npu_row.get("sum", 0.0))
        rows.append(
            {
                "Module Path": module,
                "GPU OP Calls": int(gpu_row.get("calls", 0)),
                "NPU OP Calls": int(npu_row.get("calls", 0)),
                "GPU Kernel Count": int(gpu_row.get("kernels", 0)),
                "NPU Kernel Count": int(npu_row.get("kernels", 0)),
                "GPU Kernel Sum(us)": _round(gpu_total),
                "NPU Kernel Sum(us)": _round(npu_total),
                "GPU Device Busy Sum(us)": _round(float(gpu_row.get("busy", 0.0))),
                "NPU Device Busy Sum(us)": _round(float(npu_row.get("busy", 0.0))),
                "GPU Bundle Span Sum(us)": _round(float(gpu_row.get("span", 0.0))),
                "NPU Bundle Span Sum(us)": _round(float(npu_row.get("span", 0.0))),
                "NPU/GPU Ratio": _ratio(npu_total, gpu_total),
                "Diff Time(us)": _round(npu_total - gpu_total),
            }
        )
    return rows


def build_unmatched_rows(profile: ParsedProfile, indices: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in indices:
        op = profile.ops[index]
        kernels = _op_kernels(profile, index)
        metrics = bundle_metrics(kernels)
        rows.append(
            {
                "OP Index": index,
                "OP Name": op.event.name,
                "Canonical OP": op.canonical_name,
                "Input Shapes": op.input_shapes,
                "Module Path": op.module_path,
                "Kernel Count": metrics.kernel_count,
                "Kernel Sum(us)": _round(metrics.sum_us),
                "Device Busy(us)": _round(metrics.busy_us),
                "Bundle Span(us)": _round(metrics.span_us),
                "Kernel Names": _json_list(kernel.event.name for kernel in kernels),
            }
        )
    return rows


def _profile_summary(profile: ParsedProfile) -> Dict[str, Any]:
    linked = [kernel for kernel in profile.kernels if kernel.owner_op_index is not None]
    total_duration = sum(kernel.event.duration_us for kernel in profile.kernels)
    linked_duration = sum(kernel.event.duration_us for kernel in linked)
    link_methods = Counter(kernel.link_method for kernel in profile.kernels)
    return {
        "source": profile.source_path,
        "selected_pid": profile.selected_pid,
        "selected_step": profile.selected_step,
        "available_steps": profile.step_names,
        "trace_event_count": profile.total_trace_events,
        "aten_op_count": len(profile.ops),
        "kernel_count": len(profile.kernels),
        "linked_kernel_count": len(linked),
        "linked_kernel_duration_coverage": linked_duration / total_duration if total_duration else 0.0,
        "kernel_link_methods": dict(link_methods),
        "device_process_ids": profile.device_process_ids,
        "task_type_counts": profile.task_type_counts,
        "kernel_detail_count": profile.kernel_detail_count,
        "kernel_detail_matched_count": profile.kernel_detail_matched_count,
        "kernel_detail_synthetic_count": profile.kernel_detail_synthetic_count,
    }


def write_reports(
    output_dir: str,
    gpu: ParsedProfile,
    npu: ParsedProfile,
    matches: Sequence[OpMatch],
    unmatched_gpu: Sequence[int],
    unmatched_npu: Sequence[int],
    op_map: Mapping[str, str],
) -> Dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    match_rows = build_match_rows(gpu, npu, matches)
    mapping_rows = build_gpu_kernel_mapping_rows(gpu, npu, matches)
    aten_rows = build_aten_statistic_rows(gpu, npu, matches, op_map)
    type_rows = build_kernel_type_statistic_rows(gpu, npu, matches)
    module_rows = build_module_statistic_rows(gpu, npu)
    unmatched_gpu_rows = build_unmatched_rows(gpu, unmatched_gpu)
    unmatched_npu_rows = build_unmatched_rows(npu, unmatched_npu)

    reports = {
        "matches.csv": match_rows,
        "gpu_kernel_mapping.csv": mapping_rows,
        "aten_op_statistic.csv": aten_rows,
        "kernel_type_statistic.csv": type_rows,
        "module_statistic.csv": module_rows,
        "unmatched_gpu.csv": unmatched_gpu_rows,
        "unmatched_npu.csv": unmatched_npu_rows,
    }
    for filename, rows in reports.items():
        fieldnames = list(rows[0].keys()) if rows else {
            "matches.csv": ["Match ID", "Canonical OP"],
            "gpu_kernel_mapping.csv": ["Match ID", "GPU Kernel Name", "Mapped NPU Type"],
            "aten_op_statistic.csv": ["OP Type"],
            "kernel_type_statistic.csv": ["Mapped OP Type", "Core Type"],
            "module_statistic.csv": ["Module Path"],
            "unmatched_gpu.csv": ["OP Index", "OP Name"],
            "unmatched_npu.csv": ["OP Index", "OP Name"],
        }[filename]
        _write_csv(output / filename, rows, fieldnames)

    matched_gpu_duration = sum(
        bundle_metrics(_op_kernels(gpu, match.gpu_op_index)).sum_us for match in matches
    )
    matched_npu_duration = sum(
        bundle_metrics(_op_kernels(npu, match.npu_op_index)).sum_us for match in matches
    )
    gpu_total_duration = sum(kernel.event.duration_us for kernel in gpu.kernels)
    npu_total_duration = sum(kernel.event.duration_us for kernel in npu.kernels)
    confidence_counts = Counter(match.confidence for match in matches)
    summary = {
        "gpu": _profile_summary(gpu),
        "npu": _profile_summary(npu),
        "comparison": {
            "matched_op_count": len(matches),
            "unmatched_gpu_op_count": len(unmatched_gpu),
            "unmatched_npu_op_count": len(unmatched_npu),
            "confidence_counts": dict(confidence_counts),
            "gpu_matched_kernel_duration_coverage": (
                matched_gpu_duration / gpu_total_duration if gpu_total_duration else 0.0
            ),
            "npu_matched_kernel_duration_coverage": (
                matched_npu_duration / npu_total_duration if npu_total_duration else 0.0
            ),
        },
        "reports": sorted(reports),
        "time_semantics": {
            "kernel_sum_us": "sum of kernel durations; concurrent streams may be double-counted",
            "device_busy_us": "union of kernel intervals; concurrent execution is counted once",
            "bundle_span_us": "last kernel end minus first kernel start; includes gaps",
        },
    }
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary

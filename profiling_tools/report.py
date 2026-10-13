from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .model import BundleMetrics, KernelRecord, OpMatch, OpRecord, ParsedProfile, bundle_metrics
from .call_metadata import compare_metadata, compare_match_metadata
from .parser import kernel_api_expected_ops, kernel_api_is_auxiliary, CUSTOM_COMPARABLE_OPS
from .xlsx_writer import TableSheet, write_grouped_workbook_xlsx
from .device_compare import (
    enrich_memory_mappings, memory_diagnostic_rows, scoped_indices,
    build_cpu_bundle_rows, summarize_cpu_bundles,
)
from .precision_analysis import build_precision_rows
from .whole_calls import build_whole_calls, summarize_whole_calls, annotate_type_calls
from .comparison_accounting import add_coverage_conclusions, complete_gpu_accounting
from .coverage_diagnostics import (
    build_coverage_diagnostics, write_coverage_diagnostics, validate_threshold,
)


_SHARED_GPU_BUNDLE_MODES = {"npu_gpu_bundle_shared", "gpu_child_shared_bundle", "logical_scope_shared_bundle", "manual_scope_shared_bundle"}


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


def _type_key(value: str) -> str:
    if "::" in value:
        value = value.split("::", 1)[1]
    if "." in value:
        value = value.split(".", 1)[0]
    return re.sub(r"[^a-z0-9]+", "", value.lower())


_OP_PRIMARY_TYPE_ALIASES = {
    "aten::addmm": {"matmulv3"},
    "aten::matmul": {"matmulv3"},
    "aten::mm": {"matmulv3"},
    "aten::bmm": {"batchmatmulv3"},
    "aten::clamp": {"clipbyvaluev2"},
    "aten::div": {"realdiv"},
    "aten::embedding": {"gatherv2"},
    "aten::index_select": {"gatherv2"},
    "aten::_convolution": {"conv2dv2"},
    "aten::convolution": {"conv2dv2"},
    "aten::gelu": {"geluv2"},
    "ropefunc": {"ropekernelflattened"},
    "customop::triton_target_attention_nobitwise_varlen": {"targetattentionnobitwisevarlenforwardkernel"},
    "customop::triton_layernorm_forward": {"fasterlayernormtinyforwardkernel"},
    "aten::native_layer_norm": {"layernormv4"},
    "aten::layer_norm": {"layernormv4"},
    "aten::ne": {"notequal"},
    "aten::bitwise_not": {"logicalnot"},
    "aten::where": {"selectv2"},
    "aten::silu": {"swish"},
    "aten::cat": {"concatd"},
    "aten::concat": {"concatd"},
    "aten::sum": {"reducesum"},
    "aten::mean": {"reducemean"},
    "aten::var": {"reducevar"},
    "aten::max": {"reducemax"},
    "aten::min": {"reducemin"},
    "aten::mul": {"muls"},
}


# A parent op may launch GPU work through a child aten op. For this fallback,
# require the NPU kernel Type to represent the parent operation rather than a
# helper kernel accidentally attached to the parent by the NPU trace.
_GPU_CHILD_PARENT_TYPE_ALIASES = {
    "aten::matmul": {"batchmatmulv3"},
    "aten::stack": {"concatd", "pack"},
    "aten::to": {"cast"},
    "aten::type_as": {"cast"},
    "aten::pad": {"padv3"},
    "aten::constant_pad_nd": {"padv3"},
    "aten::arange": {"range"},
    "aten::repeat": {"tile"},
    "aten::zero_": {"zeroslike"},
    "aten::ones": {"oneslike"},
    "aten::rsub": {"sub"},
    "aten::count_nonzero": {"reducesum"},
    "nestedconcatfunc": {"nestedconcatfwdkernel"},
}


# Explicit cross-backend fused-op correspondences for kernels that have no
# usable NPU flow/CPU owner. Recovery is intentionally allowed only when the
# GPU custom-op, NPU primary-kernel and every auxiliary-kernel count are equal.
_FUSED_OP_SPECS = (
    {
        "name": "FlashAttentionFunc",
        "gpu_op": "flashattentionfunc",
        "npu_primary_type": "attnfwd",
        "npu_auxiliary_types": ("genfamaskkernel",),
        "auxiliary_before_primary": True,
    },
)


def _primary_type_keys(op: OpRecord) -> set[str]:
    return {
        _type_key(op.canonical_name),
        *_OP_PRIMARY_TYPE_ALIASES.get(op.canonical_name, set()),
    }


def _is_semantic_primary_kernel(op: OpRecord, kernel: KernelRecord) -> bool:
    return (
        kernel.detail is not None
        and bool(kernel.detail.op_type)
        and _type_key(kernel.detail.op_type) in _primary_type_keys(op)
    )


def _is_gpu_child_parent_kernel(op: OpRecord, kernel: KernelRecord) -> bool:
    return (
        kernel.detail is not None
        and bool(kernel.detail.op_type)
        and _type_key(kernel.detail.op_type)
        in (
            _primary_type_keys(op)
            | _GPU_CHILD_PARENT_TYPE_ALIASES.get(op.canonical_name, set())
        )
    )


def _semantic_primary_kernels(
    op: OpRecord, kernels: Sequence[KernelRecord]
) -> List[KernelRecord]:
    return [kernel for kernel in kernels if _is_semantic_primary_kernel(op, kernel)]


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


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _is_trusted(match: OpMatch) -> bool:
    return match.confidence in {"medium", "high"}


def _is_performance_match(
    gpu: ParsedProfile, npu: ParsedProfile, match: OpMatch
) -> bool:
    return (
        _is_trusted(match)
        and bool(gpu.ops[match.gpu_op_index].kernel_indices)
        and bool(npu.ops[match.npu_op_index].kernel_indices)
    )


def _kernels_for_ops(profile: ParsedProfile, op_indices: Sequence[int]) -> List[KernelRecord]:
    kernel_indices = {
        kernel_index
        for op_index in op_indices
        for kernel_index in profile.ops[op_index].kernel_indices
    }
    return [profile.kernels[index] for index in sorted(kernel_indices)]


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
            "GPU OP Index": match.gpu_op_index,
            "NPU OP Index": match.npu_op_index,
            "Confidence": match.confidence,
            "Match Score": _round(match.score),
            "Alignment Basis": match.alignment_basis,
            "Eligible For Performance": _is_performance_match(gpu, npu, match),
            "GPU OP Name": gpu_op.event.name,
            "NPU OP Name": npu_op.event.name,
            "Canonical OP": npu_op.canonical_name,
            "GPU Input Shapes": gpu_op.input_shapes,
            "NPU Input Shapes": npu_op.input_shapes,
            "GPU Raw Input Shapes": gpu_op.raw_input_shapes,
            "NPU Raw Input Shapes": npu_op.raw_input_shapes,
            "GPU Input DTypes": gpu_op.input_dtypes,
            "NPU Input DTypes": npu_op.input_dtypes,
            "GPU Module Path": gpu_op.module_path,
            "NPU Module Path": npu_op.module_path,
            "GPU Module Key": gpu_op.module_key,
            "NPU Module Key": npu_op.module_key,
            "NPU Kernel Type": _type_signature(npu_kernels),
            "NPU Core Type": _core_signature(npu_kernels),
            "NPU Kernel Input Shapes": _json_list(_detail_shapes(npu_kernels)),
            "GPU Kernel Names": _json_list(kernel.event.name for kernel in gpu_kernels),
            "NPU Kernel Names": _json_list(kernel.event.name for kernel in npu_kernels),
            "GPU Link Methods": _json_list(kernel.link_method for kernel in gpu_kernels),
            "NPU Link Methods": _json_list(kernel.link_method for kernel in npu_kernels),
            "GPU Link Scopes": _json_list(kernel.link_scope for kernel in gpu_kernels),
            "NPU Link Scopes": _json_list(kernel.link_scope for kernel in npu_kernels),
            "NPU Flow Distance(us)": _json_list(
                kernel.link_distance_us
                for kernel in npu_kernels
                if kernel.link_distance_us is not None
            ),
            "Both Kernel Empty": gpu_metrics.kernel_count == npu_metrics.kernel_count == 0,
            "Kernel Count Equal": (
                gpu_metrics.kernel_count > 0
                and npu_metrics.kernel_count > 0
                and gpu_metrics.kernel_count == npu_metrics.kernel_count
            ),
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
        primary_npu_kernels = _semantic_primary_kernels(npu_op, npu_kernels)
        if len(gpu_kernels) == len(npu_kernels) == 1:
            mode = "one_to_one"
        elif gpu_kernels and len(gpu_kernels) == len(npu_kernels):
            mode = "ordered_equal_count"
        elif gpu_kernels and len(gpu_kernels) == len(primary_npu_kernels):
            mode = "semantic_primary"
        else:
            mode = "bundle"

        for position, gpu_kernel in enumerate(gpu_kernels):
            if mode == "semantic_primary":
                paired_npu = primary_npu_kernels[position]
            elif mode != "bundle":
                paired_npu = npu_kernels[position]
            else:
                paired_npu = None
            detail = paired_npu.detail if paired_npu is not None else None
            rows.append(
                {
                    "Match ID": match.match_id,
                    "Confidence": match.confidence,
                    "Eligible For Performance": _is_performance_match(gpu, npu, match),
                    "Mapping Mode": mode,
                    "Mapping Fidelity": (
                        "semantic_primary_in_bundle"
                        if mode == "semantic_primary"
                        else "exact"
                        if mode != "bundle"
                        else "bundle_inferred"
                    ),
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
                    "NPU Bundle Type Signature": type_signature,
                    "NPU Bundle Kernel Names": _json_list(
                        kernel.event.name for kernel in npu_kernels
                    ),
                }
            )
    return rows


def _npu_kernel_type(kernel: KernelRecord) -> str:
    if kernel.detail is not None and kernel.detail.op_type:
        return kernel.detail.op_type
    task_type = str(kernel.event.args.get("Task Type", "") or "").strip()
    return task_type or "UNRESOLVED"


def _npu_kernel_core_type(kernel: KernelRecord) -> str:
    if kernel.detail is not None and kernel.detail.core_type:
        return kernel.detail.core_type
    return str(kernel.event.args.get("Task Type", "") or "").strip() or "UNRESOLVED"


def _explicit_fused_op_occurrences(
    gpu: ParsedProfile, npu: ParsedProfile
) -> List[Dict[str, Any]]:
    """Return strictly validated occurrence-order GPU/NPU fused bundles."""
    occurrences: List[Dict[str, Any]] = []
    for spec in _FUSED_OP_SPECS:
        gpu_op_bundles = [
            (
                gpu_op_index,
                sorted(
                    gpu_op.kernel_indices,
                    key=lambda index: gpu.kernels[index].event.timestamp_us,
                ),
            )
            for gpu_op_index, gpu_op in sorted(
                enumerate(gpu.ops),
                key=lambda item: item[1].event.timestamp_us,
            )
            if gpu_op.canonical_name == spec["gpu_op"] and gpu_op.kernel_indices
        ]
        npu_primary_indices = sorted(
            (
                index
                for index, kernel in enumerate(npu.kernels)
                if kernel.detail is not None
                and _type_key(kernel.detail.op_type) == spec["npu_primary_type"]
            ),
            key=lambda index: npu.kernels[index].event.timestamp_us,
        )
        npu_auxiliary_indices_by_type = {
            auxiliary_type: sorted(
                (
                    index
                    for index, kernel in enumerate(npu.kernels)
                    if kernel.detail is not None
                    and _type_key(kernel.detail.op_type) == auxiliary_type
                ),
                key=lambda index: npu.kernels[index].event.timestamp_us,
            )
            for auxiliary_type in spec["npu_auxiliary_types"]
        }
        count = len(gpu_op_bundles)
        if (
            not count
            or len(npu_primary_indices) != count
            or any(
                len(indices) != count
                for indices in npu_auxiliary_indices_by_type.values()
            )
        ):
            continue

        spec_occurrences: List[Dict[str, Any]] = []
        valid = True
        for occurrence_index in range(count):
            primary_index = npu_primary_indices[occurrence_index]
            auxiliary_indices = [
                npu_auxiliary_indices_by_type[auxiliary_type][occurrence_index]
                for auxiliary_type in spec["npu_auxiliary_types"]
            ]
            npu_kernel_indices = sorted(
                [*auxiliary_indices, primary_index],
                key=lambda index: npu.kernels[index].event.timestamp_us,
            )
            stream_ids = {
                npu.kernels[index].stream_id
                for index in npu_kernel_indices
                if npu.kernels[index].stream_id
            }
            primary_timestamp = npu.kernels[primary_index].event.timestamp_us
            if len(stream_ids) > 1 or (
                spec["auxiliary_before_primary"]
                and any(
                    npu.kernels[index].event.timestamp_us > primary_timestamp
                    for index in auxiliary_indices
                )
            ):
                valid = False
                break
            gpu_op_index, gpu_kernel_indices = gpu_op_bundles[occurrence_index]
            spec_occurrences.append(
                {
                    "fused_op": spec["name"],
                    "occurrence": occurrence_index + 1,
                    "gpu_op_index": gpu_op_index,
                    "gpu_kernel_indices": gpu_kernel_indices,
                    "npu_primary_kernel_index": primary_index,
                    "npu_auxiliary_kernel_indices": auxiliary_indices,
                    "npu_kernel_indices": npu_kernel_indices,
                }
            )
        if valid:
            occurrences.extend(spec_occurrences)
    return occurrences


def _descendant_gpu_kernel_indices(gpu: ParsedProfile) -> List[List[int]]:
    """Index kernels by every containing GPU op without changing their owners."""
    descendants: List[List[int]] = [[] for _ in gpu.ops]
    for op in gpu.ops:
        parent_index = op.parent_index
        while parent_index is not None:
            descendants[parent_index].extend(op.kernel_indices)
            parent_index = gpu.ops[parent_index].parent_index
    return [
        sorted(set(indices), key=lambda index: gpu.kernels[index].event.timestamp_us)
        for indices in descendants
    ]


def _op_depth(ops: Sequence[OpRecord], op_index: int) -> int:
    depth = 0
    parent_index = ops[op_index].parent_index
    while parent_index is not None:
        depth += 1
        parent_index = ops[parent_index].parent_index
    return depth


def build_npu_kernel_gpu_mapping_rows(
    gpu: ParsedProfile, npu: ParsedProfile, matches: Sequence[OpMatch]
) -> List[Dict[str, Any]]:
    """Map from each NPU kernel through its CPU op to zero or more GPU kernels."""
    match_by_npu_op = {match.npu_op_index: match for match in matches}
    assignments: Dict[int, Tuple[Optional[OpMatch], str, List[int]]] = {}
    direct_gpu_op_by_npu_kernel: Dict[int, int] = {}

    for npu_op_index, npu_op in enumerate(npu.ops):
        owner_kernel_indices = sorted(
            npu_op.kernel_indices,
            key=lambda index: npu.kernels[index].event.timestamp_us,
        )
        if not owner_kernel_indices:
            continue
        match = match_by_npu_op.get(npu_op_index)
        npu_kernel_indices = []
        for kernel_index in owner_kernel_indices:
            kernel = npu.kernels[kernel_index]
            expected_ops = kernel_api_expected_ops(kernel)
            if kernel_api_is_auxiliary(kernel):
                assignments[kernel_index] = (match, "api_helper_unmapped", [])
            elif expected_ops and npu_op.canonical_name not in expected_ops:
                assignments[kernel_index] = (match, "api_semantic_conflict_unmapped", [])
            else:
                npu_kernel_indices.append(kernel_index)
        if not npu_kernel_indices:
            continue
        if match is None:
            for kernel_index in npu_kernel_indices:
                assignments[kernel_index] = (None, "unmatched_cpu_op", [])
            continue
        if not _is_trusted(match):
            for kernel_index in npu_kernel_indices:
                assignments[kernel_index] = (match, "untrusted_cpu_match", [])
            continue

        gpu_kernel_indices = sorted(
            gpu.ops[match.gpu_op_index].kernel_indices,
            key=lambda index: gpu.kernels[index].event.timestamp_us,
        )
        if npu_op.canonical_name in CUSTOM_COMPARABLE_OPS and (
            compare_metadata(npu_op, gpu.ops[match.gpu_op_index]) != ([], [])
            or any(npu.kernels[i].link_method not in {"external_id", "runtime_correlation", "torch_to_npu",
                       "host_to_device_queue", "host_to_device_thread"} for i in npu_kernel_indices)
            or any(gpu.kernels[i].link_method not in {"external_id", "runtime_correlation"} for i in gpu_kernel_indices)
            or len({(gpu.kernels[i].event.pid, str(gpu.kernels[i].event.args.get("device", "")))
                    for i in gpu_kernel_indices}) > 1
        ):
            for index in npu_kernel_indices:
                assignments[index] = (match, "untrusted_cpu_match", [])
            continue
        if not gpu_kernel_indices:
            for kernel_index in npu_kernel_indices:
                assignments[kernel_index] = (match, "gpu_op_without_kernel", [])
        elif (
            npu_op.canonical_name in {"aten::cat", "aten::concat"}
            and len(npu_kernel_indices) > 1
            and all(_type_key(_npu_kernel_type(npu.kernels[index])) == "concatd"
                    for index in npu_kernel_indices)
            and len({_npu_kernel_core_type(npu.kernels[index])
                     for index in npu_kernel_indices}) == 1
        ):
            # Cat can partition inputs differently on each backend. Even equal
            # kernel counts do not establish a per-kernel correspondence.
            for kernel_index in npu_kernel_indices:
                assignments[kernel_index] = (
                    match, "npu_gpu_bundle_shared", gpu_kernel_indices
                )
        elif len(npu_kernel_indices) == len(gpu_kernel_indices):
            mode = "one_to_one" if len(npu_kernel_indices) == 1 else "ordered_equal_count"
            for npu_kernel_index, gpu_kernel_index in zip(
                npu_kernel_indices, gpu_kernel_indices
            ):
                assignments[npu_kernel_index] = (match, mode, [gpu_kernel_index])
        elif len(npu_kernel_indices) == 1:
            assignments[npu_kernel_indices[0]] = (
                match,
                "npu_to_gpu_bundle",
                gpu_kernel_indices,
            )
        else:
            primary_npu_indices = [
                kernel_index
                for kernel_index in npu_kernel_indices
                if _is_semantic_primary_kernel(
                    npu_op, npu.kernels[kernel_index]
                )
            ]
            if npu_op.canonical_name == "aten::index_select" and compare_metadata(
                npu_op, gpu.ops[match.gpu_op_index]
            ) != ([], []):
                primary_npu_indices = []
            if len(primary_npu_indices) == 1:
                primary_set = set(primary_npu_indices)
                assignments[primary_npu_indices[0]] = (
                    match,
                    "semantic_primary_gpu_bundle",
                    gpu_kernel_indices,
                )
                for kernel_index in npu_kernel_indices:
                    if kernel_index not in primary_set:
                        assignments[kernel_index] = (
                            match,
                            "auxiliary_unmapped",
                            [],
                        )
            elif (
                len(primary_npu_indices) == len(npu_kernel_indices)
                and len({
                    (_npu_kernel_type(npu.kernels[index]),
                     _npu_kernel_core_type(npu.kernels[index]))
                    for index in npu_kernel_indices
                }) == 1
            ):
                # This CPU call launched several kernels of the same primary
                # NPU type. Compare the complete NPU and GPU bundles; there
                # is no defensible per-kernel time allocation.
                for kernel_index in npu_kernel_indices:
                    assignments[kernel_index] = (
                        match,
                        "npu_gpu_bundle_shared",
                        gpu_kernel_indices,
                    )
            elif len(primary_npu_indices) == len(gpu_kernel_indices):
                primary_set = set(primary_npu_indices)
                for npu_kernel_index, gpu_kernel_index in zip(
                    primary_npu_indices, gpu_kernel_indices
                ):
                    assignments[npu_kernel_index] = (
                        match,
                        "semantic_primary",
                        [gpu_kernel_index],
                    )
                for kernel_index in npu_kernel_indices:
                    if kernel_index not in primary_set:
                        assignments[kernel_index] = (
                            match,
                            "auxiliary_unmapped",
                            [],
                        )
            else:
                for kernel_index in npu_kernel_indices:
                    assignments[kernel_index] = (
                        match,
                        "ambiguous_bundle_unmapped",
                        [],
                    )

    # A GPU wrapper such as relu or embedding can launch its kernel from a
    # nested aten op (clamp_min or index_select). Only use the complete child
    # bundle when none of its kernels has already been claimed by a direct
    # NPU-first assignment. Process inner wrappers first so an outer wrapper
    # cannot take a child's kernels before the child itself is considered.
    descendant_kernels = _descendant_gpu_kernel_indices(gpu)
    used_gpu_kernel_indices = {
        gpu_kernel_index
        for _, _, gpu_kernel_indices in assignments.values()
        for gpu_kernel_index in gpu_kernel_indices
    }
    fallback_matches = sorted(
        (
            match
            for match in matches
            if _is_trusted(match)
            and not gpu.ops[match.gpu_op_index].kernel_indices
            and npu.ops[match.npu_op_index].kernel_indices
        ),
        key=lambda match: -_op_depth(gpu.ops, match.gpu_op_index),
    )
    for match in fallback_matches:
        guarded_scope = npu.ops[match.npu_op_index].canonical_name in (
            {"aten::_convolution", "aten::convolution"} | CUSTOM_COMPARABLE_OPS
        )
        if guarded_scope:
            if compare_metadata(npu.ops[match.npu_op_index], gpu.ops[match.gpu_op_index]) != ([], []):
                continue
        gpu_kernel_indices = descendant_kernels[match.gpu_op_index]
        if guarded_scope and (
            any(gpu.kernels[i].link_method not in {"external_id", "runtime_correlation"}
                for i in gpu_kernel_indices)
            or len({(gpu.kernels[i].event.pid, str(gpu.kernels[i].event.args.get("device", "")))
                    for i in gpu_kernel_indices}) > 1
            or any(npu.kernels[i].link_method not in {"external_id", "runtime_correlation", "torch_to_npu",
                       "host_to_device_queue", "host_to_device_thread"}
                   for i in npu.ops[match.npu_op_index].kernel_indices)
        ):
            continue
        if npu.ops[match.npu_op_index].canonical_name == "aten::count_nonzero":
            # ne/copy children have distinct semantics and may already be
            # accounted for. Only the remaining sum computes ReduceSum.
            gpu_kernel_indices = [index for index in gpu_kernel_indices
                                  if gpu.kernels[index].owner_op_index is not None
                                  and gpu.ops[gpu.kernels[index].owner_op_index].canonical_name
                                  == "aten::sum"]
        if not gpu_kernel_indices or any(
            index in used_gpu_kernel_indices for index in gpu_kernel_indices
        ):
            continue
        npu_op = npu.ops[match.npu_op_index]
        npu_kernel_indices = sorted(
            (
                index for index in npu_op.kernel_indices
                if assignments.get(index, (None, "", []))[1]
                not in {"api_semantic_conflict_unmapped", "api_helper_unmapped"}
            ),
            key=lambda index: npu.kernels[index].event.timestamp_us,
        )
        primary_npu_indices = [
            index
            for index in npu_kernel_indices
            if _is_gpu_child_parent_kernel(npu_op, npu.kernels[index])
        ]
        if not primary_npu_indices:
            continue
        if (
            len(primary_npu_indices) == len(npu_kernel_indices) > 1
            and len({
                (_npu_kernel_type(npu.kernels[index]),
                 _npu_kernel_core_type(npu.kernels[index]))
                for index in npu_kernel_indices
            }) == 1
        ):
            for npu_index in npu_kernel_indices:
                assignments[npu_index] = (
                    match, "gpu_child_shared_bundle", gpu_kernel_indices
                )
        elif len(primary_npu_indices) == len(npu_kernel_indices) == len(gpu_kernel_indices):
            mode = (
                "gpu_child_one_to_one"
                if len(npu_kernel_indices) == 1
                else "gpu_child_ordered_equal_count"
            )
            for npu_index, gpu_index in zip(npu_kernel_indices, gpu_kernel_indices):
                assignments[npu_index] = (match, mode, [gpu_index])
        elif len(npu_kernel_indices) == len(primary_npu_indices) == 1:
            assignments[npu_kernel_indices[0]] = (
                match,
                "gpu_child_bundle",
                gpu_kernel_indices,
            )
        else:
            if len(primary_npu_indices) == 1:
                assignments[primary_npu_indices[0]] = (
                    match,
                    "gpu_child_semantic_primary_bundle",
                    gpu_kernel_indices,
                )
                for npu_index in npu_kernel_indices:
                    if npu_index != primary_npu_indices[0]:
                        assignments[npu_index] = (match, "auxiliary_unmapped", [])
            elif len(primary_npu_indices) == len(gpu_kernel_indices):
                primary_set = set(primary_npu_indices)
                for npu_index, gpu_index in zip(
                    primary_npu_indices, gpu_kernel_indices
                ):
                    assignments[npu_index] = (
                        match,
                        "gpu_child_semantic_primary",
                        [gpu_index],
                    )
                for npu_index in npu_kernel_indices:
                    if npu_index not in primary_set:
                        assignments[npu_index] = (match, "auxiliary_unmapped", [])
            else:
                for npu_index in npu_kernel_indices:
                    assignments[npu_index] = (
                        match,
                        "ambiguous_bundle_unmapped",
                        [],
                    )
                continue
        used_gpu_kernel_indices.update(gpu_kernel_indices)

    # Recover a complete single-type logical scope when the NPU implementation
    # owns its kernel in a nested overload (e.g. gelu.out). Keep actual owners;
    # the CPU Match ID explicitly identifies the comparison's parent scope.
    nscope, gscope, mscope = scoped_indices(npu), scoped_indices(gpu), scoped_indices(gpu, True)
    strong_links = {"external_id", "runtime_correlation", "torch_to_npu",
                    "host_to_device_queue", "host_to_device_thread"}
    nmatch_counts = Counter(m.npu_op_index for m in matches)
    gmatch_counts = Counter(m.gpu_op_index for m in matches)
    for match in sorted(matches, key=lambda m: -_op_depth(npu.ops, m.npu_op_index)):
        ni, gi = match.npu_op_index, match.gpu_op_index
        nop, gop = npu.ops[ni], gpu.ops[gi]
        if not _is_trusted(match) or nmatch_counts[ni] != 1 or gmatch_counts[gi] != 1:
            continue
        # Scope lifting is not a generic relaxation of helper/Cast semantics.
        if nop.canonical_name not in {"aten::gelu", "aten::_convolution", "aten::convolution"} and not (
            match.alignment_basis == "layernorm_block_metadata" and nop.canonical_name in {"aten::mean", "aten::var"}
        ):
            continue
        differences, missing = compare_match_metadata(nop, gop, match)
        nk, gk = nscope[ni], gscope[gi]
        if differences or missing or not nk or not gk or mscope[gi]:
            continue
        if any(i in used_gpu_kernel_indices for i in gk):
            continue
        if any(assignments.get(i, (None, "", []))[2] for i in nk):
            continue
        if any(assignments.get(i, (None, "", []))[1] in {
            "api_semantic_conflict_unmapped", "api_helper_unmapped"
        } for i in nk):
            continue
        kernels = [npu.kernels[i] for i in nk]
        if any(k.detail is None or not _is_semantic_primary_kernel(nop, k)
               or kernel_api_is_auxiliary(k) or k.link_method not in strong_links for k in kernels):
            continue
        if len({(k.detail.device_id, k.detail.op_type, k.detail.core_type) for k in kernels}) != 1:
            continue
        if any(gpu.kernels[i].link_method not in {"external_id", "runtime_correlation"} for i in gk):
            continue
        if len({(gpu.kernels[i].event.pid, str(gpu.kernels[i].event.args.get("device", "")))
                for i in gk}) != 1:
            continue
        mode = "logical_scope_bundle" if len(nk) == 1 else "logical_scope_shared_bundle"
        for i in nk:
            assignments[i] = (match, mode, gk)
        used_gpu_kernel_indices.update(gk)

    # Some fused/custom kernels are present in kernel_details.csv but have no
    # NPU flow edge or owning aten op. For explicitly known fused equivalents,
    # map the GPU bundle only to the NPU primary kernel and retain the other NPU
    # kernels as bundle auxiliaries. Existing assignments and already-consumed
    # GPU kernels are never overwritten, so GPU time cannot be duplicated.
    fused_occurrences = _explicit_fused_op_occurrences(gpu, npu)
    fused_groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for occurrence in fused_occurrences:
        fused_groups[str(occurrence["fused_op"])].append(occurrence)
    for occurrences in fused_groups.values():
        if any(
            int(occurrence["npu_primary_kernel_index"]) in assignments
            or any(
                int(index) in assignments
                for index in occurrence["npu_auxiliary_kernel_indices"]
            )
            or any(
                int(index) in used_gpu_kernel_indices
                for index in occurrence["gpu_kernel_indices"]
            )
            for occurrence in occurrences
        ):
            continue
        for occurrence in occurrences:
            primary_index = int(occurrence["npu_primary_kernel_index"])
            gpu_op_index = int(occurrence["gpu_op_index"])
            gpu_kernel_indices = [
                int(index) for index in occurrence["gpu_kernel_indices"]
            ]
            assignments[primary_index] = (
                None,
                "fused_primary_gpu_bundle",
                gpu_kernel_indices,
            )
            direct_gpu_op_by_npu_kernel[primary_index] = gpu_op_index
            for auxiliary_index in occurrence["npu_auxiliary_kernel_indices"]:
                auxiliary_index = int(auxiliary_index)
                assignments[auxiliary_index] = (
                    None,
                    "fused_bundle_auxiliary",
                    [],
                )
                direct_gpu_op_by_npu_kernel[auxiliary_index] = gpu_op_index
            used_gpu_kernel_indices.update(gpu_kernel_indices)

    rows: List[Dict[str, Any]] = []
    for kernel_index, npu_kernel in enumerate(npu.kernels):
        # Match the scope of kernel_details.csv/op_statistic.csv. Trace-only
        # tasks such as MEMCPY_ASYNC remain available in the existing kernel
        # inventory and link diagnostics, but are not mixed into this report.
        if npu_kernel.detail is None:
            continue
        match, mode, gpu_kernel_indices = assignments.get(
            kernel_index, (None, "unlinked_npu_kernel", [])
        )
        owner = (
            npu.ops[npu_kernel.owner_op_index]
            if npu_kernel.owner_op_index is not None
            else None
        )
        direct_gpu_op_index = direct_gpu_op_by_npu_kernel.get(kernel_index)
        gpu_op = (
            gpu.ops[match.gpu_op_index]
            if match is not None
            else gpu.ops[direct_gpu_op_index]
            if direct_gpu_op_index is not None
            else None
        )
        gpu_kernels = [gpu.kernels[index] for index in gpu_kernel_indices]
        gpu_metrics = bundle_metrics(gpu_kernels)
        npu_duration = npu_kernel.event.duration_us
        comparable = bool(gpu_kernels)
        shared_bundle = mode in _SHARED_GPU_BUNDLE_MODES
        direct = len(gpu_kernels) == 1 and not shared_bundle
        accounted = comparable or mode in {
            "auxiliary_unmapped",
            "api_helper_unmapped",
            "ambiguous_bundle_unmapped",
            "fused_bundle_auxiliary",
        }
        rows.append(
            {
                "Device_id": (
                    npu_kernel.detail.device_id
                    if npu_kernel.detail is not None
                    else ""
                ),
                "OP Type": _npu_kernel_type(npu_kernel),
                "Core Type": _npu_kernel_core_type(npu_kernel),
                "NPU Kernel Index": kernel_index,
                "NPU Kernel Name": npu_kernel.event.name,
                "NPU Duration(us)": _round(npu_duration),
                "NPU Stream ID": npu_kernel.stream_id,
                "NPU Task ID": npu_kernel.task_id,
                "NPU CPU OP Index": (
                    npu_kernel.owner_op_index if owner is not None else ""
                ),
                "NPU CPU OP Name": owner.event.name if owner is not None else "",
                "NPU Canonical OP": owner.canonical_name if owner is not None else "",
                "API Expected CPU OPs": _json_list(sorted(kernel_api_expected_ops(npu_kernel))),
                "API Semantic Conflict": mode == "api_semantic_conflict_unmapped",
                "API Helper Kernel": mode == "api_helper_unmapped",
                "CPU Match ID": (
                    match.match_id
                    if match is not None
                    else f"gpu-custom:{direct_gpu_op_index}"
                    if direct_gpu_op_index is not None
                    else ""
                ),
                "CPU Match Confidence": (
                    match.confidence
                    if match is not None
                    else "high"
                    if direct_gpu_op_index is not None
                    else ""
                ),
                "CPU Alignment Basis": (
                    match.alignment_basis
                    if match is not None
                    else "explicit_fused_type_equal_count_order"
                    if direct_gpu_op_index is not None
                    else ""
                ),
                "GPU OP Name": gpu_op.event.name if gpu_op is not None else "",
                "Mapping Mode": mode,
                "Mapped": comparable,
                "Direct Kernel Pair": direct,
                "GPU Bundle Shared": shared_bundle,
                "Comparable": comparable,
                "Accounted": accounted,
                "GPU Kernel Source": (
                    "logical_cpu_scope"
                    if mode.startswith("logical_scope_")
                    else
                    "descendant_op"
                    if mode.startswith("gpu_child_")
                    else "explicit_fused_op"
                    if mode.startswith("fused_")
                    else "direct_op"
                    if comparable
                    else ""
                ),
                "GPU Kernel Count": gpu_metrics.kernel_count,
                "GPU Kernel Indices": _json_list(gpu_kernel_indices),
                "GPU Kernel Names": _json_list(
                    kernel.event.name for kernel in gpu_kernels
                ),
                "GPU Kernel Owner OP Names": _json_list(
                    gpu.ops[gpu.kernels[index].owner_op_index].event.name
                    if gpu.kernels[index].owner_op_index is not None
                    else ""
                    for index in gpu_kernel_indices
                ),
                "GPU Kernel Durations(us)": _json_list(
                    _round(kernel.event.duration_us) for kernel in gpu_kernels
                ),
                "GPU Total Time(us)": _round(gpu_metrics.sum_us),
                "GPU Device Busy(us)": _round(gpu_metrics.busy_us),
                "GPU Bundle Span(us)": _round(gpu_metrics.span_us),
                "Performance Ratio(GPU/NPU)": (
                    "" if shared_bundle else _ratio(gpu_metrics.sum_us, npu_duration)
                ),
            }
        )
    enrich_memory_mappings(gpu, npu, matches, rows)
    return rows


def build_npu_gpu_op_statistic_rows(
    mapping_rows: Sequence[Mapping[str, Any]],
    fused_mapping_rows: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Aggregate NPU-first mappings in an op_statistic.csv-like layout."""
    fused_labels_by_npu_type: Dict[str, Tuple[str, str]] = {}
    fused_type_order: Dict[str, int] = {}
    for row in fused_mapping_rows:
        primary_type = str(row["NPU Primary Type"])
        auxiliary_types = [
            str(value) for value in json.loads(str(row["NPU Auxiliary Types"]))
        ]
        npu_types = [primary_type, *auxiliary_types]
        npu_label = " + ".join(npu_types)
        gpu_label = str(row["GPU OP Name"])
        for order, npu_type in enumerate(npu_types):
            fused_labels_by_npu_type[npu_type] = (npu_label, gpu_label)
            fused_type_order[npu_type] = order

    groups: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        groups[(str(row["Device_id"]), str(row["OP Type"]), str(row["Core Type"]))].append(
            row
        )

    total_npu_time = sum(float(row["NPU Duration(us)"]) for row in mapping_rows)
    result: List[Dict[str, Any]] = []
    for (device_id, op_type, core_type), rows in groups.items():
        npu_durations = [float(row["NPU Duration(us)"]) for row in rows]
        mapped_rows = [row for row in rows if row["Mapped"]]
        direct_rows = [
            row
            for row in rows
            if row.get("Direct Kernel Pair", False)
            or (
                "Direct Kernel Pair" not in row
                and row["Mapped"]
                and len(json.loads(str(row["GPU Kernel Durations(us)"]))) == 1
            )
        ]
        accounted_rows = [
            row
            for row in rows
            if row.get(
                "Accounted",
                bool(row["Mapped"])
                or row["Mapping Mode"]
                in {
                    "auxiliary_unmapped",
                    "api_helper_unmapped",
                    "ambiguous_bundle_unmapped",
                    "fused_bundle_auxiliary",
                },
            )
        ]
        mapped_npu_durations = [
            float(row["NPU Duration(us)"]) for row in mapped_rows
        ]
        direct_npu_total = sum(float(row["NPU Duration(us)"]) for row in direct_rows)
        accounted_npu_total = sum(
            float(row["NPU Duration(us)"]) for row in accounted_rows
        )
        gpu_durations = []
        memory_durations = []
        seen_memory_indices: set[int] = set()
        seen_gpu_indices: set[int] = set()
        for row in mapped_rows:
            indices = json.loads(str(row.get("GPU Kernel Indices", "[]")))
            durations = json.loads(str(row["GPU Kernel Durations(us)"]))
            for position, duration in enumerate(durations):
                if position < len(indices):
                    index = int(indices[position])
                    if index in seen_gpu_indices:
                        continue
                    seen_gpu_indices.add(index)
                gpu_durations.append(float(duration))
            memory_indices = json.loads(str(row.get("GPU Memory Activity Indices", "[]")))
            memory_times = json.loads(str(row.get("GPU Memory Activity Durations(us)", "[]")))
            for index, duration in zip(memory_indices, memory_times):
                if int(index) not in seen_memory_indices:
                    seen_memory_indices.add(int(index))
                    memory_durations.append(float(duration))
        npu_total = sum(npu_durations)
        mapped_npu_total = sum(mapped_npu_durations)
        gpu_total = sum(gpu_durations) + sum(memory_durations)
        mode_counts = Counter(str(row["Mapping Mode"]) for row in rows)
        match_ids = {
            str(row["CPU Match ID"])
            for row in mapped_rows
            if row["CPU Match ID"] != ""
        }
        fused_npu_type, fused_gpu_type = fused_labels_by_npu_type.get(
            op_type,
            (op_type, op_type),
        )
        result.append(
            {
                "Device_id": device_id,
                "FUSED_OP_NPU_TYPE": fused_npu_type,
                "FUSED_OP_GPU_TYPE": fused_gpu_type,
                "OP Type": op_type,
                "Core Type": core_type,
                "NPU Kernel Count": len(rows),
                "Mapped NPU Kernel Count": len(mapped_rows),
                "Unmapped NPU Kernel Count": len(rows) - len(mapped_rows),
                "Mapping Coverage(%)": _round(len(mapped_rows) * 100.0 / len(rows)),
                "Direct Mapped NPU Kernel Count": len(direct_rows),
                "Direct Mapping Coverage(%)": _round(
                    len(direct_rows) * 100.0 / len(rows)
                ),
                "Accounted NPU Kernel Count": len(accounted_rows),
                "Accounted Coverage(%)": _round(
                    len(accounted_rows) * 100.0 / len(rows)
                ),
                "Comparable NPU Kernel Count": len(mapped_rows),
                "Comparable Coverage(%)": _round(
                    len(mapped_rows) * 100.0 / len(rows)
                ),
                "Mapped CPU OP Calls": len(match_ids),
                "GPU Kernel Count": len(gpu_durations),
                "GPU Memory Activity Count": len(memory_durations),
                "GPU Kernel Time(us)": _round(sum(gpu_durations)),
                "GPU Memory Time(us)": _round(sum(memory_durations)),
                "GPU Activity Kind": "+".join(kind for kind, values in
                    (("kernel", gpu_durations), ("memcpy", memory_durations)) if values),
                "NPU Total Time(us)": _round(npu_total),
                "Mapped NPU Time(us)": _round(mapped_npu_total),
                "Direct Mapped NPU Time(us)": _round(direct_npu_total),
                "Direct Time Coverage(%)": _ratio(direct_npu_total * 100.0, npu_total),
                "Accounted NPU Time(us)": _round(accounted_npu_total),
                "Accounted Time Coverage(%)": _ratio(accounted_npu_total * 100.0, npu_total),
                "Comparable NPU Time(us)": _round(mapped_npu_total),
                "Comparable Time Coverage(%)": _ratio(mapped_npu_total * 100.0, npu_total),
                "GPU Total Time(us)": _round(gpu_total),
                "Comparable Ratio(GPU/NPU)": _ratio(gpu_total, mapped_npu_total),
                "Total Time Ratio(GPU/NPU)": _ratio(gpu_total, npu_total),
                "Diff Time(GPU-NPU)(us)": _round(gpu_total - mapped_npu_total),
                "NPU Min Time(us)": _round(min(npu_durations)),
                "NPU Avg Time(us)": _round(mean(npu_durations)),
                "NPU Max Time(us)": _round(max(npu_durations)),
                "GPU Min Time(us)": _round(min(gpu_durations)) if gpu_durations else "",
                "GPU Avg Time(us)": _round(mean(gpu_durations)) if gpu_durations else "",
                "GPU Max Time(us)": _round(max(gpu_durations)) if gpu_durations else "",
                "NPU Ratio(%)": (
                    _round(npu_total * 100.0 / total_npu_time)
                    if total_npu_time
                    else 0.0
                ),
                "Mapping Mode Counts": json.dumps(
                    dict(sorted(mode_counts.items())),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    group_totals: Dict[Tuple[str, str, str], float] = defaultdict(float)
    for row in result:
        group_totals[
            (
                str(row["Device_id"]),
                str(row["FUSED_OP_NPU_TYPE"]),
                str(row["FUSED_OP_GPU_TYPE"]),
            )
        ] += float(row["NPU Total Time(us)"])
    return sorted(
        result,
        key=lambda row: (
            -group_totals[
                (
                    str(row["Device_id"]),
                    str(row["FUSED_OP_NPU_TYPE"]),
                    str(row["FUSED_OP_GPU_TYPE"]),
                )
            ],
            str(row["FUSED_OP_NPU_TYPE"]),
            fused_type_order.get(str(row["OP Type"]), 0),
            -float(row["NPU Total Time(us)"]),
            str(row["OP Type"]),
            str(row["Core Type"]),
        ),
    )


def build_fused_op_mapping_rows(
    gpu: ParsedProfile, npu: ParsedProfile
) -> List[Dict[str, Any]]:
    """Build one auditable row for each explicitly recognized fused-op call."""
    rows: List[Dict[str, Any]] = []
    for occurrence in _explicit_fused_op_occurrences(gpu, npu):
        gpu_op_index = int(occurrence["gpu_op_index"])
        gpu_kernel_indices = [
            int(index) for index in occurrence["gpu_kernel_indices"]
        ]
        npu_primary_index = int(occurrence["npu_primary_kernel_index"])
        npu_auxiliary_indices = [
            int(index) for index in occurrence["npu_auxiliary_kernel_indices"]
        ]
        npu_kernel_indices = [
            int(index) for index in occurrence["npu_kernel_indices"]
        ]
        gpu_kernels = [gpu.kernels[index] for index in gpu_kernel_indices]
        npu_kernels = [npu.kernels[index] for index in npu_kernel_indices]
        npu_primary_kernel = npu.kernels[npu_primary_index]
        npu_auxiliary_kernels = [npu.kernels[index] for index in npu_auxiliary_indices]
        gpu_metrics = bundle_metrics(gpu_kernels)
        npu_metrics = bundle_metrics(npu_kernels)
        npu_auxiliary_time = sum(
            kernel.event.duration_us for kernel in npu_auxiliary_kernels
        )
        rows.append(
            {
                "Fused OP": occurrence["fused_op"],
                "Occurrence": occurrence["occurrence"],
                "Alignment Basis": "explicit_fused_type_equal_count_order",
                "GPU OP Index": gpu_op_index,
                "GPU OP Name": gpu.ops[gpu_op_index].event.name,
                "GPU Kernel Count": gpu_metrics.kernel_count,
                "GPU Kernel Indices": _json_list(gpu_kernel_indices),
                "GPU Kernel Names": _json_list(
                    kernel.event.name for kernel in gpu_kernels
                ),
                "GPU Kernel Durations(us)": _json_list(
                    _round(kernel.event.duration_us) for kernel in gpu_kernels
                ),
                "GPU Kernel Sum(us)": _round(gpu_metrics.sum_us),
                "GPU Device Busy(us)": _round(gpu_metrics.busy_us),
                "GPU Bundle Span(us)": _round(gpu_metrics.span_us),
                "NPU Kernel Count": npu_metrics.kernel_count,
                "NPU Kernel Indices": _json_list(npu_kernel_indices),
                "NPU Kernel Names": _json_list(
                    kernel.event.name for kernel in npu_kernels
                ),
                "NPU Kernel Types": _json_list(
                    _npu_kernel_type(kernel) for kernel in npu_kernels
                ),
                "NPU Core Types": _json_list(
                    _npu_kernel_core_type(kernel) for kernel in npu_kernels
                ),
                "NPU Kernel Durations(us)": _json_list(
                    _round(kernel.event.duration_us) for kernel in npu_kernels
                ),
                "NPU Kernel Sum(us)": _round(npu_metrics.sum_us),
                "NPU Device Busy(us)": _round(npu_metrics.busy_us),
                "NPU Bundle Span(us)": _round(npu_metrics.span_us),
                "NPU Primary Kernel Index": npu_primary_index,
                "NPU Primary Type": _npu_kernel_type(npu_primary_kernel),
                "NPU Primary Time(us)": _round(
                    npu_primary_kernel.event.duration_us
                ),
                "NPU Auxiliary Kernel Indices": _json_list(npu_auxiliary_indices),
                "NPU Auxiliary Types": _json_list(
                    _npu_kernel_type(kernel) for kernel in npu_auxiliary_kernels
                ),
                "NPU Auxiliary Time(us)": _round(npu_auxiliary_time),
                "Performance Ratio(GPU/NPU Bundle)": _ratio(
                    gpu_metrics.sum_us, npu_metrics.sum_us
                ),
                "Diff Time(GPU-NPU Bundle)(us)": _round(
                    gpu_metrics.sum_us - npu_metrics.sum_us
                ),
            }
        )
    return rows


def build_fused_op_statistic_rows(
    mapping_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in mapping_rows:
        groups[str(row["Fused OP"])].append(row)

    result: List[Dict[str, Any]] = []
    for fused_op, rows in groups.items():
        gpu_times = [float(row["GPU Kernel Sum(us)"]) for row in rows]
        npu_times = [float(row["NPU Kernel Sum(us)"]) for row in rows]
        gpu_total = sum(gpu_times)
        npu_total = sum(npu_times)
        primary_total = sum(float(row["NPU Primary Time(us)"]) for row in rows)
        auxiliary_total = sum(float(row["NPU Auxiliary Time(us)"]) for row in rows)
        npu_type_counts: Counter[str] = Counter()
        for row in rows:
            npu_type_counts.update(json.loads(str(row["NPU Kernel Types"])))
        result.append(
            {
                "Fused OP": fused_op,
                "Call Count": len(rows),
                "Alignment Basis": "explicit_fused_type_equal_count_order",
                "GPU OP Name": rows[0]["GPU OP Name"],
                "GPU Kernel Count": sum(int(row["GPU Kernel Count"]) for row in rows),
                "GPU Total Time(us)": _round(gpu_total),
                "GPU Avg Bundle Time(us)": _round(mean(gpu_times)),
                "GPU Min Bundle Time(us)": _round(min(gpu_times)),
                "GPU Max Bundle Time(us)": _round(max(gpu_times)),
                "NPU Kernel Count": sum(int(row["NPU Kernel Count"]) for row in rows),
                "NPU Kernel Type Counts": json.dumps(
                    dict(sorted(npu_type_counts.items())),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                "NPU Total Time(us)": _round(npu_total),
                "NPU Avg Bundle Time(us)": _round(mean(npu_times)),
                "NPU Min Bundle Time(us)": _round(min(npu_times)),
                "NPU Max Bundle Time(us)": _round(max(npu_times)),
                "NPU Primary Type": rows[0]["NPU Primary Type"],
                "NPU Primary Total Time(us)": _round(primary_total),
                "NPU Auxiliary Types": rows[0]["NPU Auxiliary Types"],
                "NPU Auxiliary Total Time(us)": _round(auxiliary_total),
                "Performance Ratio(GPU/NPU Bundle)": _ratio(gpu_total, npu_total),
                "Diff Time(GPU-NPU Bundle)(us)": _round(gpu_total - npu_total),
            }
        )
    return sorted(result, key=lambda row: -float(row["NPU Total Time(us)"]))


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
    sequence_matched_counts: Counter[str] = Counter()
    trusted_matched_counts: Counter[str] = Counter()
    performance_matches: Dict[str, List[OpMatch]] = defaultdict(list)
    for index, op in enumerate(gpu.ops):
        gpu_groups[op_map.get(op.canonical_name, op.canonical_name)].append(index)
    for index, op in enumerate(npu.ops):
        npu_groups[op.canonical_name].append(index)
    for match in matches:
        op_type = npu.ops[match.npu_op_index].canonical_name
        sequence_matched_counts[op_type] += 1
        if _is_trusted(match):
            trusted_matched_counts[op_type] += 1
        if _is_performance_match(gpu, npu, match):
            performance_matches[op_type].append(match)

    rows: List[Dict[str, Any]] = []
    for op_type in sorted(set(gpu_groups) | set(npu_groups)):
        gpu_indices = gpu_groups.get(op_type, [])
        npu_indices = npu_groups.get(op_type, [])
        gpu_values = _duration_values(gpu, gpu_indices)
        npu_values = _duration_values(npu, npu_indices)
        gpu_total = sum(gpu_values)
        npu_total = sum(npu_values)
        npu_raw_kernel_count = sum(
            len(npu.ops[index].kernel_indices) for index in npu_indices
        )
        npu_same_type_kernel_count = sum(
            len(
                _semantic_primary_kernels(
                    npu.ops[index], _op_kernels(npu, index)
                )
            )
            for index in npu_indices
        )
        op_matches = performance_matches.get(op_type, [])
        matched_gpu_values = [
            bundle_metrics(_op_kernels(gpu, match.gpu_op_index)).sum_us
            for match in op_matches
        ]
        matched_npu_values = [
            bundle_metrics(_op_kernels(npu, match.npu_op_index)).sum_us
            for match in op_matches
        ]
        matched_gpu_total = sum(matched_gpu_values)
        matched_npu_total = sum(matched_npu_values)
        ratios = [
            npu_value / gpu_value
            for gpu_value, npu_value in zip(matched_gpu_values, matched_npu_values)
            if gpu_value > 0
        ]
        rows.append(
            {
                "OP Type": op_type,
                "GPU Calls": len(gpu_indices),
                "NPU Calls": len(npu_indices),
                "Matched Calls": sequence_matched_counts[op_type],
                "Trusted Matched Calls": trusted_matched_counts[op_type],
                "Performance Matched Calls": len(op_matches),
                "GPU Raw Kernel Count": sum(len(gpu.ops[index].kernel_indices) for index in gpu_indices),
                "NPU Raw Kernel Count": npu_raw_kernel_count,
                "NPU Same-Type Kernel Count": npu_same_type_kernel_count,
                "NPU Other-Type Kernel Count": (
                    npu_raw_kernel_count - npu_same_type_kernel_count
                ),
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
                "Matched GPU Total Time(us)": _round(matched_gpu_total),
                "Matched NPU Total Time(us)": _round(matched_npu_total),
                "Matched NPU/GPU Ratio": _ratio(matched_npu_total, matched_gpu_total),
                "Matched Ratio P50": _round(_percentile(ratios, 0.50)) if ratios else "",
                "Matched Ratio P90": _round(_percentile(ratios, 0.90)) if ratios else "",
                "Matched Ratio Outside[0.5,2]": sum(
                    ratio < 0.5 or ratio > 2.0 for ratio in ratios
                ),
            }
        )
    return rows


def build_kernel_type_statistic_rows(
    gpu: ParsedProfile, npu: ParsedProfile, matches: Sequence[OpMatch]
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[OpMatch]] = defaultdict(list)
    for match in matches:
        if not _is_performance_match(gpu, npu, match):
            continue
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
        gpu_values = [metric.sum_us for metric in gpu_metrics]
        npu_values = [metric.sum_us for metric in npu_metrics]
        ratios = [
            npu_value / gpu_value
            for gpu_value, npu_value in zip(gpu_values, npu_values)
            if gpu_value > 0
        ]
        npu_kernel_count = sum(metric.kernel_count for metric in npu_metrics)
        rows.append(
            {
                "Mapped OP Type": op_type,
                "Core Type": core_type,
                "Count": npu_kernel_count,
                "Logical Calls": len(group_matches),
                "GPU Raw Kernel Count": sum(metric.kernel_count for metric in gpu_metrics),
                "NPU Raw Kernel Count": npu_kernel_count,
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
                "Ratio P50": _round(_percentile(ratios, 0.50)) if ratios else "",
                "Ratio P90": _round(_percentile(ratios, 0.90)) if ratios else "",
                "Ratio Outside[0.5,2]": sum(
                    ratio < 0.5 or ratio > 2.0 for ratio in ratios
                ),
            }
        )
    return rows


def build_module_statistic_rows(
    gpu: ParsedProfile, npu: ParsedProfile
) -> List[Dict[str, Any]]:
    def aggregate(profile: ParsedProfile) -> Dict[str, Dict[str, Any]]:
        result: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "kernel_indices": set(), "raw_paths": set()}
        )
        for index, op in enumerate(profile.ops):
            key = op.module_key or "[NO MODULE]"
            result[key]["calls"] += 1
            result[key]["kernel_indices"].update(op.kernel_indices)
            if op.module_path:
                result[key]["raw_paths"].add(op.module_path)
        return result

    gpu_values = aggregate(gpu)
    npu_values = aggregate(npu)
    rows: List[Dict[str, Any]] = []
    for module in sorted(set(gpu_values) | set(npu_values)):
        gpu_row = gpu_values.get(module, {})
        npu_row = npu_values.get(module, {})
        gpu_kernels = [
            gpu.kernels[index] for index in sorted(gpu_row.get("kernel_indices", set()))
        ]
        npu_kernels = [
            npu.kernels[index] for index in sorted(npu_row.get("kernel_indices", set()))
        ]
        gpu_metrics = bundle_metrics(gpu_kernels)
        npu_metrics = bundle_metrics(npu_kernels)
        rows.append(
            {
                "Module Key": module,
                "GPU Raw Module Paths": _json_list(
                    sorted(gpu_row.get("raw_paths", set()))[:20]
                ),
                "NPU Raw Module Paths": _json_list(
                    sorted(npu_row.get("raw_paths", set()))[:20]
                ),
                "GPU OP Calls": int(gpu_row.get("calls", 0)),
                "NPU OP Calls": int(npu_row.get("calls", 0)),
                "GPU Kernel Count": gpu_metrics.kernel_count,
                "NPU Kernel Count": npu_metrics.kernel_count,
                "GPU Kernel Sum(us)": _round(gpu_metrics.sum_us),
                "NPU Kernel Sum(us)": _round(npu_metrics.sum_us),
                "GPU Device Busy(us)": _round(gpu_metrics.busy_us),
                "NPU Device Busy(us)": _round(npu_metrics.busy_us),
                "GPU Region Span(us)": _round(gpu_metrics.span_us),
                "NPU Region Span(us)": _round(npu_metrics.span_us),
                "NPU/GPU Ratio": _ratio(npu_metrics.sum_us, gpu_metrics.sum_us),
                "Diff Time(us)": _round(npu_metrics.sum_us - gpu_metrics.sum_us),
            }
        )
    return rows


def _op_name_signature(profile: ParsedProfile, op_indices: Sequence[int]) -> str:
    counts = Counter(profile.ops[index].canonical_name for index in op_indices)
    return _json_list(
        f"{name}x{count}" for name, count in counts.most_common(20)
    )


def build_region_match_rows(
    gpu: ParsedProfile, npu: ParsedProfile, matches: Sequence[OpMatch]
) -> List[Dict[str, Any]]:
    """Compare coarse spans bounded by trusted semantic device anchors."""
    rows: List[Dict[str, Any]] = []
    anchors = [
        match
        for match in matches
        if match.alignment_basis == "semantic_anchor"
        and _is_performance_match(gpu, npu, match)
    ]
    if len(anchors) < 2:
        anchors = [
            match for match in matches if _is_performance_match(gpu, npu, match)
        ]
    ordered = sorted(anchors, key=lambda match: (match.gpu_op_index, match.npu_op_index))
    previous_gpu = -1
    previous_npu = -1
    previous_match: Any = None
    region_id = 1
    for current_match in list(ordered) + [None]:
        current_gpu = current_match.gpu_op_index if current_match is not None else len(gpu.ops)
        current_npu = current_match.npu_op_index if current_match is not None else len(npu.ops)
        gpu_indices = list(range(previous_gpu + 1, current_gpu))
        npu_indices = list(range(previous_npu + 1, current_npu))
        if gpu_indices or npu_indices:
            gpu_kernels = _kernels_for_ops(gpu, gpu_indices)
            npu_kernels = _kernels_for_ops(npu, npu_indices)
            gpu_metrics = bundle_metrics(gpu_kernels)
            npu_metrics = bundle_metrics(npu_kernels)
            if gpu_metrics.kernel_count or npu_metrics.kernel_count:
                gpu_modules = {
                    gpu.ops[index].module_key for index in gpu_indices if gpu.ops[index].module_key
                }
                npu_modules = {
                    npu.ops[index].module_key for index in npu_indices if npu.ops[index].module_key
                }
                score = 0.35
                if previous_match is not None and current_match is not None:
                    if _is_trusted(previous_match) and _is_trusted(current_match):
                        score += 0.25
                elif (previous_match is not None and _is_trusted(previous_match)) or (
                    current_match is not None and _is_trusted(current_match)
                ):
                    score += 0.12
                if gpu_modules & npu_modules:
                    score += 0.20
                if max(len(gpu_indices), len(npu_indices)) <= 64:
                    score += 0.10
                if gpu_indices and npu_indices:
                    count_ratio = max(len(gpu_indices), len(npu_indices)) / max(
                        1, min(len(gpu_indices), len(npu_indices))
                    )
                    if count_ratio <= 4.0:
                        score += 0.10
                score = min(score, 1.0)
                confidence = "high" if score >= 0.75 else "medium" if score >= 0.55 else "low"
                eligible = (
                    confidence in {"medium", "high"}
                    and gpu_metrics.kernel_count > 0
                    and npu_metrics.kernel_count > 0
                )
                if gpu_indices and npu_indices:
                    region_type = "many_to_many"
                elif gpu_indices:
                    region_type = "gpu_only"
                else:
                    region_type = "npu_only"
                row: Dict[str, Any] = {
                    "Region ID": region_id,
                    "Confidence": confidence,
                    "Region Score": _round(score),
                    "Eligible For Performance": eligible,
                    "Region Type": region_type,
                    "Left Anchor Match ID": previous_match.match_id if previous_match else "",
                    "Right Anchor Match ID": current_match.match_id if current_match else "",
                    "GPU OP Start": gpu_indices[0] if gpu_indices else "",
                    "GPU OP End": gpu_indices[-1] if gpu_indices else "",
                    "NPU OP Start": npu_indices[0] if npu_indices else "",
                    "NPU OP End": npu_indices[-1] if npu_indices else "",
                    "GPU OP Count": len(gpu_indices),
                    "NPU OP Count": len(npu_indices),
                    "GPU OP Signature": _op_name_signature(gpu, gpu_indices),
                    "NPU OP Signature": _op_name_signature(npu, npu_indices),
                    "GPU Module Keys": _json_list(sorted(gpu_modules)),
                    "NPU Module Keys": _json_list(sorted(npu_modules)),
                    "Common Module Keys": _json_list(sorted(gpu_modules & npu_modules)),
                    "NPU/GPU Sum Ratio": _ratio(npu_metrics.sum_us, gpu_metrics.sum_us),
                    "NPU/GPU Busy Ratio": _ratio(npu_metrics.busy_us, gpu_metrics.busy_us),
                    "Sum Diff(us)": _round(npu_metrics.sum_us - gpu_metrics.sum_us),
                }
                row.update(_metric_fields("GPU", gpu_metrics))
                row.update(_metric_fields("NPU", npu_metrics))
                rows.append(row)
                region_id += 1
        if current_match is not None:
            previous_gpu = current_gpu
            previous_npu = current_npu
            previous_match = current_match
    return rows


def build_kernel_inventory_rows(profile: ParsedProfile) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[KernelRecord]] = defaultdict(list)
    for kernel in profile.kernels:
        detail = kernel.detail
        task_type = str(kernel.event.args.get("Task Type", "") or "")
        if detail and detail.op_type:
            op_type = detail.op_type
        elif task_type:
            op_type = task_type
        elif profile.side == "gpu":
            op_type = kernel.event.name
        else:
            op_type = "UNRESOLVED"
        if detail and detail.core_type:
            core_type = detail.core_type
        elif task_type:
            core_type = task_type
        elif profile.side == "gpu":
            core_type = "GPU"
        else:
            core_type = "UNRESOLVED"
        groups[(op_type, core_type, task_type)].append(kernel)

    total_time = sum(kernel.event.duration_us for kernel in profile.kernels)
    rows: List[Dict[str, Any]] = []
    for (op_type, core_type, task_type), kernels in sorted(groups.items()):
        durations = [kernel.event.duration_us for kernel in kernels]
        linked = [kernel for kernel in kernels if kernel.owner_op_index is not None]
        rows.append(
            {
                "Side": profile.side,
                "Kernel Type": op_type,
                "Core Type": core_type,
                "Task Type": task_type,
                "Count": len(kernels),
                "Linked Count": len(linked),
                "Detail Count": sum(kernel.detail is not None for kernel in kernels),
                "Total Time(us)": _round(sum(durations)),
                "Average Time(us)": _round(mean(durations)),
                "P50 Time(us)": _round(_percentile(durations, 0.50)),
                "P90 Time(us)": _round(_percentile(durations, 0.90)),
                "Max Time(us)": _round(max(durations)),
                "Time Ratio(%)": _round(sum(durations) * 100.0 / total_time)
                if total_time
                else 0.0,
            }
        )
    return rows


def build_kernel_link_rows(profile: ParsedProfile) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, kernel in enumerate(profile.kernels):
        owner = (
            profile.ops[kernel.owner_op_index]
            if kernel.owner_op_index is not None
            else None
        )
        original_owner = (
            profile.ops[kernel.original_owner_op_index]
            if kernel.original_owner_op_index is not None
            else None
        )
        rows.append(
            {
                "Side": profile.side,
                "Kernel Index": index,
                "Kernel Name": kernel.event.name,
                "Timestamp(us)": _round(kernel.event.timestamp_us),
                "Duration(us)": _round(kernel.event.duration_us),
                "Stream ID": kernel.stream_id,
                "Task ID": kernel.task_id,
                "Task Type": str(kernel.event.args.get("Task Type", "") or ""),
                "Owner OP Index": kernel.owner_op_index if owner is not None else "",
                "Owner OP Name": owner.event.name if owner is not None else "",
                "Link Method": kernel.link_method,
                "Link Scope": kernel.link_scope,
                "Link Distance(us)": _round(kernel.link_distance_us)
                if kernel.link_distance_us is not None
                else "",
                "Detail Row": kernel.detail.row_index if kernel.detail else "",
                "Detail Type": kernel.detail.op_type if kernel.detail else "",
                "Original Owner OP Index": (
                    kernel.original_owner_op_index if original_owner is not None else ""
                ),
                "Original Owner OP Name": (
                    original_owner.event.name if original_owner is not None else ""
                ),
                "Original Link Method": kernel.original_link_method,
                "Original Link Scope": kernel.original_link_scope,
                "Original Link Distance(us)": (
                    _round(kernel.original_link_distance_us)
                    if kernel.original_link_distance_us is not None
                    else ""
                ),
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
                "Raw Input Shapes": op.raw_input_shapes,
                "Module Path": op.module_path,
                "Module Key": op.module_key,
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
        "excluded_task_type_counts": profile.excluded_task_type_counts,
        "flow_link_diagnostics": profile.flow_link_diagnostics,
        "kernel_detail_count": profile.kernel_detail_count,
        "kernel_detail_matched_count": profile.kernel_detail_matched_count,
        "kernel_detail_synthetic_count": profile.kernel_detail_synthetic_count,
        "kernel_detail_link_methods": profile.kernel_detail_link_methods,
        "kernel_owner_reconciliation": profile.kernel_owner_reconciliation,
    }


def write_reports(
    output_dir: str,
    gpu: ParsedProfile,
    npu: ParsedProfile,
    matches: Sequence[OpMatch],
    unmatched_gpu: Sequence[int],
    unmatched_npu: Sequence[int],
    op_map: Mapping[str, str],
    *,
    diagnostic_coverage_threshold: float = 90.0,
    manual_links=(),
    manual_enabled=False,
) -> Dict[str, Any]:
    validate_threshold(diagnostic_coverage_threshold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    match_rows = build_match_rows(gpu, npu, matches)
    mapping_rows = build_gpu_kernel_mapping_rows(gpu, npu, matches)
    npu_kernel_mapping_rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
    if manual_enabled or manual_links:
        from .manual_mapping import apply_links
        apply_links(gpu, npu, npu_kernel_mapping_rows, manual_links)
    from .core_breakdown import annotate_counters, build_core_breakdown, statistic_rows
    annotate_counters(npu, npu_kernel_mapping_rows)
    core_breakdown = build_core_breakdown(gpu, npu_kernel_mapping_rows)
    core_statistic_rows = statistic_rows(core_breakdown)
    cpu_bundle_rows = build_cpu_bundle_rows(gpu, npu, matches, npu_kernel_mapping_rows)
    cpu_bundle_statistic_rows = summarize_cpu_bundles(cpu_bundle_rows)
    precision_rows, precision_diagnostics = build_precision_rows(gpu, npu, matches)
    fused_op_mapping_rows = build_fused_op_mapping_rows(gpu, npu)
    fused_op_statistic_rows = build_fused_op_statistic_rows(fused_op_mapping_rows)
    npu_gpu_statistic_rows = build_npu_gpu_op_statistic_rows(
        npu_kernel_mapping_rows,
        fused_op_mapping_rows,
    )
    coverage_diagnostics = build_coverage_diagnostics(
        gpu, npu, matches, npu_kernel_mapping_rows,
        threshold=diagnostic_coverage_threshold, precision_rows=precision_rows,
        cpu_bundle_rows=cpu_bundle_rows, fused_rows=fused_op_mapping_rows, op_map=op_map,
    )
    whole_call_rows = build_whole_calls(gpu, npu, matches, npu_kernel_mapping_rows,
                                      cpu_bundle_rows, fused_op_mapping_rows, op_map)
    whole_statistic_rows = summarize_whole_calls(whole_call_rows, gpu, npu)
    annotate_type_calls(npu_gpu_statistic_rows, npu_kernel_mapping_rows, whole_call_rows,
                        precision_rows, coverage_diagnostics)
    add_coverage_conclusions(npu_gpu_statistic_rows, coverage_diagnostics)
    for row in npu_gpu_statistic_rows:
        ids = {r.get('Manual Link ID') for r in npu_kernel_mapping_rows
               if (r['Device_id'], r['OP Type'], r['Core Type']) == (row['Device_id'], row['OP Type'], row['Core Type'])
               and r.get('Manual Link ID')}
        row['Manual Mapping Count'] = len(ids)
        if ids:
            row['排查结论'] = f'人工确认 {len(ids)} 组（非自动验证）；' + row.get('排查结论', '')
    npu_gpu_statistic_rows, gpu_unmapped_rows, gpu_accounting = complete_gpu_accounting(
        gpu, npu_kernel_mapping_rows, npu_gpu_statistic_rows,
        whole_call_rows=[*precision_rows, *cpu_bundle_rows, *fused_op_mapping_rows],
    )
    coverage_diagnostics["gpu_accounting"] = gpu_accounting
    coverage_diagnostics["gpu_unmapped_activities"] = gpu_unmapped_rows
    write_coverage_diagnostics(output, coverage_diagnostics)
    aten_rows = build_aten_statistic_rows(gpu, npu, matches, op_map)
    type_rows = build_kernel_type_statistic_rows(gpu, npu, matches)
    module_rows = build_module_statistic_rows(gpu, npu)
    region_rows = build_region_match_rows(gpu, npu, matches)
    kernel_inventory_rows = build_kernel_inventory_rows(gpu) + build_kernel_inventory_rows(npu)
    kernel_link_rows = build_kernel_link_rows(gpu) + build_kernel_link_rows(npu)
    unmatched_gpu_rows = build_unmatched_rows(gpu, unmatched_gpu)
    unmatched_npu_rows = build_unmatched_rows(npu, unmatched_npu)

    reports = {
        "cube_vector_statistic.csv": core_statistic_rows,
        "matches.csv": match_rows,
        "gpu_kernel_mapping.csv": mapping_rows,
        "npu_kernel_gpu_mapping.csv": npu_kernel_mapping_rows,
        "npu_gpu_op_statistic.csv": npu_gpu_statistic_rows,
        "fused_op_mapping.csv": fused_op_mapping_rows,
        "fused_op_statistic.csv": fused_op_statistic_rows,
        "aten_op_statistic.csv": aten_rows,
        "kernel_type_statistic.csv": type_rows,
        "module_statistic.csv": module_rows,
        "region_matches.csv": region_rows,
        "kernel_inventory_statistic.csv": kernel_inventory_rows,
        "kernel_link_diagnostic.csv": kernel_link_rows,
        "unmatched_gpu.csv": unmatched_gpu_rows,
        "unmatched_npu.csv": unmatched_npu_rows,
        "gpu_memory_activity.csv": memory_diagnostic_rows(gpu),
        "cpu_bundle_mapping.csv": cpu_bundle_rows,
        "cpu_bundle_statistic.csv": cpu_bundle_statistic_rows,
        "gpu_unmapped_activity.csv": gpu_unmapped_rows,
        "whole_call_mapping.csv": whole_call_rows,
        "whole_call_statistic.csv": whole_statistic_rows,
    }
    for filename, rows in reports.items():
        fieldnames = list(rows[0].keys()) if rows else {
            "matches.csv": ["Match ID", "Canonical OP"],
            "gpu_kernel_mapping.csv": ["Match ID", "GPU Kernel Name", "Mapped NPU Type"],
            "npu_kernel_gpu_mapping.csv": [
                "Device_id",
                "OP Type",
                "NPU Kernel Name",
                "Mapping Mode",
                "Mapped",
                "Direct Kernel Pair",
                "Comparable",
                "Accounted",
            ],
            "npu_gpu_op_statistic.csv": [
                "Device_id",
                "FUSED_OP_NPU_TYPE",
                "FUSED_OP_GPU_TYPE",
                "OP Type",
                "Core Type",
                "NPU Kernel Count",
                "Direct Mapping Coverage(%)",
                "Accounted Coverage(%)",
                "Comparable Coverage(%)",
                "Comparable Ratio(GPU/NPU)",
                "Total Time Ratio(GPU/NPU)",
            ],
            "fused_op_mapping.csv": ["Fused OP", "Occurrence"],
            "fused_op_statistic.csv": ["Fused OP", "Call Count"],
            "aten_op_statistic.csv": ["OP Type"],
            "kernel_type_statistic.csv": ["Mapped OP Type", "Core Type"],
            "module_statistic.csv": ["Module Key"],
            "region_matches.csv": ["Region ID", "Region Type"],
            "kernel_inventory_statistic.csv": ["Side", "Kernel Type", "Count"],
            "kernel_link_diagnostic.csv": ["Side", "Kernel Index", "Kernel Name"],
            "unmatched_gpu.csv": ["OP Index", "OP Name"],
            "unmatched_npu.csv": ["OP Index", "OP Name"],
            "gpu_memory_activity.csv": ["GPU Memory Activity Index", "Activity Kind", "Name"],
            "cpu_bundle_mapping.csv": ["CPU Bundle ID", "CPU Bundle"],
            "cpu_bundle_statistic.csv": ["CPU Bundle", "Call Count"],
            "whole_call_mapping.csv": ["Call ID", "Group ID", "Whole OP"],
            "whole_call_statistic.csv": ["Group ID", "Whole OP"],
            "gpu_unmapped_activity.csv": ["GPU Activity Kind", "GPU Activity Index", "GPU Name", "GPU Duration(us)",
                                          "GPU PID", "GPU TID", "GPU CPU OP Index", "GPU CPU OP Name",
                                          "GPU Link Method", "Status", "Whole Call References"],
        }[filename]
        _write_csv(output / filename, rows, fieldnames)

    # Keep the CSV complete and split the human-facing workbook by question.
    # Device_id only needs a visible column when several devices are present.
    device_columns = (
        ("Device_id",)
        if len({str(row["Device_id"]) for row in npu_gpu_statistic_rows if row["Row Kind"] == "npu_type"}) > 1
        else ()
    )
    identity_columns = device_columns + (
        "FUSED_OP_NPU_TYPE",
        "FUSED_OP_GPU_TYPE",
        "OP Type",
        "Core Type",
    )
    group_key_columns = (
        "Device_id",
        "FUSED_OP_NPU_TYPE",
        "FUSED_OP_GPU_TYPE",
    )
    merge_columns = ("FUSED_OP_NPU_TYPE", "FUSED_OP_GPU_TYPE")
    call_positions = {r["Call ID"]: i + 2 for i, r in enumerate(whole_call_rows)}
    group_positions = {r["Group ID"]: i + 2 for i, r in enumerate(whole_statistic_rows)}
    type_links = {}
    for i, row in enumerate(npu_gpu_statistic_rows):
        call_ids = json.loads(row.get("Whole Call IDs") or "[]")
        if call_ids:
            type_links[(i, "Related Whole OPs")] = f"'调用明细'!A{call_positions[call_ids[0]]}"
    whole_links = {(i, "Whole OP"): f"'调用明细'!A{call_positions[json.loads(r['Call IDs'])[0]]}"
                   for i, r in enumerate(whole_statistic_rows)}
    detail_links = {(i, "Whole OP"): f"'整体对比'!A{group_positions[r['Group ID']]}"
                    for i, r in enumerate(whole_call_rows)}
    display_type_rows = [dict(r) for r in npu_gpu_statistic_rows]
    for row in display_type_rows:
        if row.get("Row Kind") == "npu_type" and not row.get("Comparable NPU Kernel Count"):
            row["Comparable Ratio(GPU/NPU)"] = ""
            row["Total Time Ratio(GPU/NPU)"] = ""
    write_grouped_workbook_xlsx(
        output / "npu_gpu_op_statistic.xlsx",
        display_type_rows,
        [
            TableSheet(
                "类型对比",
                identity_columns
                + (
                    "NPU Kernel Count",
                    "NPU Total Time(us)",
                    "NPU Ratio(%)",
                    "Comparable Time Coverage(%)",
                    "Accounted Time Coverage(%)",
                    "Comparable NPU Time(us)",
                    "GPU Total Time(us)",
                    "GPU Activity Kind",
                    "Comparable Ratio(GPU/NPU)",
                    "Total Time Ratio(GPU/NPU)",
                    "GPU Kernel Count",
                    "排查结论",
                    "Whole Comparison Status",
                    "Related Whole OPs",
                ),
                group_key_columns,
                merge_columns,
                wrap_columns=("OP Type", "排查结论", "Related Whole OPs"), links=type_links,
            ),
            TableSheet(
                "整体对比",
                ("Whole OP", "Device_id", "NPU Types", "GPU Composition", "Comparable Call Count",
                 "Comparable NPU Time(us)", "Comparable GPU Time(us)", "Comparable Ratio(GPU/NPU)",
                 "Total Time Ratio(GPU/NPU)", "Comparison Status", "Population Status", "Scope",
                 "Observed Call Count", "Pending Call Count", "Overlapping Call Count",
                 "NPU Population Call Count", "GPU Population Call Count", "NPU Population Time(us)", "GPU Population Time(us)", "Group ID", "Call IDs"),
                rows=whole_statistic_rows, freeze_columns=1,
                wrap_columns=("NPU Types", "GPU Composition", "Population Status", "Scope"),
                hidden_columns=("Observed Call Count", "Pending Call Count", "Overlapping Call Count",
                                "NPU Population Call Count", "GPU Population Call Count", "NPU Population Time(us)", "GPU Population Time(us)", "Group ID", "Call IDs"),
                links=whole_links,
            ),
            TableSheet(
                "覆盖率诊断",
                identity_columns
                + (
                    "NPU Kernel Count",
                    "NPU Total Time(us)",
                    "Direct Mapped NPU Kernel Count",
                    "Direct Mapping Coverage(%)",
                    "Direct Time Coverage(%)",
                    "Comparable NPU Kernel Count",
                    "Comparable Coverage(%)",
                    "Comparable Time Coverage(%)",
                    "Accounted NPU Kernel Count",
                    "Accounted Coverage(%)",
                    "Accounted Time Coverage(%)",
                    "Unmapped NPU Kernel Count",
                    "Mapping Mode Counts",
                    "CPU Alignment Status",
                    "Kernel Comparison Status",
                    "Whole Comparison Status",
                    "Related Whole OPs",
                    "排查结论",
                    "排查建议",
                ),
                group_key_columns,
                merge_columns,
                wrap_columns=("OP Type", "排查结论", "排查建议", "CPU Alignment Status", "Kernel Comparison Status"),
                links=type_links,
            ),
            TableSheet(
                "调用明细",
                ("Call ID", "Whole OP", "NPU Types", "GPU Composition", "CPU Alignment Status",
                 "Kernel Comparison Status", "NPU Total Time(us)", "GPU Kernel Time(us)", "GPU Memory Time(us)",
                 "GPU Total Time(us)", "Comparable Ratio(GPU/NPU)", "Issues", "Overlap Call IDs",
                 "NPU CPU OP Index", "GPU CPU OP Index", "NPU Kernel Indices", "GPU Kernel Indices",
                 "GPU Memory Activity Indices", "NPU Kernel Names", "GPU Kernel Names", "GPU Memory Activity Names",
                 "Alignment Basis", "Source", "Scope", "Group ID"), rows=whole_call_rows,
                freeze_columns=2, links=detail_links,
                wrap_columns=("NPU Types", "GPU Composition", "Issues", "Scope"),
                hidden_columns=("NPU Kernel Names", "GPU Kernel Names", "GPU Memory Activity Names", "Group ID"),
            ),
            TableSheet(
                "耗时分布",
                identity_columns
                + (
                    "NPU Kernel Count",
                    "GPU Kernel Count",
                    "GPU Memory Activity Count",
                    "GPU Kernel Time(us)",
                    "GPU Memory Time(us)",
                    "Mapped CPU OP Calls",
                    "NPU Min Time(us)",
                    "NPU Avg Time(us)",
                    "NPU Max Time(us)",
                    "GPU Min Time(us)",
                    "GPU Avg Time(us)",
                    "GPU Max Time(us)",
                    "Diff Time(GPU-NPU)(us)",
                ),
                group_key_columns,
                merge_columns,
                wrap_columns=("OP Type",),
            ),
            TableSheet(
                "Cube-Vector总览", tuple(core_statistic_rows[0]), rows=core_statistic_rows,
                freeze_columns=2, wrap_columns=("Side", "Scope"),
            ),
        ],
    )

    matched_gpu_duration = sum(
        bundle_metrics(_op_kernels(gpu, match.gpu_op_index)).sum_us for match in matches
    )
    matched_npu_duration = sum(
        bundle_metrics(_op_kernels(npu, match.npu_op_index)).sum_us for match in matches
    )
    trusted_matches = [match for match in matches if _is_trusted(match)]
    performance_matches = [
        match for match in matches if _is_performance_match(gpu, npu, match)
    ]
    trusted_gpu_duration = sum(
        bundle_metrics(_op_kernels(gpu, match.gpu_op_index)).sum_us
        for match in trusted_matches
    )
    trusted_npu_duration = sum(
        bundle_metrics(_op_kernels(npu, match.npu_op_index)).sum_us
        for match in trusted_matches
    )
    performance_gpu_duration = sum(
        bundle_metrics(_op_kernels(gpu, match.gpu_op_index)).sum_us
        for match in performance_matches
    )
    performance_npu_duration = sum(
        bundle_metrics(_op_kernels(npu, match.npu_op_index)).sum_us
        for match in performance_matches
    )
    gpu_total_duration = sum(kernel.event.duration_us for kernel in gpu.kernels)
    npu_total_duration = sum(kernel.event.duration_us for kernel in npu.kernels)
    confidence_counts = Counter(match.confidence for match in matches)
    alignment_basis_counts = Counter(match.alignment_basis for match in matches)
    both_empty_count = sum(
        not gpu.ops[match.gpu_op_index].kernel_indices
        and not npu.ops[match.npu_op_index].kernel_indices
        for match in matches
    )
    eligible_region_rows = [
        row for row in region_rows if row["Eligible For Performance"]
    ]
    npu_first_mapped_rows = [
        row for row in npu_kernel_mapping_rows if row["Mapped"]
    ]
    npu_first_direct_rows = [
        row for row in npu_kernel_mapping_rows if row["Direct Kernel Pair"]
    ]
    npu_first_accounted_rows = [
        row for row in npu_kernel_mapping_rows if row["Accounted"]
    ]
    npu_first_gpu_indices = [
        int(index)
        for row in npu_first_mapped_rows
        for index in json.loads(str(row["GPU Kernel Indices"]))
    ]
    npu_first_gpu_users: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for row in npu_first_mapped_rows:
        for index in json.loads(str(row["GPU Kernel Indices"])):
            npu_first_gpu_users[int(index)].append(row)
    shared_gpu_assignment_count = 0
    duplicate_gpu_assignment_count = 0
    for users in npu_first_gpu_users.values():
        if len(users) <= 1:
            continue
        is_intentional_shared_bundle = (
            all(row["Mapping Mode"] in _SHARED_GPU_BUNDLE_MODES for row in users)
            and len({
                (str(row.get('Manual Link ID') or row["CPU Match ID"]), str(row["OP Type"]), str(row["Core Type"]))
                for row in users
            }) == 1
        )
        if is_intentional_shared_bundle:
            shared_gpu_assignment_count += len(users) - 1
        else:
            duplicate_gpu_assignment_count += len(users) - 1
    npu_first_total_time = sum(
        float(row["NPU Duration(us)"]) for row in npu_kernel_mapping_rows
    )
    npu_first_mapped_time = sum(
        float(row["NPU Duration(us)"]) for row in npu_first_mapped_rows
    )
    npu_first_direct_time = sum(
        float(row["NPU Duration(us)"]) for row in npu_first_direct_rows
    )
    npu_first_accounted_time = sum(
        float(row["NPU Duration(us)"]) for row in npu_first_accounted_rows
    )
    summary = {
        "gpu": _profile_summary(gpu),
        "npu": _profile_summary(npu),
        "comparison": {
            "matched_op_count": len(matches),
            "unmatched_gpu_op_count": len(unmatched_gpu),
            "unmatched_npu_op_count": len(unmatched_npu),
            "confidence_counts": dict(confidence_counts),
            "alignment_basis_counts": dict(alignment_basis_counts),
            "trusted_matched_op_count": len(trusted_matches),
            "performance_matched_op_count": len(performance_matches),
            "both_kernel_empty_match_count": both_empty_count,
            "gpu_matched_kernel_duration_coverage": (
                matched_gpu_duration / gpu_total_duration if gpu_total_duration else 0.0
            ),
            "npu_matched_kernel_duration_coverage": (
                matched_npu_duration / npu_total_duration if npu_total_duration else 0.0
            ),
            "gpu_trusted_kernel_duration_coverage": (
                trusted_gpu_duration / gpu_total_duration if gpu_total_duration else 0.0
            ),
            "npu_trusted_kernel_duration_coverage": (
                trusted_npu_duration / npu_total_duration if npu_total_duration else 0.0
            ),
            "gpu_performance_kernel_duration_coverage": (
                performance_gpu_duration / gpu_total_duration if gpu_total_duration else 0.0
            ),
            "npu_performance_kernel_duration_coverage": (
                performance_npu_duration / npu_total_duration if npu_total_duration else 0.0
            ),
            "region_count": len(region_rows),
            "performance_region_count": len(eligible_region_rows),
            "performance_region_gpu_kernel_time_us": _round(
                sum(float(row["GPU Kernel Sum(us)"]) for row in eligible_region_rows)
            ),
            "performance_region_npu_kernel_time_us": _round(
                sum(float(row["NPU Kernel Sum(us)"]) for row in eligible_region_rows)
            ),
            "npu_first_kernel_count": len(npu_kernel_mapping_rows),
            "npu_first_mapped_kernel_count": len(npu_first_mapped_rows),
            "npu_first_direct_kernel_count": len(npu_first_direct_rows),
            "npu_first_accounted_kernel_count": len(npu_first_accounted_rows),
            "npu_first_comparable_kernel_count": len(npu_first_mapped_rows),
            "npu_first_direct_kernel_count_coverage": (
                len(npu_first_direct_rows) / len(npu_kernel_mapping_rows)
                if npu_kernel_mapping_rows
                else 0.0
            ),
            "npu_first_accounted_kernel_count_coverage": (
                len(npu_first_accounted_rows) / len(npu_kernel_mapping_rows)
                if npu_kernel_mapping_rows
                else 0.0
            ),
            "npu_first_comparable_kernel_count_coverage": (
                len(npu_first_mapped_rows) / len(npu_kernel_mapping_rows)
                if npu_kernel_mapping_rows
                else 0.0
            ),
            "npu_first_kernel_count_coverage": (
                len(npu_first_mapped_rows) / len(npu_kernel_mapping_rows)
                if npu_kernel_mapping_rows
                else 0.0
            ),
            "npu_first_kernel_duration_coverage": (
                npu_first_mapped_time / npu_first_total_time
                if npu_first_total_time
                else 0.0
            ),
            "npu_first_direct_kernel_duration_coverage": (
                npu_first_direct_time / npu_first_total_time
                if npu_first_total_time
                else 0.0
            ),
            "npu_first_accounted_kernel_duration_coverage": (
                npu_first_accounted_time / npu_first_total_time
                if npu_first_total_time
                else 0.0
            ),
            "npu_first_comparable_kernel_duration_coverage": (
                npu_first_mapped_time / npu_first_total_time
                if npu_first_total_time
                else 0.0
            ),
            "npu_first_gpu_kernel_assignment_count": len(npu_first_gpu_indices),
            "npu_first_unique_gpu_kernel_count": len(set(npu_first_gpu_indices)),
            "npu_first_shared_gpu_bundle_assignment_count": shared_gpu_assignment_count,
            "npu_first_duplicate_gpu_assignment_count": duplicate_gpu_assignment_count,
            "fused_op_occurrence_count": len(fused_op_mapping_rows),
            "cpu_bundle_occurrence_count": len(cpu_bundle_rows),
            "gpu_memory_activity_count": len(gpu.memory_activities),
            "npu_first_unique_gpu_memory_activity_count": len({
                i for row in npu_kernel_mapping_rows
                for i in json.loads(row["GPU Memory Activity Indices"])
            }),
            "fused_op_gpu_kernel_time_us": _round(
                sum(float(row["GPU Kernel Sum(us)"]) for row in fused_op_mapping_rows)
            ),
            "fused_op_npu_kernel_time_us": _round(
                sum(float(row["NPU Kernel Sum(us)"]) for row in fused_op_mapping_rows)
            ),
        },
        "gpu_accounting": gpu_accounting,
        "core_breakdown": core_breakdown,
        "precision_analysis": {
            **precision_diagnostics,
            "pattern_counts": dict(Counter(row["Pattern"] for row in precision_rows)),
            "candidate_cause_counts": dict(Counter(row["Candidate Cause"] for row in precision_rows)),
            "cast_time_us": _round(sum(row["Cast Time(us)"] for row in precision_rows)),
            "cause_status": "not_verified; ordered dtype/shape patterns do not prove tensor dataflow or backend defects",
        },
        "coverage_diagnostics": {
            **coverage_diagnostics["summary"], "threshold_pct": diagnostic_coverage_threshold,
        },
        "reports": sorted([*reports, "npu_gpu_op_statistic.xlsx",
                           "coverage_diagnostics.json", "coverage_diagnostics.md"]),
        "time_semantics": {
            "kernel_sum_us": "sum of kernel durations; concurrent streams may be double-counted",
            "device_busy_us": "union of kernel intervals; concurrent execution is counted once",
            "bundle_span_us": "last kernel end minus first kernel start; includes gaps",
            "npu_first_gpu_total_us": "compute kernel time plus matched device memcpy time; runtime API duration is excluded",
            "total_row": "TOTAL uses every selected GPU kernel and memory activity once, including unmapped work; it is not an extra additive row or an equivalence claim",
            "row_kind": "npu_type / gpu_unmapped_kernel / gpu_unmapped_memory are detail rows; total is a non-additive independent reconciliation",
            "cpu_bundle_view": "alternative whole-call scope; never add to per-type or fused-op report totals",
            "precision_path_view": "alternative complete CPU-call scope including Cast and GPU memory activities; non-additive with other reports; ratios use only comparable calls on both sides",
            "precision_cast_time_us": "observed conversion kernel cost, not predicted speedup; GPU may convert internally without a standalone Cast",
        },
    }
    from .operator_details import build_operator_details
    from .pipeline_analysis import build_pipeline_report
    from .provenance import write_json
    pipeline_report = build_pipeline_report(npu, npu_kernel_mapping_rows)
    write_json(output / 'npu_pipeline_analysis.json', pipeline_report)
    pipeline_rows = [dict(index=k['index'], name=k['name'], op_type=k['op_type'], device_id=k['device_id'],
                          core_type=k['core_type'], time_us=k['time_us'], category=k['pipeline']['category'],
                          conclusion=k['pipeline']['label'], reason=k['pipeline']['reason'],
                          confidence=k['pipeline']['confidence'], next_check=k['pipeline']['next_check'])
                     for k in pipeline_report['kernels']]
    _write_csv(output / 'npu_pipeline_kernels.csv', pipeline_rows,
               ['index', 'name', 'op_type', 'device_id', 'core_type', 'time_us', 'category',
                'conclusion', 'reason', 'confidence', 'next_check'])
    summary['reports'].extend(['npu_pipeline_analysis.json', 'npu_pipeline_kernels.csv'])
    summary['pipeline_analysis'] = pipeline_report['summary']
    write_json(output / 'operator_details.json', build_operator_details(gpu, npu, matches, npu_kernel_mapping_rows))
    summary['reports'].append('operator_details.json')
    if manual_enabled:
        from .manual_mapping import catalog
        from .provenance import write_json
        write_json(output / 'manual_catalog.json', catalog(gpu, npu, matches, npu_kernel_mapping_rows))
        summary['reports'].append('manual_catalog.json')
    if manual_enabled or manual_links:
        from .provenance import write_json
        write_json(output / 'manual_links.json', list(manual_links))
        summary['reports'].append('manual_links.json')
        summary['manual_link_count'] = len(manual_links)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary

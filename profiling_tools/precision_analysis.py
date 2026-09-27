"""Conservative, read-only precision-path diagnostics for complete CPU calls.

These rows are an alternative view, not additional kernel assignments. Ordered
dtype/shape agreement establishes a *pattern*, never tensor dataflow or a proven
backend defect. In particular, no visible CUDA Cast does not imply FP16 opmath.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re

from .device_compare import scoped_indices
from .parser import normalize_input_dtypes, normalize_input_shapes


@dataclass(frozen=True)
class PrecisionRule:
    name: str
    cpu_ops: tuple[str, ...]
    api_prefix: str
    main_types: tuple[str, ...]


# Restrict rules to known API families. Divs may lower scalar division to Muls.
RULES = (
    PrecisionRule("divs_v1", ("aten::div", "aten::div_", "aten::true_divide"),
                  "aclnnDivs_", ("Muls", "RealDiv", "Div")),
    PrecisionRule("subs_v1", ("aten::sub", "aten::sub_", "aten::subtract"),
                  "aclnnSubs_", ("Sub",)),
)
_CONVERSIONS = {"aten::to", "aten::_to_copy", "aten::copy_", "aten::type_as"}
_STRONG_LINKS = {"external_id", "runtime_correlation", "torch_to_npu",
                 "host_to_device_queue", "host_to_device_thread"}
_LOW_PRECISION = {"float16", "bfloat16"}
_SHORT = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16"}

SUMMARY_FIELDS = (
    "CPU OP", "NPU API", "Pattern", "NPU Path", "Call Count",
    "Cast Kernel Count", "Cast Time(us)", "Cast Time Share(%)",
    "NPU Total Time(us)", "Comparable Call Count", "Comparable NPU Time(us)",
    "GPU Total Time(us)", "Comparable Ratio(GPU/NPU)", "Candidate Cause",
)
DETAIL_VIEW_FIELDS = (
    "Precision Path ID", "CPU OP", "Pattern", "NPU Path", "Cast Time(us)",
    "Cast Time Share(%)", "NPU Total Time(us)", "GPU Total Time(us)",
    "Comparable Ratio(GPU/NPU)", "Comparison Status", "GPU Cast Evidence",
    "Candidate Cause", "NPU CPU OP Index", "GPU CPU OP Index", "CPU Match Confidence",
    "Pattern Issues", "NPU Input Shapes", "NPU Input DTypes",
    "NPU Kernel Indices", "GPU Kernel Indices", "GPU Memory Activity Indices",
)


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _dtypes(value):
    # c10::Half appears in CPU trace metadata; keep parser behaviour unchanged.
    value = str(value).replace("c10::", "").replace("at::", "")
    return json.loads(normalize_input_dtypes(value) or "[]")


def _shapes(value):
    if str(value).strip().strip('"').lower() in {"", "n/a", "na", "none", "null", "unknown", "-"}:
        return []
    return json.loads(normalize_input_shapes(value) or "[]")


def _decimal(value, fallback):
    try:
        result = Decimal(str(value).strip())
        if result.is_finite():
            return result
    except InvalidOperation:
        pass
    return Decimal(str(fallback))


def _start(kernel):
    detail = kernel.detail
    return _decimal(detail.raw.get("Start Time(us)", "") if detail else "",
                    kernel.event.timestamp_us)


def _duration(kernel):
    return _decimal(kernel.detail.raw.get("Duration(us)", "") if kernel.detail else "",
                    kernel.event.duration_us)


def _parents(profile, index):
    while index is not None:
        yield index
        index = profile.ops[index].parent_index


def _node(kernel, index):
    detail = kernel.detail
    return {
        "kernel_index": index, "name": detail.name if detail else kernel.event.name,
        "type": detail.op_type if detail else "unknown",
        "input_dtypes": _dtypes(detail.input_dtypes) if detail else [],
        "output_dtypes": _dtypes(detail.output_dtypes) if detail else [],
        "input_shapes": _shapes(detail.input_shapes) if detail else [],
        "output_shapes": _shapes(detail.output_shapes) if detail else [],
        "stream_id": detail.stream_id or kernel.stream_id if detail else kernel.stream_id,
        "device_id": detail.device_id if detail else "",
        "task_id": detail.task_id if detail else kernel.task_id,
        # Exact strings retain sub-microsecond order at epoch-scale timestamps.
        "start_us": str(_start(kernel)), "duration_us": float(_duration(kernel)),
        "owner_op_index": kernel.owner_op_index, "link_method": kernel.link_method,
    }


def _path(nodes):
    def dtype_text(types):
        return "+".join(_SHORT.get(t, t) for t in dict.fromkeys(types)) or "?"
    return " -> ".join(
        n["type"] + "(" + dtype_text(n["input_dtypes"]) + ">"
        + dtype_text(n["output_dtypes"]) + ")" for n in nodes
    )


def _classify(nodes, rule):
    issues = []
    if any(not n["name"].lower().startswith(rule.api_prefix.lower()) for n in nodes):
        issues.append("mixed_or_missing_api")
    mains = [i for i, n in enumerate(nodes) if n["type"] in rule.main_types]
    if len(mains) != 1 or any(n["type"] not in (*rule.main_types, "Cast") for n in nodes):
        issues.append("non_unique_main_or_other_kernel")
    streams = {(n["device_id"], n["stream_id"]) for n in nodes}
    if len(streams) != 1 or any(not n["stream_id"] for n in nodes):
        issues.append("multiple_or_unknown_streams")
    if any(Decimal(b["start_us"]) < Decimal(a["start_us"]) + Decimal(str(a["duration_us"]))
           - Decimal("0.01") for a, b in zip(nodes, nodes[1:])):
        issues.append("overlapping_kernel_intervals")
    if any(n["duration_us"] < 0 for n in nodes):
        issues.append("negative_kernel_duration")
    if issues:
        return "ambiguous_path", issues
    for n in nodes:
        if (not n["input_dtypes"] or not n["output_dtypes"]
                or any(t not in {"float16", "float32", "bfloat16", "float64"}
                       for t in n["input_dtypes"] + n["output_dtypes"])
                or len(n["input_shapes"]) != len(n["input_dtypes"])
                or len(n["output_shapes"]) != len(n["output_dtypes"])):
            return "incomplete_metadata", ["missing_or_unsupported_dtype_shape"]
    main_index = mains[0]
    main = nodes[main_index]
    before, after = nodes[:main_index], nodes[main_index + 1:]
    if not before and not after:
        types = set(main["input_dtypes"] + main["output_dtypes"])
        return ("native_" + _SHORT[next(iter(types))], []) if len(types) == 1 and types <= set(_SHORT) else (
            "other_precision_path", ["mixed_native_dtypes"])
    if (any(len(n["input_dtypes"]) != 1 or len(n["output_dtypes"]) != 1
            or n["input_shapes"] != n["output_shapes"] for n in before + after)
            or len(after) > 1):
        return "ambiguous_path", ["cast_not_shape_preserving_or_multiple_outputs"]
    # Dtype + matching shape are necessary evidence, not proof of tensor identity.
    main_inputs = list(zip(main["input_dtypes"], main["input_shapes"]))
    main_outputs = list(zip(main["output_dtypes"], main["output_shapes"]))
    if any((n["output_dtypes"][0], n["output_shapes"][0]) not in main_inputs for n in before):
        issues.append("pre_cast_main_metadata_mismatch")
    if any((n["input_dtypes"][0], n["input_shapes"][0]) not in main_outputs for n in after):
        issues.append("post_cast_main_metadata_mismatch")
    if issues:
        return "ambiguous_path", issues
    promoted = (bool(before) and all(n["input_dtypes"][0] in _LOW_PRECISION
                                    and n["output_dtypes"] == ["float32"] for n in before)
                and set(main["input_dtypes"] + main["output_dtypes"]) == {"float32"})
    if promoted and not after:
        return "input_promotion", []
    if promoted and len(after) == 1 and all(
            n["input_dtypes"] == after[0]["output_dtypes"] for n in before):
        return "round_trip_promotion", []
    if not before and after:
        return "output_conversion", []
    return "other_precision_path", ["conversion_not_recognized"]


def _gpu_cast_evidence(gpu, indices, root):
    if not indices:
        return "unknown"
    for i in indices:
        kernel = gpu.kernels[i]
        for owner in _parents(gpu, kernel.owner_op_index):
            if owner == root:
                break
            if gpu.ops[owner].canonical_name in _CONVERSIONS:
                return "conversion_child_observed"
        # Do not match 'cast' inside broadcast or claim templated opmath is a Cast.
        if re.search(r"(?:^|::|\b)(?:cast|cast_kernel)(?:\b|<)", kernel.event.name, re.I):
            return "cast_kernel_name_observed"
    return "no_standalone_cast_observed"


def build_precision_rows(gpu, npu, matches):
    """Return disjoint NPU call rows and scan diagnostics without mutating profiles."""
    nk_scope, gk_scope, gm_scope = scoped_indices(npu), scoped_indices(gpu), scoped_indices(gpu, True)
    by_npu = defaultdict(list)
    for match in matches:
        by_npu[match.npu_op_index].append(match)
    candidates = {}
    for i, op in enumerate(npu.ops):
        for rule in RULES:
            if op.canonical_name not in rule.cpu_ops:
                continue
            mains = {k for k in nk_scope[i] if npu.kernels[k].detail
                     and npu.kernels[k].detail.name.lower().startswith(rule.api_prefix.lower())
                     and npu.kernels[k].detail.op_type in rule.main_types}
            if mains:
                candidates[i] = (rule, mains)
    # Collapse same-family overload wrappers with the same primary work. Keep
    # the wider complete scope, including casts owned by the outer CPU op.
    roots = [i for i, (rule, mains) in candidates.items() if not any(
        parent in candidates and candidates[parent] == (rule, mains)
        for parent in list(_parents(npu, i))[1:])]
    roots.sort(key=lambda i: (-len(list(_parents(npu, i))), i))
    rows, used_npu, used_gpu, used_memory = [], set(), set(), set()
    diagnostics = Counter()
    for index in roots:
        rule, _ = candidates[index]
        nk = sorted(set(nk_scope[index]), key=lambda k: (_start(npu.kernels[k]), k))
        if used_npu.intersection(nk):
            diagnostics["overlapping_npu_scope_skipped"] += 1
            continue
        used_npu.update(nk)
        nodes = [_node(npu.kernels[k], k) for k in nk]
        pattern, issues = _classify(nodes, rule)
        cpu = npu.ops[index]
        cpu_shapes, cpu_types = _shapes(cpu.input_shapes), _dtypes(cpu.input_dtypes)
        if cpu_types and nodes[0]["input_dtypes"] and cpu_types[0] != nodes[0]["input_dtypes"][0]:
            issues.append("cpu_npu_storage_dtype_mismatch")
        if cpu_shapes and nodes[0]["input_shapes"] and cpu_shapes[0] != nodes[0]["input_shapes"][0]:
            issues.append("cpu_npu_input_shape_mismatch")
        candidate_matches = by_npu[index]
        match = candidate_matches[0] if len(candidate_matches) == 1 else None
        gpu_op = gpu.ops[match.gpu_op_index] if match else None
        gk = sorted(set(gk_scope[match.gpu_op_index])) if match else []
        gm = sorted(set(gm_scope[match.gpu_op_index])) if match else []
        reasons = []
        if not match:
            reasons.append("missing_or_ambiguous_cpu_match")
        else:
            if match.confidence not in {"high", "medium"}:
                reasons.append("untrusted_cpu_match")
            if gpu_op.canonical_name not in rule.cpu_ops:
                reasons.append("cpu_semantic_mismatch")
            if (not cpu_shapes or not _shapes(gpu_op.input_shapes) or not cpu_types
                    or not _dtypes(gpu_op.input_dtypes)):
                reasons.append("missing_cpu_metadata")
            elif (_shapes(cpu.input_shapes) != _shapes(gpu_op.input_shapes)
                  or _dtypes(cpu.input_dtypes) != _dtypes(gpu_op.input_dtypes)):
                reasons.append("cpu_metadata_mismatch")
            if cpu.module_key and gpu_op.module_key and cpu.module_key != gpu_op.module_key:
                reasons.append("module_mismatch")
        if issues:
            reasons.append("unresolved_npu_path")
        if any(npu.kernels[k].link_method not in _STRONG_LINKS for k in nk):
            reasons.append("weak_npu_owner_evidence")
        if not gk:
            reasons.append("missing_gpu_compute")
        if any(r.link_method not in _STRONG_LINKS for r in
               [gpu.kernels[k] for k in gk] + [gpu.memory_activities[k] for k in gm]):
            reasons.append("weak_gpu_owner_evidence")
        if used_gpu.intersection(gk) or used_memory.intersection(gm):
            reasons.append("gpu_scope_already_used")
        comparable = not reasons
        if comparable:
            used_gpu.update(gk)
            used_memory.update(gm)
        cast_time = sum(n["duration_us"] for n in nodes if n["type"] == "Cast")
        total = sum(n["duration_us"] for n in nodes)
        gpu_time = sum(gpu.kernels[k].event.duration_us for k in gk) if comparable else None
        memory_time = sum(gpu.memory_activities[k].event.duration_us for k in gm) if comparable else None
        gpu_total = gpu_time + memory_time if comparable else None
        evidence = _gpu_cast_evidence(gpu, gk, match.gpu_op_index) if match else "unknown"
        cause = ("possible_unfused_conversion" if comparable and pattern in {
            "round_trip_promotion", "input_promotion"} and evidence == "no_standalone_cast_observed"
            else "no_external_cast" if not any(n["type"] == "Cast" for n in nodes)
            else "requires_validation")
        rows.append({
            "Precision Path ID": len(rows) + 1, "CPU OP": cpu.canonical_name,
            "NPU API": rule.api_prefix.rstrip("_"), "Rule": rule.name,
            "Pattern": pattern, "NPU Path": _path(nodes), "Candidate Cause": cause,
            "Evidence Level": "ordered_metadata_pattern", "Cause Status": "not_verified",
            "NPU CPU OP Index": index, "GPU CPU OP Index": match.gpu_op_index if match else "",
            "CPU Match ID": match.match_id if match else "",
            "CPU Match Confidence": match.confidence if match else "",
            "CPU Alignment Basis": match.alignment_basis if match else "",
            "Pattern Issues": _json(issues), "Comparison Status": ";".join(reasons) or "comparable",
            "Comparable": comparable, "GPU Cast Evidence": evidence,
            "NPU Input Shapes": cpu.input_shapes, "NPU Input DTypes": cpu.input_dtypes,
            "GPU Input Shapes": gpu_op.input_shapes if gpu_op else "",
            "GPU Input DTypes": gpu_op.input_dtypes if gpu_op else "",
            "NPU Kernel Count": len(nk), "Cast Kernel Count": sum(n["type"] == "Cast" for n in nodes),
            "Cast Time(us)": round(cast_time, 6), "Non-Cast Time(us)": round(total - cast_time, 6),
            "Cast Time Share(%)": round(100 * cast_time / total, 6) if total else "",
            "NPU Total Time(us)": round(total, 6),
            "Comparable NPU Time(us)": round(total, 6) if comparable else "",
            "GPU Kernel Time(us)": round(gpu_time, 6) if comparable else "",
            "GPU Memory Time(us)": round(memory_time, 6) if comparable else "",
            "GPU Total Time(us)": round(gpu_total, 6) if comparable else "",
            "Comparable Ratio(GPU/NPU)": round(gpu_total / total, 6) if comparable and total else "",
            "NPU Kernel Indices": _json(nk), "NPU Kernel Details": _json(nodes),
            "GPU Kernel Indices": _json(gk if comparable else []),
            "GPU Memory Activity Indices": _json(gm if comparable else []),
            "GPU Candidate Kernel Indices": _json(gk),
            "GPU Candidate Memory Activity Indices": _json(gm),
            "GPU Kernel Names": _json([gpu.kernels[k].event.name for k in gk]),
            "GPU Memory Activity Names": _json([gpu.memory_activities[k].event.name for k in gm]),
        })
    supported = {k for k, kernel in enumerate(npu.kernels) if kernel.detail and any(
        kernel.detail.name.lower().startswith(r.api_prefix.lower()) for r in RULES)}
    diagnostics.update({
        "supported_api_kernel_count": len(supported),
        "unscoped_api_kernel_count": len(supported - used_npu),
        "call_count": len(rows), "comparable_call_count": sum(r["Comparable"] for r in rows),
        "unique_gpu_kernel_count": len(used_gpu), "unique_gpu_memory_activity_count": len(used_memory),
    })
    return rows, dict(diagnostics)


def summarize_precision_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in ("CPU OP", "NPU API", "Pattern", "NPU Path", "Candidate Cause"))].append(row)
    result = []
    for key, group in groups.items():
        comparable = [r for r in group if r["Comparable"]]
        cast = sum(r["Cast Time(us)"] for r in group)
        total = sum(r["NPU Total Time(us)"] for r in group)
        npu_time = sum(r["NPU Total Time(us)"] for r in comparable)
        gpu_time = sum(r["GPU Total Time(us)"] for r in comparable)
        result.append(dict(zip(SUMMARY_FIELDS, (
            *key[:4], len(group), sum(r["Cast Kernel Count"] for r in group), round(cast, 6),
            round(100 * cast / total, 6) if total else "", round(total, 6), len(comparable),
            round(npu_time, 6) if comparable else "", round(gpu_time, 6) if comparable else "",
            round(gpu_time / npu_time, 6) if npu_time else "", key[4],
        ))))
    return sorted(result, key=lambda r: (-r["Cast Time(us)"], r["CPU OP"], r["NPU Path"]))

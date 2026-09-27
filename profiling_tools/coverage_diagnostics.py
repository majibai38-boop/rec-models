"""Coverage-driven triage for every NPU kernel type, without changing mappings.

Selection and evidence collection are independent of operator names. Specialized
analyses (precision paths, CPU bundles, fused calls) are optional evidence sources,
not the entry point. Each non-comparable kernel gets one accounting reason plus
secondary observations. Recommendations are hypotheses to verify, not fixes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import re

from .device_compare import scoped_indices
from .parser import normalize_input_dtypes, normalize_input_shapes


# No NPU Type or ACLNN API whitelist: these rules describe missing evidence or
# comparison structure and work for previously unseen kernel/operator names.
ADVICE = {
    "whole_call_available": (
        "comparison_granularity", "已有整体比较，缺少独立 kernel 分摊",
        "对应工作已纳入整体调用视图，逐类型低覆盖率不等于漏掉整次调用。",
        ["查看引用的整体调用记录，比较包含辅助 kernel 的完整 device 时间。",
         "如要分析后端额外开销，核对整条路径的源码与最小复现；不要把同一 GPU kernel 时间再次分摊给辅助类型。"],
    ),
    "npu_owner_missing": (
        "mapping_or_capture_gap", "缺少 NPU kernel → CPU 关联",
        "尚不能定位这条 NPU 工作的逻辑调用，不能据此判断 GPU 是否有对应实现。",
        ["核对 kernel 的 Device/Stream/Task ID 与 trace flow 终点，检查是否缺少 torch_to_npu、队列或运行时关联事件。",
         "确认两端采集的 step、rank、CPU PID 和自定义区域一致；补充共同的 record_function 标记后重新采集，勿按相邻时间强行归属。"],
    ),
    "npu_owner_weak": (
        "insufficient_evidence", "NPU CPU 归属依赖回退证据",
        "时间或类型顺序回退可能把 kernel 归给邻近的 helper 调用。",
        ["在 kernel_link_diagnostic.csv 中对照原始和当前 owner、Link Method/Scope/Distance。",
         "优先恢复唯一 Task/Stream、External ID 或队列关联，再判断 CPU 匹配；不要先放宽性能比较条件。"],
    ),
    "api_cpu_conflict": (
        "mapping_or_capture_gap", "API 与 CPU 算子语义冲突",
        "当前 CPU owner 与已有 API 主类型约束不一致。",
        ["检查 API Expected CPU OPs 与实际 owner，沿原始 flow/队列回溯真实下发调用。",
         "区分 API 的主 kernel 和准备 kernel；只在有调用实例证据时修正归属，不能仅按全局数量重排。"],
    ),
    "cpu_match_ambiguous": (
        "insufficient_evidence", "CPU 匹配不唯一",
        "同一 NPU 调用存在多个候选匹配，不能选择首个候选充当证据。",
        ["比较候选调用的完整输入元数据、模块、父调用与局部顺序。",
         "增加共同语义区域或调用标记，消除歧义后再恢复映射。"],
    ),
    "cpu_metadata_conflict": (
        "mapping_candidate", "两端 CPU 元数据有差异",
        "差异可能来自输入/overload 不同，也可能来自后端元数据表达方式；需要先区分。",
        ["逐字段检查报告中的 shape/dtype/module/name 差异，区分 Tensor、标量、零维 Tensor、可选参数和列表编码。",
         "核对两端实际输入、autocast 和调用源码；仅修正规范化等价项，不应把真实 dtype/shape 冲突强制配对。"],
    ),
    "cpu_match_untrusted": (
        "insufficient_evidence", "CPU 匹配置信度不足",
        "现有匹配只能提供排查候选，不足以支撑性能比较。",
        ["检查 CPU Alignment Basis、完整 shape/dtype、父调用及模块边界。",
         "补充缺失元数据或同名区域标记，重新执行局部对齐；不要仅放宽置信度门槛以增加覆盖率。"],
    ),
    "cpu_alignment_gap": (
        "mapping_candidate", "存在同模块同元数据候选，但未对齐",
        "可能是嵌套 overload、执行顺序变化或局部序列对齐规则未覆盖；候选不等于已匹配。",
        ["检查 JSON 中的候选 GPU CPU op 和最近已匹配祖先，核对父子关系、调用数和局部语义锚点。",
         "只有调用实例关系唯一且元数据一致时才增加恢复规则，并补充重复调用和顺序变化的回归测试。"],
    ),
    "cpu_match_missing": (
        "mapping_or_capture_gap", "缺少可用的跨后端 CPU 匹配",
        "可能存在框架别名、自定义作用域、输入差异或采集范围差异，当前证据不能确定是哪一种。",
        ["对照同模块候选、原始调用栈和最近已匹配祖先，检查两端算子是否落在不同 wrapper/自定义区域。",
         "确认 step/rank/输入一致，补充双端共同标记或有源码依据的 op-map；不可按同名 kernel 数量硬配。"],
    ),
    "gpu_memory_only": (
        "backend_difference_candidate", "GPU 对应调用仅有设备内存活动",
        "NPU 计算任务与 GPU memcpy/memset 的表示粒度可能不同。",
        ["检查 GPU 内存活动的方向、字节数、CPU owner 与输入输出 dtype/shape，确认工作量是否等价。",
         "等价后再评估设备活动级比较；计算 kernel 与 memcpy/memset 必须保持独立编号，不能使用 CPU runtime 时长代替 device 时间。"],
    ),
    "gpu_work_absent": (
        "mapping_or_capture_gap", "GPU CPU scope 内未发现设备工作",
        "可能是 view/空操作、融合到其他调用，也可能是采集或关联缺失；不能推断 GPU 耗时为零。",
        ["检查 GPU CPU 子调用、runtime correlation、External ID 及未归属设备事件。",
         "结合调用源码核实 view/空张量/编译融合行为；需要时扩大到共同语义区域比较，保留无法确认的缺口。"],
    ),
    "auxiliary_dtype": (
        "backend_difference_candidate", "辅助工作伴随 dtype 变化",
        "可能存在外置精度或类型准备步骤；仅凭 dtype 变化不能认定转换冗余。",
        ["沿同次 API/CPU 调用检查输入、主计算和输出 dtype 路径，核对标量、零维 Tensor、autocast 与输出要求。",
         "对照 GPU 是否在计算 kernel 内完成转换，做相同输入的最小复现与数值验证；先比较完整调用，不给辅助 kernel 重复分配 GPU 时间。"],
    ),
    "auxiliary_layout": (
        "backend_difference_candidate", "辅助工作伴随 shape/format 变化",
        "可能是布局转换、切片、广播或中间张量准备；GPU 可能在索引或读写时完成相同工作。",
        ["核对同次调用的 shape、format、stride/contiguous 状态以及 API 源码，确认是否需要物化中间张量。",
         "用相同 shape/dtype/布局重现两端路径，比较包含准备步骤的完整调用；不要把 view、拷贝和实际计算混为一类。"],
    ),
    "auxiliary_work": (
        "comparison_granularity", "辅助 kernel 无独立 GPU 对应项",
        "现有映射把 GPU 工作分配给主计算，辅助步骤无法独立分摊；尚不能确定是否可融合或省略。",
        ["查看同次 CPU/API 的全部 NPU 类型、GPU kernel 与输入输出元数据，明确辅助步骤的用途。",
         "验证工作量一致后增加完整调用比较；判断优化空间需要源码与最小复现，不能只提高逐 kernel 覆盖率。"],
    ),
    "gpu_scope_overlap": (
        "comparison_granularity", "候选 GPU 工作已被其他映射使用",
        "可能是父子调用共享或多对一关系，重复使用会虚增 GPU 总耗时。",
        ["检查候选 GPU kernel 的现有 NPU 用户及两端父子 CPU scope，区分正确共享和错误 owner。",
         "在共同调用层建立独立、去重的整体比较，或修复有证据的归属错误；不要直接复制已有 GPU 时间。"],
    ),
    "gpu_child_work": (
        "mapping_candidate", "GPU 工作位于 CPU 子调用",
        "父算子没有直接 kernel，但子调用中存在尚未使用的候选工作。",
        ["核对父调用与子 kernel 的语义、完整输入元数据及 NPU scope 是否一致。",
         "若实例关系唯一，可扩展子调用恢复规则；若包含多个不同语义的子计算，应按完整调用比较。"],
    ),
    "bundle_ambiguity": (
        "comparison_granularity", "两端 kernel bundle 粒度不一致",
        "已有逻辑调用候选，但没有可靠的逐 kernel 对应关系。",
        ["对照两端完整 kernel 列表和 NPU 输入输出元数据，检查多对一、一对多和辅助任务。",
         "工作量等价时采用去重的完整 CPU/共同区域比较；逐 kernel 性能比保持空值，不按数量相等强行配对。"],
    ),
    "unclassified": (
        "insufficient_evidence", "现有证据尚不能归因",
        "保留未解释记录；未知模式不会从缺口统计中消失。",
        ["根据记录的原始 kernel、CPU、API、flow 和候选 GPU 证据定位失败环节。",
         "补充缺失采集信息或建立最小复现，再增加可测试的诊断规则；不要直接认定为硬件或后端缺陷。"],
    ),
}


def validate_threshold(value):
    if not math.isfinite(value) or not 0 <= value <= 100:
        raise ValueError("diagnostic coverage threshold must be finite and between 0 and 100")


def _list(value):
    return json.loads(value) if isinstance(value, str) else list(value)


def _flag(value):
    # Also supports rows reloaded from exported CSV. bool('False') is unsafe.
    return value is True or value == 1 or str(value).lower() == "true"


def _shape(value):
    if str(value).strip().strip('"').lower() in {"", "n/a", "none", "null", "unknown"}:
        return []
    return json.loads(normalize_input_shapes(value) or "[]")


def _dtype(value):
    if str(value).strip().strip('"').lower() in {"", "n/a", "null", "unknown"}:
        return []
    text = str(value).replace("c10::", "").replace("at::", "")
    return [t or "none" for t in json.loads(normalize_input_dtypes(text) or "[]")]


def _api(kernel):
    name = kernel.detail.name if kernel.detail else kernel.event.name
    match = re.match(r"(aclnn[^_]+)_", name, re.I)
    return match.group(1) if match else ""


def _parents(profile, index):
    while index is not None:
        yield index
        index = profile.ops[index].parent_index


def _metadata(kernel):
    d = kernel.detail
    if d is None:
        return {}, ["missing_kernel_detail"]
    result = {"input_shapes": _shape(d.input_shapes), "output_shapes": _shape(d.output_shapes),
              "input_dtypes": _dtype(d.input_dtypes), "output_dtypes": _dtype(d.output_dtypes),
              "input_formats": d.raw.get("Input Formats", ""),
              "output_formats": d.raw.get("Output Formats", "")}
    effects = []
    inputs, outputs = result["input_dtypes"], result["output_dtypes"]
    # Additional int64 axes/shape/index tensors are not conversions of the data
    # tensor. Only a 1-input/1-output signature supports a dtype-change clue.
    if len(inputs) == len(outputs) == 1 and inputs != outputs:
        effects.append("dtypes_change")
    if len(set(inputs)) > 1:
        effects.append("mixed_input_dtypes")
    if len(inputs) > 1 or len(outputs) > 1:
        effects.append("multiple_inputs_or_outputs")
    # For multi-input tasks, a first-input/output comparison is a bounded
    # *candidate* data-tensor relation, not a reconstructed tensor dependency.
    comparable_signature = (len(inputs) == len(outputs) == 1 or
                            (len(inputs) > 1 and len(outputs) == 1 and inputs[0] == outputs[0]))
    if comparable_signature:
        a, b = result["input_shapes"], result["output_shapes"]
        if a and len(b) == 1 and a[0] != b[0]:
            effects.append("shapes_change")
        a = [f.strip() for f in result["input_formats"].split(";")]
        b = [f.strip() for f in result["output_formats"].split(";")]
        if a[0] and len(b) == 1 and b[0] and a[0] != b[0]:
            effects.append("formats_change")
    result["shape_format_comparison_basis"] = (
        "single_input_output" if len(inputs) == len(outputs) == 1 else
        "first_input_output_candidate" if comparable_signature else "not_established")
    if any(not result[k] for k in ("input_shapes", "output_shapes", "input_dtypes", "output_dtypes")):
        effects.append("incomplete_kernel_metadata")
    return result, effects


def _whole_call_index(precision_rows, cpu_bundle_rows, fused_rows):
    result = defaultdict(list)
    for rows, kind, source, identity, label in (
        (precision_rows, "precision_path", "coverage_diagnostics.json#specialized_evidence/precision_paths", "Precision Path ID", "NPU Path"),
        (cpu_bundle_rows, "cpu_bundle", "cpu_bundle_mapping.csv", "CPU Bundle ID", "CPU Bundle"),
        (fused_rows, "fused_call", "fused_op_mapping.csv", "Occurrence", "Fused OP"),
    ):
        for row in rows:
            if kind == "precision_path" and not _flag(row["Comparable"]):
                continue
            ref = {"kind": kind, "source": source, "id": row[identity], "scope": row[label]}
            if kind == "precision_path":
                ref.update({"pattern": row["Pattern"], "cause_status": row["Cause Status"],
                            "candidate_cause": row.get("Candidate Cause", ""),
                            "gpu_cast_evidence": row.get("GPU Cast Evidence", "")})
            for k in _list(row["NPU Kernel Indices"]):
                result[int(k)].append(ref)
    return result


def _reason(e):
    """Choose one accounting reason. Other observations remain in evidence."""
    if e["whole_call_references"]:
        return "whole_call_available"
    if e["mapping_mode"] == "api_semantic_conflict_unmapped":
        return "api_cpu_conflict"
    if e["npu_cpu"] is None:
        return "npu_owner_missing"
    if e["npu_link_method"] not in {"external_id", "runtime_correlation", "torch_to_npu",
                                      "host_to_device_queue", "host_to_device_thread"}:
        return "npu_owner_weak"
    if e["cpu_match_count"] > 1:
        return "cpu_match_ambiguous"
    if not e["cpu_match_count"]:
        return "cpu_alignment_gap" if e["same_module_metadata_candidate_count"] else "cpu_match_missing"
    if e["cpu_metadata_differences"]:
        return "cpu_metadata_conflict"
    if e["cpu_match_confidence"] not in {"high", "medium"}:
        return "cpu_match_untrusted"
    if not e["gpu_scope_kernel_indices"]:
        return "gpu_memory_only" if e["gpu_scope_memory_indices"] else "gpu_work_absent"
    if e["mapping_mode"] in {"api_helper_unmapped", "auxiliary_unmapped"}:
        if "dtypes_change" in e["metadata_effects"]:
            return "auxiliary_dtype"
        if any(k in e["metadata_effects"] for k in ("shapes_change", "formats_change")):
            return "auxiliary_layout"
        return "auxiliary_work"
    if e["gpu_existing_npu_users"]:
        return "gpu_scope_overlap"
    if not e["gpu_direct_kernel_indices"]:
        return "gpu_child_work"
    if e["mapping_mode"] == "ambiguous_bundle_unmapped" or e["npu_scope_kernel_count"] > 1:
        return "bundle_ambiguity"
    return "unclassified"


def build_coverage_diagnostics(gpu, npu, matches, mapping_rows, *, threshold=90.0,
                               precision_rows=(), cpu_bundle_rows=(), fused_rows=(), op_map=None):
    """Analyze every non-comparable row of every selected (device,type,core)."""
    validate_threshold(threshold)
    op_map = op_map or {}
    groups = defaultdict(list)
    seen = set()
    for row in mapping_rows:
        index = int(row["NPU Kernel Index"])
        duration = float(row["NPU Duration(us)"])
        if not math.isfinite(duration) or duration < 0:
            raise ValueError(f"invalid duration for diagnostic NPU kernel: {index}")
        if not 0 <= index < len(npu.kernels):
            raise ValueError(f"invalid diagnostic NPU kernel index: {index}")
        if index in seen:
            raise ValueError(f"duplicate NPU kernel in diagnostic input: {index}")
        seen.add(index)
        groups[tuple(str(row[k]) for k in ("Device_id", "OP Type", "Core Type"))].append(row)
    total_time = sum(float(r["NPU Duration(us)"]) for r in mapping_rows)
    selected = []
    for key, rows in groups.items():
        missing = [r for r in rows if not _flag(r["Comparable"])]
        duration = sum(float(r["NPU Duration(us)"]) for r in rows)
        missing_time = sum(float(r["NPU Duration(us)"]) for r in missing)
        count_coverage = 100 * (len(rows) - len(missing)) / len(rows)
        time_coverage = 100 * (duration - missing_time) / duration if duration else None
        if not (count_coverage < threshold or (time_coverage is not None and time_coverage < threshold)):
            continue
        selected.append({
            "device_id": key[0], "op_type": key[1], "core_type": key[2],
            "kernel_count": len(rows), "npu_total_time_us": round(duration, 6),
            "npu_time_share_pct": round(100 * duration / total_time, 6) if total_time else None,
            "comparable_count_coverage_pct": round(count_coverage, 6),
            "comparable_time_coverage_pct": round(time_coverage, 6) if time_coverage is not None else None,
            "uncomparable_kernel_count": len(missing), "uncomparable_time_us": round(missing_time, 6),
            "_rows": missing,
        })
    selected.sort(key=lambda s: (-s["uncomparable_time_us"], -s["uncomparable_kernel_count"],
                                 s["device_id"], s["op_type"], s["core_type"]))

    gk_scope, gm_scope, nk_scope = scoped_indices(gpu), scoped_indices(gpu, True), scoped_indices(npu)
    by_npu, gpu_users = defaultdict(list), defaultdict(set)
    for match in matches:
        by_npu[match.npu_op_index].append(match)
    for row in mapping_rows:
        if _flag(row["Comparable"]):
            for k in _list(row["GPU Kernel Indices"]):
                gpu_users[int(k)].add(int(row["NPU Kernel Index"]))
    whole = _whole_call_index(precision_rows, cpu_bundle_rows, fused_rows)
    gpu_names, gpu_modules = defaultdict(list), defaultdict(list)
    for i, op in enumerate(gpu.ops):
        name = op_map.get(op.canonical_name, op.canonical_name)
        gpu_names[name].append(i)
        if op.module_key:
            gpu_modules[(name, op.module_key)].append(i)

    def cpu_info(profile, index):
        op = profile.ops[index]
        return {"op_index": index, "name": op.canonical_name, "module": op.module_key,
                "input_shapes": _shape(op.input_shapes), "input_dtypes": _dtype(op.input_dtypes),
                "raw_input_shapes": op.raw_input_shapes or op.input_shapes,
                "raw_input_dtypes": op.raw_input_dtypes or op.input_dtypes,
                "parent_index": op.parent_index, "trace_event_index": op.event.index,
                "pid": op.event.pid, "tid": op.event.tid}

    @lru_cache(None)
    def gpu_cpu_info(index):
        return cpu_info(gpu, index)

    @lru_cache(None)
    def owner_context(index):
        cpu = cpu_info(npu, index)
        candidates = gpu_modules.get((cpu["name"], cpu["module"]), []) if cpu["module"] else []
        exact = [i for i in candidates if cpu["input_shapes"] and cpu["input_dtypes"]
                 and gpu_cpu_info(i)["input_shapes"] == cpu["input_shapes"]
                 and gpu_cpu_info(i)["input_dtypes"] == cpu["input_dtypes"]]
        ms = by_npu[index]
        match = ms[0] if len(ms) == 1 else None
        gp = gpu_cpu_info(match.gpu_op_index) if match else None
        differences, missing = [], []
        if gp:
            for field in ("input_shapes", "input_dtypes", "module"):
                if not cpu[field] or not gp[field]:
                    missing.append(field)
                elif cpu[field] != gp[field]:
                    differences.append(field)
            if cpu["name"] != op_map.get(gp["name"], gp["name"]):
                differences.append("name")
        ancestor = next((i for i in list(_parents(npu, index))[1:]
                         if len(by_npu[i]) == 1 and by_npu[i][0].confidence in {"high", "medium"}), None)
        scoped = nk_scope[index]
        return {
            "npu_cpu": cpu, "gpu_cpu": gp, "cpu_match_count": len(ms),
            "cpu_match_confidence": match.confidence if match else "",
            "cpu_alignment_basis": match.alignment_basis if match else "",
            "cpu_metadata_differences": differences, "cpu_missing_metadata": missing,
            "same_semantic_gpu_cpu_event_count": len(gpu_names.get(cpu["name"], [])),
            "same_module_gpu_cpu_candidate_count": len(candidates),
            "same_module_metadata_candidate_count": len(exact),
            "gpu_cpu_candidate_sample": [gpu_cpu_info(i) for i in (exact or candidates)[:5]],
            "candidate_sample_truncated": len(exact or candidates) > 5,
            "nearest_matched_npu_ancestor": ancestor,
            "nearest_matched_gpu_ancestor": by_npu[ancestor][0].gpu_op_index if ancestor is not None else None,
            "gpu_direct_kernel_indices": sorted(gpu.ops[match.gpu_op_index].kernel_indices) if match else [],
            "gpu_scope_kernel_indices": sorted(gk_scope[match.gpu_op_index]) if match else [],
            "gpu_scope_memory_indices": sorted(gm_scope[match.gpu_op_index]) if match else [],
            "npu_scope_kernel_count": len(scoped),
            "npu_scope_kernel_indices": sorted(scoped),
            "npu_scope_type_counts": dict(Counter(npu.kernels[k].detail.op_type
                if npu.kernels[k].detail else "unknown" for k in scoped)),
        }

    kernel_reports = []
    for rank, item in enumerate(selected, 1):
        item["priority_rank"] = rank
        reason_groups = defaultdict(list)
        for row in sorted(item.pop("_rows"), key=lambda r: int(r["NPU Kernel Index"])):
            k = int(row["NPU Kernel Index"])
            kernel = npu.kernels[k]
            metadata, effects = _metadata(kernel)
            evidence = dict(owner_context(kernel.owner_op_index)) if kernel.owner_op_index is not None else {"npu_cpu": None}
            evidence.update({
                "mapping_mode": row["Mapping Mode"], "accounted": _flag(row.get("Accounted", False)),
                "npu_link_method": kernel.link_method, "npu_link_scope": kernel.link_scope,
                "npu_original_owner": kernel.original_owner_op_index,
                "npu_original_link_method": kernel.original_link_method,
                "npu_task_id": kernel.task_id or (kernel.detail.task_id if kernel.detail else ""),
                "npu_stream_id": kernel.stream_id or (kernel.detail.stream_id if kernel.detail else ""),
                "npu_trace_event_index": kernel.event.index,
                "npu_api": _api(kernel), "kernel_metadata": metadata, "metadata_effects": effects,
                "whole_call_references": whole.get(k, []),
                "api_expected_cpu_ops": _list(row.get("API Expected CPU OPs", "[]")),
            })
            gk = evidence.get("gpu_scope_kernel_indices", [])
            gm = evidence.get("gpu_scope_memory_indices", [])
            evidence["gpu_existing_npu_users"] = {str(i): sorted(gpu_users[i]) for i in gk if gpu_users[i]}
            evidence["gpu_scope_kernel_names"] = [gpu.kernels[i].event.name for i in gk]
            evidence["gpu_scope_memory_names"] = [gpu.memory_activities[i].event.name for i in gm]
            reason = _reason(evidence)
            category, title, hypothesis, steps = ADVICE[reason]
            result = {
                "npu_kernel_index": k, "device_id": item["device_id"], "op_type": item["op_type"],
                "core_type": item["core_type"], "npu_kernel_name": kernel.detail.name if kernel.detail else kernel.event.name,
                "npu_duration_us": float(row["NPU Duration(us)"]),
                "reason_id": reason, "category": category, "reason": title,
                "hypothesis": hypothesis, "next_steps": list(steps), "cause_status": "requires_verification",
                "evidence": evidence,
            }
            kernel_reports.append(result)
            # Separate APIs and CPU semantics within a type: a Cast under one API
            # must not inherit another API's diagnosis or recommendations.
            reason_groups[(reason, evidence["npu_api"], evidence["npu_cpu"]["name"]
                           if evidence["npu_cpu"] else "")].append(result)
        item["diagnoses"] = []
        for (reason, api, cpu_name), rows in reason_groups.items():
            category, title, hypothesis, steps = ADVICE[reason]
            duration = sum(r["npu_duration_us"] for r in rows)
            item["diagnoses"].append({
                "reason_id": reason, "category": category, "reason": title, "npu_api": api,
                "cpu_op": cpu_name, "kernel_count": len(rows), "npu_time_us": round(duration, 6),
                "gap_time_share_pct": round(100 * duration / item["uncomparable_time_us"], 6)
                if item["uncomparable_time_us"] else None,
                "hypothesis": hypothesis, "next_steps": list(steps),
                "npu_kernel_indices": [r["npu_kernel_index"] for r in rows],
                "sample_kernel_indices": [r["npu_kernel_index"] for r in rows[:3]],
                "metadata_effect_counts": dict(Counter(effect for r in rows for effect in r["evidence"]["metadata_effects"])),
            })
        item["diagnoses"].sort(key=lambda d: (-d["npu_time_us"], -d["kernel_count"], d["reason_id"], d["npu_api"], d["cpu_op"]))
    return {
        "schema_version": 2,
        "selection": {"threshold_pct": threshold, "coverage": "Comparable", "condition": "count < threshold OR time < threshold",
                      "group_by": ["device_id", "op_type", "core_type"], "minimum_npu_time_share_pct": 0,
                      "zero_duration_policy": "time coverage unavailable; use count coverage"},
        "sources": {"gpu_trace": gpu.source_path, "npu_trace": npu.source_path,
                    "gpu_step": gpu.selected_step, "npu_step": npu.selected_step,
                    "gpu_pid": gpu.selected_pid, "npu_pid": npu.selected_pid},
        "summary": {
            "scanned_type_count": len(groups), "selected_type_count": len(selected),
            "analyzed_kernel_count": len(kernel_reports),
            "uncomparable_time_us": round(sum(r["npu_duration_us"] for r in kernel_reports), 6),
            "reason_kernel_counts": dict(Counter(r["reason_id"] for r in kernel_reports)),
            "category_kernel_counts": dict(Counter(r["category"] for r in kernel_reports)),
        },
        "limitations": [
            "Uncomparable kernel time is an observed comparison gap, not an estimate of removable overhead or speedup.",
            "One primary reason accounts for each kernel; secondary metadata effects and referenced whole calls are non-additive.",
            "Candidate CPU ops and ordered metadata patterns are not proven mappings, dataflow or backend defects.",
            "No visible GPU device work does not establish zero cost or a missing hardware capability.",
        ],
        "types": selected, "kernels": kernel_reports,
        # Keep full per-call evidence in the general report, not specialty tables.
        # This evidence is available even when its type does not meet the threshold.
        "specialized_evidence": {"precision_paths": list(precision_rows)},
    }


def _md(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("`", "'")


def _number(value):
    return "n.a." if value is None else f"{value:.2f}"


def _short(value):
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return _md(text[:180] + "...（摘要截断，完整见 JSON）" if len(text) > 180 else text)


def write_coverage_diagnostics(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "coverage_diagnostics.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary, threshold = report["summary"], report["selection"]["threshold_pct"]
    lines = ["# 低覆盖率 kernel 排查报告", "",
             f"筛选条件：Comparable 数量覆盖率或时间覆盖率 < {threshold:g}%。按 Device / OP Type / Core Type 分组，不限制算子类型或 NPU 时间占比。",
             f"扫描 {summary['scanned_type_count']} 类，命中 {summary['selected_type_count']} 类；分析 {summary['analyzed_kernel_count']} 个未比较 kernel，合计 {summary['uncomparable_time_us']:.3f} μs。", "",
             "按未比较 NPU 时间排序。这个时间是比较缺口，不是可节省时间；已有整体比较的辅助工作也可能仍有低逐类型覆盖率。",
             "每个 kernel 只计入一个主要原因。候选原因均待验证，不修改映射、覆盖率或 GPU 时间。完整证据与全部 kernel 编号见同目录 coverage_diagnostics.json。", "",
             "| 类型 | Device / Core | 数量覆盖率 | 时间覆盖率 | 未比较数量 | 未比较时间(μs) | 首要排查方向 |",
             "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for item in report["types"]:
        lines.append(f"| {_md(item['op_type'])} | {_md(item['device_id'])} / {_md(item['core_type'])} | "
                     f"{_number(item['comparable_count_coverage_pct'])}% | {_number(item['comparable_time_coverage_pct'])}" +
                     ("%" if item['comparable_time_coverage_pct'] is not None else "") +
                     f" | {item['uncomparable_kernel_count']} | {item['uncomparable_time_us']:.3f} | {_md(item['diagnoses'][0]['reason'])} |")
    by_kernel = {r["npu_kernel_index"]: r for r in report["kernels"]}
    for item in report["types"]:
        lines.extend(["", f"## {item['priority_rank']}. {_md(item['op_type'])} ({_md(item['device_id'])} / {_md(item['core_type'])})", ""])
        for d in item["diagnoses"]:
            lines.extend([f"### {_md(d['reason'])}：{d['kernel_count']} 个，{d['npu_time_us']:.3f} μs", "",
                          f"CPU：`{_md(d['cpu_op']) or '未关联'}`；API：`{_md(d['npu_api']) or '未提供'}`；分类：`{d['category']}`。",
                          f"候选解释：{d['hypothesis']}", "",
                          "证据样本（NPU Kernel Index）：" + ", ".join(map(str, d['sample_kernel_indices'])) + "。"])
            sample = by_kernel[d["sample_kernel_indices"][0]]["evidence"]
            lines.append(f"样本关联：{_md(sample['npu_link_method'])}；原映射状态：{_md(sample['mapping_mode'])}。")
            metadata = sample["kernel_metadata"]
            if metadata:
                lines.append(f"样本 NPU dtype：`{_short(metadata['input_dtypes'])}` → `{_short(metadata['output_dtypes'])}`；"
                             f"shape：`{_short(metadata['input_shapes'])}` → `{_short(metadata['output_shapes'])}`。")
            if sample.get("npu_scope_type_counts"):
                lines.append("样本完整 NPU CPU scope：" + ", ".join(
                    f"{_md(k)} × {v}" for k, v in sample["npu_scope_type_counts"].items()) + "；"
                    f"匹配 GPU CPU scope：{len(sample['gpu_scope_kernel_indices'])} 个计算 kernel、"
                    f"{len(sample['gpu_scope_memory_indices'])} 个内存活动。")
            if sample["whole_call_references"]:
                lines.append("整体记录：" + "；".join(
                    f"{ref['source']} / {ref['id']} / {_md(ref['scope'])}" for ref in sample["whole_call_references"]) + "。")
                for ref in sample["whole_call_references"]:
                    if ref.get("pattern"):
                        lines.append(f"专项观察：{_md(ref['pattern'])}；候选原因：{_md(ref.get('candidate_cause', '')) or '未确定'}；"
                                     f"原因验证状态：{_md(ref.get('cause_status', '未验证'))}。")
            if sample.get("cpu_metadata_differences"):
                lines.append("样本差异字段：" + ", ".join(sample["cpu_metadata_differences"]) + "。")
                lines.append("")
                for field in sample["cpu_metadata_differences"]:
                    lines.append(f"- {field}：NPU `{_short(sample['npu_cpu'][field])}`，"
                                 f"GPU `{_short(sample['gpu_cpu'][field])}`。")
                lines.append("")
            if sample.get("npu_cpu") and sample.get("cpu_match_count") == 0:
                lines.append(f"同模块同语义 GPU CPU 候选：{sample['same_module_gpu_cpu_candidate_count']} 条；"
                             f"其中完整输入元数据一致：{sample['same_module_metadata_candidate_count']} 条（仍未建立匹配）。")
            if d["metadata_effect_counts"]:
                lines.append("元数据观察：" + ", ".join(f"{k} × {v}" for k, v in d["metadata_effect_counts"].items()) + "。")
            lines.append("")
            lines.extend(f"{i}. {step}" for i, step in enumerate(d["next_steps"], 1))
            lines.append("")
    if not report["types"]:
        lines.extend(["", "当前阈值下没有命中类型。零时长类型只使用数量覆盖率筛选；这不代表所有后端实现已经证明等价。"])
    accounting = report.get("gpu_accounting")
    if accounting:
        lines.extend(["", "## GPU 未按 NPU 类型映射的工作", "",
                      "这里的未映射指未分配到逐类型表，不等于完全没有 CPU 匹配或整体比较。计算 kernel 与内存活动独立编号。",
                      "TOTAL 包含所选范围的全部 GPU kernel 和设备内存活动，各计一次；不包含 CPU runtime 时间。它是时长之和，不是墙钟时间或已证明等价的性能比。",
                      "整体算子、融合调用和专项证据是同一批工作的其他视图，不能再次累加。", ""])
        for kind, label in (("kernel", "计算 kernel"), ("memory", "内存活动")):
            info = accounting[kind]
            lines.append(f"- GPU {label}：总计 {info['total_count']} 个 / {info['total_time_us']:.3f} μs；"
                         f"已分配 {info['mapped_count']} 个 / {info['mapped_time_us']:.3f} μs；"
                         f"未分配 {info['unmapped_count']} 个 / {info['unmapped_time_us']:.3f} μs。"
                         f"未分配中 {info['whole_call_only_count']} 个已有整体比较。")
        lines.extend(["", f"NPU 总时间 {accounting['npu_total_time_us']:.3f} μs；GPU 总时间 {accounting['gpu_total_time_us']:.3f} μs。",
                      "计算与内存分项见表格的‘耗时分布’；逐事件编号、名称、CPU owner 和整体引用见 gpu_unmapped_activity.csv。"])
        if not accounting["type_rows_additive"]:
            lines.append("注意：存在跨类型 GPU 重用，不能直接相加类型行。总体已按 GPU 事件去重；冲突编号见 JSON gpu_accounting。")
        grouped = defaultdict(list)
        for row in report.get("gpu_unmapped_activities", []):
            grouped[(row["GPU Activity Kind"], row["Status"], row["GPU CPU OP Name"])].append(row)
        advice = {
            "whole_call_only": "已有整体比较，但没有安全的类型分摊。保留为类型表余项，不重复分配 GPU 时间。",
            "gpu_owner_missing": "检查 GPU correlation / External ID、CPU 采集范围和 step 边界，先恢复可靠的 GPU → CPU 关联。",
            "no_npu_type_assignment": "沿 GPU CPU op 检查 matches.csv、最近共同祖先和 NPU scope；核对别名、shape/dtype 及多对一粒度。不能仅按 kernel 名称或相邻时间硬配。",
        }
        for (kind, status, owner), rows in sorted(grouped.items(), key=lambda item: -sum(r["GPU Duration(us)"] for r in item[1])):
            lines.extend(["", f"### {_md(owner) or '未关联 CPU'} / {kind} / {status}", "",
                          f"{len(rows)} 个，合计 {sum(r['GPU Duration(us)'] for r in rows):.3f} μs。"
                          f"样本编号：{', '.join(str(r['GPU Activity Index']) for r in rows[:3])}。",
                          f"样本 kernel：`{_md(rows[0]['GPU Name'])}`。", advice[status]])
    paths = report.get("specialized_evidence", {}).get("precision_paths", [])
    if paths:
        lines.extend(["", "## 专项证据：完整精度路径", "",
                      "本节补充通用排查证据，不限制通用诊断的算子范围，也不改变覆盖率。Div/Sub 路径只在此报告呈现，不再生成专项表格。",
                      "下列组覆盖所有已识别调用（不受低覆盖率筛选影响）。完整逐调用 dtype/shape、kernel 编号、耗时和关联证据见 JSON specialized_evidence.precision_paths，按 Precision Path ID 查找。",
                      "有序元数据路径不是已证明的 tensor 数据流或后端缺陷；Cast 耗时也不等于可节省时间。"])
        grouped = defaultdict(list)
        for row in paths:
            grouped[(row.get("CPU OP", ""), row.get("NPU API", ""), row["Pattern"], row["NPU Path"],
                     row.get("Candidate Cause", ""), row.get("Comparison Status", ""))].append(row)
        for (cpu, api, pattern, path, cause, status), rows in grouped.items():
            lines.extend(["", f"### {_md(cpu)} / {_md(api)} / {_md(pattern)}", "",
                          f"路径：`{_md(path)}`。", f"共 {len(rows)} 次；候选解释：{_md(cause) or '待验证'}。比较状态：{_md(status)}。",
                          "样本 Precision Path ID：" + ", ".join(str(r["Precision Path ID"]) for r in rows[:3]) + "。"])
            if all("Cast Time(us)" in r for r in rows):
                lines.append(f"观测 Cast 成本：{sum(r['Cast Time(us)'] for r in rows):.3f} μs（非额外可累加工作）。")
            comparable_rows = [r for r in rows if _flag(r.get("Comparable", False))]
            if comparable_rows and all("NPU Total Time(us)" in r and "GPU Total Time(us)" in r for r in comparable_rows):
                nt = sum(r["NPU Total Time(us)"] for r in comparable_rows)
                gt = sum(r["GPU Total Time(us)"] for r in comparable_rows)
                lines.append(f"可比完整调用 {len(comparable_rows)} 次：NPU {nt:.3f} μs，GPU {gt:.3f} μs，"
                             f"GPU/NPU {_number(gt / nt if nt else None)}。这些时间与逐类型表重叠，不额外累加。")
            if rows[0].get("GPU Cast Evidence"):
                lines.append("样本 GPU 观察：" + _md(rows[0]["GPU Cast Evidence"]) + "。")
            lines.append("未观察到外置转换，可作为其他路径的对照；不能据此推断 kernel 内部计算精度。"
                         if pattern.startswith("native_") else
                         "排查建议：核对 API 实现与 autocast/标量输入，使用同 shape/dtype 的最小复现确认外置转换及 GPU 内部转换行为，并验证数值精度。")
    (output / "coverage_diagnostics.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

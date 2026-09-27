"""Reconcile the NPU-type view against every selected GPU device event.

Whole-call reports are alternative views, not additional time. An event included
only there still belongs in the per-type remainder. Kernel and memory indices
are separate namespaces. Totals are independently deduplicated, never averages
of ratios or sums of overlapping alternative views.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math


def _indices(row, field):
    value = row.get(field, "[]")
    return set(map(int, json.loads(value) if isinstance(value, str) else value))


def _flag(value):
    return value is True or str(value).lower() in {"true", "1"}


def _ratio(a, b, scale=1):
    return round(scale * a / b, 6) if b else ""


def _key(row):
    return tuple(str(row[k]) for k in ("Device_id", "OP Type", "Core Type"))


def add_coverage_conclusions(rows, report):
    """Present general observations, not Div/Sub paths or confirmed defects."""
    selected = {(str(t["device_id"]), t["op_type"], t["core_type"]): t for t in report["types"]}
    for row in rows:
        row["Row Kind"] = "npu_type"
        item = selected.get(_key(row))
        if item:
            reasons = Counter()
            advice = []
            for diagnosis in item["diagnoses"]:
                reasons[diagnosis["reason"]] += diagnosis["kernel_count"]
                step = diagnosis["next_steps"][0]
                if step not in advice:
                    advice.append(step)
            row["排查结论"] = "\n".join(f"{reason}：{count} 个" for reason, count in reasons.items())
            row["排查建议"] = "\n".join(advice) + "\n完整证据见 coverage_diagnostics.md / .json。"
        elif row["Comparable NPU Kernel Count"] == row["NPU Kernel Count"]:
            row["排查结论"] = "全部纳入逐类型比较"
            row["排查建议"] = ""
        else:
            row["排查结论"] = f"未触发排查阈值（{report['selection']['threshold_pct']:g}%），仍有未比较工作"
            row["排查建议"] = "可提高 --diagnostic-coverage-threshold 查看剩余缺口。"


def complete_gpu_accounting(gpu, mapping_rows, type_rows, *, whole_call_rows=()):
    """Return comparison rows, unmatched-event evidence and an audit summary.

Never infer a GPU-to-NPU device pairing for the synthetic rows. When a GPU event
is used by multiple NPU types, preserve the existing mapping but explicitly warn
that type rows are non-additive; the TOTAL row still counts that event once.
"""
    fields = list(type_rows[0]) if type_rows else [
        "Device_id", "FUSED_OP_NPU_TYPE", "FUSED_OP_GPU_TYPE", "OP Type", "Core Type",
        "Row Kind", "NPU Kernel Count", "NPU Total Time(us)", "GPU Kernel Count",
        "GPU Memory Activity Count", "GPU Kernel Time(us)", "GPU Memory Time(us)",
        "GPU Total Time(us)", "GPU Activity Kind", "Comparable NPU Time(us)",
        "Comparable Ratio(GPU/NPU)", "Total Time Ratio(GPU/NPU)", "排查结论", "排查建议",
    ]
    result = [dict(row) for row in type_rows]
    used, comparable, users, whole = {}, {}, {}, {}
    evidence = []
    audit = {"scope": "all selected GPU kernels and device memory activities; duration sum, not wall time",
             "unmapped_definition": "not assigned in the NPU-type view, even if included in an alternative whole-call view"}
    for kind, records, field in (("kernel", gpu.kernels, "GPU Kernel Indices"),
                                  ("memory", gpu.memory_activities, "GPU Memory Activity Indices")):
        users[kind] = defaultdict(set)
        comparable[kind] = set()
        whole[kind] = defaultdict(list)
        for row in mapping_rows:
            if not _flag(row.get("Mapped", False)):
                continue
            indices = _indices(row, field)
            if any(i < 0 or i >= len(records) for i in indices):
                raise ValueError(f"out-of-range mapped GPU {kind} index")
            for i in indices:
                users[kind][i].add(_key(row))
            if _flag(row.get("Comparable", row["Mapped"])):
                comparable[kind].update(indices)
        for call in whole_call_rows:
            if "Comparable" in call and not _flag(call["Comparable"]):
                continue
            ref = {k: call[k] for k in ("CPU Bundle ID", "CPU Bundle", "Precision Path ID", "Fused OP", "Occurrence") if k in call}
            for i in _indices(call, field):
                if i < 0 or i >= len(records):
                    raise ValueError(f"out-of-range whole-call GPU {kind} index")
                whole[kind][i].append(ref)
        used[kind] = set(users[kind])
        remaining = set(range(len(records))) - used[kind]
        times = [record.event.duration_us for record in records]
        if any(not math.isfinite(t) or t < 0 for t in times):
            raise ValueError(f"GPU {kind} durations must be finite and non-negative")
        audit[kind] = {
            "total_count": len(records), "mapped_count": len(used[kind]), "unmapped_count": len(remaining),
            "total_time_us": round(math.fsum(times), 6),
            "mapped_time_us": round(math.fsum(times[i] for i in used[kind]), 6),
            "unmapped_time_us": round(math.fsum(times[i] for i in remaining), 6),
            "whole_call_only_count": len(remaining.intersection(whole[kind])),
            "cross_type_reused_indices": sorted(i for i, keys in users[kind].items() if len(keys) > 1),
        }
        for i in sorted(remaining):
            record = records[i]
            owner = record.owner_op_index
            status = ("whole_call_only" if i in whole[kind] else
                      "gpu_owner_missing" if owner is None else "no_npu_type_assignment")
            evidence.append({
                "GPU Activity Kind": kind, "GPU Activity Index": i, "GPU Name": record.event.name,
                "GPU Duration(us)": times[i], "GPU PID": record.event.pid, "GPU TID": record.event.tid,
                "GPU CPU OP Index": owner if owner is not None else "",
                "GPU CPU OP Name": gpu.ops[owner].event.name if owner is not None else "",
                "GPU Link Method": record.link_method, "Status": status,
                "Whole Call References": json.dumps(whole[kind].get(i, []), ensure_ascii=False),
            })

    def synthetic(label, row_kind, title):
        row = dict.fromkeys(fields, "")
        row.update({"Row Kind": row_kind, "Device_id": "ALL_SELECTED",
                    "FUSED_OP_NPU_TYPE": title, "FUSED_OP_GPU_TYPE": title, "OP Type": label})
        return row

    # Always emit the requested remainder row, including a real zero when fully mapped.
    for kind, label, title in (("kernel", "GPU_UNMAPPED_KERNELS", "GPU 未按 NPU 类型映射"),
                                ("memory", "GPU_UNMAPPED_MEMORY", "GPU 未映射内存活动")):
        info = audit[kind]
        if kind == "memory" and not info["unmapped_count"]:
            continue
        row = synthetic(label, f"gpu_unmapped_{kind}", title)
        row.update({"NPU Kernel Count": 0, "NPU Total Time(us)": 0,
                    "GPU Kernel Count": info["unmapped_count"] if kind == "kernel" else 0,
                    "GPU Memory Activity Count": info["unmapped_count"] if kind == "memory" else 0,
                    "GPU Kernel Time(us)": info["unmapped_time_us"] if kind == "kernel" else 0,
                    "GPU Memory Time(us)": info["unmapped_time_us"] if kind == "memory" else 0,
                    "GPU Total Time(us)": info["unmapped_time_us"], "GPU Activity Kind": kind,
                    "排查结论": f"未分配到 NPU 类型：{info['unmapped_count']} 个；其中 {info['whole_call_only_count']} 个已在整体视图比较",
                    "排查建议": "核对通用报告 GPU 未映射部分及 gpu_unmapped_activity.csv；本行 NPU 为 0 仅表示无类型归属，覆盖率和性能比不适用。"})
        result.append(row)

    total = synthetic("TOTAL", "total", "总体（去重，非额外工作）")
    # NPU type rows are a partition, unlike the alternative whole-call views.
    for field in ("NPU Kernel Count", "Mapped NPU Kernel Count", "Unmapped NPU Kernel Count",
                  "Direct Mapped NPU Kernel Count", "Accounted NPU Kernel Count", "Comparable NPU Kernel Count",
                  "NPU Total Time(us)", "Mapped NPU Time(us)", "Direct Mapped NPU Time(us)",
                  "Accounted NPU Time(us)", "Comparable NPU Time(us)"):
        total[field] = round(math.fsum(float(r.get(field, 0)) for r in type_rows), 6)
        if field.endswith("Count"):
            total[field] = int(total[field])
    for label, prefix in (("Mapping", "Mapped"), ("Direct Mapping", "Direct Mapped"),
                          ("Accounted", "Accounted"), ("Comparable", "Comparable")):
        total[f"{label} Coverage(%)"] = _ratio(total[f"{prefix} NPU Kernel Count"], total["NPU Kernel Count"], 100)
    for label, prefix in (("Direct", "Direct Mapped"), ("Accounted", "Accounted"), ("Comparable", "Comparable")):
        total[f"{label} Time Coverage(%)"] = _ratio(total[f"{prefix} NPU Time(us)"], total["NPU Total Time(us)"], 100)
    gpu_total = round(audit["kernel"]["total_time_us"] + audit["memory"]["total_time_us"], 6)
    comparable_gpu = math.fsum(gpu.kernels[i].event.duration_us for i in comparable["kernel"]) + math.fsum(
        gpu.memory_activities[i].event.duration_us for i in comparable["memory"])
    additive = not any(audit[k]["cross_type_reused_indices"] for k in ("kernel", "memory"))
    total.update({"GPU Kernel Count": len(gpu.kernels), "GPU Memory Activity Count": len(gpu.memory_activities),
                  "GPU Kernel Time(us)": audit["kernel"]["total_time_us"],
                  "GPU Memory Time(us)": audit["memory"]["total_time_us"],
                  "GPU Total Time(us)": gpu_total, "GPU Activity Kind": "kernel+memory" if gpu.memory_activities else "kernel",
                  "NPU Ratio(%)": 100 if total["NPU Total Time(us)"] else "",
                  "Comparable Ratio(GPU/NPU)": _ratio(comparable_gpu, total["Comparable NPU Time(us)"]),
                  "Total Time Ratio(GPU/NPU)": _ratio(gpu_total, total["NPU Total Time(us)"]),
                  "Diff Time(GPU-NPU)(us)": round(gpu_total - total["NPU Total Time(us)"], 6),
                  "排查结论": "全量 GPU kernel 与内存活动各计一次；整体算子视图不累加" if additive else
                                "存在跨类型 GPU 重用：类型行不可相加，总体按事件去重",
                  "排查建议": "总体比值包含未映射工作，不代表相同类型性能；Comparable 比值仍仅使用可比较子集。多设备是所选范围汇总，不推断设备配对。"})
    result.append(total)
    audit.update({"type_rows_additive": additive, "npu_total_time_us": total["NPU Total Time(us)"],
                  "gpu_total_time_us": gpu_total, "comparable_gpu_time_us": round(comparable_gpu, 6),
                  "comparable_npu_time_us": total["Comparable NPU Time(us)"],
                  "total_time_ratio_gpu_npu": total["Total Time Ratio(GPU/NPU)"],
                  "gpu_unmapped_status_counts": dict(Counter(r["Status"] for r in evidence))})
    # Even empty NPU input must export every summary column to CSV.
    all_fields = dict.fromkeys(key for row in result for key in row)
    result = [{key: row.get(key, "") for key in all_fields} for row in result]
    return result, evidence, audit

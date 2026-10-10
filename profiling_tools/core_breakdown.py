"""NPU counter-based classification and deduplicated GPU reference groups."""
from collections import defaultdict
import json
import math


RULE = "aicore_time(us) = 0: Vector; > 0: Cube; missing/invalid: Unknown"


def counter(value):
    """Missing, malformed, negative and non-finite counters are not zero."""
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def category(value):
    value = counter(value)
    return "Unknown" if value is None else "Vector" if value == 0 else "Cube"


def annotate_counters(npu, mappings):
    for row in mappings:
        detail = npu.kernels[int(row["NPU Kernel Index"])].detail
        aic = counter(detail.aicore_time_us) if detail else None
        aiv = counter(detail.aiv_vec_time_us) if detail else None
        row.update({"NPU Compute Class": category(aic),
                    "NPU aicore_time(us)": aic, "NPU aiv_vec_time(us)": aiv})


def _ratio(numerator, denominator):
    return round(100 * numerator / denominator, 6) if denominator else None


def _flag(value):
    return value is True or str(value).lower() in ("true", "1")


def _ids(row):
    value = row.get("GPU Kernel Indices", "[]")
    return json.loads(value) if isinstance(value, str) else value


def _groups(records, categories):
    total_time = math.fsum(r["time_us"] for r in records)
    result = []
    for name in categories:
        members = [r for r in records if r["category"] == name]
        duration = math.fsum(r["time_us"] for r in members)
        result.append({"category": name, "count": len(members), "time_us": round(duration, 6),
                       "count_pct": _ratio(len(members), len(records)),
                       "time_pct": _ratio(duration, total_time)})
    return {"total_count": len(records), "total_time_us": round(total_time, 6), "groups": result}


def build_core_breakdown(gpu, mappings):
    records, gpu_users = [], defaultdict(set)
    for row in mappings:
        name = category(row.get("NPU aicore_time(us)"))
        records.append({"category": name, "time_us": float(row["NPU Duration(us)"]),
                        "aicore_time_us": counter(row.get("NPU aicore_time(us)")),
                        "aiv_vec_time_us": counter(row.get("NPU aiv_vec_time(us)"))})
        if _flag(row.get("Mapped")):
            for index in _ids(row):
                gpu_users[int(index)].add(name)
    npu = _groups(records, ("Cube", "Vector", "Unknown"))
    # Counters are separate measurements. Never add them to Duration(us).
    for group in [npu, *npu["groups"]]:
        members = records if group is npu else [r for r in records if r["category"] == group["category"]]
        for field in ("aicore_time_us", "aiv_vec_time_us"):
            values = [r[field] for r in members if r[field] is not None]
            group[field] = round(math.fsum(values), 6) if values or not members else None
            group[field.replace("_us", "_measured_count")] = len(values)
    gpu_records = []
    for index, kernel in enumerate(gpu.kernels):
        names = gpu_users.get(index, set())
        # Unknown takes precedence: a partial class cannot establish the full scope.
        name = ("Unmapped" if not names else "Unknown" if "Unknown" in names else
                "Mixed" if len(names) > 1 else next(iter(names)))
        gpu_records.append({"category": name, "time_us": kernel.event.duration_us})
    reference = _groups(gpu_records, ("Cube", "Vector", "Mixed", "Unknown", "Unmapped"))
    reference["excluded_memory_count"] = len(gpu.memory_activities)
    reference["excluded_memory_time_us"] = round(math.fsum(k.event.duration_us for k in gpu.memory_activities), 6)
    return {"schema_version": 1, "rule": RULE, "npu": npu, "gpu_reference": reference,
            "time_basis": "kernel Duration(us) sum, not wall time; counters are separate",
            "gpu_basis": "NPU-type mappings, including manual confirmations; unique GPU compute kernels only; not GPU hardware utilization"}


def statistic_rows(breakdown):
    result = []
    for side, key in (("NPU", "npu"), ("GPU (NPU class reference)", "gpu_reference")):
        for group in breakdown[key]["groups"]:
            result.append({"Side": side, "Compute Class": group["category"], "Kernel Count": group["count"],
                           "Kernel Count Ratio(%)": group["count_pct"], "Kernel Time(us)": group["time_us"],
                           "Kernel Time Ratio(%)": group["time_pct"],
                           "aicore_time(us)": group.get("aicore_time_us"),
                           "aicore Measured Count": group.get("aicore_time_measured_count"),
                           "aiv_vec_time(us)": group.get("aiv_vec_time_us"),
                           "aiv_vec Measured Count": group.get("aiv_vec_time_measured_count"),
                           "Scope": ("NPU: aicore_time=0 为 Vector，>0 为 Cube；计数器缺失为未知。" if key == "npu" else
                                     "GPU: 按已映射 NPU 分类，混合/未知/未映射各计一次；不含内存活动。") +
                                    "占比分母为各端全部计算 kernel；耗时使用 Duration，计数器独立汇总。"})
    return result

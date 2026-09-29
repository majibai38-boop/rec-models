"""Versioned, read-only access to comparison results and independent checks."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from .version import __version__, RESULT_SCHEMA_VERSION


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def file_identity(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"input changed while hashing: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest.hexdigest()}


def _true(value):
    return value is True or str(value).lower() in {"true", "1"}


def _number(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric result")
    return number


def _ids(row, field):
    value = row.get(field, "[]")
    result = json.loads(value) if isinstance(value, str) else value
    if not isinstance(result, list):
        raise ValueError(f"{field} must be a list")
    return [int(i) for i in result]


def _statistic_record(row):
    """Stable typed summary independent of presentation/translated CSV labels."""
    def number(field):
        value = row.get(field)
        return None if value in (None, "") else _number(value)

    result = {"row_kind": row.get("Row Kind"), "device_id": row["Device_id"],
              "op_type": row["OP Type"], "core_type": row["Core Type"]}
    for key, field in (("npu_kernel_count", "NPU Kernel Count"), ("gpu_kernel_count", "GPU Kernel Count"),
                       ("gpu_memory_activity_count", "GPU Memory Activity Count")):
        value = number(field)
        result[key] = None if value is None else int(value)
    for key, field in (("npu_total_time_us", "NPU Total Time(us)"), ("gpu_total_time_us", "GPU Total Time(us)"),
                       ("npu_comparable_time_us", "Comparable NPU Time(us)"), ("npu_ratio_pct", "NPU Ratio(%)"),
                       ("comparable_ratio_gpu_npu", "Comparable Ratio(GPU/NPU)"),
                       ("total_time_ratio_gpu_npu", "Total Time Ratio(GPU/NPU)")):
        result[key] = number(field)
    result["coverage"] = {
        kind: {"count_pct": number(count), "time_pct": number(time)}
        for kind, count, time in (("direct", "Direct Mapping Coverage(%)", "Direct Time Coverage(%)"),
                                  ("comparable", "Comparable Coverage(%)", "Comparable Time Coverage(%)"),
                                  ("accounted", "Accounted Coverage(%)", "Accounted Time Coverage(%)"))
    }
    if result["row_kind"] == "npu_type" and number("Comparable NPU Kernel Count") == 0:
        result["comparable_ratio_gpu_npu"] = None
        result["total_time_ratio_gpu_npu"] = None
    return result


def validate_results(output_dir):
    """Recalculate checks from inventories and mappings, not saved PASS flags.

Legacy result directories without manifests remain readable but are explicitly
unverified for provenance. Neither missing records nor NaNs count as healthy 0.
"""
    root = Path(output_dir)
    checks, warnings = [], []

    def check(name, ok, detail=""):
        checks.append({"id": name, "passed": bool(ok), "detail": detail})

    def close(a, b):
        return math.isclose(_number(a), _number(b), rel_tol=1e-9, abs_tol=1e-4)

    try:
        summary = read_json(root / "summary.json")
        schema = summary.get("schema_version")
        if schema not in (None, RESULT_SCHEMA_VERSION):
            raise ValueError(f"unsupported result schema version: {schema}")
        mappings = read_csv(root / "npu_kernel_gpu_mapping.csv")
        stats = read_csv(root / "npu_gpu_op_statistic.csv")
        links = read_csv(root / "kernel_link_diagnostic.csv")
        memory = read_csv(root / "gpu_memory_activity.csv")
        unmapped = read_csv(root / "gpu_unmapped_activity.csv")
        diagnostic = read_json(root / "coverage_diagnostics.json")
        totals = [r for r in stats if r.get("Row Kind") == "total"]
        check("one_total_row", len(totals) == 1)
        if len(totals) != 1:
            raise ValueError("expected exactly one TOTAL row")
        total = totals[0]
        inventory = {"kernel": {}, "memory": {}}
        npu_inventory = {}
        excluded_npu = []
        for r in links:
            index = int(r["Kernel Index"])
            duration = _number(r["Duration(us)"])
            check(f"duration.{r['Side']}.{index}", duration >= 0)
            if r["Side"] == "npu" and not r.get("Detail Row"):
                excluded_npu.append(duration)
                continue
            target = inventory["kernel"] if r["Side"] == "gpu" else npu_inventory
            if index in target:
                raise ValueError(f"duplicate {r['Side']} kernel index {index}")
            target[index] = duration
        if excluded_npu:
            warnings.append({"code": "npu_trace_only_excluded", "count": len(excluded_npu),
                             "time_us": round(sum(excluded_npu), 6),
                             "message": "NPU TOTAL covers kernel_details only; trace-only tasks are excluded. GPU TOTAL includes selected kernel and memory events; populations are not proven equivalent."})
        for r in memory:
            i, duration = int(r["GPU Memory Activity Index"]), _number(r["Duration(us)"])
            if i in inventory["memory"] or duration < 0:
                raise ValueError("duplicate memory index or negative memory duration")
            inventory["memory"][i] = duration
        npu_indices = [int(r["NPU Kernel Index"]) for r in mappings]
        check("npu_unique_partition", len(npu_indices) == len(set(npu_indices)) and set(npu_indices) == set(npu_inventory))
        check("npu_inventory_time", close(sum(npu_inventory.values()), sum(_number(r["NPU Duration(us)"]) for r in mappings)))
        check("npu_total_count", int(total["NPU Kernel Count"]) == len(mappings) == summary["comparison"]["npu_first_kernel_count"])
        check("npu_total_time", close(total["NPU Total Time(us)"], sum(npu_inventory.values())))
        mapped, comparable = {}, {}
        type_users = defaultdict(set)
        for kind, index_field, duration_field in (
            ("kernel", "GPU Kernel Indices", "GPU Kernel Durations(us)"),
            ("memory", "GPU Memory Activity Indices", "GPU Memory Activity Durations(us)"),
        ):
            mapped[kind], comparable[kind] = set(), set()
            for row in mappings:
                ids = _ids(row, index_field)
                times = json.loads(row.get(duration_field, "[]"))
                if len(ids) != len(times) or len(ids) != len(set(ids)):
                    raise ValueError("mapping indices/durations mismatch or duplicate within row")
                if ids and not _true(row["Mapped"]):
                    raise ValueError("unmapped row contains assigned GPU events")
                for i, t in zip(ids, times):
                    if i not in inventory[kind] or not close(t, inventory[kind][i]):
                        raise ValueError(f"mapped {kind} {i} missing from inventory or duration differs")
                    type_users[(kind, i)].add((row["Device_id"], row["OP Type"], row["Core Type"]))
                mapped[kind].update(ids)
                if _true(row.get("Comparable", row["Mapped"])):
                    comparable[kind].update(ids)
            remainder = [r for r in unmapped if r["GPU Activity Kind"] == kind]
            remaining_ids = [int(r["GPU Activity Index"]) for r in remainder]
            expected = set(inventory[kind]) - mapped[kind]
            check(f"{kind}.unmapped_partition", len(remaining_ids) == len(set(remaining_ids)) and set(remaining_ids) == expected)
            check(f"{kind}.unmapped_time", close(sum(_number(r["GPU Duration(us)"]) for r in remainder), sum(inventory[kind][i] for i in expected)))
            check(f"{kind}.count", int(total["GPU Kernel Count" if kind == "kernel" else "GPU Memory Activity Count"]) == len(inventory[kind]))
            check(f"{kind}.total_time", close(total["GPU Kernel Time(us)" if kind == "kernel" else "GPU Memory Time(us)"], sum(inventory[kind].values())))
            audit = summary["gpu_accounting"][kind]
            check(f"{kind}.saved_accounting", audit["total_count"] == len(inventory[kind]) and audit["mapped_count"] == len(mapped[kind])
                  and audit["unmapped_count"] == len(expected) and close(audit["total_time_us"], sum(inventory[kind].values())))
        gpu_time = sum(sum(items.values()) for items in inventory.values())
        comparable_gpu = sum(sum(inventory[k][i] for i in comparable[k]) for k in inventory)
        comparable_npu = sum(_number(r["NPU Duration(us)"]) for r in mappings if _true(r.get("Comparable", r["Mapped"])))
        check("gpu_total_time", close(total["GPU Total Time(us)"], gpu_time))
        for field, numerator, denominator in (
            ("Total Time Ratio(GPU/NPU)", gpu_time, _number(total["NPU Total Time(us)"])),
            ("Comparable Ratio(GPU/NPU)", comparable_gpu, comparable_npu),
        ):
            check(field, close(total[field], round(numerator / denominator, 6)) if denominator else total[field] == "")
        for row in stats:
            if row.get("Row Kind", "").startswith("gpu_unmapped"):
                check(f"{row['Row Kind']}.ratios_not_applicable", row["Comparable Ratio(GPU/NPU)"] == row["Total Time Ratio(GPU/NPU)"] == "")
        cross_type = sum(len(users) > 1 for users in type_users.values())
        check("cross_type_reuse_flag", summary["gpu_accounting"]["type_rows_additive"] == (cross_type == 0))
        if cross_type:
            warnings.append({"code": "cross_type_reuse", "message": "Type rows cannot be summed; TOTAL is independently deduplicated."})
        else:
            detail_rows = [r for r in stats if r.get("Row Kind") != "total"]
            check("gpu_detail_time_conservation", close(sum(_number(r["GPU Total Time(us)"]) for r in detail_rows), gpu_time))
        dk = diagnostic["kernels"]
        check("diagnostic_unique_kernels", len(dk) == len({r["npu_kernel_index"] for r in dk}))
        check("diagnostic_count_conservation", sum(sum(d["kernel_count"] for d in t["diagnoses"]) for t in diagnostic["types"]) == len(dk))
        check("diagnostic_time_conservation", close(sum(r["npu_duration_us"] for r in dk), diagnostic["summary"]["uncomparable_time_us"]))
        if (root / "whole_call_mapping.csv").exists():
            whole_rows = read_csv(root / "whole_call_mapping.csv")
            whole_stats = read_csv(root / "whole_call_statistic.csv")
            call_by_id = {r["Call ID"]: r for r in whole_rows}
            check("whole.unique_calls", len(call_by_id) == len(whole_rows))
            for row in whole_rows:
                call_id = row["Call ID"]
                amounts = {}
                for kind, field, source in (("npu", "NPU Kernel Indices", npu_inventory),
                                             ("kernel", "GPU Kernel Indices", inventory["kernel"]),
                                             ("memory", "GPU Memory Activity Indices", inventory["memory"])):
                    indices = _ids(row, field)
                    check(f"whole.{call_id}.{kind}.indices", len(indices) == len(set(indices)) and set(indices) <= set(source))
                    amounts[kind] = sum(source[i] for i in set(indices) if i in source)
                check(f"whole.{call_id}.npu_time", close(row["NPU Total Time(us)"], amounts["npu"]))
                check(f"whole.{call_id}.gpu_time", close(row["GPU Total Time(us)"], amounts["kernel"] + amounts["memory"]))
                comparable = _true(row["Comparable"])
                check(f"whole.{call_id}.ratio", close(row["Comparable Ratio(GPU/NPU)"],
                    (amounts["kernel"] + amounts["memory"]) / amounts["npu"]) if comparable and amounts["npu"] else row["Comparable Ratio(GPU/NPU)"] == "")
            check("whole.unique_groups", len(whole_stats) == len({r["Group ID"] for r in whole_stats}))
            check("whole.group_partition", {r["Group ID"] for r in whole_stats} == {r["Group ID"] for r in whole_rows})
            for row in whole_stats:
                group_id = row["Group ID"]
                all_ids = json.loads(row["Call IDs"])
                expected_ids = {r["Call ID"] for r in whole_rows if r["Group ID"] == group_id}
                check(f"whole.{group_id}.calls", set(all_ids) == expected_ids and len(all_ids) == len(expected_ids))
                selected = json.loads(row["Comparable Call IDs"])
                check(f"whole.{group_id}.comparable_subset", set(selected) <= expected_ids and len(selected) == len(set(selected)))
                members = [call_by_id[i] for i in selected]
                check(f"whole.{group_id}.comparable_count", len(members) == int(row["Comparable Call Count"]) and all(_true(r["Comparable"]) for r in members))
                for field in ("NPU Kernel Indices", "GPU Kernel Indices", "GPU Memory Activity Indices"):
                    indices = [i for r in members for i in _ids(r, field)]
                    check(f"whole.{group_id}.{field}.unique", len(indices) == len(set(indices)))
                nt = sum(_number(r["NPU Total Time(us)"]) for r in members)
                gt = sum(_number(r["GPU Total Time(us)"]) for r in members)
                check(f"whole.{group_id}.npu_time", close(row["Comparable NPU Time(us)"], nt))
                check(f"whole.{group_id}.gpu_time", close(row["Comparable GPU Time(us)"], gt))
                check(f"whole.{group_id}.ratio", close(row["Comparable Ratio(GPU/NPU)"], gt / nt) if nt else row["Comparable Ratio(GPU/NPU)"] == "")
                if row["Total Time Ratio(GPU/NPU)"] != "":
                    check(f"whole.{group_id}.population_ratio", _number(row["NPU Population Time(us)"]) > 0 and
                          close(row["Total Time Ratio(GPU/NPU)"], _number(row["GPU Population Time(us)"]) / _number(row["NPU Population Time(us)"])))
        if (root / "run_manifest.json").is_file():
            manifest = read_json(root / "run_manifest.json")
            check("run_identity", manifest["run_id"] == summary.get("run_id"))
            for name, identity in manifest["artifacts"].items():
                if not name or any(c in name for c in "/\\:") or name in {".", ".."}:
                    raise ValueError("unsafe artifact path in manifest")
                target = root / name
                if not target.is_file():
                    check("artifact." + name, False, "missing")
                else:
                    actual = file_identity(target)
                    check("artifact." + name, actual["sha256"] == identity["sha256"] and actual["size_bytes"] == identity["size_bytes"])
        else:
            warnings.append({"code": "legacy_no_manifest", "message": "Numerical checks only; tool/input/artifact provenance not recorded."})
    except (OSError, ValueError, TypeError, KeyError) as exc:
        check("readable_complete_results", False, str(exc))
    failed = [c for c in checks if not c["passed"]]
    # Successful per-event checks are summarized, not thousands of repeated rows.
    return {"schema_version": 1, "tool_version": __version__, "status": "invalid" if failed else "valid",
            "checks_run": len(checks), "checks_failed": len(failed), "failures": failed, "warnings": warnings}


def summarize_results(output_dir):
    root = Path(output_dir)
    summary = read_json(root / "summary.json")
    if summary.get("schema_version") not in (None, RESULT_SCHEMA_VERSION):
        raise ValueError("unsupported result schema version")
    diagnostic = read_json(root / "coverage_diagnostics.json")
    return {"schema_version": 1, "tool_version": __version__, "result_dir": str(root.resolve()),
            "run_id": summary.get("run_id"), "producer": summary.get("producer"),
            "status": summary.get("status", "legacy_unverified"),
            "scope": {side: {k: summary[side].get(k) for k in ("source", "selected_pid", "selected_step", "kernel_count")}
                      for side in ("gpu", "npu")},
            "gpu_accounting": summary.get("gpu_accounting"), "coverage": summary.get("coverage_diagnostics"),
            "overall_statistics": [_statistic_record(r) for r in read_csv(root / "npu_gpu_op_statistic.csv")
                                   if r.get("Row Kind") == "total"],
            "low_coverage_types": [{k: t[k] for k in ("device_id", "op_type", "core_type", "priority_rank",
                "comparable_count_coverage_pct", "comparable_time_coverage_pct", "uncomparable_kernel_count", "uncomparable_time_us")}
                                   for t in diagnostic["types"]],
            "warnings": summary.get("warnings", []), "saved_validation": summary.get("validation"),
            "reports": summary["reports"], "note": "Saved validation is not a fresh check; use validate before trusting changed results."}


def query_results(output_dir, *, op_type=None, npu_kernel_index=None, gpu_kernel_index=None,
                  memory_activity_index=None, cpu_op_index=None, side="npu", device_id=None,
                  core_type=None, offset=0, limit=20):
    if limit < 1 or limit > 200 or offset < 0:
        raise ValueError("limit must be 1..200 and offset >= 0")
    if all(v is None for v in (op_type, npu_kernel_index, gpu_kernel_index, memory_activity_index, cpu_op_index)):
        raise ValueError("query needs a type, kernel/activity index or CPU op index")
    if side not in {"gpu", "npu"}:
        raise ValueError("side must be gpu or npu")
    if any(v is not None and v < 0 for v in (npu_kernel_index, gpu_kernel_index, memory_activity_index, cpu_op_index)):
        raise ValueError("indices must be non-negative")
    root = Path(output_dir)
    if read_json(root / "summary.json").get("schema_version") not in (None, RESULT_SCHEMA_VERSION):
        raise ValueError("unsupported result schema version")
    matches = read_csv(root / "matches.csv")
    if cpu_op_index is not None and side == "gpu" and matches and "GPU OP Index" not in matches[0]:
        raise ValueError("legacy matches lack GPU CPU indices; regenerate results or query by kernel index")
    gpu_cpu = {r["Match ID"]: r.get("GPU OP Index", "") for r in matches}
    rows = read_csv(root / "npu_kernel_gpu_mapping.csv")
    cpu_gpu_ids, cpu_memory_ids = set(), set()
    if side == "gpu" and cpu_op_index is not None:
        cpu_gpu_ids = {int(r["Kernel Index"]) for r in read_csv(root / "kernel_link_diagnostic.csv")
                       if r["Side"] == "gpu" and r["Owner OP Index"] == str(cpu_op_index)}
        cpu_memory_ids = {int(r["GPU Memory Activity Index"]) for r in read_csv(root / "gpu_memory_activity.csv")
                         if r["Owner OP Index"] == str(cpu_op_index)}

    def selected(r):
        return ((op_type is None or r["OP Type"] == op_type)
                and (device_id is None or r["Device_id"] == device_id)
                and (core_type is None or r["Core Type"] == core_type)
                and (npu_kernel_index is None or int(r["NPU Kernel Index"]) == npu_kernel_index)
                and (gpu_kernel_index is None or gpu_kernel_index in _ids(r, "GPU Kernel Indices"))
                and (memory_activity_index is None or memory_activity_index in _ids(r, "GPU Memory Activity Indices"))
                and (cpu_op_index is None or (str(cpu_op_index) == r["NPU CPU OP Index"] if side == "npu" else
                     str(cpu_op_index) == gpu_cpu.get(r["CPU Match ID"], "") or
                     bool(cpu_gpu_ids.intersection(_ids(r, "GPU Kernel Indices"))) or
                     bool(cpu_memory_ids.intersection(_ids(r, "GPU Memory Activity Indices"))))))

    selected_rows = [r for r in rows if selected(r)]
    nk = {int(r["NPU Kernel Index"]) for r in selected_rows}
    type_keys = {(r["Device_id"], r["OP Type"], r["Core Type"]) for r in selected_rows}
    diagnostic = read_json(root / "coverage_diagnostics.json")
    def page(items):
        return {"total": len(items), "offset": offset, "limit": limit,
                "has_more": offset + limit < len(items), "items": items[offset:offset + limit]}
    calls = []
    for source, items in (("precision_paths", diagnostic.get("specialized_evidence", {}).get("precision_paths", [])),
                          ("cpu_bundle_mapping.csv", read_csv(root / "cpu_bundle_mapping.csv")),
                          ("fused_op_mapping.csv", read_csv(root / "fused_op_mapping.csv")),
                          ("whole_call_mapping.csv", read_csv(root / "whole_call_mapping.csv") if (root / "whole_call_mapping.csv").exists() else [])):
        for row in items:
            if (nk.intersection(_ids(row, "NPU Kernel Indices")) or
                (not any(v is not None for v in (op_type, npu_kernel_index, device_id, core_type)) and (
                    (gpu_kernel_index is not None and gpu_kernel_index in _ids(row, "GPU Kernel Indices")) or
                    (memory_activity_index is not None and memory_activity_index in _ids(row, "GPU Memory Activity Indices"))))):
                calls.append({"source": source, "record": row})
    # Unmapped GPU events are independently queryable, including ownerless work.
    gpu_rows = []
    npu_selector = any(v is not None for v in (op_type, npu_kernel_index, device_id, core_type)) or (side == "npu" and cpu_op_index is not None)
    if not npu_selector and (gpu_kernel_index is not None or memory_activity_index is not None or (side == "gpu" and cpu_op_index is not None)):
        for row in read_csv(root / "gpu_unmapped_activity.csv"):
            if ((gpu_kernel_index is None or (row["GPU Activity Kind"] == "kernel" and int(row["GPU Activity Index"]) == gpu_kernel_index))
                and (memory_activity_index is None or (row["GPU Activity Kind"] == "memory" and int(row["GPU Activity Index"]) == memory_activity_index))
                and (cpu_op_index is None or row["GPU CPU OP Index"] == str(cpu_op_index))):
                gpu_rows.append(row)
    type_stats = [_statistic_record(r) for r in read_csv(root / "npu_gpu_op_statistic.csv")
                  if (r["Device_id"], r["OP Type"], r["Core Type"]) in type_keys and r.get("Row Kind") == "npu_type"]
    reasons = [{"device_id": t["device_id"], "op_type": t["op_type"], "core_type": t["core_type"], **d}
               for t in diagnostic["types"] if (t["device_id"], t["op_type"], t["core_type"]) in type_keys
               for d in t["diagnoses"]]
    return {"schema_version": 1, "result_dir": str(root.resolve()),
            "filters": {"op_type": op_type, "npu_kernel_index": npu_kernel_index, "gpu_kernel_index": gpu_kernel_index,
                        "memory_activity_index": memory_activity_index, "cpu_op_index": cpu_op_index, "side": side,
                        "device_id": device_id, "core_type": core_type},
            "mappings": page(selected_rows), "diagnoses": page([r for r in diagnostic["kernels"] if r["npu_kernel_index"] in nk]),
            "type_statistics": page(type_stats), "reason_groups": page(reasons),
            "whole_calls": page(calls), "gpu_unmapped": page(gpu_rows),
            "note": "type_statistics and reason_groups describe entire matching Device/Type/Core groups, not only selected kernel IDs. CSV-backed mappings/whole_calls/gpu_unmapped preserve CSV strings; parse booleans explicitly. Whole calls overlap type rows. No diagnosis can mean the type did not meet the threshold, not that it is proven equivalent."}

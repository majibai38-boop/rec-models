"""Read-only, run-local call evidence and a reconciled overlay on the type table."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import math
from pathlib import Path
import tempfile
from html import escape

from .operator_details import identity
from .provenance import write_json
from .result_api import read_csv, read_json, validate_results, _ids, _true
from .xlsx_writer import TableSheet, write_grouped_workbook_xlsx
from .version import __version__


TYPE_FIELDS = ("OP Type", "Device_id", "Core Type")
EVENT_FIELDS = ("NPU Kernel Indices", "GPU Kernel Indices", "GPU Memory Activity Indices")
STRONG_SOURCES = {"matched_cpu_scope", "unique_child_of_matched_parent"}


def type_key(row):
    return tuple(row[k] for k in TYPE_FIELDS)


def ratio(a, b):
    return round(a / b, 6) if b else None


def _step_ok(scope, nstep, gstep):
    ns, gs = (scope[s]["selected_step"] for s in ("npu", "gpu"))
    if all(s and s != "all" for s in (ns, gs)):
        return (not nstep or nstep == ns) and (not gstep or gstep == gs)
    return bool(nstep and gstep and nstep.split("#")[-1] == gstep.split("#")[-1])


class CallEvidence:
    def __init__(self, directory):
        root = Path(directory)
        self.root = root
        summary = read_json(root / "summary.json")
        self.run_id = summary.get("run_id")
        self.verified_run = bool(self.run_id and (root / "run_manifest.json").is_file())
        self.scope = {s: {k: summary[s].get(k) for k in ("selected_pid", "selected_step")} for s in ("npu", "gpu")}
        self.stats = read_csv(root / "npu_gpu_op_statistic.csv")
        self.mappings = {int(r["NPU Kernel Index"]): r for r in read_csv(root / "npu_kernel_gpu_mapping.csv")}
        self.links = {(r["Side"], int(r["Kernel Index"])): r for r in read_csv(root / "kernel_link_diagnostic.csv")}
        self.memory = {int(r["GPU Memory Activity Index"]): r for r in read_csv(root / "gpu_memory_activity.csv")}
        self.matches = {(str(r.get("NPU OP Index")), str(r.get("GPU OP Index"))): r for r in read_csv(root / "matches.csv")}
        self.assigned = {kind: {i for r in self.mappings.values() for i in _ids(r, field)}
                         for kind, field in zip(("gpu", "memory"), EVENT_FIELDS[1:])}
        self.manual_users = defaultdict(set)
        for n, r in self.mappings.items():
            if r.get("Manual Link ID"):
                for side, field in zip(("gpu", "memory"), EVENT_FIELDS[1:]):
                    for i in _ids(r, field):
                        self.manual_users[side, i].add(n)
        self.warnings = []
        if not self.verified_run:
            self.warnings.append("缺少 manifest 或运行标识；只能查看已有统计")
        raw_calls = []
        for name, reader in (("whole_call_mapping.csv", read_csv), ("call_candidates.json", read_json)):
            if (root / name).is_file():
                raw_calls.extend(reader(root / name))
            else:
                self.warnings.append(f"缺少 {name}，对应调用证据不可用")
        diagnostics = read_json(root / "coverage_diagnostics.json")
        self.diagnoses = {(r["op_type"], r["device_id"], r["core_type"]): r["diagnoses"]
                          for r in diagnostics.get("types", [])}
        for r in diagnostics.get("specialized_evidence", {}).get("precision_paths", []):
            raw_calls.append({**r, "Call ID": f"precision:{r['Precision Path ID']}", "Whole OP": r["CPU OP"],
                              "Source": "precision_path", "Issues": "" if _true(r["Comparable"]) else r["Comparison Status"],
                              "Alignment Basis": r["CPU Alignment Basis"]})
        self.calls = {}
        self.groups = defaultdict(list)
        event_users = defaultdict(list)
        for r in raw_calls:
            call = self._call(r)
            cid = call["call_id"]
            if cid in self.calls:
                raise ValueError(f"duplicate call ID: {cid}")
            self.calls[cid] = call
            self.groups[call["group_id"]].append(call)
            if call["verified"]:
                for field in ("npu", "gpu", "memory"):
                    for i in call[field]:
                        event_users[field, i].append(cid)
        for c in self.calls.values():
            c["overlaps"] = sorted({cid for field in ("npu", "gpu", "memory") for i in c[field]
                                    for cid in event_users[field, i] if cid != c["call_id"]})

    def _call(self, row):
        nk, gk, gm = (_ids(row, f) for f in EVENT_FIELDS)
        if any(len(ids) != len(set(ids)) for ids in (nk, gk, gm)):
            raise ValueError(f"duplicate event in {row['Call ID']}")
        types = sorted({type_key(self.mappings[i]) for i in nk})
        match = self.matches.get((str(row.get("NPU CPU OP Index")), str(row.get("GPU CPU OP Index"))), {})
        nstep = row.get("NPU Step") or match.get("NPU Step", "")
        gstep = row.get("GPU Step") or match.get("GPU Step", "")
        eligible = _true(row["Comparable"]) and bool(nk and (gk or gm))
        reasons = [row["Issues"]] if row.get("Issues") else []
        if not _step_ok(self.scope, nstep, gstep):
            eligible = False
            reasons.append("缺少两侧对应 Step 的证据或 Step 冲突")
        if not self.verified_run:
            eligible = False
            reasons.append("缺少运行标识或 manifest")
        nset = set(nk)
        for side, indices, field in zip(("gpu", "memory"), (gk, gm), EVENT_FIELDS[1:]):
            available = set(indices)
            if (any(self.manual_users.get((side, i), set()) - nset for i in indices)
                or any(self.mappings[i].get("Manual Link ID") and
                       not set(_ids(self.mappings[i], field)) <= available for i in nk)):
                eligible = False
                reasons.append("已有人工关联跨越此调用范围")
                break
        # The original mapping (including manual links) always owns its events.
        single = eligible and len(types) == 1 and row["Source"] in STRONG_SOURCES and bool(gk) and not gm
        if single and (any(_true(self.mappings[i]["Comparable"]) for i in nk) or self.assigned["gpu"].intersection(gk)):
            single = False
            reasons.append("已有类型归属")
        if single and len(gk) > 1 and str(row.get("GPU Device Count", "")) != "1":
            single = False
            reasons.append("缺少单一 GPU 设备范围证据")
        if len(types) > 1:
            reasons.append("包含多个 NPU 类型，仅可整组比较")
        if not eligible and not reasons:
            reasons.append("缺少可信的完整调用配对")
        module = match.get("NPU Module Key", "") or row.get("Module", "")
        composition = row.get("GPU Composition") or row.get("GPU Kernel Names", "")
        group_id = "group:" + identity([row["Whole OP"], module, types, row.get("NPU Types", row.get("NPU Path")), composition])
        return {"call_id": str(row["Call ID"]), "group_id": group_id, "op": row["Whole OP"], "module": module,
                "types": types, "npu": nk, "gpu": gk, "memory": gm, "npu_cpu": row.get("NPU CPU OP Index"),
                "gpu_cpu": row.get("GPU CPU OP Index"), "npu_step": nstep, "gpu_step": gstep,
                "source": row["Source"], "basis": row.get("Alignment Basis", ""), "parent_anchor": row.get("Parent Anchor"),
                "anchor_metadata": {side: {k: (match or row).get(side + " " + k, "") for k in ("Input Shapes", "Input DTypes")}
                                    for side in ("NPU", "GPU")},
                "verified": eligible, "type_eligible": single, "reasons": reasons,
                "npu_us": sum(float(self.mappings[i]["NPU Duration(us)"]) for i in nk),
                "gpu_kernel_us": sum(float(self.links["gpu", i]["Duration(us)"]) for i in gk),
                "gpu_memory_us": sum(float(self.memory[i]["Duration(us)"]) for i in gm)}

    def query(self, op_type=None, device_id=None, core_type=None, group_id=None, call_id=None,
              unmapped_gpu=False, gpu_name=None, offset=0, limit=20):
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("limit must be 1..200 and offset >= 0")
        filters = (op_type, device_id, core_type)
        def selected(key):
            return all(v is None or k == v for k, v in zip(key, filters))
        if unmapped_gpu or gpu_name is not None:
            if any(x is not None for x in (*filters, group_id, call_id)):
                raise ValueError("GPU remainder query is independent of NPU type and call filters")
            items = [r for r in read_csv(self.root / "gpu_unmapped_activity.csv")
                     if gpu_name is None or r["GPU Name"] == gpu_name]
            context = {"inventory": "GPU activities without NPU type assignment"}
        elif call_id:
            c = self.calls.get(call_id)
            if not c or (group_id and c["group_id"] != group_id) or not any(map(selected, c["types"])):
                raise ValueError("unknown call for this scope")
            match = self.matches.get((str(c["npu_cpu"]), str(c["gpu_cpu"])), {})
            records = [{"side": side, **self.links[side, i]} for side in ("npu", "gpu") for i in c[side]]
            for record in records[:len(c["npu"])]:
                mapping = self.mappings[int(record["Kernel Index"])]
                record["existing_assignment"] = {"mode": mapping["Mapping Mode"], "manual": mapping.get("Manual Link ID", ""),
                                                 "gpu": _ids(mapping, EVENT_FIELDS[1]), "memory": _ids(mapping, EVENT_FIELDS[2])}
            records.extend({"side": "gpu_memory", **self.memory[i]} for i in c["memory"])
            context = {**self.call_info(c), "overlaps": c["overlaps"], "match_id": match.get("Match ID"),
                       "match_confidence": match.get("Confidence")}
            items = records
        elif group_id:
            if group_id not in self.groups:
                raise ValueError("unknown group ID")
            items = [self.call_info(c) for c in self.groups[group_id] if any(map(selected, c["types"]))]
            context = {"group_id": group_id}
        elif any(x is not None for x in filters):
            items = [self.group_info(g) for g in self.groups.values() if any(map(selected, g[0]["types"]))]
            items.sort(key=lambda g: (-g["npu_us"], g["group_id"]))
            context = {"diagnoses": [d for k, ds in self.diagnoses.items() if selected(k) for d in ds][:limit]}
        else:
            items = [{**dict(zip(TYPE_FIELDS, type_key(r))), "gap_us": round(float(r["NPU Total Time(us)"]) -
                      float(r["Comparable NPU Time(us)"]), 6)} for r in self.stats if r["Row Kind"] == "npu_type"]
            items.sort(key=lambda r: -r["gap_us"])
            context = {}
        return {"run_id": self.run_id, "scope": self.scope, "warnings": self.warnings, **context, "total": len(items),
                "has_more": offset + limit < len(items), "items": items[offset:offset + limit]}

    @staticmethod
    def call_info(c):
        fields = ("npu", "gpu", "memory", "overlaps")
        return {**{k: v for k, v in c.items() if k not in fields},
                **{k + "_count": len(c[k]) for k in fields}}

    @staticmethod
    def group_info(calls):
        c = calls[0]
        return {"group_id": c["group_id"], "op": c["op"], "module": c["module"], "types": c["types"],
                "count": len(calls), "verified": sum(x["verified"] for x in calls),
                "unique_type_candidates": sum(x["type_eligible"] and not x["overlaps"] for x in calls),
                "npu_us": sum(x["npu_us"] for x in calls), "sample_call_id": c["call_id"]}


def decisions(evidence, plan=None):
    """Resolve run-local requests. Neither group selection nor free text grants eligibility."""
    requests = {}
    if plan is not None:
        if not isinstance(plan, dict) or not evidence.run_id or plan.get("run_id") != evidence.run_id:
            raise ValueError("proposal run_id must match this result")
        entries = plan.get("proposals")
        if not isinstance(entries, list):
            raise ValueError("expected proposals array")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"evidence_id", "level", "reason"}:
                raise ValueError("proposal requires evidence_id, level and reason")
            key, level, reason = (entry[k] for k in ("evidence_id", "level", "reason"))
            if not isinstance(key, str) or level not in ("type", "call", "review") or not isinstance(reason, str) or not reason.strip():
                raise ValueError("invalid proposal")
            calls = evidence.groups.get(key) or ([evidence.calls[key]] if key in evidence.calls else [])
            if not calls:
                raise ValueError(f"unknown evidence ID: {key}")
            for c in calls:
                if c["call_id"] in requests:
                    raise ValueError("duplicate call in proposals")
                requests[c["call_id"]] = level, reason
    result = []
    for c in evidence.calls.values():
        default = "type" if c["type_eligible"] else "call"
        requested, reason = requests.get(c["call_id"], (default, ""))
        level = requested
        issues = list(c["reasons"])
        if requested == "review" or not c["verified"]:
            level = "review"
        elif c["overlaps"] and (c["call_id"] not in requests or any(
                requests.get(other, ("review", ""))[0] != "review" for other in c["overlaps"])):
            level = "review"
            issues.append("候选调用共享设备事件")
        elif c["overlaps"]:
            same_events = all(all(set(c[f]) == set(evidence.calls[other][f]) for f in ("npu", "gpu", "memory"))
                              for other in c["overlaps"])
            level = "type" if requested == "type" and c["type_eligible"] and same_events else "call"
            issues.append("按提案采用此完整调用；其他重叠候选未计入")
        elif requested == "type" and not c["type_eligible"]:
            level = "call"
            issues.append("证据仅支持整组比较")
        result.append({**c, "level": level, "agent_reason": reason, "reasons": issues})
    return result


def make_tables(ev, calls):
    promoted = defaultdict(list)
    related = defaultdict(set)
    summary = []
    by_group = defaultdict(list)
    for c in calls:
        by_group[c["group_id"]].append(c)
        for key in c["types"]:
            related[key].add(c["group_id"])
        if c["level"] == "type":
            promoted[c["types"][0]].append(c)
    for gid, members in by_group.items():
        usable = [c for c in members if c["level"] != "review"]
        nt = sum(c["npu_us"] for c in usable)
        gt = sum(c["gpu_kernel_us"] for c in usable)
        mt = sum(c["gpu_memory_us"] for c in usable)
        owners = {side: sorted({ev.links[side, i]["Owner OP Name"] for c in members for i in c[side]})
                  for side in ("npu", "gpu")}
        summary.append({"调用组": members[0]["op"], "模块": members[0]["module"], "组 ID": gid,
                        "NPU 类型": " / ".join(sorted({t[0] for t in members[0]["types"]})),
                        "NPU kernel 直接调用方": " / ".join(owners["npu"]),
                        "GPU kernel 直接调用方": " / ".join(owners["gpu"]),
                        "观测调用": len(members), "类型新增调用": sum(c["level"] == "type" for c in members),
                        "仅整组可比": sum(c["level"] == "call" for c in members), "待确认": len(members) - len(usable),
                        "可比 NPU kernel 数": sum(len(c["npu"]) for c in usable),
                        "可比 GPU kernel 数": sum(len(c["gpu"]) for c in usable),
                        "可比 NPU(us)": nt, "可比 GPU kernel(us)": gt, "可比 GPU memory(us)": mt,
                        "Comparable Ratio(GPU/NPU)": ratio(gt + mt, nt),
                        "核查说明": " / ".join(sorted({r for c in members for r in c["reasons"] if r})),
                        "Agent 说明": " / ".join(sorted({c["agent_reason"] for c in members if c["agent_reason"]}))})
    summary.sort(key=lambda r: (-r["可比 NPU(us)"], r["组 ID"]))
    types = []
    added_n, added_g = set(), set()
    for group in promoted.values():
        for c in group:
            if added_n.intersection(c["npu"]) or added_g.intersection(c["gpu"]):
                raise ValueError("overlapping promoted event sets")
            added_n.update(c["npu"])
            added_g.update(c["gpu"])
    all_n = sum(float(ev.mappings[i]["NPU Duration(us)"]) for i in added_n)
    all_g = sum(float(ev.links["gpu", i]["Duration(us)"]) for i in added_g)
    comparable_ids = {side: {i for r in ev.mappings.values() if _true(r["Comparable"]) for i in _ids(r, field)}
                      for side, field in zip(("gpu", "memory"), EVENT_FIELDS[1:])}
    group_by_id = {r["组 ID"]: r for r in summary}
    for row in ev.stats:
        kind = row["Row Kind"]
        key = type_key(row)
        gains = promoted.get(key, [])
        nsum, gsum = float(row["NPU Total Time(us)"]), float(row["GPU Total Time(us)"])
        nk, gk = int(row["NPU Kernel Count"] or 0), int(row["GPU Kernel Count"] or 0)
        nc = float(row["Comparable NPU Time(us)"] or 0)
        base_cov = ratio(100 * nc, nsum)
        count, nt, gt = len(gains), sum(c["npu_us"] for c in gains), sum(c["gpu_kernel_us"] for c in gains)
        groups = sorted(related.get(key, []))
        if kind == "gpu_unmapped_kernel":
            gsum -= all_g
            gk -= len(added_g)
        if kind == "total":
            nt, count = all_n, sum(map(len, promoted.values()))
            # TOTAL numerator includes only comparable GPU events, counted once.
            base_gpu = sum(float(ev.links["gpu", i]["Duration(us)"]) for i in comparable_ids["gpu"])
            base_gpu += sum(float(ev.memory[i]["Duration(us)"]) for i in comparable_ids["memory"])
            comp_gpu = base_gpu + all_g
        else:
            comp_gpu = gsum + gt
            gsum += gt
            gk += sum(len(c["gpu"]) for c in gains)
        comparable = nc + nt
        notes = [f"{group_by_id[g]['调用组']}（{g}）：类型新增 {group_by_id[g]['类型新增调用']}，"
                 f"仅整组 {group_by_id[g]['仅整组可比']}，待确认 {group_by_id[g]['待确认']}" for g in groups]
        if kind == "npu_type" and nsum - comparable > 1e-5:
            if not groups:
                notes.append("缺少已验证的完整调用")
            reasons = ev.diagnoses.get(key, [])
            if reasons:
                notes.append("原始缺口：" + reasons[0]["reason"])
            notes.append("按类型查询 kernel owner 和未映射 GPU 活动")
        comparable_count = int(row["Comparable NPU Kernel Count"] or 0)
        comparable_count += len(added_n) if kind == "total" else sum(len(c["npu"]) for c in gains)
        types.append({**dict(zip(TYPE_FIELDS, key)),
                      **{f: row.get(f, "") for f in ("FUSED_OP_NPU_TYPE", "FUSED_OP_GPU_TYPE")},
                      "NPU Kernel Count": nk, "GPU Kernel Count": gk,
                      "GPU Memory Activity Count": int(row["GPU Memory Activity Count"] or 0),
                      "NPU Total Time(us)": nsum, "GPU Total Time(us)": round(gsum, 6),
                      "引擎可比时间覆盖率(%)": base_cov, "新增调用": count, "新增可比 NPU(us)": round(nt, 6),
                      "Comparable NPU Time(us)": round(comparable, 6), "Comparable Time Coverage(%)": ratio(100 * comparable, nsum),
                      "Comparable NPU Kernel Count": comparable_count, "Comparable Coverage(%)": ratio(100 * comparable_count, nk),
                      "Comparable Ratio(GPU/NPU)": ratio(comp_gpu, comparable),
                      "Total Time Ratio(GPU/NPU)": ratio(gsum, nsum) if comparable or kind == "total" else None,
                      "调用关系分析": " / ".join(notes), "Row Kind": kind})
    total = next(r for r in types if r["Row Kind"] == "total")
    nrows = [r for r in types if r["Row Kind"] == "npu_type"]
    def close(a, b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-4)
    if not (close(sum(r["NPU Total Time(us)"] for r in nrows), total["NPU Total Time(us)"]) and
            close(sum(r["Comparable NPU Time(us)"] for r in nrows), total["Comparable NPU Time(us)"]) and
            not ev.assigned["gpu"].intersection(added_g)):
        raise ValueError("type table does not reconcile with TOTAL")
    remainder = [r for r in types if r["Row Kind"].startswith("gpu_unmapped")]
    remaining_gpu = sum(r["GPU Total Time(us)"] for r in remainder)
    used_gpu = sum(float(ev.links["gpu", i]["Duration(us)"]) for i in ev.assigned["gpu"] | added_g)
    used_gpu += sum(float(ev.memory[i]["Duration(us)"]) for i in ev.assigned["memory"])
    if not close(used_gpu + remaining_gpu, total["GPU Total Time(us)"]) or (
            len(ev.assigned["gpu"] | added_g) + sum(r["GPU Kernel Count"] for r in remainder) != total["GPU Kernel Count"]):
        raise ValueError("GPU kernel/memory allocation and remainder do not reconcile with TOTAL")
    return types, summary


class InvalidResults(ValueError):
    pass


@contextmanager
def staged_output(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".call-tables.lock"
    with lock.open("x"):
        pass
    published = []
    try:
        if any(entry != lock for entry in output.iterdir()):
            raise ValueError("output directory is no longer empty")
        with tempfile.TemporaryDirectory(dir=output) as tmp:
            staging = Path(tmp)
            yield staging
            for artifact in staging.iterdir():
                destination = output / artifact.name
                artifact.rename(destination)
                published.append(destination)
    except BaseException:
        for artifact in published:
            artifact.unlink()
        raise
    finally:
        lock.unlink()


def export_tables(directory, output_dir, proposals=None, caliber_note=""):
    output, source = Path(output_dir).resolve(), Path(directory).resolve()
    if output == source or source in output.parents or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("choose a new or empty output directory outside the result directory")
    validation = validate_results(source)
    if validation["status"] != "valid":
        raise InvalidResults(f"结果校验失败：{validation['failures'][:5]}")
    ev = CallEvidence(source)
    plan = read_json(proposals) if proposals else None
    records = decisions(ev, plan)
    if plan is not None and "calls" in plan:
        fields = ("call_id", "level", "npu", "gpu", "memory")
        def replay_keys(items):
            return sorted(tuple(c[f] for f in fields) for c in items)
        if replay_keys(plan["calls"]) != replay_keys(records):
            raise InvalidResults("重放结果与保存的事件或归属有差异，请重新核查提案")
    try:
        types, groups = make_tables(ev, records)
    except ValueError as exc:
        raise InvalidResults(str(exc)) from exc
    note = caliber_note or (plan or {}).get("caliber_note") or "精度、并行策略及 rank 的可比性尚未声明。"
    scope_note = "；".join(f"{s.upper()} PID {r['selected_pid']}，Step {r['selected_step']}" for s, r in ev.scope.items())
    metric_note = "时间为设备事件累计 µs。比值为 GPU/NPU，GPU 包含内存活动。类型与调用组分别阅读。"
    warnings = ev.warnings + [r["message"] for r in validation["warnings"] if "message" in r]
    report = {"schema_version": 1, "tool_version": __version__, "run_id": ev.run_id, "scope": ev.scope, "caliber_note": note,
              "warnings": warnings, "types": types, "groups": groups, "calls": records,
              "proposals": (plan or {}).get("proposals", []), "validation": validation["status"]}
    with staged_output(output) as staging:
        write_json(staging / "call_tables.json", report)
        names = ("类型对比", "调用组")
        def visible(row):
            return {k: ("" if v is None else v) for k, v in row.items() if k != "Row Kind"}
        group_columns = tuple(groups[0]) if groups else ("调用组", "观测调用", "待确认", "Comparable Ratio(GPU/NPU)")
        first_group = {}
        for i, group in enumerate(groups):
            for key in ev.groups[group["组 ID"]][0]["types"]:
                first_group.setdefault(key, i + 2)
        links = {(i, "调用关系分析"): f"'调用组'!A{first_group[type_key(row)]}" for i, row in enumerate(types)
                 if row["Row Kind"] == "npu_type" and type_key(row) in first_group}
        sheets = [TableSheet(names[0], tuple(visible(types[0])), rows=[visible(r) for r in types], links=links),
                  TableSheet(names[1], group_columns, rows=[visible(r) for r in groups]),
                  TableSheet("口径", ("说明",), rows=[{"说明": s} for s in (scope_note, note, metric_note, *warnings)], wrap_columns=("说明",))]
        write_grouped_workbook_xlsx(staging / "call_tables.xlsx", [], sheets)
        lines = ["# Profiling 调用关系对照", "", scope_note, "", note, "", metric_note, *warnings]
        for name, rows, fields in ((names[0], types, ("OP Type", "Device_id", "Core Type", "NPU Total Time(us)",
                "GPU Total Time(us)", "新增调用", "Comparable Time Coverage(%)", "Comparable Ratio(GPU/NPU)", "调用关系分析")),
                (names[1], groups, ("调用组", "组 ID", "观测调用", "类型新增调用", "仅整组可比", "待确认",
                                   "可比 NPU(us)", "可比 GPU kernel(us)", "可比 GPU memory(us)", "Comparable Ratio(GPU/NPU)", "核查说明", "Agent 说明"))):
            lines.extend(["", "## " + name, "", "| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)])
            for row in rows:
                cells = [escape(str(row[f] if row[f] is not None else "—")).replace("|", "\\|").replace("\n", " ") for f in fields]
                lines.append("| " + " | ".join(cells) + " |")
        (staging / "call_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"run_id": ev.run_id, "output_dir": str(output), "type_count": sum(r["Row Kind"] == "npu_type" for r in types),
            "group_count": len(groups), "decisions": dict(Counter(r["level"] for r in records)), "warnings": warnings}

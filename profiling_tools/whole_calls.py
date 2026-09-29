"""Alternative complete-call views. Never allocate GPU time or modify TOTAL."""
from collections import Counter, defaultdict
import hashlib
import json

from .call_metadata import compare_metadata
from .device_compare import encoded, scoped_indices


REPORT_ONLY_OPS = {"aten::div", "aten::true_divide", "aten::sub", "aten::sub_", "aten::div_"}
SCOPE = "独立整体视图；与类型表及其他重叠调用不可相加；时间为事件时长之和"


def ids(row, field):
    value = row.get(field, "[]")
    return list(json.loads(value) if isinstance(value, str) else value)


def composition(names):
    return " + ".join(f"{name} ×{count}" if count > 1 else name
                      for name, count in sorted(Counter(names).items()))


def gpu_label(name):
    if "CatArrayBatchedCopy" in name:
        return "CatArrayBatchedCopy"
    if "direct_copy_kernel_cuda" in name:
        return "copy(nocast)" if "impl_nocast" in name else "copy(generic)"
    return name.split("<", 1)[0].removeprefix("void ")


def build_whole_calls(gpu, npu, matches, mapping_rows, cpu_rows, fused_rows, op_map=None):
    """Reuse existing trusted bundles and expose bounded candidate CPU scopes.

    Missing metadata stays pending. Existing types/coverage are not promoted.
    Total-population ratios remain unavailable unless a separate full census is
    established; a list of matched scopes must not masquerade as all calls.
    """
    gscope, mscope, nscope = scoped_indices(gpu), scoped_indices(gpu, True), scoped_indices(npu)
    noncomparable = {int(r["NPU Kernel Index"]) for r in mapping_rows if not r["Comparable"]}
    by_npu, by_gpu = Counter(m.npu_op_index for m in matches), Counter(m.gpu_op_index for m in matches)
    result, signatures = [], set()

    def append(call_id, name, nk, gk, gm, ni="", gi="", *, source, basis, cpu_status,
               comparable, issues="", population_complete=False):
        nk, gk, gm = sorted(set(nk)), sorted(set(gk)), sorted(set(gm))
        if not nk or not (gk or gm):
            return
        signature = (tuple(nk), tuple(gk), tuple(gm))
        if signature in signatures:
            return
        signatures.add(signature)
        ntime = sum(npu.kernels[i].event.duration_us for i in nk)
        gtime = sum(gpu.kernels[i].event.duration_us for i in gk)
        mtime = sum(gpu.memory_activities[i].event.duration_us for i in gm)
        device = ",".join(sorted({npu.kernels[i].detail.device_id for i in nk}))
        ntypes = composition(npu.kernels[i].detail.op_type for i in nk)
        gtypes = composition([gpu_label(gpu.kernels[i].event.name) for i in gk]
                             + [gpu.memory_activities[i].event.name for i in gm])
        group_key = [name, device, ntypes, gtypes]
        group_id = "group:" + hashlib.sha256(encoded(group_key).encode()).hexdigest()[:16]
        result.append({
            "Call ID": call_id, "Group ID": group_id, "Whole OP": name,
            "Device_id": device, "NPU Types": ntypes, "GPU Composition": gtypes,
            "NPU CPU OP Index": ni, "GPU CPU OP Index": gi,
            "NPU Kernel Indices": encoded(nk), "GPU Kernel Indices": encoded(gk),
            "GPU Memory Activity Indices": encoded(gm),
            "NPU Kernel Names": encoded(npu.kernels[i].event.name for i in nk),
            "GPU Kernel Names": encoded(gpu.kernels[i].event.name for i in gk),
            "GPU Memory Activity Names": encoded(gpu.memory_activities[i].event.name for i in gm),
            "NPU Kernel Count": len(nk), "GPU Kernel Count": len(gk), "GPU Memory Activity Count": len(gm),
            "NPU Total Time(us)": round(ntime, 6), "GPU Kernel Time(us)": round(gtime, 6),
            "GPU Memory Time(us)": round(mtime, 6), "GPU Total Time(us)": round(gtime + mtime, 6),
            "Comparable": comparable, "Comparable Ratio(GPU/NPU)": round((gtime + mtime) / ntime, 6)
            if comparable and ntime else "",
            "CPU Alignment Status": cpu_status,
            "Kernel Comparison Status": "完整调用可比较，未独立归因" if comparable else "待核实整体比较",
            "Issues": issues, "Alignment Basis": basis, "Source": source,
            "Population Complete": population_complete, "Overlap Call IDs": "[]", "Scope": SCOPE,
        })

    # Preserve stronger, already bounded rules. These do not reassign type rows.
    for row in cpu_rows:
        append(row["CPU Bundle ID"], row["CPU Bundle"], ids(row, "NPU Kernel Indices"),
               ids(row, "GPU Kernel Indices"), ids(row, "GPU Memory Activity Indices"),
               row["NPU CPU OP Index"], row["GPU CPU OP Index"], source="cpu_bundle_mapping.csv",
               basis=row["Alignment Basis"], cpu_status="已对齐（完整调用规则）", comparable=True)
    for row in fused_rows:
        append("fa:" + str(row["Occurrence"]), row["Fused OP"], ids(row, "NPU Kernel Indices"),
               ids(row, "GPU Kernel Indices"), [], gi=row["GPU OP Index"], source="fused_op_mapping.csv",
               basis=row["Alignment Basis"], cpu_status="融合规则关联（非直接 CPU 配对）", comparable=True)

    for match in matches:
        ni, gi = match.npu_op_index, match.gpu_op_index
        nop, gop = npu.ops[ni], gpu.ops[gi]
        if nop.canonical_name in REPORT_ONLY_OPS or gop.canonical_name in REPORT_ONLY_OPS:
            continue
        all_nk = nscope[ni]
        nk = [i for i in all_nk if npu.kernels[i].detail is not None]
        if not nk or not (gscope[gi] or mscope[gi]):
            continue
        types = {npu.kernels[i].detail.op_type for i in nk}
        if len(types) < 2 and not noncomparable.intersection(nk):
            continue
        differences, missing = compare_metadata(nop, gop, op_map)
        trusted = match.confidence in {"high", "medium"} and by_npu[ni] == by_gpu[gi] == 1
        strong_links = {"external_id", "runtime_correlation", "torch_to_npu", "host_to_device_queue", "host_to_device_thread"}
        owner_ok = all(npu.kernels[i].link_method in strong_links for i in nk)
        gpu_owner_ok = all(gpu.kernels[i].link_method in strong_links for i in gscope[gi]) and all(
            gpu.memory_activities[i].link_method in strong_links for i in mscope[gi])
        cpu_status = "元数据真实冲突" if differences else "已对齐，但元数据不完整" if missing else "已对齐"
        if not trusted:
            cpu_status = "CPU 匹配待核实"
        issues = (["冲突：" + ",".join(differences)] if differences else []) + (["缺失：" + ",".join(missing)] if missing else [])
        if len(nk) != len(all_nk):
            issues.append("scope 含未进入 kernel_details 的 NPU 任务")
        if not owner_ok or not gpu_owner_ok:
            issues.append("kernel owner 关联证据不足")
        append("call:npu:" + str(ni), nop.canonical_name, nk, gscope[gi], mscope[gi], ni, gi,
               source="matched_cpu_scope", basis=match.alignment_basis, cpu_status=cpu_status,
               comparable=trusted and not issues, issues="；".join(issues))

    # A call may overlap a parent/child or an existing specialized bundle.
    users = defaultdict(list)
    for row in result:
        for side, field in (("npu", "NPU Kernel Indices"), ("gpu", "GPU Kernel Indices"), ("memory", "GPU Memory Activity Indices")):
            for i in ids(row, field):
                users[(side, i)].append(row["Call ID"])
    for row in result:
        overlap = set()
        for side, field in (("npu", "NPU Kernel Indices"), ("gpu", "GPU Kernel Indices"), ("memory", "GPU Memory Activity Indices")):
            for i in ids(row, field):
                overlap.update(users[(side, i)])
        row["Overlap Call IDs"] = encoded(sorted(overlap - {row["Call ID"]}))
    return sorted(result, key=lambda r: (r["Group ID"], r["Call ID"]))


def summarize_whole_calls(rows, gpu=None, npu=None):
    groups = defaultdict(list)
    for row in rows:
        groups[row["Group ID"]].append(row)
    result = []
    for group_id, group in groups.items():
        eligible, seen = [], {field: set() for field in ("NPU Kernel Indices", "GPU Kernel Indices", "GPU Memory Activity Indices")}
        # Stable representative when a parent and child have identical group
        # composition; never count the same event twice in a group ratio.
        selected = []
        for row in sorted(group, key=lambda r: (not r["Comparable"], r["Call ID"])):
            if any(seen[field].intersection(ids(row, field)) for field in seen):
                continue
            selected.append(row)
            for field in seen:
                seen[field].update(ids(row, field))
            if row["Comparable"]:
                eligible.append(row)
        nt = sum(r["NPU Total Time(us)"] for r in eligible)
        gt = sum(r["GPU Total Time(us)"] for r in eligible)
        result.append({
            **{key: group[0][key] for key in ("Group ID", "Whole OP", "Device_id", "NPU Types", "GPU Composition")},
            "Observed Call Count": len(group), "Comparable Call Count": len(eligible),
            "Pending Call Count": sum(not r["Comparable"] for r in group),
            "Overlapping Call Count": len(group) - len(selected),
            "Comparable NPU Time(us)": round(nt, 6), "Comparable GPU Time(us)": round(gt, 6),
            "Comparable Ratio(GPU/NPU)": round(gt / nt, 6) if nt else "",
            "Total Time Ratio(GPU/NPU)": "", "NPU Population Call Count": "", "GPU Population Call Count": "",
            "Population Status": "仅已关联 scope；未建立全部调用分组，总体比值不可用",
            "Comparison Status": "全部已关联调用可比较" if len(eligible) == len(group) else "部分可比较" if eligible else "待核实",
            "Call IDs": encoded(r["Call ID"] for r in group), "Scope": SCOPE,
            "Included Call IDs": encoded(r["Call ID"] for r in selected),
            "Comparable Call IDs": encoded(r["Call ID"] for r in eligible),
        })
    if gpu is not None and npu is not None:
        add_population_totals(result, gpu, npu)
    return sorted(result, key=lambda r: (-r["Comparable NPU Time(us)"], r["Whole OP"], r["Group ID"]))


def add_population_totals(groups, gpu, npu):
    """Independent census including unmatched calls, only for unique patterns.

    When unmatched scopes cannot be assigned to a cross-backend composition,
    leave populations/ratio unavailable instead of using paired-only totals.
    """
    gs, ms, ns = scoped_indices(gpu), scoped_indices(gpu, True), scoped_indices(npu)
    census = {}
    def calls(profile, name):
        indices = []
        for i, op in enumerate(profile.ops):
            if op.canonical_name != name:
                continue
            parent = op.parent_index
            while parent is not None and profile.ops[parent].canonical_name != name:
                parent = profile.ops[parent].parent_index
            if parent is None:
                indices.append(i)
        return indices
    for row in groups:
        row.update({"NPU Population Time(us)": "", "GPU Population Time(us)": ""})
        name = row["Whole OP"]
        if not name.startswith("aten::"):
            continue
        if name not in census:
            census[name] = (calls(npu, name), calls(gpu, name))
        ni, gi = census[name]
        if not ni or not gi:
            continue
        if any(not ns[i] or any(npu.kernels[k].detail is None for k in ns[i]) for i in ni):
            continue
        if any(not gs[i] and not ms[i] for i in gi):
            continue
        patterns = {composition(npu.kernels[k].detail.op_type for k in ns[i]) for i in ni}
        devices = {",".join(sorted({npu.kernels[k].detail.device_id for k in ns[i]})) for i in ni}
        gpu_patterns = {composition([gpu_label(gpu.kernels[k].event.name) for k in gs[i]] +
                                   [gpu.memory_activities[k].event.name for k in ms[i]]) for i in gi}
        if patterns != {row["NPU Types"]} or gpu_patterns != {row["GPU Composition"]} or devices != {row["Device_id"]}:
            continue
        nk = {k for i in ni for k in ns[i]}
        gk = {k for i in gi for k in gs[i]}
        gm = {k for i in gi for k in ms[i]}
        nt = sum(npu.kernels[k].event.duration_us for k in nk)
        gt = sum(gpu.kernels[k].event.duration_us for k in gk) + sum(gpu.memory_activities[k].event.duration_us for k in gm)
        row.update({"NPU Population Call Count": len(ni), "GPU Population Call Count": len(gi),
                    "NPU Population Time(us)": round(nt, 6), "GPU Population Time(us)": round(gt, 6),
                    "Total Time Ratio(GPU/NPU)": round(gt / nt, 6) if nt else "",
                    "Population Status": "同名逻辑调用全量、单一实现组合；含未配对调用，不保证工作量相等"})


def annotate_type_calls(type_rows, mapping_rows, calls, precision_rows, diagnostics):
    """Add navigation/status only; original type time/coverage remain unchanged."""
    by_kernel, precision = defaultdict(list), set()
    for call in calls:
        for i in ids(call, "NPU Kernel Indices"):
            by_kernel[i].append(call)
    for row in precision_rows:
        if row.get("Comparable") is True:
            precision.update(ids(row, "NPU Kernel Indices"))
    kernels = defaultdict(set)
    for row in mapping_rows:
        kernels[tuple(str(row[k]) for k in ("Device_id", "OP Type", "Core Type"))].add(row["NPU Kernel Index"])
    diag = defaultdict(list)
    for row in diagnostics["kernels"]:
        diag[(row["device_id"], row["op_type"], row["core_type"])].append(row)
        linked = by_kernel[row["npu_kernel_index"]]
        row["logical_call_references"] = [{"call_id": c["Call ID"], "group_id": c["Group ID"], "comparable": c["Comparable"]} for c in linked]
        if any(c["Comparable"] for c in linked):
            row["kernel_comparison_status"] = "已纳入整体比较，无法独立归因"
        elif linked:
            row["kernel_comparison_status"] = "待核实整体比较"
    for row in type_rows:
        key = tuple(str(row[k]) for k in ("Device_id", "OP Type", "Core Type"))
        indices = kernels[key]
        linked = {c["Call ID"]: c for i in indices for c in by_kernel[i]}
        covered = {i for i in indices if i in precision or any(c["Comparable"] for c in by_kernel[i])}
        row["Whole Comparison Status"] = ("全部纳入整体比较" if indices and covered == indices else
            "部分纳入整体比较" if covered else "有待核实整体调用" if linked else "尚无整体比较")
        row["Related Whole OPs"] = " / ".join(sorted({c["Whole OP"] for c in linked.values()}) + (["排查报告"] if indices & precision else []))
        row["Whole Group IDs"] = encoded(sorted({c["Group ID"] for c in linked.values()}))
        row["Whole Call IDs"] = encoded(sorted(linked))
        row["CPU Alignment Status"] = " / ".join(sorted({d["cpu_alignment_status"] for d in diag[key]}))
        row["Kernel Comparison Status"] = " / ".join(sorted({d["kernel_comparison_status"] for d in diag[key]}))

"""Compare two kernel_details populations; no trace, CPU anchor or cross-run event IDs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import re
import sys

from .parser import _normalized_id, _row_lookup, iter_kernel_details
from .core_breakdown import CORE_RULE, RULE, category, core_category, counter
from .pipeline_analysis import (ADVICE, DOMINANCE_MARGIN, LABELS, METHOD, THRESHOLD,
                                classify as classify_pipeline, summarize_buckets)
from .provenance import write_json
from .xlsx_writer import TableSheet, write_grouped_workbook_xlsx

REPORTS = ("npu_pair.xlsx", "npu_pair.json")
METRICS = ("基线数量", "对比数量", "基线耗时(us)", "对比耗时(us)", "性能比(基线/对比)")
SIGNATURE = ("OP Type", "Core Type", "输入 Shape", "输入 dtype", "输出 Shape", "输出 dtype")
# 名称 is not part of the grouping key: kernel names may vary inside one shape group, so the
# sheet shows the dominant one instead of splitting the group.
SHAPE_COLUMNS = ("OP Type", "名称", "Core Type", "输入 Shape", "输入 dtype", "输出 Shape", "输出 dtype")
BELOW_COLUMNS = ("OP Type", "名称", "Core Type", "CV", "对比数量", "基线数量",
                 "对比耗时(us)", "基线耗时(us)", "性能比(基线/对比)", "阈值",
                 "输入 Shape", "输入 dtype", "输出 Shape", "输出 dtype")
PIPELINE_OVERVIEW = ("侧", "流水类别", "标签", "次数", "耗时(us)", "耗时占比(%)", "下一步检查")
PIPELINE_TYPE = ("侧", "OP Type", "Core Type", "流水类别", "标签", "次数", "耗时(us)", "该类型内占比(%)")
PIPELINE_NOTE = ("流水初筛候选，非 Roofline 饱和结论；类别耗时为 kernel Duration 之和且互斥，"
                 "占比分母为该侧所选范围的全部 kernel，含无法分类的 Unknown。")
DEFAULT_CUBE_THRESHOLD = 2.0
DEFAULT_VECTOR_THRESHOLD = 1.1
# The band turns each threshold into four classes: >=1.5x, >=1x, >=0.5x, <0.5x.
THRESHOLD_BAND = 0.5
COLOR_RULE = ("性能比 >= 阈值 × 1.5 深绿；>= 阈值 绿；>= 阈值 × 0.5 浅黄；< 阈值 × 0.5 深黄；"
              "Mixed / Unknown 不套用阈值，不着色")


def step_id(value):
    text = _normalized_id(value)
    match = re.fullmatch(r"ProfilerStep#(\d+)", text, re.I)
    return match.group(1) if match else text


def counter_available(path):
    """True when the file carries at least one usable aicore_time value.

    Reads the header first, so a CSV without the column costs one line; a column that is
    present but empty costs one scan instead of silently reporting every kernel as Unknown.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or ()
        if not _row_lookup({name: name for name in names},
                           "aicore_time(us)", "aicore_time(μs)", "aicore_time"):
            return False
        for row in reader:
            if counter(_row_lookup(row, "aicore_time(us)", "aicore_time(μs)", "aicore_time")) is not None:
                return True
    return False


def _rows(path, side):
    try:
        for d in iter_kernel_details(path):
            if not d.op_type or d.duration_us is None or d.duration_us < 0:
                raise ValueError(f"{side} 第 {d.row_index} 行缺少 Type 或 Duration(us) 无效")
            yield d
    except (csv.Error, UnicodeError) as exc:
        raise ValueError(f"{side} 不是有效的 UTF-8 kernel_details.csv") from exc


def inspect_side(path, side):
    devices, steps = defaultdict(Counter), Counter()
    for d in _rows(path, side):
        step = step_id(d.step_id)
        devices[d.device_id][step] += 1
        steps[step] += 1
    if not devices:
        raise ValueError(f"{side} CSV 没有 kernel 明细")
    return {"devices": dict(devices), "steps": dict(steps), "count": sum(steps.values())}


def collect(path, side, step, device, source="counter"):
    # Keep each view disjoint within a side. Stream/task IDs and row order are run-local.
    types = defaultdict(lambda: [0, 0.0, set(), {}])
    signatures = defaultdict(lambda: [0, 0.0, set(), Counter()])
    pipes = defaultdict(lambda: [0, 0.0])
    buckets = ("Cube", "Vector", "Mixed", "Unknown") if source == "core" else ("Cube", "Vector", "Unknown")
    cv = {name: {"count": 0, "time_us": 0.0} for name in buckets}
    devices, selected_steps = set(), Counter()
    selected, skipped_unmarked = 0, 0
    ratios = None  # Decided from the first selected row; the CSV has one fixed schema.
    for d in _rows(path, side):
        devices.add(d.device_id)
        if device and d.device_id != device:
            continue
        label = step_id(d.step_id)
        if step and label != step:
            skipped_unmarked += not label
            continue
        selected_steps[label] += 1
        selected += 1
        signature = (d.op_type, d.core_type, d.input_shapes, d.input_dtypes, d.output_shapes, d.output_dtypes)
        # Without a counter the core type is the only evidence left, and it is labelled as such.
        cls = category(d.aicore_time_us) if source == "counter" else core_category(d.core_type)
        cv[cls]["count"] += 1
        cv[cls]["time_us"] += d.duration_us
        if ratios is None:
            ratios = any(name.endswith("_ratio") for name in d.raw)
        pipe = classify_pipeline(d)["category"] if ratios else "unknown"
        pipes[pipe][0] += 1
        pipes[pipe][1] += d.duration_us
        stats = types[signature[:2]]
        stats[0] += 1
        stats[1] += d.duration_us
        stats[2].add(cls)
        bucket = stats[3].setdefault(pipe, [0, 0.0])
        bucket[0] += 1
        bucket[1] += d.duration_us
        shape_stats = signatures[signature]
        shape_stats[0] += 1
        shape_stats[1] += d.duration_us
        shape_stats[2].add(cls)
        shape_stats[3][d.name] += 1
    if not device and len(devices) > 1:
        raise ValueError(f"{side} 含多个 Device ID，请指定一个：{', '.join(sorted(filter(None, devices)))}")
    if not selected:
        raise ValueError(f"{side} 所选 Step / Device 没有 kernel；无 Step ID 时只能选全量")
    scope = {"step": step, "device": device or next(iter(devices)), "steps": dict(selected_steps),
             "excluded_unmarked": skipped_unmarked}
    total = math.fsum(g["time_us"] for g in cv.values())
    for g in cv.values():
        g["time_pct"] = 100 * g["time_us"] / total if total else None
    scope["cv"] = cv
    scope["cv_source"] = source
    scope["cv_rule"] = RULE if source == "counter" else CORE_RULE
    scope["pipeline"] = summarize_buckets(pipes)
    return types, signatures, scope


def metrics(base, candidate):
    a, b = base[1], candidate[1]
    # Performance ratio inverts elapsed time: >1 means the candidate side needs less time.
    factor = a/b if base[0] and candidate[0] and a and b else None
    status = "两侧均有" if base[0] and candidate[0] else "仅基线" if base[0] else "仅对比"
    # Several of these are no longer shown as columns; the conclusions and the below-threshold
    # export still use them, so the row keeps the full set.
    return {
        "基线数量": base[0], "对比数量": candidate[0], "基线耗时(us)": a, "对比耗时(us)": b,
        "差值(us)": b - a, "性能比(基线/对比)": factor,
        "变化(%)": (factor-1)*100 if factor is not None else None,
        "基线平均(us)": a/base[0] if base[0] else None,
        "对比平均(us)": b/candidate[0] if candidate[0] else None,
        "状态": status,
    }


def validate_thresholds(cube, vector):
    limits = {}
    for name, value in (("cube", cube), ("vector", vector)):
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} 阈值必须是数字") from None
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} 阈值必须大于 0")
        limits[name] = number
    return limits


def ratio_color(ratio, cv, limits):
    """Four classes per CV type; a class without a threshold stays uncoloured."""
    try:
        limit = limits["cube"] if cv == "Cube" else limits["vector"] if cv == "Vector" else None
    except (KeyError, TypeError):
        limit = None
    if ratio is None or not limit:
        return None
    # Round the derived bounds so that e.g. 1.1 * 1.5 compares as 1.65, not 1.6500000000000001.
    upper = round(limit * (1 + THRESHOLD_BAND), 9)
    lower = round(limit * (1 - THRESHOLD_BAND), 9)
    if ratio >= upper:
        return "dark-green"
    if ratio >= limit:
        return "green"
    if ratio >= lower:
        return "yellow"
    return "dark-yellow"


def table(base, candidate, columns, limits, decorate=None):
    rows = []
    for k in sorted(base.keys() | candidate.keys()):
        a, b = base.get(k, (0, 0, set())), candidate.get(k, (0, 0, set()))
        classes = a[2] | b[2]
        cls = next(iter(classes)) if len(classes) == 1 else "Mixed"
        row = {**dict(zip(columns, k)), "CV": cls, **metrics(a, b)}
        if decorate:
            row.update(decorate(a, b))
        row["ratio_color"] = ratio_color(row["性能比(基线/对比)"], cls, limits)
        rows.append(row)
    return sorted(rows, key=lambda row: -abs(row["差值(us)"]))


def dominant_name(*sides):
    """Most frequent kernel name inside a shape group; kernel names may repeat across calls."""
    counts = Counter()
    for stats in sides:
        names = stats[3] if len(stats) > 3 else None
        if isinstance(names, Counter):
            counts.update(names)
    return counts.most_common(1)[0][0] if counts else ""


def excel_rows(rows):
    # Keep undefined ratios/means blank rather than rendering Python's None as text.
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


def below_threshold(rows, limits):
    """Shape-level rows that miss their CV threshold, worst performance ratio first."""
    selected = []
    for row in rows:
        cv = row.get("CV")
        limit = limits["cube"] if cv == "Cube" else limits["vector"] if cv == "Vector" else None
        ratio = row.get("性能比(基线/对比)")
        if ratio is None or not limit or ratio >= limit:
            continue
        selected.append({**row, "阈值": limit, "ratio_color": ratio_color(ratio, cv, limits)})
    return sorted(selected, key=lambda row: row.get("性能比(基线/对比)"))


def write_below_threshold_xlsx(path, rows):
    write_grouped_workbook_xlsx(Path(path), [], [
        TableSheet("未达标算子", BELOW_COLUMNS, rows=excel_rows(rows))])


def pipeline_overview(baseline, candidate):
    rows = []
    for side, scope in (("基线", baseline), ("对比", candidate)):
        for group in scope["pipeline"]["categories"]:
            rows.append({"侧": side, "流水类别": group["category"], "标签": group["label"],
                         "次数": group["count"], "耗时(us)": group["time_us"],
                         "耗时占比(%)": group["time_pct"], "下一步检查": ADVICE[group["category"]]})
    return rows


def pipeline_types(groups, side):
    rows = []
    for key in sorted(groups):
        time_us, pipes = groups[key][1], groups[key][3]
        for name, (count, cat_us) in sorted(pipes.items(), key=lambda item: -item[1][1]):
            rows.append({"侧": side, "OP Type": key[0], "Core Type": key[1], "流水类别": name,
                         "标签": LABELS[name], "次数": count, "耗时(us)": cat_us,
                         "该类型内占比(%)": 100 * cat_us / time_us if time_us else None})
    return rows


def _pipeline_headline(scope):
    """Dominant classifiable pipeline, or None when nothing could be classified."""
    groups = scope["pipeline"]["categories"]
    top = max((g for g in groups if g["category"] != "unknown"), default=None, key=lambda g: g["time_us"])
    if top is None:
        return None
    unknown = next((g for g in groups if g["category"] == "unknown"), None)
    text = f"主导 {top['label']}（{top['time_pct']:.1f}%）"
    return text + (f"，另有 {unknown['time_pct']:.1f}% 无法分类" if unknown and unknown["time_pct"] else "")


def conclusions(overall, rows, baseline, candidate):
    """Ordered, deterministic reading of the same numbers; never a new measurement."""
    items = []

    def add(kind, text):
        items.append({"kind": kind, "text": text})

    base_us, cand_us = overall["基线耗时(us)"], overall["对比耗时(us)"]
    ratio = overall["性能比(基线/对比)"]
    if ratio is None:
        add("总体", "所选范围给不出总体性能比：两侧没有同时出现 kernel，或至少一侧耗时为 0。"
                    "请先核对 Step、Device 与采集范围，再判断性能变化。")
    else:
        change = (ratio - 1) * 100
        verdict = "快" if change > 0 else "慢" if change < 0 else "持平"
        add("总体", f"对比侧 {cand_us:g} μs、基线 {base_us:g} μs，总体性能比 {ratio:.3f}，"
                    f"即对比侧{verdict}{'' if not change else f' {abs(change):.1f}%'}。"
                    "这是 kernel Duration 之和，含并发重叠，不是 Step 墙钟时间。")

    both = [row for row in rows if row["状态"] == "两侧均有"]
    rated = [row for row in both if row["性能比(基线/对比)"] is not None]
    faster = sum(row["性能比(基线/对比)"] > 1 for row in rated)
    slower = sum(row["性能比(基线/对比)"] < 1 for row in rated)
    if both:
        add("类型", f"{len(both)} 个类型两侧都有 kernel，其中 {len(rated)} 个可算比值："
                    f"变快 {faster} 个、变慢 {slower} 个、持平 {len(rated) - faster - slower} 个。")
    elif rows:
        add("类型", "两侧没有同时出现的类型，因此没有类型级性能比可用。")

    movers = [row for row in sorted(rated, key=lambda item: -abs(item["差值(us)"]))[:3]
              if row["差值(us)"]]
    if movers:
        for row in movers:
            name = f"{row['OP Type']} / {row['Core Type'] or '未标记 Core'}"
            add("主要差异", f"{name} {'多花' if row['差值(us)'] > 0 else '少花'} {abs(row['差值(us)']):g} μs，"
                            f"性能比 {row['性能比(基线/对比)']:.3f}（{row['基线数量']} → {row['对比数量']} 次）。")
    elif rated:
        add("主要差异", "两侧各类型的累计耗时差均为 0，没有可排序的主要差异。")

    only_base = [row for row in rows if row["状态"] == "仅基线"]
    only_cand = [row for row in rows if row["状态"] == "仅对比"]
    if only_base or only_cand:
        add("覆盖面", f"仅在基线出现 {len(only_base)} 个类型（{math.fsum(r['基线耗时(us)'] for r in only_base):g} μs），"
                      f"仅在对比侧出现 {len(only_cand)} 个类型（{math.fsum(r['对比耗时(us)'] for r in only_cand):g} μs）；"
                      "这些类型没有独立性能比，需先确认两侧 batch、精度与工作量是否等价。")

    shifts = [(name, baseline["cv"][name]["time_pct"], candidate["cv"][name]["time_pct"])
              for name in ("Cube", "Vector", "Unknown") if name in baseline["cv"] and name in candidate["cv"]
              if baseline["cv"][name]["time_pct"] is not None and candidate["cv"][name]["time_pct"] is not None]
    estimated = "core" in (baseline.get("cv_source"), candidate.get("cv_source"))
    basis = ("Cube/Vector 由 Accelerator Core 估计，不是实测计数器" if estimated else
             "占比分母是各侧全部 kernel 的 Duration，不是硬件利用率")
    if shifts:
        name, before, after = max(shifts, key=lambda item: abs(item[2] - item[1]))
        if baseline["cv"]["Unknown"]["time_pct"] == 100 and candidate["cv"]["Unknown"]["time_pct"] == 100:
            add("结构", "两侧都拿不到 Cube/Vector 分类：既没有 aicore_time 计数，也没有可识别的 Accelerator Core。")
        elif abs(after - before) < 0.05:
            add("结构", f"两侧 Cube/Vector/Unknown 的时间占比一致，没有可辨识的结构变化；{basis}。")
        else:
            add("结构", f"{name} 时间占比 {before:.1f}% → {after:.1f}%（{after - before:+.1f}pp），"
                        f"为两侧占比变化最大的一类；{basis}。")

    headlines = (_pipeline_headline(baseline), _pipeline_headline(candidate))
    if headlines == (None, None):
        add("流水", "两侧都没有可分类的流水指标（PipeUtilization 未采集，或核类型不支持初筛），本次不做流水初筛。")
    else:
        add("流水", "；".join(f"{side}{head or '没有可分类的流水指标（指标未采集或核类型不支持）'}"
                             for side, head in zip(("基线", "对比侧"), headlines)) +
            "。流水类别按每 kernel 的 PipeUtilization 初筛得出，不能相加，也不构成 Roofline 饱和结论。")
    add("边界", "性能比为跨采集的聚合比较，不是逐调用对齐；只在两侧都有 kernel 且耗时非零时给出。"
                "未映射、未覆盖的工作仍计入各自的总耗时。")
    return items


def compare_npus(baseline_details, candidate_details, output_dir, *, baseline_step="", candidate_step="",
                 baseline_device="", candidate_device="", all_steps=False,
                 cube_threshold=DEFAULT_CUBE_THRESHOLD, vector_threshold=DEFAULT_VECTOR_THRESHOLD):
    steps = tuple(map(step_id, (baseline_step, candidate_step)))
    if (all_steps and any(steps)) or (not all_steps and not all(steps)):
        raise ValueError("请选择两侧 Step，或明确选择 --all-steps；不可混用")
    limits = validate_thresholds(cube_threshold, vector_threshold)
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("输出目录必须是新目录或空目录")
    # Prefer the measured counter; fall back to the core type only when a side has none.
    base_source = "counter" if counter_available(baseline_details) else "core"
    cand_source = "counter" if counter_available(candidate_details) else "core"
    a, sa, base_scope = collect(baseline_details, "基线", steps[0], _normalized_id(baseline_device), base_source)
    b, sb, cand_scope = collect(candidate_details, "对比", steps[1], _normalized_id(candidate_device), cand_source)
    types = table(a, b, SIGNATURE[:2], limits)
    shapes = table(sa, sb, SIGNATURE, limits, decorate=lambda x, y: {"名称": dominant_name(x, y)})
    def totals(groups):
        return sum(x[0] for x in groups.values()), math.fsum(x[1] for x in groups.values())
    overall = metrics(totals(a), totals(b))
    findings = conclusions(overall, types, base_scope, cand_scope)
    report = {"mode": "npu_pair", "schema_version": 3, "baseline": base_scope, "candidate": cand_scope,
              "overall": overall, "conclusions": findings,
              "thresholds": {"cube": limits["cube"], "vector": limits["vector"],
                             "band": THRESHOLD_BAND, "rule": COLOR_RULE},
              "pipeline": {"version": 1, "method": METHOD, "threshold": THRESHOLD,
                           "dominance_margin": DOMINANCE_MARGIN, "note": PIPELINE_NOTE,
                           "baseline_types": pipeline_types(a, "基线"),
                           "candidate_types": pipeline_types(b, "对比")},
              "types": types, "signatures": shapes}
    out.mkdir(parents=True, exist_ok=True)
    overview = {"基线设备": base_scope["device"], "基线 Step": steps[0] or "全量",
                "对比设备": cand_scope["device"], "对比 Step": steps[1] or "全量",
                "基线未标记排除数": base_scope["excluded_unmarked"], "对比未标记排除数": cand_scope["excluded_unmarked"], **overall}
    cv_rows = [{"侧": side, "CV": cls, "数量": g["count"], "耗时(us)": g["time_us"], "占比(%)": g["time_pct"]}
               for side, scope in (("基线", base_scope), ("对比", cand_scope)) for cls, g in scope["cv"].items()]
    pipe_rows = pipeline_overview(base_scope, cand_scope)
    # One field per row reads better than a single very wide row of label/value pairs.
    overview_rows = [{"字段": name, "值": "" if value is None else value} for name, value in overview.items()]
    missed = below_threshold(shapes, limits)
    # conclusions() speaks kind/text; the sheet shows them under Chinese headers.
    finding_rows = [{"类别": row["kind"], "结论": row["text"]} for row in findings]
    write_grouped_workbook_xlsx(out/REPORTS[0], [], [
        TableSheet("结论", ("类别", "结论"), rows=finding_rows, wrap_columns=("结论",), min_row_height=30),
        TableSheet("类型对比", (*SIGNATURE[:2], "CV", *METRICS), rows=excel_rows(types)),
        TableSheet("未达标算子", BELOW_COLUMNS, rows=excel_rows(missed)),
        TableSheet("整体", ("字段", "值"), rows=excel_rows(overview_rows)),
        TableSheet("流水总览", PIPELINE_OVERVIEW, rows=excel_rows(pipe_rows), wrap_columns=("下一步检查",)),
        TableSheet("流水-类型", PIPELINE_TYPE,
                   rows=excel_rows(report["pipeline"]["baseline_types"] +
                                   report["pipeline"]["candidate_types"])),
        TableSheet("Cube-Vector总览", tuple(cv_rows[0]), rows=excel_rows(cv_rows)),
        TableSheet("Shape与精度", (*SHAPE_COLUMNS, "CV", *METRICS), rows=excel_rows(shapes))])
    write_json(out/REPORTS[1], report)
    return report


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m profiling_tools compare-npu", description=__doc__)
    for side in ("baseline", "candidate"):
        parser.add_argument(f"--{side}-details", required=True)
        parser.add_argument(f"--{side}-step", default="")
        parser.add_argument(f"--{side}-device", default="")
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--cube-threshold", type=float, default=DEFAULT_CUBE_THRESHOLD,
                        help=f"Cube 性能比达标阈值（默认 {DEFAULT_CUBE_THRESHOLD}）")
    parser.add_argument("--vector-threshold", type=float, default=DEFAULT_VECTOR_THRESHOLD,
                        help=f"Vector 性能比达标阈值（默认 {DEFAULT_VECTOR_THRESHOLD}）")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_npus(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"compare-npu: {exc}\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False))
    for item in report["conclusions"]:
        print(f"[{item['kind']}] {item['text']}")
    print(f"Reports: {args.output_dir}")
    return 0

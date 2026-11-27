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

from .parser import _normalized_id, iter_kernel_details
from .provenance import write_json
from .xlsx_writer import TableSheet, write_grouped_workbook_xlsx

REPORTS = ("npu_pair.xlsx", "npu_pair.json")
METRICS = ("基线数量", "对比数量", "基线耗时(us)", "对比耗时(us)", "差值(us)", "耗时比(对比/基线)",
           "变化(%)", "基线平均(us)", "对比平均(us)", "状态")
SIGNATURE = ("OP Type", "Core Type", "输入 Shape", "输入 dtype", "输出 Shape", "输出 dtype")


def step_id(value):
    text = _normalized_id(value)
    match = re.fullmatch(r"ProfilerStep#(\d+)", text, re.I)
    return match.group(1) if match else text


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


def collect(path, side, step, device):
    # Keep each view disjoint within a side. Stream/task IDs and row order are run-local.
    types, signatures = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    devices, selected_steps = set(), Counter()
    selected, skipped_unmarked = 0, 0
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
        for stats in (types[(d.op_type,)], signatures[signature]):
            stats[0] += 1
            stats[1] += d.duration_us
    if not device and len(devices) > 1:
        raise ValueError(f"{side} 含多个 Device ID，请指定一个：{', '.join(sorted(filter(None, devices)))}")
    if not selected:
        raise ValueError(f"{side} 所选 Step / Device 没有 kernel；无 Step ID 时只能选全量")
    scope = {"step": step, "device": device or next(iter(devices)), "steps": dict(selected_steps),
             "excluded_unmarked": skipped_unmarked}
    return types, signatures, scope


def metrics(base, candidate):
    a, b = base[1], candidate[1]
    factor = b/a if base[0] and candidate[0] and a else None
    status = "两侧均有" if base[0] and candidate[0] else "仅基线" if base[0] else "仅对比"
    return dict(zip(METRICS, (base[0], candidate[0], a, b, b-a, factor,
                             (factor-1)*100 if factor is not None else None,
                             a/base[0] if base[0] else None, b/candidate[0] if candidate[0] else None, status)))


def table(base, candidate, columns):
    rows = [{**dict(zip(columns, k)), **metrics(base.get(k, (0, 0)), candidate.get(k, (0, 0)))}
            for k in sorted(base.keys() | candidate.keys())]
    return sorted(rows, key=lambda row: -abs(row["差值(us)"]))


def compare_npus(baseline_details, candidate_details, output_dir, *, baseline_step="", candidate_step="",
                 baseline_device="", candidate_device="", all_steps=False):
    steps = tuple(map(step_id, (baseline_step, candidate_step)))
    if (all_steps and any(steps)) or (not all_steps and not all(steps)):
        raise ValueError("请选择两侧 Step，或明确选择 --all-steps；不可混用")
    out = Path(output_dir)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("输出目录必须是新目录或空目录")
    a, sa, base_scope = collect(baseline_details, "基线", steps[0], _normalized_id(baseline_device))
    b, sb, cand_scope = collect(candidate_details, "对比", steps[1], _normalized_id(candidate_device))
    types, shapes = table(a, b, SIGNATURE[:1]), table(sa, sb, SIGNATURE)
    def totals(groups):
        return sum(x[0] for x in groups.values()), math.fsum(x[1] for x in groups.values())
    overall = metrics(totals(a), totals(b))
    report = {"mode": "npu_pair", "schema_version": 1, "baseline": base_scope, "candidate": cand_scope,
              "overall": overall, "types": types, "signatures": shapes}
    out.mkdir(parents=True, exist_ok=True)
    overview = {"基线设备": base_scope["device"], "基线 Step": steps[0] or "全量",
                "对比设备": cand_scope["device"], "对比 Step": steps[1] or "全量",
                "基线未标记排除数": base_scope["excluded_unmarked"], "对比未标记排除数": cand_scope["excluded_unmarked"], **overall}
    def excel_rows(rows):
        # Keep undefined ratios/means blank rather than rendering Python's None as text.
        return [{k: v for k, v in row.items() if v is not None} for row in rows]
    write_grouped_workbook_xlsx(out/REPORTS[0], [], [
        TableSheet("类型对比", (*SIGNATURE[:1], *METRICS), rows=excel_rows(types)),
        TableSheet("整体", tuple(overview), rows=excel_rows([overview])),
        TableSheet("Shape与精度", (*SIGNATURE, *METRICS), rows=excel_rows(shapes))])
    write_json(out/REPORTS[1], report)
    return report


def main(argv):
    parser = argparse.ArgumentParser(prog="python -m profiling_tools compare-npu", description=__doc__)
    for side in ("baseline", "candidate"):
        parser.add_argument(f"--{side}-details", required=True)
        parser.add_argument(f"--{side}-step", default="")
        parser.add_argument(f"--{side}-device", default="")
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:
        report = compare_npus(**vars(args))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"compare-npu: {exc}\n")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(report["overall"], ensure_ascii=False))
    print(f"Reports: {args.output_dir}")
    return 0

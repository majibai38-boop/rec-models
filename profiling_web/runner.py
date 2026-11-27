"""One disposable analysis child. Not exposed as an HTTP command runner."""
import json
import os
from pathlib import Path
import sys
import traceback
import threading
import time

import psutil

from profiling_tools.compare import CompareConfig, InputSelectionError, compare_profiles
from profiling_tools.inspection import inspect_inputs
from profiling_tools.provenance import write_json
from .index import build_index, public_summary


def run(task_path):
    task_path = Path(task_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    attempt = task_path.parent
    def watch_parent():
        while True:
            time.sleep(1)
            try:
                parent = psutil.Process(task["parent"]["pid"])
                if abs(parent.create_time() - task["parent"]["created"]) < .01 and parent.is_running():
                    continue
            except psutil.Error:
                pass
            os._exit(143)
    if "parent" in task:
        threading.Thread(target=watch_parent, daemon=True).start()

    def progress(stage):
        write_json(attempt / "progress.json", {"stage": stage})

    try:
        if task.get("mode") == "npu_pair":
            from profiling_tools.npu_pair import REPORTS, compare_npus, inspect_side
            opts = dict(task["config"])
            opts.pop("require_explicit_scope")
            try:
                if task["kind"] == "inspect":
                    progress("inspection")
                    inventory = {k: inspect_side(opts[k+"_details"], label) for k, label in
                                 (("baseline", "基线"), ("candidate", "对比"))}
                    result = {"status": "needs_selection", "inspection": inventory}
                else:
                    progress("reports")
                    report = compare_npus(**opts)
                    from . import npu_pair_index
                    npu_pair_index.build(attempt / "npu_pair.sqlite", report)
                    summary = {k: report[k] for k in ("overall", "baseline", "candidate")}
                    summary.update(schema_version=report["schema_version"], reports=list(REPORTS), type_count=len(report["types"]))
                    result = {"status": "succeeded", "summary": summary}
            except ValueError as exc:
                message = str(exc)
                for key in ("baseline_details", "candidate_details"):
                    message = message.replace(opts[key], key)
                result = {"status": "failed", "error": message[:500]}
            write_json(attempt / "result.json", result)
            return 0 if result["status"] != "failed" else 1
        config_args = dict(task["config"])
        review = config_args.pop("export_call_tables", False)
        aggregate = config_args.pop("aggregate_all_job", "")
        config = CompareConfig(**config_args)
        if task["kind"] == "inspect":
            progress("inspection")
            report = inspect_inputs(config)
            for side, profile in report["profiles"].items():
                profile["path"] = side + "_trace"
            if "path" in report["kernel_details"]:
                report["kernel_details"]["path"] = "npu_kernel_details"
            for group in ("errors", "warnings", "selections_required"):
                for item in report[group]:
                    # Keep useful validation messages without server input paths.
                    for key in ("gpu_trace", "npu_trace", "npu_kernel_details"):
                        item["message"] = item["message"].replace(task["config"][key], key)
            write_json(attempt / "result.json", {"status": "failed" if report["status"] == "invalid" else "needs_selection", "inspection": report,
                                                 "error": "预检查失败，请查看输入错误并新建任务" if report["status"] == "invalid" else ""})
        else:
            state = task.get('manual',{}) if task['kind'] == 'manual' else {}
            base = None
            if state:
                import re
                if not re.fullmatch(r'[0-9a-f]{32}',state['base_attempt']):
                    raise ValueError('Invalid base attempt')
                base = json.loads((attempt.parent/state['base_attempt']/'reports/run_manifest.json').read_text(encoding='utf-8'))
            summary = compare_profiles(config, progress=progress, manual_enabled=True,
                                       manual_links=state.get('links',[]), manual_base=base,
                                       cache_dir=attempt.parent.parent/'analysis-cache')
            if summary["validation"]["status"] != "valid":
                raise ValueError("Result validation failed")
            progress("indexing")
            index_started = time.perf_counter()
            build_index(config.output_dir, attempt / "index.sqlite")
            summary['timing_seconds']['index_build'] = round(time.perf_counter() - index_started, 6)
            write_json(Path(config.output_dir)/'summary.json', summary)
            public = public_summary(config.output_dir)
            if review:
                from profiling_tools.call_tables import export_tables
                progress("call_review")
                try:
                    export_tables(config.output_dir, attempt / "call-review")
                except (ValueError, OSError):
                    traceback.print_exc()
                    public["review_export_error"] = "严格调用评审表未生成；主报告仍可下载，原因见 Worker 日志。"
                else:
                    public["reports"].append("call_tables.xlsx")
            if aggregate:
                from .aggregate import export_pair
                progress("aggregate")
                try:
                    export_pair(attempt, config.output_dir, aggregate)
                except (ValueError, KeyError, OSError, ImportError):
                    traceback.print_exc()
                    public["aggregate_export_error"] = "聚合表未生成，请核对全量任务、PID、明细口径和服务依赖；主报告仍可下载。"
                else:
                    public["reports"].extend(["op_type_comparison.xlsx","op_type_comparison.md"])
            write_json(attempt / "result.json", {"status": "succeeded", "summary": public})
    except InputSelectionError as exc:
        write_json(attempt / "result.json", {"status": "needs_selection", "error": "范围选择不完整，请检查 PID / step"})
    except Exception:
        traceback.print_exc()  # Private task log, never returned as an API traceback.
        write_json(attempt / "result.json", {"status": "failed", "error": "分析失败；请检查输入与参数，详细错误保存在服务器任务日志中"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))

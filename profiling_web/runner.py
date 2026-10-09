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
    config = CompareConfig(**task["config"])
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
                                       manual_links=state.get('links',[]), manual_base=base)
            if summary["validation"]["status"] != "valid":
                raise ValueError("Result validation failed")
            progress("indexing")
            build_index(config.output_dir, attempt / "index.sqlite")
            write_json(attempt / "result.json", {"status": "succeeded", "summary": public_summary(config.output_dir)})
    except InputSelectionError as exc:
        write_json(attempt / "result.json", {"status": "needs_selection", "error": "范围选择不完整，请检查 PID / step"})
    except Exception:
        traceback.print_exc()  # Private task log, never returned as an API traceback.
        write_json(attempt / "result.json", {"status": "failed", "error": "分析失败；请检查输入与参数，详细错误保存在服务器任务日志中"})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))

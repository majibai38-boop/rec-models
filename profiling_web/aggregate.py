"""Read an existing paired all-steps result; never accept a client filesystem path."""
from argparse import Namespace
from pathlib import Path
import runpy

from .store import Store


def export_pair(attempt, selected, full_id):
    store = Store(attempt.parents[3])
    other = store.get(full_id)
    if other["status"] != "succeeded" or other["mode"] != "gpu_npu" or not other["options"].get("all_steps"):
        raise ValueError("全量任务已不可用")
    full = store.path(full_id)/"attempts"/other["attempt"]/"reports"
    script = Path(__file__).resolve().parents[1]/"skills/gpu-npu-profiling/scripts/build_op_type_comparison.py"
    engine = runpy.run_path(str(script))
    args = Namespace(result_step10=selected, result_all=full, baseline_all=None,
                     npu_label="NPU", gpu_label="GPU", caliber_note="")
    tables = engine["build_report"](args)
    # Do not publish server paths. Keep Step, PID, run IDs and artifact hash gates.
    notes = tables[0]
    notes.rows = [(label, value.replace(str(Path(selected).resolve()), "当前 Step 结果")
                   .replace(str(full.resolve()), "全量任务 "+full_id) if label.endswith(" result")
                   else value.split("；",1)[-1] if label.endswith((" gpu"," npu")) else value)
                  for label,value in notes.rows]
    title = "GPU / NPU 类型与调用方聚合对照"
    output = attempt/"aggregate"
    output.mkdir()
    (output/"op_type_comparison.md").write_text(engine["markdown"](title,tables),encoding="utf-8")
    (output/"op_type_comparison.xlsx").write_bytes(engine["workbook"](title,tables))

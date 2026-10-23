"""Isolated worker. No execution of any content from uploaded files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from .profile import analyze, inventory
from .predict import predict
from .schema import Scope
from .store import Store


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024**2),b""):
            h.update(chunk)
    return h.hexdigest()


def engine_digest():
    root=Path(__file__).resolve().parent.parent
    files=sorted((root/"workload_modeler").glob("*.py"))+sorted((root/"profiling_tools").glob("*.py"))
    return hashlib.sha256("\n".join(str(p.relative_to(root))+":"+digest(p) for p in files).encode()).hexdigest()


def atomic_json(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding="utf-8")
    os.replace(tmp,path)


def safe_csv(value):
    if isinstance(value,str) and value.lstrip().startswith(("=","+","-","@")):
        return "'"+value
    return value


def execute(root,key,action,side="gpu"):
    store=Store(root)
    directory=store.job_dir(key)
    try:
        if action=="inspect":
            atomic_json(directory/"inventory.json",inventory(directory,side))
            store.transition(key,{"inspecting"},"ready")
            return
        config=store.job(key)["config"]
        if config["engine_sha256"] != engine_digest():
            raise ValueError("分析引擎在提交后发生变化，请创建新分析")
        result=predict(analyze(directory,Scope.model_validate(config["request"]["scope"])),config)
        if not all(result["audit"].values()):
            raise ValueError("记账审计失败，不发布预测")
        result["input_sha256"]={p.name:digest(p) for p in directory.iterdir() if p.name in {"trace.json","trace.json.gz","kernel_details.csv"}}
        atomic_json(directory/"result.json",result)
        fields=["op","shape","dtype","class_name","calls","kernel_count","source_us","modelled_source_us","predicted_us","coverage_pct","ratio","signature"]
        with (directory/"operators.csv").open("w",encoding="utf-8-sig",newline="") as f:
            writer=csv.DictWriter(f,fields,extrasaction="ignore")
            writer.writeheader()
            writer.writerows({k:safe_csv(v) for k,v in row.items()} for row in result["groups"])
        store.transition(key,{"running"},"completed")
    except Exception as exc:
        try:
            store.transition(key,{"running","inspecting"},"failed",error=str(exc)[:2000])
        except Exception:
            pass
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True)
    p.add_argument("--job",required=True)
    p.add_argument("--action",choices=["inspect","analyze"],required=True)
    p.add_argument("--side",choices=["gpu","npu"],default="gpu")
    a=p.parse_args()
    execute(a.root,a.job,a.action,a.side)

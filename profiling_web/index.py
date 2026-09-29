"""Build immutable per-attempt SQLite indexes; never parse traces on GET."""
import csv
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from profiling_tools.result_api import _statistic_record, read_json, summarize_results


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def public_summary(reports):
    summary = summarize_results(reports)
    summary.pop("result_dir", None)
    for side, scope in summary["scope"].items():
        scope["source"] = side + "_trace"
    # Do not offer host paths/manifests/configuration files through downloads.
    summary["reports"] = [n for n in summary["reports"] if downloadable(n)]
    return summary


def downloadable(name):
    return (Path(name).name == name and "/" not in name and "\\" not in name
            and name not in {"summary.json", "run_manifest.json", "input_inspection.json"}
            and Path(name).suffix in {".csv", ".xlsx", ".json", ".md"})


def build_index(reports, destination):
    reports = Path(reports)
    with closing(sqlite3.connect(destination)) as db:
        db.executescript("""
            CREATE TABLE stats (id INTEGER PRIMARY KEY, op_type TEXT, device_id TEXT, core_type TEXT,
                row_kind TEXT, npu_time REAL, npu_ratio REAL, coverage REAL, ratio REAL, payload TEXT);
            CREATE INDEX stats_type ON stats(op_type,device_id,core_type);
            CREATE TABLE records (id INTEGER PRIMARY KEY, kind TEXT, op_type TEXT, device_id TEXT,
                core_type TEXT, npu_index INTEGER, payload TEXT);
            CREATE INDEX records_kind_type ON records(kind,op_type,device_id,core_type);
            CREATE INDEX records_kernel ON records(kind,npu_index);
            CREATE TABLE call_types (record_id INTEGER, op_type TEXT, device_id TEXT, core_type TEXT);
            CREATE INDEX call_types_lookup ON call_types(op_type,device_id,core_type,record_id);
        """)
        for row in csv_rows(reports / "npu_gpu_op_statistic.csv"):
            record = _statistic_record(row)
            record.update({"fused_npu_type": row.get("FUSED_OP_NPU_TYPE", ""), "fused_gpu_type": row.get("FUSED_OP_GPU_TYPE", ""),
                           "conclusion": row.get("排查结论", ""), "advice": row.get("排查建议", ""),
                           "whole_status": row.get("Whole Comparison Status", ""),
                           "related_whole_ops": row.get("Related Whole OPs", ""),
                           "whole_group_ids": json.loads(row.get("Whole Group IDs") or "[]"),
                           "cpu_alignment_status": row.get("CPU Alignment Status", ""),
                           "kernel_comparison_status": row.get("Kernel Comparison Status", "")})
            db.execute("INSERT INTO stats VALUES(NULL,?,?,?,?,?,?,?,?,?)", (
                record["op_type"], record["device_id"], record["core_type"], record["row_kind"],
                record["npu_total_time_us"], record["npu_ratio_pct"], record["coverage"]["comparable"]["time_pct"],
                record["comparable_ratio_gpu_npu"], json.dumps(record, ensure_ascii=False)))

        def insert(kind, row, op_type="", device="", core="", npu=None):
            return db.execute("INSERT INTO records VALUES(NULL,?,?,?,?,?,?)", (kind, op_type, device, core, npu, json.dumps(row, ensure_ascii=False))).lastrowid

        types_by_kernel = {}
        for row in csv_rows(reports / "npu_kernel_gpu_mapping.csv"):
            insert("mappings", row, row["OP Type"], row["Device_id"], row["Core Type"], int(row["NPU Kernel Index"]))
            types_by_kernel[int(row["NPU Kernel Index"])] = (row["OP Type"], row["Device_id"], row["Core Type"])
        diag = read_json(reports / "coverage_diagnostics.json")
        for row in diag["kernels"]:
            insert("diagnoses", row, row.get("op_type", ""), row.get("device_id", ""), row.get("core_type", ""), row["npu_kernel_index"])
        for group in diag["types"]:
            for row in group["diagnoses"]:
                insert("reasons", {"op_type": group["op_type"], "device_id": group["device_id"], "core_type": group["core_type"], **row},
                       group["op_type"], group["device_id"], group["core_type"])
        for source in ("cpu_bundle_mapping.csv", "fused_op_mapping.csv"):
            for row in csv_rows(reports / source):
                insert("calls", {"source": source, **row}, row.get("Fused OP", row.get("CPU Bundle", "")))
        group_types = {}
        for filename, kind in (("whole_call_mapping.csv", "call_details"), ("whole_call_statistic.csv", "whole_calls")):
            if not (reports / filename).exists():
                continue
            for row in csv_rows(reports / filename):
                record_id = insert(kind, row, row["Whole OP"], row.get("Device_id", ""))
                if kind == "call_details":
                    keys = {types_by_kernel[i] for i in json.loads(row["NPU Kernel Indices"]) if i in types_by_kernel}
                    group_types.setdefault(row["Group ID"], set()).update(keys)
                else:
                    keys = group_types.get(row["Group ID"], set())
                db.executemany("INSERT INTO call_types VALUES(?,?,?,?)", [(record_id, *key) for key in keys])
        for row in csv_rows(reports / "gpu_unmapped_activity.csv"):
            insert("gpu_unmapped", row)
        db.commit()


def query_index(path, *, kind="operators", op_type=None, device_id=None, core_type=None,
                npu_index=None, search="", coverage_max=None, npu_ratio_min=None,
                sort="npu_time", offset=0, limit=30, group_id=None, call_id=None, related_type=None):
    if not 1 <= limit <= 200 or offset < 0:
        raise ValueError("Invalid pagination")
    if kind not in {"operators", "mappings", "diagnoses", "reasons", "calls", "gpu_unmapped", "whole_calls", "call_details"}:
        raise ValueError("Unknown record kind")
    conditions, params = [], []
    table = "stats" if kind == "operators" else "records"
    if table == "records":
        conditions.append("kind=?")
        params.append(kind)
    else:
        conditions.append("row_kind!='total'")
    for column, value in (("op_type", op_type), ("device_id", None if related_type else device_id), ("core_type", None if related_type else core_type)):
        if value is not None:
            conditions.append(column + "=?")
            params.append(value)
    if group_id is not None or call_id is not None or related_type is not None:
        if kind not in {"whole_calls", "call_details"}:
            raise ValueError("Call filters require whole_calls or call_details")
        for field, value in (("Group ID", group_id), ("Call ID", call_id)):
            if value is not None:
                conditions.append("json_extract(payload, ?) = ?")
                params.extend(['$."' + field + '"', value])
        if related_type is not None:
            clauses = ["record_id=records.id", "call_types.op_type=?"]
            params.append(related_type)
            for field, value in (("device_id", device_id), ("core_type", core_type)):
                if value is not None:
                    clauses.append("call_types." + field + "=?")
                    params.append(value)
            conditions.append("EXISTS (SELECT 1 FROM call_types WHERE " + " AND ".join(clauses) + ")")
    if npu_index is not None:
        if table != "records":
            raise ValueError("Kernel filter only applies to records")
        conditions.append("npu_index=?")
        params.append(npu_index)
    if search:
        conditions.append("instr(lower(op_type),lower(?))>0")
        params.append(search)
    if table == "stats":
        for column, operator, value in (("coverage", "<", coverage_max), ("npu_ratio", ">", npu_ratio_min)):
            if value is not None:
                conditions.append(column + operator + "?")
                params.append(value)
        order = {"npu_time": "npu_time DESC", "coverage": "coverage ASC", "ratio": "ratio DESC", "name": "op_type ASC"}.get(sort)
        if not order:
            raise ValueError("Unknown sorting field")
    else:
        order = "id ASC"
    where = " WHERE " + " AND ".join(conditions)
    # URI is built from a server-owned path, never a request path.
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        if related_type and not db.execute("SELECT 1 FROM sqlite_master WHERE name='call_types'").fetchone():
            return {"total": 0, "offset": offset, "limit": limit, "has_more": False, "items": []}
        count = db.execute("SELECT count(*) FROM " + table + where, params).fetchone()[0]
        rows = db.execute("SELECT payload FROM " + table + where + " ORDER BY " + order + ",id ASC LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
    return {"total": count, "offset": offset, "limit": limit, "has_more": offset + limit < count,
            "items": [json.loads(r[0]) for r in rows]}

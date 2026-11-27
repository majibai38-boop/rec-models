"""Paged reads of the NPU-only report tables without loading the full JSON."""
import json
import sqlite3
from contextlib import closing


def pipeline_rows(report):
    """One row per OP Type + Core Type, carrying both sides' pipeline mix for a compact bar."""
    merged = {}
    for side, key in (("基线", "baseline_types"), ("对比", "candidate_types")):
        for row in report.get("pipeline", {}).get(key, []):
            name = (row["OP Type"], row["Core Type"])
            entry = merged.setdefault(name, {"OP Type": row["OP Type"], "Core Type": row["Core Type"],
                                             "基线": {"categories": [], "count": 0, "time_us": 0.0},
                                             "对比": {"categories": [], "count": 0, "time_us": 0.0}})
            entry[side]["categories"].append({"category": row["流水类别"], "label": row["标签"],
                                              "count": row["次数"], "time_us": row["耗时(us)"],
                                              "time_pct": row["该类型内占比(%)"]})
            entry[side]["count"] += row["次数"]
            entry[side]["time_us"] += row["耗时(us)"]
    return [merged[key] for key in sorted(merged)]


def build(path, report):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE items(view TEXT, seq INTEGER, op TEXT, core TEXT, data TEXT, PRIMARY KEY(view,seq))")
        for view in ("types", "signatures"):
            db.executemany("INSERT INTO items VALUES(?,?,?,?,?)", (
                (view, i, r["OP Type"], r["Core Type"], json.dumps(r, ensure_ascii=False))
                for i, r in enumerate(report[view])))
        db.executemany("INSERT INTO items VALUES(?,?,?,?,?)", (
            ("pipeline", i, r["OP Type"], r["Core Type"], json.dumps(r, ensure_ascii=False))
            for i, r in enumerate(pipeline_rows(report))))
        db.execute("CREATE INDEX scope ON items(view,op,core,seq)")


SEARCH_FIELDS = {
    "types": ["op", "core"],
    "pipeline": ["op", "core"],
    "signatures": [
        """json_extract(data,'$."名称"')""",
        """json_extract(data,'$."输入 Shape"')""", """json_extract(data,'$."输出 Shape"')""",
        """json_extract(data,'$."输入 dtype"')""", """json_extract(data,'$."输出 dtype"')"""],
}
ORDER = {
    "difference": "seq",
    "baseline": """json_extract(data,'$."基线耗时(us)"') DESC, seq""",
    "candidate": """json_extract(data,'$."对比耗时(us)"') DESC, seq""",
    "ratio": """json_extract(data,'$."性能比(基线/对比)"') DESC, seq""",
}
PIPELINE_ORDER = {
    "baseline": """json_extract(data,'$."基线"."time_us"') DESC, seq""",
    "candidate": """json_extract(data,'$."对比"."time_us"') DESC, seq""",
}


def page(path, view, op, core, offset, limit, search="", sort="difference"):
    where, args = "view=?", [view]
    if view == "signatures":
        where += " AND op=? AND core=?"
        args.extend([op, core])
    if search:
        fields = SEARCH_FIELDS[view]
        where += " AND ("+" OR ".join("instr(lower("+f+"),lower(?))>0" for f in fields)+")"
        args.extend([search]*len(fields))
    if view == "pipeline":
        order = PIPELINE_ORDER["baseline" if sort == "baseline" else "candidate"]
    else:
        order = ORDER[sort]
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)) as db:
        total, = db.execute("SELECT COUNT(*) FROM items WHERE "+where, args).fetchone()
        rows = db.execute("SELECT data FROM items WHERE "+where+" ORDER BY "+order+" LIMIT ? OFFSET ?", [*args, limit, offset])
        return {"total": total, "items": [json.loads(r[0]) for r in rows]}

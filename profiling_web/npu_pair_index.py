"""Paged reads of the two NPU-only report tables without loading the full JSON."""
import json
import sqlite3
from contextlib import closing


def build(path, report):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE items(view TEXT, seq INTEGER, op TEXT, core TEXT, data TEXT, PRIMARY KEY(view,seq))")
        for view in ("types", "signatures"):
            db.executemany("INSERT INTO items VALUES(?,?,?,?,?)", (
                (view, i, r["OP Type"], r["Core Type"], json.dumps(r, ensure_ascii=False))
                for i, r in enumerate(report[view])))
        db.execute("CREATE INDEX scope ON items(view,op,core,seq)")


def page(path, view, op, core, offset, limit, search="", sort="difference"):
    where, args = "view=?", [view]
    if view == "signatures":
        where += " AND op=? AND core=?"
        args.extend([op, core])
    if search:
        fields = ["op","core"] if view == "types" else [
            """json_extract(data,'$."名称"')""",
            """json_extract(data,'$."输入 Shape"')""", """json_extract(data,'$."输出 Shape"')""",
            """json_extract(data,'$."输入 dtype"')""", """json_extract(data,'$."输出 dtype"')"""]
        where += " AND ("+" OR ".join("instr(lower("+f+"),lower(?))>0" for f in fields)+")"
        args.extend([search]*len(fields))
    order = {"difference":"seq", "baseline":"""json_extract(data,'$."基线耗时(us)"') DESC, seq""",
             "candidate":"""json_extract(data,'$."对比耗时(us)"') DESC, seq""",
             "ratio":"""json_extract(data,'$."性能比(基线/对比)"') DESC, seq"""}[sort]
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)) as db:
        total, = db.execute("SELECT COUNT(*) FROM items WHERE "+where, args).fetchone()
        rows = db.execute("SELECT data FROM items WHERE "+where+" ORDER BY "+order+" LIMIT ? OFFSET ?", [*args, limit, offset])
        return {"total": total, "items": [json.loads(r[0]) for r in rows]}

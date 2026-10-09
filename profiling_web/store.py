import json
from contextlib import contextmanager
from pathlib import Path
import re
import sqlite3
import time
import uuid


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.db = self.root / "tasks.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("BEGIN IMMEDIATE")
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, name TEXT, status TEXT, stage TEXT,
                created REAL, updated REAL, started REAL, kind TEXT,
                attempt TEXT, options TEXT DEFAULT '{}', inputs TEXT DEFAULT '{}',
                inspection TEXT, summary TEXT, error TEXT DEFAULT '',
                cancel INTEGER DEFAULT 0, uploading INTEGER DEFAULT 0)""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "archived" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
            if "manual" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN manual TEXT NOT NULL DEFAULT '{}'")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def path(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise KeyError(job_id)
        return self.root / "jobs" / job_id

    def get(self, job_id):
        self.path(job_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        item = dict(row)
        for key in ("inputs", "options", "inspection", "summary", "manual"):
            item[key] = json.loads(item[key]) if item[key] else None
        return item

    def create(self, name):
        job_id, now = uuid.uuid4().hex, time.time()
        (self.path(job_id) / "inputs").mkdir(parents=True)
        with self.connect() as db:
            db.execute("INSERT INTO jobs(id,name,status,stage,created,updated) VALUES(?,?,'uploading','uploading',?,?)", (job_id, name, now, now))
        return self.get(job_id)

    def list(self, offset=0, limit=50, search="", status=None, archived=False):
        clauses, params = ["archived=?", "instr(lower(name),lower(?))>0"], [int(archived), search]
        if status:
            clauses.append("status=?")
            params.append(status)
        where = " WHERE " + " AND ".join(clauses)
        with self.connect() as db:
            total = db.execute("SELECT COUNT(*) FROM jobs" + where, params).fetchone()[0]
            rows = db.execute("SELECT id,name,status,stage,created,updated,error,archived FROM jobs" + where + " ORDER BY created DESC,id LIMIT ? OFFSET ?", (*params, limit, offset)).fetchall()
        return {"total": total, "items": [dict(r) for r in rows], "offset": offset, "limit": limit}

    def edit(self, job_id, *, name=None, archived=None):
        self.get(job_id)
        fields = {"updated": time.time()}
        if name is not None:
            name = name.strip()
            if not name or len(name) > 120:
                raise ValueError("任务名称须为 1–120 个字符")
            fields["name"] = name
        if archived is not None:
            fields["archived"] = int(archived)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status,uploading FROM jobs WHERE id=?", (job_id,)).fetchone()
            if archived and (row["status"] in {"queued", "running"} or row["uploading"]):
                raise Conflict("活动任务不能归档")
            db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?", (*fields.values(), job_id))
        return self.get(job_id)

    def update(self, job_id, **fields):
        if fields.get('status') in {'failed','cancelled','interrupted'}:
            if self.restore_manual(job_id, fields.get('error','人工关联未完成')):
                return
        if fields.get('status') == 'succeeded':
            current = self.get(job_id)
            if current['manual'].get('pending'):
                fields['manual'] = {k:v for k,v in current['manual'].items() if k != 'pending'}
        allowed = {"status", "stage", "kind", "attempt", "options", "inputs", "inspection", "summary", "error", "cancel", "uploading", "started", "manual"}
        if not fields.keys() <= allowed:
            raise ValueError("Unknown task field")
        fields["updated"] = time.time()
        fields = {k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list)) else v for k, v in fields.items()}
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in fields) + " WHERE id=?", (*fields.values(), job_id))

    def start_upload(self, job_id):
        self.get(job_id)
        with self.connect() as db:
            count = db.execute("UPDATE jobs SET uploading=1 WHERE id=? AND status='uploading' AND uploading=0", (job_id,)).rowcount
        if not count:
            raise Conflict("任务输入已经冻结，或另一个文件正在上传")

    def enqueue(self, job_id, kind, options=None):
        self.get(job_id)
        allowed = ("uploading", "needs_selection", "failed", "cancelled", "interrupted")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row["status"] not in allowed or row["uploading"]:
                raise Conflict("当前任务状态不允许启动；成功结果需新建任务，不能覆盖")
            if set(json.loads(row["inputs"])) != {"gpu_trace", "npu_trace", "npu_kernel_details"}:
                raise Conflict("请先上传三个必需文件")
            if kind == "compare" and not row["inspection"]:
                raise Conflict("请先完成预检查")
            db.execute("UPDATE jobs SET status='queued',stage='queued',kind=?,options=?,cancel=0,error='',summary=NULL,attempt=NULL,started=NULL,updated=? WHERE id=?",
                       (kind, json.dumps(options or {}), time.time(), job_id))
        return self.get(job_id)

    def enqueue_manual(self, job_id, expected_attempt, links):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM jobs WHERE id=?',(job_id,)).fetchone()
            if not row or row['status'] != 'succeeded' or row['attempt'] != expected_attempt:
                raise Conflict('结果已变化或正在分析，请刷新后重试')
            old = json.loads(row['manual'])
            state = {'base_attempt': old.get('base_attempt',row['attempt']), 'links':links,
                     'pending': {'attempt':row['attempt'], 'summary':json.loads(row['summary']), 'manual':old}}
            db.execute("UPDATE jobs SET status='queued',stage='queued',kind='manual',manual=?,cancel=0,error='',updated=? WHERE id=?",
                       (json.dumps(state,ensure_ascii=False),time.time(),job_id))
        return self.get(job_id)

    def restore_manual(self, job_id, error, expected_status=None):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT manual,status FROM jobs WHERE id=?',(job_id,)).fetchone()
            if expected_status is not None and (not row or row[1] != expected_status):
                return False
            pending = json.loads(row[0]).get('pending') if row else None
            if not pending:
                return False
            db.execute("UPDATE jobs SET status='succeeded',stage='succeeded',kind='compare',attempt=?,summary=?,manual=?,error=?,cancel=0,updated=? WHERE id=?",
                       (pending['attempt'],json.dumps(pending['summary']),json.dumps(pending['manual']),
                        '人工关联未生效，已保留上一份结果：'+error,time.time(),job_id))
        return True

    def claim(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY updated LIMIT 1").fetchone()
            if row is None:
                return None
            now = time.time()
            db.execute("UPDATE jobs SET status='running',stage='starting',attempt=?,started=?,updated=? WHERE id=?", (uuid.uuid4().hex, now, now, row[0]))
        return self.get(row[0])

    def cancel(self, job_id):
        self.get(job_id)
        current = self.get(job_id)
        if current['status'] == 'queued' and self.restore_manual(job_id, '用户取消', expected_status='queued'):
            return self.get(job_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row[0] == "queued":
                db.execute("UPDATE jobs SET status='cancelled',stage='cancelled',updated=? WHERE id=?", (time.time(), job_id))
            elif row[0] == "running":
                db.execute("UPDATE jobs SET cancel=1,updated=? WHERE id=?", (time.time(), job_id))
            else:
                raise Conflict("只有排队或运行中的任务可以取消")
        return self.get(job_id)

    def recover(self):
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT id FROM jobs WHERE status='running' AND kind='manual'")]
        for ident in ids:
            self.restore_manual(ident,'Worker 已重启')
        with self.connect() as db:
            db.execute("UPDATE jobs SET status='interrupted',stage='interrupted',error='Worker 已重启，请手动重试',updated=? WHERE status='running'", (time.time(),))

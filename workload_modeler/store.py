"""Immutable revisions and SQLite-backed job state for a single local server."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .schema import Hardware, Platform


def now():
    return datetime.now(timezone.utc).isoformat()


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.db = self.root / "modeler.sqlite"
        with self.connect() as c:
            c.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS revisions (
                  kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
                  created TEXT NOT NULL, payload TEXT NOT NULL,
                  PRIMARY KEY(kind,id,revision));
                CREATE TABLE IF NOT EXISTS jobs (
                  id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                  created TEXT NOT NULL, error TEXT, config TEXT);
                CREATE TABLE IF NOT EXISTS catalog_trash (
                  kind TEXT NOT NULL, id TEXT NOT NULL, deleted_at TEXT NOT NULL,
                  PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS builtin_seeds (key TEXT PRIMARY KEY);
            """)
            columns={r['name'] for r in c.execute('PRAGMA table_info(jobs)')}
            if 'deleted_at' not in columns:
                c.execute('ALTER TABLE jobs ADD COLUMN deleted_at TEXT')

    @contextmanager
    def connect(self):
        c = sqlite3.connect(self.db, timeout=20)
        c.row_factory = sqlite3.Row
        try:
            with c:
                yield c
        finally:
            c.close()

    def get(self, kind, key, revision=None):
        with self.connect() as c:
            row = c.execute("SELECT * FROM revisions WHERE kind=? AND id=? " +
                            ("AND revision=?" if revision else "ORDER BY revision DESC LIMIT 1"),
                            (kind, key, revision) if revision else (kind, key)).fetchone()
        if row is None:
            raise KeyError("规格/平台版本不存在")
        return dict(json.loads(row["payload"]), id=row["id"], revision=row["revision"], created=row["created"])

    def list(self, kind, deleted=False):
        with self.connect() as c:
            rows = c.execute("SELECT r.*,t.deleted_at FROM revisions r JOIN (SELECT id,MAX(revision) rev FROM revisions "
                             "WHERE kind=? GROUP BY id) x ON r.id=x.id AND r.revision=x.rev "
                             "LEFT JOIN catalog_trash t ON t.kind=r.kind AND t.id=r.id WHERE r.kind=? "+
                             ("AND t.deleted_at IS NOT NULL " if deleted else "AND t.deleted_at IS NULL ")+
                             "ORDER BY r.created DESC,r.id", (kind, kind)).fetchall()
        return [dict(json.loads(r["payload"]), id=r["id"], revision=r["revision"], created=r["created"],deleted_at=r['deleted_at']) for r in rows]

    def active(self, kind, key, revision=None):
        value=self.get(kind,key,revision)
        with self.connect() as c:
            self._check_active(c,kind,key)
        return value

    @staticmethod
    def _check_active(c,kind,key):
        if c.execute('SELECT 1 FROM catalog_trash WHERE kind=? AND id=?',(kind,key)).fetchone():
            raise Conflict('该规格/平台已在回收站，请先恢复或选择其他条目')

    def trash_catalog(self,kind,key,expected,deleted=True):
        if kind not in {'hardware','platform'}:
            raise ValueError('未知资源类型')
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT revision,payload FROM revisions WHERE kind=? AND id=? ORDER BY revision DESC LIMIT 1',(kind,key)).fetchone()
            if row is None:
                raise KeyError('资源不存在')
            if row['revision']!=expected:
                raise Conflict('版本已变化，请刷新后重试')
            if deleted and kind=='hardware':
                platforms=c.execute("SELECT r.payload FROM revisions r WHERE r.kind='platform' AND r.revision=(SELECT MAX(revision) FROM revisions WHERE kind=r.kind AND id=r.id) AND NOT EXISTS(SELECT 1 FROM catalog_trash t WHERE t.kind=r.kind AND t.id=r.id)")
                names=[p['name'] for p in map(lambda r:json.loads(r['payload']),platforms) if p['hardware_id']==key]
                if names:
                    raise Conflict('芯片仍被有效平台引用，请先修改或删除这些平台：'+'、'.join(names[:5]))
            if not deleted and kind=='platform':
                self._check_active(c,'hardware',json.loads(row['payload'])['hardware_id'])
            if deleted:
                c.execute('INSERT OR IGNORE INTO catalog_trash VALUES(?,?,?)',(kind,key,now()))
            else:
                c.execute('DELETE FROM catalog_trash WHERE kind=? AND id=?',(kind,key))
        return dict(id=key,deleted=deleted)

    def history(self, kind, key):
        with self.connect() as c:
            rows = c.execute("SELECT revision FROM revisions WHERE kind=? AND id=? ORDER BY revision DESC",
                             (kind, key)).fetchall()
        return [self.get(kind, key, r[0]) for r in rows]

    def save(self, kind, payload, key=None, expected=None):
        model = {"hardware": Hardware, "platform": Platform}.get(kind)
        if model is None:
            raise ValueError("未知资源类型")
        value = model.model_validate(payload).model_dump()
        if kind == "platform":
            self.get("hardware", value["hardware_id"], value["hardware_revision"])
        new = key is None
        key = key or uuid.uuid4().hex
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            self._check_active(c,kind,key)
            if kind=='platform':
                self._check_active(c,'hardware',value['hardware_id'])
            revision = c.execute("SELECT MAX(revision) FROM revisions WHERE kind=? AND id=?", (kind, key)).fetchone()[0]
            if not new and revision is None:
                raise KeyError("资源不存在")
            if not new and expected != revision:
                raise Conflict("版本已变化，请刷新后重试")
            revision = (revision or 0) + 1
            c.execute("INSERT INTO revisions VALUES(?,?,?,?,?)", (kind, key, revision, now(), json.dumps(value, ensure_ascii=False)))
        return self.get(kind, key, revision)

    def create_job(self, name):
        key = uuid.uuid4().hex
        (self.root / "jobs" / key).mkdir()
        with self.connect() as c:
            c.execute("INSERT INTO jobs(id,name,status,created,error,config) VALUES(?,?,?,?,?,?)", (key, name, "draft", now(), None, None))
        return self.job(key)

    def job_dir(self, key):
        if len(key) != 32 or any(ch not in "0123456789abcdef" for ch in key):
            raise KeyError("任务不存在")
        self.job(key)
        return self.root / "jobs" / key

    def job(self, key, include_deleted=False):
        with self.connect() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (key,)).fetchone()
        if row is None or (row['deleted_at'] and not include_deleted):
            raise KeyError("任务不存在")
        data = dict(row)
        data["config"] = json.loads(data["config"]) if data["config"] else None
        return data

    def jobs(self, deleted=False, search='', offset=0, limit=200):
        with self.connect() as c:
            return [dict(r) for r in c.execute("SELECT id,name,status,created,error,deleted_at FROM jobs WHERE deleted_at IS "+
                ("NOT NULL" if deleted else "NULL")+" AND instr(lower(name),lower(?))>0 ORDER BY created DESC,id LIMIT ? OFFSET ?",(search,limit,offset))]

    def trash_job(self,key,deleted=True):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            row=c.execute('SELECT status FROM jobs WHERE id=?',(key,)).fetchone()
            if row is None:
                raise KeyError('任务不存在')
            if row['status'] in {'running','inspecting','uploading'}:
                raise Conflict('上传/运行中的任务不能删除，请等待或取消后再试')
            c.execute('UPDATE jobs SET deleted_at=? WHERE id=?',(now() if deleted else None,key))
        return self.job(key,include_deleted=True)

    def rename_job(self,key,name):
        with self.connect() as c:
            if c.execute('UPDATE jobs SET name=? WHERE id=? AND deleted_at IS NULL',(name,key)).rowcount!=1:
                raise KeyError('任务不存在或已删除')
        return self.job(key)

    def transition(self, key, allowed, status, *, config=None, error=None):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT status,deleted_at FROM jobs WHERE id=?", (key,)).fetchone()
            if row is None or row['deleted_at']:
                raise KeyError("任务不存在")
            if row[0] not in allowed:
                raise Conflict(f"任务当前处于 {row[0]} 状态")
            c.execute("UPDATE jobs SET status=?,error=?,config=COALESCE(?,config) WHERE id=?",
                      (status, error, json.dumps(config, ensure_ascii=False) if config is not None else None, key))


def seed_initial(store):
    """Conservative, sourced seeds; unsupported peak modes stay unknown."""
    ascend = "https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf"
    def metric(v, unit, src, evidence="official", dense=None, note=""):
        return dict(value=v, unit=unit, source=src, evidence=evidence, dense=dense, note=note)
    for cores, vector, fp16, tf32, vec, bw in [(28,56,378,189,23,1400),(32,64,432,216,27,1600)]:
        hw = store.save("hardware", dict(name=f"950PR 标卡 {cores} Cube", vendor="Ascend", metrics={
            "matrix_fp16": metric(fp16,"TFLOPS",ascend,dense=True),
            "matrix_bf16": metric(fp16,"TFLOPS",ascend,dense=True),
            "matrix_hf32": metric(tf32,"TFLOPS",ascend,dense=True,note="TF32/HF32候选档位，跨精度模式须用户确认"),
            "vector_fp32": metric(vec,"TFLOPS",ascend),
            "vector_fp16": metric(47 if cores==28 else 54,"TFLOPS",ascend),
            "vector_bf16": metric(47 if cores==28 else 54,"TFLOPS",ascend),
            "memory_bandwidth": metric(bw,"GB/s",ascend),
            "cube_count": metric(cores,"count","用户确认", "user"),
            "vector_count": metric(vector,"count",ascend)}, note="28核Vector FP32使用白皮书整数23；不混用原表23.5。峰值不等于有效吞吐。"))
        store.save("platform", dict(name=f"950PR {cores} 单卡建模平台",hardware_id=hw["id"],hardware_revision=1,
            cpu_model="AMD EPYC 9575F",cpu_count=2,cores_per_cpu=64,dimm_count=24,dimm_gb=64,memory_rated_mts=5600,
            note="主机信息按用户工作簿；单卡为建模范围，并非推断整机仅一张卡。"))
    src="https://www.ctyun.cn/document/10029787/10348867"
    hw=store.save("hardware",dict(name="NVIDIA L20",vendor="NVIDIA",metrics={
        "matrix_fp16":metric(119.5,"TFLOPS",src,dense=None,note="公开峰值，稠密/稀疏口径待确认；默认不用于矩阵预测"),
        "matrix_tf32":metric(59.8,"TFLOPS",src,dense=None,note="稠密/稀疏待确认"),
        "vector_fp32":metric(59.8,"TFLOPS",src),
        "vector_fp16":metric(59.8,"TFLOPS","NVIDIA Ada白皮书非Tensor比例推算","estimated"),
        "vector_bf16":metric(59.8,"TFLOPS","NVIDIA Ada白皮书非Tensor比例推算","estimated"),
        "memory_bandwidth":metric(864,"GB/s",src),"memory_capacity":metric(48,"GB",src)},
        note="L2及通信带宽不内置无依据占位值。预测可在新版本中补齐经核验的规格。"))
    store.save("platform",dict(name="L20 单卡建模平台",hardware_id=hw["id"],hardware_revision=1,
        cpu_model="Intel Xeon Platinum 8458P",cores_per_cpu=44,cpu_mhz=2700,dimm_count=30,dimm_gb=64,memory_rated_mts=4800,
        note="用户提供配置。CPU数量、实际内存速率与拓扑未确认，不能把2S支持当作双路实装。"))


def seed(store):
    # Check all revisions, not visible entries: a recycled catalog must not reappear.
    with store.connect() as c:
        exists=c.execute("SELECT 1 FROM revisions WHERE kind='hardware' LIMIT 1").fetchone()
    if not exists:
        seed_initial(store)
    from .presets import install_presets
    install_presets(store)

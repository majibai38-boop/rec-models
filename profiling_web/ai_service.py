"""Persisted, single-concurrency AI work queue, separate from profiling jobs."""
import json
import logging
import queue
import threading
import time
import uuid

from fastapi import HTTPException
from pydantic import Field, model_validator
from typing import Literal

from .ai_backends import AIConfig, BackendError, run_backend
from .ai_evidence import StrictModel, build_evidence, validate_analysis
from .store import Conflict


class Scope(StrictModel):
    op_type: str = Field(min_length=1, max_length=200)
    device_id: str = Field(max_length=200)
    core_type: str = Field(max_length=200)


class Request(StrictModel):
    attempt: str = Field(pattern=r'^[0-9a-f]{32}$')
    provider: Literal['deepseek', 'codex']
    question: str = Field(min_length=1, max_length=2000)
    scope: Scope | None = None
    share_evidence: bool = False

    @model_validator(mode='after')
    def check(self):
        if not self.question.strip():
            raise ValueError('请填写分析问题')
        if not self.share_evidence:
            raise ValueError('请确认将所选范围的统计、元数据和问题发送给模型服务')
        return self


class AIService:
    def __init__(self, store, config=None):
        self.store, self.config = store, config or AIConfig.from_env()
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.queue, self.cancels = queue.Queue(), {}
        self.thread = None
        with store.connect() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS ai_runs (
                id TEXT PRIMARY KEY, job_id TEXT, attempt TEXT, provider TEXT, model TEXT,
                status TEXT, created REAL, updated REAL, request TEXT, evidence TEXT,
                result TEXT, error TEXT DEFAULT '')''')
            db.execute('CREATE INDEX IF NOT EXISTS ai_runs_job ON ai_runs(job_id,created)')

    def start(self):
        # A previous API process cannot still own this queue (api.lock held).
        with self.store.connect() as db:
            db.execute("UPDATE ai_runs SET status='interrupted',error='服务重启，AI 分析已中断，请重新发起',updated=? WHERE status IN ('queued','running')", (time.time(),))
            db.execute('DELETE FROM ai_runs WHERE job_id NOT IN (SELECT id FROM jobs)')
        self.thread = threading.Thread(target=self._loop, name='profiling-ai', daemon=True)
        self.thread.start()

    def close(self):
        self.stop.set()
        with self.lock:
            for event in self.cancels.values():
                event.set()
        if self.thread:
            self.thread.join(timeout=15)
        with self.store.connect() as db:
            db.execute("UPDATE ai_runs SET status='interrupted',error='服务停止，AI 分析已中断',updated=? WHERE status IN ('queued','running')", (time.time(),))

    def current(self, job_id, attempt):
        job = self.store.get(job_id)
        if job['status'] != 'succeeded' or not job['summary'] or job['attempt'] != attempt:
            raise Conflict('分析结果已变化或尚未完成，请刷新任务后重试')
        return job

    def create(self, job_id, body):
        provider = next(p for p in self.config.availability()['providers'] if p['id'] == body.provider)
        if not provider['available']:
            raise HTTPException(503, provider['reason'])
        with self.lock, self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self.current(job_id, body.attempt)
            if db.execute("SELECT count(*) FROM ai_runs WHERE status IN ('queued','running')").fetchone()[0] >= 8:
                raise HTTPException(429, 'AI 队列已满（最多 8 个），请等待或取消任务')
            if db.execute("SELECT 1 FROM ai_runs WHERE job_id=? AND status IN ('queued','running')", (job_id,)).fetchone():
                raise Conflict('该任务已有 AI 分析正在进行')
            if db.execute('SELECT count(*) FROM ai_runs WHERE job_id=?', (job_id,)).fetchone()[0] >= 50:
                raise HTTPException(429, '每个任务最多保存 50 次 AI 分析，请先删除不需要的 AI 历史')
            ident, now = uuid.uuid4().hex, time.time()
            db.execute('INSERT INTO ai_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (ident, job_id, body.attempt, body.provider, provider['model'], 'queued', now, now,
                        body.model_dump_json(), None, None, ''))
        # Commit before dispatch; cancellation cannot be lost between the two.
        with self.lock:
            event = threading.Event()
            # A concurrent cancel may commit between INSERT and dispatch.
            with self.store.connect() as db:
                row = db.execute('SELECT status FROM ai_runs WHERE id=?', (ident,)).fetchone()
            if not row or row['status'] != 'queued':
                event.set()
            self.cancels[ident] = event
            self.queue.put(ident)
        return self.get(job_id, ident)

    def list(self, job_id):
        self.store.get(job_id)
        with self.store.connect() as db:
            return [dict(r) for r in db.execute('SELECT id,attempt,provider,model,status,created,updated,error FROM ai_runs WHERE job_id=? ORDER BY created DESC', (job_id,))]

    def get(self, job_id, ident):
        job = self.store.get(job_id)
        with self.store.connect() as db:
            row = db.execute('SELECT * FROM ai_runs WHERE id=? AND job_id=?', (ident, job_id)).fetchone()
        if not row:
            raise HTTPException(404, 'AI 分析记录不存在')
        result = dict(row)
        for field in ('request', 'evidence', 'result'):
            result[field] = json.loads(result[field]) if result[field] else None
        result['stale'] = result['attempt'] != job['attempt'] or job['status'] != 'succeeded'
        return result

    def cancel(self, job_id, ident):
        self.get(job_id, ident)
        with self.lock:
            if ident in self.cancels:
                self.cancels[ident].set()
            with self.store.connect() as db:
                db.execute("UPDATE ai_runs SET status='cancelled',error='用户取消',updated=? WHERE id=? AND status IN ('queued','running')", (time.time(), ident))
        return self.get(job_id, ident)

    def delete(self, job_id, ident):
        run = self.get(job_id, ident)
        if run['status'] in {'queued', 'running'} or ident in self.cancels:
            raise Conflict('请先取消 AI 分析并等待进程退出')
        with self.store.connect() as db:
            db.execute('DELETE FROM ai_runs WHERE id=? AND job_id=?', (ident, job_id))
        return {'deleted': ident}

    def provenance(self, job_id, body):
        if not body.ai_run_id and not body.ai_proposal_id:
            return {}
        if not body.ai_run_id or not body.ai_proposal_id:
            raise HTTPException(422, 'AI 来源需要同时提供分析和提案编号')
        run = self.get(job_id, body.ai_run_id)
        if run['stale'] or run['attempt'] != body.attempt or run['status'] != 'succeeded':
            raise Conflict('AI 提案对应的结果已变化，请重新分析')
        proposal = next((p for p in run['result']['proposals'] if p['id'] == body.ai_proposal_id), None)
        if not proposal or (proposal['target_id'], proposal['candidate_id']) != (body.target_id, body.candidate_id):
            raise HTTPException(422, '人工确认的配对与 AI 提案不一致')
        return {'ai_run_id': run['id'], 'ai_proposal_id': proposal['id'], 'ai_provider': run['provider']}

    def _update(self, ident, **fields):
        fields['updated'] = time.time()
        with self.store.connect() as db:
            # Cancellation wins even if inference completes concurrently.
            db.execute('UPDATE ai_runs SET ' + ','.join(k+'=?' for k in fields) + " WHERE id=? AND status IN ('queued','running')", (*fields.values(), ident))

    def _loop(self):
        while not self.stop.is_set():
            try:
                ident = self.queue.get(timeout=.2)
            except queue.Empty:
                continue
            event = self.cancels[ident]
            try:
                with self.store.connect() as db:
                    row = db.execute('SELECT * FROM ai_runs WHERE id=?', (ident,)).fetchone()
                if not row or row['status'] != 'queued' or event.is_set():
                    continue
                body = Request.model_validate_json(row['request'])
                job = self.current(row['job_id'], row['attempt'])
                self._update(ident, status='running')
                root = self.store.path(job['id'])/'attempts'/job['attempt']
                evidence = build_evidence(job, root/'index.sqlite', body.scope.model_dump() if body.scope else {})
                self._update(ident, evidence=json.dumps(evidence, ensure_ascii=False))
                scratch = self.store.path(job['id'])/'ai-scratch'
                scratch.mkdir(exist_ok=True, mode=0o700)
                text = run_backend(self.config, body.provider, body.question, evidence, scratch, event)
                try:
                    result = validate_analysis(text, evidence)
                except ValueError:
                    raise BackendError('AI 返回内容未通过结构/证据编号校验，未生成可应用的关联；请重试或缩小问题范围') from None
                self._update(ident, status='succeeded', result=json.dumps(result, ensure_ascii=False))
            except (BackendError, Conflict) as exc:
                self._update(ident, status='failed', error=str(exc))
            except Exception as exc:
                # Exception bodies can contain prompt/provider data. Keep only class.
                logging.getLogger(__name__).warning('AI run %s failed: %s', ident, type(exc).__name__)
                self._update(ident, status='failed', error='AI 分析未完成，请检查服务端配置、输入索引和可用空间')
            finally:
                with self.lock:
                    self.cancels.pop(ident, None)
                self.queue.task_done()


def register_routes(app, service):
    @app.get('/api/ai/config')
    def config():
        return service.config.availability()

    @app.get('/api/jobs/{job_id}/ai-runs')
    def runs(job_id: str):
        return service.list(job_id)

    @app.post('/api/jobs/{job_id}/ai-runs', status_code=202)
    def create(job_id: str, body: Request):
        return service.create(job_id, body)

    @app.get('/api/jobs/{job_id}/ai-runs/{ident}')
    def get(job_id: str, ident: str):
        return service.get(job_id, ident)

    @app.post('/api/jobs/{job_id}/ai-runs/{ident}/cancel')
    def cancel(job_id: str, ident: str):
        return service.cancel(job_id, ident)

    @app.delete('/api/jobs/{job_id}/ai-runs/{ident}')
    def delete(job_id: str, ident: str):
        return service.delete(job_id, ident)

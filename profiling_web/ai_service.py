"""Persisted, single-concurrency AI work queue, separate from profiling jobs."""
import json
import logging
import queue
import threading
import time
import uuid
import shutil

from fastapi import HTTPException
from pydantic import Field, model_validator
from typing import Literal

from .ai_backends import AIConfig, BackendError, run_backend
from .ai_evidence import StrictModel, build_evidence, validate_analysis
from .ai_trace import trace_info, trace_files, session_path, write_session, reusable_session
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
    parent_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    mode: Literal['auto','overall','coverage','shape','mapping'] = 'auto'
    exclude_types: list[str] = Field(default_factory=list, max_length=20)
    trace_mode: Literal['none', 'attach', 'reuse'] = 'none'

    @model_validator(mode='after')
    def check(self):
        if not self.question.strip():
            raise ValueError('请填写分析问题')
        if not self.share_evidence:
            raise ValueError('请确认将所选范围的统计、元数据和问题发送给模型服务')
        if any(not t.strip() or len(t)>200 for t in self.exclude_types):
            raise ValueError('排除类型必须为非空名称，且不超过 200 字符')
        if self.trace_mode == 'reuse' and not self.parent_id:
            raise ValueError('复用 Trace 必须选择要继续的会话')
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
        if body.parent_id:
            parent = self.get(job_id, body.parent_id)
            if parent['stale'] or parent['attempt'] != body.attempt or parent['status'] != 'succeeded':
                raise Conflict('只能继续当前结果版本中成功完成的分析，请新建会话')
        job = self.current(job_id, body.attempt)
        if body.trace_mode != 'none':
            try:
                if body.trace_mode == 'reuse':
                    reusable_session(self.store.path(job_id), parent, body.provider,
                                     self.config.models[body.provider], body.attempt)
                else:
                    info = trace_info(self.store.path(job_id)/'attempts'/body.attempt, job)
                    if not info['within_limit']:
                        raise HTTPException(413, f"完整 Trace 合计 {info['total_bytes']} 字节，超过发送上限 {info['max_bytes']} 字节；请缩小 Step 范围或仅发送统计证据")
            except BackendError as exc:
                raise Conflict(str(exc)) from None
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
        result['trace_context'] = {'reusable': False, 'reason': ''}
        if (result.get('evidence') or {}).get('trace'):
            try:
                if result['stale'] or result['status'] != 'succeeded':
                    raise BackendError('仅当前结果版本中成功完成的会话可复用')
                reusable_session(self.store.path(job_id), result, result['provider'],
                                 self.config.models[result['provider']], result['attempt'])
                result['trace_context']['reusable'] = True
            except BackendError as exc:
                result['trace_context']['reason'] = str(exc)
        return result

    def conversation(self, job_id, parent_id, attempt):
        turns = []
        while parent_id and len(turns)<4:
            try:
                parent = self.get(job_id, parent_id)
            except HTTPException:
                break
            if parent['attempt'] != attempt or parent['status'] != 'succeeded':
                break
            turns.append({'question':parent['request']['question'], 'scope':parent['request'].get('scope'),
                          'answer':parent['result']['summary'], 'findings':parent['result']['findings'][:4]})
            parent_id = parent['request'].get('parent_id')
        return list(reversed(turns))

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
        with self.lock, self.store.connect() as db:
            if db.execute("SELECT 1 FROM ai_runs WHERE job_id=? AND status IN ('queued','running')", (job_id,)).fetchone():
                raise Conflict('请等待本任务的 AI 分析结束后再删除历史，避免清理正在复用的会话')
            db.execute('DELETE FROM ai_runs WHERE id=? AND job_id=?', (ident, job_id))
            remaining = [json.loads(r['evidence']) for r in db.execute(
                'SELECT evidence FROM ai_runs WHERE job_id=? AND evidence IS NOT NULL', (job_id,))]
            key = (run.get('evidence') or {}).get('trace', {}).get('session_key')
            if key and not any(e.get('trace', {}).get('session_key') == key for e in remaining):
                path = session_path(self.store.path(job_id), key)
                if path.exists():
                    shutil.rmtree(path)
        return {'deleted': ident}

    def prepare_trace(self, ident, job, body, evidence, root):
        """Only explicit attach creates a session; failed reuse never resends files."""
        if body.trace_mode == 'none':
            return None, (), None
        info = trace_info(root, job)
        if body.trace_mode == 'attach' and not info['within_limit']:
            raise BackendError('完整 Trace 超过发送上限；请缩小 Step 范围或仅发送统计证据')
        info = trace_info(root, job, hashes=True)
        job_path = self.store.path(job['id'])
        if body.trace_mode == 'reuse':
            parent = self.get(job['id'], body.parent_id)
            key, state = reusable_session(job_path, parent, body.provider,
                                          self.config.models[body.provider], body.attempt)
            if state['files'] != info['files']:
                raise BackendError('Trace 内容已变化，不能复用旧会话；请重新附加 Trace')
        else:
            key = uuid.uuid4().hex
            state = {'provider': body.provider, 'model': self.config.models[body.provider],
                     'attempt': body.attempt, 'files': info['files'], 'backend_id': None}
        path = session_path(job_path, key)
        state.update(ready=False, pending_run=ident)
        write_session(path, state)  # Dirty until BOTH inference and validation succeed.
        info.update(mode=body.trace_mode, session_key=key,
                    attached_bytes=info['total_bytes'] if body.trace_mode == 'attach' else 0)
        evidence['trace'] = info
        evidence['items'].append({'id': f"E{len(evidence['items'])+1}", 'kind': 'trace_manifest', 'data': info})
        session = {'path': path, 'backend_id': state['backend_id']}
        attachments = tuple(zip(trace_files(root, job), info['files'])) if body.trace_mode == 'attach' else ()
        return session, attachments, state

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
                evidence = build_evidence(job, root/'index.sqlite', body.scope.model_dump() if body.scope else {},
                                          body.question, body.mode, body.exclude_types)
                evidence['conversation'] = self.conversation(job['id'], body.parent_id, job['attempt'])
                evidence['conversation_limit'] = '最多继承最近 4 轮；旧回答为待核实上下文，引用需以当前证据为准'
                session, attachments, state = self.prepare_trace(ident, job, body, evidence, root)
                if session:
                    evidence['conversation_limit'] += '；完整 Trace 使用独立 CLI 会话，复用时保留其后端历史（可能被模型压缩）'
                self._update(ident, evidence=json.dumps(evidence, ensure_ascii=False))
                scratch = self.store.path(job['id'])/'ai-scratch'
                scratch.mkdir(exist_ok=True, mode=0o700)
                args = (self.config, body.provider, body.question, evidence, scratch, event)
                output = run_backend(*args, session=session, attachments=attachments) if session else run_backend(*args)
                text = output['text'] if session else output
                try:
                    result = validate_analysis(text, evidence)
                except ValueError:
                    raise BackendError('AI 返回内容未通过结构/证据编号校验，未生成可应用的关联；请重试或缩小问题范围') from None
                with self.lock:
                    if event.is_set():
                        raise BackendError('AI 分析已取消')
                    if session:
                        state.update(ready=True, last_run=ident, backend_id=output['backend_id'])
                        state.pop('pending_run', None)
                        write_session(session['path'], state)
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

    @app.get('/api/jobs/{job_id}/ai-trace')
    def trace(job_id: str, attempt: str):
        job = service.current(job_id, attempt)
        try:
            return trace_info(service.store.path(job_id)/'attempts'/job['attempt'], job)
        except BackendError as exc:
            raise HTTPException(409, str(exc)) from None

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

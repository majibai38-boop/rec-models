import asyncio
from contextlib import asynccontextmanager
import csv
import hashlib
import hmac
import io
import ipaddress
import json
import math
import re
from pathlib import Path
import secrets
import shutil
import time
import uuid
from typing import Literal
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .index import call_evidence, downloadable, query_index, manual_options, manual_group, manual_batch_plan
from .settings import INPUTS, Settings, disk_usage
from .store import Conflict, Store
from .worker import worker_lock


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="GPU / NPU comparison", min_length=1, max_length=120)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if not value.strip():
            raise ValueError("任务名称不能为空")
        return value.strip()


class JobEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=120)
    archived: bool | None = None

    @model_validator(mode="after")
    def nonempty(self):
        if self.name is None and self.archived is None:
            raise ValueError("请提供名称或归档状态")
        if self.name is not None:
            self.name = JobRequest.clean_name(self.name)
        return self


class RunOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gpu_pid: str = Field(default="", max_length=64)
    npu_pid: str = Field(default="", max_length=64)
    gpu_step: str = Field(default="", max_length=128)
    npu_step: str = Field(default="", max_length=128)
    all_steps: bool = False
    include_communication: bool = False
    diagnostic_coverage_threshold: float = Field(default=90, ge=0, le=100, allow_inf_nan=False)
    flow_tolerance_us: float = Field(default=1000, ge=0, le=1000000, allow_inf_nan=False)

    @field_validator("gpu_step", "npu_step")
    @classmethod
    def canonical_step(cls, value):
        # Profiler labels and kernel_details Step ID use different spellings.
        # Restrict normalization to the standard label; custom names stay exact.
        value = value.strip()
        match = re.fullmatch(r"ProfilerStep#([0-9]+)", value, flags=re.IGNORECASE)
        return match.group(1) if match else value

    @model_validator(mode="after")
    def scope(self):
        if self.all_steps and (self.gpu_step or self.npu_step):
            raise ValueError("全量范围不能同时指定 step")
        if not self.all_steps and (not self.gpu_step.strip() or not self.npu_step.strip()):
            raise ValueError("请选择两端 step，或明确选择全量范围")
        return self


class Login(BaseModel):
    token: str = Field(max_length=512)


class ManualSeed(BaseModel):
    model_config = ConfigDict(extra='forbid')
    attempt: str = Field(pattern=r'^[0-9a-f]{32}$')
    target_id: str = Field(min_length=1, max_length=100)
    candidate_id: str = Field(min_length=1, max_length=100)


class ManualConfirmation(ManualSeed):
    reason: str = Field(min_length=1, max_length=500)


class ManualBatchSeed(ManualSeed):
    mode: Literal['strict', 'confirmed_template'] = 'strict'


class ManualBatchConfirmation(ManualBatchSeed):
    reason: str = Field(min_length=1, max_length=500)
    preview_token: str = Field(pattern=r'^[0-9a-f]{64}$')


class ManualUndo(BaseModel):
    model_config = ConfigDict(extra='forbid')
    attempt: str = Field(pattern=r'^[0-9a-f]{32}$')


def session_cookie(token, expiry):
    value = str(expiry)
    return value + "." + hmac.new(token.encode(), value.encode(), hashlib.sha256).hexdigest()


def valid_cookie(token, value):
    try:
        expiry = int(value.split(".")[0])
        return expiry > time.time() and hmac.compare_digest(session_cookie(token, expiry), value)
    except (ValueError, AttributeError):
        return False


def csv_download(path):
    """Protect spreadsheet exports from formulas while preserving numeric values."""
    def safe(value):
        if value and (value[0] in "\t\r\n" or value.lstrip().startswith(("=", "+", "-", "@"))):
            try:
                if math.isfinite(float(value)):
                    return value
            except ValueError:
                pass
            return "'" + value
        return value
    yield "\ufeff"
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        for row in csv.reader(stream):
            writer.writerow([safe(v) for v in row])
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate()


def create_app(settings=None):
    settings = (settings or Settings()).prepare()
    store = Store(settings.data_dir)
    @asynccontextmanager
    async def lifespan(app):
        with worker_lock(settings.data_dir, "api.lock"):
            with store.connect() as db:
                db.execute("UPDATE jobs SET uploading=0 WHERE uploading=1")
            yield

    app = FastAPI(title="GPU / NPU Profiling", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.store, app.state.settings = store, settings
    upload_lock = asyncio.Lock()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)

    @app.middleware("http")
    async def access(request, call_next):
        path = request.url.path
        if path.startswith("/api/"):
            if settings.auth_mode == "local":
                try:
                    peer_local = request.client is not None and ipaddress.ip_address(request.client.host).is_loopback
                except ValueError:
                    peer_local = False
                if (not peer_local or request.url.hostname not in {"localhost", "127.0.0.1", "::1"}
                        or any(h in request.headers for h in ("forwarded", "x-forwarded-for", "x-forwarded-host"))):
                    return JSONResponse({"detail": "免令牌模式仅允许本机直接访问"}, status_code=403)
            bearer = request.headers.get("authorization", "")
            token_ok = bool(settings.token) and bearer.startswith("Bearer ") and secrets.compare_digest(bearer[7:], settings.token)
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not token_ok:
                expected = f"{request.url.scheme}://{request.url.netloc}"
                if request.headers.get("origin") != expected:
                    return JSONResponse({"detail": "Origin 校验失败"}, status_code=403)
            if settings.auth_mode != "local" and path != "/api/session" and not (token_ok or valid_cookie(settings.token, request.cookies.get("profiling_session"))):
                return JSONResponse({"detail": "请登录"}, status_code=401)
            length = request.headers.get("content-length")
            try:
                limit = settings.max_upload_bytes if "/inputs/" in path else 16384
                if length is not None and (int(length) < 0 or int(length) > limit):
                    return JSONResponse({"detail": "请求超过大小上限"}, status_code=413)
            except ValueError:
                return JSONResponse({"detail": "无效 Content-Length"}, status_code=400)
            if request.method not in {"GET", "HEAD", "OPTIONS"} and "/inputs/" not in path:
                chunks, received = [], 0
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > 16384:
                        return JSONResponse({"detail": "请求超过大小上限"}, status_code=413)
                    chunks.append(chunk)
                # Starlette's middleware cached request replays this bounded body.
                request._body = b"".join(chunks)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "任务不存在"}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.post("/api/session")
    def login(body: Login, response: Response):
        if settings.auth_mode == "local":
            return {"authenticated": True, "auth_mode": "local"}
        if not secrets.compare_digest(body.token, settings.token):
            raise HTTPException(401, "访问令牌错误")
        response.set_cookie("profiling_session", session_cookie(settings.token, int(time.time()) + 28800),
                            max_age=28800, httponly=True, secure=settings.secure_cookie, samesite="strict")
        return {"authenticated": True}

    @app.delete("/api/session")
    def logout(response: Response):
        response.delete_cookie("profiling_session", httponly=True, samesite="strict", secure=settings.secure_cookie)
        return {"authenticated": False}

    @app.get("/api/config")
    def config():
        try:
            heartbeat = json.loads((settings.data_dir / "worker-heartbeat.json").read_text())["time"]
        except (OSError, ValueError, KeyError):
            heartbeat = 0
        return {"max_upload_bytes": settings.max_upload_bytes, "storage_bytes": settings.storage_bytes,
                "timeout_seconds": settings.timeout_seconds, "memory_mb": settings.memory_mb,
                "worker_online": time.time() - heartbeat < 15, "workspace_mode": "shared", "concurrency": 1,
                "auth_mode": settings.auth_mode}

    @app.get("/api/health")
    def health():
        with store.connect() as db:
            counts = dict(db.execute("SELECT status,count(*) FROM jobs GROUP BY status").fetchall())
        return {**config(), "storage_used_bytes": disk_usage(settings.data_dir), "tasks": counts}

    @app.get("/api/jobs")
    def jobs(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200),
             search: str = Query("", max_length=120), status: str | None = None, archived: bool = False):
        return store.list(offset, limit, search, status, archived)

    @app.post("/api/jobs", status_code=201)
    def create(body: JobRequest):
        return store.create(body.name)

    @app.get("/api/jobs/{job_id}")
    def get(job_id: str):
        return store.get(job_id)

    @app.patch("/api/jobs/{job_id}")
    def edit(job_id: str, body: JobEdit):
        return store.edit(job_id, name=body.name, archived=body.archived)

    @app.post("/api/jobs/{job_id}/clone", status_code=201)
    async def clone(job_id: str):
        # Serialize with uploads/deletion. Copies are independent: deleting the
        # original cannot invalidate a clone, and no arbitrary path is accepted.
        async with upload_lock:
            source = store.get(job_id)
            if source["status"] in {"uploading", "queued", "running"} or source["uploading"] or set(source["inputs"] or {}) != set(INPUTS):
                raise Conflict("请等待输入冻结且任务结束后再复用")
            needed = sum(item["size"] for item in source["inputs"].values())
            if await asyncio.to_thread(disk_usage, settings.data_dir) + needed > settings.storage_bytes:
                raise HTTPException(507, "复用输入所需空间不足，请先清理任务")
            copied = store.create((source["name"][:112] + " · 新运行")[:120])
            def copy_inputs():
                for role, filename in INPUTS.items():
                    origin = store.path(job_id) / "inputs" / filename
                    if origin.is_symlink() or not origin.is_file():
                        raise Conflict("源输入不可用")
                    destination = store.path(copied["id"]) / "inputs" / filename
                    digest, size = hashlib.sha256(), 0
                    with origin.open("rb") as reader, destination.open("xb") as writer:
                        while chunk := reader.read(1024 * 1024):
                            size += len(chunk)
                            if size > source["inputs"][role]["size"]:
                                raise Conflict("源输入大小已改变，请重新上传")
                            digest.update(chunk)
                            writer.write(chunk)
                    if size != source["inputs"][role]["size"] or digest.hexdigest() != source["inputs"][role]["sha256"]:
                        raise Conflict("源输入校验失败，请重新上传")
            try:
                await asyncio.to_thread(copy_inputs)
                store.update(copied["id"], inputs=source["inputs"])
                return store.enqueue(copied["id"], "inspect", source["options"])
            except Exception:
                store.update(copied["id"], status="failed", error="输入复用失败，可删除此任务释放空间")
                raise

    @app.delete("/api/jobs/{job_id}")
    async def delete_job(job_id: str, confirm: bool = False):
        if not confirm:
            raise HTTPException(422, "删除不可恢复，必须明确确认")
        async with upload_lock:
            store.get(job_id)
            with store.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT status,uploading FROM jobs WHERE id=?", (job_id,)).fetchone()
                if row["status"] in {"queued", "running"} or row["uploading"]:
                    raise Conflict("不能删除活动任务，请先取消并等待结束")
                target = store.path(job_id)
                if target.is_symlink() or target.resolve().parent != (settings.data_dir / "jobs").resolve():
                    raise HTTPException(409, "任务目录不安全")
                if target.exists():
                    await asyncio.to_thread(shutil.rmtree, target)
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
            return {"deleted": job_id}

    @app.put("/api/jobs/{job_id}/inputs/{role}")
    async def upload(job_id: str, role: str, request: Request):
        if role not in INPUTS:
            raise HTTPException(404, "未知文件角色")
        async with upload_lock:
            store.start_upload(job_id)
            target = store.path(job_id) / "inputs" / INPUTS[role]
            temp = target.with_suffix(".upload")
            try:
                used = await asyncio.to_thread(disk_usage, settings.data_dir)
                if used >= settings.storage_bytes:
                    raise HTTPException(507, "数据目录已达到配额")
                size, digest = 0, hashlib.sha256()
                with temp.open("wb") as stream:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > settings.max_upload_bytes:
                            raise HTTPException(413, "文件超过上传上限")
                        if used + size > settings.storage_bytes:
                            raise HTTPException(507, "数据目录将超过配额")
                        digest.update(chunk)
                        await asyncio.to_thread(stream.write, chunk)
                if size == 0:
                    raise HTTPException(422, "不能上传空文件")
                temp.replace(target)
                inputs = store.get(job_id)["inputs"]
                name = unquote(request.headers.get("x-filename", INPUTS[role]))
                name = name.replace("\\", "/").split("/")[-1]
                inputs[role] = {"name": "".join(c for c in name if c.isprintable())[:200], "size": size, "sha256": digest.hexdigest()}
                store.update(job_id, inputs=inputs)
                return inputs[role]
            finally:
                temp.unlink(missing_ok=True)
                store.update(job_id, uploading=0)

    @app.post("/api/jobs/{job_id}/inspect", status_code=202)
    def inspect(job_id: str):
        return store.enqueue(job_id, "inspect")

    @app.post("/api/jobs/{job_id}/run", status_code=202)
    def run(job_id: str, options: RunOptions):
        return store.enqueue(job_id, "compare", options.model_dump())

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        return store.cancel(job_id)

    def result_paths(job_id):
        job = store.get(job_id)
        if job["status"] != "succeeded" or not job["summary"]:
            raise HTTPException(409, "结果尚未通过校验或任务未完成")
        return job, store.path(job_id) / "attempts" / job["attempt"]

    @app.get('/api/jobs/{job_id}/manual-options')
    def manual_candidates(job_id: str, op_type: str = Query(...,max_length=200),
                          device_id: str = Query(...,max_length=64), core_type: str = Query(...,max_length=100),
                          target_id: str = Query('',max_length=100), search: str = Query('',max_length=200),
                          offset: int = Query(0,ge=0,le=100000), gpu_offset: int = Query(0,ge=0,le=100000)):
        job, root = result_paths(job_id)
        try:
            result = manual_options(root/'index.sqlite',op_type,device_id,core_type,target_id,search,offset,gpu_offset)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        result['supported'] = result['supported'] and (root/'reports/manual_catalog.json').is_file()
        from profiling_tools.manual_mapping import link_history
        result.update(attempt=job['attempt'], links=link_history([r for r in job['manual'].get('links',[])
                      if (r['op_type'],r['device_id'],r['core_type']) == (op_type,device_id,core_type)]))
        return result

    def check_manual_limit(links, added):
        if len({r.get('batch_id') or r['id'] for r in links}) >= 100 or len(links) + added > 10000:
            raise HTTPException(422, '单任务最多支持 100 次人工确认、合计 10000 对关联')

    def get_batch(job_id, body):
        from profiling_tools.manual_mapping import BatchConflict
        job, root = result_paths(job_id)
        if job['attempt'] != body.attempt:
            raise Conflict('结果已变化，请刷新后重试')
        try:
            target, plan = manual_batch_plan(root/'index.sqlite', body.target_id, body.candidate_id, body.mode)
        except BatchConflict as exc:
            raise HTTPException(422, exc.detail) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        token = hashlib.sha256(json.dumps([job['attempt'], plan], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return job, target, plan, token

    @app.post('/api/jobs/{job_id}/manual-batch/preview')
    def preview_manual_batch(job_id: str, body: ManualBatchSeed):
        job, _, plan, token = get_batch(job_id, body)
        return {**plan, 'pairs': plan['pairs'][:20], 'attempt': job['attempt'], 'preview_token': token}

    @app.post('/api/jobs/{job_id}/manual-batch', status_code=202)
    def confirm_manual_batch(job_id: str, body: ManualBatchConfirmation):
        if not body.reason.strip():
            raise HTTPException(422, '请填写人工确认理由')
        job, target, plan, token = get_batch(job_id, body)
        if not hmac.compare_digest(token, body.preview_token):
            raise Conflict('批量预览已变化，请重新预览后确认')
        links = job['manual'].get('links', [])
        check_manual_limit(links, plan['pair_count'])
        batch_id, created = uuid.uuid4().hex, time.time()
        additions = [{'id': uuid.uuid4().hex, 'batch_id': batch_id, **pair,
                      'op_type': target['op_type'], 'device_id': target['device_id'], 'core_type': target['core_type'],
                      'reason': body.reason.strip(), 'created': created, 'source_attempt': body.attempt}
                     for pair in plan['pairs']]
        return store.enqueue_manual(job_id, body.attempt, [*links, *additions])

    @app.post('/api/jobs/{job_id}/manual-links',status_code=202)
    def confirm_manual(job_id: str, body: ManualConfirmation):
        job, root = result_paths(job_id)
        if job['attempt'] != body.attempt:
            raise Conflict('结果已变化，请刷新后重试')
        if not body.reason.strip():
            raise HTTPException(422,'请填写人工确认理由')
        links = job['manual'].get('links',[])
        check_manual_limit(links, 1)
        try:
            target = manual_group(root/'index.sqlite','targets',body.target_id)
            candidate = manual_group(root/'index.sqlite','candidates',body.candidate_id)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        link = {'id':uuid.uuid4().hex, 'npu_indices':target['indices'], 'gpu_indices':candidate['indices'],
                'op_type':target['op_type'],'device_id':target['device_id'],'core_type':target['core_type'],
                'reason':body.reason.strip(), 'created':time.time(), 'source_attempt':body.attempt,
                'npu_time_us':target['time_us'], 'gpu_time_us':candidate['time_us']}
        return store.enqueue_manual(job_id,body.attempt,[*links,link])

    @app.post('/api/jobs/{job_id}/manual-links/{link_id}/undo',status_code=202)
    def undo_manual(job_id: str, link_id: str, body: ManualUndo):
        job, _ = result_paths(job_id)
        links = job['manual'].get('links',[])
        selected = [r for r in links if (r.get('batch_id') or r['id']) == link_id]
        if not selected:
            raise HTTPException(404,'人工关联不存在')
        return store.enqueue_manual(job_id,body.attempt,[r for r in links if (r.get('batch_id') or r['id']) != link_id])

    @app.get("/api/jobs/{job_id}/results")
    def results(job_id: str, kind: str = "operators", op_type: str | None = None,
                device_id: str | None = None, core_type: str | None = None,
                npu_index: int | None = Query(None, ge=0), search: str = Query("", max_length=200),
                coverage_max: float | None = Query(None, ge=0, le=100, allow_inf_nan=False),
                npu_ratio_min: float | None = Query(None, ge=0, le=100, allow_inf_nan=False),
                group_id: str | None = Query(None, max_length=120), call_id: str | None = Query(None, max_length=120),
                related_type: str | None = Query(None, max_length=200),
                comparison_status: str | None = Query(None, pattern="^(comparable|pending)$"),
                sort: str = "npu_time", offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=200)):
        _, root = result_paths(job_id)
        try:
            return query_index(root / "index.sqlite", kind=kind, op_type=op_type, device_id=device_id, core_type=core_type,
                               npu_index=npu_index, search=search, coverage_max=coverage_max, npu_ratio_min=npu_ratio_min,
                               sort=sort, offset=offset, limit=limit, group_id=group_id, call_id=call_id, related_type=related_type,
                               comparison_status=comparison_status)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/jobs/{job_id}/artifacts/{name}")
    def artifact(job_id: str, name: str):
        job, root = result_paths(job_id)
        if not downloadable(name) or name not in job["summary"]["reports"]:
            raise HTTPException(404, "文件不可下载")
        path = root / "reports" / name
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404, "文件不存在")
        if path.suffix == ".csv":
            return StreamingResponse(csv_download(path), media_type="text/csv; charset=utf-8",
                                     headers={"Content-Disposition": f'attachment; filename="{name}"'})
        return FileResponse(path, filename=name)

    @app.get("/api/jobs/{job_id}/call-evidence")
    def evidence(job_id: str, call_id: str = Query(..., max_length=120),
                 offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=200)):
        _, root = result_paths(job_id)
        return call_evidence(root / "index.sqlite", call_id, offset, limit)

    static = Path(__file__).parent / "frontend" / "dist"
    if (static / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/")
    def index():
        if not (static / "index.html").is_file():
            raise HTTPException(503, "请先构建 profiling_web/frontend，参见 README")
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    return app

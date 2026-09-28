import asyncio
from contextlib import asynccontextmanager
import csv
import hashlib
import hmac
import io
import json
import math
from pathlib import Path
import secrets
import shutil
import time
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .index import downloadable, query_index
from .settings import INPUTS, Settings, disk_usage
from .store import Conflict, Store
from .worker import worker_lock


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(default="GPU / NPU comparison", min_length=1, max_length=120)


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

    @model_validator(mode="after")
    def scope(self):
        if self.all_steps and (self.gpu_step or self.npu_step):
            raise ValueError("全量范围不能同时指定 step")
        if not self.all_steps and (not self.gpu_step.strip() or not self.npu_step.strip()):
            raise ValueError("请选择两端 step，或明确选择全量范围")
        return self


class Login(BaseModel):
    token: str = Field(max_length=512)


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
            bearer = request.headers.get("authorization", "")
            token_ok = bearer.startswith("Bearer ") and secrets.compare_digest(bearer[7:], settings.token)
            if request.method not in {"GET", "HEAD", "OPTIONS"} and not token_ok:
                expected = f"{request.url.scheme}://{request.url.netloc}"
                if request.headers.get("origin") != expected:
                    return JSONResponse({"detail": "Origin 校验失败"}, status_code=403)
            if path != "/api/session" and not (token_ok or valid_cookie(settings.token, request.cookies.get("profiling_session"))):
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
                "worker_online": time.time() - heartbeat < 15, "workspace_mode": "shared", "concurrency": 1}

    @app.get("/api/jobs")
    def jobs(offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
        return store.list(offset, limit)

    @app.post("/api/jobs", status_code=201)
    def create(body: JobRequest):
        return store.create(body.name)

    @app.get("/api/jobs/{job_id}")
    def get(job_id: str):
        return store.get(job_id)

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

    @app.get("/api/jobs/{job_id}/results")
    def results(job_id: str, kind: str = "operators", op_type: str | None = None,
                device_id: str | None = None, core_type: str | None = None,
                npu_index: int | None = Query(None, ge=0), search: str = Query("", max_length=200),
                coverage_max: float | None = Query(None, ge=0, le=100, allow_inf_nan=False),
                npu_ratio_min: float | None = Query(None, ge=0, le=100, allow_inf_nan=False),
                sort: str = "npu_time", offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=200)):
        _, root = result_paths(job_id)
        try:
            return query_index(root / "index.sqlite", kind=kind, op_type=op_type, device_id=device_id, core_type=core_type,
                               npu_index=npu_index, search=search, coverage_max=coverage_max, npu_ratio_min=npu_ratio_min,
                               sort=sort, offset=offset, limit=limit)
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

    static = Path(__file__).parent / "frontend" / "dist"
    if (static / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=static / "assets"), name="assets")

    @app.get("/")
    def index():
        if not (static / "index.html").is_file():
            raise HTTPException(503, "请先构建 profiling_web/frontend，参见 README")
        return FileResponse(static / "index.html", headers={"Cache-Control": "no-cache"})

    return app

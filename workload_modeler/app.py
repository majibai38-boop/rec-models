"""Loopback-only web service, one isolated analysis at a time."""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import psutil
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .schema import AnalysisRequest, Hardware, Platform, METRIC_UNITS, EmpiricalAssumptions
from .store import Conflict, Store, seed
from .worker import engine_digest
from .screening import assessment
from .empirical import estimate
from . import __version__

UPLOAD_LIMIT=2048*1024**2
STORAGE_LIMIT=10*1024**3
INPUT_NAMES={"trace.json","trace.json.gz","kernel_details.csv"}
STATIC=Path(__file__).with_name("static")


class Manager:
    def __init__(self,store,timeout=600,memory_mb=4096,expanded_limit_mb=4096):
        self.store=store
        self.timeout=timeout
        self.memory_mb=memory_mb
        self.expanded_limit_mb=expanded_limit_mb
        self.lock=threading.RLock()
        self.active={}

    def stop(self,key):
        with self.lock:
            process=self.active.get(key)
            if process is not None and process.poll() is None:
                try:
                    parent=psutil.Process(process.pid)
                    for child in parent.children(recursive=True):
                        child.kill()
                    parent.kill()
                except psutil.NoSuchProcess:
                    pass
                process.wait(timeout=10)

    def start(self,key,action,side="gpu",config=None):
        with self.lock:
            if self.active:
                raise Conflict("已有任务正在执行；单机首版一次处理一个任务")
            self.store.transition(key,{"draft","ready","failed"},
                                  "running" if action=="analyze" else "inspecting",config=config)
            try:
                with (self.store.job_dir(key)/"worker.log").open("w",encoding="utf-8") as log:
                    process=subprocess.Popen([sys.executable,"-m","workload_modeler.worker","--root",str(self.store.root),
                        "--job",key,"--action",action,"--side",side],cwd=Path(__file__).resolve().parent.parent,
                        stdout=log,stderr=subprocess.STDOUT,
                        env={**os.environ,"WORKLOAD_MODELER_EXPANDED_MB":str(self.expanded_limit_mb)},
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
            except Exception as exc:
                self.store.transition(key,{"running","inspecting"},"failed",error=str(exc))
                raise
            self.active[key]=process
            threading.Thread(target=self.monitor,args=(key,process),daemon=True).start()

    def monitor(self,key,process):
        start=time.monotonic()
        error=None
        while process.poll() is None:
            try:
                p=psutil.Process(process.pid)
                rss=sum(x.memory_info().rss for x in [p,*p.children(recursive=True)])
                if rss>self.memory_mb*1024**2:
                    error="超过分析内存软限制，请缩小范围"
                if time.monotonic()-start>self.timeout:
                    error="分析超时，请缩小范围"
                if error:
                    self.stop(key)
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(.25)
        process.wait()
        with self.lock:
            if self.store.job(key)["status"] in {"running","inspecting"}:
                self.store.transition(key,{"running","inspecting"},"failed",error=error or "Worker异常退出，详见本地worker.log")
            self.active.pop(key,None)


def create_app(data_dir=None, *, test_mode=False, upload_limit_mb=2048,
               expanded_limit_mb=4096, memory_mb=4096, timeout=600):
    if any(not isinstance(v,int) or v<=0 for v in (upload_limit_mb,expanded_limit_mb,memory_mb,timeout)):
        raise ValueError("上传、解压、内存和超时限制必须为正整数")
    upload_limit=upload_limit_mb*1024**2
    store=Store(data_dir or Path(__file__).with_name("data"))
    manager=Manager(store,timeout=timeout,memory_mb=memory_mb,expanded_limit_mb=expanded_limit_mb)

    @asynccontextmanager
    async def lifespan(app):
        # OS lock is released on crash, preventing a second server from corrupting task state.
        lock=(store.root/"server.lock").open("a+b")
        try:
            if os.name=="nt":
                import msvcrt
                lock.seek(0)
                if not lock.read(1):
                    lock.write(b"1"); lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            with store.connect() as c:
                c.execute("UPDATE jobs SET status='failed',error='服务重启，任务中断；请复用输入新建分析' WHERE status IN ('running','inspecting','uploading')")
            seed(store)
            yield
        finally:
            for key in list(manager.active):
                manager.stop(key)
            lock.close()

    app=FastAPI(title="Workload Modeler",version=__version__,lifespan=lifespan)
    hosts=["localhost","127.0.0.1","[::1]"]+(["testserver"] if test_mode else [])
    app.add_middleware(TrustedHostMiddleware,allowed_hosts=hosts)
    app.state.store=store
    app.state.manager=manager

    @app.middleware("http")
    async def local_security(request,call_next):
        peer=request.client.host if request.client else ""
        if peer not in {"127.0.0.1","::1"} and not test_mode:
            return JSONResponse({"detail":"仅允许本机访问"},status_code=403)
        if request.method not in {"GET","HEAD","OPTIONS"}:
            origin=request.headers.get("origin")
            if (origin and origin!=str(request.base_url).rstrip("/")) or request.headers.get("sec-fetch-site")=="cross-site":
                return JSONResponse({"detail":"拒绝跨站请求"},status_code=403)
        response=await call_next(request)
        response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Content-Security-Policy"]="default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        response.headers["Cache-Control"]="no-store"
        return response

    @app.exception_handler(KeyError)
    async def missing(request,exc):
        return JSONResponse({"detail":str(exc)},status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request,exc):
        return JSONResponse({"detail":str(exc)},status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request,exc):
        if isinstance(exc,ValidationError):
            messages=[".".join(map(str,e["loc"]))+": "+e["msg"] for e in exc.errors(include_input=False)]
            return JSONResponse({"detail":"请检查输入：\n"+"\n".join(messages)},status_code=422)
        return JSONResponse({"detail":str(exc)},status_code=422)

    async def body(request):
        chunks=bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks)>1024**2:
                raise HTTPException(413,"JSON超过1MiB")
        try:
            value=json.loads(chunks)
        except (ValueError,UnicodeError):
            raise ValueError("请求必须是有效JSON")
        if not isinstance(value,dict):
            raise ValueError("请求必须是JSON对象")
        return value

    @app.get("/")
    def home():
        return FileResponse(STATIC/"index.html")

    @app.get("/static/{name}")
    def static(name:str):
        if name == "theme.css":
            return FileResponse(STATIC.parent.parent/"web_shared"/"theme.css",media_type="text/css")
        if name not in {"app.js","catalog.js","management.js","diagnostics.js","assessment.js","empirical.js","workflow.js","style.css","forms.css"}:
            raise HTTPException(404)
        return FileResponse(STATIC/name)

    @app.get("/api/meta")
    def meta():
        return dict(version=__version__,metric_units=METRIC_UNITS,upload_limit_mb=upload_limit_mb,
                    empirical_defaults=EmpiricalAssumptions().model_dump(),
                    expanded_limit_mb=expanded_limit_mb,memory_limit_mb=memory_mb,timeout_seconds=timeout,worker_limit=1,
                    boundary="本机免令牌；仅单设备预测；不执行上传代码")

    @app.get("/api/catalog/{kind}")
    def catalog(kind:str,deleted:bool=False):
        if kind not in {"hardware","platform"}:
            raise HTTPException(404)
        return store.list(kind,deleted=deleted)

    @app.get("/api/catalog/{kind}/{key}/history")
    def history(kind:str,key:str):
        return store.history(kind,key)

    @app.post("/api/catalog/{kind}")
    async def add(kind:str,request:Request):
        return store.save(kind,await body(request))

    @app.put("/api/catalog/{kind}/{key}")
    async def update(kind:str,key:str,request:Request):
        data=await body(request)
        return store.save(kind,data["payload"],key,data["expected_revision"])

    @app.delete("/api/catalog/{kind}/{key}")
    def delete_catalog(kind:str,key:str,expected_revision:int=Query(ge=1),confirm:bool=False):
        if not confirm:
            raise ValueError("请确认移入回收站；历史版本仍将保留")
        return store.trash_catalog(kind,key,expected_revision)

    @app.post("/api/catalog/{kind}/{key}/restore")
    def restore_catalog(kind:str,key:str,expected_revision:int=Query(ge=1)):
        return store.trash_catalog(kind,key,expected_revision,deleted=False)

    @app.get("/api/jobs")
    def jobs(deleted:bool=False,search:str=Query(default="",max_length=120),
             offset:int=Query(default=0,ge=0),limit:int=Query(default=200,ge=1,le=201)):
        return store.jobs(deleted=deleted,search=search,offset=offset,limit=limit)

    @app.post("/api/jobs")
    async def new(request:Request):
        data=await body(request)
        name=str(data.get("name","新建分析")).strip()
        if not 1<=len(name)<=120:
            raise ValueError("任务名须为1–120字符")
        return store.create_job(name)

    @app.get("/api/jobs/{key}")
    def job(key:str):
        return store.job(key)

    @app.patch("/api/jobs/{key}")
    async def rename(key:str,request:Request):
        data=await body(request)
        name=str(data.get("name","")).strip()
        if not 1<=len(name)<=120:
            raise ValueError("任务名须为1–120字符")
        return store.rename_job(key,name)

    @app.delete("/api/jobs/{key}")
    def delete_job(key:str,confirm:bool=False):
        if not confirm:
            raise ValueError("请确认移入回收站；输入和结果仍将保留，不释放磁盘空间")
        with manager.lock:
            if key in manager.active:
                raise Conflict("任务进程尚未退出，请稍后再试")
            return store.trash_job(key)

    @app.post("/api/jobs/{key}/restore")
    def restore_job(key:str):
        return store.trash_job(key,deleted=False)

    @app.put("/api/jobs/{key}/inputs/{filename}")
    async def upload(key:str,filename:str,request:Request):
        if filename not in INPUT_NAMES:
            raise ValueError("只允许trace.json、trace.json.gz、kernel_details.csv")
        directory=store.job_dir(key)
        usage=sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file())
        store.transition(key,{"draft","ready","failed"},"uploading")
        tmp=directory/(filename+".upload")
        size=0
        try:
            with tmp.open("wb") as f:
                async for chunk in request.stream():
                    size+=len(chunk)
                    if size>upload_limit:
                        raise HTTPException(413,f"超过单文件 {upload_limit_mb} MiB 限制；可通过 --upload-limit-mb 调整后重启服务")
                    if usage+size>STORAGE_LIMIT:
                        raise HTTPException(413,"超过10GiB空间软配额，请清理不再使用的任务文件")
                    f.write(chunk)
            if not size:
                raise ValueError("上传文件不能为空")
            os.replace(tmp,directory/filename)
            if filename.startswith("trace."):
                (directory/("trace.json.gz" if filename=="trace.json" else "trace.json")).unlink(missing_ok=True)
            (directory/"inventory.json").unlink(missing_ok=True)
            store.transition(key,{"uploading"},"draft")
        except BaseException:
            tmp.unlink(missing_ok=True)
            store.transition(key,{"uploading"},"draft")
            raise
        return {"bytes":size,"filename":filename}

    def require_inputs(key,side):
        directory=store.job_dir(key)
        has_trace=any((directory/name).is_file() for name in ('trace.json','trace.json.gz'))
        if not has_trace and (side=='gpu' or not (directory/'kernel_details.csv').is_file()):
            raise ValueError('请先选择输入文件：GPU需要Trace；NPU需要Trace或kernel_details.csv。')

    @app.post("/api/jobs/{key}/inspect")
    async def inspect(key:str,request:Request):
        data=await body(request)
        side=data.get("side","gpu")
        if side not in {"gpu","npu"}:
            raise ValueError("side必须为gpu/npu")
        require_inputs(key,side)
        manager.start(key,"inspect",side)
        return store.job(key)

    @app.get("/api/jobs/{key}/inventory")
    def inventory_result(key:str):
        path=store.job_dir(key)/"inventory.json"
        if not path.exists():
            raise HTTPException(404,"预检查结果尚未生成")
        return FileResponse(path)

    @app.post("/api/jobs/{key}/run")
    async def run(key:str,request:Request):
        req=AnalysisRequest.model_validate(await body(request))
        if not req.scope.step and not req.scope.all_steps:
            raise ValueError("必须选择Step或明确确认全范围")
        require_inputs(key,req.scope.side)
        src=store.active("platform",req.source_platform_id,req.source_platform_revision)
        dst=store.active("platform",req.target_platform_id,req.target_platform_revision)
        if src["card_count"]!=1 or dst["card_count"]!=1:
            raise ValueError("首版只支持单设备建模；不能把多卡通信规格当作单卡倍率")
        sh=store.active("hardware",src["hardware_id"],src["hardware_revision"])
        th=store.active("hardware",dst["hardware_id"],dst["hardware_revision"])
        if (req.scope.side=="gpu" and sh["vendor"]!="NVIDIA") or (req.scope.side=="npu" and sh["vendor"]!="Ascend"):
            raise ValueError("输入类型与源硬件厂商不一致")
        config=dict(request=req.model_dump(),source_platform=src,target_platform=dst,source_hardware=sh,target_hardware=th,
                    engine_sha256=engine_digest())
        manager.start(key,"analyze",config=config)
        return store.job(key)

    @app.post("/api/jobs/{key}/cancel")
    def cancel(key:str):
        with manager.lock:
            manager.stop(key)
            store.transition(key,{"running","inspecting"},"cancelled")
        return store.job(key)

    @app.post("/api/jobs/{key}/clone")
    def clone(key:str):
        import shutil
        directory=store.job_dir(key)
        if store.job(key)["status"] in {"uploading","running","inspecting"}:
            raise Conflict("运行期间不能复制输入")
        input_size=sum((directory/name).stat().st_size for name in INPUT_NAMES if (directory/name).is_file())
        usage=sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file())
        if usage+input_size>STORAGE_LIMIT:
            raise HTTPException(413,"复制输入将超过10GiB空间软配额")
        new=store.create_job(store.job(key)["name"][:117]+" 副本")
        for name in INPUT_NAMES:
            if (directory/name).is_file():
                shutil.copyfile(directory/name,store.job_dir(new["id"])/name)
        return new

    def result(key):
        if store.job(key)["status"]!="completed":
            raise Conflict("任务未完成，无可发布结果")
        return json.loads((store.job_dir(key)/"result.json").read_text(encoding="utf-8"))

    @app.get("/api/jobs/{key}/summary")
    def summary(key:str):
        data=result(key)
        if 'assessment' not in data:
            data['assessment']=assessment(data,historical=True)
        if 'empirical' not in data:
            data['empirical']=estimate(data['profile_summary'],data['units'],data['config'])
            data['empirical']['preview']=True
            data['empirical']['note']='旧结果的默认经验假设预览；未改写原结果。下载完整JSON仍是原版本。'
        return {k:v for k,v in data.items() if k not in {"units","groups"}}

    @app.post('/api/jobs/{key}/empirical-preview')
    async def empirical_preview(key:str,request:Request):
        options=EmpiricalAssumptions.model_validate(await body(request))
        data=result(key)
        prediction=estimate(data['profile_summary'],data['units'],data['config'],options.model_dump())
        prediction.update(preview=True,note='经验假设预览；未改写原任务或精细预测。可单独导出此场景。',
            provenance=dict(job_id=key,input_sha256=data.get('input_sha256'),config=data['config'],
                            engine_version=__version__,engine_sha256=engine_digest()))
        return prediction

    @app.get("/api/jobs/{key}/groups")
    def groups(key:str,search:str="",offset:int=0,limit:int=50):
        if offset<0 or not 1<=limit<=200:
            raise ValueError("无效分页")
        rows=[g for g in result(key)["groups"] if search.lower() in (g["op"]+g["shape"]+g["dtype"]).lower()]
        return dict(total=len(rows),items=rows[offset:offset+limit])

    @app.get("/api/jobs/{key}/units")
    def units(key:str,signature:str,offset:int=0,limit:int=20):
        if offset<0 or not 1<=limit<=100:
            raise ValueError("无效分页")
        rows=[u for u in result(key)["units"] if u["signature"]==signature]
        return dict(total=len(rows),items=rows[offset:offset+limit])

    @app.get("/api/jobs/{key}/download/{filename}")
    def download(key:str,filename:str):
        if filename not in {"result.json","operators.csv"}:
            raise HTTPException(404)
        if store.job(key)["status"]!="completed":
            raise Conflict("任务未完成")
        return FileResponse(store.job_dir(key)/filename,filename=filename)

    return app

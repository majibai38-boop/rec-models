import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from profiling_tools.provenance import write_json
from .settings import INPUTS, Settings, disk_usage
from .store import Store


@contextlib.contextmanager
def worker_lock(root, name="worker.lock"):
    """An OS lock, automatically released even after a worker crash."""
    with (Path(root) / name).open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if handle.read(1) == b"":
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def kill_process(process):
    try:
        parent = psutil.Process(process.pid)
        for child in parent.children(recursive=True):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        parent.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def process_tree_rss(parent):
    """Sum RSS including venv launchers' children; shared pages count per process.

    Exited processes are normal races. AccessDenied propagates so the worker
    never silently runs an analysis without its configured memory guard.
    """
    try:
        processes = [parent, *parent.children(recursive=True)]
    except psutil.NoSuchProcess:
        return 0
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


class Worker:
    def __init__(self, settings):
        self.settings = settings.prepare()
        self.store = Store(settings.data_dir)
        self.stopping = False

    def execute(self, job):
        root = self.store.path(job["id"])
        attempt = root / "attempts" / job["attempt"]
        attempt.mkdir(parents=True)
        config = {key: str(root / "inputs" / name) for key, name in INPUTS.items()}
        config.update(job["options"] or {})
        config.update(output_dir=str(attempt / "reports"), require_explicit_scope=True)
        task = attempt / "task.json"
        write_json(task, {"kind": job["kind"], "config": config,
                          "manual": job.get('manual',{}),
                          "parent": {"pid": os.getpid(), "created": psutil.Process().create_time()}})
        with (attempt / "worker.log").open("wb") as log:
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            # Analysis needs neither the web secret nor network credentials.
            env.pop("PROFILING_WEB_TOKEN", None)
            proc = subprocess.Popen([sys.executable, "-m", "profiling_web.runner", str(task)],
                                    cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, env=env,
                                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            monitored_process = psutil.Process(proc.pid)
            write_json(attempt / "process.json", {"pid": proc.pid, "created": monitored_process.create_time()})
            started = time.monotonic()
            reason, last_stage, last_disk_check = None, None, 0
            try:
                while proc.poll() is None:
                    current = self.store.get(job["id"])
                    if self.stopping:
                        reason = ("interrupted", "Worker 停止，请手动重试")
                    elif current["cancel"]:
                        reason = ("cancelled", "用户取消了任务")
                    elif time.monotonic() - started > self.settings.timeout_seconds:
                        reason = ("failed", "任务超过运行时间上限")
                    try:
                        memory = process_tree_rss(monitored_process)
                        if memory > self.settings.memory_mb * 1024**2:
                            reason = ("failed", "任务超过内存上限")
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.AccessDenied:
                        reason = ("failed", "无法监控分析进程树内存，任务已停止")
                    if time.monotonic() - last_disk_check > 5:
                        write_json(self.settings.data_dir / "worker-heartbeat.json", {"time": time.time()})
                        if disk_usage(self.settings.data_dir) > self.settings.storage_bytes:
                            reason = ("failed", "数据目录超过存储配额，请管理员清理")
                        last_disk_check = time.monotonic()
                    try:
                        stage = json.loads((attempt / "progress.json").read_text(encoding="utf-8"))["stage"]
                        if stage != last_stage:
                            self.store.update(job["id"], stage=stage)
                            last_stage = stage
                    except (OSError, ValueError, KeyError):
                        pass
                    if reason:
                        kill_process(proc)
                        break
                    time.sleep(0.2)
            finally:
                if proc.poll() is None:
                    kill_process(proc)
            # A cancellation racing with completion still withholds publication.
            if not reason and self.store.get(job["id"])["cancel"]:
                reason = ("cancelled", "用户取消了任务")
            if reason:
                self.store.update(job["id"], status=reason[0], stage=reason[0], error=reason[1])
                return
            try:
                result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {"status": "failed", "error": "分析子进程异常退出，请查看服务器任务日志"}
            if proc.returncode != 0 and result["status"] == "succeeded":
                result = {"status": "failed", "error": "子进程异常退出，结果未发布"}
            self.store.update(job["id"], status=result["status"], stage=result["status"],
                              **{k: v for k, v in result.items() if k in {"inspection", "summary", "error"}})

    def recover(self):
        # Only kill the exact analysis process belonging to an interrupted run.
        with self.store.connect() as db:
            rows = db.execute("SELECT id,attempt FROM jobs WHERE status='running'").fetchall()
        for row in rows:
            path = self.store.path(row["id"]) / "attempts" / row["attempt"] / "process.json"
            try:
                info = json.loads(path.read_text(encoding="utf-8"))
                proc = psutil.Process(info["pid"])
                if abs(proc.create_time() - info["created"]) < 0.01 and "profiling_web.runner" in proc.cmdline():
                    proc.kill()
                    proc.wait(timeout=10)
            except (OSError, ValueError, psutil.Error):
                pass
        self.store.recover()

    def run(self):
        with worker_lock(self.settings.data_dir):
            self.recover()
            while not self.stopping:
                write_json(self.settings.data_dir / "worker-heartbeat.json", {"time": time.time()})
                job = self.store.claim()
                if job:
                    try:
                        self.execute(job)
                    except Exception:
                        self.store.update(job["id"], status="failed", stage="failed", error="Worker 异常，请查看服务器日志")
                        import traceback
                        traceback.print_exc()
                else:
                    time.sleep(0.5)


def main():
    worker = Worker(Settings())
    def stop(*_):
        worker.stopping = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    worker.run()


if __name__ == "__main__":
    main()

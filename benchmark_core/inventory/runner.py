"""Bounded command/file reads. No shell interpolation or implicit privilege escalation."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path


def _stop(process):
    if os.name == "nt":
        try:
            subprocess.run([str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/taskkill.exe"),
                            "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired):
            pass
        if process.poll() is None:
            process.kill()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_command(argv, timeout, limit):
    started = time.monotonic()
    result = {"argv": argv, "status": "ok", "returncode": None, "stdout": "", "stderr": ""}
    env = dict(os.environ, LC_ALL="C", LANG="C")
    options = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    try:
        process = subprocess.Popen(argv, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, env=env, **options)
    except (OSError, ValueError) as exc:
        result.update(status="unavailable" if isinstance(exc, FileNotFoundError) else "error", stderr=str(exc))
        result["duration_seconds"] = round(time.monotonic() - started, 4)
        return result
    chunks = {"stdout": bytearray(), "stderr": bytearray()}
    overflow = threading.Event()
    def drain(name, pipe):
        try:
            while True:
                block = pipe.read1(4096)
                if not block:
                    break
                available = limit - len(chunks[name])
                chunks[name].extend(block[:max(available, 0)])
                if len(block) > available:
                    overflow.set()
        finally:
            pipe.close()
    threads = [threading.Thread(target=drain, args=(name, getattr(process, name)), daemon=True)
               for name in chunks]
    for thread in threads:
        thread.start()
    while process.poll() is None or any(t.is_alive() for t in threads):
        if overflow.is_set():
            result["status"] = "output_limit"
            _stop(process)
            break
        if time.monotonic() - started > timeout:
            result["status"] = "timeout"
            _stop(process)
            break
        time.sleep(.02)
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
    for thread in threads:
        thread.join(timeout=.3)
    result.update({name: bytes(data).decode("utf-8", errors="replace") for name, data in chunks.items()})
    result["returncode"] = process.poll()
    if overflow.is_set():
        result["status"] = "output_limit"
    elif result["status"] == "ok" and process.returncode != 0:
        result["status"] = "error"
    result["duration_seconds"] = round(time.monotonic() - started, 4)
    return result


def read_file(paths, limit):
    errors = []
    for name in paths:
        try:
            with Path(name).open("rb") as handle:
                data = handle.read(limit + 1)
            return {"paths": paths, "selected_path": name, "status": "output_limit" if len(data) > limit else "ok",
                    "stdout": data[:limit].decode("utf-8", errors="replace"), "stderr": "", "returncode": None}
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    return {"paths": paths, "status": "unavailable", "stdout": "", "stderr": "\n".join(errors), "returncode": None}


class LocalRunner:
    def __call__(self, probe, settings, device=None):
        replacements = {"{device_id}": str((device or {}).get("physical_id") or ""),
                        "{chip_id}": str((device or {}).get("chip_id") or "")}
        if "argv" in probe:
            argv = [replacements.get(arg, arg) for arg in probe["argv"]]
            if any(not x for x in argv):
                return {"status": "error", "stdout": "", "stderr": "Missing device/chip identifier", "argv": argv}
            return run_command(argv, settings["timeout_seconds"], settings["max_output_bytes"])
        return read_file(probe["paths"], settings["max_output_bytes"])

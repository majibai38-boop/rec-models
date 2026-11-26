"""Attempt layout, device-local locks and exit codes for the first-phase workflow.

Nothing in this module starts a model, container or network transfer. P1-02..P1-04 add those
callers and must reuse the same attempt directory, lock and exit-code contract.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re
import socket
import time
import uuid
from pathlib import Path

from benchmark_core.inventory.config import ConfigError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BLOCKED = 3
EXIT_BUSY = 4
EXIT_INTERRUPTED = 5

PHASES = ("prepare", "run", "measure", "profile")

_ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ATTEMPT = re.compile(r"^attempt-([0-9]+)$")


class LockBusy(RuntimeError):
    """Another process or host already holds the requested device lock."""


def validate_id(value, where):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ConfigError(f"{where}: use a lowercase identifier, max 64 characters")
    return value


def runs_root(archive_root):
    return Path(archive_root) / "runs"


def attempt_dir(archive_root, run_id, attempt_id):
    validate_id(run_id, "run_id")
    validate_id(attempt_id, "attempt_id")
    return runs_root(archive_root) / run_id / attempt_id


def next_attempt_id(existing):
    """Return the next free attempt id; ids are never reused inside one run."""
    numbers = [int(match.group(1)) for name in existing for match in [_ATTEMPT.fullmatch(name)] if match]
    return f"attempt-{max(numbers, default=0) + 1}"


def create_attempt(archive_root, run_id, attempt_id=None):
    """Create a new attempt directory, or fail; existing attempts are never reused."""
    validate_id(run_id, "run_id")
    run_dir = runs_root(archive_root) / run_id
    existing = sorted(path.name for path in run_dir.iterdir() if path.is_dir()) if run_dir.is_dir() else []
    attempt_id = attempt_id or next_attempt_id(existing)
    validate_id(attempt_id, "attempt_id")
    target = attempt_dir(archive_root, run_id, attempt_id)
    try:
        target.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ConfigError(f"attempt already exists: {target}") from exc
    return target, attempt_id


def _now(value, where):
    now = value or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError(f"{where}: expected a timezone-aware timestamp")
    return now


def _lock_path(state_root, backend, device_id):
    if backend not in {"cuda", "ascend"}:
        raise ConfigError("lock backend: expected cuda or ascend")
    if not isinstance(device_id, str) or not device_id.strip() or len(device_id) > 128:
        raise ConfigError("lock device_id: expected nonempty text (max 128 characters)")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", device_id)
    return Path(state_root) / f"{backend}-{safe}.lock"


def _age_seconds(payload, path, now):
    acquired = payload.get("acquired_at") if isinstance(payload, dict) else None
    if isinstance(acquired, str):
        try:
            parsed = dt.datetime.fromisoformat(acquired)
            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                return (now - parsed).total_seconds()
        except ValueError:
            pass
    return now.timestamp() - path.stat().st_mtime


class DeviceLock:
    """One lock file per machine-local device, written atomically with O_EXCL."""

    def __init__(self, state_root, backend, device_id, *, command=None):
        self.path = _lock_path(state_root, backend, device_id)
        self.backend = backend
        self.device_id = device_id
        self.command = command
        self.token = None
        self.payload = None

    def holder(self):
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def acquire(self, timeout_seconds=0, poll_seconds=0.25):
        if type(timeout_seconds) not in (int, float) or not 0 <= timeout_seconds <= 3600:
            raise ConfigError("timeout_seconds: expected 0..3600")
        if type(poll_seconds) not in (int, float) or not 0 < poll_seconds <= 60:
            raise ConfigError("poll_seconds: expected 0..60")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_seconds
        while True:
            payload = {
                "token": str(uuid.uuid4()),
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "backend": self.backend,
                "device_id": self.device_id,
                "command": self.command,
                "acquired_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    held = self.holder() or {}
                    raise LockBusy(
                        f"device {self.backend}:{self.device_id} is locked by pid "
                        f"{held.get('pid')} on {held.get('hostname')}; delete {self.path} only after confirming it is stale"
                    ) from None
                time.sleep(min(poll_seconds, remaining))
                continue
            try:
                os.write(descriptor, data)
            finally:
                os.close(descriptor)
            self.token, self.payload = payload["token"], payload
            return payload

    def release(self):
        if self.token is None:
            return False
        held = self.holder()
        if not held or held.get("token") != self.token:
            raise ConfigError("lock token changed; refusing to release a lock owned by another process")
        self.path.unlink()
        self.token = self.payload = None
        return True

    def break_stale(self, older_than_seconds=3600, now=None):
        """Remove a lock only after proving it is older than the threshold."""
        if type(older_than_seconds) not in (int, float) or not 0 < older_than_seconds <= 604800:
            raise ConfigError("older_than_seconds: expected 1..604800")
        held = self.holder()
        if held is None:
            return False
        if _age_seconds(held, self.path, _now(now, "now")) <= older_than_seconds:
            return False
        self.path.unlink()
        return True

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.release()
        except ConfigError:
            if exc_type is None:
                raise
        return False


def record_interrupt(attempt_directory, phase, reason, *, exit_code=EXIT_INTERRUPTED, at=None):
    """Append one interruption record; attempts are never rewritten to hide failures."""
    if phase not in PHASES:
        raise ConfigError(f"interrupt phase: expected one of {list(PHASES)}")
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
        raise ConfigError("interrupt reason: expected nonempty text (max 2000 characters)")
    if exit_code not in {EXIT_BUSY, EXIT_INTERRUPTED, EXIT_BLOCKED}:
        raise ConfigError("interrupt exit_code: expected a defined non-success code")
    now = _now(at, "interrupt time")
    logs = Path(attempt_directory) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / "interruptions.jsonl"
    entry = {"at": now.isoformat(), "phase": phase, "reason": reason, "exit_code": exit_code,
             "hostname": socket.gethostname(), "pid": os.getpid()}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n")
    return path

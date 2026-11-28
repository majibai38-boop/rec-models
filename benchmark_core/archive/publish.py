"""Verify and atomically publish one staged attempt; never computes benchmark values.

A failed experiment can still be published: the manifest carries the missing phases. What
must never happen is publishing without a complete, verified file list or overwriting an
existing attempt with different content.
"""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

from benchmark_core.archive.layout import attempt_published, attempt_staging
from benchmark_core.inventory.config import ConfigError, load_json
from benchmark_core.planning.contracts import digest, exact
from benchmark_core.workflow.manifest import (
    artifacts_index, sha256_file, validate_artifacts, verify_artifacts, write_json,
)
from benchmark_core.workflow.state import validate_id

COMPLETE = "COMPLETE"
INDEX = "artifacts.json"
COMPLETE_VERSION = "recbench.archive-complete/v1"
RESERVED = {COMPLETE, INDEX}

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T")


def _now(value, where):
    now = value or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError(f"{where}: expected a timezone-aware timestamp")
    return now


def _validate_complete(value):
    exact(value, "schema_version run_id attempt_id artifacts_sha256 published_at", "complete")
    if value["schema_version"] != COMPLETE_VERSION:
        raise ConfigError(f"complete.schema_version: expected {COMPLETE_VERSION}")
    validate_id(value["run_id"], "complete.run_id")
    validate_id(value["attempt_id"], "complete.attempt_id")
    digest(value["artifacts_sha256"], "complete.artifacts_sha256")
    if not isinstance(value["published_at"], str) or not _TIMESTAMP.match(value["published_at"]):
        raise ConfigError("complete.published_at: expected an ISO timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value["published_at"])
    except ValueError as exc:
        raise ConfigError("complete.published_at: expected an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigError("complete.published_at: expected a timezone-aware timestamp")
    return value


def _index_of(directory, problems):
    path = Path(directory) / INDEX
    if not path.is_file():
        problems.append({"path": INDEX, "reason": "missing artifacts index"})
        return None
    try:
        index = load_json(path)
        validate_artifacts(index)
    except ConfigError as exc:
        problems.append({"path": INDEX, "reason": f"invalid artifacts index: {exc}"})
        return None
    return index


def verify(attempt_dir):
    """Recompute size and hash for every listed file; report anything unexpected or missing."""
    root = Path(attempt_dir)
    problems = []
    if not root.is_dir():
        return {"ok": False, "problems": [{"path": str(root), "reason": "not a directory"}],
                "files": 0, "total_bytes": 0, "artifacts_sha256": None}
    complete = None
    complete_path = root / COMPLETE
    if not complete_path.is_file():
        problems.append({"path": COMPLETE, "reason": "missing COMPLETE marker"})
    else:
        try:
            complete = _validate_complete(load_json(complete_path))
        except ConfigError as exc:
            problems.append({"path": COMPLETE, "reason": f"invalid COMPLETE marker: {exc}"})
    index = _index_of(root, problems)
    artifacts_sha256 = None
    if index is not None:
        index_path = root / INDEX
        artifacts_sha256 = sha256_file(index_path)
        if complete is not None and complete["artifacts_sha256"] != artifacts_sha256:
            problems.append({"path": COMPLETE, "reason": "artifacts_sha256 does not match artifacts.json"})
        if complete is not None:
            if complete["run_id"] != root.parent.name or complete["attempt_id"] != root.name:
                problems.append({"path": COMPLETE, "reason": "run_id/attempt_id do not match the directory"})
        report = verify_artifacts(index, root)
        problems.extend(report["problems"])
        listed = {entry["path"] for entry in index["files"]} | RESERVED
        for path in sorted(root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative not in listed:
                    problems.append({"path": relative, "reason": "unlisted file"})
    return {"ok": not problems, "problems": problems,
            "files": len(index["files"]) if index is not None else 0,
            "total_bytes": index["total_bytes"] if index is not None else 0,
            "artifacts_sha256": artifacts_sha256}


def _result(status, final, *, files, total_bytes, artifacts_sha256, staging=None):
    result = {"status": status, "attempt": f"{final.parent.name}/{final.name}",
              "path": str(final), "files": files, "total_bytes": total_bytes,
              "artifacts_sha256": artifacts_sha256}
    if staging is not None:
        result["staging"] = str(staging)
    return result


def publish(archive_root, run_id, attempt_id, *, published_at=None):
    """Validate the staged attempt, write the index and COMPLETE, then move it atomically."""
    staged = attempt_staging(archive_root, run_id, attempt_id)
    final = attempt_published(archive_root, run_id, attempt_id)
    if not staged.is_dir():
        raise ConfigError(f"staging attempt not found: {staged}")
    index = artifacts_index(staged)
    write_json(staged / INDEX, index)
    artifacts_sha256 = sha256_file(staged / INDEX)
    complete = {"schema_version": COMPLETE_VERSION, "run_id": run_id, "attempt_id": attempt_id,
                "artifacts_sha256": artifacts_sha256, "published_at": _now(published_at, "published_at").isoformat()}
    write_json(staged / COMPLETE, complete)
    report = verify(staged)
    if not report["ok"]:
        details = "; ".join(f"{problem['path']}: {problem['reason']}" for problem in report["problems"])
        raise ConfigError(f"staged attempt failed verification: {details}")
    if final.exists():
        existing_problems = []
        existing = _index_of(final, existing_problems)
        if existing is None:
            raise ConfigError("conflict: attempt already published but its artifacts index is invalid")
        if existing == index:
            return _result("existing", final, files=report["files"], total_bytes=report["total_bytes"],
                           artifacts_sha256=artifacts_sha256, staging=staged)
        raise ConfigError(f"conflict: attempt {attempt_id} already published with different content")
    final.parent.mkdir(parents=True, exist_ok=True)
    try:
        staged.rename(final)
    except OSError as exc:
        if final.exists():
            raise ConfigError(f"conflict: attempt {attempt_id} was published concurrently") from exc
        raise ConfigError(f"could not publish attempt {attempt_id}: {exc}") from exc
    return _result("published", final, files=report["files"], total_bytes=report["total_bytes"],
                   artifacts_sha256=artifacts_sha256)

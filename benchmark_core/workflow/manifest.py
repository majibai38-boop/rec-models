"""Minimal run manifest, metric and artifact contracts for one attempt.

The manifest records phases; metrics and artifacts are written next to it. Nothing here
computes a benchmark value: callers must supply measurements produced by the P1-04 scripts.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path

from benchmark_core.inventory.config import ConfigError
from benchmark_core.planning.contracts import digest, exact, text
from benchmark_core.workflow.state import PHASES, validate_id

MANIFEST_VERSION = "recbench.run-manifest/v1"
METRICS_VERSION = "recbench.metrics/v1"
ARTIFACTS_VERSION = "recbench.artifacts/v1"

PHASE_STATES = {"pending", "running", "succeeded", "failed", "skipped", "blocked", "interrupted"}
TERMINAL_STATES = {"succeeded", "failed", "skipped", "blocked", "interrupted"}
TRANSITIONS = {
    "pending": {"running", "skipped", "blocked"},
    "running": {"succeeded", "failed", "interrupted"},
    "blocked": {"running"},
    "succeeded": set(),
    "failed": set(),
    "skipped": set(),
    "interrupted": set(),
}
TRANSPORTS = {"local", "ssh"}
BACKENDS = {"cuda", "ascend"}
UNITS = {"samples_per_second", "milliseconds", "bytes", "ratio", "count", "boolean"}
SCOPES = {"end_to_end", "device_only", "environment", "correctness"}
AGGREGATIONS = {"single", "median", "min", "max"}
VALIDITY = {"valid", "invalid", "unsupported", "not_measured"}
SOURCES = {"measurement", "environment", "correctness"}
BENCHMARK_STATUS = {"valid", "invalid", "not_measured"}
RESERVED_FILES = {"artifacts.json", "COMPLETE"}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _timestamp(value, where, nullable=True):
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{where}: expected an ISO timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ConfigError(f"{where}: expected an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfigError(f"{where}: expected a timezone-aware timestamp")


def _number(value, where):
    if value is None or type(value) not in (int, float):
        raise ConfigError(f"{where}: expected a finite number")
    if not math.isfinite(value):
        raise ConfigError(f"{where}: expected a finite number")


def _object(value, keys, where):
    exact(value, keys, where)


def new_manifest(run_id, attempt_id, *, source, execution, archive, created_at=None):
    """Create a manifest whose phases are all pending; callers transition them explicitly."""
    validate_id(run_id, "run_id")
    validate_id(attempt_id, "attempt_id")
    now = created_at or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError("created_at: expected a timezone-aware timestamp")
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "source": source,
        "execution": execution,
        "archive": archive,
        "phases": {phase: {"state": "pending", "started_at": None, "finished_at": None,
                            "exit_code": None, "detail": None, "log": None} for phase in PHASES},
        "result": {"benchmark": "not_measured", "reason": None},
        "metrics_sha256": None,
        "artifacts_sha256": None,
    }
    validate_manifest(manifest)
    return manifest


def _validate_source(value):
    _object(value, "workflow_sha256 spec_sha256 resolved_spec_sha256 workflow_path entry_workspace hostname", "manifest.source")
    digest(value["workflow_sha256"], "manifest.source.workflow_sha256")
    digest(value["spec_sha256"], "manifest.source.spec_sha256")
    digest(value["resolved_spec_sha256"], "manifest.source.resolved_spec_sha256")
    for key in ("workflow_path", "entry_workspace", "hostname"):
        text(value[key], f"manifest.source.{key}")


def _validate_execution(value):
    _object(value, "transport backend node device_id", "manifest.execution")
    if value["transport"] not in {"local"}:
        raise ConfigError("manifest.execution.transport: phase 1 supports only local")
    if value["backend"] not in BACKENDS:
        raise ConfigError("manifest.execution.backend: expected cuda or ascend")
    text(value["device_id"], "manifest.execution.device_id")
    text(value["node"], "manifest.execution.node", True)


def _validate_archive(value):
    _object(value, "transport host root web_base_url", "manifest.archive")
    if value["transport"] not in TRANSPORTS:
        raise ConfigError("manifest.archive.transport: expected local or ssh")
    text(value["root"], "manifest.archive.root")
    if value["host"] is not None:
        text(value["host"], "manifest.archive.host")
    if value["web_base_url"] is not None:
        text(value["web_base_url"], "manifest.archive.web_base_url")


def _validate_phase(value, phase):
    where = f"manifest.phases.{phase}"
    _object(value, "state started_at finished_at exit_code detail log", where)
    if value["state"] not in PHASE_STATES:
        raise ConfigError(f"{where}.state: expected one of {sorted(PHASE_STATES)}")
    _timestamp(value["started_at"], f"{where}.started_at")
    _timestamp(value["finished_at"], f"{where}.finished_at")
    if value["exit_code"] is not None and type(value["exit_code"]) is not int:
        raise ConfigError(f"{where}.exit_code: expected an integer")
    for key in ("detail", "log"):
        if value[key] is not None:
            text(value[key], f"{where}.{key}")


def validate_manifest(value):
    _object(value, "schema_version run_id attempt_id created_at updated_at source execution archive "
                   "phases result metrics_sha256 artifacts_sha256", "manifest")
    if value["schema_version"] != MANIFEST_VERSION:
        raise ConfigError(f"manifest.schema_version: expected {MANIFEST_VERSION}")
    validate_id(value["run_id"], "manifest.run_id")
    validate_id(value["attempt_id"], "manifest.attempt_id")
    _timestamp(value["created_at"], "manifest.created_at", nullable=False)
    _timestamp(value["updated_at"], "manifest.updated_at", nullable=False)
    _validate_source(value["source"])
    _validate_execution(value["execution"])
    _validate_archive(value["archive"])
    phases = value["phases"]
    if not isinstance(phases, dict) or set(phases) != set(PHASES):
        raise ConfigError(f"manifest.phases: expected exactly {list(PHASES)}")
    for phase in PHASES:
        _validate_phase(phases[phase], phase)
    result = value["result"]
    _object(result, "benchmark reason", "manifest.result")
    if result["benchmark"] not in BENCHMARK_STATUS:
        raise ConfigError(f"manifest.result.benchmark: expected one of {sorted(BENCHMARK_STATUS)}")
    if result["reason"] is not None:
        text(result["reason"], "manifest.result.reason")
    elif result["benchmark"] == "invalid":
        raise ConfigError("manifest.result: an invalid benchmark requires a reason")
    digest(value["metrics_sha256"], "manifest.metrics_sha256", True)
    digest(value["artifacts_sha256"], "manifest.artifacts_sha256", True)
    return value


def transition(manifest, phase, state, *, at=None, exit_code=None, detail=None, log=None):
    """Move one phase and require evidence for success and a reason for failure."""
    if phase not in PHASES:
        raise ConfigError(f"phase: expected one of {list(PHASES)}")
    if state not in PHASE_STATES:
        raise ConfigError(f"phase {phase}: unknown state {state!r}")
    current = manifest["phases"][phase]
    if state not in TRANSITIONS[current["state"]]:
        raise ConfigError(f"phase {phase}: cannot move from {current['state']} to {state}")
    if state == "succeeded" and not (log or detail):
        raise ConfigError(f"phase {phase}: succeeded requires a log path or detail evidence")
    if state in {"failed", "interrupted", "blocked"} and not detail:
        raise ConfigError(f"phase {phase}: {state} requires a reason in detail")
    now = at or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError("transition time: expected a timezone-aware timestamp")
    if state == "running" and current["started_at"] is None:
        current["started_at"] = now.isoformat()
    if state in TERMINAL_STATES:
        current["finished_at"] = now.isoformat()
    current["state"] = state
    if exit_code is not None:
        if type(exit_code) is not int:
            raise ConfigError("exit_code: expected an integer")
        current["exit_code"] = exit_code
    if detail is not None:
        current["detail"] = detail
    if log is not None:
        current["log"] = log
    manifest["updated_at"] = now.isoformat()
    validate_manifest(manifest)
    return manifest


def set_result(manifest, benchmark, *, reason=None, at=None):
    if benchmark not in BENCHMARK_STATUS:
        raise ConfigError(f"result.benchmark: expected one of {sorted(BENCHMARK_STATUS)}")
    if benchmark == "invalid" and not reason:
        raise ConfigError("result: an invalid benchmark requires a reason")
    manifest["result"] = {"benchmark": benchmark, "reason": reason}
    now = at or dt.datetime.now(dt.timezone.utc)
    manifest["updated_at"] = now.isoformat()
    validate_manifest(manifest)
    return manifest


def validate_metrics(value):
    _object(value, "schema_version metrics", "metrics")
    if value["schema_version"] != METRICS_VERSION:
        raise ConfigError(f"metrics.schema_version: expected {METRICS_VERSION}")
    metrics = value["metrics"]
    if not isinstance(metrics, list) or not 1 <= len(metrics) <= 256:
        raise ConfigError("metrics.metrics: expected 1..256 entries")
    names = set()
    for metric in metrics:
        where = "metric"
        _object(metric, "name unit scope aggregation value sample_count validity reason source", where)
        text(metric["name"], "metric.name")
        if metric["name"] in names:
            raise ConfigError(f"Duplicate metric name: {metric['name']}")
        names.add(metric["name"])
        if metric["unit"] not in UNITS:
            raise ConfigError(f"metric.unit: expected one of {sorted(UNITS)}")
        if metric["scope"] not in SCOPES:
            raise ConfigError(f"metric.scope: expected one of {sorted(SCOPES)}")
        if metric["aggregation"] not in AGGREGATIONS:
            raise ConfigError(f"metric.aggregation: expected one of {sorted(AGGREGATIONS)}")
        if metric["validity"] not in VALIDITY:
            raise ConfigError(f"metric.validity: expected one of {sorted(VALIDITY)}")
        if metric["source"] not in SOURCES:
            raise ConfigError(f"metric.source: expected one of {sorted(SOURCES)}")
        if metric["value"] is not None:
            _number(metric["value"], "metric.value")
        if metric["sample_count"] is not None and (type(metric["sample_count"]) is not int
                                                    or metric["sample_count"] < 1):
            raise ConfigError("metric.sample_count: expected a positive integer or null")
        if metric["reason"] is not None:
            text(metric["reason"], "metric.reason")
        if metric["validity"] == "valid":
            if metric["value"] is None:
                raise ConfigError(f"metric {metric['name']}: a valid metric requires a value")
            if metric["sample_count"] is None and metric["aggregation"] != "single":
                raise ConfigError(f"metric {metric['name']}: aggregated metrics require sample_count")
        else:
            if not metric["reason"]:
                raise ConfigError(f"metric {metric['name']}: {metric['validity']} requires a reason")
            if metric["value"] is not None and metric["validity"] in {"unsupported", "not_measured"}:
                raise ConfigError(f"metric {metric['name']}: {metric['validity']} must not carry a value")
    return value


def _artifact_path(value):
    text(value, "artifact.path")
    if value.startswith(("/", "\\")) or "\\" in value or "\x00" in value or re.match(r"^[A-Za-z]:", value):
        raise ConfigError(f"artifact.path: use a relative POSIX path, got {value!r}")
    segments = value.split("/")
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ConfigError(f"artifact.path: empty, '.' or '..' segments are not allowed: {value!r}")
    if segments[0] in RESERVED_FILES:
        raise ConfigError(f"artifact.path: {value!r} is generated by the archive step and must not be listed")
    return value


def validate_artifacts(value):
    _object(value, "schema_version files total_bytes", "artifacts")
    if value["schema_version"] != ARTIFACTS_VERSION:
        raise ConfigError(f"artifacts.schema_version: expected {ARTIFACTS_VERSION}")
    files = value["files"]
    if not isinstance(files, list) or len(files) > 10000:
        raise ConfigError("artifacts.files: expected at most 10000 entries")
    seen, total = set(), 0
    for entry in files:
        _object(entry, "path size_bytes sha256 media_type", "artifact")
        _artifact_path(entry["path"])
        if entry["path"] in seen:
            raise ConfigError(f"Duplicate artifact path: {entry['path']}")
        seen.add(entry["path"])
        if type(entry["size_bytes"]) is not int or entry["size_bytes"] < 0:
            raise ConfigError("artifact.size_bytes: expected a nonnegative integer")
        if not isinstance(entry["sha256"], str) or not _SHA256.fullmatch(entry["sha256"]):
            raise ConfigError("artifact.sha256: expected lowercase SHA256")
        text(entry["media_type"], "artifact.media_type")
        total += entry["size_bytes"]
    if type(value["total_bytes"]) is not int or value["total_bytes"] != total:
        raise ConfigError("artifacts.total_bytes: expected the sum of file sizes")
    return value


def sha256_file(path):
    digest_value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_value.update(chunk)
    return digest_value.hexdigest()


def verify_artifacts(value, root):
    """Recompute size and hash for every listed file; never trust the manifest alone."""
    validate_artifacts(value)
    root = Path(root).resolve()
    problems = []
    for entry in value["files"]:
        try:
            resolved = (root / entry["path"]).resolve()
            resolved.relative_to(root)
        except ValueError:
            problems.append({"path": entry["path"], "reason": "path escapes the archive root"})
            continue
        if not resolved.is_file():
            problems.append({"path": entry["path"], "reason": "file missing"})
            continue
        if resolved.stat().st_size != entry["size_bytes"]:
            problems.append({"path": entry["path"], "reason": "size mismatch"})
        elif sha256_file(resolved) != entry["sha256"]:
            problems.append({"path": entry["path"], "reason": "sha256 mismatch"})
    return {"ok": not problems, "problems": problems}


def artifacts_index(root):
    """Build an artifacts.json body from files currently present under one attempt directory."""
    root = Path(root).resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in RESERVED_FILES:
            continue
        files.append({"path": relative, "size_bytes": path.stat().st_size,
                      "sha256": sha256_file(path), "media_type": _media_type(path)})
    value = {"schema_version": ARTIFACTS_VERSION, "files": files,
             "total_bytes": sum(entry["size_bytes"] for entry in files)}
    return validate_artifacts(value)


def _media_type(path):
    suffix = path.suffix.lower()
    return {".json": "application/json", ".jsonl": "application/x-ndjson", ".md": "text/markdown",
            ".txt": "text/plain", ".log": "text/plain", ".csv": "text/csv", ".gz": "application/gzip",
            ".pt": "application/octet-stream", ".trace": "application/octet-stream"}.get(suffix, "application/octet-stream")


def write_json(path, value):
    """Write JSON atomically inside the destination directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def write_manifest(path, manifest):
    validate_manifest(manifest)
    return write_json(path, manifest)

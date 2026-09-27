"""Small reproducibility manifest; raw traces are never copied into results."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import time
import uuid

from .result_api import file_identity
from .version import __version__, RULES_VERSION, RESULT_SCHEMA_VERSION


def producer():
    digest = hashlib.sha256()
    for entry in sorted(files("profiling_tools").iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".py"):
            digest.update(entry.name.encode("utf-8"))
            digest.update(entry.read_bytes())
    return {"name": "gpu-npu-profiling", "version": __version__, "rules_version": RULES_VERSION,
            "source_sha256": digest.hexdigest()}


def begin_run(config):
    paths = {key: getattr(config, key) for key in ("gpu_trace", "npu_trace", "npu_kernel_details", "op_map")
             if getattr(config, key)}
    return {"schema_version": RESULT_SCHEMA_VERSION, "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat(), "producer": producer(),
            "parameters": asdict(config), "inputs": {key: file_identity(path) for key, path in paths.items()}}


def assert_inputs_unchanged(manifest):
    for identity in manifest["inputs"].values():
        stat = Path(identity["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (identity["size_bytes"], identity["mtime_ns"]):
            raise ValueError(f"input changed during analysis: {identity['path']}")


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # Windows indexers/virus scanners may briefly hold a newly written file.
        # Keep the original intact and retry only the atomic replacement; never
        # delete the destination or fall back to a truncating write.
        for attempt in range(6):
            try:
                temp.replace(path)
                break
            except PermissionError:
                if os.name != "nt" or attempt == 5:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        if temp.exists():
            temp.unlink()

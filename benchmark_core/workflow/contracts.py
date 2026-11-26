"""recbench.workflow/v1: bind one BenchmarkSpec target to local execution and archiving.

The workflow document is deliberately separate from the strict BenchmarkSpec so archive,
profiling and destination settings never leak into the measurement contract. Resolution is
offline: it validates identity, derives a single-target spec and reports staged blockers.
"""
from __future__ import annotations

import copy
import datetime as dt
import re
from pathlib import Path

from benchmark_core.inventory.config import (
    ConfigError, fingerprint, load_config, load_json, resolve_node, text,
)
from benchmark_core.planning.contracts import exact, validate_recipe, validate_spec
from benchmark_core.planning.planner import plan
from benchmark_core.workflow.manifest import write_json

WORKFLOW_VERSION = "recbench.workflow/v1"
REQUEST_VERSION = "recbench.workflow-request/v1"
RESOLVED_VERSION = "recbench.workflow-resolved/v1"
ADMISSION_VERSION = "recbench.workflow-admission/v1"

DOCUMENTS = ("benchmark_spec", "inventory_config", "environment_recipe")
DEFAULT_ARCHIVE = Path("benchmark-results") / "archive"

_SSH_ALIAS = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,252}$")
_REMOTE_ROOT = re.compile(r"^/[A-Za-z0-9._/-]+$")
_WEB_URL = re.compile(r"^(https?)://([A-Za-z0-9][A-Za-z0-9.-]{0,252})(:[0-9]{1,5})?(/[A-Za-z0-9._~/-]*)?$")
_HASH_FIELDS = (
    ("workload", "source_sha256"),
    ("data", "artifact_sha256"),
    ("initial_state", "checkpoint_sha256"),
    ("initial_state", "training_state_sha256"),
    ("correctness", "policy_sha256"),
)


def validate_workflow(value):
    """Validate only the workflow document; no file is read and nothing is executed."""
    exact(value, "schema_version benchmark_spec inventory_config environment_recipe target_id "
                 "execution archive profile", "workflow")
    if value["schema_version"] != WORKFLOW_VERSION:
        raise ConfigError(f"workflow.schema_version: expected {WORKFLOW_VERSION}")
    for key in DOCUMENTS:
        text(value[key], f"workflow.{key}")
    text(value["target_id"], "workflow.target_id", True)
    execution = value["execution"]
    exact(execution, "transport", "workflow.execution")
    if execution["transport"] != "local":
        raise ConfigError("workflow.execution.transport: phase 1 runs only on the machine that starts the Skill/CLI")
    archive = value["archive"]
    exact(archive, "transport host root web_base_url", "workflow.archive")
    transport = archive["transport"]
    if transport not in {"local", "ssh"}:
        raise ConfigError("workflow.archive.transport: expected local or ssh")
    if transport == "local":
        if archive["host"] not in (None, "localhost"):
            raise ConfigError("workflow.archive.host: transport=local archives on the entry host; use null or localhost")
        if archive["root"] is not None:
            text(archive["root"], "workflow.archive.root")
    else:
        if not isinstance(archive["host"], str) or not _SSH_ALIAS.fullmatch(archive["host"]):
            raise ConfigError("workflow.archive.host: transport=ssh requires an existing SSH config alias")
        _remote_root(archive["root"])
    if archive["web_base_url"] is not None:
        text(archive["web_base_url"], "workflow.archive.web_base_url")
        if not _WEB_URL.fullmatch(archive["web_base_url"]):
            raise ConfigError("workflow.archive.web_base_url: expected an http(s) URL without credentials, query or fragment")
    profile = value["profile"]
    exact(profile, "enabled", "workflow.profile")
    if not isinstance(profile["enabled"], bool):
        raise ConfigError("workflow.profile.enabled: expected boolean")
    return value


def _remote_root(value):
    if not isinstance(value, str) or not _REMOTE_ROOT.fullmatch(value) or value.endswith("/"):
        raise ConfigError("workflow.archive.root: transport=ssh requires an absolute POSIX path on the archive host")
    segments = value.split("/")[1:]
    if any(segment in {"", ".", ".."} for segment in segments):
        raise ConfigError("workflow.archive.root: empty, '.' or '..' segments are not allowed")
    return value


def _document(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _target(spec, target_id):
    for target in spec["targets"]:
        if target["id"] == target_id:
            return target
    available = sorted(target["id"] for target in spec["targets"])
    raise ConfigError(f"workflow.target_id: {target_id!r} is not declared in the benchmark spec; available: {available}")


def _archive(archive, base, workspace):
    transport = archive["transport"]
    root = archive["root"]
    if transport == "local":
        if root is None:
            local_root = workspace / DEFAULT_ARCHIVE
        else:
            candidate = Path(root)
            local_root = candidate if candidate.is_absolute() else base / candidate
            local_root = local_root.resolve()
        return {"transport": "local", "host": "localhost", "root": str(local_root),
                "remote_root": None, "staging_root": None, "web_base_url": archive["web_base_url"]}
    return {"transport": "ssh", "host": archive["host"], "root": _remote_root(root),
            "remote_root": root, "staging_root": str(workspace / "benchmark-results" / "staging"),
            "web_base_url": archive["web_base_url"]}


def _admission(effective_spec, target_plan):
    blockers = target_plan["targets"][0]["blockers"]
    prepare = [dict(blocker) for blocker in blockers if blocker["code"] != "unresolved_workload"]
    workload = [{"code": "missing_hash", "message": f"{group}.{key} is unresolved; freeze it during the short model run"}
                for group, key in _HASH_FIELDS if effective_spec[group][key] is None]
    pilot = [] if effective_spec["measurement"]["measure_steps"] is not None else [{
        "code": "missing_measure_steps",
        "message": "pilot must select measurement.measure_steps before formal measurement"}]
    measurement = [{"code": "prerequisites_blocked",
                    "message": "environment, workload state and pilot must pass before measurement"}] \
        if prepare or workload or pilot else []
    stages = [
        {"id": "structure", "status": "eligible", "blockers": []},
        {"id": "environment", "status": "blocked" if prepare else "eligible", "blockers": prepare},
        {"id": "workload_state", "status": "blocked" if workload else "eligible", "blockers": workload},
        {"id": "pilot", "status": "blocked" if pilot else "eligible", "blockers": pilot},
        {"id": "measurement", "status": "blocked" if measurement else "eligible", "blockers": measurement},
    ]
    blocked = any(stage["status"] == "blocked" for stage in stages)
    return {"schema_version": ADMISSION_VERSION, "status": "blocked" if blocked else "eligible_for_measurement",
            "execution_ready": False, "stages": stages}


def resolve_workflow(workflow, *, workflow_path, entry_workspace=None, snapshot_paths=(), max_age_hours=24,
                     now=None):
    """Validate identity, derive the single-target spec and stage admission blockers."""
    validate_workflow(workflow)
    workflow_path = Path(workflow_path).resolve()
    base = workflow_path.parent
    workspace = Path(entry_workspace).resolve() if entry_workspace else Path.cwd().resolve()
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError("resolution time requires a timezone")
    documents = {key: _document(base, workflow[key]) for key in DOCUMENTS}
    spec = load_json(documents["benchmark_spec"])
    validate_spec(spec)
    recipe = load_json(documents["environment_recipe"])
    validate_recipe(recipe)
    config, profiles = load_config(documents["inventory_config"])
    target = _target(spec, workflow["target_id"])
    if recipe["id"] != target["environment"]:
        raise ConfigError(f"workflow.environment_recipe: {recipe['id']!r} does not match target "
                          f"{target['id']!r} environment {target['environment']!r}")
    node, selected = resolve_node(config, profiles, target["node"])
    if target["inventory_profile"] not in {profile["id"] for profile in selected}:
        raise ConfigError(f"target {target['id']}: inventory_profile {target['inventory_profile']!r} "
                          f"is not selected for node {node['id']}")
    effective_spec = copy.deepcopy(spec)
    effective_spec["targets"] = [copy.deepcopy(target)]
    validate_spec(effective_spec)
    archive = _archive(workflow["archive"], base, workspace)
    target_plan = plan(effective_spec, [recipe], config, profiles, list(snapshot_paths),
                       max_age_hours=max_age_hours, now=now)
    created_at = now.isoformat()
    request = {
        "schema_version": REQUEST_VERSION,
        "created_at": created_at,
        "workflow_path": str(workflow_path),
        "workflow_sha256": fingerprint(workflow),
        "base_dir": str(base),
        "entry_workspace": str(workspace),
        "workflow": copy.deepcopy(workflow),
    }
    resolved = {
        "schema_version": RESOLVED_VERSION,
        "created_at": created_at,
        "workflow_path": str(workflow_path),
        "workflow_sha256": fingerprint(workflow),
        "base_dir": str(base),
        "entry_workspace": str(workspace),
        "documents": {key: str(path) for key, path in documents.items()},
        "document_sha256": {"benchmark_spec": fingerprint(spec), "environment_recipe": fingerprint(recipe),
                            "inventory_config": fingerprint(config)},
        "spec_sha256": fingerprint(spec),
        "resolved_spec_sha256": fingerprint(effective_spec),
        "target": copy.deepcopy(target),
        "node": {"id": node["id"], "expected_hostname": node["expected_hostname"],
                 "expected_os": node["expected_os"]},
        "recipe": {"id": recipe["id"], "backend": recipe["backend"], "state": recipe["state"]},
        "archive": archive,
        "profile": copy.deepcopy(workflow["profile"]),
        "run_count": len(target_plan["runs"]),
    }
    return {"request": request, "resolved": resolved, "spec": effective_spec, "plan": target_plan,
            "admission": _admission(effective_spec, target_plan)}


def write_resolution(result, output):
    """Persist the request, resolved identity, single-target spec, plan and admission separately."""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "request.json", result["request"])
    write_json(directory / "resolved_workflow.json", result["resolved"])
    write_json(directory / "resolved_spec.json", result["spec"])
    write_json(directory / "plan.json", result["plan"])
    write_json(directory / "admission.json", result["admission"])
    return directory

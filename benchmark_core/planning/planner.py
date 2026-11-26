"""Resolve declarations against captured evidence, without touching target machines."""
from __future__ import annotations

import copy
import datetime as dt
import json
from pathlib import Path

from benchmark_core.inventory.config import ConfigError, fingerprint, integer, resolve_node
from benchmark_core.inventory.report import cell, read_snapshot
from .contracts import numeric_version, validate_recipe, validate_spec, workload_conditions, workload_fingerprint

PLAN_VERSION = "recbench.execution-plan/v1"


def evaluate_rule(rule, host, device):
    scope, key = rule["field"].split(".")
    actual = (host if scope == "host" else device).get(key)
    result = {**copy.deepcopy(rule), "observed": actual, "status": "unknown"}
    if actual is None:
        return result
    op, expected = rule["op"], rule["value"]
    if op == "version_gte":
        try:
            matched = numeric_version(actual) >= numeric_version(expected)
        except ValueError:
            return result
    elif op == "gte":
        if type(actual) is not int or actual < 0:
            return result
        matched = actual >= expected
    else:
        if type(actual) is not type(expected if op == "eq" else expected[0]):
            return result
        matched = actual == expected if op == "eq" else actual in expected
    result["status"] = "match" if matched else "mismatch"
    return result


def plan(spec, recipes, config, profiles, snapshot_paths=(), *, max_age_hours=24, now=None):
    validate_spec(spec)
    integer(max_age_hours, 1, 8760, "max_age_hours")
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ConfigError("Planning time requires a timezone")
    recipe_map = {}
    for recipe in recipes:
        validate_recipe(recipe)
        if recipe["id"] in recipe_map:
            raise ConfigError(f"Duplicate recipe: {recipe['id']}")
        recipe_map[recipe["id"]] = recipe
    # read_snapshot rejects inconsistent counts, malformed devices and unknown schema versions.
    snapshots = {}
    configured_ids = {node["id"] for node in config["nodes"]}
    for path in snapshot_paths:
        snapshot = read_snapshot(path)
        if snapshot["node_id"] in snapshots or snapshot["node_id"] not in configured_ids:
            raise ConfigError("Duplicate snapshot or snapshot node absent from inventory config")
        software = snapshot.get("software", {})
        if not isinstance(software, dict) or not isinstance(software.get("version_reports", {}), dict):
            raise ConfigError("Invalid inventory software/version_reports object")
        snapshots[snapshot["node_id"]] = snapshot
    common_issues = []
    for group, key in (("workload", "source_sha256"), ("data", "artifact_sha256"),
                       ("initial_state", "checkpoint_sha256"), ("initial_state", "training_state_sha256"),
                       ("correctness", "policy_sha256"), ("measurement", "measure_steps")):
        if spec[group][key] is None:
            common_issues.append({"code": "unresolved_workload", "message": f"Resolve {group}.{key} before admission"})
    resolved_targets = []
    for target in spec["targets"]:
        node, selected = resolve_node(config, profiles, target["node"])
        recipe = recipe_map.get(target["environment"])
        if recipe is None:
            raise ConfigError(f"Unknown environment recipe: {target['environment']}")
        if recipe["backend"] != target["backend"]:
            raise ConfigError(f"Target {target['id']} backend differs from its recipe")
        profile = next((p for p in selected if p["id"] == target["inventory_profile"]), None)
        expected_kind = "gpu" if target["backend"] == "cuda" else "npu"
        expected_vendor = "nvidia" if target["backend"] == "cuda" else "ascend"
        if profile is None or profile["kind"] != expected_kind or profile.get("vendor", "").lower() != expected_vendor:
            raise ConfigError(f"Target {target['id']}: inventory profile does not match backend/vendor")
        issues = copy.deepcopy(common_issues)

        def issue(code, message):
            issues.append({"code": code, "message": message})

        if recipe["state"] != "pinned":
            issue("draft_environment", "Pin image, runtime versions and reviewed compatibility rules")
        snapshot = snapshots.get(node["id"])
        device, rule_results = None, []
        if snapshot is None:
            issue("missing_inventory", f"No snapshot supplied for {node['id']}")
        else:
            if snapshot["status"] == "partial":
                issue("inventory_partial", "Required inventory collection failed")
            if (snapshot.get("config_sha256") != fingerprint({"node": node, "execution": config["execution"]})
                    or snapshot.get("profiles_sha256") != fingerprint(selected)
                    or snapshot["host"]["hostname"].casefold() != node["expected_hostname"].casefold()
                    or snapshot["host"]["os"] != node["expected_os"]):
                issue("inventory_stale", "Node identity, configuration or profiles changed; recollect")
            try:
                captured = dt.datetime.fromisoformat(snapshot.get("completed_at", ""))
                if captured.tzinfo is None or captured.utcoffset() is None:
                    raise ValueError("missing timezone")
                age = (now - captured).total_seconds()
                if age < 0 or age > max_age_hours * 3600:
                    issue("inventory_expired", "Snapshot is in the future or exceeds max_age_hours; recollect")
            except (ValueError, TypeError):
                issue("inventory_time_unknown", "Snapshot has no valid timezone-aware completion timestamp")
            device = next((d for d in snapshot["accelerators"] if
                           d["backend"] == target["inventory_profile"] and d["id"] == target["device_id"]), None)
            if device is None:
                issue("device_not_observed", "Requested device was not discovered; no implicit device substitution")
            else:
                vendor = device.get("vendor")
                if not isinstance(vendor, str) or vendor.lower() != expected_vendor or device.get("kind") != expected_kind:
                    issue("device_identity_mismatch", "Observed device vendor/kind differs from selected backend")
                rule_results = [evaluate_rule(r, snapshot["host"], device) for r in recipe["requirements"]]
                for result in rule_results:
                    if result["status"] != "match":
                        issue("requirement_" + result["status"], f"{result['field']} {result['op']} {result['value']!r}; observed={result['observed']!r}")
        # Recompute from included facts, never trust a supplied observed hash. This identifies
        # intended environment + selected hardware, not an observed container runtime.
        observed = None if snapshot is None else {
            "host": {k: v for k, v in snapshot["host"].items() if k != "network_interfaces"},
            "device": None if device is None else {k: v for k, v in device.items() if k not in {"sources", "health"}},
            "version_reports": snapshot.get("software", {}).get("version_reports", {}),
            "runtime_context": snapshot.get("runtime_context"),
        }
        resolved_targets.append({"target": copy.deepcopy(target), "recipe": copy.deepcopy(recipe),
            "admission": "blocked" if issues else "eligible_for_runtime_check", "blockers": issues,
            "requirements": rule_results, "snapshot_sha256": fingerprint(snapshot) if snapshot else None,
            "observed": observed, "planned_environment_sha256": fingerprint({"recipe": recipe, "observed": observed}),
            "runtime_validation": "not_run", "execution_ready": False})
    runs = []
    for batch in spec["matrix"]["batch_size"]:
        for resolved in resolved_targets:
            identity = {"workload": workload_fingerprint(spec, batch), "target": resolved["target"],
                        "environment": resolved["planned_environment_sha256"]}
            runs.append({"plan_key": fingerprint(identity), "target_id": resolved["target"]["id"],
                         "batch_size": batch, "precision": "fp32", "conditions": copy.deepcopy(workload_conditions(spec, batch)),
                         "workload_sha256": identity["workload"], "planned_environment_sha256": identity["environment"],
                         "admission": resolved["admission"]})
    return {"schema_version": PLAN_VERSION, "created_at": now.isoformat(), "spec": copy.deepcopy(spec),
            "spec_sha256": fingerprint(spec), "max_inventory_age_hours": max_age_hours,
            "status": "blocked" if any(t["blockers"] for t in resolved_targets) else "eligible_for_runtime_check",
            "execution_ready": False, "targets": resolved_targets, "runs": runs,
            "remaining_checks": ["model adapter and parameter validation", "artifact content verification",
                "container image/runtime verification", "device allocation and visibility mapping",
                "backend smoke test and correctness", "pilot and measurement protocol validation"]}


def markdown(value):
    lines = ["# Benchmark execution plan", "", f"Status: **{value['status']}**", "",
             "Offline planning only. No container started, no runtime validated, no benchmark executed.", "",
             "| Target | Backend | Node / device | Admission |", "|---|---|---|---|"]
    for resolved in value["targets"]:
        t = resolved["target"]
        lines.append("| " + " | ".join(cell(v) for v in (t["id"], t["backend"], f"{t['node']} / {t['device_id']}", resolved["admission"])) + " |")
    for resolved in value["targets"]:
        lines += ["", f"## {cell(resolved['target']['id'])}", ""]
        lines += [f"- {cell(i['code'])}: {cell(i['message'])}" for i in resolved["blockers"]] or ["No offline blockers; runtime checks remain mandatory."]
    lines += ["", f"Expanded plans: {len(value['runs'])}. Details, resolved recipes, conditions and hashes: plan.json.", "",
              "## Remaining checks", ""] + [f"- {s}" for s in value["remaining_checks"]]
    return "\n".join(lines) + "\n"


def write_plan(value, output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.md").write_text(markdown(value), encoding="utf-8")
    temporary = directory / "plan.json.tmp"
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "plan.json")
    return directory / "plan.json"

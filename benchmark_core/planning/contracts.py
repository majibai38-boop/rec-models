"""Versioned JSON contracts shared by future CLI, Web and Skill callers."""
from __future__ import annotations

import datetime as dt
import math
import re
from urllib.parse import urlsplit

from benchmark_core.inventory.config import ConfigError, fields, fingerprint, integer, strings, text

SPEC_VERSION = "recbench.benchmark-spec/v1"
RECIPE_VERSION = "recbench.environment-recipe/v1"
SHA256 = re.compile(r"^[a-f0-9]{64}$")
PIN = re.compile(r"^[0-9]+\.[0-9]+(?:[a-zA-Z0-9.+_-]*)$")
RULE_FIELDS = {
    "host.os", "host.architecture", "host.os_distribution", "host.memory_total_bytes",
    "device.vendor", "device.kind", "device.model", "device.driver_version",
    "device.firmware_version", "device.memory_capacity_bytes", "device.health",
}
NUMBER_FIELDS = {"host.memory_total_bytes", "device.memory_capacity_bytes"}
VERSION_FIELDS = {"device.driver_version", "device.firmware_version"}


def exact(obj, keys, where):
    fields(obj, keys.split(), keys.split(), where)


def choice(value, choices, where):
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{where}: expected one of {sorted(choices)}")


def digest(value, where, nullable=False):
    if value is None and nullable:
        return
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ConfigError(f"{where}: expected lowercase SHA256")


def json_value(value, where, depth=0):
    if depth > 16:
        raise ConfigError(f"{where}: JSON nesting exceeds 16")
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            json_value(item, where, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        for item in value.values():
            json_value(item, where, depth + 1)
        return
    raise ConfigError(f"{where}: expected finite JSON data")


def numeric_version(value):
    # Deliberately no guessed ordering for RC, vendor or release suffixes.
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,7}", value):
        raise ValueError("expected a dotted numeric version")
    parts = tuple(int(p) for p in value.split("."))
    return parts + (0,) * (8 - len(parts))


def validate_recipe(value):
    exact(value, "schema_version id backend state image runtime requirements compatibility", "recipe")
    choice(value["schema_version"], {RECIPE_VERSION}, "recipe.schema_version")
    text(value["id"], "recipe.id", True)
    choice(value["backend"], {"cuda", "ascend"}, "recipe.backend")
    choice(value["state"], {"draft", "pinned"}, "recipe.state")
    image = value["image"]
    if image is not None and (not isinstance(image, str) or not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._/:-]*@sha256:[a-f0-9]{64}", image)):
        raise ConfigError("recipe.image: use an immutable repository@sha256:digest")
    runtime = value["runtime"]
    exact(runtime, "python torch toolkit torch_npu", "recipe.runtime")
    for key, version in runtime.items():
        if version is not None and (not isinstance(version, str) or not PIN.fullmatch(version)):
            raise ConfigError(f"recipe.runtime.{key}: expected an exact version, not a range/tag")
    if value["backend"] == "cuda" and runtime["torch_npu"] is not None:
        raise ConfigError("CUDA recipe cannot declare torch_npu")
    requirements = value["requirements"]
    if not isinstance(requirements, list) or len(requirements) > 64:
        raise ConfigError("recipe.requirements: expected at most 64 rules")
    seen = set()
    for rule in requirements:
        exact(rule, "field op value", "requirement")
        choice(rule["field"], RULE_FIELDS, "requirement.field")
        choice(rule["op"], {"eq", "in", "gte", "version_gte"}, "requirement.op")
        key = (rule["field"], rule["op"])
        if key in seen:
            raise ConfigError(f"Duplicate requirement: {key}")
        seen.add(key)
        expected = rule["value"]
        if rule["op"] == "version_gte":
            if rule["field"] not in VERSION_FIELDS:
                raise ConfigError("version_gte only supports driver/firmware fields")
            try:
                numeric_version(expected)
            except ValueError as exc:
                raise ConfigError(str(exc)) from exc
        elif rule["op"] == "gte":
            if rule["field"] not in NUMBER_FIELDS:
                raise ConfigError("gte only supports byte capacity fields")
            integer(expected, 1, 2**63 - 1, "requirement.value")
        elif rule["op"] == "in":
            strings(expected, "requirement.value")
            if rule["field"] in NUMBER_FIELDS or len(set(expected)) != len(expected):
                raise ConfigError("in requires unique text values on a text field")
        elif rule["field"] in NUMBER_FIELDS:
            integer(expected, 1, 2**63 - 1, "requirement.value")
        else:
            text(expected, "requirement.value")
    compatibility = value["compatibility"]
    exact(compatibility, "sources reviewed_on rationale", "recipe.compatibility")
    strings(compatibility["sources"], "compatibility.sources", allow_empty=True)
    for url in compatibility["sources"]:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ConfigError("Compatibility sources must be HTTPS URLs without credentials")
    text(compatibility["rationale"], "compatibility.rationale")
    if compatibility["reviewed_on"] is not None:
        try:
            dt.date.fromisoformat(compatibility["reviewed_on"])
        except (TypeError, ValueError) as exc:
            raise ConfigError("compatibility.reviewed_on: expected ISO date") from exc
    if value["state"] == "pinned":
        required_runtime = ("python", "torch", "toolkit") + (("torch_npu",) if value["backend"] == "ascend" else ())
        if image is None or any(runtime[k] is None for k in required_runtime):
            raise ConfigError("Pinned recipes require image digest and exact runtime versions")
        required_fields = {"host.os", "host.architecture", "device.model", "device.driver_version"}
        if value["backend"] == "ascend":
            required_fields.add("device.firmware_version")
        if not required_fields <= {r["field"] for r in requirements}:
            raise ConfigError(f"Pinned recipe requires rules for {sorted(required_fields)}")
        if not compatibility["sources"] or compatibility["reviewed_on"] is None:
            raise ConfigError("Pinned recipe requires reviewed compatibility evidence")
    return value


def validate_spec(value):
    exact(value, "schema_version name workload data initial_state targets matrix measurement correctness", "spec")
    choice(value["schema_version"], {SPEC_VERSION}, "spec.schema_version")
    text(value["name"], "spec.name", True)
    workload = value["workload"]
    exact(workload, "id source_sha256 implementation_track mode execution parameters", "workload")
    text(workload["id"], "workload.id", True)
    digest(workload["source_sha256"], "workload.source_sha256", True)
    choice(workload["implementation_track"], {"common-portable"}, "workload.implementation_track")
    choice(workload["mode"], {"train_step"}, "workload.mode")
    choice(workload["execution"], {"eager"}, "workload.execution")
    if not isinstance(workload["parameters"], dict):
        raise ConfigError("workload.parameters must be an object (adapter validation is deferred)")
    json_value(workload["parameters"], "workload.parameters")
    data = value["data"]
    exact(data, "id artifact_sha256 kind", "data")
    text(data["id"], "data.id", True)
    digest(data["artifact_sha256"], "data.artifact_sha256", True)
    choice(data["kind"], {"synthetic", "real"}, "data.kind")
    state = value["initial_state"]
    exact(state, "checkpoint_sha256 training_state_sha256", "initial_state")
    for key, content in state.items():
        digest(content, f"initial_state.{key}", True)
    targets = value["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= 16:
        raise ConfigError("spec.targets requires 1..16 targets")
    names = set()
    for target in targets:
        exact(target, "id backend environment node inventory_profile device_id devices", "target")
        for key in ("id", "environment", "node", "inventory_profile"):
            text(target[key], f"target.{key}", True)
        text(target["device_id"], "target.device_id")
        choice(target["backend"], {"cuda", "ascend"}, "target.backend")
        integer(target["devices"], 1, 1, "target.devices")
        if target["id"] in names:
            raise ConfigError("Duplicate target id")
        names.add(target["id"])
    matrix = value["matrix"]
    exact(matrix, "batch_size precision", "matrix")
    batches = matrix["batch_size"]
    if not isinstance(batches, list) or not 1 <= len(batches) <= 32:
        raise ConfigError("matrix.batch_size requires 1..32 entries")
    for batch in batches:
        integer(batch, 1, 2**31 - 1, "batch_size")
    if len(set(batches)) != len(batches) or len(batches) * len(targets) > 128:
        raise ConfigError("Duplicate batch size or matrix exceeds 128 runs")
    if matrix["precision"] != ["fp32"]:
        raise ConfigError("v1 supports precision: [fp32] only")
    measurement = value["measurement"]
    exact(measurement, "scope protocol warmup_steps measure_steps independent_repeats restore_state_before_each_repeat profiler_enabled sync_policy step_components", "measurement")
    choice(measurement["scope"], {"end_to_end", "device_only"}, "measurement.scope")
    choice(measurement["protocol"], {"fixed_steps"}, "measurement.protocol")
    integer(measurement["warmup_steps"], 1, 1000000, "warmup_steps")
    if measurement["measure_steps"] is not None:
        integer(measurement["measure_steps"], 1, 10000000, "measure_steps")
    integer(measurement["independent_repeats"], 2, 1000, "independent_repeats")
    if measurement["restore_state_before_each_repeat"] is not True or measurement["profiler_enabled"] is not False:
        raise ConfigError("Restore all training state per repeat; profiling must be a separate execution")
    choice(measurement["sync_policy"], {"before_and_after_window"}, "measurement.sync_policy")
    if measurement["step_components"] != ["zero_grad", "forward", "loss", "backward", "optimizer_step"]:
        raise ConfigError("v1 train_step requires zero_grad, forward, loss, backward and optimizer_step in order")
    correctness = value["correctness"]
    exact(correctness, "policy_sha256", "correctness")
    digest(correctness["policy_sha256"], "correctness.policy_sha256", True)
    return value


def workload_conditions(spec, batch_size):
    """Target names, hardware and environment are deliberately outside this identity."""
    return {"schema_version": SPEC_VERSION, **{key: spec[key] for key in
            ("workload", "data", "initial_state", "measurement", "correctness")},
            "batch_size": batch_size, "precision": "fp32", "devices": 1}


def workload_fingerprint(spec, batch_size):
    return fingerprint(workload_conditions(spec, batch_size))

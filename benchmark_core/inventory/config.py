"""Configuration is trusted executable policy; captured output is always data."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

CONFIG_VERSION = "recbench.inventory-config/v1"
PROFILE_VERSION = "recbench.inventory-profile/v1"
CATALOG = Path(__file__).resolve().parents[2] / "benchmark_catalog" / "inventory"
ID = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
PARSERS = {
    "raw", "lscpu", "meminfo", "os_release", "windows_host", "linux_network", "windows_network",
    "nvidia_csv", "nvidia_firmware_csv", "ascend_mapping", "ascend_board",
    "ascend_memory", "ascend_health", "version", "json_devices", "system_model", "bios_version",
}
DISCOVERY_PARSERS = {"nvidia_csv", "ascend_mapping", "json_devices"}


class ConfigError(ValueError):
    pass


def load_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ConfigError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                          parse_constant=lambda x: (_ for _ in ()).throw(ConfigError(f"Invalid number: {x}")))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def fields(obj, allowed, required, where):
    if not isinstance(obj, dict):
        raise ConfigError(f"{where}: expected an object")
    unknown = obj.keys() - set(allowed)
    missing = set(required) - obj.keys()
    if unknown or missing:
        raise ConfigError(f"{where}: unknown={sorted(unknown)}, missing={sorted(missing)}")


def text(value, where, identifier=False):
    if not isinstance(value, str) or not value.strip() or len(value) > 4000 or "\x00" in value:
        raise ConfigError(f"{where}: expected nonempty text (max 4000 characters)")
    if identifier and not ID.fullmatch(value):
        raise ConfigError(f"{where}: use a lowercase identifier, max 64 characters")


def integer(value, low, high, where):
    if type(value) is not int or not low <= value <= high:
        raise ConfigError(f"{where}: expected integer {low}..{high}")


def strings(value, where, allow_empty=False):
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ConfigError(f"{where}: expected a list")
    for item in value:
        text(item, where)


def validate_profile(profile):
    fields(profile, {"schema_version", "id", "kind", "vendor", "platforms", "description", "probes"},
           {"schema_version", "id", "kind", "platforms", "probes"}, "profile")
    text(profile["kind"], "profile.kind")
    if profile["schema_version"] != PROFILE_VERSION or profile["kind"] not in {"host", "gpu", "npu", "accelerator"}:
        raise ConfigError("Unsupported profile version/kind")
    text(profile["id"], "profile.id", True)
    strings(profile["platforms"], "profile.platforms")
    if not set(profile["platforms"]) <= {"Linux", "Windows", "Darwin"}:
        raise ConfigError("Unsupported OS in profile")
    if profile["kind"] != "host":
        text(profile.get("vendor"), "profile.vendor")
    probes = profile["probes"]
    if not isinstance(probes, list) or not 1 <= len(probes) <= 32:
        raise ConfigError("A profile requires 1..32 probes")
    ids, discoveries = set(), []
    for probe in probes:
        fields(probe, {"id", "argv", "paths", "parser", "required", "per_device", "platforms"},
               {"id", "parser", "required"}, "probe")
        text(probe["id"], "probe.id", True)
        if probe["id"] in ids:
            raise ConfigError("Duplicate probe id")
        ids.add(probe["id"])
        text(probe["parser"], "probe.parser")
        if probe["parser"] not in PARSERS:
            raise ConfigError(f"Unknown parser: {probe['parser']}")
        if type(probe["required"]) is not bool or type(probe.get("per_device", False)) is not bool:
            raise ConfigError("required/per_device must be boolean")
        if ("argv" in probe) == ("paths" in probe):
            raise ConfigError("A probe requires exactly one of argv or paths")
        strings(probe.get("argv", probe.get("paths")), "probe command/paths")
        if "platforms" in probe:
            strings(probe["platforms"], "probe.platforms")
            if not set(probe["platforms"]) <= set(profile["platforms"]):
                raise ConfigError("Probe OS must be supported by its profile")
        for argument in probe.get("argv", []) + probe.get("paths", []):
            # Only these complete tokens are substituted. Shell and Python syntax are not evaluated.
            if argument in {"{device_id}", "{chip_id}"} and not probe.get("per_device"):
                raise ConfigError("Device tokens require per_device=true")
        if probe["parser"] in DISCOVERY_PARSERS:
            if probe.get("per_device") or not probe["required"]:
                raise ConfigError("Discovery must be required and not per-device")
            discoveries.append(probe)
        if probe["parser"] in {"ascend_board", "ascend_memory", "ascend_health"} and not probe.get("per_device"):
            raise ConfigError("Device detail parsers require per_device=true")
        if profile["kind"] != "host" and probe["parser"] in {"lscpu", "meminfo", "os_release", "windows_host", "linux_network", "windows_network", "system_model", "bios_version"}:
            raise ConfigError("Host parsers require a host profile")
    if profile["kind"] != "host" and (len(discoveries) != 1 or probes[0] is not discoveries[0]):
        raise ConfigError("Accelerator profile must start with exactly one discovery probe")
    if profile["kind"] == "host" and (discoveries or any(p.get("per_device") for p in probes)):
        raise ConfigError("Host profiles cannot discover accelerators or use per_device")
    return profile


def load_config(path):
    path = Path(path).resolve()
    config = load_json(path)
    fields(config, {"schema_version", "nodes", "profile_files", "execution"},
           {"schema_version", "nodes"}, "config")
    if config["schema_version"] != CONFIG_VERSION:
        raise ConfigError("Unsupported inventory config version")
    execution = config.setdefault("execution", {})
    fields(execution, {"timeout_seconds", "max_output_bytes", "max_devices"}, set(), "execution")
    for key, default, low, high in (("timeout_seconds", 15, 1, 120),
                                  ("max_output_bytes", 262144, 1024, 4194304),
                                  ("max_devices", 64, 1, 256)):
        integer(execution.setdefault(key, default), low, high, key)
    profiles = {}
    for source in sorted((CATALOG / "profiles").glob("*.json")):
        profile = validate_profile(load_json(source))
        profiles[profile["id"]] = profile
    strings(config.get("profile_files", []), "profile_files", allow_empty=True)
    custom_ids = set()
    for name in config.get("profile_files", []):
        profile = validate_profile(load_json(path.parent / name))
        if profile["id"] in custom_ids:
            raise ConfigError("Duplicate custom profile id")
        custom_ids.add(profile["id"])
        profiles[profile["id"]] = profile  # Explicit replacement; snapshot contains the resolved policy.
    nodes = config["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 1000:
        raise ConfigError("nodes must contain 1..1000 entries")
    node_ids = set()
    for node in nodes:
        fields(node, {"id", "expected_hostname", "expected_os", "profiles", "expected_devices", "access", "labels", "notes"},
               {"id", "expected_hostname", "expected_os", "profiles"}, "node")
        text(node["id"], "node.id", True)
        if node["id"] in node_ids:
            raise ConfigError("Duplicate node id")
        node_ids.add(node["id"])
        text(node["expected_hostname"], "node.expected_hostname")
        text(node["expected_os"], "node.expected_os")
        if node["expected_os"] not in {"Linux", "Windows", "Darwin"}:
            raise ConfigError("Invalid expected_os")
        strings(node["profiles"], "node.profiles")
        if len(set(node["profiles"])) != len(node["profiles"]):
            raise ConfigError("Duplicate node profile")
        for profile_id in node["profiles"]:
            if profile_id not in profiles:
                raise ConfigError(f"Unknown profile: {profile_id}")
            if node["expected_os"] not in profiles[profile_id]["platforms"]:
                raise ConfigError(f"{profile_id} does not support {node['expected_os']}")
        if sum(profiles[x]["kind"] == "host" for x in node["profiles"]) != 1:
            raise ConfigError("Select exactly one host profile")
        vendors = [profiles[x]["vendor"].casefold() for x in node["profiles"] if profiles[x]["kind"] != "host"]
        if len(set(vendors)) != len(vendors):
            raise ConfigError("Select one discovery profile per vendor; override its profile to change commands")
        expected = node.setdefault("expected_devices", {})
        if not isinstance(expected, dict):
            raise ConfigError("expected_devices must be an object")
        for key, value in expected.items():
            if key not in node["profiles"] or profiles[key]["kind"] == "host":
                raise ConfigError("Device expectations must reference selected accelerator profiles")
            integer(value, 0, 256, f"expected_devices.{key}")
        access = node.setdefault("access", {"method": "local"})
        fields(access, {"method", "target", "scheduler", "notes"}, {"method"}, "access")
        text(access["method"], "access.method")
        if access["method"] not in {"local", "ssh", "scheduler"}:
            raise ConfigError("access.method must be local, ssh or scheduler")
        for key, value in access.items():
            text(value, f"access.{key}")
        if access["method"] != "local":
            text(access.get("target"), "access.target")
        labels = node.setdefault("labels", {})
        if not isinstance(labels, dict):
            raise ConfigError("labels must be an object")
        for key, value in labels.items():
            text(key, "label key")
            text(value, "label value")
        if "notes" in node:
            text(node["notes"], "node.notes")
    return config, profiles


def resolve_node(config, profiles, node_id):
    for node in config["nodes"]:
        if node["id"] == node_id:
            return copy.deepcopy(node), [copy.deepcopy(profiles[p]) for p in node["profiles"]]
    raise ConfigError(f"Unknown node: {node_id}")

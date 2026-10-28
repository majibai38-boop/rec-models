"""Portable snapshots and an offline fleet view; no guessed missing nodes."""
from __future__ import annotations

import json
from pathlib import Path

from .collector import SNAPSHOT_VERSION
from .config import ConfigError, fingerprint, load_json


def cell(value):
    if value is None:
        return "Unknown"
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def markdown(snapshot):
    host = snapshot["host"]
    lines = [f"# Node inventory: {cell(snapshot['node_id'])}", "",
             f"Status: **{snapshot['status']}** - captured {snapshot['completed_at']}", "",
             "Inventory only. Runtime compatibility, device usability and benchmark readiness have not been validated.", "",
             "| Host field | Observed value |", "|---|---|"]
    for key in ("hostname", "os", "os_release", "os_distribution", "architecture", "cpu_model",
                "cpu_sockets", "cpu_physical_cores", "cpu_logical_count", "memory_total_bytes", "numa_nodes",
                "system_model", "bios_version"):
        lines.append(f"| {key} | {cell(host.get(key))} |")
    lines += ["", "| Backend | Observed count | Expected count | Discovery |", "|---|---|---|---|"]
    for backend, values in snapshot["backends"].items():
        lines.append(f"| {cell(backend)} | {cell(values['observed_count'])} | {cell(values['expected_count'])} | {values['discovery_status']} |")
    lines += ["", "| Backend / device | Model | PCI bus | Memory | Driver | Firmware | Health |", "|---|---|---|---|---|---|---|"]
    for device in snapshot["accelerators"]:
        memory = device.get("memory_capacity") or device.get("memory_tiers")
        if memory is None and device.get("memory_capacity_bytes") is not None:
            memory = {"value": device["memory_capacity_bytes"], "unit": "bytes"}
        lines.append("| " + " | ".join(cell(x) for x in (f"{device['backend']} / {device['id']}", device["model"],
                     device.get("pci_bus_id"), json.dumps(memory, ensure_ascii=False) if memory else None,
                     device.get("driver_version"), device.get("firmware_version"), device.get("health"))) + " |")
    lines += ["", "Counts cover selected profiles and the collector's visible scope. NVIDIA: physical GPUs; Ascend: compute chips, not boards.", "",
              "## Collection evidence", "", "| Probe | Status | Required |", "|---|---|---|"]
    for evidence in snapshot["evidence"]:
        lines.append(f"| {cell(evidence['id'])} | {evidence['status']} | {evidence['required']} |")
    lines += ["", "Commands, bounded raw output, parsed details, software versions and field provenance are in snapshot.json.", "", "## Issues", ""]
    lines += [f"- [{x['severity']}] {cell(x['code'])}: {cell(x['message'])}" for x in snapshot["issues"]] or ["No collection issues."]
    return "\n".join(lines) + "\n"


def write_snapshot(snapshot, output):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "report.md").write_text(markdown(snapshot), encoding="utf-8")
    temporary = directory / "snapshot.json.tmp"
    temporary.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(directory / "snapshot.json")
    return directory / "snapshot.json"


def read_snapshot(path):
    value = load_json(Path(path))
    if not isinstance(value, dict) or value.get("schema_version") != SNAPSHOT_VERSION:
        raise ConfigError(f"{path}: unsupported snapshot version")
    for key, kind in (("node_id", str), ("host", dict), ("accelerators", list), ("backends", dict), ("issues", list)):
        if not isinstance(value.get(key), kind):
            raise ConfigError(f"{path}: invalid {key}")
    if value.get("status") not in {"complete", "complete_with_warnings", "partial"}:
        raise ConfigError(f"{path}: invalid status")
    for key in ("hostname", "os"):
        if not isinstance(value["host"].get(key), str):
            raise ConfigError(f"{path}: invalid host.{key}")
    for backend, state in value["backends"].items():
        if not isinstance(state, dict) or "observed_count" not in state:
            raise ConfigError(f"{path}: invalid backend state")
        count = state["observed_count"]
        if count is not None and (type(count) is not int or count < 0):
            raise ConfigError(f"{path}: invalid observed count")
        devices = [d for d in value["accelerators"] if isinstance(d, dict) and d.get("backend") == backend]
        if count is not None and count != len(devices):
            raise ConfigError(f"{path}: accelerator count mismatch")
        if state.get("discovery_status") != ("unknown" if count is None else "ok"):
            raise ConfigError(f"{path}: inconsistent discovery status")
        if count is None and (devices or value["status"] != "partial"):
            raise ConfigError(f"{path}: unknown discovery cannot be complete or carry inferred devices")
    if any(not isinstance(d, dict) or d.get("backend") not in value["backends"] for d in value["accelerators"]):
        raise ConfigError(f"{path}: invalid accelerator record")
    identities = []
    for device in value["accelerators"]:
        if not isinstance(device.get("id"), str) or not device["id"] or not isinstance(device.get("model"), str):
            raise ConfigError(f"{path}: invalid device identity")
        identities.append((device["backend"], device["id"]))
    if len(set(identities)) != len(identities):
        raise ConfigError(f"{path}: duplicate device identity")
    expected_total = None if any(s["observed_count"] is None for s in value["backends"].values()) else len(value["accelerators"])
    observed_total = value.get("observed_accelerator_count")
    if observed_total != expected_total or isinstance(observed_total, bool):
        raise ConfigError(f"{path}: inconsistent observed total")
    return value


def summarize(paths, config=None, profiles=None):
    snapshots = {}
    for path in paths:
        snapshot = read_snapshot(path)
        if snapshot["node_id"] in snapshots:
            raise ConfigError("Duplicate node snapshots: select one capture per node explicitly")
        snapshots[snapshot["node_id"]] = (snapshot, str(path))
    nodes = config["nodes"] if config else [{"id": node_id} for node_id in snapshots]
    unknown = snapshots.keys() - {node["id"] for node in nodes}
    if unknown:
        raise ConfigError(f"Snapshots not in inventory config: {sorted(unknown)}")
    rows = []
    for node in nodes:
        if node["id"] not in snapshots:
            rows.append({"node_id": node["id"], "status": "not_collected", "observed_count": None})
            continue
        snapshot, path = snapshots[node["id"]]
        issues = []
        if config:
            expected_hash = fingerprint({"node": node, "execution": config["execution"]})
            profile_hash = fingerprint([profiles[p] for p in node["profiles"]])
            if snapshot.get("config_sha256") != expected_hash or snapshot.get("profiles_sha256") != profile_hash:
                issues.append("Configuration/profile changed since capture; recollect this node")
            if snapshot["host"]["hostname"].casefold() != node["expected_hostname"].casefold() or snapshot["host"]["os"] != node["expected_os"]:
                issues.append("Observed host does not match configured node")
        rows.append({"node_id": node["id"], "status": "stale" if issues else snapshot["status"],
                     "hostname": snapshot["host"]["hostname"], "os": snapshot["host"]["os"],
                     "observed_count": snapshot.get("observed_accelerator_count"), "backends": snapshot["backends"],
                     "snapshot": path, "issues": issues, "captured_at": snapshot.get("completed_at")})
    return {"schema_version": "recbench.inventory-summary/v1", "scope": "configured_nodes" if config else "supplied_snapshots",
            "status": "partial" if any(r["status"] in {"partial", "stale", "not_collected"} for r in rows) else "complete",
            "nodes": rows}

"""Collect one local node; declarations never override observed facts."""
from __future__ import annotations

import datetime as dt
import importlib.metadata
import os
import platform
import socket
import sys

from . import __version__
from .config import ConfigError, fingerprint
from .parsers import parse
from .runner import LocalRunner

SNAPSHOT_VERSION = "recbench.inventory-snapshot/v1"


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def base_host():
    return {"hostname": socket.gethostname(), "os": platform.system(), "os_release": platform.release(),
            "architecture": platform.machine(), "cpu_model": None, "cpu_logical_count": os.cpu_count(),
            "cpu_physical_cores": None, "cpu_sockets": None, "memory_total_bytes": None,
            "numa_nodes": None, "network_interfaces": None, "system_model": None, "bios_version": None,
            "cpu_affinity_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def software_inventory():
    packages = {}
    for package in ("torch", "torch-npu", "torchrec", "fbgemm-gpu", "numpy"):
        try:
            packages[package] = {"version": importlib.metadata.version(package), "status": "installed"}
        except importlib.metadata.PackageNotFoundError:
            packages[package] = {"version": None, "status": "not_installed"}
    return {"scope": "collector_python_environment", "python": platform.python_version(),
            "executable": sys.executable, "packages": packages, "version_reports": {}, "runtime_validation": "not_run"}


def collect(node, profiles, settings, runner=None, host=None):
    host = dict(base_host() if host is None else host)
    if node["expected_hostname"].casefold() != host["hostname"].casefold():
        raise ConfigError(f"Host mismatch: expected {node['expected_hostname']}, running on {host['hostname']}")
    if node["expected_os"] != host["os"]:
        raise ConfigError(f"OS mismatch: expected {node['expected_os']}, running on {host['os']}")
    runner = runner or LocalRunner()
    snapshot = {"schema_version": SNAPSHOT_VERSION, "collector_version": __version__, "started_at": utc_now(),
                "node_id": node["id"], "declared": node, "execution_scope": "local_process_visible",
                "resolved_profiles": profiles, "config_sha256": fingerprint({"node": node, "execution": settings}),
                "profiles_sha256": fingerprint(profiles), "host": host, "host_sources": {},
                "software": software_inventory(), "accelerators": [], "backends": {}, "evidence": [], "issues": [],
                "benchmark_readiness": "not_validated"}
    snapshot["runtime_context"] = {
        "container_hint": bool(os.environ.get("container") or os.path.exists("/.dockerenv")),
        "device_visibility_environment": {key: os.environ[key] for key in
             ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES") if key in os.environ},
        "resource_budget": "not_calculated",
    }
    snapshot["host_sources"] = {key: "builtin.python" for key, value in host.items() if value is not None}
    snapshot["evidence"].append({"id": "builtin.python", "status": "ok", "required": True,
                                "parser": "builtin", "captured_at": utc_now(),
                                "parsed": {key: value for key, value in host.items() if value is not None}})
    required_failed = False

    def issue(code, message, required=False, evidence=None):
        nonlocal required_failed
        required_failed |= required
        snapshot["issues"].append({"code": code, "message": message, "severity": "error" if required else "warning",
                                   "evidence_id": evidence})

    for profile in profiles:
        profile_id = profile["id"]
        devices = []
        discovery_ok = profile["kind"] == "host"
        for probe in profile["probes"]:
            if host["os"] not in probe.get("platforms", profile["platforms"]):
                snapshot["evidence"].append({"id": f"{profile_id}.{probe['id']}", "status": "unsupported_os",
                                              "required": probe["required"], "parser": probe["parser"]})
                if probe["required"]:
                    issue("unsupported_os", f"Required probe {probe['id']} unavailable on {host['os']}", True)
                continue
            targets = devices if probe.get("per_device") else [None]
            for target in targets:
                evidence_id = f"{profile_id}.{probe['id']}" + (f".{target['id']}" if target else "")
                result = runner(probe, settings, target)
                result.update(id=evidence_id, required=probe["required"], parser=probe["parser"], captured_at=utc_now())
                snapshot["evidence"].append(result)
                if result["status"] != "ok":
                    issue(result["status"], f"Probe {evidence_id} did not complete", probe["required"], evidence_id)
                    continue
                try:
                    parsed = parse(probe["parser"], result["stdout"])
                    if probe["parser"] == "version":
                        snapshot["software"]["version_reports"][evidence_id] = {
                            "text": parsed["details"]["version_text"], "source": evidence_id}
                    if "devices" in parsed:
                        found = parsed["devices"]
                        if len(found) > settings["max_devices"]:
                            raise ValueError("Device count exceeds configured max_devices")
                        if len({x["id"] for x in found}) != len(found):
                            raise ValueError("Duplicate device identities in discovery")
                        for device in found:
                            device.update(backend=profile_id, vendor=profile["vendor"], kind=profile["kind"],
                                          sources={key: evidence_id for key in device})
                            for key in ("driver_version", "firmware_version", "pci_bus_id", "memory_capacity_bytes", "health"):
                                device.setdefault(key, None)
                        devices = found
                        discovery_ok = True
                    if "host" in parsed:
                        snapshot["host"].update(parsed["host"])
                        snapshot["host_sources"].update({key: evidence_id for key in parsed["host"]})
                    if "device" in parsed:
                        if target is None:
                            raise ValueError("Device detail parser requires a per-device probe")
                        target.update(parsed["device"])
                        target["sources"].update({key: evidence_id for key in parsed["device"]})
                    if "device_updates" in parsed:
                        updates = []
                        for update in parsed["device_updates"]:
                            matches = [d for d in devices if update.get("uuid") and d.get("uuid") == update["uuid"]]
                            if len(matches) != 1:
                                raise ValueError("Cannot associate firmware row to a unique GPU UUID")
                            updates.append((matches[0], update))
                        for matched, update in updates:
                            matched.update(update)
                            matched["sources"].update({key: evidence_id for key in update})
                    result["parsed"] = parsed.get("details", {})
                except (ValueError, KeyError, TypeError, AttributeError) as exc:
                    result["status"] = "parse_error"
                    result["parse_error"] = str(exc)
                    issue("parse_error", f"{evidence_id}: {exc}", probe["required"], evidence_id)
        if profile["kind"] != "host":
            count = len(devices) if discovery_ok else None
            expected = node["expected_devices"].get(profile_id)
            snapshot["backends"][profile_id] = {"observed_count": count, "expected_count": expected,
                                                "discovery_status": "ok" if discovery_ok else "unknown"}
            if expected is not None and count != expected:
                issue("device_count_mismatch", f"{profile_id}: expected {expected}, observed {count}", True)
            snapshot["accelerators"].extend(devices)
    snapshot["detected_accelerator_count"] = len(snapshot["accelerators"])
    snapshot["observed_accelerator_count"] = (None if any(x["observed_count"] is None for x in snapshot["backends"].values())
                                               else len(snapshot["accelerators"]))
    for field in ("cpu_model", "cpu_logical_count", "memory_total_bytes"):
        if not snapshot["host"].get(field):
            issue("missing_host_field", f"Host field is unknown: {field}", True)
    snapshot["status"] = "partial" if required_failed else ("complete_with_warnings" if snapshot["issues"] else "complete")
    snapshot["completed_at"] = utc_now()
    snapshot["unknown_fields"] = [f"host.{key}" for key, value in host.items() if value is None]
    for device in snapshot["accelerators"]:
        snapshot["unknown_fields"].extend(f"{device['backend']}.{device['id']}.{key}" for key, value in device.items() if value is None)
    hardware = {"host": {k: v for k, v in host.items() if k not in {"network_interfaces"}},
                "accelerators": [{k: v for k, v in d.items() if k not in {"sources", "health"}} for d in snapshot["accelerators"]]}
    snapshot["observed_configuration_sha256"] = fingerprint({"hardware": hardware, "software": snapshot["software"]})
    return snapshot

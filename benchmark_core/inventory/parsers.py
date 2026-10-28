"""Versioned parsers; unsupported formats fail instead of inventing hardware."""
from __future__ import annotations

import csv
import io
import json
import math
import re


def number(value, integer=False):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric hardware value")
    if value is None or str(value).strip().lower() in {"", "n/a", "na", "unknown", "[not supported]", "not supported", "-"}:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0 or (integer and not result.is_integer()):
        raise ValueError(f"Invalid nonnegative number: {value}")
    return int(result) if integer else result


def optional(value):
    if value is None or str(value).strip().lower() in {"", "n/a", "na", "unknown", "[not supported]", "not supported", "-"}:
        return None
    return str(value).strip()


def key_values(output):
    result = {}
    for line in output.splitlines():
        if ":" in line:
            key, value = line.strip().split(":", 1)
            result[key.strip()] = value.strip()
    return result


def csv_rows(output, columns):
    rows = [row for row in csv.reader(io.StringIO(output)) if any(x.strip() for x in row)]
    if any(len(row) != columns for row in rows):
        raise ValueError(f"Expected {columns} CSV columns")
    return [[part.strip() for part in row] for row in rows]


def parse(parser, output):
    if parser in {"system_model", "bios_version"}:
        if not optional(output):
            raise ValueError(f"Empty {parser}")
        return {"host": {parser: output.strip()}}
    if parser in {"raw", "version"}:
        if not output.strip():
            raise ValueError("Empty evidence")
        return {"details": {"text" if parser == "raw" else "version_text": output.strip()}}
    if parser == "lscpu":
        entries = json.loads(output)["lscpu"]
        values = {x["field"].rstrip(":"): x["data"] for x in entries}
        logical = number(values.get("CPU(s)"), True)
        sockets = number(values.get("Socket(s)"), True)
        per_socket = number(values.get("Core(s) per socket"), True)
        if not logical or not optional(values.get("Model name")):
            raise ValueError("lscpu missing model or logical CPU count")
        return {"host": {"cpu_model": optional(values.get("Model name")), "cpu_logical_count": logical,
                         "cpu_sockets": sockets, "cpu_physical_cores": sockets * per_socket if sockets and per_socket else None,
                         "numa_nodes": number(values.get("NUMA node(s)"), True)}, "details": values}
    if parser == "meminfo":
        match = re.search(r"^MemTotal:\s+(\d+)\s+kB\s*$", output, re.M)
        if not match or int(match[1]) <= 0:
            raise ValueError("Missing MemTotal in /proc/meminfo")
        return {"host": {"memory_total_bytes": int(match[1]) * 1024}}
    if parser == "os_release":
        values = dict(line.split("=", 1) for line in output.splitlines() if "=" in line and not line.startswith("#"))
        name = values.get("PRETTY_NAME", "").strip('"\'')
        if not name:
            raise ValueError("Missing PRETTY_NAME")
        return {"host": {"os_distribution": name}}
    if parser == "windows_host":
        values = json.loads(output)
        required = {"cpu_model", "cpu_logical_count", "cpu_physical_cores", "cpu_sockets", "memory_total_bytes"}
        if not isinstance(values, dict) or not required <= values.keys():
            raise ValueError("Incomplete Windows CIM response")
        if values.keys() - required - {"system_model", "system_manufacturer", "bios_version", "os_distribution"}:
            raise ValueError("Unexpected Windows CIM fields")
        for field in required - {"cpu_model"}:
            values[field] = number(values[field], True)
            if not values[field]:
                raise ValueError(f"Missing {field}")
        if not optional(values["cpu_model"]):
            raise ValueError("Missing CPU model")
        return {"host": values}
    if parser == "windows_network":
        values = json.loads(output)
        if not isinstance(values, list):
            raise ValueError("Expected Windows network adapter array")
        return {"host": {"network_interfaces": [{"name": x["Name"], "connection_name": x.get("NetConnectionID"),
                                                  "status_code": x.get("NetConnectionStatus"),
                                                  "speed_bits_per_second": number(x.get("Speed"), True)} for x in values]}}
    if parser == "linux_network":
        values = json.loads(output)
        if not isinstance(values, list):
            raise ValueError("Expected ip -j link array")
        return {"host": {"network_interfaces": [{"name": x["ifname"], "state": x.get("operstate"),
                                                  "mtu": x.get("mtu")} for x in values]}}
    if parser == "nvidia_csv":
        devices = []
        for index, uuid, name, bus, memory, driver in csv_rows(output, 6):
            if not index.isdigit() or not optional(name):
                raise ValueError("Invalid NVIDIA identity")
            capacity = number(memory)
            devices.append({"id": index, "physical_id": index, "chip_id": None, "uuid": optional(uuid),
                            "model": name, "pci_bus_id": optional(bus), "driver_version": optional(driver),
                            "memory_capacity": {"value": capacity, "unit": "MiB"},
                            "memory_capacity_bytes": int(capacity * 1048576) if capacity is not None else None})
        return {"devices": devices}
    if parser == "nvidia_firmware_csv":
        return {"device_updates": [{"uuid": optional(uuid), "firmware_version": optional(version)}
                                   for uuid, version in csv_rows(output, 2)]}
    if parser == "ascend_mapping":
        if re.search(r"(?:no\s+(?:npu|device)|total\s*:\s*0\s+npu)", output, re.I):
            return {"devices": []}
        if not re.search(r"NPU\s+ID.*Chip\s+ID.*Chip\s+Logic\s+ID.*Chip\s+Name", output, re.I):
            raise ValueError("Unsupported npu-smi mapping header; expected NPU ID / Chip ID / Chip Logic ID / Chip Name")
        devices = []
        for line in output.splitlines():
            line = line.strip().strip("|").strip()
            if not re.match(r"^\d+\s", line):
                continue
            match = re.fullmatch(r"(\d+)\s+(\d+)\s+(\d+|-|NA)\s+(.+)", line)
            if not match:
                raise ValueError("Unsupported npu-smi device row")
            physical, chip, logic, name = match.groups()
            if name.strip().upper() in {"MCU", "CPU"}:
                continue
            devices.append({"id": f"{physical}:{chip}", "physical_id": physical, "chip_id": chip,
                            "runtime_id": optional(logic), "model": name.strip(), "uuid": None})
        if not devices:
            raise ValueError("Mapping has no accelerator rows and did not explicitly report zero")
        return {"devices": devices}
    if parser == "ascend_board":
        values = key_values(output)
        mapping = {"PCIe Bus Info": "pci_bus_id", "Firmware Version": "firmware_version",
                   "Driver Version": "driver_version", "Chip Name": "model"}
        result = {target: optional(values[key]) for key, target in mapping.items() if key in values}
        if not result:
            raise ValueError("No recognized board fields")
        return {"device": result, "details": values}
    if parser == "ascend_memory":
        values, memories = key_values(output), {}
        for kind in ("HBM", "DDR"):
            key = f"{kind} Capacity(MB)"
            if key in values:
                memories[kind.lower()] = {"value": number(values[key]), "unit": "MB", "unit_basis": "vendor_reported"}
        if not memories:
            raise ValueError("No recognized HBM/DDR capacity fields")
        return {"device": {"memory_tiers": memories}, "details": values}
    if parser == "ascend_health":
        values = key_values(output)
        value = next((values[key] for key in ("Health", "Healthy", "NPU Health") if key in values), None)
        if value is None:
            raise ValueError("No recognized health field")
        return {"device": {"health": optional(value)}, "details": values}
    if parser == "json_devices":
        devices = json.loads(output)
        allowed = {"id", "physical_id", "chip_id", "runtime_id", "model", "uuid", "pci_bus_id",
                   "memory_capacity_bytes", "driver_version", "firmware_version", "health"}
        if not isinstance(devices, list):
            raise ValueError("json_devices expects a normalized JSON array")
        for device in devices:
            if not isinstance(device, dict) or not {"id", "model"} <= device.keys() or device.keys() - allowed:
                raise ValueError("Invalid json_devices fields")
            for key, value in device.items():
                if key == "memory_capacity_bytes":
                    if value is not None and (type(value) is not int or value < 0):
                        raise ValueError("memory_capacity_bytes must be a nonnegative integer or null")
                elif value is not None and (not isinstance(value, str) or not value.strip()):
                    raise ValueError(f"{key} must be nonempty text or null")
            if not device["id"] or not device["model"]:
                raise ValueError("Missing device identity")
            device.setdefault("physical_id", device["id"])
        return {"devices": devices}
    raise ValueError(f"Unknown parser: {parser}")

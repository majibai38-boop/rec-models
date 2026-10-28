"""python -m benchmark_core.inventory --help"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

from .collector import collect
from .config import CONFIG_VERSION, ConfigError, load_config, resolve_node, text
from .report import summarize, write_snapshot


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only node inventory; collect runs ON the selected node.")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write a config for this local host; never overwrite existing files")
    init.add_argument("--output", required=True)
    init.add_argument("--node", default="local")
    init.add_argument("--backend", choices=["none", "nvidia", "ascend"], default="none")
    init.add_argument("--expected-devices", type=int)
    for name in ("check", "plan", "collect"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True)
        if name != "check":
            cmd.add_argument("--node", required=True)
        if name == "collect":
            cmd.add_argument("--output", required=True, help="A NEW directory for snapshot.json and report.md")
    summary = sub.add_parser("summary", help="Offline aggregation; a config reveals missing/stale nodes")
    summary.add_argument("snapshots", nargs="+")
    summary.add_argument("--config")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            os_name = platform.system()
            if os_name not in {"Linux", "Windows"} or (args.backend == "ascend" and os_name != "Linux"):
                raise ConfigError("Built-in host profiles support Linux/Windows; Ascend requires Linux")
            if args.expected_devices is not None and (args.backend == "none" or not 0 <= args.expected_devices <= 256):
                raise ConfigError("--expected-devices needs an accelerator backend and a count in 0..256")
            text(args.node, "node", True)
            selected = ["linux-host" if os_name == "Linux" else "windows-host"]
            if args.backend != "none":
                selected.append(args.backend)
            config = {"schema_version": CONFIG_VERSION, "profile_files": [],
                      "nodes": [{"id": args.node, "expected_hostname": socket.gethostname(), "expected_os": os_name,
                                 "profiles": selected, "expected_devices": {} if args.expected_devices is None else {args.backend: args.expected_devices},
                                 "access": {"method": "local"}, "labels": {}}]}
            with Path(args.output).open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            emit({"status": "created", "config": args.output})
            return 0
        if args.command == "summary":
            config, profiles = load_config(args.config) if args.config else (None, None)
            result = summarize(args.snapshots, config, profiles)
            emit(result)
            return 3 if result["status"] == "partial" else 0
        config, profiles = load_config(args.config)
        if args.command == "check":
            emit({"status": "valid", "nodes": [n["id"] for n in config["nodes"]],
                  "note": "Configuration only; no node contacted or command executed."})
            return 0
        node, selected = resolve_node(config, profiles, args.node)
        if args.command == "plan":
            emit({"node": node, "execution": config["execution"], "profiles": selected,
                  "note": "Run collect locally on expected_hostname. access is metadata; no SSH is executed."})
            return 0
        if Path(args.output).exists():
            raise ConfigError("Output directory already exists; use a new capture directory")
        snapshot = collect(node, selected, config["execution"])
        path = write_snapshot(snapshot, args.output)
        emit({"status": snapshot["status"], "node_id": node["id"], "snapshot": str(path),
              "report": str(path.parent / "report.md"), "observed_accelerator_count": snapshot["observed_accelerator_count"],
              "issues": snapshot["issues"]})
        return 3 if snapshot["status"] == "partial" else 0
    except (ConfigError, OSError, ValueError) as exc:
        emit({"status": "error", "error": str(exc)})
        return 2


if __name__ == "__main__":
    sys.exit(main())

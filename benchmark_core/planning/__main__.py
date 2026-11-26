"""python -m benchmark_core.planning --help"""
import argparse
import json
from pathlib import Path
import sys

from benchmark_core.inventory.config import ConfigError, load_config, load_json
from .contracts import validate_recipe, validate_spec
from .planner import plan, write_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline environment and benchmark planning; never executes commands")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate one document's structure, not compatibility/readiness")
    check.add_argument("kind", choices=["spec", "recipe"])
    check.add_argument("path")
    resolve = sub.add_parser("plan", help="Expand matrix and explain blockers against inventory snapshots")
    resolve.add_argument("--spec", required=True)
    resolve.add_argument("--recipe", action="append", required=True, help="Repeat for each environment recipe")
    resolve.add_argument("--inventory-config", required=True)
    resolve.add_argument("--snapshot", action="append", default=[])
    resolve.add_argument("--max-age-hours", type=int, default=24)
    resolve.add_argument("--output", required=True, help="NEW directory for plan.json and report.md")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            value = load_json(Path(args.path))
            (validate_spec if args.kind == "spec" else validate_recipe)(value)
            result = {"status": "valid_structure", "execution_ready": False}
            code = 0
        else:
            if Path(args.output).exists():
                raise ConfigError("Output directory already exists; choose a new directory")
            config, profiles = load_config(args.inventory_config)
            value = plan(load_json(Path(args.spec)), [load_json(Path(p)) for p in args.recipe],
                         config, profiles, args.snapshot, max_age_hours=args.max_age_hours)
            path = write_plan(value, args.output)
            result = {"status": value["status"], "execution_ready": False, "runs": len(value["runs"]),
                      "plan": str(path), "report": str(path.parent / "report.md")}
            code = 3 if value["status"] == "blocked" else 0
    except (ConfigError, OSError, ValueError) as exc:
        result, code = {"status": "error", "error": str(exc)}, 2
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())

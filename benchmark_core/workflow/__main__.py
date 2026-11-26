"""python -m benchmark_core.workflow --help"""
import argparse
import json
import sys
from pathlib import Path

from benchmark_core.inventory.config import ConfigError, load_json
from benchmark_core.workflow.contracts import resolve_workflow, write_resolution
from benchmark_core.workflow.state import EXIT_BLOCKED, EXIT_OK, EXIT_USAGE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and resolve recbench.workflow/v1; never runs models, containers or transfers")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="Validate the workflow, its documents and target identity")
    check.add_argument("path")
    resolve = sub.add_parser("resolve", help="Write request, resolved spec, plan and staged admission")
    resolve.add_argument("path")
    resolve.add_argument("--output", required=True, help="NEW directory for the resolution artifacts")
    resolve.add_argument("--snapshot", action="append", default=[],
                         help="Repeat for each inventory snapshot; only the selected target's node is required")
    resolve.add_argument("--max-age-hours", type=int, default=24)
    resolve.add_argument("--entry-workspace", default=None,
                         help="Workspace that owns benchmark-results/and default archive root (default: cwd)")
    args = parser.parse_args(argv)
    try:
        workflow_path = Path(args.path)
        workflow = load_json(workflow_path)
        if args.command == "check":
            result = resolve_workflow(workflow, workflow_path=workflow_path)
            summary = {"status": "valid_structure", "execution_ready": False,
                       "target": result["resolved"]["target"]["id"],
                       "recipe": result["resolved"]["recipe"]["id"],
                       "archive": result["resolved"]["archive"],
                       "stages": {stage["id"]: stage["status"] for stage in result["admission"]["stages"]}}
            code = EXIT_OK
        else:
            if Path(args.output).exists():
                raise ConfigError("Output directory already exists; choose a new directory")
            result = resolve_workflow(workflow, workflow_path=workflow_path,
                                      entry_workspace=args.entry_workspace,
                                      snapshot_paths=args.snapshot, max_age_hours=args.max_age_hours)
            directory = write_resolution(result, args.output)
            summary = {"status": result["admission"]["status"], "execution_ready": False,
                       "target": result["resolved"]["target"]["id"],
                       "output": str(directory),
                       "request": str(directory / "request.json"),
                       "resolved_spec": str(directory / "resolved_spec.json"),
                       "plan": str(directory / "plan.json"),
                       "admission": str(directory / "admission.json"),
                       "blocked_stages": [stage["id"] for stage in result["admission"]["stages"]
                                          if stage["status"] == "blocked"]}
            code = EXIT_BLOCKED if result["admission"]["status"] == "blocked" else EXIT_OK
    except (ConfigError, OSError, ValueError) as exc:
        summary, code = {"status": "error", "error": str(exc)}, EXIT_USAGE
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())

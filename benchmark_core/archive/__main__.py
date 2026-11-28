"""python -m benchmark_core.archive --help"""
import argparse
import json
import sys

from benchmark_core.archive.layout import create_staging
from benchmark_core.archive.publish import publish, verify
from benchmark_core.inventory.config import ConfigError
from benchmark_core.workflow.state import EXIT_BLOCKED, EXIT_OK, EXIT_USAGE


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stage, verify and publish one benchmark attempt; never runs models")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("new", help="Create a new staging attempt directory")
    create.add_argument("--root", required=True, help="Archive root (contains staging/ and runs/)")
    create.add_argument("--run-id", required=True)
    create.add_argument("--attempt-id", default=None)
    publish_cmd = sub.add_parser("publish", help="Verify a staged attempt and publish it atomically")
    publish_cmd.add_argument("--root", required=True)
    publish_cmd.add_argument("--run-id", required=True)
    publish_cmd.add_argument("--attempt-id", required=True)
    verify_cmd = sub.add_parser("verify", help="Recompute the published attempt's file list and hashes")
    verify_cmd.add_argument("path")
    args = parser.parse_args(argv)
    try:
        if args.command == "new":
            directory, attempt_id = create_staging(args.root, args.run_id, args.attempt_id)
            result = {"status": "created", "attempt_id": attempt_id, "path": str(directory)}
            code = EXIT_OK
        elif args.command == "publish":
            result = publish(args.root, args.run_id, args.attempt_id)
            code = EXIT_OK
        else:
            result = verify(args.path)
            code = EXIT_OK if result["ok"] else EXIT_BLOCKED
    except (ConfigError, OSError, ValueError) as exc:
        result, code = {"status": "error", "error": str(exc)}, EXIT_USAGE
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    sys.exit(main())

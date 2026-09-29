import argparse
import os
from pathlib import Path
import subprocess
import sys

from .settings import Settings


def main():
    parser = argparse.ArgumentParser(description="Single-machine GPU/NPU profiling service")
    parser.add_argument("command", choices=("serve", "worker", "all"), nargs="?", default="all")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir")
    parser.add_argument("--auth", choices=("auto", "local", "token"), default=os.environ.get("PROFILING_WEB_AUTH", "auto"))
    args = parser.parse_args()
    if args.data_dir:
        os.environ["PROFILING_WEB_DATA"] = str(Path(args.data_dir).resolve())
    local_host = args.host in {"localhost", "127.0.0.1", "::1"}
    mode = args.auth
    if mode == "auto":
        mode = "local" if local_host and not os.environ.get("PROFILING_WEB_TOKEN") else "token"
    if mode == "local" and not local_host:
        parser.error("Local no-token mode may only bind a loopback address")
    os.environ["PROFILING_WEB_AUTH"] = mode
    if args.command != "worker" and args.host not in {"localhost", "127.0.0.1", "::1"} and not os.environ.get("PROFILING_WEB_TOKEN"):
        parser.error("Non-loopback deployment requires an explicit PROFILING_WEB_TOKEN")
    settings = Settings().prepare()
    print(f"Data directory: {settings.data_dir}", flush=True)
    if mode == "local":
        print("Local mode: no token required; loopback access only", flush=True)
    elif not os.environ.get("PROFILING_WEB_TOKEN"):
        print(f"Access token saved in: {settings.data_dir / 'access-token'} (keep private)", flush=True)
    if args.command == "worker":
        from .worker import main as run_worker
        run_worker()
        return
    worker = None
    if args.command == "all":
        worker = subprocess.Popen([sys.executable, "-m", "profiling_web", "worker"],
                                  creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        import uvicorn
        from .app import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port, workers=1, proxy_headers=mode != "local")
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()


if __name__ == "__main__":
    main()

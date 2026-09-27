"""Launch the pinned bundle, or the single source implementation in this repo."""
from pathlib import Path
import subprocess
import sys


def main():
    bundled = Path(__file__).resolve().with_name("profiling_tools.pyz")
    if bundled.is_file():
        return subprocess.run([sys.executable, str(bundled), *sys.argv[1:]], check=False).returncode
    repo = Path(__file__).resolve().parents[3]
    if not (repo / "profiling_tools" / "version.py").is_file():
        print("Missing profiling engine. Use the built skill bundle, not a standalone copy of the source skill folder.", file=sys.stderr)
        return 2
    sys.path.insert(0, str(repo))
    from profiling_tools.__main__ import main as run
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

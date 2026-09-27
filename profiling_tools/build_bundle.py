"""Build a portable stdlib-only application and skill from one source checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import zipfile

from .result_api import file_identity
from .version import __version__


def build_bundle(output_dir):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError("bundle destination already exists; choose a new directory")
    source = Path(__file__).resolve().parent
    skill_source = source.parent / "skills" / "gpu-npu-profiling"
    if not skill_source.is_dir():
        raise ValueError("building the skill bundle requires the source checkout")
    output.mkdir(parents=True)

    def write_app(path):
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("__main__.py", "from profiling_tools.__main__ import main\nraise SystemExit(main())\n")
            # Only package modules: never include example traces, reports or tests.
            for module in sorted(source.glob("*.py")):
                archive.write(module, "profiling_tools/" + module.name)

    app = output / "profiling_tools.pyz"
    write_app(app)
    skill = output / "gpu-npu-profiling"
    names = ["SKILL.md", "agents/openai.yaml", "scripts/run_compare.py"]
    names += ["references/" + p.name for p in sorted((skill_source / "references").glob("*.md"))]
    for name in names:
        target = skill / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(skill_source / name, target)
    shutil.copyfile(app, skill / "scripts" / app.name)
    manifest = {"schema_version": 1, "tool_version": __version__,
                "application": str(app), "skill": str(skill), "application_sha256": file_identity(app)["sha256"]}
    (output / "bundle.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(build_bundle(args.output_dir), ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"build_bundle: {exc}\n")


if __name__ == "__main__":
    main()

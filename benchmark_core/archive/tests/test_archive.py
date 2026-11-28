"""Local archive protocol tests; no remote destination or benchmark execution is involved."""
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from benchmark_core.archive.__main__ import main
from benchmark_core.archive.layout import (
    attempt_published, attempt_staging, create_staging, existing_attempts,
)
from benchmark_core.archive.publish import COMPLETE, INDEX, publish, verify
from benchmark_core.inventory.config import ConfigError, load_json
from benchmark_core.workflow.manifest import sha256_file, write_json
from benchmark_core.workflow.state import EXIT_BLOCKED, EXIT_OK, EXIT_USAGE


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "archive"

    def stage(self, run_id="run-1", attempt_id=None, files=None):
        directory, attempt_id = create_staging(self.root, run_id, attempt_id)
        for name, content in (files or {}).items():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return directory, attempt_id

    def published(self, attempt_id="attempt-1"):
        return attempt_published(self.root, "run-1", attempt_id)

    def run_cli(self, *argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(list(argv))
        return code, json.loads(buffer.getvalue())

    def test_create_staging_allocates_sequential_attempts(self):
        first, attempt = self.stage()
        self.assertEqual(attempt, "attempt-1")
        self.assertTrue(first.is_dir())
        _, second = self.stage()
        self.assertEqual(second, "attempt-2")
        self.assertEqual(existing_attempts(self.root, "run-1"), ["attempt-1", "attempt-2"])
        with self.assertRaises(ConfigError):
            create_staging(self.root, "run-1", "attempt-1")
        with self.assertRaises(ConfigError):
            create_staging(self.root, "Bad Run")
        self.assertEqual(attempt_staging(self.root, "run-1", "attempt-1"), first)

    def test_publish_writes_index_and_complete_then_moves(self):
        staged, attempt_id = self.stage(files={"manifest.json": "{}\n", "logs/prepare.log": "ok\n"})
        result = publish(self.root, "run-1", attempt_id)
        final = self.published()
        self.assertEqual(result["status"], "published")
        self.assertEqual(Path(result["path"]), final)
        self.assertFalse(staged.exists())
        index = load_json(final / INDEX)
        self.assertEqual({entry["path"] for entry in index["files"]},
                         {"manifest.json", "logs/prepare.log"})
        self.assertEqual(index["total_bytes"], sum(entry["size_bytes"] for entry in index["files"]))
        complete = load_json(final / COMPLETE)
        self.assertEqual(complete["run_id"], "run-1")
        self.assertEqual(complete["attempt_id"], "attempt-1")
        self.assertEqual(complete["artifacts_sha256"], sha256_file(final / INDEX))
        self.assertEqual(result["artifacts_sha256"], complete["artifacts_sha256"])

    def test_verify_accepts_published_attempt(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n"})
        publish(self.root, "run-1", attempt_id)
        report = verify(self.published())
        self.assertEqual(report["ok"], True)
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["files"], 1)

    def test_republish_identical_content_returns_existing(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n"})
        publish(self.root, "run-1", attempt_id)
        final = self.published()
        staged = attempt_staging(self.root, "run-1", attempt_id)
        shutil.copytree(final, staged)
        result = publish(self.root, "run-1", attempt_id)
        self.assertEqual(result["status"], "existing")
        self.assertEqual(result["path"], str(final))
        self.assertTrue(staged.exists())
        self.assertEqual(load_json(final / INDEX), load_json(staged / INDEX))

    def test_republish_different_content_conflicts(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n"})
        publish(self.root, "run-1", attempt_id)
        staged = attempt_staging(self.root, "run-1", attempt_id)
        staged.mkdir(parents=True)
        (staged / "metrics.json").write_text('{"throughput": 1}\n', encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            publish(self.root, "run-1", attempt_id)
        self.assertIn("conflict", str(caught.exception))

    def test_publish_requires_staging(self):
        with self.assertRaises(ConfigError) as caught:
            publish(self.root, "run-1", "attempt-9")
        self.assertIn("staging attempt not found", str(caught.exception))

    def test_verify_detects_tampered_missing_and_unlisted_files(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n", "logs/run.log": "line\n"})
        publish(self.root, "run-1", attempt_id)
        final = self.published()
        (final / "logs/run.log").write_text("tampered content\n", encoding="utf-8")
        report = verify(final)
        self.assertEqual(report["ok"], False)
        self.assertIn("size mismatch", {problem["reason"] for problem in report["problems"]})
        (final / "logs/run.log").write_text("line\n", encoding="utf-8")
        (final / "extra.bin").write_bytes(b"x")
        report = verify(final)
        self.assertIn("unlisted file", {problem["reason"] for problem in report["problems"]})
        (final / "extra.bin").unlink()
        (final / "metrics.json").unlink()
        report = verify(final)
        self.assertIn("file missing", {problem["reason"] for problem in report["problems"]})
        self.assertEqual(verify(final)["ok"], False)

    def test_verify_detects_index_and_complete_tampering(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n"})
        publish(self.root, "run-1", attempt_id)
        final = self.published()
        index = load_json(final / INDEX)
        index["files"][0]["sha256"] = "0" * 64
        write_json(final / INDEX, index)
        reasons = {problem["reason"] for problem in verify(final)["problems"]}
        self.assertIn("sha256 mismatch", reasons)
        self.assertIn("artifacts_sha256 does not match artifacts.json", reasons)
        complete = load_json(final / COMPLETE)
        complete["run_id"] = "other-run"
        write_json(final / COMPLETE, complete)
        reasons = {problem["reason"] for problem in verify(final)["problems"]}
        self.assertIn("run_id/attempt_id do not match the directory", reasons)
        (final / COMPLETE).unlink()
        self.assertIn("missing COMPLETE marker", {problem["reason"] for problem in verify(final)["problems"]})

    def test_verify_rejects_path_traversal_and_invalid_index(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n"})
        publish(self.root, "run-1", attempt_id)
        final = self.published()
        bad = {"schema_version": "recbench.artifacts/v1", "total_bytes": 0,
               "files": [{"path": "../escape.json", "size_bytes": 0, "sha256": "d" * 64,
                          "media_type": "application/json"}]}
        write_json(final / INDEX, bad)
        report = verify(final)
        self.assertEqual(report["ok"], False)
        self.assertTrue(any(problem["reason"].startswith("invalid artifacts index")
                            for problem in report["problems"]))

    def test_failed_experiment_can_still_be_published(self):
        manifest = {"schema_version": "recbench.run-manifest/v1", "result": {"benchmark": "invalid",
                    "reason": "loss diverged"}}
        staged, attempt_id = self.stage(files={"manifest.json": json.dumps(manifest),
                                                 "logs/measure.log": "error\n"})
        self.assertTrue(staged.is_dir())
        result = publish(self.root, "run-1", attempt_id)
        self.assertEqual(result["status"], "published")
        self.assertEqual(verify(self.published())["ok"], True)

    def test_attempt_ids_skip_published_attempts(self):
        _, attempt_id = self.stage(files={"a.log": "a\n"})
        publish(self.root, "run-1", attempt_id)
        _, next_id = self.stage()
        self.assertEqual(next_id, "attempt-2")

    def test_cli_new_publish_verify(self):
        code, created = self.run_cli("new", "--root", str(self.root), "--run-id", "run-2")
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(created["status"], "created")
        staged = Path(created["path"])
        (staged / "metrics.json").write_text("{}\n", encoding="utf-8")
        code, published = self.run_cli("publish", "--root", str(self.root), "--run-id", "run-2",
                                       "--attempt-id", created["attempt_id"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(published["status"], "published")
        code, report = self.run_cli("verify", published["path"])
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(report["ok"], True)
        (Path(published["path"]) / "metrics.json").write_text("tampered\n", encoding="utf-8")
        code, report = self.run_cli("verify", published["path"])
        self.assertEqual(code, EXIT_BLOCKED)
        self.assertEqual(report["ok"], False)

    def test_cli_reports_conflict_as_usage_error(self):
        _, attempt_id = self.stage(files={"metrics.json": "{}\n"})
        self.run_cli("publish", "--root", str(self.root), "--run-id", "run-1", "--attempt-id", attempt_id)
        staged = attempt_staging(self.root, "run-1", attempt_id)
        staged.mkdir(parents=True)
        (staged / "metrics.json").write_text("different\n", encoding="utf-8")
        code, result = self.run_cli("publish", "--root", str(self.root), "--run-id", "run-1",
                                    "--attempt-id", attempt_id)
        self.assertEqual(code, EXIT_USAGE)
        self.assertEqual(result["status"], "error")
        self.assertIn("conflict", result["error"])


if __name__ == "__main__":
    unittest.main()

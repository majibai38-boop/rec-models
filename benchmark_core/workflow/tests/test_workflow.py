"""Offline contract tests; none of these results prove hardware execution."""
import copy
import datetime as dt
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from benchmark_core.inventory.collector import collect
from benchmark_core.inventory.config import ConfigError, fingerprint, load_config, load_json, resolve_node
from benchmark_core.inventory.report import write_snapshot
from benchmark_core.inventory.tests.test_inventory import HOST, fixture_runner
from benchmark_core.workflow.__main__ import main
from benchmark_core.workflow.contracts import resolve_workflow, validate_workflow
from benchmark_core.workflow.manifest import (
    ARTIFACTS_VERSION, MANIFEST_VERSION, METRICS_VERSION, artifacts_index, new_manifest, set_result,
    transition, validate_artifacts, validate_manifest, validate_metrics, verify_artifacts, write_json,
)
from benchmark_core.workflow.state import (
    EXIT_BLOCKED, EXIT_OK, EXIT_USAGE, DeviceLock, LockBusy, attempt_dir, create_attempt,
    next_attempt_id, record_interrupt, validate_id,
)

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "benchmark_catalog"
NOW = dt.datetime(2026, 10, 29, 8, tzinfo=dt.timezone.utc)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.spec = load_json(CATALOG / "experiments/dlrm-small-pair.json")
        self.recipe = load_json(CATALOG / "environments/cuda.template.json")
        self.raw_config = load_json(CATALOG / "inventory/nodes.example.json")
        for node in self.raw_config["nodes"]:
            node["expected_hostname"] = HOST["hostname"]
            node["expected_devices"] = {}
        self.workflow = load_json(CATALOG / "workflows/dlrm-small-local.json")
        self.workflow.update(benchmark_spec="experiments.json", environment_recipe="cuda.json",
                             inventory_config="nodes.json")
        self.workspace = self.dir / "workspace"
        self.write_documents()

    def write_documents(self):
        write_json(self.dir / "experiments.json", self.spec)
        write_json(self.dir / "cuda.json", self.recipe)
        write_json(self.dir / "nodes.json", self.raw_config)
        write_json(self.dir / "workflow.json", self.workflow)

    def pin(self):
        for group, key in (("workload", "source_sha256"), ("data", "artifact_sha256"),
                           ("initial_state", "checkpoint_sha256"), ("initial_state", "training_state_sha256"),
                           ("correctness", "policy_sha256")):
            self.spec[group][key] = "a" * 64
        self.spec["measurement"]["measure_steps"] = 200
        self.recipe.update(state="pinned", image="example.invalid/synthetic-test@sha256:" + "b" * 64,
                           runtime={"python": "3.12.0", "torch": "2.5.0", "toolkit": "12.4.0", "torch_npu": None})
        self.recipe["compatibility"].update(reviewed_on="2026-10-29",
                                            sources=["https://example.invalid/test-only"],
                                            rationale="Synthetic fixture, no hardware support claim")
        self.recipe["requirements"] += [
            {"field": "host.architecture", "op": "in", "value": ["x86_64"]},
            {"field": "device.model", "op": "eq", "value": "Test GPU A"},
            {"field": "device.driver_version", "op": "version_gte", "value": "550.9"},
        ]
        self.write_documents()

    def snapshot(self):
        config, profiles = load_config(self.dir / "nodes.json")
        node, selected = resolve_node(config, profiles, "gpu-01")
        snapshot = collect(node, selected, config["execution"], runner=fixture_runner, host=HOST)
        snapshot["completed_at"] = NOW.isoformat()
        return write_snapshot(snapshot, self.dir / "snapshot")

    def resolve(self, **kwargs):
        workflow = load_json(self.dir / "workflow.json")
        kwargs.setdefault("workflow_path", self.dir / "workflow.json")
        kwargs.setdefault("entry_workspace", self.workspace)
        kwargs.setdefault("now", NOW)
        return resolve_workflow(workflow, **kwargs)

    def run_cli(self, *argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(list(argv))
        return code, json.loads(buffer.getvalue())

    def test_catalog_workflow_checks_and_defaults_archive_to_entry_workspace(self):
        workflow = load_json(CATALOG / "workflows/dlrm-small-local.json")
        result = resolve_workflow(workflow, workflow_path=CATALOG / "workflows/dlrm-small-local.json",
                                  entry_workspace=self.workspace, now=NOW)
        self.assertEqual(result["resolved"]["target"]["id"], "gpu")
        self.assertEqual(result["resolved"]["recipe"]["id"], "cuda-portable")
        expected = self.workspace.resolve() / "benchmark-results" / "archive"
        self.assertEqual(Path(result["resolved"]["archive"]["root"]), expected)
        self.assertTrue(result["resolved"]["documents"]["benchmark_spec"].endswith("dlrm-small-pair.json"))
        self.assertEqual(result["admission"]["status"], "blocked")

    def test_relative_archive_root_resolves_against_workflow_directory(self):
        self.workflow["archive"]["root"] = "archive-here"
        self.write_documents()
        result = self.resolve()
        self.assertEqual(Path(result["resolved"]["archive"]["root"]), (self.dir / "archive-here").resolve())

    def test_ssh_archive_records_remote_root_and_local_staging(self):
        self.workflow["archive"] = {"transport": "ssh", "host": "bench-archive",
                                    "root": "/srv/rec-benchmark/archive",
                                    "web_base_url": "http://bench-archive:8080"}
        self.write_documents()
        archive = self.resolve()["resolved"]["archive"]
        self.assertEqual(archive["remote_root"], "/srv/rec-benchmark/archive")
        self.assertEqual(archive["host"], "bench-archive")
        self.assertEqual(archive["transport"], "ssh")
        self.assertTrue(Path(archive["staging_root"]).is_absolute())
        self.assertEqual(archive["web_base_url"], "http://bench-archive:8080")

    def test_workflow_rejects_unknown_fields_and_unsupported_execution(self):
        broken = copy.deepcopy(self.workflow)
        broken["extra"] = True
        with self.assertRaises(ConfigError):
            validate_workflow(broken)
        broken = copy.deepcopy(self.workflow)
        broken["execution"] = {"transport": "ssh"}
        with self.assertRaises(ConfigError):
            validate_workflow(broken)
        broken = copy.deepcopy(self.workflow)
        broken["archive"]["host"] = "bench-archive"
        with self.assertRaises(ConfigError):
            validate_workflow(broken)
        broken = copy.deepcopy(self.workflow)
        broken["archive"] = {"transport": "ssh", "host": "bench-archive", "root": "relative/path",
                             "web_base_url": None}
        with self.assertRaises(ConfigError):
            validate_workflow(broken)
        broken = copy.deepcopy(self.workflow)
        broken["archive"] = {"transport": "ssh", "host": "bench-archive", "root": "/srv/../escape",
                             "web_base_url": None}
        with self.assertRaises(ConfigError):
            validate_workflow(broken)
        broken = copy.deepcopy(self.workflow)
        broken["archive"]["web_base_url"] = "http://user:secret@bench-archive:8080"
        with self.assertRaises(ConfigError):
            validate_workflow(broken)
        broken = copy.deepcopy(self.workflow)
        broken["profile"] = {"enabled": "yes"}
        with self.assertRaises(ConfigError):
            validate_workflow(broken)

    def test_unknown_target_lists_available_targets(self):
        self.workflow["target_id"] = "tpu"
        self.write_documents()
        with self.assertRaises(ConfigError) as caught:
            self.resolve()
        self.assertIn("tpu", str(caught.exception))
        self.assertIn("gpu", str(caught.exception))

    def test_recipe_must_match_selected_target_environment(self):
        self.recipe["id"] = "some-other-recipe"
        self.write_documents()
        with self.assertRaises(ConfigError) as caught:
            self.resolve()
        self.assertIn("does not match", str(caught.exception))

    def test_resolve_writes_separate_request_and_single_target_spec(self):
        output = self.dir / "resolution"
        code, summary = self.run_cli("resolve", str(self.dir / "workflow.json"), "--output", str(output),
                                     "--entry-workspace", str(self.workspace))
        self.assertEqual(code, EXIT_BLOCKED)
        self.assertEqual(summary["status"], "blocked")
        request = load_json(output / "request.json")
        resolved = load_json(output / "resolved_workflow.json")
        spec = load_json(output / "resolved_spec.json")
        plan = load_json(output / "plan.json")
        admission = load_json(output / "admission.json")
        self.assertEqual(request["workflow"]["benchmark_spec"], "experiments.json")
        self.assertEqual(len(spec["targets"]), 1)
        self.assertEqual(spec["targets"][0]["id"], "gpu")
        self.assertEqual(plan["spec"]["targets"], spec["targets"])
        self.assertEqual(admission["status"], "blocked")
        self.assertEqual(resolved["workflow_sha256"], request["workflow_sha256"])
        self.assertEqual(resolved["resolved_spec_sha256"], fingerprint(spec))
        stages = {stage["id"]: stage for stage in admission["stages"]}
        self.assertEqual(stages["environment"]["status"], "blocked")
        self.assertIn("draft_environment", {b["code"] for b in stages["environment"]["blockers"]})
        self.assertEqual(stages["structure"]["status"], "eligible")

    def test_resolve_refuses_existing_output_directory(self):
        output = self.dir / "resolution"
        self.run_cli("resolve", str(self.dir / "workflow.json"), "--output", str(output),
                     "--entry-workspace", str(self.workspace))
        code, summary = self.run_cli("resolve", str(self.dir / "workflow.json"), "--output", str(output),
                                     "--entry-workspace", str(self.workspace))
        self.assertEqual(code, EXIT_USAGE)
        self.assertEqual(summary["status"], "error")
        self.assertIn("already exists", summary["error"])

    def test_blockers_reference_only_the_selected_target(self):
        code, summary = self.run_cli("check", str(self.dir / "workflow.json"))
        self.assertEqual(code, EXIT_OK)
        self.assertEqual(summary["stages"]["environment"], "blocked")
        admission = self.resolve()["admission"]
        text = json.dumps(admission)
        self.assertNotIn("npu-01", text)
        self.assertIn("gpu-01", text)

    def test_pinned_recipe_and_single_snapshot_admit_measurement(self):
        self.pin()
        snapshot_path = self.snapshot()
        result = self.resolve(snapshot_paths=[str(snapshot_path)])
        self.assertEqual(result["admission"]["status"], "eligible_for_measurement")
        self.assertEqual(result["plan"]["status"], "eligible_for_runtime_check")
        self.assertEqual(len(result["plan"]["runs"]), 2)
        self.assertEqual([target["id"] for target in result["spec"]["targets"]], ["gpu"])

    def test_missing_hash_and_pilot_steps_are_reported_per_stage(self):
        stages = {stage["id"]: stage for stage in self.resolve()["admission"]["stages"]}
        self.assertEqual({b["code"] for b in stages["workload_state"]["blockers"]}, {"missing_hash"})
        self.assertEqual(len(stages["workload_state"]["blockers"]), 5)
        self.assertEqual({b["code"] for b in stages["pilot"]["blockers"]}, {"missing_measure_steps"})
        self.assertEqual({b["code"] for b in stages["measurement"]["blockers"]}, {"prerequisites_blocked"})

    def test_check_reports_invalid_workflow_with_exit_code(self):
        self.workflow["target_id"] = "tpu"
        self.write_documents()
        code, summary = self.run_cli("check", str(self.dir / "workflow.json"))
        self.assertEqual(code, EXIT_USAGE)
        self.assertEqual(summary["status"], "error")
        self.assertIn("available", summary["error"])


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = {"workflow_sha256": "a" * 64, "spec_sha256": "b" * 64,
                       "resolved_spec_sha256": "c" * 64, "workflow_path": str(self.root / "workflow.json"),
                       "entry_workspace": str(self.root), "hostname": "test-host"}
        self.execution = {"transport": "local", "backend": "cuda", "node": "gpu-01", "device_id": "0"}
        self.archive = {"transport": "local", "host": "localhost", "root": str(self.root / "archive"),
                        "web_base_url": None}

    def manifest(self, **kwargs):
        return new_manifest("run-1", "attempt-1", source=self.source, execution=self.execution,
                            archive=self.archive, created_at=NOW, **kwargs)

    def test_new_manifest_starts_pending_and_round_trips(self):
        value = self.manifest()
        self.assertEqual(value["schema_version"], MANIFEST_VERSION)
        self.assertEqual({phase["state"] for phase in value["phases"].values()}, {"pending"})
        self.assertEqual(value["result"], {"benchmark": "not_measured", "reason": None})
        validate_manifest(json.loads(json.dumps(value)))

    def test_phase_transitions_require_evidence_and_respect_order(self):
        value = self.manifest()
        transition(value, "prepare", "running", at=NOW)
        self.assertIsNotNone(value["phases"]["prepare"]["started_at"])
        with self.assertRaises(ConfigError):
            transition(value, "prepare", "succeeded", at=NOW)
        transition(value, "prepare", "succeeded", at=NOW, log="logs/prepare.log")
        with self.assertRaises(ConfigError):
            transition(value, "prepare", "running", at=NOW)
        with self.assertRaises(ConfigError):
            transition(value, "run", "succeeded", at=NOW, detail="no log")
        with self.assertRaises(ConfigError):
            transition(value, "measure", "failed", at=NOW)
        transition(value, "measure", "blocked", at=NOW, detail="no device")
        transition(value, "measure", "running", at=NOW)
        transition(value, "measure", "interrupted", at=NOW, detail="operator cancelled", exit_code=5)
        self.assertEqual(value["phases"]["measure"]["state"], "interrupted")

    def test_manifest_rejects_naive_timestamps_and_unknown_phases(self):
        value = self.manifest()
        value["created_at"] = "2026-10-29T08:00:00"
        with self.assertRaises(ConfigError):
            validate_manifest(value)
        value = self.manifest()
        value["phases"]["extra"] = value["phases"]["prepare"]
        with self.assertRaises(ConfigError):
            validate_manifest(value)
        value = self.manifest()
        value["phases"]["run"]["state"] = "done"
        with self.assertRaises(ConfigError):
            validate_manifest(value)

    def test_invalid_result_requires_reason(self):
        value = self.manifest()
        with self.assertRaises(ConfigError):
            set_result(value, "invalid")
        set_result(value, "invalid", reason="loss diverged", at=NOW)
        self.assertEqual(value["result"]["benchmark"], "invalid")

    def metric(self, **overrides):
        value = {"name": "throughput", "unit": "samples_per_second", "scope": "end_to_end",
                 "aggregation": "median", "value": 1234.5, "sample_count": 5, "validity": "valid",
                 "reason": None, "source": "measurement"}
        value.update(overrides)
        return {"schema_version": METRICS_VERSION, "metrics": [value]}

    def test_metric_contract_separates_valid_from_unsupported(self):
        validate_metrics(self.metric())
        with self.assertRaises(ConfigError):
            validate_metrics(self.metric(validity="unsupported"))
        validate_metrics(self.metric(validity="unsupported", value=None, reason="backend has no memory counter"))
        with self.assertRaises(ConfigError):
            validate_metrics(self.metric(validity="not_measured", value=1.0, reason="not run"))
        with self.assertRaises(ConfigError):
            validate_metrics(self.metric(aggregation="median", sample_count=None))
        with self.assertRaises(ConfigError):
            validate_metrics(self.metric(value=float("nan")))
        with self.assertRaises(ConfigError):
            validate_metrics({"schema_version": METRICS_VERSION,
                              "metrics": [self.metric()["metrics"][0], self.metric()["metrics"][0]]})

    def test_artifacts_index_and_verification(self):
        attempt = self.root / "runs" / "run-1" / "attempt-1"
        write_json(attempt / "manifest.json", self.manifest())
        (attempt / "logs").mkdir(parents=True)
        (attempt / "logs" / "prepare.log").write_text("prepared\n", encoding="utf-8")
        (attempt / "metrics.json").write_text("{}\n", encoding="utf-8")
        index = artifacts_index(attempt)
        validate_artifacts(index)
        self.assertEqual(verify_artifacts(index, attempt)["ok"], True)
        (attempt / "logs" / "prepare.log").write_text("tampered\n", encoding="utf-8")
        report = verify_artifacts(index, attempt)
        self.assertEqual(report["ok"], False)
        self.assertEqual(report["problems"][0]["reason"], "sha256 mismatch")

    def test_artifacts_reject_escaping_and_reserved_paths(self):
        index = {"schema_version": ARTIFACTS_VERSION, "total_bytes": 0, "files": [
            {"path": "../escape.json", "size_bytes": 0, "sha256": "d" * 64, "media_type": "application/json"}]}
        with self.assertRaises(ConfigError):
            validate_artifacts(index)
        index["files"][0]["path"] = "artifacts.json"
        with self.assertRaises(ConfigError):
            validate_artifacts(index)
        index["files"][0]["path"] = "logs/prepare.log"
        index["total_bytes"] = 0
        index["files"][0]["size_bytes"] = 0
        index["files"].append(dict(index["files"][0]))
        with self.assertRaises(ConfigError):
            validate_artifacts(index)


class StateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_attempt_ids_increment_and_existing_attempts_are_refused(self):
        self.assertEqual(next_attempt_id([]), "attempt-1")
        self.assertEqual(next_attempt_id(["attempt-1", "attempt-7"]), "attempt-8")
        directory, attempt = create_attempt(self.root / "archive", "run-1")
        self.assertEqual(attempt, "attempt-1")
        self.assertEqual(directory, attempt_dir(self.root / "archive", "run-1", "attempt-1"))
        _, second = create_attempt(self.root / "archive", "run-1")
        self.assertEqual(second, "attempt-2")
        with self.assertRaises(ConfigError):
            create_attempt(self.root / "archive", "run-1", "attempt-1")
        with self.assertRaises(ConfigError):
            validate_id("Bad Id", "run_id")

    def test_device_lock_is_exclusive_releasable_and_detects_stale_holders(self):
        state = self.root / "state"
        first = DeviceLock(state, "cuda", "0", command="test")
        first.acquire()
        self.assertEqual(first.holder()["pid"], os.getpid())
        with self.assertRaises(LockBusy):
            DeviceLock(state, "cuda", "0").acquire()
        other = DeviceLock(state, "cuda", "1")
        other.acquire()
        other.release()
        first.release()
        second = DeviceLock(state, "cuda", "0")
        second.acquire()
        self.assertNotEqual(second.holder()["token"], "other")
        second.release()

    def test_stale_lock_requires_age_before_breaking(self):
        state = self.root / "state"
        state.mkdir()
        path = state / "cuda-0.lock"
        path.write_text(json.dumps({"token": "old", "pid": 1, "hostname": "gone",
                                    "acquired_at": "2026-10-29T07:50:00+00:00"}), encoding="utf-8")
        lock = DeviceLock(state, "cuda", "0")
        self.assertFalse(lock.break_stale(older_than_seconds=3600, now=NOW))
        self.assertTrue(lock.break_stale(older_than_seconds=3600, now=NOW + dt.timedelta(hours=24)))
        self.assertFalse(path.exists())

    def test_interruption_record_appends_json_lines(self):
        attempt = self.root / "attempt-1"
        path = record_interrupt(attempt, "run", "device unavailable", at=NOW)
        record_interrupt(attempt, "run", "retried after operator confirmation",
                         exit_code=EXIT_BLOCKED, at=NOW)
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([line["phase"] for line in lines], ["run", "run"])
        self.assertEqual(lines[0]["exit_code"], 5)
        self.assertEqual(lines[1]["exit_code"], EXIT_BLOCKED)
        with self.assertRaises(ConfigError):
            record_interrupt(attempt, "unknown", "bad phase", at=NOW)
        with self.assertRaises(ConfigError):
            record_interrupt(attempt, "run", "bad code", exit_code=0, at=NOW)


if __name__ == "__main__":
    unittest.main()

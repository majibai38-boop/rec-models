"""Offline fixtures only; never evidence of CUDA/Ascend execution or compatibility."""
import copy
import datetime as dt
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

from benchmark_core.inventory.collector import collect
from benchmark_core.inventory.config import ConfigError, load_config, load_json, resolve_node
from benchmark_core.inventory.report import write_snapshot
from benchmark_core.inventory.tests.test_inventory import fixture_runner, HOST
from benchmark_core.planning.__main__ import main
from benchmark_core.planning.contracts import validate_recipe, validate_spec, workload_fingerprint
from benchmark_core.planning.planner import evaluate_rule, plan, write_plan

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "benchmark_catalog"
NOW = dt.datetime(2026, 10, 29, 8, tzinfo=dt.timezone.utc)


class PlanningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = load_json(CATALOG / "experiments/dlrm-small-pair.json")
        self.recipes = [load_json(CATALOG / f"environments/{backend}.template.json") for backend in ("cuda", "ascend")]
        raw_config = load_json(CATALOG / "inventory/nodes.example.json")
        for node in raw_config["nodes"]:
            node["expected_hostname"] = HOST["hostname"]
            node["expected_devices"] = {}
        self.config_path = self.root / "nodes.json"
        self.config_path.write_text(json.dumps(raw_config), encoding="utf-8")
        self.config, self.profiles = load_config(self.config_path)
        self.snapshots, self.paths = [], []
        for node_id in ("gpu-01", "npu-01"):
            node, profiles = resolve_node(self.config, self.profiles, node_id)
            snapshot = collect(node, profiles, self.config["execution"], runner=fixture_runner, host=HOST)
            snapshot["completed_at"] = NOW.isoformat()
            for device in snapshot["accelerators"]:
                if device["kind"] == "npu":
                    device["driver_version"] = "1.2.0"
            self.snapshots.append(snapshot)
            self.paths.append(write_snapshot(snapshot, self.root / node_id))
        self.spec["targets"][1]["device_id"] = "2:0"

    def pin(self):
        for group, key in (("workload", "source_sha256"), ("data", "artifact_sha256"),
                           ("initial_state", "checkpoint_sha256"), ("initial_state", "training_state_sha256"),
                           ("correctness", "policy_sha256")):
            self.spec[group][key] = "a" * 64
        self.spec["measurement"]["measure_steps"] = 200
        for index, recipe in enumerate(self.recipes):
            recipe["state"] = "pinned"
            recipe["image"] = "example.invalid/synthetic-test@sha256:" + str(index) * 64
            recipe["runtime"] = {"python": "3.11.0", "torch": "2.0.0", "toolkit": "1.0.0", "torch_npu": "2.0.0" if index else None}
            recipe["compatibility"].update(reviewed_on="2026-10-29", sources=["https://example.invalid/test-only"], rationale="Synthetic fixture, no hardware support claim")
            recipe["requirements"] += [
                {"field": "host.architecture", "op": "in", "value": ["x86_64"]},
                {"field": "device.model", "op": "eq", "value": "910B3" if index else "Test GPU A"},
                {"field": "device.driver_version", "op": "version_gte", "value": "1.2" if index else "550.9"}]
            if index:
                recipe["requirements"].append({"field": "device.firmware_version", "op": "eq", "value": "7.1"})

    def resolve(self, paths=None, **kwargs):
        return plan(self.spec, self.recipes, self.config, self.profiles,
                    self.paths if paths is None else paths, now=NOW, **kwargs)

    def save_snapshot(self, index=0):
        self.paths[index].write_text(json.dumps(self.snapshots[index]), encoding="utf-8")

    def codes(self, result, target=0):
        return {b["code"] for b in result["targets"][target]["blockers"]}

    def test_templates_validate_but_block_planning(self):
        validate_spec(self.spec)
        for recipe in self.recipes:
            validate_recipe(recipe)
        result = self.resolve(paths=[])
        self.assertEqual(result["status"], "blocked")
        self.assertTrue({"unresolved_workload", "draft_environment", "missing_inventory"} <= self.codes(result))
        self.assertEqual(len(result["runs"]), 4)

    def test_pinned_pair_and_fingerprint_separation(self):
        self.pin()
        result = self.resolve()
        self.assertEqual(result["status"], "eligible_for_runtime_check")
        self.assertFalse(result["execution_ready"])
        a, b, c, _ = result["runs"]
        self.assertEqual(a["workload_sha256"], b["workload_sha256"])
        self.assertNotEqual(a["planned_environment_sha256"], b["planned_environment_sha256"])
        self.assertNotEqual(a["workload_sha256"], c["workload_sha256"])
        self.assertNotEqual(a["plan_key"], b["plan_key"])

    def test_environment_changes_preserve_workload_identity(self):
        self.pin()
        before = self.resolve()
        self.recipes[0]["runtime"]["torch"] = "2.1.0"
        after = self.resolve()
        self.assertEqual(before["runs"][0]["workload_sha256"], after["runs"][0]["workload_sha256"])
        self.assertNotEqual(before["runs"][0]["planned_environment_sha256"], after["runs"][0]["planned_environment_sha256"])

    def test_all_comparison_conditions_affect_hash(self):
        self.pin()
        baseline = workload_fingerprint(self.spec, 256)
        for group, key, value in (("workload", "source_sha256", "b" * 64), ("data", "artifact_sha256", "b" * 64),
                                  ("measurement", "warmup_steps", 51), ("measurement", "scope", "device_only"),
                                  ("initial_state", "training_state_sha256", "b" * 64), ("correctness", "policy_sha256", "b" * 64)):
            changed = copy.deepcopy(self.spec)
            changed[group][key] = value
            self.assertNotEqual(baseline, workload_fingerprint(changed, 256))
        self.spec["name"] = "renamed-experiment"
        self.assertEqual(baseline, workload_fingerprint(self.spec, 256))

    def test_unknown_and_mismatched_requirements_block(self):
        self.pin()
        self.snapshots[0]["accelerators"][0]["driver_version"] = None
        self.save_snapshot()
        self.assertIn("requirement_unknown", self.codes(self.resolve()))
        self.snapshots[0]["accelerators"][0]["driver_version"] = "549.999"
        self.save_snapshot()
        self.assertIn("requirement_mismatch", self.codes(self.resolve()))

    def test_versions_compare_numerically_and_suffixes_remain_unknown(self):
        rule = {"field": "device.driver_version", "op": "version_gte", "value": "550.9"}
        for actual, status in (("550.10", "match"), ("550.8.99", "mismatch"), ("550.9.0", "match"), ("550.10-rc1", "unknown"), (True, "unknown")):
            self.assertEqual(evaluate_rule(rule, {}, {"driver_version": actual})["status"], status)

    def test_expired_future_and_invalid_timestamps_block(self):
        self.pin()
        for timestamp, code in (("2026-10-20T08:00:00+00:00", "inventory_expired"),
                                ("2026-10-30T08:00:00+00:00", "inventory_expired"),
                                ("2026-10-29T08:00:00", "inventory_time_unknown"), (None, "inventory_time_unknown")):
            self.snapshots[0]["completed_at"] = timestamp
            self.save_snapshot()
            self.assertIn(code, self.codes(self.resolve()))

    def test_changed_config_or_profile_blocks_stale_snapshots(self):
        self.pin()
        self.config["nodes"][0]["labels"]["new"] = "value"
        self.assertIn("inventory_stale", self.codes(self.resolve()))
        self.profiles["ascend"]["description"] = "updated profile"
        self.assertIn("inventory_stale", self.codes(self.resolve(), 1))

    def test_partial_snapshot_and_missing_device_block(self):
        self.pin()
        self.snapshots[0]["status"] = "partial"
        self.save_snapshot()
        self.assertIn("inventory_partial", self.codes(self.resolve()))
        self.spec["targets"][1]["device_id"] = "999:0"
        self.assertIn("device_not_observed", self.codes(self.resolve(), 1))

    def test_collector_python_packages_are_not_container_runtime(self):
        self.pin()
        before = self.resolve()
        self.snapshots[0]["software"]["packages"]["torch"] = {"version": "999.0", "status": "installed"}
        self.save_snapshot()
        after = self.resolve()
        self.assertEqual(after["status"], "eligible_for_runtime_check")
        self.assertEqual(before["runs"][0]["planned_environment_sha256"], after["runs"][0]["planned_environment_sha256"])
        self.assertNotEqual(before["targets"][0]["snapshot_sha256"], after["targets"][0]["snapshot_sha256"])

    def test_wrong_references_and_duplicate_inputs_fail(self):
        self.pin()
        with self.assertRaises(ConfigError):
            self.resolve(paths=self.paths + self.paths[:1])
        self.recipes.append(copy.deepcopy(self.recipes[0]))
        with self.assertRaises(ConfigError):
            self.resolve()
        self.recipes.pop()
        self.spec["targets"][0]["environment"] = "ascend-portable"
        with self.assertRaises(ConfigError):
            self.resolve()
        self.spec["targets"][0]["environment"] = "cuda-portable"
        self.spec["targets"][0]["inventory_profile"] = "linux-host"
        with self.assertRaises(ConfigError):
            self.resolve()

    def test_invalid_recipe_rules_and_mutable_images_rejected(self):
        self.pin()
        for modify in (lambda r: r.update(image="repo:latest"),
                       lambda r: r["runtime"].update(torch=">=2.0"),
                       lambda r: r["requirements"].append({"field": "host.architecture", "op": "version_gte", "value": "1.0"}),
                       lambda r: r["requirements"].append({"field": "device.memory_capacity_bytes", "op": "gte", "value": True}),
                       lambda r: r["requirements"].append({"field": "device.unknown", "op": "eq", "value": "x"}),
                       lambda r: r.update(requirements=[]),
                       lambda r: r["compatibility"].update(sources=[])):
            recipe = copy.deepcopy(self.recipes[0])
            modify(recipe)
            with self.assertRaises(ConfigError):
                validate_recipe(recipe)

    def test_unsupported_or_unsafe_measurement_configs_rejected(self):
        for modify in (lambda s: s.update(typo=True), lambda s: s["matrix"].update(batch_size=[True]),
                       lambda s: s["matrix"].update(precision=["fp16"]), lambda s: s["matrix"].update(batch_size=[1, 1]),
                       lambda s: s["measurement"].update(profiler_enabled=True),
                       lambda s: s["measurement"].update(restore_state_before_each_repeat=False),
                       lambda s: s["measurement"].update(step_components=["forward"]),
                       lambda s: s["workload"]["parameters"].update(bad=float("nan"))):
            spec = copy.deepcopy(self.spec)
            modify(spec)
            with self.assertRaises(ConfigError):
                validate_spec(spec)

    def test_plan_roundtrip_is_immutable_and_does_not_overwrite(self):
        self.pin()
        result = self.resolve()
        path = write_plan(result, self.root / "plan")
        self.assertEqual(load_json(path), result)
        self.spec["workload"]["parameters"]["seed"] = 999
        self.assertEqual(result["spec"]["workload"]["parameters"]["seed"], 2026)
        # Conditions must also be frozen independently of caller mutation.
        self.assertEqual(result["runs"][0]["conditions"]["workload"]["parameters"]["seed"], 2026)
        with self.assertRaises(FileExistsError):
            write_plan(result, self.root / "plan")

    def test_cli_draft_plan_exit_and_invalid_json(self):
        spec_path = self.root / "spec.json"
        spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        args = ["plan", "--spec", str(spec_path), "--inventory-config", str(self.config_path),
                "--output", str(self.root / "cli-plan")]
        for backend in ("cuda", "ascend"):
            args += ["--recipe", str(CATALOG / f"environments/{backend}.template.json")]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["check", "spec", str(spec_path)]), 0)
            self.assertEqual(main(args), 3)
            self.assertEqual(main(args), 2)
            spec_path.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
            self.assertEqual(main(["check", "spec", str(spec_path)]), 2)


if __name__ == "__main__":
    unittest.main()

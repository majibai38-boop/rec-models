"""Synthetic command fixtures, not hardware benchmark or real accelerator validation."""
import copy
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from benchmark_core.inventory.__main__ import main
from benchmark_core.inventory.collector import collect
from benchmark_core.inventory.config import CATALOG, CONFIG_VERSION, ConfigError, load_config, load_json, resolve_node, validate_profile
from benchmark_core.inventory.parsers import parse
from benchmark_core.inventory.report import read_snapshot, summarize, write_snapshot
from benchmark_core.inventory.runner import LocalRunner, read_file, run_command


NVIDIA = '0, GPU-a, Test GPU A, 00000000:01:00.0, 40960, 550.54\n2, GPU-b, Test GPU B, 00000000:81:00.0, N/A, 550.54\n'
ASCEND = 'NPU ID    Chip ID    Chip Logic ID    Chip Name\n2         0          0                910B3\n2         1          1                910B3\n9         0          2                910B3\n'
LSCPU = json.dumps({"lscpu": [{"field": k + ":", "data": v} for k, v in {
    "Model name": "Test CPU", "CPU(s)": "64", "Socket(s)": "2", "Core(s) per socket": "16", "NUMA node(s)": "2"}.items()]})
HOST = {"hostname": "test-host", "os": "Linux", "architecture": "x86_64", "cpu_model": None,
        "cpu_logical_count": 64, "memory_total_bytes": None}


def fixture_runner(probe, settings, device=None):
    outputs = {"lscpu": LSCPU, "meminfo": "MemTotal:       1048576 kB\n", "os_release": 'PRETTY_NAME="Test OS"\n',
               "nvidia_csv": NVIDIA, "nvidia_firmware_csv": 'GPU-a, 1.2\nGPU-b, 1.3\n',
               "ascend_mapping": ASCEND, "ascend_board": "Chip Name : 910B3\nPCIe Bus Info : 0000:81:00.0\nFirmware Version : 7.1\n",
               "ascend_memory": "HBM Capacity(MB) : 65536\nDDR Capacity(MB) : 8192\n",
               "ascend_health": "Health : OK\n", "linux_network": '[{"ifname":"eth0","operstate":"UP","mtu":1500}]',
               "raw": "fixture topology", "version": "Version=1.2", "system_model": "Test Server", "bios_version": "1.2"}
    return {"status": "ok", "stdout": outputs[probe["parser"]], "stderr": "", "returncode": 0,
            "argv": probe.get("argv"), "paths": probe.get("paths")}


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "nodes.json"
        self.config_data = {"schema_version": CONFIG_VERSION, "nodes": [{"id": "test", "expected_hostname": "test-host",
                            "expected_os": "Linux", "profiles": ["linux-host", "nvidia"], "expected_devices": {"nvidia": 2}}]}
        self.write_config()

    def write_config(self):
        self.path.write_text(json.dumps(self.config_data), encoding="utf-8")

    def context(self):
        config, profiles = load_config(self.path)
        node, selected = resolve_node(config, profiles, "test")
        return config, profiles, node, selected

    def capture(self, runner=fixture_runner):
        config, profiles, node, selected = self.context()
        return collect(node, selected, config["execution"], runner=runner, host=HOST)

    def test_builtin_profiles_and_example_validate(self):
        config, profiles = load_config(CATALOG / "nodes.example.json")
        self.assertEqual(len(config["nodes"]), 2)
        self.assertEqual(set(profiles), {"linux-host", "windows-host", "nvidia", "ascend"})

    def test_nvidia_inventory_preserves_unknown_and_joins_firmware_by_uuid(self):
        snapshot = self.capture()
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["observed_accelerator_count"], 2)
        self.assertEqual(snapshot["host"]["memory_total_bytes"], 1073741824)
        self.assertEqual(snapshot["host"]["cpu_physical_cores"], 32)
        self.assertEqual(snapshot["accelerators"][0]["memory_capacity_bytes"], 40960 * 1048576)
        self.assertIsNone(snapshot["accelerators"][1]["memory_capacity_bytes"])
        self.assertEqual(snapshot["accelerators"][1]["firmware_version"], "1.3")
        self.assertEqual(snapshot["accelerators"][1]["sources"]["firmware_version"], "nvidia.firmware")
        self.assertEqual(snapshot["benchmark_readiness"], "not_validated")

    def test_discovery_failure_is_unknown_not_zero(self):
        def fail(probe, settings, device=None):
            if probe["parser"] == "nvidia_csv":
                return {"status": "unavailable", "stdout": "", "stderr": "not found"}
            if probe["parser"] == "nvidia_firmware_csv":
                return {"status": "unavailable", "stdout": "", "stderr": "not found"}
            return fixture_runner(probe, settings, device)
        snapshot = self.capture(fail)
        self.assertEqual(snapshot["status"], "partial")
        self.assertIsNone(snapshot["observed_accelerator_count"])
        self.assertIsNone(snapshot["backends"]["nvidia"]["observed_count"])
        self.assertEqual(snapshot["detected_accelerator_count"], 0)

    def test_malformed_or_duplicate_discovery_keeps_evidence_without_devices(self):
        for output in ("bad header", NVIDIA + NVIDIA.splitlines()[0] + "\n"):
            with self.subTest(output=output):
                def run(probe, settings, device=None):
                    result = fixture_runner(probe, settings, device)
                    if probe["parser"] == "nvidia_csv":
                        result["stdout"] = output
                    return result
                snapshot = self.capture(run)
                self.assertEqual(snapshot["status"], "partial")
                self.assertIsNone(snapshot["observed_accelerator_count"])
                record = next(e for e in snapshot["evidence"] if e["id"] == "nvidia.devices")
                self.assertEqual(record["status"], "parse_error")
                self.assertEqual(record["stdout"], output)

    def test_expected_count_never_replaces_observed_count(self):
        self.config_data["nodes"][0]["expected_devices"]["nvidia"] = 8
        self.write_config()
        snapshot = self.capture()
        self.assertEqual(snapshot["observed_accelerator_count"], 2)
        self.assertEqual(snapshot["status"], "partial")

    def test_ascend_multiple_chips_are_distinct_and_memory_tiers_not_summed(self):
        node = self.config_data["nodes"][0]
        node["profiles"] = ["linux-host", "ascend"]
        node["expected_devices"] = {"ascend": 3}
        self.write_config()
        snapshot = self.capture()
        self.assertEqual(snapshot["observed_accelerator_count"], 3)
        self.assertEqual([d["id"] for d in snapshot["accelerators"]], ["2:0", "2:1", "9:0"])
        self.assertEqual(snapshot["accelerators"][0]["memory_tiers"]["hbm"]["unit"], "MB")
        self.assertIsNone(snapshot["accelerators"][0]["memory_capacity_bytes"])
        self.assertEqual(snapshot["accelerators"][2]["sources"]["firmware_version"], "ascend.board.9:0")

    def test_ascend_unrecognized_header_and_missing_device_rows_fail(self):
        for output in ("NPU Chip Model\n0 0 910B3", "NPU ID Chip ID Chip Logic ID Chip Name\n"):
            with self.assertRaises(ValueError):
                parse("ascend_mapping", output)
        self.assertEqual(parse("ascend_mapping", "Total : 0 NPU in system")["devices"], [])

    def test_host_guard_rejects_control_machine_before_any_probe(self):
        config, profiles, node, selected = self.context()
        for hostname, os_name in (("wrong-host", "Linux"), ("test-host", "Windows")):
            with self.assertRaises(ConfigError):
                collect(node, selected, config["execution"], runner=lambda *_: self.fail("probe executed"),
                        host=dict(HOST, hostname=hostname, os=os_name))

    def test_optional_failure_warns_but_required_failure_is_partial(self):
        for parser_name, expected in (("version", "complete_with_warnings"), ("meminfo", "partial")):
            def run(probe, settings, device=None):
                if probe["parser"] == parser_name:
                    return {"status": "timeout", "stdout": "", "stderr": ""}
                return fixture_runner(probe, settings, device)
            self.assertEqual(self.capture(run)["status"], expected)

    def test_fingerprint_excludes_collection_time_and_tracks_driver_change(self):
        first, second = self.capture(), self.capture()
        self.assertEqual(first["observed_configuration_sha256"], second["observed_configuration_sha256"])
        def run(probe, settings, device=None):
            result = fixture_runner(probe, settings, device)
            result["stdout"] = result["stdout"].replace("550.54", "560.12")
            return result
        self.assertNotEqual(first["observed_configuration_sha256"], self.capture(run)["observed_configuration_sha256"])

    def test_toolkit_version_reports_participate_in_configuration_fingerprint(self):
        first = self.capture()
        def run(probe, settings, device=None):
            result = fixture_runner(probe, settings, device)
            if probe["parser"] == "version":
                result["stdout"] = "Toolkit version=updated"
            return result
        second = self.capture(run)
        self.assertNotEqual(first["observed_configuration_sha256"], second["observed_configuration_sha256"])
        self.assertEqual(second["software"]["version_reports"]["nvidia.cuda-toolkit"]["source"], "nvidia.cuda-toolkit")

    def test_strict_configuration_rejects_typos_duplicates_and_wrong_types(self):
        cases = []
        typo = copy.deepcopy(self.config_data); typo["execuion"] = {}; cases.append(typo)
        duplicate = copy.deepcopy(self.config_data); duplicate["nodes"] *= 2; cases.append(duplicate)
        boolean = copy.deepcopy(self.config_data); boolean["execution"] = {"timeout_seconds": True}; cases.append(boolean)
        malformed = copy.deepcopy(self.config_data); malformed["nodes"][0]["expected_os"] = []; cases.append(malformed)
        for data in cases:
            self.path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(self.path)
        self.path.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_json(self.path)

    def test_custom_profile_can_override_command_and_changes_profile_hash(self):
        before = self.capture()
        profile = load_json(CATALOG / "profiles/nvidia.json")
        profile["probes"][0]["argv"][0] = "/opt/vendor/nvidia-smi"
        (self.root / "custom.json").write_text(json.dumps(profile), encoding="utf-8")
        self.config_data["profile_files"] = ["custom.json"]
        self.write_config()
        after = self.capture()
        self.assertNotEqual(before["profiles_sha256"], after["profiles_sha256"])
        self.assertEqual(after["resolved_profiles"][1]["probes"][0]["argv"][0], "/opt/vendor/nvidia-smi")

    def test_json_device_plugin_validates_units_and_identities(self):
        result = parse("json_devices", '[{"id":"0","model":"New XPU","memory_capacity_bytes":1024}]')
        self.assertEqual(result["devices"][0]["physical_id"], "0")
        for invalid in ('[{"id":"0","model":"X","memory_capacity_bytes":true}]',
                        '[{"id":"0","model":"X","unknown":2}]'):
            with self.assertRaises(ValueError):
                parse("json_devices", invalid)

    def test_snapshot_and_fleet_missing_stale_duplicate_detection(self):
        snapshot = self.capture()
        path = write_snapshot(snapshot, self.root / "capture")
        self.assertEqual(read_snapshot(path)["status"], "complete")
        with self.assertRaises(FileExistsError):
            write_snapshot(snapshot, self.root / "capture")
        config, profiles, _, _ = self.context()
        self.assertEqual(summarize([path], config, profiles)["status"], "complete")
        config["nodes"].append(dict(config["nodes"][0], id="missing"))
        summary = summarize([path], config, profiles)
        self.assertEqual(summary["nodes"][1]["status"], "not_collected")
        config["nodes"][0]["expected_devices"]["nvidia"] = 4
        self.assertEqual(summarize([path], config, profiles)["nodes"][0]["status"], "stale")
        with self.assertRaises(ConfigError):
            summarize([path, path])

    def test_unknown_discovery_summary_is_not_a_zero_device_success(self):
        snapshot = self.capture()
        snapshot["backends"]["nvidia"]["observed_count"] = 9
        path = write_snapshot(snapshot, self.root / "bad")
        with self.assertRaises(ConfigError):
            read_snapshot(path)

    def test_cli_check_and_plan_do_not_run_probes(self):
        with patch("benchmark_core.inventory.__main__.collect", side_effect=AssertionError("called")):
            for command in (("check",), ("plan", "--node", "test")):
                with redirect_stdout(io.StringIO()) as output:
                    status = main([*command, "--config", str(self.path)])
                self.assertEqual(status, 0, output.getvalue())

    def test_device_limit_does_not_silently_truncate_inventory(self):
        self.config_data["execution"] = {"max_devices": 1}
        self.write_config()
        snapshot = self.capture()
        self.assertEqual(snapshot["status"], "partial")
        self.assertIsNone(snapshot["observed_accelerator_count"])
        self.assertEqual(snapshot["accelerators"], [])

    def test_new_vendor_profile_uses_json_without_changing_collector(self):
        profile = {"schema_version": "recbench.inventory-profile/v1", "id": "new-xpu", "kind": "accelerator",
                   "vendor": "ExampleVendor", "platforms": ["Linux"], "probes": [
                       {"id": "devices", "argv": ["vendor-inventory", "--json"], "parser": "json_devices", "required": True}]}
        (self.root / "new.json").write_text(json.dumps(profile), encoding="utf-8")
        self.config_data["profile_files"] = ["new.json"]
        self.config_data["nodes"][0]["profiles"] = ["linux-host", "new-xpu"]
        self.config_data["nodes"][0]["expected_devices"] = {"new-xpu": 1}
        self.write_config()
        def run(probe, settings, device=None):
            if probe["parser"] == "json_devices":
                return {"status": "ok", "stdout": '[{"id":"3","model":"Test XPU","memory_capacity_bytes":1024}]', "stderr": ""}
            return fixture_runner(probe, settings, device)
        snapshot = self.capture(run)
        self.assertEqual(snapshot["status"], "complete")
        self.assertEqual(snapshot["accelerators"][0]["vendor"], "ExampleVendor")

    def test_windows_cim_and_single_adapter_array(self):
        host = parse("windows_host", json.dumps({"cpu_model": "Test CPU", "cpu_sockets": 1,
                     "cpu_physical_cores": 8, "cpu_logical_count": 16, "memory_total_bytes": 32768}))
        self.assertEqual(host["host"]["cpu_physical_cores"], 8)
        network = parse("windows_network", '[{"Name":"NIC","Speed":"1000000000","NetConnectionStatus":2}]')
        self.assertEqual(network["host"]["network_interfaces"][0]["speed_bits_per_second"], 1000000000)
        with self.assertRaises(ValueError):
            parse("windows_network", '{"Name":"NIC"}')

    def test_invalid_firmware_association_never_partially_updates_devices(self):
        def run(probe, settings, device=None):
            result = fixture_runner(probe, settings, device)
            if probe["parser"] == "nvidia_firmware_csv":
                result["stdout"] = 'GPU-a, 9.0\nGPU-not-present, 8.0\n'
            return result
        snapshot = self.capture(run)
        self.assertEqual(snapshot["status"], "complete_with_warnings")
        self.assertIsNone(snapshot["accelerators"][0]["firmware_version"])

    def test_invalid_total_or_complete_unknown_snapshot_is_rejected(self):
        for index in range(2):
            snapshot = self.capture()
            if index == 0:
                snapshot["observed_accelerator_count"] = 99
            else:
                snapshot["backends"]["nvidia"]["observed_count"] = None
            path = write_snapshot(snapshot, self.root / f"invalid-{index}")
            with self.assertRaises(ConfigError):
                read_snapshot(path)

    def test_cli_init_no_overwrite_and_partial_capture_exit_code(self):
        output_path = self.root / "init.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["init", "--output", str(output_path)]), 0)
            self.assertEqual(main(["init", "--output", str(output_path)]), 2)
        snapshot = self.capture()
        snapshot["status"] = "partial"
        with patch("benchmark_core.inventory.__main__.collect", return_value=snapshot), redirect_stdout(io.StringIO()):
            status = main(["collect", "--config", str(self.path), "--node", "test", "--output", str(self.root / "partial")])
        self.assertEqual(status, 3)
        self.assertTrue((self.root / "partial/snapshot.json").is_file())


class RunnerTest(unittest.TestCase):
    def test_missing_command_is_explicit(self):
        result = run_command(["recbench-nonexistent-tool-1234"], 1, 1024)
        self.assertEqual(result["status"], "unavailable")

    def test_shell_metacharacters_are_literal_arguments(self):
        value = 'a; echo unexpected && $(whoami) `x`'
        result = run_command([sys.executable, "-c", "import sys; print(sys.argv[1])", value], 5, 1024)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stdout"].strip(), value)

    def test_timeout_and_output_limit(self):
        result = run_command([sys.executable, "-c", "import time; time.sleep(5)"], .15, 1024)
        self.assertEqual(result["status"], "timeout")
        result = run_command([sys.executable, "-c", "print('x'*100000)"], 5, 1024)
        self.assertEqual(result["status"], "output_limit")
        self.assertLessEqual(len(result["stdout"].encode()), 1024)

    def test_file_fallback_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "version.info"
            path.write_text("Version=1.2", encoding="utf-8")
            result = read_file([str(path) + ".missing", str(path)], 1024)
            self.assertEqual(result["selected_path"], str(path))
            self.assertEqual(read_file([str(path)], 3)["status"], "output_limit")

    def test_chip_template_requires_observed_chip_id(self):
        result = LocalRunner()({"argv": ["npu-smi", "-c", "{chip_id}"]},
                               {"timeout_seconds": 1, "max_output_bytes": 1024}, {"chip_id": None})
        self.assertEqual(result["status"], "error")


if __name__ == "__main__":
    unittest.main()

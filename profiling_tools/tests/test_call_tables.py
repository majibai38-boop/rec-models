import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from xml.etree import ElementTree

from profiling_tools.call_candidates import child_candidates
from profiling_tools.call_tables import CallEvidence, InvalidResults, decisions, export_tables
from profiling_tools.compare import compare_profiles
from profiling_tools.model import OpMatch
from profiling_tools.result_api import read_json
from profiling_tools.tests.test_compare import complete
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.tests.test_manual_mapping import nested_fixture


class CallTablesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_ten_steps_nested_wrapper_all_templates_and_replay(self):
        cfg = nested_fixture(self.root, count=260, templates=26, steps=10)
        compare_profiles(cfg)
        ev = CallEvidence(cfg.output_dir)
        group, = ev.query(op_type="IndexAdd")["items"]
        self.assertEqual(group["unique_type_candidates"], 260)
        last = ev.query(group_id=group["group_id"], offset=259)["items"][0]
        events = ev.query(call_id=last["call_id"])
        self.assertEqual(events["npu_step"], events["gpu_step"])
        self.assertEqual(json.loads(events["items"][0]["Owner OP Path"]), ["aten::index_add_", "aten::index_add"])
        self.assertEqual(ev.query(unmapped_gpu=True)["total"], 260)
        output = self.root / "tables"
        report = export_tables(cfg.output_dir, output)
        self.assertEqual(report["decisions"], {"type": 260})
        saved = read_json(output / "call_tables.json")
        row = saved["types"][0]
        self.assertEqual((row["引擎可比时间覆盖率(%)"], row["Comparable Time Coverage(%)"]), (0, 100))
        self.assertEqual(row["Comparable Ratio(GPU/NPU)"], .5)
        self.assertEqual(saved["types"][1]["GPU Total Time(us)"], 0)
        self.assertEqual(saved["types"][-1]["GPU Total Time(us)"], 520)
        with zipfile.ZipFile(output / "call_tables.xlsx") as archive:
            for name in archive.namelist():
                if name.endswith(".xml"):
                    ElementTree.fromstring(archive.read(name))
        plan = {"run_id": ev.run_id, "proposals": [{"level": "call", "reason": "compare outer call only",
                                                     "evidence_id": group["group_id"]}]}
        planpath = self.root / "plan.json"
        planpath.write_text(json.dumps(plan))
        selected = self.root / "selected"
        export_tables(cfg.output_dir, selected, planpath)
        replay = self.root / "replay"
        export_tables(cfg.output_dir, replay, selected / "call_tables.json")
        self.assertEqual((replay / "call_tables.json").read_bytes(), (selected / "call_tables.json").read_bytes())
        self.assertEqual(read_json(replay / "call_tables.json")["types"][0]["Comparable Time Coverage(%)"], 0)
        with self.assertRaisesRegex(ValueError, "run_id"):
            decisions(ev, {**plan, "run_id": "another-run"})
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            decisions(ev, {"run_id": ev.run_id, "proposals": [{**plan["proposals"][0], "evidence_id": "invented"}]})
        with self.assertRaises(ValueError):
            export_tables(cfg.output_dir, output)
        (Path(cfg.output_dir) / "whole_call_mapping.csv").write_text("broken", encoding="utf-8")
        with self.assertRaises(InvalidResults):
            export_tables(cfg.output_dir, self.root / "bad")
        self.assertFalse((self.root / "bad").exists())

    def test_transpose_under_two_callers_and_shared_gpu(self):
        cfg = nested_fixture(self.root, count=2)
        for side in ("npu", "gpu"):
            path = self.root / (side + ".json")
            events = read_json(path)["traceEvents"]
            for e in events:
                if e["name"] == "aten::index_add_":
                    e["name"] = "aten::copy_" if e["ts"] < 100 else "aten::mm"
            if side == "npu":
                for i in range(2):
                    events.append(complete("transpose", "kernel", 36 + i*100, 2, 4, 7,
                                           {"External id": 2 + 2*i, "stream": 7}))
            path.write_text(json.dumps({"traceEvents": events}))
        with (self.root / "details.csv").open("a", encoding="utf-8") as f:
            f.write("transpose,Transpose,2,36,7,0,1\ntranspose,Transpose,2,136,7,0,1\n")
        compare_profiles(cfg)
        ev = CallEvidence(cfg.output_dir)
        groups = ev.query(op_type="Transpose")["items"]
        self.assertEqual({g["op"] for g in groups}, {"aten::copy_", "aten::mm"})
        plan = {"run_id":ev.run_id, "proposals":[{"evidence_id": g["group_id"], "level":"type",
                "reason":"test shared GPU event"} for g in groups]}
        self.assertEqual({r["level"] for r in decisions(ev, plan)}, {"call"})
        out = self.root / "tables"
        export_tables(cfg.output_dir, out)
        report = read_json(out / "call_tables.json")
        self.assertEqual(sum(r["可比 GPU kernel(us)"] for r in report["groups"]), 4)
        row = next(r for r in report["types"] if r["OP Type"] == "Transpose")
        self.assertEqual(row["新增可比 NPU(us)"], 0)
        self.assertIn("aten::copy_", row["调用关系分析"])

    def test_missing_metadata_and_cross_step_are_pending(self):
        cfg = nested_fixture(self.root, count=4, steps=2)
        path = self.root / "gpu.json"
        events = read_json(path)
        for e in events["traceEvents"]:
            if e["name"] == "ProfilerStep#2":
                e["name"] = "ProfilerStep#99"
            if e["name"] == "aten::index_add_" and e["ts"] == 10:
                del e["args"]["Input type"]
        path.write_text(json.dumps(events))
        compare_profiles(cfg)
        records = decisions(CallEvidence(cfg.output_dir))
        self.assertEqual(sum(r["level"] == "type" for r in records), 1)
        self.assertTrue(all(r["level"] == "review" for r in records if r["gpu_step"].endswith("#99")))

    def test_structural_candidate_requires_unique_child_and_parent_match(self):
        def make(side, duplicate=False):
            ops = [op("aten::outer", 0), op("aten::child", 1, parent=0)]
            if duplicate:
                ops.append(op("aten::child", 2, parent=0))
            p = profile(side, ops, [(1, "k")])
            p.kernels[0].link_method = "external_id"
            return p
        base = [OpMatch(1, 0, 0, 1, "high")]
        rows = [{"NPU Kernel Index":0, "Comparable":False}]
        gpu, npu = make("gpu"), make("npu")
        candidate, = child_candidates(gpu, npu, base, rows, {})
        self.assertTrue(candidate["Comparable"])
        self.assertEqual(candidate["Parent Anchor"]["match_id"], 1)
        self.assertEqual(child_candidates(make("gpu", True), npu, base, rows, {}), [])
        self.assertEqual(child_candidates(gpu, npu, [], rows, {}), [])

    def test_no_cpu_leaves_gpu_in_remainder_and_type_ratio_blank(self):
        cfg = nested_fixture(self.root, count=4)
        for side in ("gpu", "npu"):
            path = self.root / (side + ".json")
            events = [e for e in read_json(path)["traceEvents"] if e["cat"] != "cpu_op"]
            for i, e in enumerate(events):
                if e["cat"] == "kernel" and side == "gpu":
                    e["args"]["stream"] = 7 + i % 2
            path.write_text(json.dumps({"traceEvents": events}))
        compare_profiles(cfg)
        out = self.root / "tables"
        export_tables(cfg.output_dir, out)
        data = read_json(out / "call_tables.json")
        self.assertEqual(data["groups"], [])
        self.assertIsNone(data["types"][0]["Comparable Ratio(GPU/NPU)"])
        self.assertEqual(data["types"][1]["GPU Kernel Count"], 4)
        self.assertIn("缺少已验证", data["types"][0]["调用关系分析"])

    def test_agent_selects_one_overlapping_view_and_manual_conflict_blocks_it(self):
        from profiling_tools.result_api import read_csv
        cfg = nested_fixture(self.root)
        compare_profiles(cfg)
        root = Path(cfg.output_dir)
        row, = read_csv(root / "whole_call_mapping.csv")
        duplicate = {**row, "Call ID": "alternative", "Source": "unique_child_of_matched_parent"}
        (root / "call_candidates.json").write_text(json.dumps([duplicate]))
        ev = CallEvidence(root)
        self.assertEqual([r["level"] for r in decisions(ev)], ["review", "review"])
        plan = {"run_id": ev.run_id, "proposals": [{"evidence_id": "alternative", "reason": "verified child",
                                                    "level": "type"}]}
        self.assertEqual([r["level"] for r in decisions(ev, plan)], ["review", "type"])
        ev.manual_users["gpu", 0].add(999)
        self.assertFalse(ev._call(duplicate)["verified"])

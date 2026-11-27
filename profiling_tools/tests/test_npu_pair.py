import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from profiling_tools.npu_pair import compare_npus, inspect_side


class PairTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def details(self, name, rows):
        p = self.root/name
        with p.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(("Step Id", "Device_id", "Type", "Duration(us)", "Input Shapes"))
            w.writerows(rows)
        return p

    def test_counts_and_workload_shift(self):
        a = self.details("a.csv", [(1, 7, "Add", 3, "2;3"), (1, 7, "Cast", 4, ""), ("", 7, "Add", 90, "")])
        b = self.details("b.csv", [(9, 0, "Add", 2, "2;3"), (9, 0, "Add", 2, "4;3"), (9, 0, "Fill", 0, "")])
        out = self.root/"result"
        r = compare_npus(a, b, out, baseline_step="ProfilerStep#1", candidate_step="9")
        types = {v["OP Type"]: v for v in r["types"]}
        self.assertEqual(r["overall"]["基线耗时(us)"], 7)
        self.assertEqual(r["overall"]["耗时比(对比/基线)"], 4/7)
        self.assertEqual(r["baseline"]["excluded_unmarked"], 1)
        self.assertEqual(types["Add"]["耗时比(对比/基线)"], 4/3)
        self.assertEqual(types["Add"]["对比平均(us)"], 2)
        self.assertIsNone(types["Cast"]["耗时比(对比/基线)"])
        self.assertEqual(types["Fill"]["状态"], "仅对比")
        self.assertEqual(len(r["signatures"]), 4)
        self.assertEqual(json.loads((out/"npu_pair.json").read_text(encoding="utf8")), r)
        with zipfile.ZipFile(out/"npu_pair.xlsx") as z:
            self.assertIsNone(z.testzip())
        with self.assertRaisesRegex(ValueError, "空目录"):
            compare_npus(a,b,out,all_steps=True)

    def test_require_scope_and_valid_durations(self):
        a = self.details("a.csv", [("", 0, "A", 0, ""), (2, 1, "A", 5, "")])
        self.assertEqual(inspect_side(a,"基线")["devices"], {"0":{"":1}, "1":{"2":1}})
        for opts in [{}, {"all_steps":True}, {"baseline_step":"5","candidate_step":"2","baseline_device":"0"}]:
            with self.assertRaises(ValueError):
                compare_npus(a,a,self.root/"bad",**opts)
        r = compare_npus(a,a,self.root/"out",all_steps=True,baseline_device="0",candidate_device="1")
        self.assertIsNone(r["overall"]["耗时比(对比/基线)"])
        for value in ("NaN", -1, "", "invalid"):
            b = self.details("b.csv", [("", 0, "A", value, "")])
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                inspect_side(b, "对比")
        self.assertFalse((self.root/"bad").exists())

    def test_core_scopes_colors_and_time_shares(self):
        for side, durations in [('a',[10]*6), ('b',[20,21,11,12,50,40])]:
            with (self.root/side).open('w',newline='',encoding='utf8') as f:
                w=csv.writer(f)
                w.writerow(('Type','Accelerator Core','aicore_time(us)','Duration(us)'))
                for (op,core,counter),t in zip([('A','AI_CORE',1),('A','MIX_AIC',3),('V','AI_VECTOR_CORE',0),
                          ('W','AI_CORE',0),('M','AI_CORE',0),('M','AI_CORE',None)], durations):
                    w.writerow((op,core,counter,t))
        report=compare_npus(self.root/'a',self.root/'b',self.root/'result',all_steps=True)
        groups={(r['OP Type'],r['Core Type']): (r['CV'],r['ratio_color']) for r in report['types']}
        self.assertEqual(groups,{('A','AI_CORE'):('Cube','yellow'),('A','MIX_AIC'):('Cube','green'),
                        ('V','AI_VECTOR_CORE'):('Vector','yellow'),('W','AI_CORE'):('Vector','green'),
                        ('M','AI_CORE'):('Mixed','yellow')})
        self.assertEqual(report['baseline']['cv']['Vector']['time_pct'],50)
        self.assertEqual(report['candidate']['cv']['Unknown']['time_us'],40)

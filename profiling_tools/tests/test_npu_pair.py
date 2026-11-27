import csv
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from profiling_tools.npu_pair import (below_threshold, compare_npus, dominant_name, inspect_side,
                                      ratio_color, validate_thresholds, write_below_threshold_xlsx)


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
        self.assertEqual(r["overall"]["性能比(基线/对比)"], 7/4)
        self.assertEqual(r["baseline"]["excluded_unmarked"], 1)
        self.assertEqual(types["Add"]["性能比(基线/对比)"], 3/4)
        self.assertEqual(types["Add"]["变化(%)"], -25)
        self.assertEqual(types["Add"]["对比平均(us)"], 2)
        self.assertIsNone(types["Cast"]["性能比(基线/对比)"])
        self.assertEqual(types["Fill"]["状态"], "仅对比")
        self.assertEqual(len(r["signatures"]), 4)
        kinds = {item["kind"]: item["text"] for item in r["conclusions"]}
        self.assertIn("1.750", kinds["总体"])          # 7 / 4
        self.assertIn("aicore_time", kinds["结构"])     # no counter columns in this CSV
        self.assertIn("不做流水初筛", kinds["流水"])    # no PipeUtilization columns either
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
        self.assertIsNone(r["overall"]["性能比(基线/对比)"])
        for value in ("NaN", -1, "", "invalid"):
            b = self.details("b.csv", [("", 0, "A", value, "")])
            with self.assertRaisesRegex(ValueError, "第 2 行"):
                inspect_side(b, "对比")
        self.assertFalse((self.root/"bad").exists())

    def test_pipeline_screen_feeds_the_conclusion(self):
        header = ("Step Id", "Device_id", "Type", "Accelerator Core", "Duration(us)",
                  "aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
                  "aic_fixpipe_ratio", "aiv_vec_ratio", "aiv_scalar_ratio",
                  "aiv_mte2_ratio", "aiv_mte3_ratio")

        def cube(mac, mte2):
            return {'aic_mac_ratio': mac, 'aic_scalar_ratio': .1, 'aic_mte1_ratio': .1,
                    'aic_mte2_ratio': mte2, 'aic_fixpipe_ratio': .1}

        def vector(scalar, mte2=.1):
            return {'aiv_vec_ratio': .1, 'aiv_scalar_ratio': scalar,
                    'aiv_mte2_ratio': mte2, 'aiv_mte3_ratio': .05}

        # Baseline: cube-bound MatMul plus a memory-bound Add. Candidate: MatMul moves to
        # mte2 (memory) while getting faster; Add turns scalar-bound and slows down.
        sides = {'a': [('MatMul', 'AI_CORE', 100, {**cube(.9, .2), **vector(.1)}),
                       ('Add', 'AI_VECTOR_CORE', 30, {**cube(.1, .1), **vector(.1, .95)})],
                 'b': [('MatMul', 'AI_CORE', 60, {**cube(.2, .95), **vector(.1)}),
                       ('Add', 'AI_VECTOR_CORE', 45, {**cube(.1, .1), **vector(.9)})]}
        for side, data in sides.items():
            with (self.root/side).open('w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for op, core, duration, ratios in data:
                    writer.writerow((1, 0, op, core, duration,
                                     *(ratios.get(name, '') for name in header[5:])))
        report = compare_npus(self.root/'a', self.root/'b', self.root/'result', all_steps=True)

        def shares(side):
            return {c['category']: c['time_us'] for c in report[side]['pipeline']['categories']}
        self.assertEqual(report['pipeline']['method'], 'pipe_utilization_v2')
        self.assertEqual(shares('baseline'), {'compute': 100, 'memory': 30})
        self.assertEqual(shares('candidate'), {'memory': 60, 'scalar': 45})

        screen = {(row['侧'], row['OP Type']): row for row in
                  report['pipeline']['baseline_types'] + report['pipeline']['candidate_types']}
        self.assertEqual(screen[('基线', 'MatMul')]['流水类别'], 'compute')
        self.assertEqual(screen[('对比', 'MatMul')]['流水类别'], 'memory')
        self.assertEqual(screen[('对比', 'Add')]['该类型内占比(%)'], 100)

        kinds = {item['kind']: item['text'] for item in report['conclusions']}
        self.assertIn('1.238', kinds['总体'])
        self.assertIn('搬运主导候选', kinds['流水'])
        self.assertEqual(json.loads((self.root/'result'/'npu_pair.json').read_text(encoding='utf8')), report)
        with zipfile.ZipFile(self.root/'result'/'npu_pair.xlsx') as archive:
            workbook = archive.read('xl/workbook.xml').decode()
        for sheet in ('结论', '流水总览', '流水-类型'):
            self.assertIn(sheet, workbook)

    def test_core_scopes_colors_and_time_shares(self):
        for side, durations in [('a',[10]*6), ('b',[5,4,9,12,3,4])]:
            with (self.root/side).open('w',newline='',encoding='utf8') as f:
                w=csv.writer(f)
                w.writerow(('Type','Accelerator Core','aicore_time(us)','Duration(us)'))
                for (op,core,counter),t in zip([('A','AI_CORE',1),('A','MIX_AIC',3),('V','AI_VECTOR_CORE',0),
                          ('W','AI_CORE',0),('M','AI_CORE',0),('M','AI_CORE',None)], durations):
                    w.writerow((op,core,counter,t))
        report=compare_npus(self.root/'a',self.root/'b',self.root/'result',all_steps=True)
        groups={(r['OP Type'],r['Core Type']): (r['CV'],r['ratio_color']) for r in report['types']}
        # Default thresholds (Cube 2.0, Vector 1.1): A/MIX_AIC is a 2.5x speedup but stays in the
        # 1.0x-1.5x band, W is a slow-down, and Mixed has no threshold to apply.
        self.assertEqual(groups,{('A','AI_CORE'):('Cube','green'),('A','MIX_AIC'):('Cube','green'),
                        ('V','AI_VECTOR_CORE'):('Vector','green'),('W','AI_CORE'):('Vector','yellow'),
                        ('M','AI_CORE'):('Mixed',None)})
        self.assertEqual(report['thresholds']['cube'],2.0)
        self.assertEqual(report['thresholds']['vector'],1.1)
        self.assertEqual(report['thresholds']['band'],0.5)
        self.assertEqual(report['baseline']['cv']['Vector']['time_pct'],50)
        self.assertEqual(report['candidate']['cv']['Unknown']['time_us'],4)

    def test_thresholds_split_each_class_into_four_colours(self):
        limits = validate_thresholds(2.0, 1.1)
        self.assertEqual(ratio_color(3.0, 'Cube', limits), 'dark-green')     # exactly 1.5x
        self.assertEqual(ratio_color(2.99, 'Cube', limits), 'green')
        self.assertEqual(ratio_color(1.0, 'Cube', limits), 'yellow')         # exactly 0.5x
        self.assertEqual(ratio_color(0.99, 'Cube', limits), 'dark-yellow')
        self.assertEqual(ratio_color(1.65, 'Vector', limits), 'dark-green')  # per-class threshold
        self.assertEqual(ratio_color(1.6, 'Vector', limits), 'green')
        self.assertEqual(ratio_color(0.6, 'Vector', limits), 'yellow')
        self.assertIsNone(ratio_color(9.0, 'Mixed', limits))
        self.assertIsNone(ratio_color(9.0, 'Unknown', limits))
        self.assertIsNone(ratio_color(None, 'Cube', limits))
        for bad in (0, -1, 'x', None, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                validate_thresholds(bad, 1.1)
            with self.assertRaises(ValueError):
                validate_thresholds(2.0, bad)

    def test_custom_thresholds_flow_into_the_report(self):
        for side, rows in (('a', [('CubeOp', 'AI_CORE', 1, 100), ('VecOp', 'AI_VECTOR_CORE', 0, 100)]),
                           ('b', [('CubeOp', 'AI_CORE', 1, 50), ('VecOp', 'AI_VECTOR_CORE', 0, 50)])):
            with (self.root/side).open('w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(('Type', 'Accelerator Core', 'aicore_time(us)', 'Duration(us)'))
                writer.writerows(rows)
        default = compare_npus(self.root/'a', self.root/'b', self.root/'r1', all_steps=True)
        self.assertEqual({r['OP Type']: r['ratio_color'] for r in default['types']},
                         {'CubeOp': 'green', 'VecOp': 'dark-green'})
        tighter = compare_npus(self.root/'a', self.root/'b', self.root/'r2', all_steps=True,
                               cube_threshold=1.5, vector_threshold=1.5)
        self.assertEqual({r['OP Type']: r['ratio_color'] for r in tighter['types']},
                         {'CubeOp': 'green', 'VecOp': 'green'})
        self.assertEqual(tighter['thresholds']['cube'], 1.5)
        self.assertEqual(json.loads((self.root/'r2'/'npu_pair.json').read_text(encoding='utf8'))['thresholds'],
                         tighter['thresholds'])
        with self.assertRaises(ValueError):
            compare_npus(self.root/'a', self.root/'b', self.root/'r3', all_steps=True, cube_threshold=0)

    def test_below_threshold_export_lists_misses_with_shapes(self):
        header = ('Type', 'Name', 'Accelerator Core', 'aicore_time(us)', 'Duration(us)',
                  'Input Shapes', 'Input Data Types', 'Output Shapes', 'Output Data Types')
        sides = {
            'a': [('cube_ok', 'k_cube_ok', 'AI_CORE', 5, 100), ('cube_bad', 'k_cube_bad', 'AI_CORE', 5, 100),
                  ('vec_ok', 'k_vec_ok', 'AI_VECTOR_CORE', 0, 100), ('vec_bad', 'k_vec_bad', 'AI_VECTOR_CORE', 0, 100)],
            'b': [('cube_ok', 'k_cube_ok', 'AI_CORE', 5, 40), ('cube_bad', 'k_cube_bad', 'AI_CORE', 5, 80),
                  ('vec_ok', 'k_vec_ok', 'AI_VECTOR_CORE', 0, 80), ('vec_bad', 'k_vec_bad', 'AI_VECTOR_CORE', 0, 200)],
        }
        for side, rows in sides.items():
            with (self.root/side).open('w', newline='', encoding='utf8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for op, name, core, aicore, duration in rows:
                    writer.writerow((op, name, core, aicore, duration,
                                     f'{op}-in', 'FLOAT16', f'{op}-out', 'FLOAT16'))
        report = compare_npus(self.root/'a', self.root/'b', self.root/'result', all_steps=True)
        missed = below_threshold(report['signatures'], validate_thresholds(2.0, 1.1))
        self.assertEqual([row['OP Type'] for row in missed], ['vec_bad', 'cube_bad'])
        self.assertEqual([row['名称'] for row in missed], ['k_vec_bad', 'k_cube_bad'])
        self.assertEqual([row['阈值'] for row in missed], [1.1, 2.0])
        self.assertEqual(missed[0]['ratio_color'], 'dark-yellow')   # 0.5 against a 1.1 threshold
        self.assertEqual(missed[1]['ratio_color'], 'yellow')        # 1.25 against a 2.0 threshold
        self.assertEqual(missed[0]['输入 Shape'], 'vec_bad-in')
        self.assertEqual(missed[0]['输出 dtype'], 'FLOAT16')
        self.assertEqual(dominant_name([0, 0, set(), Counter({'a': 2, 'b': 1})]), 'a')
        with zipfile.ZipFile(self.root/'result'/'npu_pair.xlsx') as archive:
            self.assertIn('未达标算子', archive.read('xl/workbook.xml').decode())
        standalone = self.root/'misses.xlsx'
        write_below_threshold_xlsx(standalone, missed)
        with zipfile.ZipFile(standalone) as archive:
            sheet = archive.read('xl/worksheets/sheet1.xml').decode()
        self.assertIn('k_vec_bad', sheet)
        self.assertIn('<c r="E2" s="18"', sheet)   # dark-yellow ratio cell
        self.assertIn('<c r="E3" s="10"', sheet)   # yellow ratio cell
        self.assertIn('vec_bad-in', sheet)

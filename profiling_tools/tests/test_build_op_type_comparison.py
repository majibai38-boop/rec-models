"""Known-answer report regression: no customer traces or optional Excel dependency needed."""
import csv
from dataclasses import replace
from hashlib import sha256
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from profiling_tools.build_bundle import build_bundle
from profiling_tools.compare import compare_profiles
from profiling_tools.tests.test_matmul_backward import fixture as engine_fixture

SCRIPT = Path(__file__).resolve().parents[2]/'skills/gpu-npu-profiling/scripts/build_op_type_comparison.py'
spec = importlib.util.spec_from_file_location('op_type_report', SCRIPT)
report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = report
spec.loader.exec_module(report)


def csv_file(root, name, header, rows):
    with (root/name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def seal(root):
    manifest = {'run_id':root.name, 'artifacts':{k:{'sha256':sha256((root/k).read_bytes()).hexdigest()}
                 for k in report.INPUT_CSVS}, 'inputs':{k:{'sha256':k} for k in ('gpu_trace','npu_trace','npu_kernel_details')}}
    (root/'run_manifest.json').write_text(json.dumps(manifest))


def fixture(root, scope):
    root.mkdir()
    # 12 kernels, each 2 ms raw. 11 with 1 ms busy, one double-zero AICPU falls back to 2 ms.
    kinds = ['MatMulV3','BatchMatMulV3','Cast','InplaceIndexAdd','Sort','LinearizeIndex',
             'JaggedUniqueIndicesLength','DelinearizeUniqueIndex','find_with_digest_kernel',
             '_ZN17custom','GatherV2','AiCpu']
    names = ['aclnnMm_Gemm','aclnnMatmul_Bmm','aclnnAddmm_Cast'] + kinds[3:]
    csv_file(root, 'npu_kernel_gpu_mapping.csv', ['OP Type','NPU Kernel Name','NPU Duration(us)','NPU Task ID'],
             [(t,n,2000,i) for i,(t,n) in enumerate(zip(kinds,names))])
    csv_file(root, 'npu_kernel_details.csv', ['Type','Name','Duration(us)','aiv_time(us)','aicore_time(us)','Task ID'],
             [(t,n,2000,0 if t=='AiCpu' else 1000,0 if t=='AiCpu' else 500,i) for i,(t,n) in enumerate(zip(kinds,names))])
    csv_file(root, 'npu_gpu_op_statistic.csv', ['Row Kind','OP Type','GPU Kernel Count','GPU Memory Activity Count','GPU Total Time(us)'],
             [('npu_type','Cast',1,0,750),('total','TOTAL',9,1,99000)])
    # Vary the chain segments; identical values would conceal a shared GPU numerator.
    gpu_names = ['cub::DeviceRadixSort','cub::DeviceScan','cub::DeviceSelect','fbgemm_gpu::linearize_index_wo_infos',
                 'fbgemm_gpu::unique_indices_length','fbgemm_gpu::delinearize_unique_index','cuco','fbgemm_gpu::vectorized_gather_kernel']
    csv_file(root, 'gpu_kerenel_details.csv', ['Name','Duration(us)'], [('long_template_'*20+s+'<1>', (i+1)*1000) for i,s in enumerate(gpu_names)])
    csv_file(root, 'aten_op_statistic.csv', ['OP Type','NPU Raw Kernel Count','NPU Total Time(us)','GPU Raw Kernel Count','GPU Total Time(us)'],
             [('aten::'+k,int(n>0),n,int(g>0),g) for k,n,g in [('mm',1000,4000),('addmm',1000,6000),('bmm',0,8000),
              ('matmul',12000,0),('matmul_backward',4000,0),('index_add',2000,0),('index_add_',0,1000),('view',2000,9000)]])
    csv_file(root, 'gpu_memory_activity.csv', ['Name','Duration(us)'], [('Memcpy DtoD',1000)])
    csv_file(root, 'whole_call_mapping.csv', ['Whole OP','Issues'], [('aten::foo','冲突：input_shapes,input_dtypes；缺失：module'),
             ('aten::foo','缺失：input_dtypes'),('aten::bar','冲突：input_shapes')])
    summary = {'status':'complete_with_gaps','validation':{'status':'valid'},'producer':{'version':'test'}, 'run_id':root.name,
               'comparison':{'npu_first_kernel_count':12},'gpu_accounting':{'npu_total_time_us':24000,
               'kernel':{'total_time_us':36000,'total_count':8},'memory':{'total_time_us':1000,'total_count':1}},
               **{s:{'selected_step':scope,'selected_pid':1,'source':s} for s in ('npu','gpu')}}
    (root/'summary.json').write_text(json.dumps(summary))
    seal(root)


class OpTypeComparisonTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.step, self.full = self.root/'step', self.root/'full'
        fixture(self.step, 'ProfilerStep#10')
        fixture(self.full, 'all')

    def cli(self, *extra, script=SCRIPT, no_site=False):
        return subprocess.run([sys.executable, *(['-S'] if no_site else []), str(script), '--result-step10', str(self.step),
           '--result-all', str(self.full), '--output-md', str(self.root/'out.md'), '--output-xlsx', str(self.root/'out.xlsx'),
           *map(str,extra)], capture_output=True, encoding='utf-8', env={**os.environ, 'PYTHONUTF8':'1'})

    def test_known_answer_rows_and_denominators(self):
        run = report.load(self.step, False)
        table = report.type_table('selected', run, list(run.types))
        rows = {r[0]:r for r in table.rows}
        self.assertEqual(rows['AiCpu'][3], 2)
        self.assertEqual(rows['GEMM 家族'][1:6], (3,6,3,18,3))  # includes auxiliary Cast
        self.assertEqual(rows['TOTAL'][1:6], (12,24,13,36,1.5))  # 1ms DtoD excluded
        self.assertNotEqual(sum(r[4] or 0 for r in table.rows[:-1]), 36)
        self.assertIsNone(rows['MatMulV3'][5])
        self.assertIsNone(rows['custom_ascendc'][5])
        golden = '| TOTAL | 12 | 24.00 | 13.00 | 36.00 | 1.500 | 2.769 | summary：仅 GPU compute kernel；GPU 混源列不可相加 | — |'
        self.assertIn(golden.encode(), report.markdown('test', [table]).encode().splitlines())
        callers = {r[0]:r for r in report.caller_table(run).rows}
        self.assertEqual(callers['index_add'][1:6], (1,1,2,1,.5))
        self.assertEqual(callers['bmm'][3:6], (16,8,.5))
        self.assertIsNone(callers['t/view/transpose'][5])
        chain = report.chain_table(run)
        self.assertEqual([r[4] for r in chain.rows], [6,15,7,23])
        self.assertIsNone(chain.rows[-1][5])
        self.assertEqual((run.conflicts,run.missing), ({'aten::foo':1},{'aten::foo':1}))

    def test_optional_excel_colors_and_baseline(self):
        try:
            import openpyxl
        except ImportError:
            self.skipTest('optional openpyxl')
        res = self.cli('--baseline-step10',self.step,'--baseline-all',self.full)
        self.assertEqual(res.returncode, 0, res.stderr)
        with (self.root/'out.xlsx').open('rb') as f:
            wb = openpyxl.load_workbook(io.BytesIO(f.read()))
        self.assertIn('§5 NPU 两版类型变化', wb.sheetnames)
        cells = {row[0].value:row for row in wb['§1 Step 类型'].rows}
        self.assertEqual(cells['TOTAL'][5].value, 1.5)
        self.assertTrue(cells['TOTAL'][5].fill.fgColor.rgb.endswith('D9F2DC'))
        self.assertTrue(cells['InplaceIndexAdd'][5].fill.fgColor.rgb.endswith('FFE4C4'))
        self.assertIsNone(cells['GatherV2'][5].fill.patternType)
        self.assertEqual(self.cli().returncode, 2)  # do not overwrite

    def test_bundle_in_isolation_without_openpyxl(self):
        package = Path(build_bundle(self.root/'bundle')['skill'])
        self.assertTrue((package/'references/op-type-comparison-tables.md').is_file())
        out = self.cli(script=package/'scripts/build_op_type_comparison.py', no_site=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn('openpyxl', out.stderr)
        self.assertNotIn('## §4', (self.root/'out.md').read_text(encoding='utf-8'))
        self.assertFalse((self.root/'out.xlsx').exists())

    def test_reference_diff_precedes_output(self):
        self.assertEqual(self.cli(no_site=True).returncode, 0)
        ref = self.root/'out.md'
        other = self.root/'other.md'
        args = ('--reference-md', ref, '--output-md', other)
        self.assertEqual(self.cli(*args,no_site=True).returncode, 0)
        self.assertEqual(other.read_bytes(), ref.read_bytes())
        ref.write_text(ref.read_text(encoding='utf-8').replace(' | 36.00 | 1.500 |',' | 99.00 | 1.500 |'), encoding='utf-8')
        fail = self.root/'fail.md'
        self.assertEqual(self.cli('--reference-md', ref, '--output-md', fail, no_site=True).returncode, 2)
        self.assertFalse(fail.exists())

    def test_reject_stale_ambiguous_or_wrong_scope(self):
        path = self.step/'npu_kernel_details.csv'
        original = path.read_text(encoding='utf-8-sig')
        path.write_text(original.replace('aclnnMm_Gemm','different'), encoding='utf-8-sig')
        with self.assertRaisesRegex(ValueError, '发生变化'):
            report.load(self.step, False)
        seal(self.step)
        with self.assertRaisesRegex(ValueError, '身份不一致'):
            report.load(self.step, False)
        self.assertEqual(self.cli('--output-md', self.step/'new.md').returncode, 2)
        self.assertFalse((self.step/'new.md').exists())
        self.assertEqual(self.cli('--baseline-all', self.full).returncode, 2)
        self.assertEqual(self.cli('--result-step10', self.full).returncode, 2)

    def test_run_on_actual_engine_exports_with_no_pmu(self):
        config = engine_fixture(self.root)
        compare_profiles(config)
        self.step, self.full = Path(config.output_dir), self.root/'engine_all'
        compare_profiles(replace(config, gpu_step='', npu_step='', output_dir=str(self.full)))
        out = self.cli(no_site=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        run = report.load(self.step, False)
        self.assertGreater(run.total.raw, 0)
        self.assertIsNone(run.total.busy)

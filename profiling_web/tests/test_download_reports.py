import csv
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace

from profiling_tools.compare import compare_profiles
from profiling_tools.tests.test_skill_interfaces import fixture
from profiling_web.index import public_summary


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


class DownloadReportTests(unittest.TestCase):
    def test_selected_step_exports_original_columns_and_json_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = replace(fixture(tmp, multiple=True), gpu_pid='1', npu_pid='3')
            for side in ('gpu', 'npu'):
                path=Path(tmp)/f'{side}.json'
                data=json.loads(path.read_text(encoding='utf-8'))
                data['displayTimeUnit']='us'
                data['traceEvents'][2]['args']['device']=0
                data['traceEvents'][2]['custom_field']='preserved'
                path.write_text(json.dumps(data), encoding='utf-8')
            detail=Path(config.npu_kernel_details)
            detail.write_text('Name,Type,Duration(us),Start Time(us),Stream ID,Device_id,Step ID,Custom\n'
                'add_kernel,Add,4,20,7,0,1,original\n'
                'later,Add,99,120,7,0,2,later\n'
                'no_step,Other,2,30,7,0,,fallback\n'
                'later_no_step,Other,7,130,7,0,,outside\n',encoding='utf-8')
            summary=compare_profiles(config)
            self.assertEqual(summary['validation']['status'],'valid')
            root=Path(config.output_dir)
            npu=rows(root/'npu_kernel_details.csv')
            self.assertEqual([r['Custom'] for r in npu],['original','fallback'])
            self.assertEqual(list(npu[0]),['Name','Type','Duration(us)','Start Time(us)','Stream ID','Device_id','Step ID','Custom'])
            stats=rows(root/'npu_op_statistic.csv')
            self.assertEqual(sum(float(r['Total Time(us)']) for r in stats),6)
            gpu=rows(root/'gpu_kerenel_details.csv')
            self.assertEqual(gpu[0]['Device_id'],'0')
            self.assertEqual(gpu[0]['CPU Input Shapes'],'[[4],[4]]')
            self.assertNotIn('OP Type',gpu[0])
            self.assertEqual(rows(root/'gpu_op_statistic.csv')[0]['Count'],'1')
            report=public_summary(root)
            for side in ('gpu','npu'):
                exported=json.loads((root/f'{side}_trace.json').read_text(encoding='utf-8'))
                self.assertEqual(exported['displayTimeUnit'],'us')
                self.assertFalse(any(e.get('ts',0)>=100 for e in exported['traceEvents']))
                self.assertEqual(next(e for e in exported['traceEvents'] if e['cat']=='kernel')['custom_field'],'preserved')
                self.assertIn(f'{side}_trace.json',report['reports'])

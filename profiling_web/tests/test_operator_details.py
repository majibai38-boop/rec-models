import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from profiling_tools.model import OpMatch
from profiling_tools.operator_details import build_operator_details, tensor_metadata
from profiling_tools.report import build_npu_kernel_gpu_mapping_rows, write_reports
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_web.index import build_index
from profiling_web.operator_details import build_details_index, query_details


def case():
    gpu = profile('gpu', [op('aten::add', i) for i in range(4)], [(i, 'gpu_add') for i in range(4)])
    npu = profile('npu', [op('aten::add', i) for i in range(4)], [(i, 'Add') for i in range(4)])
    matches = [OpMatch(i, i, i, 1, 'high') for i in range(4)]
    for i, k in enumerate(npu.kernels):
        k.detail.input_shapes = k.detail.output_shapes = '4' if i < 3 else '2,2'
        k.detail.input_dtypes = k.detail.output_dtypes = 'FLOAT16'
        k.event.duration_us = i + 1
    gpu.kernels[0].event.duration_us = 6
    gpu.kernels[1].event.duration_us = 2
    return gpu, npu, matches, build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)


class OperatorDetailsTest(unittest.TestCase):
    def test_pipeline_report_index_and_full_filter_partition(self):
        from profiling_tools.tests.test_pipeline_analysis import detail
        from profiling_web.index import query_index
        g,n,m,r = case()
        for i,k in enumerate(n.kernels):
            k.detail.raw = detail(aiv_vec_ratio=.9 if i==0 else .2).raw
            k.detail.core_type = 'AI_VECTOR_CORE'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_reports(str(root/'reports'),g,n,m,[],[],{},diagnostic_coverage_threshold=0)
            source = root/'reports'/'npu_pipeline_analysis.json'
            report = json.loads(source.read_text(encoding='utf-8'))
            self.assertEqual(report['summary']['count'],4)
            self.assertEqual(report['summary']['time_us'],10)
            self.assertTrue((root/'reports'/'npu_pipeline_kernels.csv').is_file())
            build_index(root/'reports',root/'index.sqlite')
            result = query_details(root/'index.sqlite',op_type='Add',device_id='',core_type='AI_VECTOR_CORE',limit=1)
            self.assertTrue(result['has_more'])
            self.assertEqual(result['summary']['pipeline']['count'],4)
            self.assertEqual(result['summary']['pipeline']['time_us'],10)
            self.assertEqual(sum(c['time_pct'] for c in result['summary']['pipeline']['categories']),100)
            type_rows = query_index(root/'index.sqlite')['items']
            row = next(r for r in type_rows if r['row_kind']=='npu_type')
            self.assertEqual(row['pipeline']['time_us'],10)
            group = result['items'][0]
            filtered = query_details(root/'index.sqlite',op_type='Add',device_id='',core_type='AI_VECTOR_CORE',group_id=group['group_id'])
            self.assertEqual(filtered['summary']['pipeline']['time_us'],group['npu_time_us'])

    def test_metadata_outputs_and_unknown_gpu(self):
        g,n,m,r = case()
        g.ops[0].event.args.update({'Output Shapes': [[4]], 'Output Types': ['float']})
        report = build_operator_details(g,n,m,r)
        u = report['units'][0]
        self.assertEqual(u['npu_kernels'][0]['output_shapes'], [[4]])
        self.assertEqual(u['npu_kernels'][0]['output_dtypes'], ['float16'])
        self.assertEqual(u['gpu_kernels'][0]['input_shapes'], [])
        self.assertEqual(u['gpu_scopes'][0]['output_dtypes'], ['float32'])
        self.assertEqual(u['gpu_scopes'][0]['output_shapes'], [[4]])
        self.assertEqual(len({u['group_id'] for u in report['units']}), 3)  # Output metadata differs too.
        self.assertEqual(tensor_metadata('N/A')['input_shapes'], [])

    def test_shared_bundle_counted_once_and_cross_type_not_comparable(self):
        g,n,m,r = case()
        for row in r[:2]:
            row.update({'GPU Kernel Indices': '[0]', 'GPU Bundle Shared': True, 'Mapping Mode':'manual_scope_shared_bundle', 'Manual Link ID':'manual', 'Manual Reason':'confirmed'})
        report = build_operator_details(g,n,m,r)
        u = report['units'][0]
        self.assertEqual(len(u['npu_kernels']), 2)
        self.assertEqual(u['npu_time_us'], 3)
        self.assertEqual(u['gpu_time_us'], 6)
        self.assertEqual(u['ratio'], 2)
        self.assertTrue(u['manual'])
        r[1]['OP Type'] = 'Other'
        report = build_operator_details(g,n,m,r)
        self.assertTrue(all(u['ratio'] is None for u in report['units'][:2]))

    def test_overlapping_gpu_sets_and_unmapped(self):
        g,n,m,r = case()
        r[0]['GPU Kernel Indices']='[0,1]'
        r[3].update({'GPU Kernel Indices':'[]','Comparable':False})
        u = build_operator_details(g,n,m,r)['units']
        self.assertIsNone(u[0]['ratio'])
        self.assertIsNone(u[1]['ratio'])
        self.assertIsNone(u[3]['gpu_time_us'])
        self.assertIsNone(u[3]['ratio'])

    def test_index_sums_pagination_stable_positions_and_read_only_query(self):
        g,n,m,r = case()
        report = build_operator_details(g,n,m,r)
        original = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'details.json'; index=root/'index.sqlite'
            source.write_text(json.dumps(report),encoding='utf-8')
            with closing(sqlite3.connect(index)) as db, db:
                build_details_index(db, source)
            with patch('pathlib.Path.read_text',side_effect=AssertionError('GET must not read reports')):
                result=query_details(index,op_type='Add',device_id='',core_type='AI_CORE',limit=1)
                self.assertEqual(result['total'],2)
                self.assertTrue(result['has_more'])
                self.assertEqual(result['summary']['npu_time_us'],10)
                first=result['items'][0]
                self.assertEqual(first['count'],3)
                self.assertAlmostEqual(first['ratio'],10/6)  # Not mean(6, 1, 2/3).
                self.assertEqual(first['npu_mean_us'],2)
                units=query_details(index,op_type='Add',device_id='',core_type='AI_CORE',view='units',group_id=first['group_id'],sort='position',offset=1,limit=1)
                self.assertEqual(units['items'][0]['position']['index'],2)
                self.assertEqual(units['items'][0]['template_position']['total'],3)
                self.assertEqual(query_details(index,op_type='Add',device_id='wrong',core_type='AI_CORE')['total'],0)
                self.assertEqual(query_details(index,op_type="' OR 1=1 --",device_id='',core_type='AI_CORE')['total'],0)
                self.assertEqual(query_details(index,op_type='Add',device_id='',core_type='AI_CORE',search='float16')['total'],2)
                with self.assertRaises(ValueError):
                    query_details(index,op_type='Add',device_id='',core_type='',sort='SQL')
            self.assertEqual(report,original)

    def test_report_all_rows_independent_of_threshold_and_old_index(self):
        g,n,m,r = case()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_reports(str(root/'reports'),g,n,m,[],[],{},diagnostic_coverage_threshold=0)
            build_index(root/'reports',root/'new.sqlite')
            self.assertEqual(query_details(root/'new.sqlite',op_type='Add',device_id='',core_type='AI_CORE')['summary']['count'],4)
            with closing(sqlite3.connect(root/'old.sqlite')) as db, db:
                db.execute('CREATE TABLE records(id INTEGER)')
            self.assertFalse(query_details(root/'old.sqlite',op_type='Add',device_id='',core_type='AI_CORE')['supported'])

    def test_nested_owner_keeps_outer_comparison_scope(self):
        g,n,m,r = case()
        n.ops.append(op('aten::inner',4,parent=0))
        n.kernels[0].owner_op_index=4
        u=build_operator_details(g,n,m,r)['units'][0]
        self.assertEqual(u['npu_scopes'][0]['name'],'aten::add')
        self.assertEqual(u['npu_owners'][0]['name'],'aten::inner')

    def test_memory_namespace_is_separate_and_zero_time_is_not_a_ratio(self):
        from profiling_tools.tests.test_device_compare import memory
        g,n,m,r = case()
        g.memory_activities=[memory(0)]
        r[1].update({'GPU Kernel Indices':'[]','GPU Memory Activity Indices':'[0]'})
        n.kernels[2].event.duration_us=0
        before=copy.deepcopy(r)
        units=build_operator_details(g,n,m,r)['units']
        self.assertTrue(units[0]['comparable'])  # kernel:0 does not conflict with memory:0.
        self.assertTrue(units[1]['comparable'])
        self.assertEqual(units[1]['gpu_time_us'],.5)
        self.assertEqual(units[1]['gpu_kernels'][0]['kind'],'memory')
        self.assertIsNone(units[2]['ratio'])
        self.assertEqual(r,before)

    def test_module_and_input_dtype_remain_distinct(self):
        g,n,m,r = case()
        n.ops[0].module_key='other_module'
        g.ops[1].input_dtypes='["float16","scalar"]'
        report=build_operator_details(g,n,m,r)
        self.assertEqual(len({u['group_id'] for u in report['units']}),4)


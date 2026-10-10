import copy
import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from profiling_tools.compare import compare_profiles
from profiling_tools.core_breakdown import annotate_counters, build_core_breakdown
from profiling_tools.parser import load_kernel_details
from profiling_tools.result_api import summarize_results, validate_results
from profiling_tools.tests.test_comparison_accounting import mapping
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.tests.test_manual_mapping import nested_fixture
from profiling_tools.tests.test_skill_interfaces import fixture


class CoreBreakdownTest(unittest.TestCase):
    def test_missing_invalid_and_zero_counters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'details.csv'
            path.write_text('Name,Step ID,aicore_time(μs),aiv_vec_time(us)\n'
                            'vector,15,0,1.5\ncube,15,2,3\nmissing,15,N/A,\n'
                            'nan,15,nan,inf\nnegative,15,-1,-2\nout,16,9,9\n', encoding='utf-8-sig')
            details = load_kernel_details(str(path), '15')
            npu = profile('npu', [op('aten::x', 0)], [(0, d.name) for d in details])
            for k, d in zip(npu.kernels, details):
                k.detail = d
            rows = [mapping(i, gpu_indices=()) for i in range(len(details))]
            annotate_counters(npu, rows)
            self.assertEqual([r['NPU Compute Class'] for r in rows], ['Vector', 'Cube', 'Unknown', 'Unknown', 'Unknown'])
            result = build_core_breakdown(profile('gpu', []), rows)
            self.assertEqual(result['npu']['aicore_time_us'], 2)
            self.assertEqual(result['npu']['aicore_time_measured_count'], 2)
            self.assertEqual(result['npu']['aiv_vec_time_us'], 4.5)
            self.assertEqual(result['npu']['aiv_vec_time_measured_count'], 2)
            self.assertTrue(all(g['time_pct'] is None for g in result['gpu_reference']['groups']))

    def test_shared_and_mixed_gpu_groups_deduplicate(self):
        gpu = profile('gpu', [op('aten::x', 0)], [(0, 'k')] * 5)
        gpu.memory_activities = [copy.deepcopy(gpu.kernels[0])]
        rows = [mapping(0, gpu_indices=(0, 1)), mapping(1, gpu_indices=(1, 2)),
                mapping(2, gpu_indices=(0,)), mapping(3, gpu_indices=(3,))]
        for row, aic in zip(rows, (2, 0, 1, None)):
            row['NPU aicore_time(us)'] = aic
            row['NPU aiv_vec_time(us)'] = 1
        result = build_core_breakdown(gpu, rows)
        groups = {g['category']: g for g in result['gpu_reference']['groups']}
        self.assertEqual({k: g['count'] for k, g in groups.items()}, dict(Cube=1, Vector=1, Mixed=1, Unknown=1, Unmapped=1))
        self.assertEqual(sum(g['time_us'] for g in groups.values()), 10)
        self.assertTrue(all(g['time_pct'] == 20 for g in groups.values()))
        self.assertEqual(result['gpu_reference']['excluded_memory_time_us'], 2)
        # A group containing unknown NPU classification cannot be assumed Cube.
        rows[3]['GPU Kernel Indices'] = '[0, 3]'
        result = build_core_breakdown(gpu, rows)
        self.assertEqual(result['gpu_reference']['groups'][3]['count'], 2)

    def test_full_reports_and_tampered_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp, extra_gpu=True)
            Path(config.npu_kernel_details).write_text('Name,Type,Duration(us),Start Time(us),Stream ID,Device_id,Step ID,aicore_time(us),aiv_vec_time(us)\n'
                                                     'add_kernel,Add,4,20,7,0,1,0,1.5\n', encoding='utf-8')
            compare_profiles(config)
            result = summarize_results(config.output_dir)['core_breakdown']
            self.assertEqual(result['npu']['groups'][1]['time_us'], 4)
            self.assertEqual(result['npu']['aiv_vec_time_us'], 1.5)
            self.assertEqual(result['gpu_reference']['groups'][1]['time_pct'], 40)
            self.assertEqual(result['gpu_reference']['groups'][-1]['time_pct'], 60)
            with zipfile.ZipFile(Path(config.output_dir) / 'npu_gpu_op_statistic.xlsx') as xlsx:
                self.assertIn('Cube-Vector总览', xlsx.read('xl/workbook.xml').decode())
            with (Path(config.output_dir) / 'cube_vector_statistic.csv').open(encoding='utf-8-sig', newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 8)
            self.assertEqual(validate_results(config.output_dir)['status'], 'valid')
            path = Path(config.output_dir) / 'summary.json'
            data = json.loads(path.read_text(encoding='utf-8'))
            data['core_breakdown']['gpu_reference']['groups'][1]['time_us'] = 999
            path.write_text(json.dumps(data), encoding='utf-8')
            audit = validate_results(config.output_dir)
            self.assertEqual(audit['status'], 'invalid')
            self.assertTrue(any(f['id'] == 'core.gpu_reference.Vector.count_time' for f in audit['failures']))

    def test_manual_assignment_updates_reference_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = nested_fixture(tmp)
            Path(cfg.npu_kernel_details).write_text('Name,Type,Duration(us),Start Time(us),Stream ID,Device_id,Step ID,aicore_time(us),aiv_vec_time(us)\n'
                                                  'index_kernel,IndexAdd,4,30,7,0,1,0,1\n', encoding='utf-8')
            before = compare_profiles(cfg)
            after = compare_profiles(replace(cfg, output_dir=str(Path(tmp)/'manual')), manual_links=[
                {'id': 'test', 'npu_indices': [0], 'gpu_indices': [0], 'reason': 'verified parent scope'}])
            self.assertEqual(before['core_breakdown']['gpu_reference']['groups'][-1]['count'], 1)
            self.assertEqual(after['core_breakdown']['gpu_reference']['groups'][1]['count'], 1)
            self.assertEqual(before['core_breakdown']['gpu_reference']['total_time_us'], after['core_breakdown']['gpu_reference']['total_time_us'])

    def test_zero_duration_does_not_invent_ratio(self):
        gpu = profile('gpu', [op('aten::x', 0)], [(0, 'k')])
        gpu.kernels[0].event.duration_us = 0
        row = mapping(duration=0)
        row['NPU aicore_time(us)'] = 0
        result = build_core_breakdown(gpu, [row])
        for side in ('npu', 'gpu_reference'):
            self.assertEqual(result[side]['groups'][1]['count_pct'], 100)
            self.assertIsNone(result[side]['groups'][1]['time_pct'])

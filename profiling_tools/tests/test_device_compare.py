import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from profiling_tools.device_compare import build_cpu_bundle_rows, summarize_cpu_bundles
from profiling_tools.matcher import match_ops
from profiling_tools.model import KernelRecord, TraceEvent, OpMatch
from profiling_tools.parser import parse_trace
from profiling_tools.report import build_npu_kernel_gpu_mapping_rows, build_npu_gpu_op_statistic_rows
from profiling_tools.tests.test_compare import complete
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.xlsx_writer import TableSheet, write_grouped_workbook_xlsx


def memory(owner, name='Memcpy DtoD (Device -> Device)', size=32):
    return KernelRecord(TraceEvent(90, name, 'gpu_memcpy', 'X', 200, .5, '2', '7',
                                   {'bytes': size}), owner_op_index=owner)


class DeviceCompareTest(unittest.TestCase):
    def test_memory_is_separate_and_uses_device_not_runtime_duration(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.json'
            events = [complete('ProfilerStep#0', 'user_annotation', 0, 50, 1, 1),
                      complete('aten::repeat', 'cpu_op', 10, 30, 1, 1, {'External id': 1}),
                      complete('aten::copy_', 'cpu_op', 15, 20, 1, 1, {'External id': 2}),
                      complete('cudaMemcpyAsync', 'cuda_runtime', 17, 12, 1, 1, {'correlation': 9}),
                      complete('Memcpy DtoD', 'gpu_memcpy', 100, 2, 2, 7,
                               {'External id': 2, 'correlation': 9, 'bytes': 32, 'Device Type': 'cuda'})]
            path.write_text(json.dumps({'traceEvents': events}), encoding='utf-8')
            gpu = parse_trace(str(path), 'gpu', requested_step='0')
            self.assertEqual(len(gpu.kernels), 0)
            self.assertEqual(len(gpu.memory_activities), 1)
            self.assertEqual(gpu.memory_activities[0].owner_op_index, 1)
            self.assertEqual(gpu.memory_activities[0].event.duration_us, 2)
            # Runtime-only linking works without an external ID.
            events[-1]['args'].pop('External id')
            path.write_text(json.dumps({'traceEvents': events}), encoding='utf-8')
            gpu = parse_trace(str(path), 'gpu', requested_step='0')
            self.assertEqual(gpu.memory_activities[0].link_method, 'runtime_correlation')
            self.assertEqual(gpu.memory_activities[0].owner_op_index, 1)
            events[-1]['args']['External id'] = 2
            # A correlation reused by another host PID must not be adopted.
            events.append(complete('cudaMemcpyAsync', 'cuda_runtime', 17, 1, 3, 3, {'correlation': 9}))
            path.write_text(json.dumps({'traceEvents': events}), encoding='utf-8')
            gpu = parse_trace(str(path), 'gpu')
            self.assertIsNone(gpu.memory_activities[0].owner_op_index)

    def test_tile_memory_counts_and_time(self):
        gpu = profile('gpu', [op('aten::repeat', 0), op('aten::copy_', 1, parent=0)])
        gpu.memory_activities = [memory(1)]
        npu = profile('npu', [op('aten::repeat', 0)], [(0, 'Tile')])
        rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])
        self.assertTrue(rows[0]['Comparable'])
        self.assertFalse(rows[0]['Direct Kernel Pair'])
        self.assertEqual(rows[0]['GPU Kernel Count'], 0)
        stat = build_npu_gpu_op_statistic_rows(rows)[0]
        self.assertEqual(stat['GPU Memory Activity Count'], 1)
        self.assertEqual(stat['GPU Total Time(us)'], .5)
        self.assertEqual(stat['GPU Kernel Time(us)'], 0)
        self.assertEqual(stat['Comparable Ratio(GPU/NPU)'], .25)
        gpu.memory_activities[0].event.name = 'Memcpy HtoD'
        self.assertFalse(build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])[0]['Comparable'])

    def test_memory_fallback_does_not_change_cast(self):
        gpu = profile('gpu', [op('aten::copy_', 0)])
        gpu.memory_activities = [memory(0)]
        npu = profile('npu', [op('aten::copy_', 0)], [(0, 'Cast')])
        row = build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])[0]
        self.assertFalse(row['Comparable'])
        self.assertEqual(row['GPU Memory Activity Count'], 0)

    def test_memory_does_not_override_existing_compute(self):
        gpu = profile('gpu', [op('aten::repeat', 0), op('aten::copy_', 1, parent=0)], [(1, 'copy')])
        gpu.memory_activities = [memory(1)]
        npu = profile('npu', [op('aten::repeat', 0)], [(0, 'Tile')])
        row = build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])[0]
        self.assertEqual(row['GPU Kernel Count'], 1)
        self.assertEqual(row['GPU Memory Activity Count'], 0)

    def test_muls_primary_is_distinct_from_transpose(self):
        gpu = profile('gpu', [op('aten::mul', 0)], [(0, 'multiply')])
        npu = profile('npu', [op('aten::mul', 0)], [(0, 'Transpose'), (0, 'Muls')])
        rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])
        self.assertFalse(rows[0]['Comparable'])
        self.assertTrue(rows[1]['Comparable'])

    def test_noncast_slice_copy_recovery(self):
        gpu = profile('gpu', [op('aten::clone', 0), op('aten::copy_', 1, parent=0)], [(1, 'copy')])
        npu = profile('npu', [op('aten::clone', 0), op('aten::copy_', 1, parent=0)], [(1, 'Slice')])
        from profiling_tools.matcher import _recover_slice_copies
        self.assertEqual(_recover_slice_copies(gpu, npu, []), [(1, 1, 'noncast_clone_copy_recovery')])
        npu.kernels[0].detail.op_type = 'Cast'
        self.assertEqual(_recover_slice_copies(gpu, npu, []), [])

    def test_stack_is_whole_call_not_three_gpu_assignments(self):
        gpu = profile('gpu', [op('aten::stack', 0), op('aten::cat', 1, parent=0)], [(1, 'cat')])
        npu = profile('npu', [op('aten::stack', 0)], [(0, 'Pack'), (0, 'Pack'), (0, 'ConcatD')])
        matches = [OpMatch(1, 0, 0, 1, 'high')]
        mapping = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
        bundles = build_cpu_bundle_rows(gpu, npu, matches, mapping)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0]['GPU Kernel Count'], 1)
        self.assertEqual(bundles[0]['NPU Kernel Count'], 3)
        self.assertTrue(all(r['Comparable In CPU Bundle'] for r in mapping))
        self.assertTrue(all(not r['Comparable'] for r in mapping))

    def scalar_case(self):
        gpu = profile('gpu', [op('aten::lift_fresh', 0, shape='[[]]', dtype='["bool"]'),
                              op('aten::index_put_', 1),
                              op('aten::_index_put_impl_', 2, parent=1,
                                 shape='[[4],[],[],[],[]]', dtype='["bool","none","bool","scalar","scalar"]'),
                              op('aten::copy_', 3, parent=2)], [(2, 'index_put')])
        gpu.memory_activities = [memory(3, 'Memcpy HtoD', 1)]
        npu = profile('npu', [op('aten::scalar_tensor', 0),
                              op('aten::fill_', 1, parent=0, shape='[[],[]]', dtype='["bool","scalar"]'),
                              op('aten::index_put_', 2),
                              op('aten::_index_put_impl_', 3, parent=2,
                                 shape='[[4],[],[],[],[]]', dtype='["bool","none","bool","scalar","scalar"]')],
                      [(1, 'Fill'), (3, 'BroadcastTo'), (3, 'IndexPutV2')])
        return gpu, npu, [OpMatch(1, 2, 3, 1, 'high')]

    def test_scalar_preparation_bundle_and_rejections(self):
        gpu, npu, matches = self.scalar_case()
        mapping = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
        bundles = build_cpu_bundle_rows(gpu, npu, matches, mapping)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0]['GPU Total Time(us)'], 2.5)
        self.assertEqual(bundles[0]['GPU Kernel Count'], 1)
        self.assertEqual(bundles[0]['GPU Memory Activity Count'], 1)
        self.assertFalse(mapping[0]['Comparable'])
        self.assertTrue(mapping[0]['Accounted'])
        self.assertEqual(summarize_cpu_bundles(bundles)[0]['Call Count'], 1)
        gpu.memory_activities[0].event.args['bytes'] = 4
        self.assertEqual(build_cpu_bundle_rows(gpu, npu, matches, mapping), [])
        gpu.memory_activities[0].event.args['bytes'] = 1
        gpu.ops[2].input_dtypes = '["float32","none","float32","scalar","scalar"]'
        self.assertEqual(build_cpu_bundle_rows(gpu, npu, matches, mapping), [])
        gpu.ops[2].input_dtypes = npu.ops[3].input_dtypes
        npu.ops[1].input_dtypes = '["float32","scalar"]'
        self.assertEqual(build_cpu_bundle_rows(gpu, npu, matches, mapping), [])

    def test_independent_sheet_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.xlsx'
            write_grouped_workbook_xlsx(path, [{'A': 99}], [TableSheet('Types', ('A',)),
                                        TableSheet('Bundles', ('A',), rows=[{'A': 3}, {'A': 4}])])
            with zipfile.ZipFile(path) as archive:
                content = archive.read('xl/worksheets/sheet2.xml').decode()
                self.assertIn('<v>3</v>', content)
                self.assertNotIn('<v>99</v>', content)


if __name__ == '__main__':
    unittest.main()

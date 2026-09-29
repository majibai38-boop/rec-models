import copy
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

from profiling_tools.call_metadata import compare_metadata
from profiling_tools.model import OpMatch
from profiling_tools.report import build_npu_kernel_gpu_mapping_rows
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.whole_calls import build_whole_calls, summarize_whole_calls
from profiling_tools.xlsx_writer import TableSheet, write_grouped_workbook_xlsx


class WholeCallsTest(unittest.TestCase):
    def case(self, name='aten::cat'):
        gpu = profile('gpu', [op(name, 0)], [(0, 'cat')])
        npu = profile('npu', [op(name, 0)], [(0, 'Slice'), (0, 'ConcatD')])
        for p in (gpu, npu):
            for kernel in p.kernels:
                kernel.link_method = 'external_id'
        matches = [OpMatch(1, 0, 0, 1, 'high')]
        mapping = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
        return gpu, npu, matches, mapping

    def test_tensorlist_missing_is_not_conflict_or_wildcard(self):
        n = op('aten::cat', 0, dtype='["TensorList","Scalar"]')
        g = copy.deepcopy(n)
        n.raw_input_shapes = ';'
        g.raw_input_shapes = '[[[61,200],[61,8]],[]]'
        self.assertEqual(compare_metadata(n, g), ([], ['input_shapes']))
        n.raw_input_shapes = '[[[61,201],[61,8]],[]]'
        self.assertEqual(compare_metadata(n, g), (['input_shapes'], []))
        n.raw_input_shapes = g.raw_input_shapes
        self.assertEqual(compare_metadata(n, g), ([], []))
        n.raw_input_shapes = g.raw_input_shapes = '[[],[]]'
        n.input_dtypes = g.input_dtypes = '["float32","scalar"]'
        self.assertEqual(compare_metadata(n, g), ([], []))

    def test_full_call_does_not_change_type_allocations(self):
        gpu, npu, matches, mapping = self.case()
        before = copy.deepcopy(mapping)
        rows = build_whole_calls(gpu, npu, matches, mapping, [], [])
        self.assertEqual(mapping, before)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['Comparable'])
        self.assertEqual(rows[0]['NPU Total Time(us)'], 4)
        self.assertEqual(rows[0]['GPU Total Time(us)'], 2)
        self.assertEqual(rows[0]['Comparable Ratio(GPU/NPU)'], .5)

    def test_scalar_encoded_shape_retains_existing_normalization(self):
        n, g = op('aten::copy_', 0), op('aten::copy_', 0)
        n.raw_input_shapes, g.raw_input_shapes = '1', '[[1]]'
        self.assertEqual(compare_metadata(n, g), ([], []))

    def test_pending_call_has_no_comparable_ratio(self):
        gpu, npu, matches, mapping = self.case()
        npu.ops[0].raw_input_shapes = ';'
        gpu.ops[0].raw_input_shapes = '[[[4]],[]]'
        npu.ops[0].input_dtypes = gpu.ops[0].input_dtypes = '["tensorlist","scalar"]'
        rows = build_whole_calls(gpu, npu, matches, mapping, [], [])
        self.assertFalse(rows[0]['Comparable'])
        self.assertIn('元数据不完整', rows[0]['CPU Alignment Status'])
        self.assertEqual(rows[0]['Comparable Ratio(GPU/NPU)'], '')
        self.assertEqual(summarize_whole_calls(rows)[0]['Comparable Call Count'], 0)

    def test_report_only_paths_never_enter_new_table(self):
        for name in ('aten::div', 'aten::sub', 'aten::true_divide'):
            gpu, npu, matches, mapping = self.case(name)
            self.assertEqual(build_whole_calls(gpu, npu, matches, mapping, [], []), [])

    def test_overlap_is_visible_and_group_sum_is_unique(self):
        gpu, npu, matches, mapping = self.case()
        rows = build_whole_calls(gpu, npu, matches, mapping, [], [])
        duplicate = {**rows[0], 'Call ID': 'another-parent'}
        summary = summarize_whole_calls(rows + [duplicate])[0]
        self.assertEqual(summary['Observed Call Count'], 2)
        self.assertEqual(summary['Comparable Call Count'], 1)
        self.assertEqual(summary['Overlapping Call Count'], 1)
        self.assertEqual(summary['Comparable NPU Time(us)'], 4)

    def test_population_includes_unmatched_calls_and_ratio_is_not_average(self):
        gpu, npu, matches, mapping = self.case()
        gpu = profile('gpu', [op('aten::cat', 0), op('aten::cat', 1)], [(0, 'cat'), (1, 'cat')])
        gpu.kernels[0].link_method = 'external_id'
        gpu.kernels[1].event.duration_us = 8
        rows = build_whole_calls(gpu, npu, matches, mapping, [], [])
        summary = summarize_whole_calls(rows, gpu, npu)[0]
        self.assertEqual(summary['Comparable Ratio(GPU/NPU)'], .5)
        self.assertEqual(summary['NPU Population Call Count'], 1)
        self.assertEqual(summary['GPU Population Call Count'], 2)
        self.assertEqual(summary['Total Time Ratio(GPU/NPU)'], 2.5)
        gpu.kernels[1].event.name = 'different_implementation'
        self.assertEqual(summarize_whole_calls(rows, gpu, npu)[0]['Total Time Ratio(GPU/NPU)'], '')

    def test_namespaces_do_not_confuse_memory_zero_with_kernel_zero(self):
        from profiling_tools.tests.test_device_compare import memory
        gpu, npu, matches, mapping = self.case()
        gpu.memory_activities = [memory(0)]
        gpu.memory_activities[0].link_method = 'external_id'
        row = build_whole_calls(gpu, npu, matches, mapping, [], [])[0]
        self.assertEqual(row['GPU Kernel Indices'], '[0]')
        self.assertEqual(row['GPU Memory Activity Indices'], '[0]')
        self.assertEqual(row['GPU Total Time(us)'], 2.5)

    def test_xlsx_links_hidden_columns_and_schema_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'links.xlsx'
            write_grouped_workbook_xlsx(path, [{'Name': 'cat', 'ID': 'one'}], [
                TableSheet('整体对比', ('Name', 'ID'), hidden_columns=('ID',), links={(0, 'Name'): "'调用明细'!A2"}),
                TableSheet('调用明细', ('Name', 'ID'))])
            with zipfile.ZipFile(path) as archive:
                sheet = ET.fromstring(archive.read('xl/worksheets/sheet1.xml'))
            ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            self.assertEqual(sheet.find('x:hyperlinks/x:hyperlink', ns).attrib['location'], "'调用明细'!A2")
            self.assertEqual(sheet.find('x:cols/x:col[@min="2"]', ns).attrib['hidden'], '1')
            tags = [e.tag.rsplit('}', 1)[-1] for e in sheet]
            self.assertLess(tags.index('autoFilter'), tags.index('hyperlinks'))

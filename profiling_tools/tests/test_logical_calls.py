import json
import tempfile
import unittest
from pathlib import Path

from profiling_tools.matcher import match_ops
from profiling_tools.model import KernelDetail, KernelRecord, OpMatch, OpRecord, ParsedProfile, TraceEvent
from profiling_tools.parser import parse_trace
from profiling_tools.report import build_npu_kernel_gpu_mapping_rows
from profiling_tools.tests.test_compare import complete


def op(name, index, *, parent=None, shape='[[4],[]]', dtype='["float32","scalar"]'):
    return OpRecord(TraceEvent(index, name, 'cpu_op', 'X', index * 10, 5, '1', '1', {}),
                    name, input_shapes=shape, input_dtypes=dtype,
                    module_key='block_0', parent_index=parent)


def profile(side, ops, types=()):
    kernels = []
    for owner, name in types:
        i = len(kernels)
        ops[owner].kernel_indices.append(i)
        kernels.append(KernelRecord(
            TraceEvent(i, name, 'kernel', 'X', 100 + i * 10, 2, '2', '7', {}),
            detail=KernelDetail(i, name=name, op_type=name, core_type='AI_CORE') if side == 'npu' else None,
            owner_op_index=owner))
    return ParsedProfile(side, '', ops, kernels, '1', 'all', [], 0)


class LogicalCallsTest(unittest.TestCase):
    def test_reordered_index_put_uses_dtype(self):
        for name in ['aten::index_put_', 'aten::_index_put_impl_']:
            gpu = profile('gpu', [op(name, 0, dtype='["bool","scalar"]'), op(name, 1)])
            npu = profile('npu', [op(name, 0), op(name, 1, dtype='["bool","scalar"]')])
            matches, _, _ = match_ops(gpu, npu, {})
            self.assertEqual({(m.gpu_op_index, m.npu_op_index) for m in matches}, {(0, 1), (1, 0)})
            self.assertTrue(all(m.confidence == 'high' for m in matches))

    def test_fill_parent_conflict_releases_kernel(self):
        gpu = profile('gpu', [op('aten::zero_', 0), op('aten::fill_', 1, parent=0)], [(1, 'fill')])
        npu = profile('npu', [op('aten::scalar_tensor', 0), op('aten::fill_', 1, parent=0)], [(1, 'Fill')])
        matches, _, _ = match_ops(gpu, npu, {})
        self.assertEqual(matches, [])

    def test_overload_mismatch_is_not_recovered(self):
        gpu = profile('gpu', [op('aten::sum', 0, shape='[[4],[],[],[]]')])
        npu = profile('npu', [op('aten::sum', 0)])
        self.assertEqual(match_ops(gpu, npu, {})[0], [])

    def test_custom_kernel_count_mismatch_not_promoted(self):
        name = 'nestedconcatfunc'
        gpu = profile('gpu', [op(name, 0, shape='', dtype='')], [(0, 'nested_concat_fwd_kernel')])
        npu = profile('npu', [op(name, 0, shape='', dtype=''), op(name, 1, shape='', dtype='')],
                      [(0, 'nested_concat_fwd_kernel'), (1, 'nested_concat_fwd_kernel')])
        matches, _, _ = match_ops(gpu, npu, {})
        self.assertTrue(all(m.confidence == 'low' for m in matches))

    def test_cast_ops_are_not_repaired(self):
        gpu = profile('gpu', [op('aten::copy_', 0)], [(0, 'copy')])
        npu = profile('npu', [op('aten::copy_', 0, shape='[[],[]]')], [(0, 'Cast')])
        matches, _, _ = match_ops(gpu, npu, {})
        self.assertEqual(len(matches), 1)
        self.assertNotEqual(matches[0].alignment_basis, 'logical_call_metadata')

    def test_same_name_children_not_independent_calls(self):
        for name, kernel_type in [('aten::sum', 'ReduceSum'), ('aten::arange', 'Range')]:
            gpu = profile('gpu', [op(name, 0), op(name, 1, parent=0),
                                  op(name, 2), op(name, 3, parent=2)], [(1, 'compute'), (3, 'compute')])
            npu = profile('npu', [op(name, 0), op(name, 1)], [(0, kernel_type), (1, kernel_type)])
            matches, _, _ = match_ops(gpu, npu, {})
            self.assertEqual([(m.gpu_op_index, m.npu_op_index) for m in matches], [(0, 0), (2, 1)])
            rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
            self.assertTrue(all(r['Comparable'] for r in rows))
            self.assertEqual([r['GPU Kernel Indices'] for r in rows], ['[0]', '[1]'])

    def test_child_aliases(self):
        for parent, child, kind in [('aten::repeat', 'aten::copy_', 'Tile'),
                                    ('aten::zero_', 'aten::fill_', 'ZerosLike'),
                                    ('aten::ones', 'aten::fill_', 'OnesLike'),
                                    ('aten::rsub', 'aten::sub', 'Sub')]:
            gpu = profile('gpu', [op(parent, 0), op(child, 1, parent=0)], [(1, 'compute')])
            npu = profile('npu', [op(parent, 0)], [(0, kind)])
            rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])
            self.assertTrue(rows[0]['Comparable'])

    def test_count_nonzero_does_not_reuse_ne_or_copy(self):
        gpu = profile('gpu', [op('aten::count_nonzero', 0), op('aten::ne', 1, parent=0),
                              op('aten::copy_', 2, parent=0), op('aten::sum', 3, parent=0)],
                      [(1, 'ne'), (2, 'copy'), (3, 'sum')])
        npu = profile('npu', [op('aten::count_nonzero', 0), op('aten::ne', 1, parent=0)],
                      [(0, 'ReduceSum'), (1, 'NotEqual')])
        rows = build_npu_kernel_gpu_mapping_rows(gpu, npu,
                                                [OpMatch(1, 0, 0, 1, 'high'), OpMatch(2, 1, 1, 1, 'high')])
        self.assertEqual(rows[0]['GPU Kernel Indices'], '[2]')
        self.assertEqual(rows[1]['GPU Kernel Indices'], '[0]')

    def test_custom_python_scope_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.json'
            path.write_text(json.dumps({'traceEvents': [
                complete('aten::empty', 'cpu_op', 1, 1, 1, 1),
                complete('/src/position_concat.py(65): position_concat, callsite:10',
                         'python_function', 10, 10, 1, 1, {'External id': 2}),
                complete('position_concat_fwd_kernel', 'kernel', 30, 3, 2, 7, {'External id': 2}),
            ]}), encoding='utf-8')
            parsed = parse_trace(str(path), 'gpu')
            self.assertEqual(parsed.ops[1].canonical_name, 'position_concat_fwd_kernel')
            self.assertEqual(parsed.kernels[0].owner_op_index, 1)


if __name__ == '__main__':
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from profiling_tools.matcher import match_ops
from profiling_tools.model import OpMatch
from profiling_tools.parser import CUSTOM_COMPARABLE_OPS, parse_trace
from profiling_tools.report import build_npu_kernel_gpu_mapping_rows, build_npu_gpu_op_statistic_rows
from profiling_tools.tests.test_compare import complete
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.logical_recovery import recover_layernorm_blocks
from profiling_tools.call_metadata import compare_metadata, compare_match_metadata


def strong(*profiles):
    for p in profiles:
        for k in p.kernels:
            k.link_method = 'external_id' if p.side == 'gpu' else 'torch_to_npu'


class ScopeRecoveryTest(unittest.TestCase):
    def gelu(self, types=('GeluV2',)):
        gpu = profile('gpu', [op('aten::gelu', 0)], [(0, 'gelu')])
        npu = profile('npu', [op('aten::gelu', 0), op('aten::gelu', 1, parent=0,
                      shape='[[4],[],[4]]')], [(1, name) for name in types])
        strong(gpu, npu)
        return gpu, npu, [OpMatch(1, 0, 0, 1, 'high')]

    def test_nested_npu_gelu_uses_outer_metadata_and_preserves_owner(self):
        gpu, npu, matches = self.gelu()
        row = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)[0]
        self.assertTrue(row['Comparable'])
        self.assertEqual(row['Mapping Mode'], 'logical_scope_bundle')
        self.assertEqual(row['NPU CPU OP Index'], 1)
        self.assertEqual(row['CPU Match ID'], 1)

    def test_shared_scope_deduplicates_gpu_time(self):
        gpu, npu, matches = self.gelu(('GeluV2', 'GeluV2'))
        rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
        self.assertTrue(all(r['GPU Bundle Shared'] and not r['Direct Kernel Pair'] for r in rows))
        stats = build_npu_gpu_op_statistic_rows(rows)
        self.assertEqual(stats[0]['GPU Total Time(us)'], 2)
        self.assertEqual(stats[0]['NPU Total Time(us)'], 4)

    def test_scope_rejects_mixed_types_metadata_and_weak_links(self):
        for kind in ('mixed', 'shape', 'dtype', 'module', 'link', 'gpu_link', 'trace_only', 'device', 'gpu_device'):
            with self.subTest(kind=kind):
                gpu, npu, matches = self.gelu(('GeluV2', 'Cast') if kind == 'mixed' else ('GeluV2',))
                if kind == 'shape': npu.ops[0].input_shapes = '[[8],[]]'
                if kind == 'dtype': npu.ops[0].input_dtypes = '["float16","scalar"]'
                if kind == 'module': npu.ops[0].module_key = ''
                if kind == 'link': npu.kernels[0].link_method = 'fallback'
                if kind == 'gpu_link': gpu.kernels[0].link_method = 'fallback'
                if kind == 'trace_only': npu.kernels[0].detail = None
                if kind == 'device':
                    gpu, npu, matches = self.gelu(('GeluV2', 'GeluV2'))
                    npu.kernels[1].detail.device_id = 'other'
                if kind == 'gpu_device':
                    gpu = profile('gpu', [op('aten::gelu', 0)], [(0, 'gelu'), (0, 'gelu')])
                    strong(gpu)
                    gpu.kernels[1].event.args['device'] = 1
                self.assertFalse(any(r['Comparable'] for r in build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)))

    def test_outer_scope_does_not_reuse_gpu_kernel(self):
        gpu, npu, matches = self.gelu()
        matches.append(OpMatch(2, 0, 1, 1, 'high'))
        rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)
        self.assertNotEqual(rows[0]['Mapping Mode'], 'logical_scope_bundle')

    def test_convolution_child_and_real_metadata_conflict(self):
        gpu = profile('gpu', [op('aten::_convolution', 0), op('aten::cudnn_convolution', 1, parent=0)], [(1, 'conv')])
        npu = profile('npu', [op('aten::_convolution', 0)], [(0, 'Conv2DV2')])
        strong(gpu, npu)
        match = [OpMatch(1, 0, 0, 1, 'high')]
        self.assertTrue(build_npu_kernel_gpu_mapping_rows(gpu, npu, match)[0]['Comparable'])
        npu.ops[0].input_shapes = '[[9],[]]'
        self.assertFalse(build_npu_kernel_gpu_mapping_rows(gpu, npu, match)[0]['Comparable'])

    def test_new_child_scopes_require_kernel_ownership_evidence(self):
        for name, kernel_type in [('aten::_convolution', 'Conv2DV2'), ('ropefunc', 'rope_kernel_flattened')]:
            gpu = profile('gpu', [op(name, 0), op('aten::child', 1, parent=0)], [(1, 'compute')])
            npu = profile('npu', [op(name, 0)], [(0, kernel_type)])
            matches = [OpMatch(1, 0, 0, 1, 'high')]
            self.assertFalse(build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)[0]['Comparable'])
            strong(gpu, npu)
            self.assertTrue(build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)[0]['Comparable'])

    def test_index_select_primary_does_not_allocate_transpose(self):
        gpu = profile('gpu', [op('aten::index_select', 0)], [(0, 'indexSelectLargeIndex')])
        npu = profile('npu', [op('aten::index_select', 0)], [(0, 'Transpose'), (0, 'GatherV2')])
        rows = build_npu_kernel_gpu_mapping_rows(gpu, npu, [OpMatch(1, 0, 0, 1, 'high')])
        self.assertEqual([r['Comparable'] for r in rows], [False, True])
        self.assertEqual(rows[1]['GPU Kernel Indices'], '[0]')

    def test_custom_scope_selection_and_ownership_both_sides(self):
        for name in CUSTOM_COMPARABLE_OPS:
            for side in ('gpu', 'npu'):
                with self.subTest(name=name, side=side), tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / 'trace.json'
                    path.write_text(json.dumps({'traceEvents': [
                        complete('aten::empty', 'cpu_op', 0, 1, 1, 1),
                        complete(name, 'cpu_op', 10, 10, 1, 1, {'External id': 77}),
                        complete('custom_kernel', 'kernel', 50, 2, 2, 7, {'External id': 77}),
                    ]}), encoding='utf-8')
                    parsed = parse_trace(str(path), side)
                    self.assertEqual(parsed.ops[1].canonical_name, name)
                    self.assertEqual(parsed.kernels[0].owner_op_index, 1)

    def test_custom_sequence_missing_or_conflicting_call_stays_untrusted(self):
        for variant in ('ok', 'missing', 'shape', 'metadata', 'module'):
            gpu = profile('gpu', [op('ropefunc', 0), op('ropefunc', 1)])
            npu = profile('npu', [op('ropefunc', 0), op('ropefunc', 1)])
            if variant == 'missing': npu.ops.pop()
            if variant == 'shape': npu.ops[1].input_shapes = '[[9],[]]'
            if variant == 'metadata': npu.ops[1].input_shapes = ''
            if variant == 'module': npu.ops[1].module_key = ''
            matches, _, _ = match_ops(gpu, npu, {})
            if variant == 'ok': self.assertTrue(all(m.alignment_basis == 'custom_scope_metadata_sequence' for m in matches))
            else: self.assertTrue(all(m.confidence == 'low' for m in matches))

    def test_custom_no_flow_does_not_promote(self):
        gpu = profile('gpu', [op('ropefunc', 0)], [(0, 'rope_kernel')])
        npu = profile('npu', [op('ropefunc', 0)], [(0, 'rope_kernel_flattened')])
        matches, _, _ = match_ops(gpu, npu, {})
        self.assertFalse(build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)[0]['Comparable'])
        strong(gpu, npu)
        self.assertTrue(build_npu_kernel_gpu_mapping_rows(gpu, npu, matches)[0]['Comparable'])

    def norm(self, module, repeats=1):
        ops = [op('aten::before', 0)]
        for repeat in range(repeats):
            for name in ('mean', 'sub', 'var', 'div', 'add'):
                item = op('aten::' + name, len(ops))
                item.module_key = module + str(repeat)
                ops.append(item)
        ops.append(op('aten::after', len(ops)))
        return profile('gpu', ops)

    def test_layernorm_recovers_only_complete_unique_block_between_anchors(self):
        gpu, npu = self.norm('layernorm_2'), self.norm('layernorm_5')
        pairs = [(0, 0, 'semantic_anchor'), (6, 6, 'semantic_anchor')]
        recovered = recover_layernorm_blocks(gpu, npu, pairs, [(0, 0), (6, 6)])
        self.assertEqual(len(recovered), 7)
        self.assertTrue(all(basis == 'layernorm_block_metadata' for _, _, basis in recovered[2:]))
        m = OpMatch(1, 1, 1, 1, 'high', 'layernorm_block_metadata')
        self.assertEqual(compare_metadata(npu.ops[1], gpu.ops[1]), (['module'], []))
        self.assertEqual(compare_match_metadata(npu.ops[1], gpu.ops[1], m), ([], []))

    def test_layernorm_ambiguous_or_incomplete_blocks_not_recovered(self):
        for variant in ('duplicate', 'shape', 'no_bounds', 'missing'):
            gpu, npu = self.norm('layernorm_2', 2 if variant == 'duplicate' else 1), self.norm('layernorm_5')
            bounds = [(0, 0), (len(gpu.ops)-1, len(npu.ops)-1)]
            if variant == 'shape': npu.ops[2].input_shapes = '[[9],[]]'
            if variant == 'missing': npu.ops[2].input_dtypes = ''
            if variant == 'no_bounds': bounds = bounds[:1]
            pairs = [(g,n,'semantic_anchor') for g,n in bounds]
            self.assertEqual(recover_layernorm_blocks(gpu,npu,pairs,bounds), pairs)

    def test_layernorm_conflicting_same_name_pair_is_not_kept(self):
        gpu, npu = self.norm('layernorm_2'), self.norm('layernorm_5')
        npu.ops[3].input_shapes = '[[9],[]]'
        anchors = [(0,0),(6,6)]
        self.assertEqual(recover_layernorm_blocks(gpu,npu,[(3,3,'same_name_recovery')],anchors), [])

    def test_layernorm_protects_existing_valid_pair(self):
        gpu, npu = self.norm('layernorm_2'), self.norm('layernorm_5')
        gpu.ops.append(op('aten::mean', 7))
        gpu.ops[7].module_key = npu.ops[1].module_key
        pairs = [(7,1,'semantic_anchor')]
        self.assertEqual(recover_layernorm_blocks(gpu,npu,pairs,[(0,0),(6,6)]), pairs)

import json
from pathlib import Path
import tempfile
import unittest

from profiling_tools.model import OpRecord, TraceEvent
from workload_modeler.parameters import candidates
from workload_modeler.profile import analyze
from workload_modeler.schema import Scope


def op(name, shapes, parent=None, module=''):
    return OpRecord(TraceEvent(1, name, 'cpu_op', 'X', 0, 1, '1', '1', {}), name,
                    input_shapes=json.dumps(shapes), input_dtypes=json.dumps(['float']*len(shapes)),
                    parent_index=parent, module_path=module)


class ParameterTests(unittest.TestCase):
    def test_linear_and_embedding_exclude_batch_and_lookup_count(self):
        ops = [op('aten::linear', [[32, 128], [64, 128], [64]]),
               op('aten::embedding', [[10000, 16], [32, 20]])]
        r = candidates(ops, {0, 1})
        self.assertEqual(r['dense_candidates'], 64*128+64)
        self.assertEqual(r['embedding_candidates'], 10000*16)
        self.assertIsNone(r['exact_parameters'])

    def test_repeated_steps_and_nested_lowering_not_multiplied(self):
        ops = [op('aten::linear', [[32, 128], [64, 128], [64]]),
               op('aten::addmm', [[64], [32, 128], [128, 64]], parent=0),
               op('aten::linear', [[32, 128], [64, 128], [64]])]
        r = candidates(ops, {1, 2})
        self.assertEqual(r['dense_candidates'], 8256)
        self.assertIsNone(r['ambiguous_matrix_elements'])
        self.assertEqual([g['calls'] for g in r['groups']], [2, 2])

    def test_module_identity_and_shape_only_limit(self):
        ops = [op('aten::linear', [[2, 8], [4, 8]], module=m) for m in ['layer1', 'layer2']]
        self.assertEqual(candidates(ops, {0, 1})['dense_candidates'], 64)
        for item in ops:
            item.module_path = ''
        r = candidates(ops, {0, 1})
        self.assertEqual(r['dense_candidates'], 32)
        self.assertEqual(r['groups'][0]['dedup_basis'], 'shape_only')

    def test_matrix_activation_not_in_parameter_total(self):
        r = candidates([op('aten::bmm', [[32, 4, 8], [32, 8, 4]])], {0})
        self.assertIsNone(r['candidate_parameters'])
        self.assertEqual(r['ambiguous_matrix_elements'], 1024)

    def test_norm_excludes_running_stats_and_invalid_shapes(self):
        ops = [op('aten::batch_norm', [[32, 8], [8], [8], [8], [8]]),
               op('aten::linear', [[32, 8], [-1, 8], []])]
        r = candidates(ops, {0, 1})
        self.assertEqual(r['dense_candidates'], 16)
        self.assertEqual(r['missing_shape_calls'], 1)
        self.assertEqual(candidates([], [])['status'], 'unavailable')

    def test_device_owner_scope_and_backward_exclusion(self):
        ops = [op('aten::linear', [[32, 8], [4, 8]]),
               op('LinearBackward', []), op('aten::mm', [[32, 8], [8, 4]], parent=1)]
        self.assertIsNone(candidates(ops, {2})['candidate_parameters'])
        self.assertEqual(candidates(ops, {0})['candidate_parameters'], 32)

    def test_trace_adapter_persists_parameter_summary(self):
        source = Path(__file__).resolve().parents[1]/'examples'/'gpu_trace.json'
        events = json.loads(source.read_text())
        for e in events['traceEvents']:
            if e['name'] == 'aten::add':
                e['name'] = 'aten::linear'
                e['args']['Input Dims'] = [[32, 8], [4, 8], [4]]
                e['args']['Input type'] = ['float', 'float', 'float']
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'trace.json').write_text(json.dumps(events))
            p = analyze(directory, Scope(side='gpu', step='ProfilerStep#1'))
        self.assertEqual(p['summary']['parameters']['candidate_parameters'], 36)
        self.assertEqual(p['summary']['kernel_count'], 2)


if __name__ == '__main__':
    unittest.main()

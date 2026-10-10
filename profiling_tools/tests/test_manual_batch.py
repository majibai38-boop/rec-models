import copy
import unittest

from profiling_tools.manual_mapping import batch_plan, catalog, link_history
from profiling_tools.model import OpMatch
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.tests.test_comparison_accounting import mapping
from profiling_tools.tests.test_scope_recovery import strong


def repeated_catalog(count=260):
    gpu = profile('gpu', [op('aten::index_add_', i) for i in range(count)], [(i, 'index_kernel') for i in range(count)])
    npu = profile('npu', [op('aten::index_add_', i) for i in range(count)] +
                  [op('aten::index_add', count+i, parent=i) for i in range(count)],
                  [(count+i, 'IndexAdd') for i in range(count)])
    strong(gpu, npu)
    matches = [OpMatch(i, i, i, 1.0, 'high') for i in range(count)]
    return catalog(gpu, npu, matches, [mapping(i, gpu_indices=()) for i in range(count)])


class ManualBatchTest(unittest.TestCase):
    def test_260_aligned_calls_and_template_order(self):
        c = repeated_catalog()
        for aligned in (True, False):
            if not aligned:
                for t in c['targets']:
                    t['aligned'] = []
            plan = batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')
            self.assertEqual(plan['pair_count'], 260)
            self.assertEqual(plan['gpu_kernel_count'], 260)
            self.assertEqual(plan['npu_kernel_count'], 260)
            self.assertEqual(plan['basis_counts'], {'aligned_cpu_scope' if aligned else 'manual_template_order': 260})
            self.assertTrue(all(p['npu_indices'] == p['gpu_indices'] for p in plan['pairs']))
            self.assertEqual(len({i for p in plan['pairs'] for i in p['gpu_indices']}), 260)

    def test_ambiguous_missing_or_overlapping_pairs_are_excluded(self):
        for kind in ('weak', 'metadata', 'occupied', 'overlap', 'thread', 'device', 'module', 'ambiguous'):
            c = repeated_catalog(5)
            if kind == 'weak': c['targets'][1]['weak_owner'] = True
            if kind == 'metadata': c['targets'][1]['batch_context']['metadata_complete'] = False
            if kind == 'occupied': c['candidates'].pop(1)
            if kind == 'overlap': c['candidates'][1]['indices'] = [2]
            if kind == 'thread': c['candidates'][1]['batch_context']['tid'] = 'other'
            if kind == 'device': c['candidates'][1]['device_key'] = ['other', '0']
            if kind == 'module': c['candidates'][1]['module'] = 'other'
            if kind == 'ambiguous': c['targets'][1]['aligned'].append('op:2')
            with self.subTest(kind=kind):
                plan = batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')
                self.assertEqual(plan['pair_count'], 3 if kind == 'overlap' else 4)
                self.assertGreater(plan['excluded_count'], 0)

    def test_manual_order_requires_equal_counts_and_selected_anchor(self):
        c = repeated_catalog(5)
        for t in c['targets']: t['aligned'] = []
        with self.assertRaises(ValueError): batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:1')
        with self.assertRaises(ValueError): batch_plan(c['targets'], c['candidates'][:-1], 'npu:0', 'op:0')
        c['targets'][2]['aligned'] = ['op:3']
        with self.assertRaises(ValueError): batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')

    def test_unavailable_legacy_catalog_and_batch_history(self):
        c = repeated_catalog(2)
        del c['targets'][0]['batch_context']
        with self.assertRaisesRegex(ValueError, '旧报告'): batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')
        links = [{'id': str(i), 'batch_id': 'batch1', 'reason': 'checked', 'npu_indices': [i], 'gpu_indices': [i]}
                 for i in range(260)]
        original = copy.deepcopy(links)
        history = link_history(links)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['pair_count'], 260)
        self.assertEqual(history[0]['id'], 'batch1')
        self.assertEqual(links, original)

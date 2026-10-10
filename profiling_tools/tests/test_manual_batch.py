import copy
import unittest

from profiling_tools.manual_mapping import BatchConflict, batch_plan, catalog, link_history
from profiling_tools.model import OpMatch
from profiling_tools.tests.test_logical_calls import op, profile
from profiling_tools.tests.test_comparison_accounting import mapping
from profiling_tools.tests.test_scope_recovery import strong


def repeated_catalog(count=260, outer_shape='[[4],[]]', aligned=True, inner_metadata=True):
    gpu = profile('gpu', [op('aten::index_add_', i) for i in range(count)], [(i, 'index_kernel') for i in range(count)])
    npu = profile('npu', [op('aten::index_add_', i, shape=outer_shape) for i in range(count)] +
                  [op('aten::index_add', count+i, parent=i,
                      shape='[[4],[],[4]]' if inner_metadata else '',
                      dtype='["float32","scalar","float32"]' if inner_metadata else '') for i in range(count)],
                  [(count+i, 'IndexAdd') for i in range(count)])
    strong(gpu, npu)
    matches = [OpMatch(i, i, i, 1.0, 'high') for i in range(count)] if aligned else []
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
            if kind == 'metadata':
                for scope in c['targets'][1]['comparison_scopes']:
                    scope['batch_context']['metadata_complete'] = False
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
        c['targets'][0].pop('comparison_scopes')
        with self.assertRaisesRegex(ValueError, '旧报告'): batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')
        links = [{'id': str(i), 'batch_id': 'batch1', 'reason': 'checked', 'npu_indices': [i], 'gpu_indices': [i]}
                 for i in range(260)]
        original = copy.deepcopy(links)
        history = link_history(links)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['pair_count'], 260)
        self.assertEqual(history[0]['id'], 'batch1')
        self.assertEqual(links, original)

    def test_outer_metadata_with_missing_inner_metadata_keeps_owner(self):
        c = repeated_catalog(inner_metadata=False)
        plan = batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')
        self.assertEqual(plan['pair_count'], 260)
        self.assertEqual(plan['comparison']['npu']['cpu_name'], 'aten::index_add_')
        self.assertEqual(plan['pairs'][0]['npu_cpu_index'], 260)
        self.assertEqual(plan['pairs'][0]['npu_scope_cpu_index'], 0)
        self.assertEqual(c['targets'][0]['cpu_name'], 'aten::index_add')
        self.assertEqual(c['targets'][0]['shapes'], [])

    def test_confirmed_template_maps_different_signatures_and_legacy_catalog(self):
        for legacy in (False, True):
            c = repeated_catalog(260, outer_shape='[[4],[],[4]]')
            if legacy:
                for t in c['targets']:
                    t.pop('comparison_scopes')
                    t.pop('cpu_aligned')
            with self.subTest(legacy=legacy):
                with self.assertRaisesRegex(ValueError, '人工确认模板'):
                    batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0')
                before = copy.deepcopy(c)
                plan = batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0', 'confirmed_template')
                self.assertEqual(plan['pair_count'], 260)
                basis = 'manual_confirmed_template_order' if legacy else 'manual_confirmed_cpu_scope'
                self.assertEqual(plan['basis_counts'], {basis: 260})
                self.assertIn('shapes', plan['comparison']['different_fields'])
                self.assertEqual(len({i for p in plan['pairs'] for i in p['gpu_indices']}), 260)
                self.assertEqual(c, before)

    def test_confirmed_template_only_expands_selected_signatures(self):
        c = repeated_catalog(5, outer_shape='[[4],[],[4]]', aligned=False)
        c['targets'][2]['comparison_scopes'][1]['shapes'] = [[8], [], [8]]
        c['candidates'][2]['shapes'] = [[8], []]
        plan = batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0', 'confirmed_template')
        self.assertEqual(plan['pair_count'], 4)
        self.assertNotIn('npu:2', [p['target_id'] for p in plan['pairs']])

    def test_wrong_seed_reports_positions_and_correct_candidate_for_260_calls(self):
        c = repeated_catalog(260, outer_shape='[[4],[],[4]]', aligned=False)
        with self.assertRaises(BatchConflict) as caught:
            batch_plan(c['targets'], c['candidates'], 'npu:179', 'op:0', 'confirmed_template')
        detail = caught.exception.detail
        self.assertEqual(detail['code'], 'seed_pair_mismatch')
        self.assertEqual(detail['position']['npu'], 180)
        self.assertEqual(detail['position']['gpu'], 1)
        self.assertEqual(detail['suggested_candidate']['id'], 'op:179')
        plan = batch_plan(c['targets'], c['candidates'], 'npu:179',
                          detail['suggested_candidate']['id'], 'confirmed_template')
        self.assertEqual(plan['pair_count'], 260)
        self.assertEqual(plan['selection_position']['paired_gpu'], 180)
        self.assertTrue(all(p['npu_indices'] == p['gpu_indices'] for p in plan['pairs']))

    def test_complete_cpu_correspondence_wins_over_different_call_order(self):
        c = repeated_catalog(5, outer_shape='[[4],[],[4]]')
        c['targets'][0]['cpu_aligned'] = ['op:1']
        c['targets'][1]['cpu_aligned'] = ['op:0']
        plan = batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:1', 'confirmed_template')
        self.assertEqual(plan['pair_count'], 5)
        self.assertEqual(plan['basis_counts'], {'manual_confirmed_cpu_scope': 5})
        self.assertEqual([p['gpu_indices'] for p in plan['pairs']], [[1], [0], [2], [3], [4]])
        self.assertEqual(plan['selection_position']['gpu'], 2)
        with self.assertRaises(BatchConflict) as caught:
            batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0', 'confirmed_template')
        self.assertEqual(caught.exception.detail['suggested_candidate']['id'], 'op:1')

    def test_partial_conflicting_cpu_alignment_is_reported_not_overridden(self):
        c = repeated_catalog(5, outer_shape='[[4],[],[4]]', aligned=False)
        c['targets'][3]['cpu_aligned'] = ['op:4']
        with self.assertRaises(BatchConflict) as caught:
            batch_plan(c['targets'], c['candidates'], 'npu:0', 'op:0', 'confirmed_template')
        detail = caught.exception.detail
        self.assertEqual(detail['code'], 'cpu_alignment_conflict')
        self.assertEqual(detail['conflict_count'], 1)
        self.assertEqual(detail['conflicts'][0]['npu_position'], 4)
        self.assertNotIn('suggested_candidate', detail)

    def test_confirmed_template_rejects_counts_order_missing_and_alignment_conflicts(self):
        for case in ('counts', 'seed_order', 'aligned_conflict', 'weak', 'metadata', 'overlap'):
            c = repeated_catalog(5, outer_shape='[[4],[],[4]]')
            candidate = 'op:0'
            if case == 'counts': c['candidates'].pop()
            if case == 'seed_order': candidate = 'op:1'
            if case == 'aligned_conflict': c['targets'][1]['cpu_aligned'] = ['op:2']
            if case == 'weak': c['candidates'][1]['weak_owner'] = True
            if case == 'metadata': c['candidates'][1]['batch_context']['tensor_metadata_complete'] = False
            if case == 'overlap': c['candidates'][1]['indices'] = [0]
            with self.subTest(case=case), self.assertRaises(ValueError):
                batch_plan(c['targets'], c['candidates'], 'npu:0', candidate, 'confirmed_template')

    def test_ancestor_with_sibling_work_is_not_comparison_scope(self):
        gpu = profile('gpu', [op('aten::index_add_', 0)], [(0, 'index_kernel')])
        npu = profile('npu', [op('aten::index_add_', 0), op('aten::index_add', 1, parent=0),
                              op('aten::copy_', 2, parent=0)], [(1, 'IndexAdd'), (2, 'Cast')])
        strong(gpu, npu)
        c = catalog(gpu, npu, [OpMatch(0, 0, 0, 1, 'high')],
                    [mapping(0, gpu_indices=()), {**mapping(1, gpu_indices=()), 'OP Type': 'Cast'}])
        self.assertEqual([s['cpu_index'] for s in c['targets'][0]['comparison_scopes']], [1])
        self.assertEqual(c['targets'][0]['aligned'], [])

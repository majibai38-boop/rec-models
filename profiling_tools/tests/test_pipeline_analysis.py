import copy
import unittest

from profiling_tools.model import KernelDetail
from profiling_tools.pipeline_analysis import classify, parse_ratio, summarize, build_pipeline_report
from profiling_tools.tests.test_logical_calls import profile, op


def detail(core='AI_VECTOR_CORE', **values):
    raw = {f'aiv_{p}_ratio': .1 for p in ('vec', 'scalar', 'mte2', 'mte3')}
    if core != 'AI_VECTOR_CORE':
        raw.update({f'aic_{p}_ratio': .1 for p in ('mac', 'scalar', 'mte1', 'mte2', 'fixpipe')})
    raw.update(values)
    return KernelDetail(0, core_type=core, raw=raw)


class PipelineAnalysisTest(unittest.TestCase):
    def test_scale_and_invalid_data(self):
        self.assertEqual(parse_ratio('85%'), .85)
        for invalid in ('N/A', '', None, 'nan', 'inf', -1, 85):
            self.assertIsNone(parse_ratio(invalid))
        d = detail(aiv_vec_ratio=.99)
        del d.raw['aiv_mte3_ratio']
        self.assertEqual(classify(d)['category'], 'unknown')
        self.assertIn('aiv_mte3_ratio', classify(d)['reason'])
        self.assertEqual(classify(detail(**{f'aiv_{p}_ratio': 0 for p in ('vec','scalar','mte2','mte3')}))['category'], 'unknown')
        self.assertEqual(classify(KernelDetail(0, core_type='AI_CPU'))['category'], 'unknown')

    def test_boundary_competing_pipelines_and_scalar(self):
        self.assertEqual(classify(detail(aiv_vec_ratio=.8))['category'], 'no_bound')
        self.assertEqual(classify(detail(aiv_vec_ratio=.801))['category'], 'compute')
        self.assertEqual(classify(detail(aiv_vec_ratio=.9, aiv_mte2_ratio=.85))['category'], 'mixed')
        self.assertEqual(classify(detail(aiv_vec_ratio=.85, aiv_mte2_ratio=.75))['category'], 'mixed')
        self.assertEqual(classify(detail(aiv_scalar_ratio=.91))['category'], 'scalar')

    def test_write_and_on_chip_paths(self):
        self.assertEqual(classify(detail(aiv_mte3_ratio=.92))['category'], 'memory')
        for pipe in ('mte1', 'fixpipe'):
            result = classify(detail('AI_CORE', **{f'aic_{pipe}_ratio': .95}))
            self.assertEqual(result['category'], 'memory')
            self.assertIn('不能直接归因于 HBM', result['next_check'])

    def test_mix_keeps_both_domains_and_incomplete_side(self):
        d = detail('MIX_AIC', aic_mac_ratio=.92, aiv_mte2_ratio=.94)
        before = copy.deepcopy(d)
        result = classify(d)
        self.assertEqual(result['category'], 'mixed')
        self.assertEqual([s['category'] for s in result['sides']], ['compute', 'memory'])
        self.assertEqual(d, before)
        del d.raw['aiv_mte3_ratio']
        self.assertEqual(classify(d)['category'], 'unknown')

    def test_zero_aic_fields_do_not_hide_vector_memory(self):
        d = detail(aiv_mte2_ratio=.95)
        d.raw.update({f'aic_{p}_ratio': 0 for p in ('mac', 'mte2', 'mte3', 'scalar')})
        self.assertEqual(classify(d)['category'], 'memory')

    def test_legacy_and_missing_cube_output_path(self):
        d = KernelDetail(0, core_type='AI_CORE', raw={p+'_ratio': .1 for p in ('vec','mac','scalar','mte1','mte2','mte3')})
        d.raw['mac_ratio'] = .9
        self.assertEqual(classify(d)['category'], 'compute')
        d = detail('AI_CORE', aic_mac_ratio=.95)
        del d.raw['aic_fixpipe_ratio']
        self.assertEqual(classify(d)['category'], 'unknown')

    def test_time_partition_and_shape_groups_independent_of_gpu(self):
        n = profile('npu', [op('aten::add', i) for i in range(3)], [(i,'Add') for i in range(3)])
        rows = []
        for i,k in enumerate(n.kernels):
            k.detail = detail(aiv_vec_ratio=.9 if i==0 else .2)
            k.detail.input_shapes = '4' if i<2 else '8'
            k.detail.input_dtypes = 'FLOAT16'
            rows.append({'NPU Kernel Index':i, 'OP Type':'Add', 'Device_id':'0', 'Core Type':'AI_VECTOR_CORE',
                         'NPU Duration(us)': 90 if i==0 else 5, 'Comparable':False})
        report = build_pipeline_report(n, rows)
        self.assertEqual(len(report['shapes']), 2)
        categories = {r['category']:r for r in report['summary']['categories']}
        self.assertEqual(categories['compute']['time_pct'],90)
        self.assertEqual(categories['no_bound']['count'],2)
        self.assertEqual(sum(r['time_us'] for r in categories.values()),100)
        self.assertEqual(summarize([{'time_us':0}])['categories'][0]['time_pct'],None)
        with self.assertRaises(ValueError):
            build_pipeline_report(n, rows + rows[:1])

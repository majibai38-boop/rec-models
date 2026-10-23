import copy
import json
import unittest
from workload_modeler.features import canonical_dtype, normalize_dtypes_json, extract_features, ModelIssue
from workload_modeler.predict import predict
from workload_modeler.screening import screen, assessment


def fixture(parts):
    units = []
    for i, (category, duration) in enumerate(parts):
        units.append(dict(id=str(i), op='aten::custom', kind='compute', class_name=category,
            input_shapes='[[4],[4]]', input_dtypes='["float32","float32"]', signature=str(i),
            source_us=duration, kernel_count=1,
            events=[dict(id=str(i), duration=duration, start=i*1000., stream='1', name='test')]))
    total = sum(duration for _,duration in parts)
    summary = dict(compute_sum_us=total, device_sum_us=total, kernel_count=len(parts),
        memory_sum_us=0., communication_sum_us=0., busy_us=total, step_us=1000.,
        serial_step_eligible=False, classification_basis='synthetic',
        compute_breakdown=[dict(category=c,time_us=t) for c,t in parts])
    config = dict(source_hardware=dict(name='source', vendor='NVIDIA', metrics={}),
        target_hardware=dict(name='target', vendor='Ascend', metrics={}),
        request=dict(source_platform_id='s', source_platform_revision=1,
                     target_platform_id='t', target_platform_revision=1, scope=dict(step='1')))
    return dict(summary=summary, units=units, warnings=[]),config


class ScreeningTests(unittest.TestCase):
    def test_zero_prediction_still_has_screening(self):
        profile,config=fixture([('Cube-like',36),('Vector-like',37),('Mixed',5),('Unknown',22)])
        result=predict(profile,config)
        self.assertEqual(result['prediction']['coverage_pct'],0)
        self.assertEqual(result['assessment']['screening']['status'],'mixed_workload')
        self.assertEqual(result['assessment']['screening']['metrics']['classified_pct'],78)
        self.assertEqual(result['assessment']['partial']['status'],'unavailable')
        self.assertEqual(result['assessment']['step']['status'],'unavailable')
        self.assertTrue(all(result['audit'].values()))

    def test_partial_ratio_is_not_step_ratio(self):
        profile,config=fixture([('Cube-like',60),('Unknown',40)])
        config['request']['calibrations']=[dict(signature='0',source_us=60,target_us=30,evidence='synthetic calibration')]
        result=predict(profile,config)
        self.assertEqual(result['assessment']['partial']['coverage_pct'],60)
        self.assertEqual(result['assessment']['partial']['ratio'],2)
        self.assertIsNone(result['assessment']['step']['prediction'])
        self.assertEqual(result['assessment']['partial']['uncovered_source_us'],40)

    def test_full_model_still_requires_step_conditions(self):
        profile,config=fixture([('Vector-like',60),('Vector-like',40)])
        config['request']['calibrations']=[dict(signature=str(i),source_us=u['source_us'],target_us=u['source_us']/2,
            evidence='synthetic calibration') for i,u in enumerate(profile['units'])]
        result=predict(profile,config)
        self.assertEqual(result['prediction']['coverage_pct'],100)
        self.assertEqual(result['assessment']['step']['status'],'unavailable')
        config['request']['assumptions']=dict(fixed_gap_serial_step=True)
        profile['summary']['serial_step_eligible']=True
        result=predict(profile,config)
        self.assertAlmostEqual(result['assessment']['step']['prediction']['ratio'],1000/950)

    def test_unknown_time_not_renormalized_or_defaulted(self):
        profile,_=fixture([('Cube-like',20),('Unknown',80)])
        s=screen(profile['summary'],profile['units'],'Ascend')
        self.assertEqual(s['status'],'classification_gap')
        self.assertEqual(s['metrics']['cube_pct'],20)
        self.assertEqual(s['metrics']['cube_bound_coverage_pct'],0)
        self.assertEqual(s['metrics']['cube_memory_lower_pct'],0)
        self.assertEqual(s['metrics']['cube_memory_upper_pct'],100)

    def test_memory_risk_uses_all_cube_time_not_only_known_subset(self):
        profile,_=fixture([('Cube-like',40),('Cube-like',40),('Vector-like',20)])
        profile['units'][0]['target_terms']={'dominant':'memory'}
        s=screen(profile['summary'],profile['units'],'Ascend')
        self.assertEqual(s['status'],'matrix_candidate')
        self.assertEqual(s['metrics']['cube_memory_lower_pct'],50)
        profile['units'][1]['target_terms']={'dominant':'memory'}
        self.assertEqual(screen(profile['summary'],profile['units'],'Ascend')['status'],'matrix_memory_risk')

    def test_short_and_fused_are_overlapping_observations(self):
        profile,_=fixture([('Mixed',4),('Vector-like',5),('Unknown',91)])
        profile['units'][0]['op']='FlashAttentionFunc'
        s=screen(profile['summary'],profile['units'],'Ascend')
        self.assertEqual(s['metrics']['small_kernel_count'],1)
        self.assertEqual(s['metrics']['small_kernel_time_pct'],4)
        self.assertEqual(s['metrics']['fused_time_pct'],4)
        self.assertEqual(s['metrics']['mixed_pct'],4)

    def test_empty_compute_and_no_step_do_not_fabricate_metrics(self):
        profile,_=fixture([])
        profile['summary']['step_us']=None
        s=screen(profile['summary'],[],'Ascend')
        self.assertEqual(s['status'],'no_compute')
        self.assertIsNone(s['metrics']['cube_pct'])
        self.assertIsNone(s['metrics']['small_kernel_time_pct'])

    def test_legacy_assessment_does_not_mutate_result(self):
        profile,config=fixture([('Vector',100)])
        result=predict(profile,config)
        result.pop('assessment')
        original=copy.deepcopy(result)
        enriched=assessment(result,historical=True)
        self.assertTrue(enriched['historical'])
        self.assertEqual(enriched['screening']['status'],'vector_risk')
        self.assertEqual(result,original)

    def test_dtype_aliases_and_raw_parameters(self):
        for alias in ['c10::Half','torch.float16','half','at::Half','scalar_type::Half']:
            self.assertEqual(canonical_dtype(alias),'fp16')
        self.assertEqual(canonical_dtype('c10::BFloat16'),'bf16')
        self.assertIsNone(canonical_dtype('Scalar'))
        self.assertEqual(json.loads(normalize_dtypes_json('["c10::Half","Scalar","int64"]')),['float16','Scalar','int64'])
        unit=dict(op='aten::bmm',input_shapes='[[16,61,3840],[16,3840,3072]]',
            input_dtypes='["c10::Half","c10::Half"]')
        f=extract_features(unit)
        self.assertEqual(f['dtype'],'fp16')
        self.assertEqual(f['flops'],2*16*61*3840*3072)

    def test_semantic_parameters_and_unsupported_op_not_unknown_dtype(self):
        unit=dict(op='aten::add',input_shapes='[[4],[4],[]]',input_dtypes='["float32","float32","Scalar"]')
        with self.assertRaises(ModelIssue) as caught:
            extract_features(unit)
        self.assertEqual(caught.exception.code,'parameters')
        unit['op']='aten::addmm'
        with self.assertRaises(ModelIssue) as caught:
            extract_features(unit)
        self.assertEqual(caught.exception.code,'unsupported_operator')

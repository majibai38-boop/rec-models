import copy
import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient
from workload_modeler.app import create_app
from workload_modeler.empirical import estimate
from workload_modeler.predict import predict
from workload_modeler.schema import EmpiricalAssumptions
from workload_modeler.tests.test_screening import fixture


def single_ratio(value):
    return dict(conservative=value,baseline=value,optimistic=value)


class EmpiricalTests(unittest.TestCase):
    def test_zero_precise_coverage_still_has_overall(self):
        p,c=fixture([('Cube-like',50),('Vector-like',50)])
        r=predict(p,c)
        self.assertEqual(r['prediction']['coverage_pct'],0)
        self.assertIsNone(r['step_prediction'])
        self.assertEqual(r['empirical']['status'],'available')
        self.assertEqual(r['empirical']['device']['baseline']['ratio'],1)
        self.assertEqual(r['empirical']['direct_assumption_time_pct'],100)
        self.assertTrue(any('缺规格' in n for n in r['empirical']['notices']))

    def test_harmonic_total_and_hardware_ratio_direction(self):
        p,c=fixture([('Cube-like',50),('Unknown',50)])
        for hw,peak in [(c['source_hardware'],100),(c['target_hardware'],200)]:
            hw['metrics']={'matrix_fp16':dict(value=peak,dense=True,evidence='user')}
        e=estimate(p['summary'],p['units'],c,dict(cube_compute_fraction=1,hardware_factors=single_ratio(1)))
        self.assertAlmostEqual(e['device']['baseline']['target_us'],75)
        self.assertAlmostEqual(e['device']['baseline']['ratio'],4/3)
        c['source_hardware'],c['target_hardware']=c['target_hardware'],c['source_hardware']
        e=estimate(p['summary'],p['units'],c,dict(cube_compute_fraction=1,hardware_factors=single_ratio(1)))
        self.assertAlmostEqual(e['device']['baseline']['ratio'],2/3)

    def test_partition_short_fused_and_noncompute(self):
        p,c=fixture([('Cube',20),('Mixed',4),('Vector',10),('Memory',3),('Communication',2),('Unknown',11)])
        p['units'][1]['op']='FlashAttentionFunc'
        p['units'][3]['kind']='memory'
        p['units'][4]['kind']='communication'
        e=estimate(p['summary'],p['units'],c)
        times={b['key']:b['source_us'] for b in e['buckets']}
        self.assertEqual(times,dict(matrix_compute=10,matrix_memory=10,short=4,vector=10,transfer=3,communication=2,unknown=11))
        self.assertEqual(sum(times.values()),p['summary']['device_sum_us'])
        self.assertTrue(e['audit']['source_time_conserved'])

    def test_short_is_per_event_not_call_sum(self):
        p,c=fixture([('Mixed',8)])
        p['units'][0]['events']=[dict(duration=4),dict(duration=4)]
        self.assertEqual(estimate(p['summary'],p['units'],c)['buckets'][0]['key'],'short')
        e=estimate(p['summary'],p['units'],c,dict(short_kernel_us=0))
        self.assertEqual(e['buckets'][0]['key'],'fused')

    def test_multi_stream_overlap_and_gap_not_double_counted(self):
        p,c=fixture([('Cube-like',100)])
        p['summary'].update(step_us=200,step_busy_us=50,activities_in_step=True,streams=['1','2'],overlap_us=50)
        e=estimate(p['summary'],p['units'],c,dict(cube_compute_fraction=1,matrix_ratio=2,hardware_factors=single_ratio(1)))
        self.assertEqual(e['device']['baseline']['ratio'],2)
        self.assertEqual(e['step']['scenarios']['baseline']['busy_us'],25)
        self.assertEqual(e['step']['scenarios']['baseline']['target_us'],175)
        self.assertAlmostEqual(e['step']['scenarios']['baseline']['ratio'],200/175)
        self.assertFalse(p['summary']['serial_step_eligible'])

    def test_step_guard_and_csv_still_have_device(self):
        p,c=fixture([('Vector',100)])
        for update in [dict(step_us=None),dict(step_us=200,step_busy_us=100,activities_in_step=False),
                       dict(step_us=200,step_busy_us=101,activities_in_step=True)]:
            s={**p['summary'],**update}
            e=estimate(s,p['units'],c)
            self.assertIsNone(e['step'])
            self.assertEqual(e['status'],'available')
            self.assertTrue(e['step_reason'])

    def test_modes_overrides_missing_spec_and_notices(self):
        p,c=fixture([('Cube',100)])
        c['source_hardware']['metrics']={'matrix_tf32':dict(value=10,dense=None,evidence='estimated')}
        c['target_hardware']['metrics']={'matrix_hf32':dict(value=40,dense=True,evidence='user')}
        options=dict(cube_compute_fraction=1,source_matrix_mode='tf32',target_matrix_mode='hf32')
        e=estimate(p['summary'],p['units'],c,options)
        self.assertEqual(e['device']['baseline']['ratio'],4)
        self.assertTrue(any('稠密口径未确认' in n for n in e['notices']))
        self.assertTrue(any('estimated' in n for n in e['notices']))
        options['matrix_ratio']=3
        self.assertEqual(estimate(p['summary'],p['units'],c,options)['device']['baseline']['ratio'],3)

    def test_scenarios_ordered_and_invalid_assumptions(self):
        p,c=fixture([('Cube',50),('Vector',20),('Unknown',30)])
        e=estimate(p['summary'],p['units'],c)
        d=e['device']
        self.assertLessEqual(d['conservative']['ratio'],d['baseline']['ratio'])
        self.assertLessEqual(d['baseline']['ratio'],d['optimistic']['ratio'])
        for option in [dict(matrix_ratio=0),dict(cube_compute_fraction=2),dict(short_kernel_us=float('nan')),
                       dict(unknown_ratio=dict(conservative=2,baseline=1,optimistic=3))]:
            with self.assertRaises(ValueError):
                EmpiricalAssumptions.model_validate(option)

    def test_conservation_rejects_missing_or_negative_time(self):
        p,c=fixture([('Unknown',100)])
        p['summary']['device_sum_us']=101
        with self.assertRaisesRegex(ValueError,'不守恒'):estimate(p['summary'],p['units'],c)
        p['summary']['device_sum_us']=100
        p['units'][0]['events']=[dict(duration=101),dict(duration=-1)]
        with self.assertRaises(ValueError):estimate(p['summary'],p['units'],c)

    def test_disabled_empty_and_no_mutation(self):
        p,c=fixture([('Unknown',100)])
        before=copy.deepcopy((p,c))
        self.assertEqual(estimate(p['summary'],p['units'],c,dict(enabled=False))['status'],'disabled')
        estimate(p['summary'],p['units'],c)
        self.assertEqual((p,c),before)
        p,c=fixture([])
        self.assertEqual(estimate(p['summary'],[],c)['status'],'empty')

    def test_api_historical_preview_does_not_change_saved_files(self):
        root=Path(__file__).resolve().parents[1]/'data'/'tests'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            app=create_app(tmp,test_mode=True)
            with TestClient(app) as client:
                self.assertEqual(client.get('/static/empirical.js').status_code,200)
                self.assertTrue(client.get('/api/meta').json()['empirical_defaults']['enabled'])
                store=app.state.store
                job=store.create_job('synthetic legacy')['id']
                p,c=fixture([('Unknown',100)])
                r=predict(p,c);r.pop('empirical')
                path=store.job_dir(job)/'result.json'
                path.write_text(json.dumps(r),encoding='utf-8')
                original=path.read_bytes()
                store.transition(job,{'draft'},'completed',config=c)
                base='/api/jobs/'+job
                summary=client.get(base+'/summary').json()
                self.assertTrue(summary['empirical']['preview'])
                response=client.post(base+'/empirical-preview',json=dict(unknown_ratio=single_ratio(2)))
                self.assertEqual(response.status_code,200,response.text)
                e=response.json()
                self.assertEqual(e['device']['baseline']['ratio'],2)
                self.assertEqual(e['provenance']['job_id'],job)
                self.assertEqual(path.read_bytes(),original)
                self.assertEqual(store.job(job)['config'],c)
                self.assertEqual(client.post(base+'/empirical-preview',json=dict(matrix_ratio=0)).status_code,422)
                self.assertEqual(client.get(base+'/summary').json()['empirical']['device']['baseline']['ratio'],1)
                unfinished=store.create_job('draft')['id']
                self.assertEqual(client.post('/api/jobs/'+unfinished+'/empirical-preview',json={}).status_code,409)


if __name__=='__main__':
    unittest.main()

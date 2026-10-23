import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from workload_modeler.app import create_app
from workload_modeler.profile import analyze, inventory
from workload_modeler.predict import predict
from workload_modeler.features import extract_features, ModelIssue
from workload_modeler.schema import Hardware, Scope
from workload_modeler.store import Store, Conflict, seed

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'gpu_trace.json'


class ModelerTests(unittest.TestCase):
    def setUp(self):
        test_root = Path(__file__).resolve().parents[1] / 'data' / 'tests'
        test_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=test_root)
        self.root = Path(self.temp.name)
        (self.root / 'trace.json').write_bytes(EXAMPLE.read_bytes())

    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        store = Store(self.root / 'db')
        seed(store)
        hw = next(h for h in store.list('hardware') if h['vendor'] == 'NVIDIA')
        platform = next(p for p in store.list('platform') if p['hardware_id'] == hw['id'])
        return dict(source_hardware=hw, target_hardware=copy.deepcopy(hw), request=dict(
            source_platform_id=platform['id'], source_platform_revision=1,
            target_platform_id=platform['id'], target_platform_revision=1,
            scope=dict(step='1'), assumptions=dict(fixed_gap_serial_step=True)))

    def test_inventory_and_conservation(self):
        self.assertEqual(inventory(self.root, 'gpu')['devices'], ['device:0'])
        result = analyze(self.root, Scope(step='1'))
        self.assertEqual(result['summary']['kernel_count'], 2)
        self.assertEqual(result['summary']['device_sum_us'], 20)
        self.assertEqual(result['summary']['busy_us'], 20)
        self.assertTrue(result['summary']['serial_step_eligible'])
        predicted = predict(result, self.config())
        self.assertEqual(predicted['prediction']['coverage_pct'], 100)
        self.assertAlmostEqual(predicted['step_prediction']['ratio'], 1)
        self.assertTrue(all(predicted['audit'].values()))

    def test_unknown_does_not_become_one(self):
        profile = analyze(self.root, Scope(step='1'))
        config = self.config()
        config['target_hardware']['metrics'].pop('memory_bandwidth')
        result = predict(profile, config)
        self.assertEqual(result['prediction']['coverage_pct'], 0)
        self.assertIsNone(result['prediction']['modelled_device_ratio'])
        self.assertIsNone(result['step_prediction'])

    def test_scope_and_overlap(self):
        with self.assertRaises(ValueError):
            analyze(self.root, Scope())
        data = json.loads(EXAMPLE.read_text())
        data['traceEvents'][-1]['ts'] = 25
        (self.root / 'trace.json').write_text(json.dumps(data))
        result = analyze(self.root, Scope(step='1'))
        self.assertEqual(result['summary']['busy_us'], 15)
        self.assertFalse(result['summary']['serial_step_eligible'])

    def test_revision_and_validation(self):
        store = Store(self.root / 'db')
        payload = dict(name='test', vendor='Other', metrics={})
        first = store.save('hardware', payload)
        store.save('hardware', dict(payload, name='updated'), first['id'], 1)
        self.assertEqual(store.get('hardware', first['id'], 1)['name'], 'test')
        with self.assertRaises(Conflict):
            store.save('hardware', payload, first['id'], 1)
        with self.assertRaises(ValueError):
            Hardware.model_validate(dict(payload, metrics={'memory_bandwidth':dict(value=1,unit='GB/s')}))

    def test_large_trace_timestamps_and_dtype_provenance(self):
        data=json.loads(EXAMPLE.read_text())
        for event in data['traceEvents']:
            event['ts']+=2_845_214_366_879.589
            if event['cat']=='kernel':
                event['dur']=1.001
            if event['cat']=='cpu_op':
                event['args']['Input type']=['c10::Half','c10::Half']
        (self.root/'trace.json').write_text(json.dumps(data))
        result=analyze(self.root,Scope(step='1'))
        self.assertAlmostEqual(result['summary']['busy_us'],2.002,places=8)
        self.assertAlmostEqual(result['summary']['step_busy_us'],2.002,places=8)
        self.assertTrue(result['summary']['serial_step_eligible'])
        self.assertEqual(json.loads(result['units'][0]['input_dtypes']),['float16','float16'])
        self.assertEqual(json.loads(result['units'][0]['raw_input_dtypes']),['c10::Half','c10::Half'])

    def test_npu_csv_classification(self):
        (self.root / 'trace.json').unlink()
        (self.root / 'kernel_details.csv').write_text(
            'Name,Type,Duration(us),Start Time(us),Device ID,Stream ID,aicore_time(us),aiv_vec_time(us)\n'
            'add,Add,10,1,0,1,0,8\n'
            'mm,MatMulV3,20,20,0,1,18,2\n'
            'other,Unknown,5,50,0,1,,\n', encoding='utf-8')
        result=analyze(self.root,Scope(side='npu',all_steps=True))
        groups={r['category']:r for r in result['summary']['compute_breakdown']}
        self.assertEqual(groups['Vector']['time_us'],10)
        self.assertEqual(groups['Cube']['time_us'],20)
        self.assertEqual(groups['Unknown']['time_us'],5)
        self.assertIsNone(result['summary']['step_us'])

    def test_calibration_and_precision_guard(self):
        profile=analyze(self.root,Scope(step='1'))
        config=self.config()
        config['request']['calibrations']=[dict(signature=profile['units'][0]['signature'],
            source_us=10,target_us=5,evidence='synthetic test only')]
        result=predict(profile,config)
        self.assertEqual(result['prediction']['modelled_device_ratio'],2)
        config['request']['calibrations'][0]['signature']='not-in-profile'
        with self.assertRaises(ValueError):
            predict(profile,config)
        config['request']['calibrations']=[]
        for unit in profile['units']:
            unit.update(op='aten::mm',input_shapes='[[2,2],[2,2]]')
        result=predict(profile,config)
        self.assertEqual(result['prediction']['coverage_pct'],0)
        self.assertIn('模式未确认',result['units'][0]['reason'])

    def test_csv_metadata_and_safe_matrix_guard(self):
        (self.root / 'trace.json').unlink()
        sample=EXAMPLE.parent / 'kernel_details.csv'
        (self.root / 'kernel_details.csv').write_bytes(sample.read_bytes())
        profile=analyze(self.root,Scope(side='npu',step='1'))
        add=next(u for u in profile['units'] if u['op']=='aten::add')
        self.assertEqual(json.loads(add['input_shapes']),[[32,128],[32,128]])
        self.assertEqual(json.loads(add['input_dtypes']),['float32','float32'])
        self.assertEqual(add['raw_input_shapes'],'32,128;32,128')
        self.assertEqual(add['metadata_source'],'npu_kernel')
        self.assertEqual(extract_features(add)['logical_bytes'],32*128*4*3)
        result=predict(profile,self.config())
        self.assertEqual(result['prediction']['coverage_pct'],37.5)
        self.assertEqual(result['diagnostics'][0]['code'],'implementation')
        self.assertEqual(result['diagnostics'][0]['source_us'],50)
        self.assertTrue(result['audit']['diagnostic_time_conserved'])
        self.assertIsNone(result['step_prediction'])

    def test_output_mismatch_and_invalid_metadata_are_not_modelled(self):
        unit=dict(op='aten::add',input_shapes='[[4],[4]]',input_dtypes='["float32","float32"]',output_shapes='[[8]]')
        with self.assertRaises(ModelIssue) as error:
            extract_features(unit)
        self.assertEqual(error.exception.code,'implementation')
        unit.update(output_shapes='[[4]]',output_dtypes='["float16"]')
        with self.assertRaises(ModelIssue) as error:
            extract_features(unit)
        self.assertEqual(error.exception.code,'dtype')
        unit.update(input_shapes='[[null],[4]]')
        with self.assertRaises(ModelIssue) as error:
            extract_features(unit)
        self.assertEqual(error.exception.code,'metadata')

    def test_diagnostics_aggregate_all_units_not_only_top_twenty(self):
        profile=analyze(self.root,Scope(step='1'))
        source=profile['units'][0]
        profile['units']=[]
        for i in range(30):
            u=copy.deepcopy(source)
            u.update(id=str(i),op='aten::custom',signature=str(i))
            u['events'][0]['id']='test:'+str(i)
            profile['units'].append(u)
        profile['summary']['device_sum_us']=300
        result=predict(profile,self.config())
        self.assertEqual(len(result['risks']),20)
        self.assertEqual(result['diagnostics'][0]['unit_count'],30)
        self.assertEqual(result['diagnostics'][0]['source_us'],300)
        self.assertTrue(all(result['audit'].values()))

    def test_catalog_api_versions_and_form_assets(self):
        with TestClient(create_app(self.root/'catalog',test_mode=True)) as client:
            for name in ['catalog.js','diagnostics.js','assessment.js','workflow.js','forms.css']:
                self.assertEqual(client.get('/static/'+name).status_code,200)
            hw=client.post('/api/catalog/hardware',json={'name':'  custom  ','vendor':'Other','metrics':{}}).json()
            self.assertEqual(hw['name'],'custom')
            payload={k:v for k,v in hw.items() if k not in {'id','created','revision'}}
            platform=client.post('/api/catalog/platform',json={'name':'p','hardware_id':hw['id'],'hardware_revision':1}).json()
            self.assertEqual(client.put('/api/catalog/hardware/'+hw['id'],json={'expected_revision':1,'payload':dict(payload,name='custom v2')}).status_code,200)
            self.assertEqual(client.put('/api/catalog/hardware/'+hw['id'],json={'expected_revision':1,'payload':payload}).status_code,409)
            persisted=next(p for p in client.get('/api/catalog/platform').json() if p['id']==platform['id'])
            self.assertEqual(persisted['hardware_revision'],1)
            self.assertEqual(client.post('/api/catalog/hardware',json=dict(payload,name='   ')).status_code,422)

    def test_web_worker_end_to_end(self):
        app = create_app(self.root / 'web', test_mode=True)
        with TestClient(app) as client:
            self.assertEqual(client.get('/').status_code, 200)
            self.assertEqual(client.post('/api/jobs', json={}, headers={'Origin':'https://evil.example'}).status_code, 403)
            job = client.post('/api/jobs', json={'name':'example'}).json()['id']
            base = '/api/jobs/' + job
            self.assertEqual(client.put(base+'/inputs/trace.json', content=EXAMPLE.read_bytes()).status_code, 200)
            self.assertEqual(client.post(base+'/inspect',json={'side':'gpu'}).status_code, 200)
            self.wait_job(client, app, job, 'ready')
            platforms = client.get('/api/catalog/platform').json()
            hardware = client.get('/api/catalog/hardware').json()
            gpu = next(h for h in hardware if h['vendor']=='NVIDIA')
            platform = next(p for p in platforms if p['hardware_id']==gpu['id'])
            request = dict(source_platform_id=platform['id'], source_platform_revision=1,
                           target_platform_id=platform['id'], target_platform_revision=1,
                           scope={'step':'1'}, assumptions={'fixed_gap_serial_step':True})
            response = client.post(base+'/run',json=request)
            self.assertEqual(response.status_code,200,response.text)
            self.wait_job(client, app, job, 'completed')
            summary=client.get(base+'/summary').json()
            self.assertEqual(summary['prediction']['coverage_pct'],100)
            self.assertEqual(summary['schema_version'],4)
            self.assertEqual(summary['empirical']['status'],'available')
            self.assertTrue(summary['empirical']['audit']['source_time_conserved'])
            self.assertEqual(summary['assessment']['partial']['status'],'available')
            self.assertEqual(summary['assessment']['step']['status'],'available')
            self.assertEqual(summary['diagnostics'],[])
            self.assertEqual(summary['step_blockers'],[])
            self.assertEqual(client.get(base+'/groups').json()['total'],1)
            self.assertEqual(client.get(base+'/download/operators.csv').status_code,200)
            self.assertEqual(client.post(base+'/run',json=request).status_code,409)
            self.assertEqual(client.post(base+'/clone').status_code,200)

    def simple_request(self,client):
        hw=next(h for h in client.get('/api/catalog/hardware').json() if h['vendor']=='NVIDIA')
        platform=next(p for p in client.get('/api/catalog/platform').json() if p['hardware_id']==hw['id'])
        return dict(source_platform_id=platform['id'],source_platform_revision=platform['revision'],
                    target_platform_id=platform['id'],target_platform_revision=platform['revision'],scope={'step':'1'})

    def test_direct_analysis_without_inventory_and_missing_input(self):
        app=create_app(self.root/'direct',test_mode=True)
        with TestClient(app) as client:
            job=client.post('/api/jobs',json={'name':'direct'}).json()['id']
            base='/api/jobs/'+job
            request=self.simple_request(client)
            self.assertEqual(client.post(base+'/run',json=request).status_code,422)
            self.assertEqual(client.post(base+'/inspect',json={'side':'gpu'}).status_code,422)
            self.assertEqual(client.get(base).json()['status'],'draft')
            client.put(base+'/inputs/trace.json',content=EXAMPLE.read_bytes())
            no_scope=copy.deepcopy(request);no_scope['scope']={}
            self.assertEqual(client.post(base+'/run',json=no_scope).status_code,422)
            response=client.post(base+'/run',json=request)
            self.assertEqual(response.status_code,200,response.text)
            self.wait_job(client,app,job,'completed')
            self.assertEqual(client.get(base+'/inventory').status_code,404)
            self.assertEqual(client.get(base+'/summary').json()['profile_summary']['device_sum_us'],20)

    def test_direct_ambiguous_pid_requires_choice_and_can_retry(self):
        app=create_app(self.root/'retry',test_mode=True)
        with TestClient(app) as client:
            job=client.post('/api/jobs',json={}).json()['id'];base='/api/jobs/'+job
            data=json.loads(EXAMPLE.read_text());other=copy.deepcopy(data['traceEvents'][1]);other['pid']=9
            data['traceEvents'].append(other)
            client.put(base+'/inputs/trace.json',content=json.dumps(data).encode())
            request=self.simple_request(client)
            self.assertEqual(client.post(base+'/run',json=request).status_code,200)
            self.wait_job(client,app,job,'failed')
            self.assertIn('多个CPU PID',client.get(base).json()['error'])
            request['scope']['pid']='1'
            self.assertEqual(client.post(base+'/run',json=request).status_code,200)
            self.wait_job(client,app,job,'completed')
            self.assertEqual(client.get(base+'/summary').json()['config']['request']['scope']['pid'],'1')

    def test_failed_job_can_replace_input_without_inspection(self):
        app=create_app(self.root/'replace',test_mode=True)
        with TestClient(app) as client:
            job=client.post('/api/jobs',json={}).json()['id'];base='/api/jobs/'+job
            app.state.store.transition(job,{'draft'},'failed',error='synthetic failure')
            self.assertEqual(client.put(base+'/inputs/trace.json',content=EXAMPLE.read_bytes()).status_code,200)
            self.assertEqual(client.get(base).json()['status'],'draft')
            self.assertEqual(client.post(base+'/run',json=self.simple_request(client)).status_code,200)
            self.wait_job(client,app,job,'completed')

    def wait_job(self, client, app, job, expected):
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            status=client.get('/api/jobs/'+job).json()
            if status['status'] in {expected,'failed'} and not app.state.manager.active:
                self.assertEqual(status['status'],expected,status)
                return
            time.sleep(.1)
        self.fail('Worker did not finish')


if __name__ == '__main__':
    unittest.main()

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from profiling_web.app import create_app
from profiling_web.settings import Settings
from profiling_web import __main__ as launcher
from profiling_web.tests import test_web as fixtures


class LocalModeTest(unittest.TestCase):
    def test_local_mode_requires_loopback_host_peer_and_same_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(data_dir=Path(tmp), auth_mode="local")
            app = create_app(settings)
            self.assertFalse((Path(tmp) / "access-token").exists())
            with TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 1234)) as client:
                self.assertEqual(client.get('/api/config').json()['auth_mode'], 'local')
                self.assertEqual(client.get('/api/jobs').status_code, 200)
                self.assertEqual(client.post('/api/jobs', json={'name':'local'}).status_code, 403)
                self.assertEqual(client.post('/api/jobs', json={'name':'local'}, headers={'Origin':'https://evil.example'}).status_code, 403)
                self.assertEqual(client.post('/api/jobs', json={'name':'local'}, headers={'Origin':'http://127.0.0.1'}).status_code, 201)
                self.assertIn(client.get('/api/jobs', headers={'Host':'evil.example'}).status_code, (400,403))
                self.assertEqual(client.get('/api/jobs', headers={'X-Forwarded-For':'127.0.0.1'}).status_code, 403)
            with TestClient(app, base_url='http://127.0.0.1', client=('192.168.1.5',1234)) as client:
                self.assertEqual(client.get('/api/jobs').status_code, 403)

    def test_launcher_default_local_and_refuses_public_no_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with patch('sys.argv',['profiling_web','serve','--data-dir',tmp]), patch('uvicorn.run') as run:
                launcher.main()
                self.assertFalse(run.call_args.kwargs['proxy_headers'])
                self.assertEqual(run.call_args.args[0].state.settings.auth_mode, 'local')
            with patch('sys.argv',['profiling_web','serve','--host','0.0.0.0','--auth','local','--data-dir',tmp]):
                with self.assertRaises(SystemExit): launcher.main()


class UsabilityTest(unittest.TestCase):
    setUp = fixtures.WebTest.setUp
    tearDown = fixtures.WebTest.tearDown
    new = fixtures.WebTest.new
    upload = fixtures.WebTest.upload
    inspect = fixtures.WebTest.inspect
    complete = fixtures.WebTest.complete

    def test_rename_archive_and_search_pagination(self):
        job = self.new()
        for name in ('', '   ', 'a'*121):
            self.assertEqual(self.client.patch(f'/api/jobs/{job}',json={'name':name}).status_code,422)
        self.assertEqual(self.client.patch(f'/api/jobs/{job}',json={}).status_code,422)
        response=self.client.patch(f'/api/jobs/{job}',json={'name':'  新名字 <script>  '})
        self.assertEqual(response.json()['name'],'新名字 <script>')
        self.assertEqual(self.client.get('/api/jobs',params={'search':'新名字'}).json()['total'],1)
        for i in range(52): self.store.create(f'历史 {i}')
        self.assertEqual(self.client.get('/api/jobs',params={'offset':50,'limit':10}).json()['total'],53)
        self.assertEqual(len(self.client.get('/api/jobs',params={'offset':50,'limit':10}).json()['items']),3)
        self.client.patch(f'/api/jobs/{job}',json={'archived':True}).raise_for_status()
        self.assertEqual(self.client.get('/api/jobs',params={'search':'新名字'}).json()['total'],0)
        self.assertEqual(self.client.get('/api/jobs',params={'archived':True}).json()['total'],1)
        self.client.patch(f'/api/jobs/{job}',json={'archived':False}).raise_for_status()
        self.assertEqual(self.store.get(job)['archived'],0)

    def test_clone_reuses_verified_inputs_and_preserves_original(self):
        job=self.new();self.upload(job);self.inspect(job)
        self.store.update(job,options={'gpu_step':'1','npu_step':'1'})
        original=self.store.get(job)
        response=self.client.post(f'/api/jobs/{job}/clone')
        self.assertEqual(response.status_code,201,response.text)
        clone=response.json()
        self.assertNotEqual(clone['id'],job)
        self.assertEqual(clone['status'],'queued')
        self.assertEqual(clone['options'],original['options'])
        self.assertEqual(self.store.get(job),original)
        for role, meta in original['inputs'].items():
            from profiling_web.settings import INPUTS
            self.assertEqual((self.store.path(job)/'inputs'/INPUTS[role]).read_bytes(),(self.store.path(clone['id'])/'inputs'/INPUTS[role]).read_bytes())
        self.assertEqual(self.client.patch(f"/api/jobs/{clone['id']}",json={'archived':True}).status_code,409)
        self.client.delete(f'/api/jobs/{job}?confirm=true').raise_for_status()
        self.assertTrue((self.store.path(clone['id'])/'inputs'/'gpu.json').is_file())

    def test_clone_refuses_unfrozen_quota_and_changed_content(self):
        job=self.new();self.upload(job)
        self.assertEqual(self.client.post(f'/api/jobs/{job}/clone').status_code,409)
        self.inspect(job)
        with patch.object(self.settings,'storage_bytes',1):
            self.assertEqual(self.client.post(f'/api/jobs/{job}/clone').status_code,507)
        (self.store.path(job)/'inputs'/'gpu.json').write_bytes(b'bad')
        self.assertEqual(self.client.post(f'/api/jobs/{job}/clone').status_code,409)
        self.assertEqual(self.store.get(job)['status'],'needs_selection')

    def test_evidence_filters_match_type_filters_and_health_is_private(self):
        job=self.complete()
        url=f'/api/jobs/{job}/results'
        for kind in ('operators','mappings','diagnoses','reasons'):
            self.assertEqual(self.client.get(url,params={'kind':kind,'search':'does-not-exist'}).json()['total'],0)
            self.assertEqual(self.client.get(url,params={'kind':kind,'coverage_max':0}).json()['total'],0)
        health=self.client.get('/api/health').json()
        self.assertGreater(health['storage_used_bytes'],0)
        self.assertNotIn(str(self.root),str(health))
        self.assertEqual(self.client.get(f'/api/jobs/{job}/call-evidence',params={'call_id':'missing'}).json()['event_total'],0)
        self.client.headers.pop('Authorization')
        self.assertEqual(self.client.get('/api/health').status_code,401)

    def test_browser_profiler_step_label_matches_csv_step_id(self):
        job=self.new();self.upload(job);self.inspect(job)
        result=self.client.post(f'/api/jobs/{job}/run',json={'gpu_pid':'1','npu_pid':'3','gpu_step':'ProfilerStep#1','npu_step':'ProfilerStep#1'})
        self.assertEqual(result.status_code,202,result.text)
        self.assertEqual(result.json()['options']['npu_step'],'1')
        from profiling_web.worker import Worker
        Worker(self.settings).execute(self.store.claim())
        self.assertEqual(self.store.get(job)['status'],'succeeded')
        from profiling_web.app import RunOptions
        self.assertEqual(RunOptions(gpu_step='custom#1',npu_step='custom#1').npu_step,'custom#1')

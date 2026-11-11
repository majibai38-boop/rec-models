from pathlib import Path
import unittest
from unittest.mock import patch
from profiling_web.tests import test_web as base
from profiling_tools.operator_details import kernel_metadata, cpu_metadata
from profiling_web.tests.test_operator_details import case


class TraceViewerTest(unittest.TestCase):
    setUp, tearDown = base.WebTest.setUp, base.WebTest.tearDown
    new, upload, inspect, complete = base.WebTest.new, base.WebTest.upload, base.WebTest.inspect, base.WebTest.complete

    def test_scoped_trace_auth_version_and_no_reparse(self):
        ident=self.complete()
        job=self.store.get(ident)
        root=self.store.path(ident)/'attempts'/job['attempt']/'reports'
        with patch('profiling_tools.parser._load_trace_events',side_effect=AssertionError('GET cannot parse')):
            for side in ('npu','gpu'):
                url=f'/api/jobs/{ident}/trace-view/{side}'
                response=self.client.get(url,params={'attempt':job['attempt']})
                self.assertEqual(response.status_code,200)
                self.assertEqual(response.content,(root/(side+'_trace.json')).read_bytes())
                self.assertEqual(response.headers['cache-control'],'no-store')
                self.assertEqual(self.client.get(url,params={'attempt':'f'*32}).status_code,409)
            self.assertEqual(self.client.get(f'/api/jobs/{ident}/trace-view/other',params={'attempt':job['attempt']}).status_code,422)
        self.client.headers.pop('Authorization')
        self.assertEqual(self.client.get(f'/api/jobs/{ident}/trace-view/gpu',params={'attempt':job['attempt']}).status_code,401)
        # Browser transport uses the existing HTTP-only session, never URL tokens.
        self.client.post('/api/session',json={'token':self.settings.token},headers={'Origin':'http://testserver'}).raise_for_status()
        self.assertEqual(self.client.get(f'/api/jobs/{ident}/trace-view/gpu',params={'attempt':job['attempt']}).status_code,200)
        (root/'gpu_trace.json').unlink()
        self.assertEqual(self.client.get(f'/api/jobs/{ident}/trace-view/gpu',params={'attempt':job['attempt']}).status_code,404)

    def test_installation_requires_restart_to_mount(self):
        from fastapi.testclient import TestClient
        from profiling_web.app import create_app
        with patch('profiling_web.perfetto.installed',return_value=False):
            app=create_app(self.settings)
        with patch('profiling_web.perfetto.installed',return_value=True):
            client=TestClient(app,headers={'Authorization':'Bearer '+self.settings.token})
            self.assertFalse(client.get('/api/trace-viewer').json()['available'])

    def test_viewer_config_and_security_headers(self):
        result=self.client.get('/api/trace-viewer')
        self.assertEqual(result.status_code,200)
        config=result.json()
        if config['available']:
            response=self.client.get(config['url'])
            self.assertEqual(response.status_code,200)
            self.assertEqual(response.headers['x-frame-options'],'SAMEORIGIN')
            self.assertIn("frame-ancestors 'self'",response.headers['content-security-policy'])
            self.assertIn("connect-src 'self'",response.headers['content-security-policy'])
        self.assertEqual(self.client.get('/api/health').headers['x-frame-options'],'DENY')

    def test_trace_metadata_and_synthetic_events(self):
        g,n,_,_=case()
        self.assertTrue(kernel_metadata(n,0)['trace_present'])
        self.assertEqual(cpu_metadata(g,0)['time_us'],g.ops[0].event.duration_us)
        n.kernels[0].event.index=-1
        self.assertFalse(kernel_metadata(n,0)['trace_present'])

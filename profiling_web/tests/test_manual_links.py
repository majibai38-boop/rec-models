import json
from pathlib import Path
import unittest
from unittest.mock import patch

from profiling_tools.tests.test_manual_mapping import nested_fixture
from profiling_web.tests import test_web as base
from profiling_web.settings import INPUTS
from profiling_web.worker import Worker


class ManualWebTest(unittest.TestCase):
    setUp = base.WebTest.setUp
    tearDown = base.WebTest.tearDown
    new = base.WebTest.new
    inspect = base.WebTest.inspect
    complete = base.WebTest.complete

    def upload(self, ident):
        config = nested_fixture(self.root)
        for role in INPUTS:
            response = self.client.put(f'/api/jobs/{ident}/inputs/{role}',content=Path(getattr(config,role)).read_bytes())
            self.assertEqual(response.status_code,200,response.text)

    def panel(self, ident, **params):
        query = {'op_type':'IndexAdd','device_id':'0','core_type':'UNRESOLVED',**params}
        # The sample detail has no core column; use the exact published group.
        types = self.client.get(f'/api/jobs/{ident}/results').json()['items']
        row = next(r for r in types if r['op_type']=='IndexAdd')
        query['core_type'] = row['core_type']
        response = self.client.get(f'/api/jobs/{ident}/manual-options',params=query)
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def request_link(self, ident):
        panel = self.panel(ident)
        return {'attempt':panel['attempt'],'target_id':panel['target']['id'],
                'candidate_id':panel['candidates'][0]['id'],'reason':'人工核对 index_add_ 外层调用一致'}

    def execute(self):
        Worker(self.settings).execute(self.store.claim())

    def test_confirm_export_conservation_and_undo(self):
        ident = self.complete()
        old = self.store.get(ident)
        original = self.store.path(ident)/'attempts'/old['attempt']/'reports/npu_gpu_op_statistic.xlsx'
        original_bytes = original.read_bytes()
        with patch('profiling_tools.parser._load_trace_events',side_effect=AssertionError('GET must not parse traces')):
            panel = self.panel(ident)
        self.assertEqual(panel['candidates'][0]['recommendation'],'优先推荐')
        payload = self.request_link(ident)
        response = self.client.post(f'/api/jobs/{ident}/manual-links',json=payload)
        self.assertEqual(response.status_code,202,response.text)
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/manual-links',json=payload).status_code,409)
        self.execute()
        new = self.store.get(ident)
        self.assertEqual(new['status'],'succeeded',new['error'])
        self.assertEqual(new['error'],'')
        self.assertNotEqual(old['attempt'],new['attempt'])
        self.assertEqual(original_bytes,original.read_bytes())
        for k in ('gpu_total_time_us','npu_total_time_us'):
            self.assertEqual(old['summary']['gpu_accounting'][k],new['summary']['gpu_accounting'][k])
        row = next(r for r in self.client.get(f'/api/jobs/{ident}/results').json()['items'] if r['op_type']=='IndexAdd')
        self.assertEqual(row['manual_count'],1)
        self.assertEqual(row['coverage']['comparable']['time_pct'],100)
        self.assertEqual(row['coverage']['direct']['time_pct'],0)
        self.assertIn('人工确认',row['conclusion'])
        self.assertEqual(self.client.get(f'/api/jobs/{ident}/artifacts/npu_gpu_op_statistic.xlsx').content[:2],b'PK')
        self.assertEqual(self.client.get(f'/api/jobs/{ident}/artifacts/manual_catalog.json').status_code,404)
        self.assertEqual(len(self.client.get(f'/api/jobs/{ident}/artifacts/manual_links.json').json()),1)
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/manual-links',json=payload).status_code,409)
        panel = self.panel(ident)
        self.assertEqual(panel['target_total'],0)
        undo = self.client.post(f"/api/jobs/{ident}/manual-links/{panel['links'][0]['id']}/undo",json={'attempt':new['attempt']})
        self.assertEqual(undo.status_code,202,undo.text)
        self.execute()
        self.assertEqual(self.store.get(ident)['summary']['gpu_accounting'],old['summary']['gpu_accounting'])
        self.assertEqual(len(self.panel(ident)['links']),0)

    def test_invalid_payload_and_cancel_restore_previous(self):
        ident = self.complete()
        old = self.store.get(ident)
        payload = self.request_link(ident)
        for change in ({'reason':' '},{'candidate_id':'op:999'},{'gpu_indices':[0]}, {'attempt':'../evil'}):
            self.assertEqual(self.client.post(f'/api/jobs/{ident}/manual-links',json={**payload,**change}).status_code,422)
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/manual-links',json=payload).status_code,202)
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/cancel').status_code,200)
        restored = self.store.get(ident)
        self.assertEqual(restored['attempt'],old['attempt'])
        self.assertEqual(restored['status'],'succeeded')
        self.assertEqual(restored['manual'],{})

    def test_changed_input_failure_restores_previous_result(self):
        ident = self.complete()
        old = self.store.get(ident)
        self.client.post(f'/api/jobs/{ident}/manual-links',json=self.request_link(ident))
        gpu = self.store.path(ident)/'inputs/gpu.json'
        gpu.write_text(gpu.read_text(encoding='utf-8')+' ',encoding='utf-8')
        self.execute()
        restored = self.store.get(ident)
        self.assertEqual(restored['attempt'],old['attempt'])
        self.assertEqual(restored['status'],'succeeded')
        self.assertIn('未生效',restored['error'])
        self.assertEqual(restored['summary'],old['summary'])

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from profiling_tools.tests.test_manual_mapping import nested_fixture
from profiling_tools.compare import compare_profiles
from profiling_tools.inspection import inspect_inputs
from profiling_web.index import build_index, manual_options
from profiling_web.tests import test_web as web_tests
from profiling_web.settings import INPUTS
from profiling_web.worker import Worker


class DeviceGroupsTest(unittest.TestCase):
    def test_no_cpu_multistream_group_compare_and_undo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = nested_fixture(root, count=26)
            for side in ('gpu', 'npu'):
                path = root/f'{side}.json'
                payload = json.loads(path.read_text(encoding='utf-8'))
                payload['traceEvents'] = [e for e in payload['traceEvents'] if e['cat'] != 'cpu_op']
                for i, event in enumerate(payload['traceEvents']):
                    if event['cat'] == 'kernel' and side == 'gpu':
                        event['args']['stream'] = i % 2 + 7
                path.write_text(json.dumps(payload), encoding='utf-8')
            initial = compare_profiles(config, manual_enabled=True)
            self.assertEqual(initial['validation']['status'], 'valid')
            base = json.loads((root/'out/run_manifest.json').read_text(encoding='utf-8'))
            cat = json.loads((root/'out/manual_catalog.json').read_text(encoding='utf-8'))
            self.assertEqual(len(cat['targets']), 1)
            self.assertEqual(len(cat['candidates']), 1)
            target, candidate = cat['targets'][0], cat['candidates'][0]
            self.assertEqual(target['count'], 26)
            self.assertEqual(candidate['count'], 26)
            self.assertEqual(len(candidate['stream_counts']), 2)
            self.assertEqual(target['owner_diagnosis'], 'cpu_events_absent')
            build_index(root/'out', root/'index.sqlite')
            options = manual_options(root/'index.sqlite',target['op_type'],target['device_id'],target['core_type'])
            self.assertEqual(options['candidates'][0]['recommendation'], '整组候选，需人工确认')
            link = dict(id='group', reason='verified same workload', basis='manual_device_group',
                        npu_indices=target['indices'], gpu_indices=candidate['indices'])
            after = compare_profiles(replace(config, output_dir=str(root/'after')), manual_enabled=True,
                                     manual_links=[link], manual_base=base)
            self.assertEqual(after['validation']['status'], 'valid')
            self.assertEqual(after['comparison']['npu_first_comparable_kernel_count'],26)
            self.assertEqual(after['gpu_accounting']['gpu_total_time_us'],initial['gpu_accounting']['gpu_total_time_us'])
            undo = compare_profiles(replace(config, output_dir=str(root/'undo')), manual_enabled=True, manual_base=base)
            self.assertEqual(undo['comparison'], initial['comparison'])
            for side in ('gpu', 'npu'):
                path=root/f'{side}.json'
                payload=json.loads(path.read_text(encoding='utf-8'))
                payload['traceEvents']=[e for e in payload['traceEvents'] if e['cat']=='kernel']
                path.write_text(json.dumps(payload),encoding='utf-8')
            full=replace(config,gpu_step='',npu_step='',all_steps=True,output_dir=str(root/'full'))
            self.assertEqual(inspect_inputs(full)['status'],'ready')
            self.assertEqual(compare_profiles(full,manual_enabled=True)['validation']['status'],'valid')


class DeviceGroupsWebTest(unittest.TestCase):
    setUp, tearDown = web_tests.WebTest.setUp, web_tests.WebTest.tearDown
    new, inspect, complete = web_tests.WebTest.new, web_tests.WebTest.inspect, web_tests.WebTest.complete

    def upload(self, ident):
        config = nested_fixture(self.root, count=26)
        for role in INPUTS:
            content = Path(getattr(config, role)).read_bytes()
            if role.endswith('trace'):
                payload = json.loads(content)
                payload['traceEvents'] = [e for e in payload['traceEvents'] if e['cat'] != 'cpu_op']
                content = json.dumps(payload).encode()
            self.assertEqual(self.client.put(f'/api/jobs/{ident}/inputs/{role}', content=content).status_code,200)

    def test_web_confirmation_and_undo(self):
        ident = self.complete()
        job = self.store.get(ident)
        row = next(r for r in self.client.get(f'/api/jobs/{ident}/results').json()['items'] if r['op_type']=='IndexAdd')
        options = self.client.get(f'/api/jobs/{ident}/manual-options',params={k:row[k] for k in ('op_type','device_id','core_type')}).json()
        body = dict(attempt=job['attempt'], target_id=options['target']['id'],
                    candidate_id=options['candidates'][0]['id'],reason='人工核对整个采集范围等价')
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/manual-links',json=body).status_code,422)
        body['confirm_device_group']=True
        self.assertEqual(self.client.post(f'/api/jobs/{ident}/manual-links',json=body).status_code,202)
        Worker(self.settings).execute(self.store.claim())
        after=self.store.get(ident)
        self.assertEqual(after['status'],'succeeded',after['error'])
        link=after['manual']['links'][0]
        self.assertEqual(link['basis'],'manual_device_group')
        self.assertEqual(len(link['npu_indices']),26)
        self.assertEqual(len(link['gpu_indices']),26)
        response=self.client.post(f"/api/jobs/{ident}/manual-links/{link['id']}/undo",json={'attempt':after['attempt']})
        self.assertEqual(response.status_code,202)
        Worker(self.settings).execute(self.store.claim())
        self.assertEqual(self.store.get(ident)['summary']['gpu_accounting'],job['summary']['gpu_accounting'])

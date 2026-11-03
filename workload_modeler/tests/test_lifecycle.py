"""Recycling, revision preservation and additive catalog upgrades."""
import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

from fastapi.testclient import TestClient
from workload_modeler.app import create_app
from workload_modeler.presets import device_presets
from workload_modeler.store import Store, seed


def payload(item):
    return {k:v for k,v in item.items() if k not in {'id','revision','created','deleted_at'}}


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'data'/'tests'
        root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=root)
        self.app=create_app(self.temp.name,test_mode=True)
        self.client=TestClient(self.app)
        self.client.__enter__()
        self.store=self.app.state.store

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.temp.cleanup()

    def delete(self,kind,item):
        return self.client.delete(f'/api/catalog/{kind}/{item["id"]}',params={'confirm':True,'expected_revision':item['revision']})

    def restore(self,kind,item):
        return self.client.post(f'/api/catalog/{kind}/{item["id"]}/restore',params={'expected_revision':item['revision']})

    def test_task_roundtrip_retains_bytes_and_snapshot(self):
        job=self.store.create_job('保留快照');key=job['id'];base='/api/jobs/'+key
        folder=self.store.job_dir(key)
        (folder/'trace.json').write_bytes(b'original input')
        (folder/'result.json').write_bytes(b'{"saved":true}')
        config={'immutable':'snapshot'}
        self.store.transition(key,{'draft'},'completed',config=config)
        self.assertEqual(self.client.delete(base).status_code,422)
        self.assertEqual(self.client.delete(base+'?confirm=true').status_code,200)
        self.assertNotIn(key,[j['id'] for j in self.client.get('/api/jobs').json()])
        self.assertEqual(self.client.get('/api/jobs?deleted=true').json()[0]['id'],key)
        for suffix in ('','/summary','/inventory','/download/result.json'):
            self.assertEqual(self.client.get(base+suffix).status_code,404)
        self.assertEqual(self.client.patch(base,json={'name':'no'}).status_code,404)
        self.assertEqual(self.client.post(base+'/clone').status_code,404)
        self.assertEqual(self.client.put(base+'/inputs/trace.json',content=b'bad').status_code,404)
        self.assertEqual((folder/'trace.json').read_bytes(),b'original input')
        self.assertEqual((folder/'result.json').read_bytes(),b'{"saved":true}')
        restored=self.client.post(base+'/restore').json()
        self.assertEqual(restored['config'],config)
        self.assertEqual(restored['status'],'completed')
        self.assertIsNone(restored['deleted_at'])
        self.assertEqual(self.client.get(base+'/download/result.json').content,b'{"saved":true}')
        self.assertEqual(self.client.get('/api/jobs?deleted=true').json(),[])

    def test_busy_and_live_process_cannot_be_recycled(self):
        for status in ('running','inspecting','uploading'):
            job=self.store.create_job(status);self.store.transition(job['id'],{'draft'},status)
            self.assertEqual(self.client.delete('/api/jobs/'+job['id']+'?confirm=true').status_code,409)
        job=self.store.create_job('process not released')
        self.app.state.manager.active[job['id']]=object()
        try:
            self.assertEqual(self.client.delete('/api/jobs/'+job['id']+'?confirm=true').status_code,409)
        finally:
            self.app.state.manager.active.clear()

    def test_catalog_dependency_guard_and_restore_order(self):
        platform=self.store.list('platform')[0];hardware=self.store.get('hardware',platform['hardware_id'])
        before=self.store.history('hardware',hardware['id'])
        self.assertEqual(self.delete('hardware',hardware).status_code,409)
        self.assertEqual(self.delete('platform',platform).status_code,200)
        self.assertEqual(self.delete('hardware',hardware).status_code,200)
        self.assertEqual(self.restore('platform',platform).status_code,409)
        self.assertEqual(self.store.history('hardware',hardware['id']),before)
        self.assertEqual(self.client.put('/api/catalog/hardware/'+hardware['id'],json={'payload':payload(hardware),'expected_revision':1}).status_code,409)
        self.assertEqual(self.client.post('/api/catalog/platform',json=payload(platform)).status_code,409)
        self.assertEqual(self.restore('hardware',hardware).status_code,200)
        self.assertEqual(self.restore('platform',platform).status_code,200)
        self.assertEqual(self.store.get('platform',platform['id'])['revision'],platform['revision'])

    def test_catalog_stale_revision_and_confirmation(self):
        hw=self.store.save('hardware',dict(name='test',vendor='Other'))
        url='/api/catalog/hardware/'+hw['id']
        self.assertEqual(self.client.delete(url+'?expected_revision=1').status_code,422)
        self.store.save('hardware',dict(name='new',vendor='Other'),hw['id'],1)
        self.assertEqual(self.delete('hardware',hw).status_code,409)
        self.assertEqual(self.client.delete(url+'?confirm=true&expected_revision=0').status_code,422)

    def test_deleted_platform_is_not_usable_for_new_run(self):
        platform=next(p for p in self.store.list('platform') if p['name'].startswith('A100'))
        job=self.store.create_job('no stale platform');base='/api/jobs/'+job['id']
        self.client.put(base+'/inputs/trace.json',content=b'{"traceEvents":[]}')
        self.delete('platform',platform)
        request=dict(source_platform_id=platform['id'],target_platform_id=platform['id'],
                     source_platform_revision=1,target_platform_revision=1,scope={'step':'1'})
        self.assertEqual(self.client.post(base+'/run',json=request).status_code,409)
        self.assertEqual(self.store.job(job['id'])['status'],'draft')

    def test_search_and_pagination_beyond_two_hundred(self):
        for i in range(205):self.store.create_job(f'任务{i:03}')
        first=self.client.get('/api/jobs?limit=200').json()
        last=self.client.get('/api/jobs?offset=200').json()
        self.assertEqual(len(first),200);self.assertEqual(len(last),5)
        self.assertEqual(len({j['id'] for j in first+last}),205)
        found=self.client.get('/api/jobs?search=任务000').json()
        self.assertEqual(len(found),1)
        self.client.delete('/api/jobs/'+found[0]['id']+'?confirm=true')
        self.assertEqual(self.client.get('/api/jobs?search=任务000').json(),[])
        self.assertEqual(len(self.client.get('/api/jobs?deleted=true&search=任务000').json()),1)
        for query in ('offset=-1','limit=202','limit=0'):
            self.assertEqual(self.client.get('/api/jobs?'+query).status_code,422)

    def test_cross_origin_deletion_is_rejected(self):
        job=self.store.create_job('protected')
        response=self.client.delete('/api/jobs/'+job['id']+'?confirm=true',headers={'Origin':'https://example.org'})
        self.assertEqual(response.status_code,403)
        self.assertEqual(self.store.job(job['id'])['name'],'protected')

    def test_seed_preserves_edits_and_never_revives_trash(self):
        self.assertEqual(len(self.store.list('hardware')),10)
        self.assertEqual(len(self.store.list('platform')),10)
        platform=next(p for p in self.store.list('platform') if p['name'].startswith('A100'))
        hw=self.store.get('hardware',platform['hardware_id'])
        edited=self.store.save('hardware',dict(payload(hw),name='My measured A100'),hw['id'],1)
        seed(self.store)
        self.assertEqual(self.store.get('hardware',hw['id'])['name'],'My measured A100')
        self.delete('platform',platform);self.delete('hardware',edited)
        seed(self.store)
        self.assertEqual(len(self.store.list('hardware')),9)
        self.assertEqual(len(self.store.list('platform')),9)
        self.assertEqual(self.store.list('hardware',deleted=True)[0]['revision'],2)


class UpgradeTests(unittest.TestCase):
    def test_legacy_database_migration_and_additive_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            db=Path(directory)/'modeler.sqlite'
            with closing(sqlite3.connect(db)) as c:
                with c:
                    c.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY,name TEXT,status TEXT,created TEXT,error TEXT,config TEXT)')
                    c.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)',('a'*32,'old','completed','2026-01-01',None,json.dumps({'old':True})))
            store=Store(directory)
            custom=store.save('hardware',dict(name='Custom untouched',vendor='Other'))
            seed(store);seed(store)
            self.assertEqual(store.get('hardware',custom['id']),custom)
            self.assertEqual(len(store.list('hardware')),8)
            self.assertEqual(store.job('a'*32)['config'],{'old':True})
            self.assertIsNone(store.job('a'*32)['deleted_at'])

    def test_preset_variants_and_unknowns(self):
        presets=list(device_presets());self.assertEqual(len(presets),7)
        for key,hw,p in presets:
            m=hw['metrics'];self.assertTrue(m['matrix_fp16']['dense'])
            if key.startswith('a100'):
                self.assertEqual(m['matrix_fp16']['value'],312)
                self.assertEqual(m['matrix_tf32']['value'],156)
                self.assertEqual(m['vector_bf16']['value'],39)
                self.assertNotIn('cpu_model',p)
                expected=1935 if '80-pcie' in key else 2039 if '80-sxm' in key else 1555
                self.assertEqual(m['memory_bandwidth']['value'],expected)
            else:
                cores=m['cube_count']['value']
                self.assertEqual(m['matrix_fp16']['value'],cores*13.5)
                self.assertEqual(m['vector_count']['value'],cores*2)
                self.assertEqual(m['memory_bandwidth']['value'],4000)
                self.assertIsNone(m['memory_capacity']['value'])
                if cores==28:self.assertEqual(p['dimm_count'],24)


if __name__=='__main__':unittest.main()

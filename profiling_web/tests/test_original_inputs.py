import tempfile
from pathlib import Path
import unittest
from urllib.parse import quote, unquote
from fastapi.testclient import TestClient

from profiling_web.app import create_app
from profiling_web.settings import Settings, INPUTS


class OriginalInputsTests(unittest.TestCase):
    def test_original_bytes_names_and_auth(self):
        with tempfile.TemporaryDirectory() as tmp:
            token='original-test-token-1234567890'
            app=create_app(Settings(data_dir=Path(tmp),token=token,auth_mode='token',allowed_hosts=['testserver']))
            with TestClient(app,headers={'Authorization':'Bearer '+token}) as c:
                job=c.post('/api/jobs',json={'name':'inputs'}).json()['id']
                for role in INPUTS:
                    name='原始 '+INPUTS[role]
                    content=b'\xef\xbb\xbfName,Value\r\n=original,"a,b"\r\n' if role=='npu_kernel_details' else b'{"traceEvents":[], "unmodified": true}\n'
                    url=f'/api/jobs/{job}/inputs/{role}'
                    self.assertEqual(c.put(url,content=content,headers={'x-filename':quote('../../'+name)}).status_code,200)
                    r=c.get(url)
                    self.assertEqual(r.status_code,200)
                    self.assertEqual(r.content,content)
                    self.assertIn(name,unquote(r.headers['content-disposition']))
                    self.assertNotIn('../',r.headers['content-disposition'])
                    self.assertEqual(r.headers['content-type'],'application/octet-stream')
                c.headers.pop('authorization')
                self.assertEqual(c.get(url).status_code,401)

    def test_missing_uploading_and_modified_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=create_app(Settings(data_dir=Path(tmp),auth_mode='none',allowed_hosts=['testserver']))
            with TestClient(app,headers={'Origin':'http://testserver'}) as c:
                job=c.post('/api/jobs',json={'name':'inputs'}).json()['id']
                url=f'/api/jobs/{job}/inputs/gpu_trace'
                self.assertEqual(c.get(url).status_code,404)
                self.assertEqual(c.get(f'/api/jobs/{job}/inputs/access-token').status_code,404)
                c.put(url,content=b'{}')
                store=app.state.store
                store.update(job,uploading=1)
                self.assertEqual(c.get(url).status_code,409)
                store.update(job,uploading=0)
                path=store.path(job)/'inputs'/INPUTS['gpu_trace']
                path.write_bytes(b'changed')
                self.assertEqual(c.get(url).status_code,409)
                path.unlink()
                self.assertEqual(c.get(url).status_code,404)

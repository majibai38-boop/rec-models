import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from profiling_web.app import create_app as profiling_app
from profiling_web.settings import Settings
from profiling_web import __main__ as profiling_cli
from workload_modeler.app import create_app as workload_app
from workload_modeler import __main__ as workload_cli


class NetworkAccessTests(unittest.TestCase):
    def test_remote_none_and_token_cookie_bearer_and_origin(self):
        token='test-secret-'+'a'*32
        for kind in ('profiling','workload'):
            for mode in ('none','token'):
                with self.subTest(kind=kind,mode=mode), tempfile.TemporaryDirectory() as tmp:
                    app=(profiling_app(Settings(data_dir=Path(tmp),auth_mode=mode,token=token,allowed_hosts=['192.168.1.10']))
                         if kind=='profiling' else workload_app(tmp,auth_mode=mode,token=token,allowed_hosts=['192.168.1.10']))
                    origin='http://192.168.1.10:8770'
                    with TestClient(app,base_url=origin,client=('192.168.1.20',4567)) as c:
                        self.assertEqual(c.get('/api/jobs').status_code,401 if mode=='token' else 200)
                        self.assertEqual(c.post('/api/jobs',json={'name':'bad'},headers={'Origin':'http://evil.example'}).status_code,403)
                        self.assertEqual(c.get('/api/jobs',headers={'Host':'wrong.example'}).status_code,400 if mode=='none' else 401)
                        if mode=='token':
                            self.assertEqual(c.post('/api/session',json={'token':'wrong'},headers={'Origin':origin}).status_code,401)
                            self.assertEqual(c.get('/api/jobs',headers={'Authorization':'Bearer '+token}).status_code,200)
                            r=c.post('/api/session',json={'token':token},headers={'Origin':origin})
                            self.assertEqual(r.status_code,200)
                            self.assertIn('httponly',r.headers['set-cookie'].lower())
                            self.assertEqual(c.get('/api/jobs').status_code,200)
                        r=c.post('/api/jobs',json={'name':'remote task'},headers={'Origin':origin})
                        self.assertIn(r.status_code,(200,201))
                        self.assertNotIn(token,r.text)
                        if mode=='token':
                            c.delete('/api/session',headers={'Origin':origin})
                            self.assertEqual(c.get('/api/jobs').status_code,401)

    def test_workload_local_rejects_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            with TestClient(workload_app(tmp),base_url='http://127.0.0.1',client=('192.168.1.20',4567)) as c:
                self.assertEqual(c.get('/api/jobs').status_code,403)

    def test_launchers_explicit_none_and_default_fail_closed(self):
        for module in (profiling_cli,workload_cli):
            with self.subTest(module=module.__name__),tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{},clear=True):
                argv=['service']+(['serve'] if module is profiling_cli else [])+['--host','0.0.0.0','--auth','none','--data-dir',tmp]
                with patch('sys.argv',argv),patch('uvicorn.run') as run:
                    module.main()
                    self.assertEqual(run.call_args.kwargs['host'],'0.0.0.0')
                    self.assertFalse(run.call_args.kwargs['proxy_headers'])
                argv[argv.index('none')]='auto'
                with patch('sys.argv',argv),self.assertRaises(SystemExit):
                    module.main()

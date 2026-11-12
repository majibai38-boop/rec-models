import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from profiling_web.app import create_app
from profiling_web.settings import Settings


class StorageAccessTest(unittest.TestCase):
    def test_existing_jobs_directory_is_probed(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=Settings(data_dir=Path(tmp),auth_mode='none')
            with patch('profiling_web.settings.os.scandir',side_effect=PermissionError('denied')):
                with self.assertRaisesRegex(ValueError,'任务目录无法读写'):
                    settings.prepare()

    def test_storage_errors_are_actionable_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=create_app(Settings(data_dir=Path(tmp),auth_mode='none',allowed_hosts=['testserver']))
            with TestClient(app,headers={'Origin':'http://testserver'}) as client:
                with patch.object(app.state.store,'create',side_effect=PermissionError('private path')):
                    response=client.post('/api/jobs',json={'name':'test'})
                    self.assertEqual(response.status_code,503)
                    self.assertIn('目录',response.json()['detail'])
                    self.assertNotIn('private path',response.text)
                with patch.object(app.state.store,'list',side_effect=sqlite3.OperationalError('private path')):
                    response=client.get('/api/jobs')
                    self.assertEqual(response.status_code,503)
                    self.assertIn('数据库',response.json()['detail'])
                job=app.state.store.create('missing index')
                app.state.store.update(job['id'],status='succeeded',attempt='a'*32,summary={'test':True})
                response=client.get(f"/api/jobs/{job['id']}/results")
                self.assertEqual(response.status_code,409)
                self.assertIn('索引',response.json()['detail'])

import gzip
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from workload_modeler.app import create_app
from workload_modeler.profile import materialize_trace


class UploadLimitsTests(unittest.TestCase):
    def test_upload_boundary_and_failed_replacement(self):
        with tempfile.TemporaryDirectory() as root:
            app = create_app(root, test_mode=True, upload_limit_mb=1, expanded_limit_mb=2)
            with TestClient(app) as client:
                meta = client.get('/api/meta').json()
                self.assertEqual(meta['upload_limit_mb'], 1)
                self.assertEqual(meta['expanded_limit_mb'], 2)
                key = client.post('/api/jobs', json={'name': 'limit test'}).json()['id']
                url = f'/api/jobs/{key}/inputs/trace.json'
                content = b' ' * (1024**2)
                self.assertEqual(client.put(url, content=content).status_code, 200)
                result = client.put(url, content=content+b' ')
                self.assertEqual(result.status_code, 413)
                self.assertIn('1 MiB', result.json()['detail'])
                directory = app.state.store.job_dir(key)
                self.assertEqual((directory/'trace.json').read_bytes(), content)
                self.assertFalse((directory/'trace.json.upload').exists())
                self.assertEqual(client.get(f'/api/jobs/{key}').json()['status'], 'draft')

    def test_expansion_limit_and_cleanup(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'trace.json.gz'
            path.write_bytes(gzip.compress(b' '*(1024**2+1)))
            with patch.dict(os.environ, {'WORKLOAD_MODELER_EXPANDED_MB': '1'}):
                with self.assertRaisesRegex(ValueError, '1 MiB'):
                    materialize_trace(path)
                self.assertFalse((path.parent/'expanded.json').exists())
            with patch.dict(os.environ, {'WORKLOAD_MODELER_EXPANDED_MB': '2'}):
                    self.assertEqual(materialize_trace(path).stat().st_size, 1024**2+1)

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from profiling_web.app import create_app
from profiling_web.settings import Settings
from profiling_web.worker import Worker


class NpuPairWebTest(unittest.TestCase):
    def test_upload_compare_export_and_clone(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Settings(auth_mode="none", data_dir=Path(tmp), allowed_hosts=["testserver"])
            app = create_app(cfg)
            worker = Worker(cfg)
            with TestClient(app, headers={"Origin": "http://testserver"}) as client:
                created = client.post("/api/jobs",json={"mode":"npu_pair","name":"NPU regression"})
                self.assertEqual(created.status_code, 201, created.text)
                job = created.json()
                base = "/api/jobs/"+job["id"]
                self.assertEqual(client.put(base+"/inputs/gpu_trace",content=b"unknown").status_code,404)
                self.assertEqual(client.post(base+"/inspect").status_code,409)
                for role,t in (("baseline_details",4),("candidate_details",6)):
                    data = f"Type,Step Id,Duration(us)\nAdd,15,{t}\n"
                    response = client.put(base+"/inputs/"+role, content=data,
                                          headers={"X-Filename": "kernel_details.csv"})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["name"], "kernel_details.csv")
                self.assertEqual(client.post(base+"/inspect").status_code,202)
                worker.execute(app.state.store.claim())
                self.assertEqual(client.get(base).json()["inspection"]["baseline"]["steps"],{"15":1})
                self.assertEqual(client.post(base+"/run",json={"all_steps":True}).status_code,409)
                self.assertEqual(client.post(base+"/run-npu-pair",json={"baseline_step":"15","candidate_step":"15"}).status_code,202)
                worker.execute(app.state.store.claim())
                result = client.get(base).json()
                self.assertEqual(result["status"], "succeeded", result.get("error"))
                self.assertEqual(result["summary"]["overall"]["耗时比(对比/基线)"],1.5)
                self.assertTrue(client.get(base+"/artifacts/npu_pair.xlsx").content.startswith(b"PK"))
                for role, t in (("baseline_details", 4), ("candidate_details", 6)):
                    self.assertEqual(client.get(base+"/inputs/"+role).text,
                                     f"Type,Step Id,Duration(us)\nAdd,15,{t}\n")
                self.assertEqual(client.get(base+"/artifacts/gpu_trace.json").status_code,404)
                cloned = client.post(base+"/clone").json()
                self.assertEqual(cloned["mode"],"npu_pair")
                self.assertEqual(cloned["inputs"],result["inputs"])

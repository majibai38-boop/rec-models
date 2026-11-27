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
                self.assertEqual(client.post(base+"/run-npu-pair",json={"baseline_step":"15","candidate_step":"15",
                                                                        "cube_threshold":3,"vector_threshold":1.5}).status_code,202)
                worker.execute(app.state.store.claim())
                result = client.get(base).json()
                self.assertEqual(result["status"], "succeeded", result.get("error"))
                self.assertAlmostEqual(result["summary"]["overall"]["性能比(基线/对比)"],4/6,places=6)
                self.assertEqual(result["summary"]["candidate"]["cv"]["Unknown"]["time_pct"], 100)
                self.assertEqual([c["category"] for c in result["summary"]["candidate"]["pipeline"]["categories"]], ["unknown"])
                self.assertEqual(result["summary"]["pipeline"]["method"], "pipe_utilization_v2")
                self.assertEqual(result["summary"]["thresholds"]["cube"], 3.0)
                self.assertEqual(result["summary"]["thresholds"]["vector"], 1.5)
                self.assertEqual(client.post(base+"/run-npu-pair",json={"baseline_step":"15","candidate_step":"15",
                                                                        "cube_threshold":0}).status_code,422)
                self.assertTrue(result["summary"]["conclusions"])
                page = client.get(base+"/npu-pair-table").json()
                self.assertEqual(page["total"], 1)
                self.assertEqual(client.get(base+"/npu-pair-table",params={"search":"ad","sort":"candidate"}).json()["total"],1)
                self.assertEqual(client.get(base+"/npu-pair-table",params={"search":"other","sort":"ratio"}).json()["total"],0)
                self.assertEqual(client.get(base+"/npu-pair-table",params={"view":"signatures","op":"Add","search":"float32"}).json()["total"],0)
                self.assertEqual(page["items"][0]["CV"], "Unknown")
                params = {"view":"signatures", "op":"Add", "core":""}
                self.assertEqual(client.get(base+"/npu-pair-table",params=params).json()["items"][0]["对比耗时(us)"],6)
                self.assertFalse(client.get(base+"/npu-pair-table",params={**params,"offset":1}).json()["items"])
                self.assertTrue(client.get(base+"/artifacts/npu_pair.xlsx").content.startswith(b"PK"))
                pipeline = client.get(base+"/npu-pair-table", params={"view": "pipeline"}).json()
                self.assertEqual(pipeline["total"], 1)
                self.assertEqual([c["category"] for c in pipeline["items"][0]["对比"]["categories"]], ["unknown"])
                self.assertEqual(client.get(base+"/npu-pair-table",
                                            params={"view": "pipeline", "search": "add"}).json()["total"], 1)
                self.assertEqual(client.get(base+"/npu-pair-table",
                                            params={"view": "pipeline", "search": "other"}).json()["total"], 0)
                missed = client.get(base+"/npu-pair/below-threshold.xlsx", params={"cube": 3, "vector": 1.5})
                self.assertEqual(missed.status_code, 200, missed.text[:200])
                self.assertTrue(missed.content.startswith(b"PK"))
                self.assertEqual(client.get(base+"/npu-pair/below-threshold.xlsx",
                                            params={"cube": 0}).status_code, 422)
                for role, t in (("baseline_details", 4), ("candidate_details", 6)):
                    self.assertEqual(client.get(base+"/inputs/"+role).text,
                                     f"Type,Step Id,Duration(us)\nAdd,15,{t}\n")
                self.assertEqual(client.get(base+"/artifacts/gpu_trace.json").status_code,404)
                cloned = client.post(base+"/clone").json()
                self.assertEqual(cloned["mode"],"npu_pair")
                self.assertEqual(cloned["inputs"],result["inputs"])

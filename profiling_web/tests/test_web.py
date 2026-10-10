import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from profiling_tools.tests.test_skill_interfaces import fixture
from profiling_web.app import create_app, csv_download, session_cookie, valid_cookie
from profiling_web.index import build_index, query_index
from profiling_web.settings import INPUTS, Settings
from profiling_web.store import Conflict, Store
from profiling_web.worker import Worker, worker_lock


class WebTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(data_dir=self.root / "data", token="test-only-not-a-real-secret-12345", allowed_hosts=["testserver"])
        self.app = create_app(self.settings)
        self.client = TestClient(self.app, headers={"Authorization": "Bearer " + self.settings.token})
        self.client.__enter__()
        self.store = self.app.state.store

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def new(self):
        response = self.client.post("/api/jobs", json={"name": "中文测试 <script>"})
        self.assertEqual(response.status_code, 201, response.text)
        return response.json()["id"]

    def upload(self, job_id):
        config = fixture(self.root, extra_gpu=True)
        for role in INPUTS:
            content = Path(getattr(config, role)).read_bytes()
            response = self.client.put(f"/api/jobs/{job_id}/inputs/{role}", content=content, headers={"x-filename": "../../" + INPUTS[role]})
            self.assertEqual(response.status_code, 200, response.text)

    def inspect(self, job_id):
        response = self.client.post(f"/api/jobs/{job_id}/inspect")
        self.assertEqual(response.status_code, 202, response.text)
        Worker(self.settings).execute(self.store.claim())
        state = self.store.get(job_id)
        self.assertEqual(state["status"], "needs_selection", state)

    def complete(self):
        job_id = self.new()
        self.upload(job_id)
        self.inspect(job_id)
        response = self.client.post(f"/api/jobs/{job_id}/run", json={"gpu_step": "1", "npu_step": "1"})
        self.assertEqual(response.status_code, 202, response.text)
        Worker(self.settings).execute(self.store.claim())
        state = self.store.get(job_id)
        if state["status"] != "succeeded":
            log = self.store.path(job_id) / "attempts" / state["attempt"] / "worker.log"
            self.fail(log.read_text(encoding="utf-8") + repr(state))
        return job_id

    def test_end_to_end_upload_inspect_compare_query_download(self):
        job_id = self.complete()
        job = self.client.get(f"/api/jobs/{job_id}").json()
        self.assertEqual(job["summary"]["status"], "complete_with_gaps")
        self.assertEqual(job["summary"]["overall_statistics"][0]["total_time_ratio_gpu_npu"], 1.25)
        core = job["summary"]["core_breakdown"]
        self.assertEqual(core["npu"]["groups"][2]["count"], 1)
        self.assertIsNone(core["npu"]["aicore_time_us"])
        self.assertEqual(core["gpu_reference"]["groups"][3]["time_us"], 2)
        self.assertEqual(core["gpu_reference"]["groups"][4]["time_us"], 3)
        self.assertNotIn(str(self.root), json.dumps(job))
        result = self.client.get(f"/api/jobs/{job_id}/results", params={"search": "Add", "limit": 1}).json()
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["items"][0]["comparable_ratio_gpu_npu"], .5)
        for kind in ('whole_calls', 'call_details'):
            response = self.client.get(f"/api/jobs/{job_id}/results", params={"kind": kind, "related_type": "Add"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['items'], [])
        with patch("profiling_tools.parser._load_trace_events", side_effect=AssertionError("must not reparse")), patch("profiling_web.index.csv_rows", side_effect=AssertionError("must use index")):
            records = self.client.get(f"/api/jobs/{job_id}/results", params={"kind": "mappings", "op_type": "Add"}).json()
            self.assertEqual(records["total"], 1)
        export = self.client.get(f"/api/jobs/{job_id}/artifacts/npu_gpu_op_statistic.xlsx")
        self.assertEqual(export.status_code, 200)
        self.assertEqual(export.content[:2], b"PK")
        self.assertIn("attachment", export.headers["content-disposition"])
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/artifacts/run_manifest.json").status_code, 404)
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/artifacts/task.json").status_code, 404)

    def test_auth_cookie_expiry_origin_and_host(self):
        self.client.headers.pop("Authorization")
        self.assertEqual(self.client.get("/api/jobs").status_code, 401)
        self.assertEqual(self.client.post("/api/session", json={"token": self.settings.token}).status_code, 403)
        login = self.client.post("/api/session", json={"token": self.settings.token}, headers={"Origin": "http://testserver"})
        self.assertEqual(login.status_code, 200)
        self.assertIn("httponly", login.headers["set-cookie"].lower())
        self.assertEqual(self.client.get("/api/jobs").status_code, 200)
        self.assertEqual(self.client.post("/api/jobs", json={}, headers={"Origin": "https://attacker.example"}).status_code, 403)
        self.assertEqual(self.client.post("/api/jobs", json={}, headers={"Origin": "http://testserver"}).status_code, 201)
        self.assertEqual(self.client.get("/api/jobs", headers={"Host": "attacker.example"}).status_code, 400)
        self.assertFalse(valid_cookie(self.settings.token, session_cookie(self.settings.token, 1)))
        self.assertFalse(valid_cookie(self.settings.token, "tampered"))

    def test_inputs_are_role_restricted_and_frozen(self):
        job_id = self.new()
        self.assertEqual(self.client.put(f"/api/jobs/{job_id}/inputs/evil", content=b"a").status_code, 404)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/inspect").status_code, 409)
        self.upload(job_id)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/inspect").status_code, 202)
        self.assertEqual(self.client.put(f"/api/jobs/{job_id}/inputs/gpu_trace", content=b"{}").status_code, 409)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/inspect").status_code, 409)
        self.assertTrue((self.store.path(job_id) / "inputs" / "gpu.json").exists())
        self.assertFalse((self.settings.data_dir / "gpu.json").exists())

    def test_upload_limit_empty_and_storage_quota(self):
        job_id = self.new()
        self.settings.max_upload_bytes = 8
        response = self.client.put(f"/api/jobs/{job_id}/inputs/gpu_trace", content=b"x" * 9)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.client.put(f"/api/jobs/{job_id}/inputs/gpu_trace", content=b"").status_code, 422)
        self.settings.storage_bytes = 1
        self.assertEqual(self.client.put(f"/api/jobs/{job_id}/inputs/gpu_trace", content=b"{}").status_code, 507)
        self.assertEqual(self.store.get(job_id)["uploading"], 0)
        self.assertEqual(list(self.store.path(job_id).rglob("*.upload")), [])

    def test_chunked_upload_and_json_limits(self):
        job_id = self.new()
        self.settings.max_upload_bytes = 5
        response = self.client.put(f"/api/jobs/{job_id}/inputs/gpu_trace", content=iter([b"123", b"456"]))
        self.assertEqual(response.status_code, 413)
        response = self.client.post("/api/jobs", content=iter([b"x" * 9000, b"x" * 9000]), headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 413)

    def test_explicit_scope_and_no_path_options(self):
        job_id = self.new()
        self.upload(job_id)
        self.inspect(job_id)
        for data in ({}, {"all_steps": True, "gpu_step": "1"}, {"all_steps": True, "output_dir": "C:/"}, {"all_steps": True, "diagnostic_coverage_threshold": 101}):
            self.assertEqual(self.client.post(f"/api/jobs/{job_id}/run", json=data).status_code, 422)
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/run", json={"all_steps": True}).status_code, 202)

    def test_cancel_queue_and_running_attempt(self):
        job_id = self.new()
        self.upload(job_id)
        self.client.post(f"/api/jobs/{job_id}/inspect")
        self.assertEqual(self.client.post(f"/api/jobs/{job_id}/cancel").json()["status"], "cancelled")
        self.assertIsNone(self.store.claim())
        self.client.post(f"/api/jobs/{job_id}/inspect")
        claimed = self.store.claim()
        self.client.post(f"/api/jobs/{job_id}/cancel")
        Worker(self.settings).execute(claimed)
        self.assertEqual(self.store.get(job_id)["status"], "cancelled")
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}/results").status_code, 409)

    def test_memory_and_timeout_fail_without_publishing(self):
        job_id = self.new()
        self.upload(job_id)
        for field, value, text in (("memory_mb", 1, "内存"), ("timeout_seconds", .001, "时间")):
            original = getattr(self.settings, field)
            setattr(self.settings, field, value)
            self.client.post(f"/api/jobs/{job_id}/inspect")
            Worker(self.settings).execute(self.store.claim())
            state = self.store.get(job_id)
            self.assertEqual(state["status"], "failed")
            self.assertIn(text, state["error"])
            self.assertIsNone(state["summary"])
            setattr(self.settings, field, original)

    def test_worker_recovery_and_single_worker_lock(self):
        job_id = self.new()
        self.upload(job_id)
        self.client.post(f"/api/jobs/{job_id}/inspect")
        self.store.claim()
        Worker(self.settings).recover()
        self.assertEqual(self.store.get(job_id)["status"], "interrupted")
        with worker_lock(self.settings.data_dir):
            with self.assertRaises(OSError):
                with worker_lock(self.settings.data_dir):
                    pass

    def test_bad_trace_returns_validation_error_not_success(self):
        job_id = self.new()
        self.upload(job_id)
        self.client.put(f"/api/jobs/{job_id}/inputs/gpu_trace", content=b"not json")
        self.client.post(f"/api/jobs/{job_id}/inspect")
        Worker(self.settings).execute(self.store.claim())
        state = self.store.get(job_id)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["inspection"]["status"], "invalid")
        self.assertIsNone(state["summary"])

    def test_query_bounds_and_injection(self):
        job_id = self.complete()
        for params in ({"kind": "sqlite_master"}, {"sort": "DROP TABLE stats"}, {"limit": 500}, {"offset": -1}):
            self.assertEqual(self.client.get(f"/api/jobs/{job_id}/results", params=params).status_code, 422)
        response = self.client.get(f"/api/jobs/{job_id}/results", params={"op_type": "' OR 1=1 --"})
        self.assertEqual(response.json()["total"], 0)
        response = self.client.get(f"/api/jobs/{job_id}/results", params={"search": "Add", "offset": 1, "limit": 1})
        self.assertEqual(response.json()["items"], [])
        self.assertEqual(response.json()["total"], 1)
        self.assertEqual(self.client.get("/api/jobs/not-a-uuid").status_code, 404)

    def test_confirmed_delete_only_terminal_tasks(self):
        job_id = self.new()
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}").status_code, 422)
        self.upload(job_id)
        self.client.post(f"/api/jobs/{job_id}/inspect")
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}?confirm=true").status_code, 409)
        self.client.post(f"/api/jobs/{job_id}/cancel")
        self.assertEqual(self.client.delete(f"/api/jobs/{job_id}?confirm=true").status_code, 200)
        self.assertFalse(self.store.path(job_id).exists())
        self.assertEqual(self.client.get(f"/api/jobs/{job_id}").status_code, 404)

    def test_csv_export_formula_neutralization(self):
        path = self.root / "test.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows([["name", "time"], ['=HYPERLINK("bad")', "-1.25"], ["@SUM(1)", "+2"]])
        text = "".join(csv_download(path)).lstrip("\ufeff")
        rows = list(csv.reader(io.StringIO(text)))
        self.assertTrue(rows[1][0].startswith("'="))
        self.assertEqual(rows[1][1], "-1.25")
        self.assertTrue(rows[2][0].startswith("'@"))
        self.assertEqual(rows[2][1], "+2")


if __name__ == "__main__":
    unittest.main()

"""Opt-in HTTP workflow check against a locally deployed service (needs httpx)."""
import argparse
import os
from pathlib import Path
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    for key in ("gpu-trace", "npu-trace", "npu-kernel-details", "gpu-step", "npu-step"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--name", default="HTTP end-to-end verification")
    args = parser.parse_args()
    token = os.environ["PROFILING_WEB_TOKEN"]
    with httpx.Client(base_url=args.url, headers={"Authorization": "Bearer " + token}, timeout=600) as client:
        def request(method, path, **kwargs):
            response = client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()
        job = request("POST", "/api/jobs", json={"name": args.name})
        path = "/api/jobs/" + job["id"]
        print("Created task:", job["id"], flush=True)
        for role in ("gpu_trace", "npu_trace", "npu_kernel_details"):
            file = Path(getattr(args, role))
            with file.open("rb") as stream:
                request("PUT", path + "/inputs/" + role, content=stream,
                        headers={"Content-Type": "application/octet-stream", "Content-Length": str(file.stat().st_size)})
            print("Uploaded:", role, file.stat().st_size, flush=True)
        def wait():
            last, deadline = None, time.monotonic() + 1800
            while time.monotonic() < deadline:
                state = request("GET", path)
                if state["stage"] != last:
                    print(state["stage"], flush=True)
                    last = state["stage"]
                if state["status"] not in {"queued", "running"}:
                    return state
                time.sleep(1)
            raise TimeoutError("Task did not finish")
        request("POST", path + "/inspect")
        state = wait()
        if state["status"] != "needs_selection":
            raise RuntimeError(state["error"])
        options = {"gpu_step": args.gpu_step, "npu_step": args.npu_step}
        for side in ("gpu", "npu"):
            candidates = state["inspection"]["profiles"][side]["cpu_candidates"]
            if len(candidates) != 1:
                raise ValueError("Ambiguous CPU PID: confirm scope in the browser")
            options[side + "_pid"] = candidates[0]["pid"]
        request("POST", path + "/run", json=options)
        state = wait()
        if state["status"] != "succeeded":
            raise RuntimeError(state["error"])
        assert state["summary"]["saved_validation"]["status"] == "valid"
        page = request("GET", path + "/results?limit=1")
        assert page["items"]
        for kind in ("reasons", "mappings", "calls", "gpu_unmapped"):
            request("GET", path + "/results?kind=" + kind + "&limit=1")
        file = client.get(path + "/artifacts/npu_gpu_op_statistic.xlsx")
        file.raise_for_status()
        assert file.content[:2] == b"PK"
        print("PASS:", state["summary"]["overall_statistics"], flush=True)


if __name__ == "__main__":
    main()

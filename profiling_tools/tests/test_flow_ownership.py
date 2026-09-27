from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from profiling_tools.model import KernelDetail
from profiling_tools.parser import attach_kernel_details, parse_trace
from profiling_tools.tests.test_compare import complete


def flow(name, category, flow_id, start, finish, *, pid=1, tid=1):
    return [
        {"name": name, "cat": category, "id": flow_id, "ph": "s",
         "pid": pid, "tid": tid, "ts": start},
        {"name": name, "cat": category, "id": flow_id, "ph": "f",
         "pid": 4, "tid": 7, "ts": finish},
    ]


class FlowOwnershipTest(unittest.TestCase):
    def parse(self, events):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.json"
            trace.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            return parse_trace(str(trace), "npu")

    def test_torch_flow_wins_regardless_of_order_and_targets_exact_kernel_start(self):
        host = flow("HostToDevice", "HostToDevice", 88, 55, 100, pid=9)
        direct = flow("torch_to_npu", "async_npu", 88, 15, 100)
        # IDs intentionally overlap across categories; the two Cat calls also
        # have the same name, so a type-only validation cannot find this bug.
        for flows in (host + direct, direct + host):
            with self.subTest(first_category=flows[0]["cat"]):
                profile = self.parse([
                    complete("aten::cat", "cpu_op", 10, 20, 1, 1),
                    complete("aten::cat", "cpu_op", 50, 20, 1, 1),
                    *flows,
                    complete("previous", "kernel", 95, 5, 4, 7),
                    complete("concat", "kernel", 100, 5, 4, 7),
                ])
                self.assertIsNone(profile.kernels[0].owner_op_index)
                self.assertEqual(profile.kernels[1].owner_op_index, 0)
                self.assertEqual(profile.kernels[1].link_method, "torch_to_npu")
                self.assertEqual(profile.flow_link_diagnostics["flow_pair_count"], 2)

    def test_host_flow_uses_queue_correlation_for_all_launched_kernels(self):
        profile = self.parse([
            complete("aten::cat", "cpu_op", 10, 20, 1, 1),
            complete("aten::reshape", "cpu_op", 50, 30, 1, 1),
            complete("Enqueue@aclnnCat", "enqueue", 15, 1, 1, 1,
                     {"correlation_id": 42}),
            complete("Dequeue@aclnnCat", "dequeue", 50, 20, 1, 3,
                     {"correlation_id": 42}),
            {"name": "enqueue_to_dequeue", "cat": "async_task_queue", "id": 42,
             "ph": "s", "pid": 1, "tid": 1, "ts": 15},
            {"name": "enqueue_to_dequeue", "cat": "async_task_queue", "id": 42,
             "ph": "f", "pid": 1, "tid": 3, "ts": 50},
            *flow("launch1", "HostToDevice", 101, 55, 100, pid=9, tid=3),
            *flow("launch2", "HostToDevice", 102, 60, 105, pid=9, tid=3),
            complete("concat1", "kernel", 100, 5, 4, 7),
            complete("concat2", "kernel", 105, 5, 4, 7),
        ])
        self.assertEqual([k.owner_op_index for k in profile.kernels], [0, 0])
        self.assertEqual({k.link_method for k in profile.kernels}, {"host_to_device_queue"})
        self.assertEqual(profile.flow_link_diagnostics["ignored_queue_flow_event_count"], 2)

    def test_device_flow_without_source_evidence_does_not_jump_cpu_threads(self):
        for category, name in (("HostToDevice", "launch"), ("async_npu", "torch_to_npu")):
            with self.subTest(category=category):
                profile = self.parse([
                    complete("aten::reshape", "cpu_op", 50, 30, 1, 1),
                    *flow(name, category, 1, 55, 100, pid=9, tid=3),
                    complete("concat", "kernel", 100, 5, 4, 7),
                ])
                self.assertIsNone(profile.kernels[0].owner_op_index)

    def test_ambiguous_queue_id_is_not_used_as_strong_evidence(self):
        profile = self.parse([
            complete("aten::cat", "cpu_op", 10, 30, 1, 1),
            complete("Enqueue@aclnnCat", "enqueue", 15, 1, 1, 1,
                     {"correlation_id": 42}),
            complete("Enqueue@aclnnCat", "enqueue", 25, 1, 1, 1,
                     {"correlation_id": 42}),
            complete("Dequeue@aclnnCat", "dequeue", 50, 20, 1, 3,
                     {"correlation_id": 42}),
            *flow("launch", "HostToDevice", 1, 55, 100, pid=9, tid=3),
            complete("concat", "kernel", 100, 5, 4, 7),
        ])
        self.assertIsNone(profile.kernels[0].owner_op_index)

    def test_missing_device_task_does_not_claim_nearby_kernel(self):
        profile = self.parse([
            complete("aten::cat", "cpu_op", 10, 20, 1, 1),
            *flow("torch_to_npu", "async_npu", 1, 15, 80),
            complete("unrelated", "kernel", 100, 5, 4, 7),
        ])
        self.assertIsNone(profile.kernels[0].owner_op_index)
        self.assertEqual(profile.flow_link_diagnostics["flow_kernel_not_found_count"], 1)

    def test_conflicting_direct_flows_remain_unresolved(self):
        profile = self.parse([
            complete("aten::stack", "cpu_op", 10, 20, 1, 1),
            complete("aten::stack", "cpu_op", 50, 20, 1, 1),
            *flow("torch_to_npu", "async_npu", 1, 15, 100),
            *flow("torch_to_npu", "async_npu", 2, 55, 100),
            *flow("torch_to_npu", "async_npu", 3, 55, 200),
            complete("concat1", "kernel", 100, 5, 4, 7,
                     {"Task ID": 1, "Stream ID": 7}),
            complete("concat2", "kernel", 200, 5, 4, 7,
                     {"Task ID": 2, "Stream ID": 7}),
        ])
        attach_kernel_details(profile, [
            KernelDetail(row_index=i, name="aclnnStack_ConcatD_ConcatD",
                         op_type="ConcatD", task_id=str(i), stream_id="7",
                         start_us=start, duration_us=5)
            for i, start in ((1, 100), (2, 200))
        ])
        self.assertIsNone(profile.kernels[0].owner_op_index)
        self.assertEqual(profile.kernels[0].link_method, "ambiguous_flow")
        self.assertEqual(profile.kernels[1].owner_op_index, 1)
        self.assertEqual(profile.flow_link_diagnostics["flow_unresolved_owner_conflict_count"], 1)

    def test_exact_flow_is_not_overwritten_by_equal_count_order_repair(self):
        profile = self.parse([
            complete("aten::stack", "cpu_op", 10, 20, 1, 1),
            complete("aten::stack", "cpu_op", 50, 20, 1, 1),
            *flow("torch_to_npu", "async_npu", 1, 55, 100),
            *flow("torch_to_npu", "async_npu", 2, 15, 200),
            complete("concat1", "kernel", 100, 5, 4, 7,
                     {"Task ID": 1, "Stream ID": 7}),
            complete("concat2", "kernel", 200, 5, 4, 7,
                     {"Task ID": 2, "Stream ID": 7}),
        ])
        attach_kernel_details(profile, [
            KernelDetail(row_index=i, name="aclnnStack_ConcatD_ConcatD",
                         op_type="ConcatD", task_id=str(i), stream_id="7",
                         start_us=start, duration_us=5)
            for i, start in ((1, 100), (2, 200))
        ])
        self.assertEqual([k.owner_op_index for k in profile.kernels], [1, 0])
        self.assertEqual({k.link_method for k in profile.kernels}, {"torch_to_npu"})


if __name__ == "__main__":
    unittest.main()

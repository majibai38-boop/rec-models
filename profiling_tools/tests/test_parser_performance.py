import unittest
from unittest.mock import patch

from profiling_tools import parser
from profiling_tools.model import TraceEvent


class ParserPerformanceTests(unittest.TestCase):
    def test_kernel_outside_cpu_step_is_linked_before_filtering(self):
        events = [
            TraceEvent(0, "ProfilerStep#1", "user_annotation", "X", 0, 50, "1", "1", {}),
            TraceEvent(1, "aten::add", "cpu_op", "X", 10, 10, "1", "1", {"External id": 7}),
            TraceEvent(2, "add", "kernel", "X", 100, 5, "2", "1",
                       {"External id": 7, "Task Type": "AI_CORE"}),
            TraceEvent(3, "unrelated", "kernel", "X", 110, 5, "2", "1", {}),
        ]
        profile = parser.parse_trace("unused.json", "gpu", requested_pid="1",
                                     requested_step="1", trace_data=(events, len(events)))
        self.assertEqual([kernel.event.name for kernel in profile.kernels], ["add"])
        self.assertEqual(profile.kernels[0].owner_op_index, 0)
        self.assertEqual(profile.task_type_counts, {})

    def test_normalized_keys_preserve_alias_and_mutation_semantics(self):
        args = {"Task Type": "first", "task_type": None, "fallback": "other"}
        self.assertIsNone(parser.get_arg(args, "Task Type", "fallback"))
        args["task_type"] = "changed"
        self.assertEqual(parser.get_arg(args, "Task Type"), "changed")
        self.assertEqual(parser.get_arg(args, "fallback", "Task Type"), "other")
        self.assertEqual(parser._normalized_key(["A B"]), "ab")

    def test_field_cache_is_bounded(self):
        parser._normalized_key_text.cache_clear()
        for index in range(5000):
            parser._normalized_key("field-" + str(index))
        self.assertLessEqual(parser._normalized_key_text.cache_info().currsize, 4096)

    def test_kernel_classification_preserves_precedence(self):
        cases = [
            ("hccl_event_wait", "kernel", {}, False, False),
            ("hccl_event_wait", "kernel", {}, True, True),
            ("event_wait", "kernel", {}, True, False),
            ("copy", "gpu_memcpy", {"Device Type": "cuda"}, True, False),
            ("compute", "kernel", {}, False, True),
            ("compute", "", {"Device Type": "npu"}, False, True),
            ("compute", "", {"Task Type": "AI_CORE"}, False, True),
            ("compute", "cpu_op", {}, False, False),
        ]
        for name, category, args, communication, expected in cases:
            with self.subTest(name=name, args=args, communication=communication):
                event = TraceEvent(0, name, category, "X", 0, 1, "1", "1", args)
                with patch.object(parser, "_task_type", wraps=parser._task_type) as lookup:
                    self.assertEqual(parser._is_kernel(event, communication, []), expected)
                    self.assertEqual(lookup.call_count, 1)
        event = TraceEvent(0, "compute", "", "X", 0, 1, "device", "1", {})
        self.assertTrue(parser._is_kernel(event, False, ["device"]))
        event.duration_us = 0
        self.assertFalse(parser._is_kernel(event, False, ["device"]))

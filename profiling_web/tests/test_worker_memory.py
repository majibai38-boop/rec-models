from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import psutil

from profiling_web.worker import process_tree_rss


class ProcessTreeMemoryTests(unittest.TestCase):
    def process(self, rss):
        process = Mock()
        process.memory_info.return_value = SimpleNamespace(rss=rss)
        return process

    def test_counts_launcher_interpreter_and_grandchildren(self):
        parent = self.process(4 * 1024**2)
        parent.children.return_value = [self.process(80 * 1024**2), self.process(16 * 1024**2)]
        self.assertEqual(process_tree_rss(parent), 100 * 1024**2)
        parent.children.assert_called_once_with(recursive=True)

    def test_exited_child_does_not_hide_live_children(self):
        parent = self.process(4)
        exited = self.process(0)
        exited.memory_info.side_effect = psutil.NoSuchProcess(123)
        parent.children.return_value = [exited, self.process(80)]
        self.assertEqual(process_tree_rss(parent), 84)

    def test_exited_parent_and_permission_failure(self):
        parent = self.process(4)
        parent.children.side_effect = psutil.NoSuchProcess(123)
        self.assertEqual(process_tree_rss(parent), 0)
        parent.children.side_effect = psutil.AccessDenied(123)
        with self.assertRaises(psutil.AccessDenied):
            process_tree_rss(parent)


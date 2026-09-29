from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling_tools.report import write_reports
from profiling_tools.tests import test_whole_calls
from profiling_web.index import build_index, call_evidence, query_index


class WholeCallIndexTest(unittest.TestCase):
    def test_type_group_and_detail_filters_use_only_index(self):
        gpu, npu, matches, _ = test_whole_calls.WholeCallsTest().case()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_reports(str(root / 'reports'), gpu, npu, matches, [], [], {})
            index = root / 'index.sqlite'
            build_index(root / 'reports', index)
            with patch('profiling_web.index.csv_rows', side_effect=AssertionError('GET must use SQLite')):
                groups = query_index(index, kind='whole_calls', related_type='Slice', device_id='', core_type='AI_CORE')
                self.assertEqual(groups['total'], 1)
                group = groups['items'][0]
                rows = query_index(index, kind='call_details', group_id=group['Group ID'])
                self.assertEqual(rows['total'], 1)
                self.assertEqual(float(rows['items'][0]['GPU Total Time(us)']), 2.0)
                self.assertEqual(query_index(index, kind='call_details', call_id=rows['items'][0]['Call ID'])['total'], 1)
                self.assertEqual(query_index(index, kind='call_details', comparison_status='comparable')['total'], 1)
                self.assertEqual(query_index(index, kind='whole_calls', comparison_status='pending')['total'], 0)
                self.assertEqual(query_index(index, kind='whole_calls', search='absent')['total'], 0)
                self.assertEqual(query_index(index, kind='whole_calls', sort='ratio')['total'], 1)
                evidence = call_evidence(index, rows['items'][0]['Call ID'], limit=1)
                self.assertEqual(len(evidence['event_rows']), 1)
                self.assertGreater(evidence['event_total'], 1)
                self.assertTrue(evidence['has_more'])
                self.assertEqual(query_index(index, kind='whole_calls', related_type='Slice', device_id='wrong')['total'], 0)
                self.assertEqual(query_index(index, kind='whole_calls', related_type="' OR 1=1 --")['total'], 0)
                with self.assertRaises(ValueError):
                    query_index(index, kind='operators', group_id=group['Group ID'])

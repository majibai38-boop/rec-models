import copy
import unittest

from profiling_tools.tests.test_manual_batch import multiple_template_catalog
from profiling_web.manual_positions import with_call_positions, comparison_positions


class ManualPositionsTest(unittest.TestCase):
    def test_family_and_template_positions_follow_call_order_not_input_order(self):
        c = multiple_template_catalog()
        before = copy.deepcopy(c)
        gpu = {r['id']: r for r in with_call_positions(list(reversed(c['candidates'])), gpu=True)}
        npu = comparison_positions(list(reversed(c['targets'])), c['candidates'][100])
        for row in (gpu['op:100'], npu['npu:100']):
            self.assertEqual(row['call_position'], {'index': 101, 'total': 260})
            self.assertEqual(row['template_position'], {'index': 4, 'total': 10})
        self.assertEqual(npu['npu:100']['cpu_name'], 'aten::index_add_')
        self.assertEqual(c, before)

    def test_independent_families_and_unknown_context_are_not_mixed(self):
        c = multiple_template_catalog()
        for field, value in (('module', 'other'), ('device_key', ['other', '0'])):
            rows = copy.deepcopy(c['candidates'][:3])
            rows[1][field] = value
            result = with_call_positions(rows, gpu=True)
            self.assertEqual(result[1]['call_position'], {'index': 1, 'total': 1})
            self.assertEqual(result[2]['call_position'], {'index': 2, 'total': 2})
        rows = copy.deepcopy(c['candidates'][:3])
        rows[0].pop('batch_context')
        result = with_call_positions(rows, gpu=True)
        self.assertIsNone(result[0]['call_position'])
        self.assertIsNone(result[0]['template_position'])
        self.assertEqual(result[1]['call_position'], {'index': 1, 'total': 2})

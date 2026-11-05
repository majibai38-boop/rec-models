import json
from pathlib import Path
import tempfile
import unittest

from profiling_tools.compare import compare_profiles
from profiling_tools.tests.test_skill_interfaces import fixture
from profiling_web.index import public_summary, downloadable


class ParameterReportTests(unittest.TestCase):
    def test_report_api_and_download_preserve_independent_estimates(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            for side in ('gpu', 'npu'):
                path = Path(tmp)/f'{side}.json'
                data = json.loads(path.read_text(encoding='utf-8'))
                event = data['traceEvents'][1]
                event['name'] = 'aten::linear' if side == 'gpu' else 'aten::embedding'
                event['args'].update({'Input Dims': [[32, 8], [4, 8], [4]] if side == 'gpu' else [[1000, 8], [32]],
                                      'Input type': ['float', 'float', 'float'] if side == 'gpu' else ['float', 'long']})
                path.write_text(json.dumps(data), encoding='utf-8')
            result = compare_profiles(config)
            self.assertEqual(result['validation']['status'], 'valid')
            report = public_summary(config.output_dir)
            estimates = report['parameters']
            self.assertEqual(estimates['gpu']['candidate_parameters'], 36)
            self.assertEqual(estimates['npu']['embedding_candidates'], 8000)
            self.assertIsNone(estimates['npu']['dense_candidates'])
            self.assertIn('parameter_estimates.json', report['reports'])
            self.assertTrue(downloadable('parameter_estimates.json'))
            saved = json.loads((Path(config.output_dir)/'parameter_estimates.json').read_text(encoding='utf-8'))
            self.assertEqual(saved, estimates)

    def test_missing_parameters_and_legacy_report_remain_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = fixture(tmp)
            compare_profiles(config)
            report = public_summary(config.output_dir)
            self.assertIsNone(report['parameters']['gpu']['candidate_parameters'])
            path = Path(config.output_dir)/'summary.json'
            raw = json.loads(path.read_text(encoding='utf-8'))
            del raw['parameters']
            path.write_text(json.dumps(raw), encoding='utf-8')
            self.assertIsNone(public_summary(config.output_dir)['parameters'])

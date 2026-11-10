from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from profiling_tools.compare import compare_profiles
from profiling_tools.tests.test_manual_mapping import nested_fixture


class AnalysisCacheTests(unittest.TestCase):
    def test_manual_and_undo_reuse_parse_and_exports_with_identical_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = nested_fixture(root, count=6, templates=2, steps=2)
            cache = root/'cache'
            initial = compare_profiles(config, manual_enabled=True, cache_dir=cache)
            base = json.loads((root/'out/run_manifest.json').read_text(encoding='utf-8'))
            catalog = json.loads((root/'out/manual_catalog.json').read_text(encoding='utf-8'))
            links = [{'id': 'test', 'npu_indices': catalog['targets'][0]['indices'],
                      'gpu_indices': catalog['candidates'][0]['indices'], 'reason': 'verified'}]
            full = compare_profiles(replace(config, output_dir=str(root/'full')),
                                    manual_enabled=True, manual_links=links, manual_base=base)
            with (patch('profiling_tools.compare.inspect_inputs', side_effect=AssertionError('no inspection')),
                 patch('profiling_tools.compare.parse_trace', side_effect=AssertionError('no parsing')),
                 patch('profiling_tools.compare.match_ops', side_effect=AssertionError('no matching')),
                 patch('profiling_tools.download_reports.write_selected_trace', side_effect=AssertionError('no JSON export')),
                 patch('profiling_tools.download_reports.write_download_csvs', side_effect=AssertionError('no CSV export'))):
                fast = compare_profiles(replace(config, output_dir=str(root/'fast')),
                    manual_enabled=True, manual_links=links, manual_base=base, cache_dir=cache)
                undo = compare_profiles(replace(config, output_dir=str(root/'undo')),
                    manual_enabled=True, manual_base=base, cache_dir=cache)
            self.assertTrue(fast['analysis_cache']['hit'])
            self.assertEqual(fast['validation']['status'], 'valid')
            self.assertEqual(fast['comparison'], full['comparison'])
            self.assertEqual(fast['gpu_accounting'], full['gpu_accounting'])
            self.assertEqual(undo['comparison'], initial['comparison'])
            for path in (root/'full').iterdir():
                if path.suffix == '.csv' or path.name in ('operator_details.json', 'manual_catalog.json', 'parameter_estimates.json', 'gpu_trace.json', 'npu_trace.json'):
                    self.assertEqual(path.read_bytes(), (root/'fast'/path.name).read_bytes(), path.name)

    def test_corrupt_cache_and_exports_fall_back_and_scope_changes_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = nested_fixture(root)
            cache = root/'cache'
            compare_profiles(config, cache_dir=cache)
            (cache/'state.json').write_text('{}', encoding='utf-8')
            second = compare_profiles(replace(config, output_dir=str(root/'second')), cache_dir=cache)
            self.assertFalse(second['analysis_cache']['hit'])
            self.assertEqual(second['analysis_cache']['reason'], 'cache_checksum_mismatch')
            (root/'second/gpu_trace.json').write_text('{}', encoding='utf-8')
            third = compare_profiles(replace(config, output_dir=str(root/'third')), cache_dir=cache)
            self.assertEqual(third['analysis_cache']['reason'], 'export_checksum_mismatch')
            changed = compare_profiles(replace(config, output_dir=str(root/'changed'), diagnostic_coverage_threshold=100), cache_dir=cache)
            self.assertFalse(changed['analysis_cache']['hit'])
            self.assertEqual(changed['validation']['status'], 'valid')

    def test_changed_input_is_still_rejected_on_manual_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = nested_fixture(root)
            compare_profiles(config, cache_dir=root/'cache')
            base = json.loads((root/'out/run_manifest.json').read_text(encoding='utf-8'))
            with Path(config.gpu_trace).open('a', encoding='utf-8') as stream:
                stream.write(' ')
            with self.assertRaisesRegex(ValueError, '输入或引擎'):
                compare_profiles(replace(config, output_dir=str(root/'changed')),
                                 manual_base=base, cache_dir=root/'cache')

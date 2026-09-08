import tempfile
import unittest
from pathlib import Path

import torch

from behavior_and_multi_task.training_framework.utils.reporting import (
    PROJECT_ROOT,
    get_device_result_name,
    resolve_results_dir,
    save_benchmark_results,
    summarize_latencies,
)


class ReportingTests(unittest.TestCase):
    def test_default_results_directory_is_repository_sibling(self):
        self.assertEqual(get_device_result_name("cuda:0"), "GPU")
        self.assertEqual(
            resolve_results_dir("GPU"),
            PROJECT_ROOT.parent / "save_results_GPU",
        )

    def test_latency_summary_uses_milliseconds_and_lower_percentiles(self):
        result = summarize_latencies([0.001, 0.002, 0.003, 0.004], 128, "ETA")

        self.assertEqual(result["Batch_size"], 128)
        self.assertEqual(result["model_name"], "ETA")
        self.assertAlmostEqual(result["AVG Latency"], 2.5)
        self.assertAlmostEqual(result["P90 Latency"], 3.0)
        self.assertAlmostEqual(result["P999 Latency"], 3.0)

    def test_benchmark_results_are_appended_and_saved(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = save_benchmark_results(
                latency_seconds=[0.001, 0.002],
                batch_size=64,
                model_name="eta",
                device_type="cpu",
                inductor_enabled=False,
                precision_results={"final_loss": 0.25, "step": 12},
                results_dir=temporary_directory,
            )

            performance_text = result["performance_path"].read_text(
                encoding="utf-8"
            )
            precision_text = result["precision_path"].read_text(
                encoding="utf-8"
            )
            self.assertIn("performance: CPU_ETA_eager_", performance_text)
            self.assertIn("CPU_ETA_eager_", precision_text)
            self.assertEqual(result["precision"]["ms/step"], 1.5)
            self.assertEqual(
                result["artifact_path"].parent,
                Path(temporary_directory) / "ETA",
            )
            saved_result = torch.load(
                result["artifact_path"], weights_only=True
            )
            self.assertEqual(saved_result["final_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()

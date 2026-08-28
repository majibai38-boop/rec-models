"""Tests for the shared, sharded Ali-CCP data loader."""

from pathlib import Path
import json
import sys
import unittest


try:
    import torch
except ImportError:
    torch = None

TRAINING_TORCH_ROOT = Path(__file__).resolve().parents[2]
if str(TRAINING_TORCH_ROOT) not in sys.path:
    sys.path.append(str(TRAINING_TORCH_ROOT))

if torch is not None:
    from behavior_and_multi_task.data_process.aliccp import get_spec, load_data
    from behavior_and_multi_task.training_framework.config import AttrDict


def make_payload(offset):
    return {
        "format_version": 1,
        "num_samples": 3,
        "features": {
            "one": torch.tensor([offset, offset + 1, offset + 2], dtype=torch.int32),
            "seq": torch.tensor(
                [[offset, -1], [offset + 1, -1], [offset + 2, -1]],
                dtype=torch.int32,
            ),
        },
        "labels": {
            "y": torch.tensor([0.0, 1.0, 0.0]),
            "z": torch.tensor([0.0, 0.0, 0.0]),
        },
    }


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AliccpDataTest(unittest.TestCase):
    def test_real_modes_require_the_generated_dataset_spec(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary_dir:
            params = AttrDict(
                data_dir=temporary_dir,
                mode="train",
                extra_fields=0,
            )
            with self.assertRaisesRegex(FileNotFoundError, "spec.json"):
                get_spec(params)

    def test_loader_combines_rows_across_shards_into_full_batches(self):
        import tempfile

        spec = {
            "data_format": "pytorch",
            "format_version": 1,
            "one_hot_fields": ["one"],
            "multi_hot_fields": ["seq"],
            "special_fields": [],
            "vocab_length": {"one": 20, "seq": 20},
            "parts": {"train": 2, "val": 2, "test": 2},
            "shard_sizes": {
                "train": [3, 3],
                "val": [3, 3],
                "test": [3, 3],
            },
            "dataset_size": {"train": 6, "val": 6, "test": 6},
        }

        with tempfile.TemporaryDirectory() as temporary_dir:
            data_dir = Path(temporary_dir)
            (data_dir / "spec.json").write_text(
                json.dumps(spec), encoding="utf-8"
            )
            for mode in ("train", "val", "test"):
                split_dir = data_dir / mode
                split_dir.mkdir()
                torch.save(make_payload(1), split_dir / f"data_{mode}.pt.00000")
                torch.save(make_payload(4), split_dir / f"data_{mode}.pt.00001")

            params = AttrDict(
                data_dir=str(data_dir),
                batch_size=4,
                device="cpu",
                seed=7,
                mode="train",
                extra_fields=0,
            )
            train_loader, test_loader, val_loader = load_data(params)

            self.assertEqual(len(train_loader), 1)
            self.assertEqual(len(test_loader), 1)
            self.assertEqual(len(val_loader), 1)
            features, labels = next(iter(val_loader))
            self.assertEqual(features["one"].dtype, torch.long)
            self.assertEqual(features["one"].tolist(), [1, 2, 3, 4])
            self.assertEqual(features["seq"].shape, (4, 2))
            self.assertEqual(labels["y"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()

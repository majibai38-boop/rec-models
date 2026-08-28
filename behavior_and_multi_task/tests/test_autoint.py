"""CPU tests for the Ali-CCP AutoInt implementation."""

import argparse
from pathlib import Path
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
    from behavior_and_multi_task.models.autoint import (
        AUTOINT_DEFAULTS,
        AliccpAutoInt,
        AutoInt,
        InteractingLayer,
        configure_autoint,
    )
    from behavior_and_multi_task.data_process.aliccp import get_spec
    from behavior_and_multi_task.training_framework.config import AttrDict


def tiny_spec():
    one_hot_fields = ["101", "121", "122"]
    multi_hot_fields = ["110_14", "127_14"]
    special_fields = ["853"]
    fields = one_hot_fields + multi_hot_fields + special_fields
    return {
        "one_hot_fields": one_hot_fields,
        "multi_hot_fields": multi_hot_fields,
        "special_fields": special_fields,
        "vocab_length": {field: 5 for field in fields},
    }


def tiny_batch(spec, batch_size=4):
    features = {
        field: torch.randint(0, 6, (batch_size,))
        for field in spec["one_hot_fields"]
    }
    for field in spec["multi_hot_fields"]:
        values = torch.randint(0, 6, (batch_size, 4))
        values[:, -1] = -1
        features[field] = values
    for field in spec["special_fields"]:
        values = torch.randint(0, 6, (batch_size, 3))
        values[:, -1] = -1
        features[field] = values
    labels = {"y": torch.tensor([0.0, 1.0, 1.0, 0.0])}
    return features, labels


def tiny_params(**overrides):
    values = {
        "embedding_size": 4,
        "autoint_attention_layers": 2,
        "autoint_num_heads": 2,
        "autoint_residual": True,
        "autoint_scaling": False,
        "autoint_dnn_hidden_units": (8, 4),
        "autoint_dropout": 0.0,
        "autoint_positive_class_weight": 1.0,
    }
    values.update(overrides)
    return AttrDict(values)


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AutoIntTest(unittest.TestCase):
    def test_interacting_layer_attends_within_each_sample(self):
        torch.manual_seed(7)
        layer = InteractingLayer(
            embedding_size=4,
            head_num=2,
            use_residual=True,
        )
        first = torch.randn(1, 3, 4)
        second = torch.randn(1, 3, 4) * 20.0

        isolated = layer(first)
        batched = layer(torch.cat((first, second), dim=0))[:1]
        self.assertTrue(
            torch.allclose(isolated, batched, atol=1e-7, rtol=1e-6)
        )

    def test_core_supports_deepctr_attention_and_dnn_branches(self):
        inputs = torch.randn(3, 6, 4)
        linear_logit = torch.randn(3, 1)
        configurations = (
            {"attention_layers": 2, "dnn_hidden_units": (8, 4)},
            {"attention_layers": 2, "dnn_hidden_units": ()},
            {"attention_layers": 0, "dnn_hidden_units": (8, 4)},
        )
        for configuration in configurations:
            with self.subTest(configuration=configuration):
                model = AutoInt(
                    field_count=6,
                    embedding_size=4,
                    num_heads=2,
                    **configuration,
                )
                output = model(inputs, linear_logit)
                self.assertEqual(output.shape, (3,))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.all((0 <= output) & (output <= 1)))

    def test_aliccp_forward_loss_and_backward(self):
        spec = tiny_spec()
        model = AliccpAutoInt(tiny_params(), spec)
        features, labels = tiny_batch(spec)

        predictions = model(features, "train")
        self.assertEqual(set(predictions), {"ctr"})
        self.assertEqual(predictions["ctr"].shape, (4,))
        self.assertTrue(torch.isfinite(predictions["ctr"]).all())

        loss = model.loss(predictions, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_sequence_pooling_keeps_oov_zero_and_masks_minus_one(self):
        spec = tiny_spec()
        model = AliccpAutoInt(tiny_params(), spec)
        field = spec["multi_hot_fields"][0]
        values = torch.tensor(
            [[0, -1, -1], [1, 0, -1], [-1, -1, -1]]
        )

        encoded = model.encoder._encode_sequence(
            field, values, model.encoder.deep_embeddings
        )
        embedding = model.encoder.deep_embeddings[field].weight
        self.assertTrue(torch.allclose(encoded[0], embedding[0]))
        self.assertTrue(
            torch.allclose(encoded[1], (embedding[1] + embedding[0]) / 2)
        )
        self.assertTrue(torch.equal(encoded[2], torch.zeros_like(encoded[2])))

    def test_first_order_linear_branch_is_part_of_prediction(self):
        model = AutoInt(
            field_count=2,
            embedding_size=4,
            attention_layers=1,
            num_heads=2,
            dnn_hidden_units=(),
        )
        with torch.no_grad():
            model.output_layer.weight.zero_()
            model.prediction_bias.zero_()
        field_embeddings = torch.randn(2, 2, 4)
        linear_logit = torch.tensor([[0.0], [2.0]])
        output = model(field_embeddings, linear_logit)
        self.assertTrue(
            torch.allclose(output, torch.sigmoid(linear_logit[:, 0]))
        )

    def test_defaults_keep_reference_benchmark_attention_shape(self):
        self.assertEqual(AUTOINT_DEFAULTS["autoint_attention_layers"], 8)
        self.assertEqual(AUTOINT_DEFAULTS["autoint_num_heads"], 8)
        self.assertEqual(AUTOINT_DEFAULTS["autoint_dnn_hidden_units"], ())

        spec = get_spec(
            AttrDict(
                mode="test_qps",
                data_dir="",
                extra_fields=AUTOINT_DEFAULTS["extra_fields"],
            )
        )
        field_count = sum(
            len(spec[group])
            for group in (
                "one_hot_fields",
                "multi_hot_fields",
                "special_fields",
            )
        )
        self.assertEqual(field_count, 300)

    def test_configuration_rejects_invalid_architectures(self):
        params = tiny_params(embedding_size=6, autoint_num_heads=4)
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "divisible"
        ):
            configure_autoint(params)

        params = tiny_params(
            autoint_attention_layers=0,
            autoint_dnn_hidden_units=(),
        )
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "branch must be enabled"
        ):
            configure_autoint(params)

        params = tiny_params(autoint_dropout=1.0)
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "autoint_dropout"
        ):
            configure_autoint(params)


if __name__ == "__main__":
    unittest.main()

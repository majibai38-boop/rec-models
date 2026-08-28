"""Small CPU contract tests for the unified model adapters."""

import argparse
import inspect
import pickle
from pathlib import Path
from collections import OrderedDict
import sys
import tempfile
import unittest


try:
    import torch
except ImportError:
    torch = None

TRAINING_TORCH_ROOT = Path(__file__).resolve().parents[2]
if str(TRAINING_TORCH_ROOT) not in sys.path:
    sys.path.append(str(TRAINING_TORCH_ROOT))

if torch is not None:
    from behavior_and_multi_task.models.dien import (
        AliccpDIEN,
        DIEN,
        DIEN_DEFAULTS,
        LocalActivationUnit,
        MaskedGRU,
        configure_dien,
    )
    from behavior_and_multi_task.models.essm import (
        AliccpESSM,
        ESSM,
        configure_essm,
    )
    from behavior_and_multi_task.models.eta import (
        ETA,
        LoneAttention as FixedLoneAttention,
    )
    from behavior_and_multi_task.models.eta_legacy import (
        ETA as LegacyETA,
        LoneAttention as LegacyLoneAttention,
    )
    from behavior_and_multi_task.models.registry import list_models
    from behavior_and_multi_task.training_framework.config import AttrDict
    from behavior_and_multi_task.training_framework.handler import (
        ModelHandler,
        get_opts,
        get_params,
    )
    from behavior_and_multi_task.training_framework.utils.serialization import (
        safe_weights_load,
    )
    from behavior_and_multi_task.main import select_model


def tiny_spec():
    one_hot_fields = [
        "101",
        "121",
        "122",
        "124",
        "125",
        "126",
        "127",
        "128",
        "129",
        "205",
        "206",
        "207",
        "216",
        "508",
        "509",
        "702",
        "301",
    ]
    multi_hot_fields = ["110_14", "127_14", "109_14", "150_14"]
    special_fields = ["853", "210"]
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
        values = torch.randint(0, 6, (batch_size, 5))
        values[:, -1] = -1
        features[field] = values
    for field in spec["special_fields"]:
        values = torch.randint(0, 6, (batch_size, 4))
        values[:, -2:] = -1
        features[field] = values

    labels = {
        "y": torch.tensor([0.0, 1.0, 1.0, 1.0]),
        "z": torch.tensor([0.0, 0.0, 1.0, 0.0]),
    }
    return features, labels


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ModelContractTest(unittest.TestCase):
    def test_checkpoint_flags_default_to_disabled_and_support_aliases(self):
        params = get_opts(["main.py", "--mode=test_qps"], get_params())
        self.assertFalse(params.load_checkpoint)
        self.assertFalse(params.save_checkpoint)

        params = get_opts(
            [
                "main.py",
                "--mode=test_qps",
                "--model_dir=checkpoint",
                "--load_weights=true",
                "--save_weights=true",
            ],
            get_params(),
        )
        self.assertTrue(params.load_checkpoint)
        self.assertTrue(params.save_checkpoint)

    def test_disabled_checkpoint_io_does_not_create_or_load(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            handler = ModelHandler.__new__(ModelHandler)
            handler.params = AttrDict(
                model_dir=temporary_dir,
                model="eta",
                load_checkpoint=False,
                save_checkpoint=False,
                device="cpu",
            )
            handler.saved_dir = handler.get_saved_path()
            handler.model = torch.nn.Linear(2, 1)

            self.assertFalse(Path(handler.saved_dir).exists())
            handler.load_check_point()
            self.assertFalse(Path(handler.saved_dir).exists())

            handler.params.save_checkpoint = True
            self.assertEqual(handler.get_saved_path(), handler.saved_dir)
            self.assertTrue(Path(handler.saved_dir).is_dir())

    def test_enabled_checkpoint_load_restores_state_dict(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_dir = Path(temporary_dir) / "eta"
            checkpoint_dir.mkdir()
            expected_model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                expected_model.weight.fill_(3.0)
                expected_model.bias.fill_(2.0)
            torch.save(
                expected_model.state_dict(),
                checkpoint_dir / "best_val.pth",
            )

            handler = ModelHandler.__new__(ModelHandler)
            handler.params = AttrDict(
                model_dir=temporary_dir,
                model="eta",
                load_checkpoint=True,
                save_checkpoint=False,
                device="cpu",
            )
            handler.saved_dir = handler.get_saved_path()
            handler.model = torch.nn.Linear(2, 1)
            handler.load_check_point()

            self.assertTrue(
                torch.equal(handler.model.weight, expected_model.weight)
            )
            self.assertTrue(
                torch.equal(handler.model.bias, expected_model.bias)
            )

    def test_safe_weights_load_allows_getattr_in_restricted_context(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            checkpoint_path = Path(temporary_dir) / "getattr.pth"
            torch.save(getattr, checkpoint_path)
            get_safe_globals = torch.serialization.get_safe_globals
            initial_safe_globals = get_safe_globals()
            torch.serialization.clear_safe_globals()
            try:
                with self.assertRaises(pickle.UnpicklingError):
                    torch.load(
                        checkpoint_path,
                        map_location="cpu",
                        weights_only=True,
                    )

                loaded = safe_weights_load(
                    checkpoint_path, map_location="cpu"
                )
                self.assertIs(loaded, getattr)
                self.assertEqual(get_safe_globals(), [])

                torch.serialization.add_safe_globals([getattr])
                safe_weights_load(checkpoint_path, map_location="cpu")
                self.assertIn(getattr, get_safe_globals())
            finally:
                torch.serialization.clear_safe_globals()
                torch.serialization.add_safe_globals(initial_safe_globals)

    def test_registry_exposes_all_models(self):
        self.assertEqual(
            list_models(),
            ("autoint", "dien", "essm", "eta", "eta_legacy"),
        )

    def test_model_preparser_does_not_treat_mode_as_model_abbreviation(self):
        selected = select_model(
            ["main.py", "--model=eta", "--mode=train"]
        )
        self.assertEqual(selected, "eta")

        selected = select_model(
            ["main.py", "--model=eta_legacy", "--mode=train"]
        )
        self.assertEqual(selected, "eta_legacy")

        selected = select_model(
            ["main.py", "--model=dien", "--mode=train"]
        )
        self.assertEqual(selected, "dien")

        selected = select_model(
            ["main.py", "--model=autoint", "--mode=train"]
        )
        self.assertEqual(selected, "autoint")

    def test_eta_legacy_preserves_original_lsh_ranking(self):
        class ReturnSelectedKeys(torch.nn.Module):
            def forward(self, query, key, value, mask):
                return key

        params = AttrDict(
            num_heads=1,
            embedding_size=[2],
            reuse_hash=True,
            hash_bits=2,
            topk=1,
            device="cpu",
        )
        fixed_attention = FixedLoneAttention(params)
        legacy_attention = LegacyLoneAttention(params)
        self.assertIn("reuse_hash", fixed_attention.state_dict())
        self.assertNotIn("reuse_hash", legacy_attention.state_dict())
        projection = torch.eye(2)
        fixed_attention.reuse_hash.copy_(projection)
        legacy_attention.reuse_hash.copy_(projection)
        fixed_attention.attention = ReturnSelectedKeys()
        legacy_attention.attention = ReturnSelectedKeys()

        target = torch.tensor([[[1.0, 1.0]]])
        sequence = torch.tensor(
            [[[1.0, 1.0], [-1.0, -1.0]]]
        )
        mask = torch.tensor([[[True, True]]])

        fixed_selected = fixed_attention(target, sequence, mask)
        legacy_selected = legacy_attention(target, sequence, mask)
        self.assertTrue(torch.equal(fixed_selected, sequence[:, :1]))
        self.assertTrue(torch.equal(legacy_selected, sequence[:, 1:]))

    def test_essm_configuration_is_validated_before_model_creation(self):
        params = AttrDict(
            dropout=1.0,
            ctr_loss_weight=1.0,
            ctcvr_loss_weight=1.0,
        )
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "dropout"):
            configure_essm(params)

        params.dropout = 0.0
        params.ctr_loss_weight = 0.0
        params.ctcvr_loss_weight = 0.0
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "at least one"):
            configure_essm(params)

    def test_legacy_compiled_checkpoint_prefixes_are_removed(self):
        state_dict = OrderedDict(
            {
                "module._orig_mod.weight": torch.ones(2, 2),
                "module._orig_mod.bias": torch.ones(2),
            }
        )
        normalized = ModelHandler.normalize_checkpoint_keys(state_dict)
        self.assertEqual(list(normalized), ["weight", "bias"])

        reversed_wrappers = OrderedDict(
            {"_orig_mod.module.weight": torch.ones(2, 2)}
        )
        normalized = ModelHandler.normalize_checkpoint_keys(reversed_wrappers)
        self.assertEqual(list(normalized), ["weight"])

    def test_ddp_wraps_initialized_model_before_torch_compile(self):
        source = inspect.getsource(ModelHandler.main_ddp)
        device_position = source.index("self.params.device =")
        hf32_position = source.index("self.set_hf32()")
        init_position = source.index("self.init_model()")
        ddp_position = source.index("self.model = DDP")
        compile_position = source.index("self.set_compile_model()")

        self.assertLess(device_position, hf32_position)
        self.assertLess(hf32_position, init_position)
        self.assertLess(init_position, ddp_position)
        self.assertLess(ddp_position, compile_position)

    def test_dense_essm_contract_is_preserved(self):
        model = ESSM(input_dim=8, hidden_dims=(4,), dropout=0.0)
        outputs = model(torch.randn(3, 8))
        self.assertEqual(outputs.shape, (3, 2))
        self.assertTrue(torch.all(outputs[:, 1] <= outputs[:, 0]))

    def test_dense_dien_core_supports_interest_evolution_variants(self):
        context = torch.randn(3, 8)
        query = torch.randn(3, 6)
        history = torch.randn(3, 5, 6)
        valid_mask = torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, True, True, True],
                [False, False, False, False, False],
            ]
        )
        for gru_type in ("GRU", "AIGRU", "AGRU", "AUGRU"):
            with self.subTest(gru_type=gru_type):
                model = DIEN(
                    context_dim=8,
                    interest_dim=6,
                    dnn_hidden_dims=(8, 4),
                    attention_hidden_dims=(8, 4),
                    gru_type=gru_type,
                    dropout=0.0,
                )
                output = model(context, query, history, valid_mask)
                self.assertEqual(output.shape, (3,))
                self.assertTrue(torch.isfinite(output).all())
                self.assertTrue(torch.all((0 <= output) & (output <= 1)))

    def test_masked_gru_matches_torch_gru_cell_for_one_step(self):
        masked_gru = MaskedGRU(input_size=3, hidden_size=4)
        reference = torch.nn.GRUCell(input_size=3, hidden_size=4)
        self.assertEqual(masked_gru.bias_input.shape, (12,))
        self.assertEqual(masked_gru.bias_hidden.shape, (12,))

        with torch.no_grad():
            reference.weight_ih.copy_(masked_gru.weight_input.T)
            reference.weight_hh.copy_(masked_gru.weight_hidden.T)
            reference.bias_ih.copy_(masked_gru.bias_input)
            reference.bias_hh.copy_(masked_gru.bias_hidden)

        inputs = torch.randn(5, 1, 3)
        initial_hidden = torch.zeros(5, 4)
        expected = reference(inputs[:, 0], initial_hidden)
        actual = masked_gru(
            inputs, torch.ones(5, 1, dtype=torch.bool)
        )[:, 0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-6))

    def test_dien_defaults_to_deepctr_normalized_attention(self):
        model = DIEN(
            context_dim=8,
            interest_dim=6,
            dnn_hidden_dims=(4,),
            attention_hidden_dims=(4,),
        )
        self.assertTrue(model.local_attention.weight_normalization)
        self.assertTrue(
            DIEN_DEFAULTS["dien_attention_weight_normalization"]
        )

    def test_dien_dense_layers_use_deepctr_xavier_initialization(self):
        torch.manual_seed(11)
        model = DIEN(
            context_dim=1018,
            interest_dim=6,
            dnn_hidden_dims=(512,),
            attention_hidden_dims=(64,),
        )
        first_dense_weight = model.dnn.network[0].weight
        expected_std = (2.0 / (1024 + 512)) ** 0.5
        actual_std = first_dense_weight.std(unbiased=False).item()
        self.assertAlmostEqual(
            actual_std,
            expected_std,
            delta=expected_std * 0.05,
        )
        self.assertTrue(torch.count_nonzero(model.dnn.network[0].bias) == 0)
        # DeepCTR still uses init_std for the final bias-free logit layer.
        self.assertLess(model.output_layer.weight.std().item(), 5e-4)

    def test_dien_dropout_applies_only_to_prediction_dnn(self):
        model = DIEN(
            context_dim=8,
            interest_dim=6,
            dnn_hidden_dims=(4,),
            attention_hidden_dims=(4,),
            dropout=0.5,
        )
        self.assertTrue(
            any(
                isinstance(module, torch.nn.Dropout)
                for module in model.dnn.modules()
            )
        )
        self.assertFalse(
            any(
                isinstance(module, torch.nn.Dropout)
                for module in model.local_attention.modules()
            )
        )

    def test_dien_prediction_layer_has_trainable_scalar_bias(self):
        model = DIEN(
            context_dim=8,
            interest_dim=6,
            dnn_hidden_dims=(4,),
            attention_hidden_dims=(4,),
        )
        self.assertIsInstance(model.prediction_bias, torch.nn.Parameter)
        self.assertEqual(model.prediction_bias.shape, (1,))
        self.assertTrue(model.prediction_bias.requires_grad)

        with torch.no_grad():
            model.output_layer.weight.zero_()
            model.prediction_bias.fill_(1.25)
        output = model(
            torch.randn(2, 8),
            torch.randn(2, 6),
            torch.randn(2, 3, 6),
            torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertTrue(
            torch.allclose(output, torch.sigmoid(torch.tensor(1.25)).expand(2))
        )

    def test_dien_attention_supports_raw_and_normalized_weights(self):
        attention = LocalActivationUnit(
            interest_dim=2,
            hidden_dims=(2,),
            weight_normalization=False,
        )
        with torch.no_grad():
            for parameter in attention.parameters():
                parameter.zero_()
            attention.projection.bias.fill_(2.0)

        query = torch.zeros(1, 2)
        interests = torch.zeros(1, 3, 2)
        valid_mask = torch.tensor([[True, True, False]])
        scores = attention(query, interests, valid_mask)
        self.assertTrue(
            torch.equal(scores, torch.tensor([[2.0, 2.0, 0.0]]))
        )

        normalized_attention = LocalActivationUnit(
            interest_dim=2,
            hidden_dims=(2,),
            weight_normalization=True,
        )
        with torch.no_grad():
            for parameter in normalized_attention.parameters():
                parameter.zero_()
            normalized_attention.projection.bias.fill_(2.0)
        normalized = normalized_attention(query, interests, valid_mask)
        self.assertTrue(
            torch.allclose(normalized, torch.tensor([[0.5, 0.5, 0.0]]))
        )

    def test_aliccp_essm_forward_loss_and_backward(self):
        spec = tiny_spec()
        params = AttrDict(
            embedding_size=4,
            hidden_dims=(8, 4),
            dropout=0.0,
            ctr_loss_weight=1.0,
            ctcvr_loss_weight=1.0,
        )
        model = AliccpESSM(params, spec)
        features, labels = tiny_batch(spec)

        predictions = model(features, "train")
        self.assertEqual(set(predictions), {"ctr", "cvr", "ctcvr"})
        self.assertTrue(all(value.shape == (4,) for value in predictions.values()))
        self.assertTrue(torch.all(predictions["ctcvr"] <= predictions["ctr"]))

        loss = model.loss(predictions, labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_sequence_padding_does_not_hide_oov_id_zero(self):
        spec = tiny_spec()
        params = AttrDict(
            embedding_size=4,
            hidden_dims=(8,),
            dropout=0.0,
            ctr_loss_weight=1.0,
            ctcvr_loss_weight=1.0,
        )
        model = AliccpESSM(params, spec)
        field = spec["multi_hot_fields"][0]
        values = torch.tensor([[0, -1, -1], [1, 0, -1]])

        encoded = model.encoder._encode_sequence(field, values)
        embedding = model.encoder.embeddings[field].weight
        self.assertTrue(torch.allclose(encoded[0], embedding[0]))
        self.assertTrue(
            torch.allclose(encoded[1], (embedding[1] + embedding[0]) / 2)
        )

    def test_aliccp_dien_forward_loss_padding_and_backward(self):
        spec = tiny_spec()
        params = AttrDict(
            embedding_size=4,
            dnn_hidden_size=(8, 4),
            att_hidden_size=(8, 4),
            gru_type="GRU",
            dien_dropout=0.0,
            dien_positive_class_weight=(1.0 - 0.14) / 0.14,
        )
        model = AliccpDIEN(params, spec)
        features, labels = tiny_batch(spec)

        # Exercise both a valid OOV ID 0 and an entirely padded history row.
        for field in ("109_14", "110_14", "127_14"):
            features[field][0] = -1
            features[field][1, 0] = 0
        embedded, valid = model._embed_history(
            "109_14", torch.tensor([[0, -1]])
        )
        self.assertEqual(valid.tolist(), [[True, False]])
        self.assertTrue(
            torch.equal(embedded[0, 0], model.embeddings["109_14"].weight[0])
        )

        predictions = model(features, "train")
        default_mode_predictions = model(features)
        self.assertEqual(set(predictions), {"ctr"})
        self.assertEqual(predictions["ctr"].shape, (4,))
        self.assertTrue(torch.isfinite(predictions["ctr"]).all())
        self.assertTrue(
            torch.all((0 <= predictions["ctr"]) & (predictions["ctr"] <= 1))
        )
        self.assertTrue(torch.isfinite(default_mode_predictions["ctr"]).all())

        loss = model.loss(predictions, labels)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_aliccp_dien_target_fields_remain_in_all_padding_context(self):
        spec = tiny_spec()
        params = AttrDict(
            embedding_size=1,
            dnn_hidden_size=(1,),
            att_hidden_size=(2,),
            gru_type="GRU",
            dien_dropout=0.0,
            dien_positive_class_weight=1.0,
        )
        model = AliccpDIEN(params, spec)
        self.assertEqual(model.context_fields, tuple(spec["one_hot_fields"]))
        for target_field in ("206", "207", "216"):
            self.assertIn(target_field, model.context_fields)
        self.assertTrue(model.dien.local_attention.weight_normalization)

        features, _ = tiny_batch(spec)
        for history_field in ("109_14", "110_14", "127_14"):
            features[history_field].fill_(-1)
        features["206"].fill_(1)
        changed_features = {field: values.clone() for field, values in features.items()}
        changed_features["206"].fill_(2)

        target_context_offset = model.context_fields.index("206")
        with torch.no_grad():
            target_embedding = model.embeddings["206"].weight
            target_embedding.zero_()
            target_embedding[1].fill_(1.0)
            target_embedding[2].fill_(2.0)

            first_dense = model.dien.dnn.network[0]
            first_dense.weight.zero_()
            first_dense.bias.zero_()
            first_dense.weight[0, target_context_offset] = 1.0
            model.dien.output_layer.weight.fill_(1.0)
            model.dien.prediction_bias.zero_()

        original = model(features)["ctr"]
        changed = model(changed_features)["ctr"]
        self.assertTrue(torch.all(changed > original))

    def test_dien_configuration_rejects_invalid_values(self):
        params = AttrDict(
            dnn_hidden_size=(8,),
            att_hidden_size=(4,),
            gru_type="GRU",
            dien_dropout=1.0,
            dien_attention_weight_normalization=False,
            dien_positive_class_weight=1.0,
        )
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "dropout"):
            configure_dien(params)

        params.dien_dropout = 0.0
        params.gru_type = "invalid"
        with self.assertRaisesRegex(argparse.ArgumentTypeError, "gru_type"):
            configure_dien(params)

        params.gru_type = "GRU"
        params.dien_positive_class_weight = 0.0
        with self.assertRaisesRegex(
            argparse.ArgumentTypeError, "positive_class_weight"
        ):
            configure_dien(params)

    def test_eta_implements_the_same_framework_contract(self):
        spec = tiny_spec()
        params = AttrDict(
            max_seq_len=5,
            num_heads=2,
            deep_layers=[16],
            reuse_hash=True,
            hash_bits=8,
            topk=2,
            extra_fields=0,
            embedding_size=[4],
            device="cpu",
        )
        model = ETA(params, spec)
        features, labels = tiny_batch(spec)
        for field in ("109_14", "110_14", "127_14", "150_14"):
            features[field][0] = -1

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

    def test_eta_legacy_implements_the_framework_contract(self):
        spec = tiny_spec()
        params = AttrDict(
            max_seq_len=5,
            num_heads=2,
            deep_layers=[16],
            reuse_hash=True,
            hash_bits=8,
            topk=2,
            extra_fields=0,
            embedding_size=[4],
            device="cpu",
        )
        model = LegacyETA(params, spec)
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


if __name__ == "__main__":
    unittest.main()

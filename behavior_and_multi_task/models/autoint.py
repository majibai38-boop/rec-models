"""AutoInt model and Ali-CCP adapter for the unified training framework.

The model follows DeepCTR-Torch's AutoInt layout: a first-order linear term,
stacked interacting layers, an optional DNN branch, a bias-free projection and
a final binary prediction bias.  The Ali-CCP adapter turns every scalar or
variable-length feature group into one field embedding.  Sequence padding uses
``-1`` while feature ID ``0`` remains a valid, trainable OOV bucket.

The reference implementation used ``nn.MultiheadAttention`` directly on
batch-first tensors without enabling ``batch_first`` and applied
``CrossEntropyLoss`` to a single sigmoid output.  This implementation keeps its
benchmark-oriented 8-layer/8-head defaults, but uses the field-wise attention
and binary loss semantics of the DeepCTR reference.
"""

import argparse
import math
from typing import Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


AUTOINT_EMBEDDING_STD = 0.0001
AUTOINT_ATTENTION_STD = 0.05


class InteractingLayer(nn.Module):
    """Multi-head self-attention over feature fields.

    Inputs and outputs both have shape ``[batch, field_count, embedding_size]``.
    This is equivalent to DeepCTR-Torch's ``InteractingLayer`` while keeping the
    batch dimension explicit throughout the implementation.
    """

    def __init__(
        self,
        embedding_size: int,
        head_num: int = 2,
        use_residual: bool = True,
        scaling: bool = False,
    ) -> None:
        super().__init__()
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if head_num <= 0:
            raise ValueError("head_num must be positive")
        if embedding_size % head_num != 0:
            raise ValueError(
                "embedding_size must be an integer multiple of head_num"
            )

        self.embedding_size = int(embedding_size)
        self.head_num = int(head_num)
        self.head_dim = self.embedding_size // self.head_num
        self.use_residual = bool(use_residual)
        self.scaling = bool(scaling)

        self.query_weight = nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)
        )
        self.key_weight = nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)
        )
        self.value_weight = nn.Parameter(
            torch.empty(self.embedding_size, self.embedding_size)
        )
        if self.use_residual:
            self.residual_weight = nn.Parameter(
                torch.empty(self.embedding_size, self.embedding_size)
            )
        else:
            self.register_parameter("residual_weight", None)

        for parameter in self.parameters():
            nn.init.normal_(
                parameter, mean=0.0, std=AUTOINT_ATTENTION_STD
            )

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, field_count, _ = values.shape
        return values.reshape(
            batch_size, field_count, self.head_num, self.head_dim
        ).transpose(1, 2)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3 or inputs.size(-1) != self.embedding_size:
            raise ValueError(
                "inputs must have shape [batch, fields, {}], got {}".format(
                    self.embedding_size, tuple(inputs.shape)
                )
            )

        queries = self._split_heads(torch.matmul(inputs, self.query_weight))
        keys = self._split_heads(torch.matmul(inputs, self.key_weight))
        values = self._split_heads(torch.matmul(inputs, self.value_weight))

        attention_logits = torch.matmul(queries, keys.transpose(-2, -1))
        if self.scaling:
            attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_scores = torch.softmax(attention_logits, dim=-1)
        result = torch.matmul(attention_scores, values)
        result = result.transpose(1, 2).reshape(
            inputs.size(0), inputs.size(1), self.embedding_size
        )

        if self.residual_weight is not None:
            result = result + torch.matmul(inputs, self.residual_weight)
        return F.relu(result)


class AutoIntDNN(nn.Module):
    """DeepCTR-compatible dense branch with Xavier-normal initialization."""

    def __init__(
        self,
        input_dim: int,
        hidden_units: Sequence[int],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dimensions = (int(input_dim),) + tuple(
            int(unit) for unit in hidden_units
        )
        if len(dimensions) <= 1 or any(unit <= 0 for unit in dimensions):
            raise ValueError("DNN dimensions must be positive and non-empty")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.linears = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
        self.dropout = nn.Dropout(float(dropout))
        self.output_dim = dimensions[-1]

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for linear in self.linears:
            output = self.dropout(F.relu(linear(output)))
        return output


class AutoInt(nn.Module):
    """Dataset-independent AutoInt core operating on field embeddings."""

    def __init__(
        self,
        field_count: int,
        embedding_size: int,
        attention_layers: int = 3,
        num_heads: int = 2,
        use_residual: bool = True,
        scaling: bool = False,
        dnn_hidden_units: Sequence[int] = (256, 128),
        dnn_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.field_count = int(field_count)
        self.embedding_size = int(embedding_size)
        self.attention_layer_count = int(attention_layers)
        hidden_units = tuple(int(unit) for unit in dnn_hidden_units)

        if self.field_count <= 0:
            raise ValueError("field_count must be positive")
        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if self.attention_layer_count < 0:
            raise ValueError("attention_layers must be non-negative")
        if self.attention_layer_count == 0 and not hidden_units:
            raise ValueError("attention or DNN branch must be enabled")
        if any(unit <= 0 for unit in hidden_units):
            raise ValueError("dnn_hidden_units must contain positive integers")

        self.interacting_layers = nn.ModuleList(
            InteractingLayer(
                embedding_size=self.embedding_size,
                head_num=int(num_heads),
                use_residual=use_residual,
                scaling=scaling,
            )
            for _ in range(self.attention_layer_count)
        )
        self.dnn = (
            AutoIntDNN(
                input_dim=self.field_count * self.embedding_size,
                hidden_units=hidden_units,
                dropout=float(dnn_dropout),
            )
            if hidden_units
            else None
        )

        projection_dim = 0
        if self.interacting_layers:
            projection_dim += self.field_count * self.embedding_size
        if self.dnn is not None:
            projection_dim += self.dnn.output_dim

        # DeepCTR's AutoInt deliberately uses a bias-free projection and adds
        # the scalar prediction bias after the first-order linear logit.
        self.output_layer = nn.Linear(projection_dim, 1, bias=False)
        self.prediction_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self, field_embeddings: Tensor, linear_logit: Tensor = None
    ) -> Tensor:
        expected_shape = (self.field_count, self.embedding_size)
        if field_embeddings.ndim != 3 or tuple(
            field_embeddings.shape[1:]
        ) != expected_shape:
            raise ValueError(
                "field_embeddings must have shape [batch, {}, {}], got {}".format(
                    self.field_count,
                    self.embedding_size,
                    tuple(field_embeddings.shape),
                )
            )

        branch_outputs = []
        if self.interacting_layers:
            attention_output = field_embeddings
            for layer in self.interacting_layers:
                attention_output = layer(attention_output)
            branch_outputs.append(torch.flatten(attention_output, start_dim=1))

        flattened_embeddings = torch.flatten(field_embeddings, start_dim=1)
        if self.dnn is not None:
            branch_outputs.append(self.dnn(flattened_embeddings))

        combined = (
            branch_outputs[0]
            if len(branch_outputs) == 1
            else torch.cat(branch_outputs, dim=-1)
        )
        if linear_logit is None:
            linear_logit = field_embeddings.new_zeros(
                (field_embeddings.size(0), 1)
            )
        elif linear_logit.ndim == 1:
            linear_logit = linear_logit.unsqueeze(-1)
        if linear_logit.shape != (field_embeddings.size(0), 1):
            raise ValueError(
                "linear_logit must have shape [batch] or [batch, 1], got {}".format(
                    tuple(linear_logit.shape)
                )
            )

        logits = (
            linear_logit + self.output_layer(combined) + self.prediction_bias
        )
        return torch.sigmoid(logits).squeeze(-1)


class AliccpAutoIntFeatureEncoder(nn.Module):
    """Encode every Ali-CCP feature group as one AutoInt field token."""

    def __init__(self, spec: Mapping, embedding_size: int) -> None:
        super().__init__()
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")

        self.one_hot_fields = tuple(spec["one_hot_fields"])
        self.sequence_fields = tuple(spec["multi_hot_fields"]) + tuple(
            spec["special_fields"]
        )
        self.field_names = self.one_hot_fields + self.sequence_fields
        if not self.field_names:
            raise ValueError("Ali-CCP spec must contain at least one feature field")
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("Ali-CCP feature groups contain duplicate field names")

        vocab_lengths = spec["vocab_length"]
        missing_vocab = [field for field in self.field_names if field not in vocab_lengths]
        if missing_vocab:
            raise ValueError(
                f"Missing vocab_length entries for fields: {missing_vocab}"
            )

        self.deep_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(
                    int(vocab_lengths[field]) + 1, int(embedding_size)
                )
                for field in self.field_names
            }
        )
        self.linear_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(vocab_lengths[field]) + 1, 1)
                for field in self.field_names
            }
        )
        for embeddings in (self.deep_embeddings, self.linear_embeddings):
            for embedding in embeddings.values():
                nn.init.normal_(
                    embedding.weight,
                    mean=0.0,
                    std=AUTOINT_EMBEDDING_STD,
                )

        self.embedding_size = int(embedding_size)
        self.field_count = len(self.field_names)

    @staticmethod
    def _safe_ids(values: Tensor) -> Tuple[Tensor, Tensor]:
        valid = values.ge(0)
        return torch.where(valid, values, torch.zeros_like(values)), valid

    def _encode_one_hot(
        self, field: str, values: Tensor, embeddings: nn.ModuleDict
    ) -> Tensor:
        if values.ndim == 2 and values.size(1) == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(
                f"One-hot field {field} must have shape [batch], "
                f"got {tuple(values.shape)}"
            )
        safe_values, valid = self._safe_ids(values)
        encoded = embeddings[field](safe_values)
        return encoded * valid.unsqueeze(-1).to(encoded.dtype)

    def _encode_sequence(
        self, field: str, values: Tensor, embeddings: nn.ModuleDict
    ) -> Tensor:
        if values.ndim != 2:
            raise ValueError(
                f"Sequence field {field} must have shape [batch, length], "
                f"got {tuple(values.shape)}"
            )
        safe_values, valid = self._safe_ids(values)
        encoded = embeddings[field](safe_values)
        mask = valid.unsqueeze(-1).to(encoded.dtype)
        summed = torch.sum(encoded * mask, dim=1)
        counts = torch.sum(mask, dim=1).clamp_min(1.0)
        return summed / counts

    def _encode_with(self, features: Mapping[str, Tensor], embeddings):
        encoded = [
            self._encode_one_hot(field, features[field], embeddings)
            for field in self.one_hot_fields
        ]
        encoded.extend(
            self._encode_sequence(field, features[field], embeddings)
            for field in self.sequence_fields
        )
        return encoded

    def forward(self, features: Mapping[str, Tensor]):
        missing = [field for field in self.field_names if field not in features]
        if missing:
            raise KeyError(f"Missing Ali-CCP AutoInt feature fields: {missing}")

        field_embeddings = torch.stack(
            self._encode_with(features, self.deep_embeddings), dim=1
        )
        linear_terms = self._encode_with(features, self.linear_embeddings)
        linear_logit = torch.stack(linear_terms, dim=1).sum(dim=1)
        return field_embeddings, linear_logit


class AliccpAutoInt(nn.Module):
    """Ali-CCP adapter implementing the shared framework model contract."""

    def __init__(self, params, spec: Mapping) -> None:
        super().__init__()
        embedding_size = int(params.embedding_size)
        self.encoder = AliccpAutoIntFeatureEncoder(spec, embedding_size)
        self.autoint = AutoInt(
            field_count=self.encoder.field_count,
            embedding_size=embedding_size,
            attention_layers=int(params.autoint_attention_layers),
            num_heads=int(params.autoint_num_heads),
            use_residual=bool(params.autoint_residual),
            scaling=bool(params.autoint_scaling),
            dnn_hidden_units=tuple(params.autoint_dnn_hidden_units),
            dnn_dropout=float(params.autoint_dropout),
        )
        self.positive_class_weight = float(
            params.autoint_positive_class_weight
        )
        if self.positive_class_weight <= 0:
            raise ValueError("autoint_positive_class_weight must be positive")

    def forward(self, features: Mapping[str, Tensor], mode: str = "train"):
        del mode
        field_embeddings, linear_logit = self.encoder(features)
        return {"ctr": self.autoint(field_embeddings, linear_logit)}

    def loss(self, predictions, labels):
        probabilities = predictions["ctr"]
        targets = labels["y"].to(probabilities.dtype)
        if targets.shape != probabilities.shape:
            targets = targets.reshape(probabilities.shape)
        sample_weights = 1.0 + (
            self.positive_class_weight - 1.0
        ) * targets
        return F.binary_cross_entropy(
            probabilities, targets, weight=sample_weights
        )


# Compatibility names matching the conventions used by the other adapters.
AutoIntHandler = AliccpAutoInt
MyAutoInt = AutoInt


# The test_qps schema starts with 23 fields.  The Ali-CCP adapter's historical
# extra-field generator adds 232 scalar fields and 45 paired sequence fields,
# resulting in the 300 fields used by the reference AutoInt benchmark.
AUTOINT_DEFAULTS = {
    "model": "autoint",
    "autoint_attention_layers": 8,
    "autoint_num_heads": 8,
    "autoint_residual": True,
    "autoint_scaling": False,
    # The benchmark model is attention-only; users can enable DeepCTR's
    # parallel DNN branch with --autoint_dnn_hidden_units.
    "autoint_dnn_hidden_units": (),
    "autoint_dropout": 0.0,
    "autoint_positive_class_weight": 1.0,
    "extra_fields": 232,
    "find_unused_parameters": False,
}


def _parse_hidden_units(value):
    text = str(value).strip()
    if not text or text.lower() in {"none", "off"}:
        return ()
    try:
        dimensions = tuple(int(item) for item in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "autoint_dnn_hidden_units must be a comma-separated integer list"
        ) from exc
    if any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError(
            "autoint_dnn_hidden_units must contain positive integers"
        )
    return dimensions


def add_autoint_arguments(parser):
    """Add AutoInt-only architecture and loss arguments to the parser."""
    parser.add_argument(
        "--autoint_attention_layers",
        type=int,
        default=AUTOINT_DEFAULTS["autoint_attention_layers"],
        help="number of stacked AutoInt interacting layers",
    )
    parser.add_argument(
        "--autoint_num_heads",
        type=int,
        default=AUTOINT_DEFAULTS["autoint_num_heads"],
        help="number of attention heads in each interacting layer",
    )
    parser.add_argument(
        "--autoint_residual",
        type=str.lower,
        choices=("true", "false"),
        default=AUTOINT_DEFAULTS["autoint_residual"],
        help="enable the learned residual projection",
    )
    parser.add_argument(
        "--autoint_scaling",
        type=str.lower,
        choices=("true", "false"),
        default=AUTOINT_DEFAULTS["autoint_scaling"],
        help="scale attention logits by sqrt(head dimension)",
    )
    parser.add_argument(
        "--autoint_dnn_hidden_units",
        type=_parse_hidden_units,
        default=AUTOINT_DEFAULTS["autoint_dnn_hidden_units"],
        help="optional comma-separated DeepCTR DNN widths; empty disables it",
    )
    parser.add_argument(
        "--autoint_dropout",
        type=float,
        default=AUTOINT_DEFAULTS["autoint_dropout"],
        help="dropout used by the optional DNN branch",
    )
    parser.add_argument(
        "--autoint_positive_class_weight",
        type=float,
        default=AUTOINT_DEFAULTS["autoint_positive_class_weight"],
        help="positive-sample multiplier in the CTR loss",
    )
    return parser


def _as_bool(value, name):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise argparse.ArgumentTypeError(f"{name} must be true or false")


def configure_autoint(params):
    """Validate AutoInt options before distributed model construction."""
    params.autoint_attention_layers = int(params.autoint_attention_layers)
    params.autoint_num_heads = int(params.autoint_num_heads)
    params.autoint_dnn_hidden_units = tuple(
        int(unit) for unit in params.autoint_dnn_hidden_units
    )
    params.autoint_residual = _as_bool(
        params.autoint_residual, "autoint_residual"
    )
    params.autoint_scaling = _as_bool(
        params.autoint_scaling, "autoint_scaling"
    )

    if params.autoint_attention_layers < 0:
        raise argparse.ArgumentTypeError(
            "autoint_attention_layers must be non-negative"
        )
    if params.autoint_num_heads <= 0:
        raise argparse.ArgumentTypeError("autoint_num_heads must be positive")
    if (
        params.autoint_attention_layers > 0
        and params.embedding_size % params.autoint_num_heads != 0
    ):
        raise argparse.ArgumentTypeError(
            "embedding_size must be divisible by autoint_num_heads"
        )
    if any(unit <= 0 for unit in params.autoint_dnn_hidden_units):
        raise argparse.ArgumentTypeError(
            "autoint_dnn_hidden_units must contain positive integers"
        )
    if (
        params.autoint_attention_layers == 0
        and not params.autoint_dnn_hidden_units
    ):
        raise argparse.ArgumentTypeError(
            "attention or DNN branch must be enabled"
        )
    if not 0.0 <= params.autoint_dropout < 1.0:
        raise argparse.ArgumentTypeError("autoint_dropout must be in [0, 1)")
    if params.autoint_positive_class_weight <= 0:
        raise argparse.ArgumentTypeError(
            "autoint_positive_class_weight must be positive"
        )
    return params


def build_autoint(params, spec):
    """Build the Ali-CCP AutoInt adapter through the framework contract."""
    return AliccpAutoInt(params, spec)


__all__ = [
    "AUTOINT_DEFAULTS",
    "AliccpAutoInt",
    "AliccpAutoIntFeatureEncoder",
    "AutoInt",
    "AutoIntDNN",
    "AutoIntHandler",
    "InteractingLayer",
    "MyAutoInt",
    "add_autoint_arguments",
    "build_autoint",
    "configure_autoint",
]

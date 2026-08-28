# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Self-contained DIEN model adapted to the unified Ali-CCP framework.

The source implementation used ``deepctr_torch.models.DIEN`` and replaced its
packed-sequence GRUs with project-local layers.  This version follows the
DeepCTR execution semantics while removing the undeclared ``deepctr_torch``
runtime dependency:

* current item features ``206``, ``207`` and ``216`` form the target query;
* ``109_14``, ``110_14`` and ``127_14`` form the aligned behavior sequence;
* one GRU extracts latent interests and a second GRU evolves them;
* target-aware local attention produces the final interest representation;
* a deep network predicts click-through probability.

Ali-CCP uses ``-1`` for sequence padding and keeps ID ``0`` as a valid OOV ID.
Padding is therefore masked before embedding instead of being rewritten into
an unmasked zero ID.  Negative history is not present in ``aliccp_out`` and the
source model disabled negative sampling, so the auxiliary DIEN loss remains
disabled here as well.
"""

import argparse
import math
from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MULTIHOT_MAP = {
    "109_14": "206",
    "110_14": "207",
    "127_14": "216",
    "150_14": "210",
}

# The remote model intentionally used only these three scalar target features.
# ``210`` is itself a sequence field in Ali-CCP, so its mapped history is not a
# compatible scalar-query pair and is not part of the interest path.
DIEN_TARGET_FIELDS = ("206", "207", "216")
DIEN_HISTORY_FIELDS = {
    target: history
    for history, target in MULTIHOT_MAP.items()
    if target in DIEN_TARGET_FIELDS
}
SUPPORTED_GRU_TYPES = ("GRU", "AIGRU", "AGRU", "AUGRU")
DIEN_INIT_STD = 0.0001


class _MLP(nn.Module):
    """A small reusable ReLU MLP using DeepCTR's dense initialization."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(int(dim) <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            if dropout:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = current_dim

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class LocalActivationUnit(nn.Module):
    """Compute target-aware attention weights over an interest sequence."""

    def __init__(
        self,
        interest_dim: int,
        hidden_dims: Sequence[int],
        dropout: float = 0.0,
        weight_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.interest_dim = int(interest_dim)
        self.weight_normalization = bool(weight_normalization)
        self.mlp = _MLP(4 * self.interest_dim, hidden_dims, dropout)
        self.projection = nn.Linear(self.mlp.output_dim, 1)
        nn.init.xavier_normal_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(
        self,
        query: Tensor,
        interests: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if query.ndim != 2 or query.size(-1) != self.interest_dim:
            raise ValueError(
                "query must have shape [batch, {}]".format(self.interest_dim)
            )
        if interests.ndim != 3 or interests.size(-1) != self.interest_dim:
            raise ValueError(
                "interests must have shape [batch, length, {}]".format(
                    self.interest_dim
                )
            )
        if valid_mask.shape != interests.shape[:2]:
            raise ValueError("valid_mask must match the first two interest dimensions")

        expanded_query = query.unsqueeze(1).expand(-1, interests.size(1), -1)
        attention_inputs = torch.cat(
            (
                expanded_query,
                interests,
                expanded_query - interests,
                expanded_query * interests,
            ),
            dim=-1,
        )
        scores = self.projection(self.mlp(attention_inputs)).squeeze(-1)

        if not self.weight_normalization:
            return torch.where(valid_mask, scores, torch.zeros_like(scores))

        # ``finfo.min`` avoids NaNs for an all-padding row.  Multiplication and
        # renormalization below then turn that row into all-zero weights.
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid_mask.to(weights.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


class MaskedGRU(nn.Module):
    """DeepCTR-compatible GRU equations without packed sequences.

    Separate input/hidden biases are required for parity with ``nn.GRU`` and
    DeepCTR's AGRU/AUGRU cells: the hidden candidate bias is multiplied by the
    reset gate and therefore cannot be folded into the input bias.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        zero_bias: bool = False,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("GRU dimensions must be positive")
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.zero_bias = bool(zero_bias)
        self.weight_input = nn.Parameter(
            torch.empty(self.input_size, 3 * self.hidden_size)
        )
        self.weight_hidden = nn.Parameter(
            torch.empty(self.hidden_size, 3 * self.hidden_size)
        )
        self.bias_input = nn.Parameter(torch.empty(3 * self.hidden_size))
        self.bias_hidden = nn.Parameter(torch.empty(3 * self.hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # DeepCTR overwrites every recurrent weight with N(0, init_std).
        nn.init.normal_(self.weight_input, mean=0.0, std=DIEN_INIT_STD)
        nn.init.normal_(self.weight_hidden, mean=0.0, std=DIEN_INIT_STD)
        if self.zero_bias:
            # DeepCTR's custom AGRU/AUGRU cells explicitly zero both biases.
            nn.init.zeros_(self.bias_input)
            nn.init.zeros_(self.bias_hidden)
        else:
            # Its nn.GRU paths retain PyTorch's default uniform bias init.
            bound = 1.0 / math.sqrt(self.hidden_size)
            nn.init.uniform_(self.bias_input, -bound, bound)
            nn.init.uniform_(self.bias_hidden, -bound, bound)

    def forward(
        self,
        inputs: Tensor,
        valid_mask: Tensor,
        attention_scores: Optional[Tensor] = None,
        gru_type: str = "GRU",
    ) -> Tensor:
        if inputs.ndim != 3 or inputs.size(-1) != self.input_size:
            raise ValueError(
                "inputs must have shape [batch, length, {}]".format(
                    self.input_size
                )
            )
        if valid_mask.shape != inputs.shape[:2]:
            raise ValueError("valid_mask must match the first two input dimensions")
        if gru_type not in ("GRU", "AGRU", "AUGRU"):
            raise ValueError(f"unsupported recurrent gate type: {gru_type}")
        if gru_type in ("AGRU", "AUGRU"):
            if attention_scores is None or attention_scores.shape != valid_mask.shape:
                raise ValueError(
                    f"{gru_type} requires one attention score per sequence step"
                )

        hidden = inputs.new_zeros(inputs.size(0), self.hidden_size)
        outputs = []
        for step in range(inputs.size(1)):
            input_gates = (
                inputs[:, step] @ self.weight_input + self.bias_input
            )
            hidden_gates = hidden @ self.weight_hidden + self.bias_hidden
            reset, keep = torch.sigmoid(
                input_gates[:, : 2 * self.hidden_size]
                + hidden_gates[:, : 2 * self.hidden_size]
            ).chunk(2, dim=-1)
            candidate = torch.tanh(
                input_gates[:, 2 * self.hidden_size :]
                + reset * hidden_gates[:, 2 * self.hidden_size :]
            )

            candidate_weight = 1.0 - keep
            if gru_type == "AGRU":
                candidate_weight = attention_scores[:, step].unsqueeze(-1)
            elif gru_type == "AUGRU":
                # Match DeepCTR-Torch's AUGRU cell: its update gate is the
                # candidate-state coefficient and is scaled by attention.
                candidate_weight = (
                    keep * attention_scores[:, step].unsqueeze(-1)
                )
            next_hidden = (
                candidate_weight * candidate
                + (1.0 - candidate_weight) * hidden
            )
            hidden = torch.where(
                valid_mask[:, step].unsqueeze(-1), next_hidden, hidden
            )
            outputs.append(hidden)

        if not outputs:
            return inputs.new_zeros(inputs.size(0), 0, self.hidden_size)
        return torch.stack(outputs, dim=1)


class DIEN(nn.Module):
    """Dataset-independent dense DIEN core.

    ``context`` contains every scalar feature, including the current-item
    fields. ``query`` and ``history`` additionally contain aligned current-item
    and behavior representations for interest evolution.
    """

    def __init__(
        self,
        context_dim: int,
        interest_dim: int,
        dnn_hidden_dims: Sequence[int] = (256, 128),
        attention_hidden_dims: Sequence[int] = (64, 16),
        gru_type: str = "GRU",
        dropout: float = 0.0,
        attention_weight_normalization: bool = True,
    ) -> None:
        super().__init__()
        gru_type = str(gru_type).upper()
        if gru_type not in SUPPORTED_GRU_TYPES:
            raise ValueError(
                f"gru_type must be one of {SUPPORTED_GRU_TYPES}, got {gru_type!r}"
            )
        if context_dim <= 0 or interest_dim <= 0:
            raise ValueError("DIEN input dimensions must be positive")

        self.context_dim = int(context_dim)
        self.interest_dim = int(interest_dim)
        self.gru_type = gru_type
        self.interest_extractor = MaskedGRU(interest_dim, interest_dim)
        self.local_attention = LocalActivationUnit(
            interest_dim,
            attention_hidden_dims,
            # DeepCTR fixes local-attention dropout at zero. ``dnn_dropout``
            # applies only to the final prediction network.
            dropout=0.0,
            weight_normalization=attention_weight_normalization,
        )
        self.interest_evolution = MaskedGRU(
            interest_dim,
            interest_dim,
            zero_bias=gru_type in ("AGRU", "AUGRU"),
        )
        self.dnn = _MLP(
            context_dim + interest_dim,
            dnn_hidden_dims,
            dropout,
        )
        self.output_layer = nn.Linear(self.dnn.output_dim, 1, bias=False)
        nn.init.normal_(
            self.output_layer.weight, mean=0.0, std=DIEN_INIT_STD
        )
        # DeepCTR's bias-free final Linear is followed by PredictionLayer,
        # whose default binary path owns one trainable scalar bias.
        self.prediction_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        context: Tensor,
        query: Tensor,
        history: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if context.ndim != 2 or context.size(-1) != self.context_dim:
            raise ValueError(
                "context must have shape [batch, {}]".format(self.context_dim)
            )
        if query.ndim != 2 or query.size(-1) != self.interest_dim:
            raise ValueError(
                "query must have shape [batch, {}]".format(self.interest_dim)
            )
        if history.ndim != 3 or history.size(-1) != self.interest_dim:
            raise ValueError(
                "history must have shape [batch, length, {}]".format(
                    self.interest_dim
                )
            )

        extracted = self.interest_extractor(history, valid_mask)
        if self.gru_type == "GRU":
            evolved_sequence = self.interest_evolution(extracted, valid_mask)
            attention = self.local_attention(
                query, evolved_sequence, valid_mask
            )
            evolved_interest = torch.sum(
                evolved_sequence * attention.unsqueeze(-1), dim=1
            )
        else:
            attention = self.local_attention(query, extracted, valid_mask)
            if self.gru_type == "AIGRU":
                evolved_sequence = self.interest_evolution(
                    extracted * attention.unsqueeze(-1), valid_mask
                )
            else:
                evolved_sequence = self.interest_evolution(
                    extracted,
                    valid_mask,
                    attention_scores=attention,
                    gru_type=self.gru_type,
                )
            evolved_interest = evolved_sequence[:, -1]

        deep_inputs = torch.cat((context, evolved_interest), dim=-1)
        logits = self.output_layer(self.dnn(deep_inputs)) + self.prediction_bias
        return torch.sigmoid(logits).squeeze(-1)


class AliccpDIEN(nn.Module):
    """Ali-CCP adapter implementing the shared framework model contract."""

    def __init__(self, params, spec: Mapping) -> None:
        super().__init__()
        embedding_dim = int(params.embedding_size)
        if embedding_dim <= 0:
            raise ValueError("embedding_size must be positive")

        one_hot_fields = tuple(spec["one_hot_fields"])
        self.history_pairs: Tuple[Tuple[str, str], ...] = tuple(
            (target, DIEN_HISTORY_FIELDS[target])
            for target in DIEN_TARGET_FIELDS
        )
        required_targets = {target for target, _ in self.history_pairs}
        required_histories = {history for _, history in self.history_pairs}
        missing_targets = sorted(required_targets.difference(one_hot_fields))
        missing_histories = sorted(
            required_histories.difference(spec["multi_hot_fields"])
        )
        if missing_targets or missing_histories:
            raise ValueError(
                "Ali-CCP DIEN fields are missing; targets={}, histories={}".format(
                    missing_targets, missing_histories
                )
            )

        # DeepCTR concatenates every scalar sparse embedding (including the
        # current-item fields) with the evolved interest before the final DNN.
        # Non-history variable-length fields are not part of that computation.
        self.context_fields = one_hot_fields
        if not self.context_fields:
            raise ValueError("DIEN requires at least one scalar context field")

        embedding_fields = self.context_fields + tuple(
            history for _, history in self.history_pairs
        )
        if len(set(embedding_fields)) != len(embedding_fields):
            raise ValueError("DIEN embedding field names must be unique")
        vocab_lengths = spec["vocab_length"]
        missing_vocab = [field for field in embedding_fields if field not in vocab_lengths]
        if missing_vocab:
            raise ValueError(f"Missing vocab_length entries for: {missing_vocab}")

        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(vocab_lengths[field]) + 1, embedding_dim)
                for field in embedding_fields
            }
        )
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=0.0001)

        interest_dim = len(self.history_pairs) * embedding_dim
        context_dim = len(self.context_fields) * embedding_dim
        attention_weight_normalization = params.get(
            "dien_attention_weight_normalization", True
        )
        if isinstance(attention_weight_normalization, str):
            attention_weight_normalization = (
                attention_weight_normalization.lower() == "true"
            )
        self.dien = DIEN(
            context_dim=context_dim,
            interest_dim=interest_dim,
            dnn_hidden_dims=tuple(int(dim) for dim in params.dnn_hidden_size),
            attention_hidden_dims=tuple(
                int(dim) for dim in params.att_hidden_size
            ),
            gru_type=params.gru_type,
            dropout=float(params.dien_dropout),
            attention_weight_normalization=attention_weight_normalization,
        )
        self.positive_class_weight = float(params.dien_positive_class_weight)
        if self.positive_class_weight <= 0:
            raise ValueError("dien_positive_class_weight must be positive")

    def _embed_one_hot(self, field: str, values: Tensor) -> Tensor:
        if values.ndim == 2 and values.size(1) == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(
                f"One-hot field {field} must have shape [batch], "
                f"got {tuple(values.shape)}"
            )
        valid = values.ge(0)
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        return self.embeddings[field](safe_values) * valid.unsqueeze(-1).to(
            self.embeddings[field].weight.dtype
        )

    def _embed_history(self, field: str, values: Tensor) -> Tuple[Tensor, Tensor]:
        if values.ndim != 2:
            raise ValueError(
                f"History field {field} must have shape [batch, length], "
                f"got {tuple(values.shape)}"
            )
        valid = values.ge(0)
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        return self.embeddings[field](safe_values), valid

    def forward(self, features: Mapping[str, Tensor], mode: str = "train"):
        del mode
        required_fields = self.context_fields + tuple(
            history for _, history in self.history_pairs
        )
        missing = [field for field in required_fields if field not in features]
        if missing:
            raise KeyError(f"Missing Ali-CCP DIEN feature fields: {missing}")

        context = torch.cat(
            [
                self._embed_one_hot(field, features[field])
                for field in self.context_fields
            ],
            dim=-1,
        )

        query_parts = []
        history_parts = []
        history_masks = []
        sequence_shape = None
        for target_field, history_field in self.history_pairs:
            query_parts.append(
                self._embed_one_hot(target_field, features[target_field])
            )
            embedded_history, valid = self._embed_history(
                history_field, features[history_field]
            )
            if sequence_shape is None:
                sequence_shape = valid.shape
            elif valid.shape != sequence_shape:
                raise ValueError("DIEN history fields must have identical shapes")
            history_parts.append(embedded_history)
            history_masks.append(valid)

        # The three history attributes describe the same behavior event.  A
        # timestep is valid only when every component is present.
        valid_mask = torch.stack(history_masks, dim=0).all(dim=0)
        mask = valid_mask.unsqueeze(-1).to(history_parts[0].dtype)
        history = torch.cat(
            [embedded * mask for embedded in history_parts], dim=-1
        )
        query = torch.cat(query_parts, dim=-1)
        return {"ctr": self.dien(context, query, history, valid_mask)}

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


# Compatibility names retained from the source file.
MyDIEN = DIEN
DIENHandler = AliccpDIEN


# The source benchmark doubled each listed width immediately before creating
# the model.  Store the final widths directly so command-line values are never
# transformed implicitly a second time.
DIEN_DEFAULT_DNN_HIDDEN_SIZE = tuple(
    width * 2
    for width in (
        1024,
        1024,
        1024,
        1024,
        256,
        256,
        256,
        256,
        128,
        128,
        128,
        128,
    )
)
DIEN_DEFAULT_ATT_HIDDEN_SIZE = tuple(
    width * 2
    for width in (
        1024,
        1024,
        1024,
        1024,
        256,
        256,
        256,
        256,
        128,
        128,
        128,
        128,
        64,
        64,
        64,
        64,
        16,
        16,
        16,
        16,
    )
)

DIEN_DEFAULTS = {
    "model": "dien",
    "dnn_hidden_size": DIEN_DEFAULT_DNN_HIDDEN_SIZE,
    "att_hidden_size": DIEN_DEFAULT_ATT_HIDDEN_SIZE,
    "gru_type": "GRU",
    "dien_dropout": 0.0,
    # Match DeepCTR DIEN rather than the legacy wrapper's dropped argument.
    "dien_attention_weight_normalization": True,
    # Equivalent to the source loss's (1 - 0.14) / 0.14 multiplier.
    "dien_positive_class_weight": (1.0 - 0.14) / 0.14,
    "extra_fields": 100,
    "find_unused_parameters": False,
}


def _parse_positive_ints(value):
    try:
        dimensions = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "value must be a comma-separated integer list"
        ) from exc
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError("all dimensions must be positive")
    return dimensions


def add_dien_arguments(parser):
    """Add DIEN-only architecture and loss arguments to the shared parser."""
    parser.add_argument(
        "--dnn_hidden_size",
        type=_parse_positive_ints,
        default=DIEN_DEFAULT_DNN_HIDDEN_SIZE,
        help="comma-separated widths of DIEN's prediction network",
    )
    parser.add_argument(
        "--att_hidden_size",
        type=_parse_positive_ints,
        default=DIEN_DEFAULT_ATT_HIDDEN_SIZE,
        help="comma-separated widths of DIEN's local attention network",
    )
    parser.add_argument(
        "--gru_type",
        type=str.upper,
        choices=SUPPORTED_GRU_TYPES,
        default=DIEN_DEFAULTS["gru_type"],
        help="interest evolution variant",
    )
    parser.add_argument(
        "--dien_dropout",
        type=float,
        default=DIEN_DEFAULTS["dien_dropout"],
        help="dropout after prediction-DNN hidden activations",
    )
    parser.add_argument(
        "--dien_attention_weight_normalization",
        type=str.lower,
        choices=("true", "false"),
        default=DIEN_DEFAULTS["dien_attention_weight_normalization"],
        help="normalize local attention scores with softmax",
    )
    parser.add_argument(
        "--dien_positive_class_weight",
        type=float,
        default=DIEN_DEFAULTS["dien_positive_class_weight"],
        help="positive-sample multiplier in the CTR loss",
    )
    return parser


def configure_dien(params):
    """Validate DIEN parameters before distributed model construction."""
    params.dnn_hidden_size = tuple(int(dim) for dim in params.dnn_hidden_size)
    params.att_hidden_size = tuple(int(dim) for dim in params.att_hidden_size)
    if not params.dnn_hidden_size or any(
        dim <= 0 for dim in params.dnn_hidden_size
    ):
        raise argparse.ArgumentTypeError(
            "dnn_hidden_size must contain positive integers"
        )
    if not params.att_hidden_size or any(
        dim <= 0 for dim in params.att_hidden_size
    ):
        raise argparse.ArgumentTypeError(
            "att_hidden_size must contain positive integers"
        )
    params.gru_type = str(params.gru_type).upper()
    if params.gru_type not in SUPPORTED_GRU_TYPES:
        raise argparse.ArgumentTypeError(
            f"gru_type must be one of {SUPPORTED_GRU_TYPES}"
        )
    if not 0.0 <= params.dien_dropout < 1.0:
        raise argparse.ArgumentTypeError("dien_dropout must be in [0, 1)")
    normalization = params.dien_attention_weight_normalization
    if isinstance(normalization, str):
        normalization = normalization == "true"
    if not isinstance(normalization, bool):
        raise argparse.ArgumentTypeError(
            "dien_attention_weight_normalization must be true or false"
        )
    params.dien_attention_weight_normalization = normalization
    if params.dien_positive_class_weight <= 0:
        raise argparse.ArgumentTypeError(
            "dien_positive_class_weight must be positive"
        )
    return params


def build_dien(params, spec):
    """Build the Ali-CCP DIEN adapter through the framework contract."""
    return AliccpDIEN(params, spec)


__all__ = [
    "AliccpDIEN",
    "DIEN",
    "DIENHandler",
    "DIEN_DEFAULTS",
    "LocalActivationUnit",
    "MULTIHOT_MAP",
    "MaskedGRU",
    "MyDIEN",
    "add_dien_arguments",
    "build_dien",
    "configure_dien",
]

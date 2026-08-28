"""PyTorch ESMM core, Ali-CCP adapter and framework registration hooks.

``ESSM`` remains a dataset-independent dense core. ``AliccpESSM`` adds the
field embeddings, sequence pooling, framework output contract and two-task
loss needed to train that core on ``aliccp_out``.

The model returns two probabilities in this order::

    [P(click), P(click and conversion)]

The second probability is constrained by ESMM's defining equation:

    P(click and conversion) = P(click) * P(conversion | click)
"""

import argparse
from typing import Mapping, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class _Tower(nn.Module):
    """Task-specific multilayer perceptron."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        # Keep the final layer bias-free to match this project's ESMM.
        layers.append(nn.Linear(current_dim, 1, bias=False))
        self.network = nn.Sequential(*layers)

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class ESSM(nn.Module):
    """Entire Space Multi-Task Model implemented with plain PyTorch.

    Args:
        input_dim: Number of input features after encoding/concatenation.
        hidden_dims: Hidden-layer widths used by both task towers.
        dropout: Dropout probability applied after every hidden activation.

    Input:
        A floating-point tensor shaped ``[batch_size, input_dim]``.

    Output:
        A tensor shaped ``[batch_size, 2]``. Column 0 is CTR and column 1
        is CTCVR. CVR is an internal auxiliary prediction rather than a
        directly supervised output.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not hidden_dims or any(dim <= 0 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive integers")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.input_dim = input_dim
        self.ctr_tower = _Tower(input_dim, hidden_dims, dropout)
        self.cvr_tower = _Tower(input_dim, hidden_dims, dropout)

    def probabilities(self, inputs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Return CTR, CVR and CTCVR probabilities separately."""
        if inputs.ndim != 2 or inputs.size(1) != self.input_dim:
            raise ValueError(
                "inputs must have shape [batch_size, {}], got {}".format(
                    self.input_dim, tuple(inputs.shape)
                )
            )

        ctr = torch.sigmoid(self.ctr_tower(inputs))
        cvr = torch.sigmoid(self.cvr_tower(inputs))
        ctcvr = ctr * cvr
        return ctr, cvr, ctcvr

    def forward(self, inputs: Tensor) -> Tensor:
        ctr, _, ctcvr = self.probabilities(inputs)
        return torch.cat((ctr, ctcvr), dim=1)


class AliccpFeatureEncoder(nn.Module):
    """Embed and pool an Ali-CCP feature dictionary into one dense tensor."""

    def __init__(self, spec: Mapping, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.one_hot_fields = tuple(spec["one_hot_fields"])
        self.sequence_fields = tuple(spec["multi_hot_fields"]) + tuple(
            spec["special_fields"]
        )
        self.field_names = self.one_hot_fields + self.sequence_fields
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("Ali-CCP feature groups contain duplicate field names")

        vocab_lengths = spec["vocab_length"]
        missing_vocab = [field for field in self.field_names if field not in vocab_lengths]
        if missing_vocab:
            raise ValueError(
                f"Missing vocab_length entries for fields: {missing_vocab}"
            )

        self.embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(int(vocab_lengths[field]) + 1, embedding_dim)
                for field in self.field_names
            }
        )
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, mean=0.0, std=(2 / 512) ** 0.5)

        self.output_dim = len(self.field_names) * embedding_dim

    def _encode_one_hot(self, field: str, values: Tensor) -> Tensor:
        if values.ndim == 2 and values.size(1) == 1:
            values = values[:, 0]
        if values.ndim != 1:
            raise ValueError(
                f"One-hot field {field} must have shape [batch], "
                f"got {tuple(values.shape)}"
            )
        return self.embeddings[field](values)

    def _encode_sequence(self, field: str, values: Tensor) -> Tensor:
        if values.ndim != 2:
            raise ValueError(
                f"Sequence field {field} must have shape [batch, length], "
                f"got {tuple(values.shape)}"
            )

        # -1 is padding, while 0 is a valid OOV feature ID.
        valid = values.ge(0)
        safe_values = torch.where(valid, values, torch.zeros_like(values))
        embedded = self.embeddings[field](safe_values)
        mask = valid.unsqueeze(-1).to(embedded.dtype)
        summed = torch.sum(embedded * mask, dim=1)
        counts = torch.sum(mask, dim=1).clamp_min(1.0)
        return summed / counts

    def forward(self, features: Mapping[str, Tensor]) -> Tensor:
        missing = [field for field in self.field_names if field not in features]
        if missing:
            raise KeyError(f"Missing Ali-CCP feature fields: {missing}")

        encoded = [
            self._encode_one_hot(field, features[field])
            for field in self.one_hot_fields
        ]
        encoded.extend(
            self._encode_sequence(field, features[field])
            for field in self.sequence_fields
        )
        return torch.cat(encoded, dim=1)


class AliccpESSM(nn.Module):
    """Ali-CCP adapter implementing the shared framework's model contract."""

    def __init__(self, params, spec: Mapping) -> None:
        super().__init__()
        embedding_dim = int(params.embedding_size)
        hidden_dims = tuple(int(dim) for dim in params.hidden_dims)
        self.encoder = AliccpFeatureEncoder(spec, embedding_dim)
        self.esmm = ESSM(
            input_dim=self.encoder.output_dim,
            hidden_dims=hidden_dims,
            dropout=float(params.dropout),
        )
        self.ctr_loss_weight = float(params.get("ctr_loss_weight", 1.0))
        self.ctcvr_loss_weight = float(params.get("ctcvr_loss_weight", 1.0))
        if self.ctr_loss_weight < 0 or self.ctcvr_loss_weight < 0:
            raise ValueError("ESSM loss weights must be non-negative")
        if self.ctr_loss_weight == 0 and self.ctcvr_loss_weight == 0:
            raise ValueError("At least one ESSM loss weight must be positive")

    def forward(self, features: Mapping[str, Tensor], mode: str = "train"):
        del mode
        dense_inputs = self.encoder(features)
        ctr, cvr, ctcvr = self.esmm.probabilities(dense_inputs)
        return {
            "ctr": ctr.squeeze(-1),
            "cvr": cvr.squeeze(-1),
            "ctcvr": ctcvr.squeeze(-1),
        }

    def loss(self, predictions, labels):
        ctr_loss = F.binary_cross_entropy(
            predictions["ctr"], labels["y"].to(predictions["ctr"].dtype)
        )
        ctcvr_loss = F.binary_cross_entropy(
            predictions["ctcvr"],
            labels["z"].to(predictions["ctcvr"].dtype),
        )
        return (
            self.ctr_loss_weight * ctr_loss
            + self.ctcvr_loss_weight * ctcvr_loss
        )


# The paper and the original project use "ESMM"; keep it as a convenient alias.
ESMM = ESSM


ESSM_DEFAULTS = {
    "model": "essm",
    "hidden_dims": (256, 128),
    "dropout": 0.0,
    "ctr_loss_weight": 1.0,
    "ctcvr_loss_weight": 1.0,
    "extra_fields": 0,
    "find_unused_parameters": False,
}


def parse_hidden_dims(value):
    """Parse positive ESSM tower widths from the command line."""
    try:
        dimensions = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "hidden_dims must be a comma-separated integer list"
        ) from exc
    if not dimensions or any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError(
            "hidden_dims must contain positive integers"
        )
    return dimensions


def add_essm_arguments(parser):
    """Add ESSM-only architecture and loss arguments to the shared parser."""
    parser.add_argument(
        "--hidden_dims",
        type=parse_hidden_dims,
        default=ESSM_DEFAULTS["hidden_dims"],
        help="comma-separated widths of the CTR and CVR towers",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=ESSM_DEFAULTS["dropout"],
        help="dropout used after each hidden activation",
    )
    parser.add_argument(
        "--ctr_loss_weight",
        type=float,
        default=ESSM_DEFAULTS["ctr_loss_weight"],
    )
    parser.add_argument(
        "--ctcvr_loss_weight",
        type=float,
        default=ESSM_DEFAULTS["ctcvr_loss_weight"],
    )
    return parser


def build_essm(params, spec):
    """Build the Ali-CCP ESSM adapter through the framework contract."""
    return AliccpESSM(params, spec)


def configure_essm(params):
    """Validate ESSM parameters before distributed workers initialize models."""
    if not 0.0 <= params.dropout < 1.0:
        raise argparse.ArgumentTypeError("dropout must be in [0, 1)")
    if params.ctr_loss_weight < 0 or params.ctcvr_loss_weight < 0:
        raise argparse.ArgumentTypeError("ESSM loss weights must be non-negative")
    if params.ctr_loss_weight == 0 and params.ctcvr_loss_weight == 0:
        raise argparse.ArgumentTypeError(
            "at least one ESSM loss weight must be positive"
        )
    return params

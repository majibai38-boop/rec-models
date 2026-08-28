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

"""ETA behavior before the LSH and padding fixes.

This module intentionally retains the original ETA candidate-ranking and
padding behavior so results can be compared with the corrected ``eta`` model.
Shared methods that were unchanged by the fix are inherited from ``eta.ETA``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .eta import ETA as CurrentETA
from .eta import ETA_DEFAULTS


class MultiHeadAttention(nn.Module):
    """Original ETA multi-head attention implementation."""

    def __init__(self, params):
        super().__init__()
        self.num_heads = params.num_heads
        self.attention_dim = params.embedding_size[-1]

        self.q_linear = nn.Linear(
            self.attention_dim, self.attention_dim, bias=False
        )
        self.k_linear = nn.Linear(
            self.attention_dim, self.attention_dim, bias=False
        )
        self.v_linear = nn.Linear(
            self.attention_dim, self.attention_dim, bias=False
        )
        self.out_linear = nn.Linear(
            self.attention_dim, self.attention_dim, bias=False
        )

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)

        q = q.view(
            batch_size,
            -1,
            self.num_heads,
            q.shape[-1] // self.num_heads,
        ).transpose(1, 2)
        k = k.view(
            batch_size,
            -1,
            self.num_heads,
            k.shape[-1] // self.num_heads,
        ).transpose(1, 2)
        v = v.view(
            batch_size,
            -1,
            self.num_heads,
            v.shape[-1] // self.num_heads,
        ).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores /= torch.sqrt(
            torch.tensor(query.shape[-1] // self.num_heads)
        )

        paddings = torch.ones_like(scores) * (-(2**32) + 1)
        scores = torch.where(
            torch.tile(
                torch.reshape(mask, [-1, 1, 1, key.shape[1]]),
                [1, self.num_heads, 1, 1],
            ),
            scores,
            paddings,
        )

        attention = F.softmax(scores, dim=-1)
        context = torch.matmul(attention, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.attention_dim
        )
        return self.out_linear(context)


class LoneAttention(nn.Module):
    """Original long-interest attention, including legacy LSH ranking."""

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.attention = MultiHeadAttention(params)
        if self.params.reuse_hash:
            self.reuse_hash = torch.randn(
                self.params.embedding_size[-1],
                self.params.hash_bits,
            ).to(self.params.device)
        self.relu = nn.ReLU()

    def lsh_hash(self, vec, rand_rotate):
        rotated_vec = torch.matmul(vec, rand_rotate)
        return self.relu(torch.sign(rotated_vec))

    def forward(self, target_input, seq_input, mask):
        random_rotations = (
            self.reuse_hash
            if self.params.reuse_hash
            else torch.randn(
                self.params.embedding_dim,
                self.params.hash_bits,
            )
        )

        target_hash = self.lsh_hash(target_input, random_rotations)
        seq_hash = self.lsh_hash(seq_input, random_rotations)
        sim_hash = torch.sum(
            torch.abs(seq_hash - target_hash), dim=-1, keepdim=False
        )

        min_int = -(2**32) + 1
        paddings = torch.zeros_like(sim_hash) + min_int
        sim_hash = torch.where(
            torch.reshape(mask, (-1, sim_hash.shape[-1])),
            sim_hash,
            paddings,
        )
        _, topk_index = torch.topk(
            sim_hash,
            k=self.params.topk,
            dim=-1,
            sorted=True,
        )

        topk_emb = torch.gather(
            seq_input,
            index=topk_index[..., None].expand(
                -1, -1, seq_input.size(-1)
            ),
            dim=1,
        )
        topk_mask = torch.gather(
            mask, index=topk_index.unsqueeze(dim=1), dim=-1
        )
        return self.attention(
            target_input, topk_emb, topk_emb, topk_mask
        )


class ETA(CurrentETA):
    """Original ETA model behavior adapted to the shared model contract."""

    def __init__(self, params, spec):
        nn.Module.__init__(self)
        self.params = params
        self.spec = spec
        self.reuse_hash = None
        self.emb_weights = nn.ModuleDict()

        extra_multis = []
        extra_ones = []
        if params.get("extra_multi_hots", None) is not None:
            for i in range(params.extra_multi_hots):
                extra_multis.append(str(1000 + i) + "_14")
                extra_ones.append(str(1000 + i))

        self.target_field_name = ["206", "207", "216", "210"] + extra_ones
        self.target_multi_field = [
            "109_14",
            "110_14",
            "127_14",
            "150_14",
        ] + extra_multis
        self.embedding_init()

        self.masks = {}
        self.emb_cats = []

        layer_list = []
        for i in range(len(self.params.deep_layers)):
            layer_list.append(
                nn.Linear(
                    in_features=self.deep_layers_num[i],
                    out_features=self.deep_layers_num[i + 1],
                    bias=False,
                )
            )
            layer_list.append(nn.ReLU())

        self.short_attention = MultiHeadAttention(self.params)
        self.long_attention = LoneAttention(self.params)
        self.deep_layers = nn.Sequential(*layer_list)
        self.out_layer = nn.Linear(
            in_features=self.deep_layers_num[-1], out_features=1
        )

    def embedding_preprocess(self, features):
        embeddings = {}
        for key in self.spec["one_hot_fields"]:
            embeddings[key] = self.emb_weights[key](features[key])
            embeddings[key] = torch.reshape(
                embeddings[key], [-1, 1, embeddings[key].shape[-1]]
            )

        for key in self.spec["multi_hot_fields"]:
            feature_dense = features.get(key)
            self.masks[key] = (feature_dense >= 0)[:, None]
            feature_dense = torch.where(
                feature_dense == -1,
                torch.zeros_like(feature_dense),
                feature_dense,
            )
            embeddings[key] = self.emb_weights[key](feature_dense)

        for key in self.spec["special_fields"]:
            feature_sparse = features.get(key)
            sparse_mask = (feature_sparse >= 0).to(torch.float32)[..., None]
            feature_sparse = torch.where(
                feature_sparse == -1,
                torch.zeros_like(feature_sparse),
                feature_sparse,
            )
            sparse_lookup_embedding = (
                self.emb_weights[key](feature_sparse) * sparse_mask
            )
            embeddings[key] = torch.sum(
                sparse_lookup_embedding, dim=1
            )[:, None]
        return embeddings

    def forward(self, features, mode="train"):
        embeddings = self.embedding_preprocess(features)

        self.emb_cats = [
            self.long_emb_padding(field, embeddings, self.masks)
            for field in self.target_multi_field
        ]
        short_attentions = []
        for emb_cat, target_name in zip(
            self.emb_cats, self.target_field_name
        ):
            emb_target = embeddings.get(target_name)
            emb_short = emb_cat[0]
            mask_short = emb_cat[2]
            short_attentions.append(
                self.short_attention(
                    emb_target, emb_short, emb_short, mask_short
                )
            )

        long_attentions = []
        for emb_cat, target_name in zip(
            self.emb_cats, self.target_field_name
        ):
            emb_target = embeddings.get(target_name)
            emb_long = emb_cat[1]
            mask_long = emb_cat[3]
            long_attentions.append(
                self.long_attention(emb_target, emb_long, mask_long)
            )

        embedding = torch.concat(
            [
                embeddings.get(field_name)
                for field_name in self.spec["one_hot_fields"]
            ]
            + [
                embeddings.get(field_name)
                for field_name in self.spec["special_fields"]
            ]
            + short_attentions
            + long_attentions,
            dim=-1,
        )

        x_deep = torch.reshape(embedding, [-1, embedding.shape[-1]])
        x_deep = self.deep_layers(x_deep)
        y_deep = self.out_layer(x_deep)
        return {"ctr": torch.sigmoid(torch.reshape(y_deep, shape=[-1]))}


ETA_LEGACY_DEFAULTS = {
    **ETA_DEFAULTS,
    "model": "eta_legacy",
}


def build_eta_legacy(params, spec):
    """Build the pre-fix ETA variant through the framework contract."""
    return ETA(params, spec)

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

"""Ali-CCP ESMM using the standard TorchRec 1.2 DMP interfaces.

CUDA and NPU share the same ``EmbeddingBagCollection``, sharder, planner and
``DistributedModelParallel`` code. The NPU-adapted TorchRec wheel and
``fbgemm_ascend`` package provide the backend implementation without
introducing a separate Dynamic Embedding model API.
"""

import argparse
from importlib.metadata import version as package_version
import os
from typing import Mapping, Optional

import torch
import torch.distributed as dist
from torch import Tensor, nn
import torch.nn.functional as F

from .esmm import ESMM, ESMM_DEFAULTS, add_esmm_arguments, configure_esmm


_TORCH_NPU_IMPORT_ERROR: Optional[Exception] = None
try:
    import torch_npu  # noqa: F401  # Register the torch.npu device extension.
except Exception as exc:
    _TORCH_NPU_IMPORT_ERROR = exc

_TORCHREC_DMP_IMPORT_ERROR: Optional[Exception] = None
try:
    import torchrec
    import torch.distributed.checkpoint as dcp
    from torch.distributed.optim import _apply_optimizer_in_backward as apply_optimizer_in_backward
    from torchrec.distributed.comm import get_local_size
    from torchrec.distributed.embedding_types import EmbeddingComputeKernel
    from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
    from torchrec.distributed.model_parallel import DistributedModelParallel
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.planner.types import ParameterConstraints
    from torchrec.distributed.train_pipeline import TrainPipelineSparseDist
    from torchrec.distributed.types import ShardingType
    from torchrec.modules.embedding_configs import DataType, EmbeddingBagConfig, PoolingType
    from torchrec.modules.embedding_modules import EmbeddingBagCollection
    from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
    from torchrec.optim.optimizers import in_backward_optimizer_filter
    TORCHREC_DMP_VERSION = getattr(torchrec, "__version__", "") or package_version("torchrec")
    if not str(TORCHREC_DMP_VERSION).startswith("1.2.0"):
        raise RuntimeError(f"esmm_torchrec_dmp requires TorchRec 1.2.0, found {TORCHREC_DMP_VERSION or 'unknown'}")
except Exception as exc:  # TorchRec may fail on an incompatible FBGEMM build.
    dcp = None
    apply_optimizer_in_backward = None
    CombinedOptimizer = None
    DataType = None
    DistributedModelParallel = None
    EmbeddingBagCollection = None
    EmbeddingBagCollectionSharder = None
    EmbeddingBagConfig = None
    EmbeddingComputeKernel = None
    EmbeddingShardingPlanner = None
    get_local_size = None
    KeyedOptimizerWrapper = None
    ParameterConstraints = None
    PoolingType = None
    ShardingType = None
    Topology = None
    TrainPipelineSparseDist = None
    TORCHREC_DMP_VERSION = None
    in_backward_optimizer_filter = None
    _TORCHREC_DMP_IMPORT_ERROR = exc

_FBGEMM_ASCEND_IMPORT_ERROR: Optional[Exception] = None
try:
    # Importing the operator package registers the NPU implementations with PyTorch.
    import fbgemm_ascend  # noqa: F401
except Exception as exc:
    _FBGEMM_ASCEND_IMPORT_ERROR = exc


TORCHREC_DMP_AVAILABLE = _TORCHREC_DMP_IMPORT_ERROR is None


def require_torchrec_dmp(device_type=None):
    """Validate the standard TorchRec stack and optional NPU registrations."""
    if not TORCHREC_DMP_AVAILABLE:
        raise RuntimeError("esmm_torchrec_dmp requires TorchRec 1.2.0 and a compatible PyTorch/FBGEMM installation") from _TORCHREC_DMP_IMPORT_ERROR
    if device_type == "npu" and _TORCH_NPU_IMPORT_ERROR is not None:
        raise RuntimeError("NPU TorchRec DMP requires the PyTorch 2.7.1 compatible torch-npu package") from _TORCH_NPU_IMPORT_ERROR
    if device_type == "npu" and _FBGEMM_ASCEND_IMPORT_ERROR is not None:
        raise RuntimeError("NPU TorchRec DMP requires the compatible fbgemm_ascend 1.2.0 package") from _FBGEMM_ASCEND_IMPORT_ERROR


def _normal_embedding_init(weight):
    return nn.init.normal_(weight, mean=0.0, std=(2 / 512) ** 0.5)


class AliccpTorchRecDmpFeatureEncoder(nn.Module):
    """Encode Ali-CCP fields with a meta-device TorchRec EBC."""

    def __init__(self, spec: Mapping, embedding_dim: int) -> None:
        super().__init__()
        require_torchrec_dmp()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

        self.one_hot_fields = tuple(spec["one_hot_fields"])
        self.sequence_fields = tuple(spec["multi_hot_fields"]) + tuple(spec["special_fields"])
        self.field_names = self.one_hot_fields + self.sequence_fields
        if not self.field_names:
            raise ValueError("Ali-CCP spec must contain at least one feature field")
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("Ali-CCP feature groups contain duplicate field names")

        vocab_lengths = spec["vocab_length"]
        missing_vocab = [field for field in self.field_names if field not in vocab_lengths]
        if missing_vocab:
            raise ValueError(f"Missing vocab_length entries for fields: {missing_vocab}")

        self.table_configs = [EmbeddingBagConfig(name=f"table_{index}", embedding_dim=embedding_dim, num_embeddings=int(vocab_lengths[field]) + 1, feature_names=[field], data_type=DataType.FP32, pooling=PoolingType.SUM, init_fn=_normal_embedding_init) for index, field in enumerate(self.field_names)]
        self.embedding_bags = EmbeddingBagCollection(tables=self.table_configs, is_weighted=False, device=torch.device("meta"))
        self.embedding_dim = embedding_dim
        self.output_dim = len(self.field_names) * embedding_dim

    def forward(self, batch) -> Tensor:
        pooled = self.embedding_bags(batch.sparse_features).values()
        pooled = pooled.reshape(pooled.size(0), len(self.field_names), self.embedding_dim)
        pooled = pooled / batch.normalizers.to(pooled.dtype).unsqueeze(-1)
        return pooled.flatten(start_dim=1)


class AliccpTorchRecDmpESMM(nn.Module):
    """Sharded TorchRec EBC plus the shared dense ESMM towers."""

    def __init__(self, params, spec: Mapping) -> None:
        super().__init__()
        require_torchrec_dmp(str(params.device).split(":", 1)[0])
        embedding_dim = int(params.embedding_size)
        hidden_dims = tuple(int(dim) for dim in params.hidden_dims)
        self.encoder = AliccpTorchRecDmpFeatureEncoder(spec, embedding_dim)
        self.esmm = ESMM(input_dim=self.encoder.output_dim, hidden_dims=hidden_dims, dropout=float(params.dropout))
        self.ctr_loss_weight = float(params.get("ctr_loss_weight", 1.0))
        self.ctcvr_loss_weight = float(params.get("ctcvr_loss_weight", 1.0))
        if self.ctr_loss_weight < 0 or self.ctcvr_loss_weight < 0:
            raise ValueError("ESMM loss weights must be non-negative")
        if self.ctr_loss_weight == 0 and self.ctcvr_loss_weight == 0:
            raise ValueError("At least one ESMM loss weight must be positive")

    def forward(self, batch):
        dense_inputs = self.encoder(batch).to(self.esmm.ctr_tower.network[0].weight.dtype)
        ctr, cvr, ctcvr = self.esmm.probabilities(dense_inputs)
        predictions = {"ctr": ctr.squeeze(-1), "cvr": cvr.squeeze(-1), "ctcvr": ctcvr.squeeze(-1)}
        loss = self.loss(predictions, {"y": batch.y, "z": batch.z})
        predictions["loss"] = loss.detach()
        return loss, predictions

    def loss(self, predictions, labels):
        ctr_loss = F.binary_cross_entropy(predictions["ctr"], labels["y"].to(predictions["ctr"].dtype))
        ctcvr_loss = F.binary_cross_entropy(predictions["ctcvr"], labels["z"].to(predictions["ctcvr"].dtype))
        return self.ctr_loss_weight * ctr_loss + self.ctcvr_loss_weight * ctcvr_loss

    def build_distributed(self, params, device, dense_dtype):
        """Collectively plan EBC shards and build standard TorchRec DMP."""
        require_torchrec_dmp(device.type)
        if not dist.is_initialized():
            raise RuntimeError("TorchRec DMP requires an initialized process group")

        self.esmm.to(device=device, dtype=dense_dtype)
        apply_optimizer_in_backward(torch.optim.Adagrad, self.encoder.embedding_bags.parameters(), {"lr": float(params.learning_rate), "eps": 1e-8})

        strategy = params.torchrec_sharding_strategy
        constraints = {}
        for config in self.encoder.table_configs:
            sharding_types = None
            if strategy == "row_wise":
                sharding_types = [ShardingType.ROW_WISE.value]
            elif strategy == "table_wise":
                sharding_types = [ShardingType.TABLE_WISE.value]
            elif strategy == "hybrid":
                sharding_types = [ShardingType.ROW_WISE.value if config.num_embeddings >= params.torchrec_row_wise_min_rows else ShardingType.TABLE_WISE.value]
            constraints[config.name] = ParameterConstraints(sharding_types=sharding_types, compute_kernels=[EmbeddingComputeKernel.FUSED.value])
        sharder = EmbeddingBagCollectionSharder()
        world_size = dist.get_world_size()
        local_world_size = int(get_local_size())
        if local_world_size <= 0 or world_size % local_world_size != 0:
            raise RuntimeError(f"Invalid TorchRec topology: world_size={world_size}, local_world_size={local_world_size}")
        planner = EmbeddingShardingPlanner(topology=Topology(world_size=world_size, local_world_size=local_world_size, compute_device=device.type), constraints=constraints, batch_size=int(params.batch_size))
        plan = planner.collective_plan(self, [sharder], dist.GroupMember.WORLD)
        model = DistributedModelParallel(module=self, sharders=[sharder], device=device, plan=plan)
        dense_optimizer = KeyedOptimizerWrapper(dict(in_backward_optimizer_filter(model.named_parameters())), lambda parameters: torch.optim.Adam(parameters, lr=float(params.learning_rate)))
        optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])
        return model, optimizer, plan

    @staticmethod
    def build_train_pipeline(model, optimizer, device):
        return TrainPipelineSparseDist(model=model, optimizer=optimizer, device=device, execute_all_batches=True)

    @staticmethod
    def build_eval_pipeline(model, optimizer, device):
        # The TorchRec 1.2 NPU adaptation covers TrainPipelineSparseDist streams,
        # while EvalPipelineSparseDist's background loader lacks NPU device setup.
        # With model.eval(), the train pipeline skips backward and optimizer steps.
        return TrainPipelineSparseDist(model=model, optimizer=optimizer, device=device, execute_all_batches=True)

    @staticmethod
    def _checkpoint_path(saved_dir):
        return os.path.join(saved_dir, "best_val_dmp")

    def load_checkpoint(self, model, saved_dir):
        path = self._checkpoint_path(saved_dir)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"TorchRec DMP checkpoint does not exist: {path}")
        state_dict = {"model": model.state_dict()}
        dcp.load(state_dict=state_dict, checkpoint_id=path)
        model.load_state_dict(state_dict["model"])

    def save_checkpoint(self, model, saved_dir):
        dcp.save(state_dict={"model": model.state_dict()}, checkpoint_id=self._checkpoint_path(saved_dir))


ESMM_TORCHREC_DMP_DEFAULTS = dict(ESMM_DEFAULTS, model="esmm_torchrec_dmp", distributed_strategy="torchrec_dmp", embedding_type="float32", torchrec_sharding_strategy="row_wise", torchrec_row_wise_min_rows=100000, torchrec_kjt_dtype="int32")


def add_esmm_torchrec_dmp_arguments(parser):
    parser = add_esmm_arguments(parser)
    parser.add_argument("--torchrec_sharding_strategy", type=str.lower, default="row_wise", choices=("hybrid", "auto", "row_wise", "table_wise"), help="TorchRec embedding sharding strategy; the NPU-adapted TorchRec 1.2.0 build supports row_wise only")
    parser.add_argument("--torchrec_row_wise_min_rows", type=int, default=100000, help="hybrid mode uses row-wise sharding for tables at or above this row count")
    parser.add_argument("--torchrec_kjt_dtype", type=str.lower, default="int32", choices=("int32", "int64"), help="integer dtype used by KJT values")
    return parser


def configure_esmm_torchrec_dmp(params):
    params = configure_esmm(params)
    if params.distributed_strategy == "auto":
        params.distributed_strategy = "torchrec_dmp"
    if params.distributed_strategy != "torchrec_dmp":
        raise argparse.ArgumentTypeError("esmm_torchrec_dmp requires --distributed_strategy=torchrec_dmp")
    if params.device not in ("cuda", "npu"):
        raise argparse.ArgumentTypeError("esmm_torchrec_dmp supports CUDA or NPU execution")
    if params.embedding_type != "float32" and not (params.embedding_type == "model" and params.model_type == "float32"):
        raise argparse.ArgumentTypeError("TorchRec DMP embeddings are configured as FP32; use --embedding_type=float32")
    incompatible = [name for name in ("compile", "graph", "shape_handle", "check_precision") if params.get(name, False)]
    if incompatible:
        raise argparse.ArgumentTypeError(f"esmm_torchrec_dmp does not support these options yet: {', '.join(incompatible)}")
    if params.mode == "get_layer_result":
        raise argparse.ArgumentTypeError("esmm_torchrec_dmp does not support get_layer_result mode")
    if params.torchrec_sharding_strategy not in ("hybrid", "auto", "row_wise", "table_wise"):
        raise argparse.ArgumentTypeError(f"Unsupported TorchRec sharding strategy: {params.torchrec_sharding_strategy}")
    if str(params.device).split(":", 1)[0] == "npu" and params.torchrec_sharding_strategy != "row_wise":
        raise argparse.ArgumentTypeError("The NPU-adapted TorchRec 1.2.0 FUSED EBC supports only --torchrec_sharding_strategy=row_wise")
    if params.torchrec_kjt_dtype not in ("int32", "int64"):
        raise argparse.ArgumentTypeError(f"Unsupported TorchRec KJT dtype: {params.torchrec_kjt_dtype}")
    if params.torchrec_row_wise_min_rows <= 0:
        raise argparse.ArgumentTypeError("torchrec_row_wise_min_rows must be positive")
    return params


def build_esmm_torchrec_dmp(params, spec):
    """Build the unsharded meta model; the framework invokes DMP collectively."""
    return AliccpTorchRecDmpESMM(params, spec)


__all__ = [
    "AliccpTorchRecDmpESMM",
    "AliccpTorchRecDmpFeatureEncoder",
    "ESMM_TORCHREC_DMP_DEFAULTS",
    "TORCHREC_DMP_AVAILABLE",
    "TORCHREC_DMP_VERSION",
    "add_esmm_torchrec_dmp_arguments",
    "build_esmm_torchrec_dmp",
    "configure_esmm_torchrec_dmp",
    "require_torchrec_dmp",
]

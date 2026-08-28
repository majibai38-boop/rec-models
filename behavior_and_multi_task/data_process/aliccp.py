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

"""Ali-CCP sharded dataset loading for the unified training package."""

import os
import json
import stat
import glob

import torch
from torch.utils.data import DataLoader, IterableDataset

from ..training_framework.handler import TestHandler
from ..training_framework.utils.logger import logger


class AliccpDataset(IterableDataset):
    """Stream pre-batched examples from sharded PyTorch tensor files."""

    def __init__(self, params, filepaths, spec, mode, shuffle=False):
        super().__init__()
        self.params = params
        self.spec = spec
        self.filepaths = list(filepaths)
        self.mode = mode
        self.shuffle = shuffle
        self.batch_size = int(params.batch_size)
        self.device = params.device
        self.seed = int(params.get("seed", 42))
        self._epoch = 0
        self.feature_names = (
            list(spec["one_hot_fields"])
            + list(spec["multi_hot_fields"])
            + list(spec["special_fields"])
        )
        if not self.filepaths:
            raise FileNotFoundError(f"No PyTorch shards were provided for {mode}")

        dataset_size = int(spec["dataset_size"][mode])
        shard_sizes = spec.get("shard_sizes", {}).get(mode)
        if shard_sizes is not None:
            if len(shard_sizes) != len(self.filepaths):
                raise ValueError(
                    f"spec.json declares {len(shard_sizes)} {mode} shards, but "
                    f"{len(self.filepaths)} files were found"
                )
            if sum(shard_sizes) != int(spec["dataset_size"][mode]):
                raise ValueError(f"Inconsistent shard sizes for {mode} in spec.json")

        self.rank = 0
        self.world_size = 1
        if mode == "train" and torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        if self.world_size > 1 and shard_sizes is None:
            raise ValueError(
                "Distributed training requires shard_sizes in spec.json; regenerate "
                "the dataset with data_process/run.sh"
            )

        self.shard_segments = self._build_shard_segments(
            shard_sizes, dataset_size
        )
        samples_for_rank = dataset_size // self.world_size
        self.length = samples_for_rank // self.batch_size
        if self.length == 0:
            raise ValueError(
                f"The {mode} split does not contain a complete batch of "
                f"{self.batch_size} samples per rank"
            )

    def __len__(self):
        return self.length

    def _build_shard_segments(self, shard_sizes, dataset_size):
        if shard_sizes is None:
            return [(path, 0, None) for path in self.filepaths]

        if self.world_size == 1:
            return [
                (path, 0, int(size))
                for path, size in zip(self.filepaths, shard_sizes)
            ]

        samples_per_rank = dataset_size // self.world_size
        rank_start = self.rank * samples_per_rank
        rank_end = rank_start + samples_per_rank
        segments = []
        shard_start = 0
        for path, size in zip(self.filepaths, shard_sizes):
            size = int(size)
            shard_end = shard_start + size
            segment_start = max(rank_start, shard_start)
            segment_end = min(rank_end, shard_end)
            if segment_start < segment_end:
                segments.append(
                    (
                        path,
                        segment_start - shard_start,
                        segment_end - shard_start,
                    )
                )
            shard_start = shard_end
        return segments

    @staticmethod
    def _load_shard(path):
        try:
            return torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            # PyTorch 2.7 supports the safe/mmap path; keep older releases usable.
            return torch.load(path, map_location="cpu")

    def _validate_shard(self, shard, path):
        if shard.get("format_version") != self.spec.get("format_version", 1):
            raise ValueError(f"Unsupported data format version in {path}")
        if set(shard.get("features", {})) != set(self.feature_names):
            raise ValueError(f"Feature schema mismatch in {path}")
        if set(shard.get("labels", {})) != {"y", "z"}:
            raise ValueError(f"Label schema mismatch in {path}")
        sample_count = int(shard.get("num_samples", -1))
        for name, tensor in {**shard["features"], **shard["labels"]}.items():
            if tensor.shape[0] != sample_count:
                raise ValueError(
                    f"{name} has {tensor.shape[0]} rows, expected {sample_count} "
                    f"in {path}"
                )
        return sample_count

    @staticmethod
    def _take_rows(tensor, start, count, order):
        if order is None:
            return tensor[start : start + count]
        return tensor.index_select(0, order[start : start + count])

    def __iter__(self):
        epoch = self._epoch
        self._epoch += 1
        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)

        shard_segments = self.shard_segments
        if self.shuffle:
            file_order = torch.randperm(
                len(shard_segments), generator=generator
            ).tolist()
            shard_segments = [shard_segments[index] for index in file_order]

        feature_parts = {name: [] for name in self.feature_names}
        label_parts = {name: [] for name in ("y", "z")}
        pending_count = 0
        emitted_batches = 0

        for path, segment_start, segment_end in shard_segments:
            shard = self._load_shard(path)
            sample_count = self._validate_shard(shard, path)
            segment_end = sample_count if segment_end is None else segment_end
            if segment_end > sample_count:
                raise ValueError(
                    f"spec.json declares more samples than are stored in {path}"
                )
            segment_size = segment_end - segment_start
            row_order = (
                torch.randperm(segment_size, generator=generator) + segment_start
                if self.shuffle
                else None
            )
            position = 0
            while position < segment_size and emitted_batches < self.length:
                take_count = min(
                    self.batch_size - pending_count, segment_size - position
                )
                row_start = (
                    position if row_order is not None else segment_start + position
                )
                for name in self.feature_names:
                    feature_parts[name].append(
                        self._take_rows(
                            shard["features"][name], row_start, take_count, row_order
                        )
                    )
                for name in ("y", "z"):
                    label_parts[name].append(
                        self._take_rows(
                            shard["labels"][name], row_start, take_count, row_order
                        )
                    )
                position += take_count
                pending_count += take_count

                if pending_count == self.batch_size:
                    features = {
                        name: torch.cat(parts, dim=0).to(
                            self.device, dtype=torch.long, non_blocking=True
                        )
                        for name, parts in feature_parts.items()
                    }
                    labels = {
                        name: torch.cat(parts, dim=0).to(
                            self.device, non_blocking=True
                        )
                        for name, parts in label_parts.items()
                    }
                    yield features, labels
                    feature_parts = {name: [] for name in self.feature_names}
                    label_parts = {name: [] for name in ("y", "z")}
                    pending_count = 0
                    emitted_batches += 1

            if emitted_batches == self.length:
                break

        if emitted_batches != self.length:
            raise RuntimeError(
                f"Loaded {emitted_batches} full {self.mode} batches, but spec.json "
                f"declares {self.length}"
            )


def get_spec(params):
    root_path = os.path.abspath(__file__)
    root_path = os.path.sep.join(root_path.split(os.path.sep)[:-1])
    bundled_spec_path = os.path.join(root_path, "resources", "aliccp_spec.json")
    dataset_spec_path = os.path.join(params.data_dir, "spec.json")
    if params.mode == "test_qps":
        spec_json_path = bundled_spec_path
    elif params.data_dir and os.path.isfile(dataset_spec_path):
        spec_json_path = dataset_spec_path
    else:
        raise FileNotFoundError(
            f"Ali-CCP schema does not exist: {dataset_spec_path}. "
            "Run data_process/run.sh before training."
        )
    local_spec = json_file_load("spec", spec_json_path)
    logger.info(f"spec_json_path: {spec_json_path}")
    if params.mode == "test_qps":
        one_hot_list = []
        multi_hot_list = []
        one_hot_num = len(local_spec["one_hot_fields"])
        multi_hot_num = len(local_spec["multi_hot_fields"])
        one_hots = int(
            params.extra_fields * (one_hot_num / (one_hot_num + multi_hot_num))
        )
        multi_hots = params.extra_fields - one_hots
        params.extra_multi_hots = multi_hots
        logger.info(one_hots)

        # 先同步生成one_hot和multi_hot特征，多余的生成one_hot特征
        for i in range(multi_hots):
            one_hot_list.append(f"{1000+i}")
            multi_hot_list.append(f"{1000+i}_14")
        for i in range(multi_hots, params.extra_fields):
            one_hot_list.append(f"{1000+i}")
        local_spec["multi_hot_fields"].extend(multi_hot_list)
        local_spec["one_hot_fields"].extend(one_hot_list)
        for key in multi_hot_list:
            local_spec["vocab_length"][key] = 10000
            local_spec["train_max_length"][key] = 50
            local_spec["test_max_length"][key] = 50
            local_spec["val_max_length"][key] = 50
        for key in one_hot_list:
            local_spec["vocab_length"][key] = 50
            local_spec["train_max_length"][key] = 1
            local_spec["test_max_length"][key] = 1
            local_spec["val_max_length"][key] = 1
    return local_spec


def json_file_load(json_name: str, json_path: str) -> dict:
    """
    Load a JSON file from the specified path.
    """
    flags = os.O_RDONLY
    modes = stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    try:
        with os.fdopen(os.open(json_path, flags, modes), "r") as fp:
            json_re = json.load(fp)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{json_name} file not found: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error loading {json_name} file: {e}") from e

    return json_re


def generate_dataloader(dataset):
    # The iterable dataset emits complete batches while it streams each shard.
    return DataLoader(dataset, batch_size=None, num_workers=0)


def _find_shards(data_dir, mode, spec):
    candidates = sorted(
        glob.glob(os.path.join(data_dir, mode, f"data_{mode}.pt.*"))
    )
    shard_files = [path for path in candidates if path.rsplit(".", 1)[-1].isdigit()]
    expected_parts = spec.get("parts", {}).get(mode)
    if expected_parts is not None and len(shard_files) != int(expected_parts):
        raise FileNotFoundError(
            f"Expected {expected_parts} PyTorch shards for {mode}, found "
            f"{len(shard_files)} under {os.path.join(data_dir, mode)}"
        )
    if not shard_files:
        raise FileNotFoundError(
            f"No data_{mode}.pt.* files found under {os.path.join(data_dir, mode)}. "
            "Run data_process/run.sh to generate the PyTorch dataset."
        )
    return shard_files


def load_data(params):
    spec = get_spec(params)
    if spec.get("data_format") != "pytorch":
        raise ValueError(
            "The dataset is not in the PyTorch shard format. Re-run "
            "data_process/run.sh and point --data_dir at its aliccp_out directory."
        )
    tr_files = _find_shards(params.data_dir, "train", spec)
    va_files = _find_shards(params.data_dir, "val", spec)
    te_files = _find_shards(params.data_dir, "test", spec)

    train_dataset = AliccpDataset(
        params,
        tr_files,
        spec=spec,
        mode="train",
        shuffle=True,
    )
    test_dataset = AliccpDataset(
        params,
        te_files,
        spec=spec,
        mode="test",
    )
    val_dataset = AliccpDataset(
        params,
        va_files,
        spec=spec,
        mode="val",
    )

    train_loader = generate_dataloader(train_dataset)
    test_loader = generate_dataloader(test_dataset)
    val_loader = generate_dataloader(val_dataset)

    return train_loader, test_loader, val_loader


def load_generate_data(params):
    spec_json_path = os.path.join(params.data_dir, "spec.json")
    local_spec = json_file_load("spec", spec_json_path)
    return local_spec


class TestAliccpHandler(TestHandler):
    def __init__(self, params, spec):
        super().__init__(params)
        self.spec = spec

    def generate_data(self, batch_size):

        features = {}
        device = self.params.device
        for key in self.spec["one_hot_fields"]:
            features[key] = torch.randint(
                low=0, high=self.spec["vocab_length"][key], size=(batch_size, 1)
            )[:, 0].to(device)

        for key in self.spec["multi_hot_fields"]:
            features[key] = torch.randint(
                low=0, high=self.spec["vocab_length"][key], size=(batch_size, 50)
            ).to(device)

        for key in self.spec["special_fields"]:
            features[key] = torch.randint(
                low=0, high=self.spec["vocab_length"][key], size=(batch_size, 38)
            ).to(device)
        return features

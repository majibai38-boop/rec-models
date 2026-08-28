"""Generate the shared Ali-CCP schema for the sharded PyTorch dataset."""

import argparse
import glob
import json
import os
from itertools import repeat, takewhile
from pathlib import Path

import torch

from path_validator import validate_read_file, validate_save_path


FORMAT_VERSION = 1
FIELDS = (
    "101", "109_14", "110_14", "127_14", "150_14", "121", "122",
    "124", "125", "126", "127", "128", "129", "205", "206", "207",
    "210", "216", "508", "509", "702", "853", "301",
)
MULTI_HOT_FIELDS = ("109_14", "110_14", "127_14", "150_14")
SPECIAL_FIELDS = ("210", "853")
ONE_HOT_FIELDS = tuple(
    field
    for field in FIELDS
    if field not in MULTI_HOT_FIELDS and field not in SPECIAL_FIELDS
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default="./aliccp_out")
    # run.sh passes the common preprocessing arguments to every step.
    args, _ = parser.parse_known_args()
    return args


def iter_count(file_name):
    validate_read_file(file_name)
    buffer_size = 1024 * 1024
    with open(file_name, "r", encoding="utf-8") as source:
        buffers = takewhile(
            lambda value: value,
            (source.read(buffer_size) for _ in repeat(None)),
        )
        return sum(buffer.count("\n") for buffer in buffers)


def load_shard(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        # mmap/weights_only are unavailable in older PyTorch versions.
        return torch.load(path, map_location="cpu")


def inspect_shards(output_dir):
    shard_paths = sorted(glob.glob(os.path.join(output_dir, "*", "data_*.pt.*")))
    if not shard_paths:
        raise FileNotFoundError(f"No PyTorch data shards found under {output_dir}")

    indices_by_part = {}
    sizes_by_part = {}
    for shard_path in shard_paths:
        filename = os.path.basename(shard_path)
        part = Path(shard_path).parent.name
        try:
            shard_index = int(filename.rsplit(".", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid PyTorch shard name: {shard_path}") from error

        shard = load_shard(shard_path)
        if shard.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Unsupported format version in {shard_path}: "
                f"{shard.get('format_version')}"
            )
        sample_count = int(shard["num_samples"])
        if set(shard["features"]) != set(FIELDS):
            raise ValueError(f"Feature schema mismatch in {shard_path}")
        if set(shard["labels"]) != {"y", "z"}:
            raise ValueError(f"Label schema mismatch in {shard_path}")
        for name, tensor in {**shard["features"], **shard["labels"]}.items():
            if tensor.shape[0] != sample_count:
                raise ValueError(
                    f"{name} has {tensor.shape[0]} rows but {sample_count} were "
                    f"declared in {shard_path}"
                )

        indices_by_part.setdefault(part, []).append(shard_index)
        sizes_by_part.setdefault(part, {})[shard_index] = sample_count

    parts = {}
    shard_sizes = {}
    for part, indices in indices_by_part.items():
        indices.sort()
        expected = list(range(len(indices)))
        if indices != expected:
            raise ValueError(
                f"Shard indices for {part} must be contiguous: expected {expected}, "
                f"got {indices}"
            )
        parts[part] = len(indices)
        shard_sizes[part] = [sizes_by_part[part][index] for index in indices]
    return parts, shard_sizes


def main():
    args = parse_args()
    parts, shard_sizes = inspect_shards(args.output_dir)
    required_parts = {"train", "val", "test"}
    if set(parts) != required_parts:
        raise ValueError(
            f"Expected dataset parts {sorted(required_parts)}, got {sorted(parts)}"
        )

    spec = {
        "data_format": "pytorch",
        "format_version": FORMAT_VERSION,
        "one_hot_fields": list(ONE_HOT_FIELDS),
        "multi_hot_fields": list(MULTI_HOT_FIELDS),
        "special_fields": list(SPECIAL_FIELDS),
        "vocab_length": {},
        "parts": parts,
        "shard_sizes": shard_sizes,
        "dataset_size": {
            part: sum(sizes) for part, sizes in shard_sizes.items()
        },
    }

    vocab_pattern = os.path.join(args.output_dir, "vocab", "vocab_*")
    for vocab_file in sorted(glob.glob(vocab_pattern)):
        key = os.path.basename(vocab_file).split("_", 1)[1]
        spec["vocab_length"][key] = iter_count(vocab_file)

    missing_vocabs = set(FIELDS) - set(spec["vocab_length"])
    if missing_vocabs:
        raise ValueError(f"Missing vocabularies for fields: {sorted(missing_vocabs)}")

    for part in ("train", "val", "test"):
        max_length_path = os.path.join(args.input_dir, f"data_{part}_max_length.json")
        validate_read_file(max_length_path)
        with open(max_length_path, "r", encoding="utf-8") as max_length_file:
            spec[f"{part}_max_length"] = json.load(max_length_file)

    spec_path = os.path.join(args.output_dir, "spec.json")
    validate_save_path(spec_path)
    with open(spec_path, "w", encoding="utf-8") as spec_file:
        json.dump(spec, spec_file, indent=2)
        spec_file.write("\n")
    print(f"Wrote PyTorch dataset specification to {spec_path}")


if __name__ == "__main__":
    main()

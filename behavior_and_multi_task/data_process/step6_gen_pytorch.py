"""Convert merged Ali-CCP CSV files to shared sharded PyTorch tensors."""

import argparse
import glob
import json
import os
from multiprocessing import Pool
from pathlib import Path

import torch

from path_validator import validate_read_file


FORMAT_VERSION = 1
FIELDS = (
    "101", "109_14", "110_14", "127_14", "150_14", "121", "122",
    "124", "125", "126", "127", "128", "129", "205", "206", "207",
    "210", "216", "508", "509", "702", "853", "301",
)
MULTI_HOT_FIELDS = {"109_14", "110_14", "127_14", "150_14"}
SPECIAL_FIELDS = {"210", "853"}
SEQUENCE_FIELDS = MULTI_HOT_FIELDS | SPECIAL_FIELDS


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    # Kept for CLI compatibility with the earlier preprocessing steps. Sequence
    # truncation has already happened before this conversion step.
    parser.add_argument("--length", type=float, default=-1)
    parser.add_argument("--proc", type=int, default=1, help="number of workers")
    parser.add_argument(
        "--padding",
        type=parse_bool,
        nargs="?",
        const=True,
        default=True,
        help="pad sequence fields to their split-specific maximum length",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=int,
        default=128,
        help="maximum input CSV bytes handled by one output shard",
    )
    parser.add_argument("--output-dir", default="./aliccp_out")
    args = parser.parse_args()
    if args.proc < 1:
        parser.error("--proc must be at least 1")
    if args.chunk_size_mb < 1:
        parser.error("--chunk-size-mb must be at least 1")
    if not args.padding:
        parser.error(
            "ETA requires fixed-length tensors; run with --padding=True "
            "(the default)"
        )
    return args


def chunkify_file(input_path, output_path, chunk_size_mb):
    """Split a CSV on line boundaries without copying the source file."""
    validate_read_file(input_path)
    file_size = os.path.getsize(input_path)
    chunk_size = chunk_size_mb * 1024 * 1024
    tasks = []
    with open(input_path, "rb") as source:
        chunk_start = 0
        shard_index = 0
        while chunk_start < file_size:
            source.seek(min(chunk_start + chunk_size, file_size))
            if source.tell() < file_size:
                source.readline()
            chunk_end = source.tell()
            tasks.append(
                (
                    chunk_start,
                    chunk_end - chunk_start,
                    input_path,
                    output_path,
                    input_path.replace(".csv", "_max_length.json"),
                    shard_index,
                )
            )
            chunk_start = chunk_end
            shard_index += 1
    return tasks


def _parse_sequence(raw_value, field, target_length, context):
    values = [int(value) for value in raw_value.split("#")]
    if len(values) > target_length:
        raise ValueError(
            f"{field} has {len(values)} values, greater than its declared maximum "
            f"{target_length} ({context})"
        )
    return values + [-1] * (target_length - len(values))


def generate_shard(task):
    (
        chunk_start,
        chunk_size,
        input_path,
        output_path,
        max_length_path,
        shard_index,
    ) = task
    torch.set_num_threads(1)
    validate_read_file(input_path)
    validate_read_file(max_length_path)
    with open(max_length_path, "r", encoding="utf-8") as max_length_file:
        max_lengths = json.load(max_length_file)

    with open(input_path, "rb") as source:
        source.seek(chunk_start)
        raw_chunk = source.read(chunk_size)
    if raw_chunk and raw_chunk[-1] != ord("\n"):
        raise ValueError(
            f"CSV chunk is not line aligned: {input_path}, offset={chunk_start}, "
            f"size={chunk_size}"
        )

    feature_rows = {field: [] for field in FIELDS}
    labels = {"y": [], "z": []}
    lines = raw_chunk.decode("utf-8").splitlines()
    for row_offset, line in enumerate(lines):
        cells = line.strip().split(",")
        if len(cells) != len(FIELDS) + 3:
            raise ValueError(
                f"Expected {len(FIELDS) + 3} columns, got {len(cells)} in "
                f"{input_path} chunk {shard_index}, row {row_offset}"
            )
        _, y_value, z_value, *field_values = cells
        labels["y"].append(float(y_value))
        labels["z"].append(float(z_value))
        for field, raw_value in zip(FIELDS, field_values):
            context = f"{input_path} chunk {shard_index}, row {row_offset}"
            if field in SEQUENCE_FIELDS:
                value = _parse_sequence(
                    raw_value, field, int(max_lengths[field]), context
                )
            else:
                if "#" in raw_value:
                    raise ValueError(f"One-hot field {field} is multi-valued ({context})")
                value = int(raw_value)
            feature_rows[field].append(value)

    features = {
        field: torch.tensor(values, dtype=torch.int32)
        for field, values in feature_rows.items()
    }
    label_tensors = {
        name: torch.tensor(values, dtype=torch.float32)
        for name, values in labels.items()
    }
    payload = {
        "format_version": FORMAT_VERSION,
        "num_samples": len(lines),
        "features": features,
        "labels": label_tensors,
    }

    shard_path = Path(f"{output_path}.{shard_index:05d}")
    temporary_path = shard_path.with_name(f"{shard_path.name}.tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, shard_path)
    return str(shard_path), len(lines)


def main():
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for input_path in sorted(glob.glob("./data_*.csv")):
        part = Path(input_path).stem.split("_", 1)[1]
        output_folder = output_root / part
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"data_{part}.pt"
        for existing_shard in output_folder.glob(f"data_{part}.pt.*"):
            if existing_shard.name.rsplit(".", 1)[-1].isdigit():
                existing_shard.unlink()
        tasks.extend(
            chunkify_file(input_path, str(output_path), args.chunk_size_mb)
        )

    if not tasks:
        raise FileNotFoundError("No data_*.csv files were found in the current directory")

    with Pool(processes=args.proc) as pool:
        results = list(pool.imap_unordered(generate_shard, tasks))
    total_samples = sum(sample_count for _, sample_count in results)
    print(f"Generated {len(results)} PyTorch shards containing {total_samples} samples")


if __name__ == "__main__":
    main()

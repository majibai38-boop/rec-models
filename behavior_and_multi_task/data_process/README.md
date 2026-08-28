# Ali-CCP preprocessing for ETA, DIEN, ESSM, and AutoInt

These scripts create the single `aliccp_out` dataset consumed by all four models
in `behavior_and_multi_task`.

The pipeline:

1. removes invalid samples (`y == 0 && z == 1`);
2. removes feature IDs that occur only once in the training split;
3. remaps every field to a compact ID range (`0` is OOV; `-1` is sequence padding);
4. splits the original test data equally into validation and test data;
5. joins `common_feature` and `sample_skeleton` into merged CSV files;
6. converts the CSV files to sharded PyTorch `.pt` tensor dictionaries;
7. writes `spec.json`, including the schema, split sizes, and shard sizes.

TensorFlow and TFRecord are not used.

## Input files

Put the four downloaded Ali-CCP files in this directory:

```text
common_features_test.csv
common_features_train.csv
sample_skeleton_test.csv
sample_skeleton_train.csv
```

The preprocessing environment needs Python, NumPy, and PyTorch. Run:

```bash
bash run.sh
```

The main settings in `run.sh` are:

- `MAX_LENGTH`: maximum retained length of long sequence fields;
- `NUM_OF_PROC`: preprocessing worker count;
- `CHUNK_SIZE_MB`: maximum merged CSV size represented by one `.pt` shard;
- `PADDING`: keep `true`, because ETA consumes dense, fixed-length tensors.

## Output format

The output is written below `./aliccp_out`:

```text
aliccp_out/
|-- spec.json
|-- vocab/
|-- train/data_train.pt.00000
|-- val/data_val.pt.00000
`-- test/data_test.pt.00000
```

There may be multiple numbered files in each split. Each shard is created with
`torch.save` and contains:

```python
{
    "format_version": 1,
    "num_samples": int,
    "features": {field_name: torch.int32 tensor},
    "labels": {"y": torch.float32 tensor, "z": torch.float32 tensor},
}
```

Feature IDs are stored as `int32` to reduce disk usage and converted to
`torch.long` per batch before model execution. The training loader memory-maps
and streams one shard at a time. It shuffles the
training shard order and rows, joins rows across shard boundaries into complete
batches, and drops only the final incomplete batch of a split. During DDP
training, every rank receives an equal, non-overlapping sample range.

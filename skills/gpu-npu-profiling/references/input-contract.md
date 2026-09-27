# Inputs and safe scope selection

Required: a GPU PyTorch Chrome trace (`traceEvents` object or event list), an Ascend NPU trace, and the corresponding `kernel_details.csv`. Work offline. Do not silently substitute `op_statistic.csv` for kernel details. CUDA/Ascend capture versions can differ; unsupported/missing fields remain warnings or errors, not guessed mappings.

`inspect` returns `profiles`, CPU PID candidates with their step intervals, timestamp-window kernel candidates, device-event PIDs, metadata-presence counts, flow/correlation clues, selected NPU detail counts and structured warnings. Device PIDs are not rank IDs. Multiple device-event PIDs require scope review; the engine does not yet expose independent device filtering.

Metadata presence is not metadata correctness. Missing duration in kernel details may be recovered from a matched trace event; a missing detail file or empty detail scope is not a valid new comparison. Missing shape/dtype/flow can reduce confidence and should be reported. A zero trace-kernel inventory on NPU may be a format/capture issue or recoverable from details; examine warnings before claiming zero work.

Complete (`X`) trace events require finite timestamps and non-negative finite durations. Supplied invalid detail timings are errors, not missing values or zero work. Discarded non-complete records are reported because missing flow records can affect ownership.

New `compare` requires each side's step, or explicit `--all-steps`. When multiple CPU PIDs have operators, supply `--gpu-pid` / `--npu-pid`. Available steps do not determine which is stable or equivalent: ask about warmup, input shapes, batch size, rank, precision/autocast and capture scope when material information is missing. These choices cannot be inferred from equal step numbers.

Exit codes: 0 = requested operation completed (coverage gaps may remain), 2 = invalid input/command, 3 = scope selection required, 4 = result validation failed. Read the JSON status and warnings as well as the code.

New runs record tool/rules/source versions, input SHA-256 and size/mtime, run parameters, output hashes, input inspection and validation. This enables reproducibility; it does not prove capture equivalence. Old result directories can be analyzed without original traces but have a provenance warning. Do not rerun merely to remove that warning when the user only asked to read results.

The historical command without a subcommand keeps automatic scope selection for compatibility and emits warnings. The skill must use `inspect` and the stricter `compare` workflow, not that compatibility path.

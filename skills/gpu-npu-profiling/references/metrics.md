# Metrics and non-additive views

- Direct: NPU kernels with independent direct pairs. Comparable: NPU work eligible for the established comparison granularity. Accounted: work whose location/purpose is accounted for, not necessarily independently comparable. Each has count and NPU-time coverage; do not substitute one for another.
- Ordinary type rows: Comparable Ratio = assigned GPU device time / comparable NPU time. Total Time Ratio = the same assigned GPU time / all NPU time of that type. Low coverage means the latter is not a complete same-type backend comparison.
- TOTAL: all selected GPU compute kernels plus memory activities, each once, divided by NPU **kernel_details-backed** kernel time. NPU trace-only tasks are excluded and counted in validation warnings. Thus TOTAL is a scoped duration comparison, not necessarily an equivalent-work performance ratio.
- TOTAL Comparable Ratio uses only the unique comparable GPU subset and its corresponding NPU subset. Do not divide TOTAL's full GPU time by Comparable NPU Time. Exact numerator/denominator are in `gpu_accounting`.
- GPU_UNMAPPED_KERNELS includes kernels not assigned to an NPU type, even if already compared in an alternative whole-call view. GPU_UNMAPPED_MEMORY is a separate namespace. Their NPU zero means no type allocation, not zero required NPU work; ratios and coverage are not applicable.
- CPU bundles, FlashAttention fused calls and precision paths overlap the type table. Never add them to TOTAL. TOTAL itself is a non-additive summary row, not another workload.
- Duration sums include overlapping streams. They are not elapsed wall time. Busy-time unions and spans are distinct metrics.
- Ratio >=2.0 is dark green, >=1.0 and <2.0 light green, <1.0 yellow in Excel. Color is not evidence that workloads or hardware resources are equivalent.

Use the engine's `validate` result: event IDs/namespace uniqueness, mapped+unmapped partition, counts/times, ratio denominators, diagnostic conservation, and output hashes. Cross-type reuse makes type rows non-additive; the engine warns and TOTAL still deduplicates. A successful audit proves those accounting checks, not semantic correctness of every mapping.

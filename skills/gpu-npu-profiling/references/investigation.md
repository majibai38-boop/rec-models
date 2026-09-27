# Evidence-led investigation

Start with the general report, not a Cast whitelist. Selection is Comparable count OR time coverage strictly below the configured threshold, per Device/Type/Core. A type outside the threshold can still have gaps. Unknown patterns remain unknown.

Query only the relevant evidence; each returned collection has `total`, `offset`, `limit`, and `has_more`. Filters are exact and combine with AND. Start with `--limit 5`; page as needed. Use `--device-id` and `--core-type` to disambiguate a type.

```text
query --result-dir <results> --op-type Cast --limit 5
query --result-dir <results> --npu-kernel-index 42
query --result-dir <results> --gpu-kernel-index 17
query --result-dir <results> --memory-activity-index 3
query --result-dir <results> --side npu --cpu-op-index 99
```

Queries return mappings, selected low-coverage diagnoses, overlapping whole-call references, and independently queried GPU remainder events. No diagnosis may mean the type was not selected for triage. No mapping may mean the GPU event is in the remainder. Read the relevant collection before declaring evidence absent. Older results without CPU index columns cannot support a reliable GPU CPU-index query; use kernel IDs or explain the version limitation.

`summary.overall_statistics` and `query.type_statistics` contain typed counts, times, both ratios and split coverage. `query.reason_groups` aggregates low-coverage reasons, evidence sample IDs and next checks. These two query collections always describe the **entire matching Device/Type/Core group**, even for a single-kernel query; they are context, not the filtered kernel subtotal. CSV-backed `mappings`, `whole_calls` and `gpu_unmapped` retain original column names/strings; `"False"` is false, not a truthy flag. NPU type/device selectors exclude GPU remainder events with no such assignment; query those by GPU ID or GPU CPU owner separately.

Investigate in evidence order:

1. Check input/capture scope and association quality before interpreting a backend difference.
2. Missing/weak NPU owner: inspect Task/Stream, flow and original owner; do not use adjacency alone.
3. CPU mismatch: inspect name normalization, module/parents, overload and complete input metadata. Same counts are insufficient.
4. GPU child work, memory-only work or missing work: inspect runtime correlation and descendant scope. A view, fusion or capture gap are hypotheses, not established causes.
5. Auxiliary dtype/layout work: examine the complete call and input/output metadata. Mixed data+index dtypes are not automatically a Cast; ordered metadata does not prove tensor dataflow.
6. Shared/fused work: prefer a verified complete logical call. Do not give multiple NPU types the same GPU duration.

Div/Sub evidence lives in `coverage_diagnostics.json -> specialized_evidence.precision_paths` and the Markdown report. `Precision Path ID` identifies a complete call. Native paths are controls, not proof about internal opmath; observed conversion cost is not predicted savings.

For each finding report: observation with IDs, established facts, plausible cause, missing evidence and a specific next check. Source inspection or a same-input minimal experiment can verify a cause, but do not change code, dtype, captures or mapping rules unless the user asked for implementation.

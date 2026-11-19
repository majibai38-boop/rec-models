# Type tables and call investigation

Use the skill's `scripts/run_compare.py` entrypoint for the commands below. Select one completed result directory. Check `summary` and `validate`. Record both Step/PID selections, precision, distributed strategy, ranks and device-time population from available evidence. Identify unknowns in the caliber note.

1. Run `call-evidence --result-dir RESULTS` and page through types in descending order of uncovered NPU time. For a requested type, use its exact name, device and core. Investigate groups with `--op-type TYPE`, instances with `--group-id ID`, and source events with `--call-id ID`. All three views support `--offset N --limit N`. Check the actual outer CPU pair, module, input shapes/dtypes, direct owners, Step, existing assignments and competing calls. Instances in a group can have different shapes and verification outcomes.
2. Use `query --result-dir RESULTS --npu-kernel-index N` to inspect alternative owners and general diagnoses. Use `call-evidence --result-dir RESULTS --unmapped-gpu` to page through the GPU remainder or `--gpu-name NAME` for an exact filter. GPU kernel and memory IDs have separate namespaces. If no anchor links the two sides, report what is missing and the affected time.
3. The default table credits eligible single-type scopes and shows verified multi-type scopes as whole calls. Investigate overlaps individually. When the trace evidence supports a choice, write a proposal using a returned group or call ID. Explain the observed path, not a confidence score. A group proposal expands across all its calls; the engine checks each instance. Choose `type` for exclusive single-type attribution, `call` for a complete, possibly multi-type operation, or `review` for unresolved work. To choose one overlapping view, request that view alone and cite the anchor in the reason. Mutually conflicting selections stay pending.

Example proposal (replace identifiers with values returned for this run):

```json
{
  "run_id": "<run_id>",
  "proposals": [
    {"evidence_id": "<group_id>", "level": "call", "reason": "The aligned outer call owns both Transpose and MatMulV3 on NPU and one fused GPU kernel."}
  ]
}
```

4. Run `tables --result-dir RESULTS --output-dir NEW_DIR --caliber-note NOTE`, adding `--proposals PLAN.json` when needed. Read the reported decisions and link the XLSX and Markdown. Cite a call ID for a difficult group and distinguish the original and revised comparable time coverage. The JSON records every call decision, event IDs and supplied proposals. Replay by passing that JSON as `--proposals` for the same result and a fresh output directory. Old reports may lack per-call Steps, call paths or child candidates; report those evidence gaps.

For nested `index_add_`, compare the matched outer calls even if the NPU leaf has a different signature. For a shared GPU kernel across multiple NPU types, explain the parent-level ratio and leave the component time unallocated. The workbook TOTAL includes the unmapped GPU remainder exactly once. A sum of device durations over streams measures cumulative work.

"""Read-only input inventory and scope checks; no inferred cross-device pairing."""
from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path

from .parser import (_load_trace_events, _device_process_ids, _is_complete_event,
                     _is_kernel, _is_step_event, _step_matches, _event_in_window,
                     _normalized_id, get_arg, load_kernel_details)
from .version import __version__
from .coverage_diagnostics import validate_threshold
from .matcher import load_op_map


def inspect_inputs(config, *, retain_data=False):
    report = {"schema_version": 1, "tool_version": __version__, "status": "ready",
              "profiles": {}, "kernel_details": {}, "errors": [], "selections_required": [], "warnings": []}
    loaded = {}

    def issue(bucket, code, side, message):
        report[bucket].append({"code": code, "side": side, "message": message})

    try:
        validate_threshold(config.diagnostic_coverage_threshold)
        if not math.isfinite(config.flow_tolerance_us) or config.flow_tolerance_us < 0:
            raise ValueError("flow tolerance must be finite and non-negative")
        load_op_map(config.op_map)
    except (OSError, ValueError, TypeError) as exc:
        issue("errors", "invalid_configuration", "both", str(exc))

    for side in ("gpu", "npu"):
        path = Path(getattr(config, side + "_trace"))
        try:
            events, raw_count = _load_trace_events(path, strict=True)
            if retain_data:
                loaded[side] = (events, raw_count)
            by_pid, steps = defaultdict(list), defaultdict(list)
            for event in events:
                if event.phase.upper() == "X" and (not math.isfinite(event.duration_us) or
                       not math.isfinite(event.timestamp_us) or event.duration_us < 0):
                    raise ValueError("non-finite timestamp/duration or negative duration")
                if _is_complete_event(event) and event.name.lower().startswith(config.op_prefix.lower()):
                    by_pid[event.pid].append(event)
                if _is_complete_event(event) and _is_step_event(event):
                    steps[event.pid].append(event)
            requested_pid = _normalized_id(getattr(config, side + "_pid"))
            requested_step = getattr(config, side + "_step")
            if not by_pid:
                raise ValueError(f"no CPU operators with prefix {config.op_prefix!r}")
            if requested_pid and requested_pid not in by_pid:
                raise ValueError(f"requested PID {requested_pid} has no selected CPU operators")
            selected_pid = requested_pid or max(by_pid, key=lambda p: len(by_pid[p]))
            if not requested_pid and len(by_pid) > 1:
                issue("selections_required", "ambiguous_pid", side, f"Specify --{side}-pid; several CPU PIDs contain operators.")
            window = None
            if requested_step:
                candidates = [e for e in steps[selected_pid] if _step_matches(e.name, requested_step)]
                if len(candidates) != 1:
                    raise ValueError(f"step {requested_step!r} must identify exactly one interval; found {len(candidates)}")
                window = (candidates[0].timestamp_us, candidates[0].end_us)
            elif not getattr(config, "all_steps", False):
                # Even one marked step may leave warmup/unmarked events outside it.
                issue("selections_required", "step_scope_unspecified", side,
                      f"Specify --{side}-step or explicitly choose --all-steps (also required when no step markers exist).")
            selected_ops = [e for e in by_pid[selected_pid] if _event_in_window(e, window)]
            if not selected_ops:
                raise ValueError("the selected CPU scope is empty")
            device_pids = _device_process_ids(events)
            kernels = [e for e in events if _is_complete_event(e) and _is_kernel(e, config.include_communication, device_pids)
                       and _event_in_window(e, window)]
            memory = [e for e in events if _is_complete_event(e) and e.category.lower() in {"gpu_memcpy", "gpu_memset"}
                      and _event_in_window(e, window)]
            shapes = sum(get_arg(e.args, "Input Dims", "Input Shapes", "input_shapes") not in (None, "", "N/A") for e in selected_ops)
            dtypes = sum(get_arg(e.args, "Input type", "Input Types", "Input Data Types") not in (None, "", "N/A") for e in selected_ops)
            info = {
                "path": str(path.resolve()), "trace_event_count": raw_count,
                "discarded_trace_event_count": raw_count - len(events),
                "cpu_candidates": [{"pid": p, "op_count": len(ops),
                                    "steps": [{"name": e.name, "start_us": e.timestamp_us, "duration_us": e.duration_us}
                                              for e in steps[p]]} for p, ops in sorted(by_pid.items())],
                "selected_scope": {"pid": selected_pid, "step": requested_step or "all",
                                   "pid_was_automatic": not bool(requested_pid)},
                "scope_op_count": len(selected_ops), "shape_present_count": shapes, "dtype_present_count": dtypes,
                "window_kernel_candidate_count": len(kernels), "window_memory_candidate_count": len(memory),
                "device_event_pids": sorted({e.pid for e in kernels + memory}),
                "external_id_present_count": sum(get_arg(e.args, "External id", "external_id") is not None for e in kernels),
                "correlation_present_count": sum(get_arg(e.args, "correlation", "correlation id") is not None for e in kernels),
                "flow_event_count": sum(e.phase in {"s", "t", "f"} for e in events),
                "inventory_note": "Timestamp-window candidates before ownership linking; not final kernel counts. Metadata presence does not prove completeness.",
            }
            report["profiles"][side] = info
            if raw_count != len(events):
                issue("warnings", "discarded_trace_events", side,
                      "Some non-complete records could not be parsed; verify capture format and flow completeness.")
            if shapes < len(selected_ops) or dtypes < len(selected_ops):
                issue("warnings", "partial_cpu_metadata", side, "Some CPU operators lack shape/dtype fields; do not infer equivalence from counts.")
            if not kernels:
                issue("warnings", "no_trace_kernel_candidates", side, "No device kernel candidates in the timestamp window; inspect capture/format and NPU detail recovery.")
            if len(info["device_event_pids"]) > 1:
                issue("warnings", "multiple_device_pids", side, "Device PIDs are not rank/device pairing; verify capture scope. CPU PID selection is not a device filter.")
        except (OSError, ValueError, TypeError) as exc:
            issue("errors", "invalid_trace", side, str(exc))

    details = []
    try:
        details = load_kernel_details(config.npu_kernel_details, config.npu_step)
        if not details:
            issue("errors" if getattr(config, "require_explicit_scope", False) else "warnings",
                  "empty_kernel_details", "npu", "No kernel detail rows for the selected scope; NPU-type comparison is unavailable.")
        if any(not d.name or not d.op_type or (d.duration_us is not None and
               (not math.isfinite(d.duration_us) or d.duration_us < 0)) for d in details):
            raise ValueError("kernel_details requires Name and Type; provided durations must be finite and non-negative")
        for detail in details:
            for field, aliases in (("start", ("Start Time(us)", "Start Time(μs)")),
                                   ("duration", ("Duration(us)", "Duration(μs)"))):
                raw = get_arg(detail.raw, *aliases)
                if raw is None or str(raw).strip() in {"", "N/A"}:
                    continue
                number = float(raw)
                if not math.isfinite(number) or (field == "duration" and number < 0):
                    raise ValueError(f"kernel_details row {detail.row_index}: invalid {field}")
        if any(d.duration_us is None for d in details):
            issue("warnings", "detail_duration_missing", "npu", "Some detail durations are unavailable; only recovered trace timing is usable.")
        report["kernel_details"] = {
            "path": str(Path(config.npu_kernel_details).resolve()), "selected_row_count": len(details),
            "device_counts": dict(Counter(d.device_id or "unknown" for d in details)),
            "step_counts": dict(Counter(d.step_id or "unknown" for d in details)),
            "shape_present_count": sum(bool(d.input_shapes) for d in details),
            "dtype_present_count": sum(bool(d.input_dtypes) for d in details),
        }
        if config.npu_step and any(not d.step_id for d in details):
            issue("warnings", "detail_step_unavailable", "npu", "Some detail rows have no Step ID; verify that the CSV belongs to the selected capture/step.")
    except (OSError, ValueError, TypeError) as exc:
        issue("errors", "invalid_kernel_details", "npu", str(exc))
    if getattr(config, "all_steps", False) and (config.gpu_step or config.npu_step):
        issue("errors", "conflicting_step_options", "both", "--all-steps cannot be combined with individual step selectors.")
    if bool(config.gpu_step) != bool(config.npu_step):
        issue("warnings", "asymmetric_step_scope", "both", "Only one backend selects a step. Confirm equal workloads before interpreting performance.")
    report["status"] = "invalid" if report["errors"] else "needs_selection" if report["selections_required"] else "ready"
    return (report, loaded, details) if retain_data else report

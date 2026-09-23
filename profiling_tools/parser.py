from __future__ import annotations

import bisect
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .model import KernelDetail, KernelRecord, OpRecord, ParsedProfile, TraceEvent


_NORMALIZE_KEY_RE = re.compile(r"[^a-z0-9]+")
_FLOAT_SUFFIX_RE = re.compile(r"\.0+$")


def _normalized_key(value: Any) -> str:
    return _NORMALIZE_KEY_RE.sub("", str(value).lower())


def _normalized_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return _FLOAT_SUFFIX_RE.sub("", text)


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def get_arg(args: Mapping[str, Any], *aliases: str) -> Any:
    if not args:
        return None
    normalized = {_normalized_key(key): value for key, value in args.items()}
    for alias in aliases:
        key = _normalized_key(alias)
        if key in normalized:
            return normalized[key]
    return None


def canonical_op_name(name: str) -> str:
    value = " ".join(name.strip().split()).lower()
    if "::" in value:
        namespace, op = value.split("::", 1)
        if "." in op:
            op = op.split(".", 1)[0]
        return f"{namespace}::{op}"
    return value


def normalize_kernel_name(name: str) -> str:
    value = name.lower()
    value = re.sub(r"<.*?>", "", value)
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def _stable_json(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except (ValueError, TypeError):
            return " ".join(text.split())
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_trace_events(path: Path) -> Tuple[List[TraceEvent], int]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    raw_events = payload.get("traceEvents", []) if isinstance(payload, dict) else payload
    if not isinstance(raw_events, list):
        raise ValueError(f"{path}: traceEvents must be a list")

    events: List[TraceEvent] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, dict):
            continue
        phase = str(raw.get("ph", ""))
        timestamp = _safe_float(raw.get("ts"))
        if timestamp is None:
            if phase.upper() == "M":
                timestamp = 0.0
            else:
                continue
        duration = _safe_float(raw.get("dur")) or 0.0
        args = dict(raw.get("args")) if isinstance(raw.get("args"), dict) else {}
        if "id" in raw:
            args.setdefault("__flow_id__", raw["id"])
        if "bind_id" in raw:
            args.setdefault("__bind_id__", raw["bind_id"])
        events.append(
            TraceEvent(
                index=index,
                name=str(raw.get("name", "")),
                category=str(raw.get("cat", "")),
                phase=phase,
                timestamp_us=timestamp,
                duration_us=duration,
                pid=_normalized_id(raw.get("pid")),
                tid=_normalized_id(raw.get("tid")),
                args=args,
            )
        )
    return events, len(raw_events)


def _is_complete_event(event: TraceEvent) -> bool:
    return event.phase.upper() == "X" and event.duration_us >= 0


def _device_process_ids(events: Sequence[TraceEvent]) -> List[str]:
    result = set()
    for event in events:
        if event.phase.upper() != "M" or event.name.lower() != "process_name":
            continue
        process_name = str(get_arg(event.args, "name") or "").lower()
        if "ascend hardware" in process_name or "npu hardware" in process_name:
            result.add(event.pid)
    return sorted(result)


def _task_type(event: TraceEvent) -> str:
    return str(get_arg(event.args, "Task Type", "task_type") or "").strip()


def _is_communication_event(event: TraceEvent) -> bool:
    category = event.category.lower()
    task_type = _task_type(event).lower()
    name = event.name.lower()
    return any(
        token in category or token in task_type or token in name
        for token in ("communication", "nccl", "hccl")
    )


def _is_kernel(
    event: TraceEvent, include_communication: bool, device_process_ids: Sequence[str]
) -> bool:
    if _is_communication_event(event):
        return include_communication
    category = event.category.lower()
    if "kernel" in category:
        return True
    device_type = str(get_arg(event.args, "Device Type", "device_type") or "").lower()
    if device_type in {"cuda", "npu", "privateuse1", "xpu"}:
        return True
    # Ascend trace_view.json commonly omits cat="kernel". Device tasks are
    # instead placed under the "Ascend Hardware" process and carry Task Type.
    if event.pid in device_process_ids:
        return bool(_task_type(event)) or event.duration_us > 0
    return bool(_task_type(event))


def _is_context_event(event: TraceEvent) -> bool:
    category = event.category.lower()
    name = event.name.lower()
    if "user_annotation" in category:
        return True
    return "python_function" in category and ("nn.module" in name or "module:" in name)


def _is_step_event(event: TraceEvent) -> bool:
    return "profilerstep#" in event.name.lower() or event.name.lower().startswith("step#")


def _step_matches(name: str, requested: str) -> bool:
    if name == requested:
        return True
    lowered = name.lower()
    value = requested.lower()
    return lowered.endswith(f"#{value}") or lowered.endswith(f"step {value}")


def _select_pid(events: Sequence[TraceEvent], op_prefix: str, requested_pid: str) -> str:
    if requested_pid:
        return _normalized_id(requested_pid)
    counts: Dict[str, int] = defaultdict(int)
    for event in events:
        if _is_complete_event(event) and event.name.lower().startswith(op_prefix.lower()):
            counts[event.pid] += 1
    if not counts:
        raise ValueError(f"No operator whose name starts with {op_prefix!r} was found")
    return max(counts.items(), key=lambda item: item[1])[0]


def _find_selected_step(
    events: Sequence[TraceEvent], selected_pid: str, requested_step: str
) -> Tuple[Optional[Tuple[float, float]], str, List[str]]:
    steps = [
        event
        for event in events
        if _is_complete_event(event) and _is_step_event(event) and event.pid == selected_pid
    ]
    names = [event.name for event in steps]
    if not requested_step:
        return None, "all", names
    for event in steps:
        if _step_matches(event.name, requested_step):
            return (event.timestamp_us, event.end_us), event.name, names
    available = ", ".join(names[:20]) or "none"
    raise ValueError(f"Step {requested_step!r} was not found; available steps: {available}")


def _event_in_window(event: TraceEvent, window: Optional[Tuple[float, float]]) -> bool:
    if window is None:
        return True
    return window[0] <= event.timestamp_us < window[1]


def _build_op_hierarchy_and_context(
    ops: List[OpRecord], contexts: Sequence[TraceEvent]
) -> None:
    items_by_thread: Dict[Tuple[str, str], List[Tuple[str, int, TraceEvent]]] = defaultdict(list)
    for index, op in enumerate(ops):
        items_by_thread[(op.event.pid, op.event.tid)].append(("op", index, op.event))
    for index, context in enumerate(contexts):
        items_by_thread[(context.pid, context.tid)].append(("context", index, context))

    for items in items_by_thread.values():
        items.sort(key=lambda item: (item[2].timestamp_us, -item[2].end_us, item[0] != "context"))
        stack: List[Tuple[str, int, TraceEvent]] = []
        for kind, index, event in items:
            while stack and stack[-1][2].end_us < event.end_us:
                stack.pop()
            if kind == "op":
                parent = next((item for item in reversed(stack) if item[0] == "op"), None)
                if parent is not None:
                    ops[index].parent_index = parent[1]
                module_names = [
                    item[2].name
                    for item in stack
                    if item[0] == "context" and not _is_step_event(item[2])
                ]
                deduplicated: List[str] = []
                for name in module_names:
                    if not deduplicated or deduplicated[-1] != name:
                        deduplicated.append(name)
                ops[index].module_path = " / ".join(deduplicated)
            if event.duration_us > 0:
                stack.append((kind, index, event))


def _find_containing_op(
    ops: Sequence[OpRecord],
    starts_by_thread: Mapping[Tuple[str, str], Tuple[List[float], List[int]]],
    pid: str,
    tid: str,
    timestamp_us: float,
) -> Optional[int]:
    starts, indices = starts_by_thread.get((pid, tid), ([], []))
    position = bisect.bisect_right(starts, timestamp_us) - 1
    best: Optional[int] = None
    best_duration = float("inf")
    for cursor in range(position, max(-1, position - 256), -1):
        op_index = indices[cursor]
        event = ops[op_index].event
        if event.timestamp_us <= timestamp_us <= event.end_us and event.duration_us < best_duration:
            best = op_index
            best_duration = event.duration_us
    return best


def _build_starts_by_thread(
    ops: Sequence[OpRecord],
) -> Dict[Tuple[str, str], Tuple[List[float], List[int]]]:
    grouped: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        grouped[(op.event.pid, op.event.tid)].append(index)
    result: Dict[Tuple[str, str], Tuple[List[float], List[int]]] = {}
    for key, indices in grouped.items():
        indices.sort(key=lambda index: ops[index].event.timestamp_us)
        result[key] = ([ops[index].event.timestamp_us for index in indices], indices)
    return result


def _build_starts_by_pid(
    ops: Sequence[OpRecord],
) -> Dict[str, Tuple[List[float], List[int]]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        grouped[op.event.pid].append(index)
    result: Dict[str, Tuple[List[float], List[int]]] = {}
    for pid, indices in grouped.items():
        indices.sort(key=lambda index: ops[index].event.timestamp_us)
        result[pid] = ([ops[index].event.timestamp_us for index in indices], indices)
    return result


def _find_containing_from_sorted(
    ops: Sequence[OpRecord],
    indexed_ops: Tuple[List[float], List[int]],
    timestamp_us: float,
) -> Optional[int]:
    starts, indices = indexed_ops
    position = bisect.bisect_right(starts, timestamp_us) - 1
    best: Optional[int] = None
    best_duration = float("inf")
    for cursor in range(position, max(-1, position - 2048), -1):
        op_index = indices[cursor]
        event = ops[op_index].event
        if event.timestamp_us <= timestamp_us <= event.end_us and event.duration_us < best_duration:
            best = op_index
            best_duration = event.duration_us
    return best


def _find_containing_op_relaxed(
    ops: Sequence[OpRecord],
    starts_by_thread: Mapping[Tuple[str, str], Tuple[List[float], List[int]]],
    starts_by_pid: Mapping[str, Tuple[List[float], List[int]]],
    all_starts: Tuple[List[float], List[int]],
    pid: str,
    tid: str,
    timestamp_us: float,
) -> Optional[int]:
    exact = _find_containing_op(ops, starts_by_thread, pid, tid, timestamp_us)
    if exact is not None:
        return exact
    same_process = _find_containing_from_sorted(
        ops, starts_by_pid.get(pid, ([], [])), timestamp_us
    )
    if same_process is not None:
        return same_process
    return _find_containing_from_sorted(ops, all_starts, timestamp_us)


def _build_kernel_start_index(
    kernels: Sequence[KernelRecord], key_kind: str
) -> Dict[Any, Tuple[List[float], List[int]]]:
    grouped: Dict[Any, List[int]] = defaultdict(list)
    for index, kernel in enumerate(kernels):
        if key_kind == "thread":
            key: Any = (kernel.event.pid, kernel.event.tid)
        elif key_kind == "tid":
            key = kernel.event.tid
        else:
            key = "all"
        grouped[key].append(index)
    result: Dict[Any, Tuple[List[float], List[int]]] = {}
    for key, indices in grouped.items():
        indices.sort(key=lambda index: kernels[index].event.timestamp_us)
        result[key] = (
            [kernels[index].event.timestamp_us for index in indices],
            indices,
        )
    return result


def _nearest_kernel(
    kernels: Sequence[KernelRecord],
    indexed_kernels: Tuple[List[float], List[int]],
    timestamp_us: float,
    tolerance_us: float = 1000.0,
) -> Optional[int]:
    starts, indices = indexed_kernels
    if not starts:
        return None
    position = bisect.bisect_left(starts, timestamp_us)
    candidates = indices[max(0, position - 4) : min(len(indices), position + 5)]
    if not candidates:
        return None

    def distance(index: int) -> float:
        event = kernels[index].event
        if event.timestamp_us <= timestamp_us <= event.end_us:
            return 0.0
        return min(
            abs(event.timestamp_us - timestamp_us), abs(event.end_us - timestamp_us)
        )

    best = min(candidates, key=distance)
    return best if distance(best) <= tolerance_us else None


def _choose_external_op(
    candidate_indices: Sequence[int], ops: Sequence[OpRecord], timestamp_us: Optional[float]
) -> Optional[int]:
    if not candidate_indices:
        return None
    if timestamp_us is None:
        return candidate_indices[-1]
    containing = [
        index
        for index in candidate_indices
        if ops[index].event.timestamp_us <= timestamp_us <= ops[index].event.end_us
    ]
    if containing:
        return min(containing, key=lambda index: ops[index].event.duration_us)
    preceding = [index for index in candidate_indices if ops[index].event.timestamp_us <= timestamp_us]
    return preceding[-1] if preceding else candidate_indices[0]


def _flow_key(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _normalized_id(value)


def _link_kernels_to_ops(
    events: Sequence[TraceEvent], ops: List[OpRecord], kernels: List[KernelRecord]
) -> None:
    for op in ops:
        op.kernel_indices.clear()
    for kernel in kernels:
        kernel.owner_op_index = None
        kernel.link_method = "unmatched"

    external_to_ops: Dict[str, List[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        if op.external_id:
            external_to_ops[op.external_id].append(index)

    starts_by_thread = _build_starts_by_thread(ops)
    starts_by_pid = _build_starts_by_pid(ops)
    all_indices = sorted(range(len(ops)), key=lambda index: ops[index].event.timestamp_us)
    all_starts = ([ops[index].event.timestamp_us for index in all_indices], all_indices)
    runtime_by_correlation: Dict[str, List[TraceEvent]] = defaultdict(list)
    for event in events:
        if not _is_complete_event(event):
            continue
        name = event.name.lower()
        if any(token in name for token in ("launchkernel", "launch_kernel", "rtkernellaunch", "aclrtlaunch")):
            correlation = _normalized_id(get_arg(event.args, "correlation", "correlation id"))
            if correlation:
                runtime_by_correlation[correlation].append(event)

    kernels_by_thread = _build_kernel_start_index(kernels, "thread")
    kernels_by_tid = _build_kernel_start_index(kernels, "tid")
    all_kernels = _build_kernel_start_index(kernels, "all").get("all", ([], []))

    flow_starts: Dict[str, List[TraceEvent]] = defaultdict(list)
    flow_finishes: Dict[str, List[TraceEvent]] = defaultdict(list)
    for event in events:
        phase = event.phase.lower()
        if phase not in {"s", "t", "f"}:
            continue
        raw_id = get_arg(event.args, "id")
        if raw_id is None:
            raw_id = get_arg(event.args, "__bind_id__", "bind_id")
        if raw_id is None:
            # Chrome flow IDs usually live at the event top-level. The loader stores
            # only args, so accept common exporters that duplicate it as flow_id.
            raw_id = get_arg(event.args, "flow_id", "flow id")
        key = _flow_key(raw_id)
        if not key:
            continue
        if phase == "s":
            flow_starts[key].append(event)
        elif phase == "f":
            flow_finishes[key].append(event)

    flow_links: Dict[int, int] = {}
    for key, starts in flow_starts.items():
        finishes = flow_finishes.get(key, [])
        for start, finish in zip(sorted(starts, key=lambda event: event.timestamp_us), sorted(finishes, key=lambda event: event.timestamp_us)):
            op_index = _find_containing_op_relaxed(
                ops,
                starts_by_thread,
                starts_by_pid,
                all_starts,
                start.pid,
                start.tid,
                start.timestamp_us,
            )
            if op_index is None:
                continue
            # Ascend flow endpoints are often emitted on a Host PID while the
            # actual task lives under the "Ascend Hardware" PID. Prefer the
            # exact thread, then the same TID/stream across PIDs, then time.
            kernel_index = _nearest_kernel(
                kernels,
                kernels_by_thread.get((finish.pid, finish.tid), ([], [])),
                finish.timestamp_us,
            )
            if kernel_index is None:
                kernel_index = _nearest_kernel(
                    kernels,
                    kernels_by_tid.get(finish.tid, ([], [])),
                    finish.timestamp_us,
                )
            if kernel_index is None:
                kernel_index = _nearest_kernel(kernels, all_kernels, finish.timestamp_us)
            if kernel_index is None:
                continue
            flow_links[kernel_index] = op_index

    for kernel_index, kernel in enumerate(kernels):
        owner: Optional[int] = None
        method = "unmatched"

        if kernel.external_id:
            owner = _choose_external_op(
                external_to_ops.get(kernel.external_id, []), ops, None
            )
            if owner is not None:
                method = "external_id"

        if owner is None and kernel.correlation_id:
            runtime_candidates = runtime_by_correlation.get(kernel.correlation_id, [])
            if runtime_candidates:
                runtime = min(
                    runtime_candidates,
                    key=lambda event: abs(event.timestamp_us - kernel.event.timestamp_us),
                )
                runtime_external = _normalized_id(
                    get_arg(runtime.args, "External id", "external_id")
                )
                if runtime_external:
                    owner = _choose_external_op(
                        external_to_ops.get(runtime_external, []), ops, runtime.timestamp_us
                    )
                if owner is None:
                    owner = _find_containing_op(
                        ops,
                        starts_by_thread,
                        runtime.pid,
                        runtime.tid,
                        runtime.timestamp_us,
                    )
                if owner is not None:
                    method = "runtime_correlation"

        if owner is None and kernel_index in flow_links:
            owner = flow_links[kernel_index]
            method = "flow"

        if owner is not None:
            kernel.owner_op_index = owner
            kernel.link_method = method
            ops[owner].kernel_indices.append(kernel_index)


def parse_trace(
    path: str,
    side: str,
    *,
    op_prefix: str = "aten::",
    requested_pid: str = "",
    requested_step: str = "",
    include_communication: bool = False,
) -> ParsedProfile:
    trace_path = Path(path)
    events, total_trace_events = _load_trace_events(trace_path)
    device_process_ids = _device_process_ids(events)
    task_type_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        task_type = _task_type(event)
        if _is_complete_event(event) and task_type:
            task_type_counts[task_type] += 1
    selected_pid = _select_pid(events, op_prefix, requested_pid)
    step_window, selected_step, step_names = _find_selected_step(
        events, selected_pid, requested_step
    )

    op_events = [
        event
        for event in events
        if _is_complete_event(event)
        and event.pid == selected_pid
        and event.name.lower().startswith(op_prefix.lower())
        and _event_in_window(event, step_window)
    ]
    op_events.sort(key=lambda event: (event.timestamp_us, -event.duration_us, event.index))
    ops = [
        OpRecord(
            event=event,
            canonical_name=canonical_op_name(event.name),
            external_id=_normalized_id(get_arg(event.args, "External id", "external_id")),
            input_shapes=_stable_json(
                get_arg(event.args, "Input Dims", "Input Shapes", "input_shapes")
            ),
            input_dtypes=_stable_json(
                get_arg(event.args, "Input type", "Input Types", "Input Data Types")
            ),
            call_stack=str(get_arg(event.args, "Call stack", "Call Stack") or ""),
        )
        for event in op_events
    ]

    contexts = [
        event
        for event in events
        if _is_complete_event(event)
        and event.pid == selected_pid
        and _is_context_event(event)
        and _event_in_window(event, step_window)
    ]
    _build_op_hierarchy_and_context(ops, contexts)

    kernels: List[KernelRecord] = []
    for event in events:
        if not _is_complete_event(event) or not _is_kernel(
            event, include_communication, device_process_ids
        ):
            continue
        kernels.append(
            KernelRecord(
                event=event,
                external_id=_normalized_id(get_arg(event.args, "External id", "external_id")),
                correlation_id=_normalized_id(
                    get_arg(event.args, "correlation", "correlation id")
                ),
                task_id=_normalized_id(get_arg(event.args, "Task Id", "Task ID", "task_id")),
                stream_id=_normalized_id(
                    get_arg(event.args, "Stream Id", "Stream ID", "stream", "stream_id")
                ),
            )
        )
    kernels.sort(key=lambda kernel: (kernel.event.timestamp_us, kernel.event.index))
    _link_kernels_to_ops(events, ops, kernels)

    if step_window is not None:
        retained = [
            index
            for index, kernel in enumerate(kernels)
            if kernel.owner_op_index is not None or _event_in_window(kernel.event, step_window)
        ]
        old_to_new = {old: new for new, old in enumerate(retained)}
        kernels = [kernels[index] for index in retained]
        for op in ops:
            op.kernel_indices = [
                old_to_new[index] for index in op.kernel_indices if index in old_to_new
            ]

    return ParsedProfile(
        side=side,
        source_path=str(trace_path),
        ops=ops,
        kernels=kernels,
        selected_pid=selected_pid,
        selected_step=selected_step,
        step_names=step_names,
        total_trace_events=total_trace_events,
        trace_events=events,
        device_process_ids=device_process_ids,
        task_type_counts=dict(task_type_counts),
    )


def _row_lookup(row: Mapping[str, str], *aliases: str) -> str:
    normalized = {_normalized_key(key): value for key, value in row.items()}
    for alias in aliases:
        key = _normalized_key(alias)
        if key in normalized:
            return str(normalized[key] or "").strip()
    return ""


def load_kernel_details(path: str, requested_step: str = "") -> List[KernelDetail]:
    csv_path = Path(path)
    details: List[KernelDetail] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV header is missing")
        for row_index, row in enumerate(reader, start=2):
            step_id = _row_lookup(row, "Step Id", "Step ID")
            if requested_step and step_id and not _step_matches(step_id, requested_step):
                continue
            details.append(
                KernelDetail(
                    row_index=row_index,
                    name=_row_lookup(row, "Name", "Kernel Name", "OP Name"),
                    op_type=_row_lookup(row, "Type", "OP Type"),
                    core_type=_row_lookup(row, "Accelerator Core", "Core Type"),
                    task_id=_normalized_id(_row_lookup(row, "Task ID", "Task Id")),
                    stream_id=_normalized_id(_row_lookup(row, "Stream ID", "Stream Id")),
                    device_id=_normalized_id(_row_lookup(row, "Device_id", "Device ID")),
                    step_id=step_id,
                    start_us=_safe_float(_row_lookup(row, "Start Time(us)", "Start Time(μs)")),
                    duration_us=_safe_float(_row_lookup(row, "Duration(us)", "Duration(μs)")),
                    input_shapes=_row_lookup(row, "Input Shapes"),
                    input_dtypes=_row_lookup(row, "Input Data Types"),
                    output_shapes=_row_lookup(row, "Output Shapes"),
                    output_dtypes=_row_lookup(row, "Output Data Types"),
                    raw=dict(row),
                )
            )
    return details


def _detail_kernel_cost(detail: KernelDetail, kernel: KernelRecord) -> float:
    cost = 0.0
    detail_name = normalize_kernel_name(detail.name)
    kernel_name = normalize_kernel_name(kernel.event.name)
    if detail_name and kernel_name:
        if detail_name == kernel_name:
            cost -= 5.0
        elif detail_name in kernel_name or kernel_name in detail_name:
            cost -= 2.0
        else:
            cost += 2.0
    if detail.duration_us is not None and max(detail.duration_us, kernel.event.duration_us) > 0:
        cost += abs(detail.duration_us - kernel.event.duration_us) / max(
            detail.duration_us, kernel.event.duration_us
        )
    if detail.start_us is not None:
        delta = abs(detail.start_us - kernel.event.timestamp_us)
        if delta < 1_000_000:
            cost += min(delta / 1000.0, 3.0)
    return cost


def attach_kernel_details(profile: ParsedProfile, details: Sequence[KernelDetail]) -> None:
    profile.kernel_detail_count = len(details)
    profile.kernel_detail_matched_count = 0
    profile.kernel_detail_synthetic_count = 0
    unmatched_kernel_indices = set(range(len(profile.kernels)))
    remaining_details: Dict[int, KernelDetail] = {
        detail.row_index: detail for detail in details
    }

    def assign(detail: KernelDetail, kernel_index: int) -> None:
        profile.kernels[kernel_index].detail = detail
        unmatched_kernel_indices.discard(kernel_index)
        remaining_details.pop(detail.row_index, None)

    kernels_by_ids: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, kernel in enumerate(profile.kernels):
        if kernel.task_id and kernel.stream_id:
            kernels_by_ids[(kernel.task_id, kernel.stream_id)].append(index)

    for detail in details:
        candidates = [
            index
            for index in kernels_by_ids.get((detail.task_id, detail.stream_id), [])
            if index in unmatched_kernel_indices
        ]
        if candidates:
            best = min(candidates, key=lambda index: _detail_kernel_cost(detail, profile.kernels[index]))
            assign(detail, best)

    details_by_stream: Dict[str, List[KernelDetail]] = defaultdict(list)
    kernels_by_stream: Dict[str, List[int]] = defaultdict(list)
    for detail in remaining_details.values():
        details_by_stream[detail.stream_id].append(detail)
    for index in unmatched_kernel_indices:
        kernels_by_stream[profile.kernels[index].stream_id].append(index)

    for stream_id, stream_details in details_by_stream.items():
        stream_kernels = kernels_by_stream.get(stream_id, [])
        stream_details.sort(key=lambda detail: (detail.start_us is None, detail.start_us or 0.0, detail.row_index))
        stream_kernels.sort(key=lambda index: profile.kernels[index].event.timestamp_us)
        if len(stream_details) == len(stream_kernels):
            for detail, kernel_index in zip(stream_details, stream_kernels):
                assign(detail, kernel_index)
            continue
        remaining = set(stream_kernels)
        for detail in stream_details:
            if not remaining:
                break
            name_matches = [
                index
                for index in remaining
                if normalize_kernel_name(detail.name)
                == normalize_kernel_name(profile.kernels[index].event.name)
            ]
            candidates = name_matches or list(remaining)
            best = min(candidates, key=lambda index: _detail_kernel_cost(detail, profile.kernels[index]))
            if _detail_kernel_cost(detail, profile.kernels[best]) <= 4.0:
                assign(detail, best)
                remaining.remove(best)

    # Some Ascend trace variants omit Stream ID or use a different device PID.
    # Make one final global pass using name, timestamp and duration.
    for detail in list(remaining_details.values()):
        if not unmatched_kernel_indices:
            break
        name_matches = [
            index
            for index in unmatched_kernel_indices
            if normalize_kernel_name(detail.name)
            == normalize_kernel_name(profile.kernels[index].event.name)
        ]
        candidates = name_matches or list(unmatched_kernel_indices)
        best = min(candidates, key=lambda index: _detail_kernel_cost(detail, profile.kernels[index]))
        if _detail_kernel_cost(detail, profile.kernels[best]) <= 4.0:
            assign(detail, best)

    profile.kernel_detail_matched_count = sum(
        kernel.detail is not None for kernel in profile.kernels
    )

    # kernel_details.csv is the authoritative NPU task inventory. If a row has
    # no trace event, preserve it as a synthetic kernel so bundle counts do not
    # silently become zero. Flow timestamps can still associate it with aten.
    synthetic_kernels: List[KernelRecord] = []
    for detail in remaining_details.values():
        if detail.start_us is None or detail.duration_us is None:
            continue
        args: Dict[str, Any] = {
            "Task ID": detail.task_id,
            "Stream ID": detail.stream_id,
            "Task Type": detail.core_type or detail.op_type,
            "Synthetic Source": "kernel_details.csv",
        }
        synthetic_kernels.append(
            KernelRecord(
                event=TraceEvent(
                    index=-detail.row_index,
                    name=detail.name or detail.op_type or f"kernel_details_row_{detail.row_index}",
                    category="kernel_details",
                    phase="X",
                    timestamp_us=detail.start_us,
                    duration_us=detail.duration_us,
                    pid=f"ascend-hardware:{detail.device_id or 'unknown'}",
                    tid=detail.stream_id,
                    args=args,
                ),
                task_id=detail.task_id,
                stream_id=detail.stream_id,
                detail=detail,
            )
        )

    if synthetic_kernels:
        profile.kernels.extend(synthetic_kernels)
        profile.kernels.sort(key=lambda kernel: (kernel.event.timestamp_us, kernel.event.index))
        profile.kernel_detail_synthetic_count = len(synthetic_kernels)
        _link_kernels_to_ops(profile.trace_events, profile.ops, profile.kernels)

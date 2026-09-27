from __future__ import annotations

import bisect
import csv
import json
import math
import re
from collections import Counter, defaultdict
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


_CUSTOM_SCOPE_NAMES = (
    (re.compile(r"(?:^|[/\\])position_concat\.py\(\d+\): position_concat(?:,|$)"),
     "position_concat_fwd_kernel"),
    (re.compile(r"(?:^|[/\\])position_offset_creator\.py\(\d+\): create_position_offset(?:,|$)"),
     "create_position_offset_fwd_kernel"),
)


def canonical_op_name(name: str) -> str:
    value = " ".join(name.strip().split()).lower()
    if "::" in value:
        namespace, op = value.split("::", 1)
        if "." in op:
            op = op.split(".", 1)[0]
        return f"{namespace}::{op}"
    # Explicit Python scopes for custom kernels without record_function on CUDA.
    if ".py(" in value:
        for pattern, canonical in _CUSTOM_SCOPE_NAMES:
            if pattern.search(value):
                return canonical
    return value


_EXTRA_OPERATOR_NAMES_BY_SIDE = {
    # FlashAttentionFunc is a record_function/custom autograd op in CUDA
    # traces, not an aten:: event.  Keep it as an operator so its directly
    # launched fwd_kernel bundle can participate in explicit cross-backend
    # mappings without broadening the default operator prefix.
    "gpu": {"flashattentionfunc", "nestedconcatfunc",
            "position_concat_fwd_kernel", "create_position_offset_fwd_kernel"},
    "npu": {"nestedconcatfunc", "position_concat_fwd_kernel",
            "create_position_offset_fwd_kernel"},
}


def _is_selected_operator_name(name: str, side: str, op_prefix: str) -> bool:
    return name.lower().startswith(op_prefix.lower()) or canonical_op_name(name) in (
        _EXTRA_OPERATOR_NAMES_BY_SIDE.get(side.lower(), set())
    )


def normalize_kernel_name(name: str) -> str:
    value = name.lower()
    value = re.sub(r"<.*?>", "", value)
    value = re.sub(r"[^a-z0-9]+", "", value)
    return value


def _operator_type_key(name: str) -> str:
    """Return the comparable part of an aten name or kernel detail type."""
    value = canonical_op_name(name)
    if "::" in value:
        value = value.split("::", 1)[1]
    return normalize_kernel_name(value)


_KERNEL_API_TO_ATEN = (
    (re.compile(r"^aclnnStack_", re.IGNORECASE), "aten::stack", "concatd"),
    (re.compile(r"^aclnnAddmm_", re.IGNORECASE), "aten::addmm", "matmulv3"),
    (re.compile(r"^aclnnClamp_", re.IGNORECASE), "aten::clamp", "clipbyvaluev2"),
    (re.compile(r"^aclnnEmbedding_", re.IGNORECASE), "aten::embedding", "gatherv2"),
    (
        re.compile(r"^aclnnLayerNormWithImplMode_", re.IGNORECASE),
        "aten::native_layer_norm",
        "layernormv4",
    ),
    (re.compile(r"^aclnnBitwiseNot_", re.IGNORECASE), "aten::bitwise_not", "logicalnot"),
    (re.compile(r"^aclnnNe(?:Tensor|Scalar)_", re.IGNORECASE), "aten::ne", "notequal"),
    (re.compile(r"^aclnnSilu_", re.IGNORECASE), "aten::silu", "swish"),
    (re.compile(r"^aclnnSWhere_", re.IGNORECASE), "aten::where", "selectv2"),
    (re.compile(r"^aclnnMatmul_", re.IGNORECASE), "aten::matmul", "matmulv3"),
    (re.compile(r"^aclnnCumsum_", re.IGNORECASE), "aten::cumsum", "cumsum"),
    (re.compile(r"^aclnnInplaceFillScalar_", re.IGNORECASE), "aten::fill_", "fill"),
)


_KERNEL_API_EXPECTED_ATEN = (
    (re.compile(r"^aclnnCat_", re.IGNORECASE), {"aten::cat", "aten::concat"}, "concatd"),
    (re.compile(r"^aclnnRepeat_", re.IGNORECASE), {"aten::repeat", "aten::repeat_interleave"}, "tile"),
    (re.compile(r"^aclnnSubs_", re.IGNORECASE), {"aten::sub", "aten::sub_"}, "sub"),
    (re.compile(r"^aclnnPowTensorScalar_", re.IGNORECASE), {"aten::pow"}, "square"),
    (re.compile(r"^aclnnReduceSum_", re.IGNORECASE), {"aten::sum", "aten::count_nonzero"}, "reducesum"),
    (re.compile(r"^aclnnInplaceCopy_", re.IGNORECASE), {"aten::copy_"}, "cast"),
)

_KNOWN_CAST_HELPER_API = re.compile(
    r"^aclnn(?:Cat|Cumsum|Divs|Subs|ReduceSum|GtScalar|PowTensorScalar|Stack)_",
    re.IGNORECASE,
)

_KNOWN_AUXILIARY_API_TYPES = (
    (re.compile(r"^aclnnIndexPutImpl_", re.IGNORECASE), {"broadcastto"}),
    (re.compile(r"^aclnnMaxDim_", re.IGNORECASE), {"slice"}),
)


def _kernel_api_target_op(kernel: KernelRecord) -> str:
    if kernel.detail is None:
        return ""
    detail_type_key = _operator_type_key(kernel.detail.op_type)
    for pattern, canonical_name, primary_type_key in _KERNEL_API_TO_ATEN:
        # An ACLNN call name can prefix both its main compute kernel and helper
        # kernels (Cast/Tile/Concat, etc.).  Only the authoritative Type that
        # represents the op itself may participate in owner reconciliation.
        if pattern.match(kernel.detail.name) and detail_type_key == primary_type_key:
            return canonical_name
    return ""


def kernel_api_expected_ops(kernel: KernelRecord) -> set[str]:
    """Return aten owners supported by the ACLNN API and its primary Type.

    Helper kernels from the same ACLNN call deliberately have no constraint.
    """
    target = _kernel_api_target_op(kernel)
    if target:
        return {target}
    if kernel.detail is None:
        return set()
    detail_type_key = _operator_type_key(kernel.detail.op_type)
    for pattern, expected, primary_type_key in _KERNEL_API_EXPECTED_ATEN:
        if pattern.match(kernel.detail.name) and detail_type_key == primary_type_key:
            return expected
    return set()


def kernel_api_is_auxiliary(kernel: KernelRecord) -> bool:
    """Identify confirmed setup kernels that must not consume GPU compute time."""
    if kernel.detail is None:
        return False
    name = kernel.detail.name
    type_key = _operator_type_key(kernel.detail.op_type)
    if type_key == "cast" and _KNOWN_CAST_HELPER_API.match(name):
        return True
    if re.match(r"^aclnnCat_", name, re.IGNORECASE) and type_key != "concatd":
        return True
    return any(
        pattern.match(name) and type_key in helper_types
        for pattern, helper_types in _KNOWN_AUXILIARY_API_TYPES
    )


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


def _normalize_dimension(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    text = str(value).strip().strip('"').strip("'")
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return text.lower()


def _normalize_shape_item(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [_normalize_dimension(item) for item in value]
    text = str(value).strip().strip('"').strip("'")
    if not text or text in {"[]", "()"}:
        return []
    text = text.strip("[]()")
    if not text:
        return []
    return [
        _normalize_dimension(item)
        for item in re.split(r"\s*(?:,|x|\*)\s*", text)
        if item.strip()
    ]


def normalize_input_shapes(value: Any) -> str:
    """Normalize GPU JSON shapes and Ascend semicolon shapes to one representation."""
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        text = value.strip()
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if decoded is not None and decoded != text:
            return normalize_input_shapes(decoded)
        parts = text.split(";")
        shapes = [_normalize_shape_item(part) for part in parts]
    elif isinstance(value, (list, tuple)):
        if not value:
            shapes = []
        elif all(not isinstance(item, (list, tuple, dict)) for item in value):
            shapes = [_normalize_shape_item(value)]
        else:
            shapes = [_normalize_shape_item(item) for item in value]
    else:
        shapes = [_normalize_shape_item(value)]
    return json.dumps(shapes, ensure_ascii=False, separators=(",", ":"))


_DTYPE_ALIASES = {
    "bool": "bool",
    "byte": "uint8",
    "char": "int8",
    "double": "float64",
    "float": "float32",
    "float32": "float32",
    "float64": "float64",
    "half": "float16",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "long": "int64",
    "long int": "int64",
    "int64": "int64",
    "int": "int32",
    "int32": "int32",
    "short": "int16",
    "int16": "int16",
    "uint8": "uint8",
}


def _normalize_dtype(value: Any) -> str:
    text = str(value).strip().strip('"').strip("'").lower()
    text = text.replace("torch.", "").replace("scalar_type::", "")
    return _DTYPE_ALIASES.get(text, text)


def normalize_input_dtypes(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        text = value.strip()
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            decoded = None
        if decoded is not None and decoded != text:
            return normalize_input_dtypes(decoded)
        values = text.split(";")
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    return json.dumps(
        [_normalize_dtype(item) for item in values],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def normalize_module_key(module_path: str) -> str:
    if not module_path:
        return ""
    parts = re.split(r"\s*/\s*", module_path)
    cleaned: List[str] = []
    for part in parts:
        value = re.sub(r",?\s*callsite\s*:\s*[^,]+", "", part, flags=re.IGNORECASE)
        value = re.sub(r"^(?:nn\.)?module\s*:\s*", "", value, flags=re.IGNORECASE)
        value = " ".join(value.split()).strip(" ,")
        if value:
            cleaned.append(value.lower())
    return cleaned[-1] if cleaned else ""


def _load_trace_events(path: Path, *, strict: bool = False) -> Tuple[List[TraceEvent], int]:
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
        parsed_duration = _safe_float(raw.get("dur"))
        if strict and phase.upper() == "X" and (
            timestamp is None or parsed_duration is None or parsed_duration < 0
        ):
            raise ValueError(f"{path}: event {index} requires finite ts and non-negative finite dur")
        if timestamp is None:
            if phase.upper() == "M":
                timestamp = 0.0
            else:
                continue
        duration = parsed_duration or 0.0
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


def _is_marker_task(event: TraceEvent) -> bool:
    task_type = _task_type(event).lower()
    name = event.name.lower()
    return any(
        token in task_type or token in name
        for token in (
            "event_record",
            "event_wait",
            "place_holder",
            "placeholder",
            "notify_record",
            "notify_wait",
        )
    )


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
    if _is_marker_task(event):
        return False
    category = event.category.lower()
    if category in {"gpu_memcpy", "gpu_memset"}:
        return False
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
                ops[index].module_key = normalize_module_key(ops[index].module_path)
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


def _kernel_distance(kernel: KernelRecord, timestamp_us: float) -> float:
    event = kernel.event
    if event.timestamp_us <= timestamp_us <= event.end_us:
        return 0.0
    return min(
        abs(event.timestamp_us - timestamp_us), abs(event.end_us - timestamp_us)
    )


def _nearest_kernel(
    kernels: Sequence[KernelRecord],
    indexed_kernels: Tuple[List[float], List[int]],
    timestamp_us: float,
    tolerance_us: float = 1000.0,
) -> Tuple[Optional[int], Optional[float], int]:
    starts, indices = indexed_kernels
    if not starts:
        return None, None, 0
    position = bisect.bisect_left(starts, timestamp_us)
    candidates = indices[max(0, position - 4) : min(len(indices), position + 5)]
    if not candidates:
        return None, None, 0

    valid = [
        index
        for index in candidates
        if _kernel_distance(kernels[index], timestamp_us) <= tolerance_us
    ]
    if not valid:
        return None, None, 0
    # Adjacent/overlapping kernels can both contain the endpoint. An exact
    # start timestamp identifies the target more precisely than containment.
    best = min(valid, key=lambda index: (
        _kernel_distance(kernels[index], timestamp_us),
        abs(kernels[index].event.timestamp_us - timestamp_us),
    ))
    return best, _kernel_distance(kernels[best], timestamp_us), len(valid)


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


def _queue_owner_index(
    events: Sequence[TraceEvent],
    ops: Sequence[OpRecord],
    starts_by_thread: Mapping[Tuple[str, str], Tuple[List[float], List[int]]],
) -> Dict[str, Tuple[List[float], List[float], List[Tuple[float, int]]]]:
    """Associate worker dequeue intervals with the original enqueue's aten.

    Host runtime events may use a synthetic PID, so lookup uses the worker TID.
    Correlation matching itself stays within the originating process/API.
    """
    enqueues: Dict[Tuple[str, str, str], List[TraceEvent]] = defaultdict(list)
    dequeues: Dict[Tuple[str, str, str], List[TraceEvent]] = defaultdict(list)
    for event in events:
        if not _is_complete_event(event):
            continue
        name = event.name.lower()
        if not name.startswith(("enqueue@", "dequeue@")):
            continue
        correlation = _normalized_id(get_arg(event.args, "correlation_id"))
        if not correlation:
            continue
        key = (event.pid, correlation, name.split("@", 1)[1])
        (enqueues if name.startswith("enqueue@") else dequeues)[key].append(event)

    ranges: Dict[str, List[Tuple[float, float, int]]] = defaultdict(list)
    for key, dequeue_events in dequeues.items():
        enqueue_events = enqueues.get(key, [])
        # Reused/cropped correlation IDs require additional evidence. Never
        # turn an ambiguous queue association into a strong CPU owner.
        if len(enqueue_events) != 1 or len(dequeue_events) != 1:
            continue
        enqueue, dequeue = enqueue_events[0], dequeue_events[0]
        if enqueue.timestamp_us > dequeue.timestamp_us:
            continue
        owner = _find_containing_op(
            ops, starts_by_thread, enqueue.pid, enqueue.tid, enqueue.timestamp_us
        )
        if owner is not None:
            ranges[dequeue.tid].append((dequeue.timestamp_us, dequeue.end_us, owner))

    result = {}
    for tid, records in ranges.items():
        records.sort()
        starts, max_ends, entries = [], [], []
        max_end = float("-inf")
        for start, end, owner in records:
            starts.append(start)
            max_end = max(max_end, end)
            max_ends.append(max_end)
            entries.append((end, owner))
        result[tid] = (starts, max_ends, entries)
    return result


def _queue_owner_at(
    index: Mapping[str, Tuple[List[float], List[float], List[Tuple[float, int]]]],
    tid: str,
    timestamp_us: float,
) -> Optional[int]:
    starts, max_ends, entries = index.get(tid, ([], [], []))
    cursor = bisect.bisect_right(starts, timestamp_us) - 1
    owners = set()
    while cursor >= 0 and max_ends[cursor] >= timestamp_us:
        end, owner = entries[cursor]
        if end >= timestamp_us:
            owners.add(owner)
        cursor -= 1
    return next(iter(owners)) if len(owners) == 1 else None


def _link_kernels_to_ops(
    events: Sequence[TraceEvent],
    ops: List[OpRecord],
    kernels: List[KernelRecord],
    flow_tolerance_us: float = 1000.0,
) -> Dict[str, Any]:
    for op in ops:
        op.kernel_indices.clear()
    for kernel in kernels:
        kernel.owner_op_index = None
        kernel.link_method = "unmatched"
        kernel.link_scope = ""
        kernel.link_distance_us = None

    diagnostics: Dict[str, Any] = {
        "flow_pair_count": 0,
        "flow_unpaired_start_count": 0,
        "flow_unpaired_finish_count": 0,
        "flow_kernel_not_found_count": 0,
        "flow_kernel_collision_count": 0,
        "flow_ambiguous_candidate_count": 0,
        "flow_scope_counts": {},
        "flow_max_distance_us": 0.0,
        "flow_tolerance_us": flow_tolerance_us,
        "ignored_queue_flow_event_count": 0,
        "flow_source_owner_not_found_count": 0,
        "flow_owner_conflict_count": 0,
    }

    external_to_ops: Dict[str, List[int]] = defaultdict(list)
    for index, op in enumerate(ops):
        if op.external_id:
            external_to_ops[op.external_id].append(index)

    starts_by_thread = _build_starts_by_thread(ops)
    starts_by_pid = _build_starts_by_pid(ops)
    all_indices = sorted(range(len(ops)), key=lambda index: ops[index].event.timestamp_us)
    all_starts = ([ops[index].event.timestamp_us for index in all_indices], all_indices)
    queue_owners = _queue_owner_index(events, ops, starts_by_thread)
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

    flow_starts: Dict[Tuple[str, str], List[TraceEvent]] = defaultdict(list)
    flow_finishes: Dict[Tuple[str, str], List[TraceEvent]] = defaultdict(list)
    for event in events:
        phase = event.phase.lower()
        if phase not in {"s", "t", "f"}:
            continue
        if event.category.lower() == "async_task_queue" or event.name.lower() == "enqueue_to_dequeue":
            # This edge ends on a CPU worker, never on a device kernel.
            diagnostics["ignored_queue_flow_event_count"] += 1
            continue
        raw_id = get_arg(event.args, "__flow_id__", "id")
        if raw_id is None:
            raw_id = get_arg(event.args, "__bind_id__", "bind_id")
        if raw_id is None:
            # Chrome flow IDs usually live at the event top-level. The loader stores
            # only args, so accept common exporters that duplicate it as flow_id.
            raw_id = get_arg(event.args, "flow_id", "flow id")
        flow_id = _flow_key(raw_id)
        if not flow_id:
            continue
        key = (event.category, flow_id)
        if phase == "s":
            flow_starts[key].append(event)
        elif phase == "f":
            flow_finishes[key].append(event)

    flow_links: Dict[int, Tuple[int, str, float, int, str]] = {}
    flow_scores: Dict[int, Tuple[int, int, float, float]] = {}
    ambiguous_flow_kernels: set[int] = set()
    for key, starts in flow_starts.items():
        finishes = flow_finishes.get(key, [])
        ordered_starts = sorted(starts, key=lambda event: event.timestamp_us)
        ordered_finishes = sorted(finishes, key=lambda event: event.timestamp_us)
        diagnostics["flow_unpaired_start_count"] += max(0, len(ordered_starts) - len(ordered_finishes))
        diagnostics["flow_unpaired_finish_count"] += max(0, len(ordered_finishes) - len(ordered_starts))
        for start, finish in zip(ordered_starts, ordered_finishes):
            diagnostics["flow_pair_count"] += 1
            is_torch_flow = start.category.lower() == "async_npu" or start.name.lower() == "torch_to_npu"
            is_host_flow = start.category.lower() == "hosttodevice"
            op_index = _find_containing_op(
                ops, starts_by_thread, start.pid, start.tid, start.timestamp_us
            )
            method, evidence_rank = "flow", 3
            if is_torch_flow:
                method, evidence_rank = "torch_to_npu", 0
            elif is_host_flow:
                queued_owner = _queue_owner_at(queue_owners, start.tid, start.timestamp_us)
                if queued_owner is not None:
                    op_index = queued_owner
                    method, evidence_rank = "host_to_device_queue", 1
                else:
                    # Synchronous launches may use another PID but retain the
                    # CPU thread ID. Never jump to an unrelated thread by time.
                    same_tid_owners = {
                        owner
                        for pid, tid in starts_by_thread
                        if tid == start.tid
                        for owner in [_find_containing_op(
                            ops, starts_by_thread, pid, tid, start.timestamp_us
                        )]
                        if owner is not None
                    }
                    op_index = next(iter(same_tid_owners)) if len(same_tid_owners) == 1 else None
                    method, evidence_rank = "host_to_device_thread", 2
            elif op_index is None:
                op_index = _find_containing_op_relaxed(
                    ops, starts_by_thread, starts_by_pid, all_starts,
                    start.pid, start.tid, start.timestamp_us,
                )
                evidence_rank = 4
            if op_index is None:
                diagnostics["flow_source_owner_not_found_count"] += 1
                continue
            # Ascend flow endpoints are often emitted on a Host PID while the
            # actual task lives under the "Ascend Hardware" PID. Prefer the
            # exact thread, then the same TID/stream across PIDs, then time.
            kernel_index, distance_us, candidate_count = _nearest_kernel(
                kernels,
                kernels_by_thread.get((finish.pid, finish.tid), ([], [])),
                finish.timestamp_us,
                flow_tolerance_us,
            )
            scope = "same_thread"
            if kernel_index is None:
                kernel_index, distance_us, candidate_count = _nearest_kernel(
                    kernels,
                    kernels_by_tid.get(finish.tid, ([], [])),
                    finish.timestamp_us,
                    flow_tolerance_us,
                )
                scope = "same_stream"
            if kernel_index is None and not (is_torch_flow or is_host_flow):
                kernel_index, distance_us, candidate_count = _nearest_kernel(
                    kernels, all_kernels, finish.timestamp_us, flow_tolerance_us
                )
                scope = "global_time"
            if kernel_index is None:
                diagnostics["flow_kernel_not_found_count"] += 1
                continue
            assert distance_us is not None
            if (is_torch_flow or is_host_flow) and distance_us != 0:
                # A missing/filtered device task must not claim a neighboring
                # task merely because it falls within the legacy tolerance.
                diagnostics["flow_kernel_not_found_count"] += 1
                continue
            if candidate_count > 1:
                diagnostics["flow_ambiguous_candidate_count"] += 1
            previous = flow_links.get(kernel_index)
            score = (
                evidence_rank,
                {"same_thread": 0, "same_stream": 1, "global_time": 2}[scope],
                distance_us,
                abs(kernels[kernel_index].event.timestamp_us - finish.timestamp_us),
            )
            if previous is not None:
                diagnostics["flow_kernel_collision_count"] += 1
                if flow_scores[kernel_index] == score and previous[0] != op_index:
                    ambiguous_flow_kernels.add(kernel_index)
                    diagnostics["flow_owner_conflict_count"] += 1
                if flow_scores[kernel_index] <= score:
                    continue
            ambiguous_flow_kernels.discard(kernel_index)
            flow_scores[kernel_index] = score
            flow_links[kernel_index] = (op_index, scope, distance_us, candidate_count, method)
            diagnostics["flow_max_distance_us"] = max(
                diagnostics["flow_max_distance_us"], distance_us
            )

    for kernel_index, kernel in enumerate(kernels):
        owner: Optional[int] = None
        method = "unmatched"

        if kernel.external_id:
            owner = _choose_external_op(
                external_to_ops.get(kernel.external_id, []), ops, None
            )
            if owner is not None:
                method = "external_id"
                kernel.link_scope = "external_id"

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
                    kernel.link_scope = "runtime_correlation"

        if owner is None and kernel_index in flow_links and kernel_index not in ambiguous_flow_kernels:
            owner, scope, distance_us, _, method = flow_links[kernel_index]
            kernel.link_scope = scope
            kernel.link_distance_us = distance_us

        if owner is not None:
            kernel.owner_op_index = owner
            kernel.link_method = method
            ops[owner].kernel_indices.append(kernel_index)
        elif kernel_index in ambiguous_flow_kernels:
            # Preserve the conflict so a later equal-count/order heuristic
            # cannot silently reinterpret it as missing evidence.
            kernel.link_method = "ambiguous_flow"
            kernel.link_scope = "conflicting_flow_sources"

    diagnostics["flow_scope_counts"] = dict(
        Counter(link[1] for index, link in flow_links.items() if index not in ambiguous_flow_kernels)
    )
    diagnostics["flow_selected_method_counts"] = dict(
        Counter(kernel.link_method for kernel in kernels if kernel.link_method in {
            "flow", "torch_to_npu", "host_to_device_queue", "host_to_device_thread"
        })
    )
    diagnostics["flow_unresolved_owner_conflict_count"] = len(ambiguous_flow_kernels)
    return diagnostics


def _gpu_memory_activities(events, ops, selected_pid, step_window):
    """Link device memory events using unique external/runtime evidence only."""
    external = defaultdict(list)
    runtimes = defaultdict(list)
    starts = _build_starts_by_thread(ops)
    for index, op in enumerate(ops):
        if op.external_id:
            external[op.external_id].append(index)
    for event in events:
        if event.category == "cuda_runtime" and _is_complete_event(event):
            correlation = _normalized_id(get_arg(event.args, "correlation", "correlation id"))
            if correlation:
                runtimes[correlation].append(event)
    result = []
    for event in events:
        if not (_is_complete_event(event) and event.category in {"gpu_memcpy", "gpu_memset"}):
            continue
        ext = _normalized_id(get_arg(event.args, "External id"))
        corr = _normalized_id(get_arg(event.args, "correlation", "correlation id"))
        candidates = set(external.get(ext, []))
        runtime_candidates = set()
        runtime_events = runtimes.get(corr, [])
        ambiguous_runtime = len(runtime_events) > 1 or any(
            e.pid != selected_pid or not _event_in_window(e, step_window) for e in runtime_events
        )
        if len(runtime_events) == 1 and not ambiguous_runtime:
            runtime = runtimes[corr][0]
            owner = _find_containing_op(ops, starts, runtime.pid, runtime.tid, runtime.timestamp_us)
            if owner is not None:
                runtime_candidates.add(owner)
        # Conflicting strong evidence remains visible but unassigned.
        owners = candidates | runtime_candidates
        if ambiguous_runtime:
            owners = set()
        owner = next(iter(owners)) if len(owners) == 1 else None
        if owner is None and not _event_in_window(event, step_window):
            continue
        result.append(KernelRecord(
            event=event, external_id=ext, correlation_id=corr,
            stream_id=_normalized_id(get_arg(event.args, "stream", "Stream ID")),
            owner_op_index=owner,
            link_method=("external_id" if candidates else "runtime_correlation")
            if owner is not None else "unmatched",
            link_scope="same_thread" if owner is not None else "",
        ))
    return sorted(result, key=lambda item: (item.event.timestamp_us, item.event.index))


def parse_trace(
    path: str,
    side: str,
    *,
    op_prefix: str = "aten::",
    requested_pid: str = "",
    requested_step: str = "",
    include_communication: bool = False,
    flow_tolerance_us: float = 1000.0,
    trace_data: Optional[Tuple[List[TraceEvent], int]] = None,
) -> ParsedProfile:
    trace_path = Path(path)
    events, total_trace_events = trace_data if trace_data is not None else _load_trace_events(trace_path)
    device_process_ids = _device_process_ids(events)
    selected_pid = _select_pid(events, op_prefix, requested_pid)
    step_window, selected_step, step_names = _find_selected_step(
        events, selected_pid, requested_step
    )
    task_type_counts: Dict[str, int] = defaultdict(int)
    excluded_task_type_counts: Dict[str, int] = defaultdict(int)
    for event in events:
        task_type = _task_type(event)
        if not (_is_complete_event(event) and task_type and _event_in_window(event, step_window)):
            continue
        if _is_kernel(event, include_communication, device_process_ids):
            task_type_counts[task_type] += 1
        else:
            excluded_task_type_counts[task_type] += 1

    op_events = [
        event
        for event in events
        if _is_complete_event(event)
        and event.pid == selected_pid
        and _is_selected_operator_name(event.name, side, op_prefix)
        and not (side == "npu" and event.category == "python_function"
                 and canonical_op_name(event.name) in _EXTRA_OPERATOR_NAMES_BY_SIDE["npu"])
        and _event_in_window(event, step_window)
    ]
    op_events.sort(key=lambda event: (event.timestamp_us, -event.duration_us, event.index))
    ops: List[OpRecord] = []
    for event in op_events:
        raw_shapes = get_arg(event.args, "Input Dims", "Input Shapes", "input_shapes")
        raw_dtypes = get_arg(event.args, "Input type", "Input Types", "Input Data Types")
        ops.append(
            OpRecord(
                event=event,
                canonical_name=canonical_op_name(event.name),
                external_id=_normalized_id(
                    get_arg(event.args, "External id", "external_id")
                ),
                input_shapes=normalize_input_shapes(raw_shapes),
                input_dtypes=normalize_input_dtypes(raw_dtypes),
                raw_input_shapes=_stable_json(raw_shapes),
                raw_input_dtypes=_stable_json(raw_dtypes),
                call_stack=str(get_arg(event.args, "Call stack", "Call Stack") or ""),
            )
        )

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
        stream_id = _normalized_id(
            get_arg(event.args, "Stream Id", "Stream ID", "stream", "stream_id")
        )
        if not stream_id and event.pid in device_process_ids:
            stream_id = event.tid
        kernels.append(
            KernelRecord(
                event=event,
                external_id=_normalized_id(get_arg(event.args, "External id", "external_id")),
                correlation_id=_normalized_id(
                    get_arg(event.args, "correlation", "correlation id")
                ),
                task_id=_normalized_id(get_arg(event.args, "Task Id", "Task ID", "task_id")),
                stream_id=stream_id,
            )
        )
    kernels.sort(key=lambda kernel: (kernel.event.timestamp_us, kernel.event.index))
    flow_link_diagnostics = _link_kernels_to_ops(
        events, ops, kernels, flow_tolerance_us
    )

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
        excluded_task_type_counts=dict(excluded_task_type_counts),
        flow_link_diagnostics=flow_link_diagnostics,
        memory_activities=_gpu_memory_activities(events, ops, selected_pid, step_window)
        if side == "gpu" else [],
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
    detail_link_methods: Dict[str, int] = defaultdict(int)
    unmatched_kernel_indices = set(range(len(profile.kernels)))
    remaining_details: Dict[int, KernelDetail] = {
        detail.row_index: detail for detail in details
    }
    kernel_normalized_names = [
        normalize_kernel_name(kernel.event.name) for kernel in profile.kernels
    ]

    def assign(detail: KernelDetail, kernel_index: int, method: str) -> None:
        profile.kernels[kernel_index].detail = detail
        unmatched_kernel_indices.discard(kernel_index)
        remaining_details.pop(detail.row_index, None)
        detail_link_methods[method] += 1

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
            if len(candidates) == 1 or _detail_kernel_cost(detail, profile.kernels[best]) <= 4.0:
                assign(detail, best, "task_stream_id")

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
                assign(detail, kernel_index, "stream_order")
            continue
        remaining = set(stream_kernels)
        for detail in stream_details:
            if not remaining:
                break
            name_matches = [
                index
                for index in remaining
                if normalize_kernel_name(detail.name)
                == kernel_normalized_names[index]
            ]
            candidates = name_matches or list(remaining)
            best = min(candidates, key=lambda index: _detail_kernel_cost(detail, profile.kernels[index]))
            if _detail_kernel_cost(detail, profile.kernels[best]) <= 4.0:
                assign(detail, best, "stream_fuzzy")
                remaining.remove(best)

    # Some Ascend trace variants omit Stream ID or use a different device PID.
    # Make one final indexed global pass using name, timestamp and duration.
    kernels_by_name: Dict[str, List[int]] = defaultdict(list)
    for index in unmatched_kernel_indices:
        kernels_by_name[kernel_normalized_names[index]].append(index)
    global_indices = sorted(
        unmatched_kernel_indices,
        key=lambda index: profile.kernels[index].event.timestamp_us,
    )
    global_starts = [profile.kernels[index].event.timestamp_us for index in global_indices]
    for detail in list(remaining_details.values()):
        if not unmatched_kernel_indices:
            break
        detail_name = normalize_kernel_name(detail.name)
        name_matches = [
            index
            for index in kernels_by_name.get(detail_name, [])
            if index in unmatched_kernel_indices
        ]
        candidates = name_matches
        if not candidates and detail.start_us is not None:
            position = bisect.bisect_left(global_starts, detail.start_us)
            candidates = [
                index
                for index in global_indices[
                    max(0, position - 8) : min(len(global_indices), position + 9)
                ]
                if index in unmatched_kernel_indices
            ]
        if not candidates:
            continue
        best = min(candidates, key=lambda index: _detail_kernel_cost(detail, profile.kernels[index]))
        if _detail_kernel_cost(detail, profile.kernels[best]) <= 4.0:
            assign(detail, best, "global_fuzzy")

    profile.kernel_detail_matched_count = sum(
        kernel.detail is not None for kernel in profile.kernels
    )
    profile.kernel_detail_link_methods = dict(detail_link_methods)

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
        profile.kernel_detail_link_methods["synthetic"] = len(synthetic_kernels)
        flow_tolerance_us = float(
            profile.flow_link_diagnostics.get("flow_tolerance_us", 1000.0)
        )
        profile.flow_link_diagnostics = _link_kernels_to_ops(
            profile.trace_events,
            profile.ops,
            profile.kernels,
            flow_tolerance_us,
        )

    _reconcile_kernel_owners_by_detail_type(profile)
    _reconcile_kernel_owners_by_api_family(profile)


def _reconcile_kernel_owners_by_detail_type(profile: ParsedProfile) -> None:
    """Repair ambiguous NPU flow ownership with kernel_details type order.

    Ascend flow starts may occur inside a nested helper aten call. Choosing the
    narrowest containing event can therefore attach an ``Index`` task to a
    nearby ``reshape``/``slice`` instead of the enclosing ``aten::index``.  If
    an aten type and an exact kernel-details type have equal cardinality, and
    existing flow links provide enough ordered evidence, the two sequences are
    paired by occurrence order.  The original flow owner is retained on the
    kernel for diagnostics.
    """
    ops_by_type: Dict[str, List[int]] = defaultdict(list)
    kernels_by_type: Dict[str, List[int]] = defaultdict(list)
    for op_index, op in enumerate(profile.ops):
        key = _operator_type_key(op.canonical_name)
        if key:
            ops_by_type[key].append(op_index)
    for kernel_index, kernel in enumerate(profile.kernels):
        if kernel.detail is None or not kernel.detail.op_type:
            continue
        key = _operator_type_key(kernel.detail.op_type)
        if key:
            kernels_by_type[key].append(kernel_index)

    reconciled_groups: Dict[str, Dict[str, int]] = {}
    moved_kernel_count = 0
    for key in sorted(set(ops_by_type) & set(kernels_by_type)):
        op_indices = sorted(
            ops_by_type[key], key=lambda index: profile.ops[index].event.timestamp_us
        )
        kernel_indices = sorted(
            kernels_by_type[key],
            key=lambda index: profile.kernels[index].event.timestamp_us,
        )
        if not op_indices or len(op_indices) != len(kernel_indices):
            continue

        op_rank = {op_index: rank for rank, op_index in enumerate(op_indices)}
        evidence: List[Tuple[int, int]] = []
        for kernel_rank, kernel_index in enumerate(kernel_indices):
            kernel = profile.kernels[kernel_index]
            owner = kernel.owner_op_index
            if owner in op_rank and kernel.link_method == "flow":
                evidence.append((kernel_rank, op_rank[owner]))

        group_size = len(op_indices)
        minimum_evidence = 1 if group_size <= 3 else max(2, math.ceil(group_size * 0.25))
        if len(evidence) < minimum_evidence:
            continue
        order_tolerance = max(1, math.ceil(group_size * 0.02))
        ordered_evidence = sum(
            abs(kernel_rank - owner_rank) <= order_tolerance
            for kernel_rank, owner_rank in evidence
        )
        if ordered_evidence / len(evidence) < 0.8:
            continue

        # External-id and runtime-correlation ownership are stronger than this
        # sequence-level repair. Never override them merely because counts
        # happen to be equal.
        has_strong_conflict = any(
            profile.kernels[kernel_index].owner_op_index != target_op_index
            and profile.kernels[kernel_index].link_method
            not in {"flow", "unmatched", "detail_type_order"}
            for target_op_index, kernel_index in zip(op_indices, kernel_indices)
        )
        if has_strong_conflict:
            continue

        group_moved = 0
        for target_op_index, kernel_index in zip(op_indices, kernel_indices):
            kernel = profile.kernels[kernel_index]
            previous_owner = kernel.owner_op_index
            if previous_owner == target_op_index:
                continue
            if kernel.original_link_method == "":
                kernel.original_owner_op_index = previous_owner
                kernel.original_link_method = kernel.link_method
                kernel.original_link_scope = kernel.link_scope
                kernel.original_link_distance_us = kernel.link_distance_us
            if previous_owner is not None:
                profile.ops[previous_owner].kernel_indices = [
                    index
                    for index in profile.ops[previous_owner].kernel_indices
                    if index != kernel_index
                ]
            kernel.owner_op_index = target_op_index
            kernel.link_method = "detail_type_order"
            kernel.link_scope = "equal_type_count"
            kernel.link_distance_us = None
            profile.ops[target_op_index].kernel_indices.append(kernel_index)
            group_moved += 1

        if group_moved:
            for op_index in op_indices:
                profile.ops[op_index].kernel_indices.sort(
                    key=lambda index: profile.kernels[index].event.timestamp_us
                )
            moved_kernel_count += group_moved
            reconciled_groups[key] = {
                "count": group_size,
                "flow_evidence_count": len(evidence),
                "ordered_evidence_count": ordered_evidence,
                "moved_kernel_count": group_moved,
            }

    profile.kernel_owner_reconciliation = {
        "group_count": len(reconciled_groups),
        "moved_kernel_count": moved_kernel_count,
        "groups": reconciled_groups,
    }


def _reconcile_kernel_owners_by_api_family(profile: ParsedProfile) -> None:
    """Use the ACLNN API prefix when kernel Type and aten name are aliases.

    Examples include ``aclnnClamp_* / ClipByValueV2 -> aten::clamp`` and
    ``aclnnAddmm_* / MatMulV3 -> aten::addmm``. Reassignment is allowed only
    when the API-family kernel count exactly equals the target aten call count
    and existing flow links confirm a stable occurrence order. The explicit
    Stack/ConcatD family also permits an exact-count, time-local order repair:
    its flow links can all point to neighboring reduction calls.
    """
    ops_by_name: Dict[str, List[int]] = defaultdict(list)
    kernels_by_target: Dict[str, List[int]] = defaultdict(list)
    for op_index, op in enumerate(profile.ops):
        ops_by_name[op.canonical_name].append(op_index)
    for kernel_index, kernel in enumerate(profile.kernels):
        target = _kernel_api_target_op(kernel)
        if target:
            kernels_by_target[target].append(kernel_index)

    reconciled_groups: Dict[str, Dict[str, int]] = {}
    moved_kernel_count = 0
    for target in sorted(set(ops_by_name) & set(kernels_by_target)):
        op_indices = sorted(
            ops_by_name[target], key=lambda index: profile.ops[index].event.timestamp_us
        )
        kernel_indices = sorted(
            kernels_by_target[target],
            key=lambda index: profile.kernels[index].event.timestamp_us,
        )
        if not op_indices or len(op_indices) != len(kernel_indices):
            continue

        op_rank = {op_index: rank for rank, op_index in enumerate(op_indices)}
        evidence: List[Tuple[int, int]] = []
        for kernel_rank, kernel_index in enumerate(kernel_indices):
            kernel = profile.kernels[kernel_index]
            owner = kernel.owner_op_index
            was_flow_linked = (
                kernel.link_method == "flow"
                or kernel.original_link_method == "flow"
            )
            if owner in op_rank and was_flow_linked:
                evidence.append((kernel_rank, op_rank[owner]))

        group_size = len(op_indices)
        minimum_evidence = 1 if group_size <= 3 else max(2, math.ceil(group_size * 0.25))
        order_only_stack = (
            target == "aten::stack"
            and len(evidence) < minimum_evidence
            and len({profile.kernels[index].stream_id for index in kernel_indices}) == 1
            and all(
                abs(
                    profile.kernels[kernel_index].event.timestamp_us
                    - profile.ops[op_index].event.timestamp_us
                )
                <= float(profile.flow_link_diagnostics.get("flow_tolerance_us", 1000.0))
                for op_index, kernel_index in zip(op_indices, kernel_indices)
            )
        )
        if len(evidence) < minimum_evidence and not order_only_stack:
            continue
        order_tolerance = max(1, math.ceil(group_size * 0.02))
        ordered_evidence = sum(
            abs(kernel_rank - owner_rank) <= order_tolerance
            for kernel_rank, owner_rank in evidence
        )
        if evidence and ordered_evidence / len(evidence) < 0.8:
            continue

        has_strong_conflict = any(
            profile.kernels[kernel_index].owner_op_index != target_op_index
            and profile.kernels[kernel_index].link_method
            not in {"flow", "unmatched", "detail_type_order", "kernel_api_order"}
            for target_op_index, kernel_index in zip(op_indices, kernel_indices)
        )
        if has_strong_conflict:
            continue

        group_moved = 0
        for target_op_index, kernel_index in zip(op_indices, kernel_indices):
            kernel = profile.kernels[kernel_index]
            previous_owner = kernel.owner_op_index
            if previous_owner == target_op_index:
                continue
            if kernel.original_link_method == "":
                kernel.original_owner_op_index = previous_owner
                kernel.original_link_method = kernel.link_method
                kernel.original_link_scope = kernel.link_scope
                kernel.original_link_distance_us = kernel.link_distance_us
            if previous_owner is not None:
                profile.ops[previous_owner].kernel_indices = [
                    index
                    for index in profile.ops[previous_owner].kernel_indices
                    if index != kernel_index
                ]
            kernel.owner_op_index = target_op_index
            kernel.link_method = "kernel_api_order"
            kernel.link_scope = f"api_family:{target}"
            kernel.link_distance_us = None
            profile.ops[target_op_index].kernel_indices.append(kernel_index)
            group_moved += 1

        if group_moved:
            for op_index in op_indices:
                profile.ops[op_index].kernel_indices.sort(
                    key=lambda index: profile.kernels[index].event.timestamp_us
                )
            moved_kernel_count += group_moved
            reconciled_groups[target] = {
                "count": group_size,
                "flow_evidence_count": len(evidence),
                "ordered_evidence_count": ordered_evidence,
                "order_only_stack": order_only_stack,
                "moved_kernel_count": group_moved,
            }

    existing = profile.kernel_owner_reconciliation
    existing["api_group_count"] = len(reconciled_groups)
    existing["api_moved_kernel_count"] = moved_kernel_count
    existing["api_groups"] = reconciled_groups
    existing["moved_kernel_count"] = (
        int(existing.get("moved_kernel_count", 0)) + moved_kernel_count
    )

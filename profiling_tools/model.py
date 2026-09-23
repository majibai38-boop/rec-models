from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class TraceEvent:
    index: int
    name: str
    category: str
    phase: str
    timestamp_us: float
    duration_us: float
    pid: str
    tid: str
    args: Dict[str, Any]

    @property
    def end_us(self) -> float:
        return self.timestamp_us + self.duration_us


@dataclass
class KernelDetail:
    row_index: int
    name: str = ""
    op_type: str = ""
    core_type: str = ""
    task_id: str = ""
    stream_id: str = ""
    device_id: str = ""
    step_id: str = ""
    start_us: Optional[float] = None
    duration_us: Optional[float] = None
    input_shapes: str = ""
    input_dtypes: str = ""
    output_shapes: str = ""
    output_dtypes: str = ""
    raw: Dict[str, str] = field(default_factory=dict)


@dataclass
class KernelRecord:
    event: TraceEvent
    external_id: str = ""
    correlation_id: str = ""
    task_id: str = ""
    stream_id: str = ""
    detail: Optional[KernelDetail] = None
    owner_op_index: Optional[int] = None
    link_method: str = "unmatched"


@dataclass
class OpRecord:
    event: TraceEvent
    canonical_name: str
    external_id: str = ""
    input_shapes: str = ""
    input_dtypes: str = ""
    call_stack: str = ""
    module_path: str = ""
    parent_index: Optional[int] = None
    kernel_indices: List[int] = field(default_factory=list)


@dataclass
class ParsedProfile:
    side: str
    source_path: str
    ops: List[OpRecord]
    kernels: List[KernelRecord]
    selected_pid: str
    selected_step: str
    step_names: List[str]
    total_trace_events: int
    kernel_detail_count: int = 0
    kernel_detail_matched_count: int = 0


@dataclass
class BundleMetrics:
    kernel_count: int
    sum_us: float
    busy_us: float
    span_us: float


@dataclass
class OpMatch:
    match_id: int
    gpu_op_index: int
    npu_op_index: int
    score: float
    confidence: str


def interval_union_us(intervals: Sequence[Tuple[float, float]]) -> float:
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    total = 0.0
    current_start, current_end = valid[0]
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def bundle_metrics(kernels: Sequence[KernelRecord]) -> BundleMetrics:
    if not kernels:
        return BundleMetrics(kernel_count=0, sum_us=0.0, busy_us=0.0, span_us=0.0)
    intervals = [(kernel.event.timestamp_us, kernel.event.end_us) for kernel in kernels]
    return BundleMetrics(
        kernel_count=len(kernels),
        sum_us=sum(kernel.event.duration_us for kernel in kernels),
        busy_us=interval_union_us(intervals),
        span_us=max(end for _, end in intervals) - min(start for start, _ in intervals),
    )

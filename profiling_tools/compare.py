from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .matcher import load_op_map, match_ops
from .parser import attach_kernel_details, load_kernel_details, parse_trace
from .report import write_reports


@dataclass(frozen=True)
class CompareConfig:
    gpu_trace: str
    npu_trace: str
    npu_kernel_details: str
    output_dir: str
    gpu_step: str = ""
    npu_step: str = ""
    gpu_pid: str = ""
    npu_pid: str = ""
    op_prefix: str = "aten::"
    op_map: str = ""
    include_communication: bool = False


def compare_profiles(config: CompareConfig) -> Dict[str, Any]:
    gpu = parse_trace(
        config.gpu_trace,
        "gpu",
        op_prefix=config.op_prefix,
        requested_pid=config.gpu_pid,
        requested_step=config.gpu_step,
        include_communication=config.include_communication,
    )
    npu = parse_trace(
        config.npu_trace,
        "npu",
        op_prefix=config.op_prefix,
        requested_pid=config.npu_pid,
        requested_step=config.npu_step,
        include_communication=config.include_communication,
    )
    details = load_kernel_details(config.npu_kernel_details, config.npu_step)
    attach_kernel_details(npu, details)
    op_map = load_op_map(config.op_map)
    matches, unmatched_gpu, unmatched_npu = match_ops(gpu, npu, op_map)
    return write_reports(
        config.output_dir,
        gpu,
        npu,
        matches,
        unmatched_gpu,
        unmatched_npu,
        op_map,
    )

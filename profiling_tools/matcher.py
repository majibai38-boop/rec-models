from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .model import OpMatch, OpRecord, ParsedProfile
from .parser import canonical_op_name


def load_op_map(path: str) -> Dict[str, str]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Operator map must be a JSON object: {gpu_name: npu_name}")
    return {
        canonical_op_name(str(gpu_name)): canonical_op_name(str(npu_name))
        for gpu_name, npu_name in payload.items()
    }


def _mapped_gpu_names(ops: Sequence[OpRecord], op_map: Mapping[str, str]) -> List[str]:
    return [op_map.get(op.canonical_name, op.canonical_name) for op in ops]


def _suffix_similarity(left: str, right: str) -> bool:
    if not left or not right:
        return False
    left_tail = left.replace("\\", "/").split("/")[-1]
    right_tail = right.replace("\\", "/").split("/")[-1]
    return left_tail == right_tail


def match_score(
    gpu_op: OpRecord,
    npu_op: OpRecord,
    gpu_index: int,
    npu_index: int,
    gpu_count: int,
    npu_count: int,
    op_map: Mapping[str, str],
) -> Tuple[float, str]:
    mapped_name = op_map.get(gpu_op.canonical_name, gpu_op.canonical_name)
    if mapped_name != npu_op.canonical_name:
        return 0.0, "none"

    score = 0.45
    if gpu_op.input_shapes and npu_op.input_shapes and gpu_op.input_shapes == npu_op.input_shapes:
        score += 0.20
    if gpu_op.input_dtypes and npu_op.input_dtypes and gpu_op.input_dtypes == npu_op.input_dtypes:
        score += 0.05
    if gpu_op.module_path and npu_op.module_path:
        if gpu_op.module_path == npu_op.module_path:
            score += 0.15
        elif _suffix_similarity(gpu_op.module_path, npu_op.module_path):
            score += 0.08
    if gpu_op.call_stack and npu_op.call_stack:
        if gpu_op.call_stack == npu_op.call_stack:
            score += 0.10
        elif _suffix_similarity(gpu_op.call_stack, npu_op.call_stack):
            score += 0.05

    gpu_position = gpu_index / max(gpu_count - 1, 1)
    npu_position = npu_index / max(npu_count - 1, 1)
    score += 0.05 * max(0.0, 1.0 - abs(gpu_position - npu_position) * 4.0)
    score = min(score, 1.0)
    confidence = "high" if score >= 0.80 else "medium" if score >= 0.60 else "low"
    return score, confidence


def match_ops(
    gpu: ParsedProfile, npu: ParsedProfile, op_map: Mapping[str, str]
) -> Tuple[List[OpMatch], List[int], List[int]]:
    gpu_names = _mapped_gpu_names(gpu.ops, op_map)
    npu_names = [op.canonical_name for op in npu.ops]
    matcher = difflib.SequenceMatcher(a=gpu_names, b=npu_names, autojunk=False)

    matches: List[OpMatch] = []
    used_gpu = set()
    used_npu = set()
    next_match_id = 1
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            gpu_index = block.a + offset
            npu_index = block.b + offset
            score, confidence = match_score(
                gpu.ops[gpu_index],
                npu.ops[npu_index],
                gpu_index,
                npu_index,
                len(gpu.ops),
                len(npu.ops),
                op_map,
            )
            matches.append(
                OpMatch(
                    match_id=next_match_id,
                    gpu_op_index=gpu_index,
                    npu_op_index=npu_index,
                    score=score,
                    confidence=confidence,
                )
            )
            next_match_id += 1
            used_gpu.add(gpu_index)
            used_npu.add(npu_index)

    return (
        matches,
        [index for index in range(len(gpu.ops)) if index not in used_gpu],
        [index for index in range(len(npu.ops)) if index not in used_npu],
    )

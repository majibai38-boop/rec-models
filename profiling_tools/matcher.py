from __future__ import annotations

import bisect
import difflib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .model import OpMatch, OpRecord, ParsedProfile
from .parser import canonical_op_name, CUSTOM_COMPARABLE_OPS
from .call_metadata import compare_metadata
from .logical_recovery import recover_layernorm_blocks


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


def _semantic_token(op: OpRecord, mapped_name: str) -> Tuple[str, str, str, str]:
    return (mapped_name, op.input_shapes, op.input_dtypes, op.module_key)


def _unique_semantic_anchors(
    gpu_tokens: Sequence[Tuple[str, str, str, str]],
    npu_tokens: Sequence[Tuple[str, str, str, str]],
) -> List[Tuple[int, int]]:
    """Return a monotonic LIS of tokens that occur exactly once on each side."""
    gpu_positions: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    npu_positions: Dict[Tuple[str, str, str, str], List[int]] = defaultdict(list)
    for index, token in enumerate(gpu_tokens):
        gpu_positions[token].append(index)
    for index, token in enumerate(npu_tokens):
        npu_positions[token].append(index)
    candidates = sorted(
        (gpu_indices[0], npu_positions[token][0])
        for token, gpu_indices in gpu_positions.items()
        if len(gpu_indices) == 1 and len(npu_positions.get(token, [])) == 1
    )
    if not candidates:
        return []

    tails: List[int] = []
    tail_candidate_indices: List[int] = []
    previous = [-1] * len(candidates)
    for candidate_index, (_, npu_index) in enumerate(candidates):
        position = bisect.bisect_left(tails, npu_index)
        if position == len(tails):
            tails.append(npu_index)
            tail_candidate_indices.append(candidate_index)
        else:
            tails[position] = npu_index
            tail_candidate_indices[position] = candidate_index
        if position > 0:
            previous[candidate_index] = tail_candidate_indices[position - 1]

    result: List[Tuple[int, int]] = []
    cursor = tail_candidate_indices[-1]
    while cursor >= 0:
        result.append(candidates[cursor])
        cursor = previous[cursor]
    result.reverse()
    return result


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

    score = 0.35
    if gpu_op.input_shapes and npu_op.input_shapes and gpu_op.input_shapes == npu_op.input_shapes:
        score += 0.30
    if gpu_op.input_dtypes and npu_op.input_dtypes and gpu_op.input_dtypes == npu_op.input_dtypes:
        score += 0.10
    if gpu_op.module_key and npu_op.module_key:
        if gpu_op.module_key == npu_op.module_key:
            score += 0.15
        elif _suffix_similarity(gpu_op.module_key, npu_op.module_key):
            score += 0.08
    if gpu_op.call_stack and npu_op.call_stack:
        if gpu_op.call_stack == npu_op.call_stack:
            score += 0.05
        elif _suffix_similarity(gpu_op.call_stack, npu_op.call_stack):
            score += 0.03

    gpu_position = gpu_index / max(gpu_count - 1, 1)
    npu_position = npu_index / max(npu_count - 1, 1)
    score += 0.05 * max(0.0, 1.0 - abs(gpu_position - npu_position) * 4.0)
    score = min(score, 1.0)
    confidence = "high" if score >= 0.75 else "medium" if score >= 0.55 else "low"
    return score, confidence


def _repair_logical_calls(
    gpu: ParsedProfile, npu: ParsedProfile, pairs: Sequence[Tuple[int, int, str]]
) -> List[Tuple[int, int, str]]:
    """Reconcile selected overloaded calls within a module and parent scope.

    Nested sum/arange overloads are implementation children, not additional
    logical invocations. Exact metadata buckets allow independent index_put
    calls to change order without matching boolean writes to float writes.
    Never touch copy_/to (Cast) or relax a metadata mismatch.
    """
    names = {"aten::sum", "aten::arange", "aten::fill_", "aten::index_put_",
             "aten::_index_put_impl_"}

    def descriptor(profile, index):
        op = profile.ops[index]
        if op.canonical_name not in names or not op.module_key:
            return None
        parent = profile.ops[op.parent_index] if op.parent_index is not None else None
        if (op.canonical_name in {"aten::sum", "aten::arange"}
                and parent is not None and parent.canonical_name == op.canonical_name):
            return ("nested",)
        context = parent.canonical_name if parent else ""
        try:
            shapes, dtypes = json.loads(op.input_shapes), json.loads(op.input_dtypes)
        except (ValueError, TypeError):
            return None
        if not shapes or not dtypes:
            return None
        # Preserve every argument/overload, not just the first tensor. CUDA
        # uses an empty type for optional None while Ascend spells it "none".
        metadata = (op.input_shapes, json.dumps([value or "none" for value in dtypes]))
        return (op.canonical_name, op.module_key, context, metadata)

    descriptors = [[descriptor(profile, i) for i in range(len(profile.ops))]
                   for profile in (gpu, npu)]
    groups = []
    for keys in descriptors:
        buckets = defaultdict(list)
        for i, key in enumerate(keys):
            if key and key != ("nested",):
                buckets[key].append(i)
        groups.append(buckets)
    recovered = []
    repaired_gpu, repaired_npu = set(), set()
    for key in groups[0].keys() & groups[1].keys():
        left, right = groups[0][key], groups[1][key]
        if len(left) != len(right):
            continue
        recovered.extend((g, n, "logical_call_metadata") for g, n in zip(left, right))
        repaired_gpu.update(left)
        repaired_npu.update(right)
    kept = []
    for g, n, basis in pairs:
        if g in repaired_gpu or n in repaired_npu:
            continue
        a, b = descriptors[0][g], descriptors[1][n]
        # Explicitly contradictory parent/shape information must not be
        # promoted by module-name agreement alone.
        if a and b and a != b:
            continue
        kept.append((g, n, basis))
    return kept + recovered


def _recover_slice_copies(gpu, npu, pairs):
    """Recover unmatched clone/copy calls only for NPU Slice, never Cast."""
    used = [{p[side] for p in pairs} for side in (0, 1)]
    buckets = []
    for side, profile in enumerate((gpu, npu)):
        groups = defaultdict(list)
        for i, op in enumerate(profile.ops):
            if i in used[side] or op.canonical_name != "aten::copy_" or not op.module_key:
                continue
            if op.parent_index is None or profile.ops[op.parent_index].canonical_name != "aten::clone":
                continue
            if not op.input_shapes or not op.input_dtypes or not op.kernel_indices:
                continue
            if side == 1 and not all(profile.kernels[k].detail is not None
                    and profile.kernels[k].detail.op_type == "Slice" for k in op.kernel_indices):
                continue
            parent = profile.ops[op.parent_index]
            grandparent = profile.ops[parent.parent_index].canonical_name if parent.parent_index is not None else ""
            groups[(op.module_key, op.input_shapes, op.input_dtypes, grandparent)].append(i)
        buckets.append(groups)
    recovered = []
    for key in buckets[0].keys() & buckets[1].keys():
        # Unique residual call per semantic parent scope. Repeated ambiguity
        # is left unresolved instead of using global copy_ cardinality.
        if len(buckets[0][key]) == len(buckets[1][key]) == 1:
            recovered.append((buckets[0][key][0], buckets[1][key][0], "noncast_clone_copy_recovery"))
    return list(pairs) + recovered


def match_ops(
    gpu: ParsedProfile, npu: ParsedProfile, op_map: Mapping[str, str]
) -> Tuple[List[OpMatch], List[int], List[int]]:
    gpu_names = _mapped_gpu_names(gpu.ops, op_map)
    npu_names = [op.canonical_name for op in npu.ops]
    gpu_tokens = [_semantic_token(op, gpu_names[index]) for index, op in enumerate(gpu.ops)]
    npu_tokens = [_semantic_token(op, npu_names[index]) for index, op in enumerate(npu.ops)]

    semantic_anchors = _unique_semantic_anchors(gpu_tokens, npu_tokens)

    paired_indices: List[Tuple[int, int, str]] = []
    previous_gpu = -1
    previous_npu = -1
    for anchor_gpu, anchor_npu in semantic_anchors + [(len(gpu.ops), len(npu.ops))]:
        gpu_start = previous_gpu + 1
        npu_start = previous_npu + 1
        gpu_gap = gpu_names[gpu_start:anchor_gpu]
        npu_gap = npu_names[npu_start:anchor_npu]
        if gpu_gap == npu_gap:
            fallback_blocks = [(0, 0, len(gpu_gap))]
        elif not gpu_gap or not npu_gap:
            fallback_blocks = []
        else:
            fallback = difflib.SequenceMatcher(
                a=gpu_gap,
                b=npu_gap,
                autojunk=False,
            )
            fallback_blocks = [
                (block.a, block.b, block.size)
                for block in fallback.get_matching_blocks()
                if block.size
            ]
        for block_gpu, block_npu, block_size in fallback_blocks:
            for offset in range(block_size):
                paired_indices.append(
                    (
                        gpu_start + block_gpu + offset,
                        npu_start + block_npu + offset,
                        "name_sequence",
                    )
                )
        if anchor_gpu < len(gpu.ops) and anchor_npu < len(npu.ops):
            paired_indices.append((anchor_gpu, anchor_npu, "semantic_anchor"))
        previous_gpu = anchor_gpu
        previous_npu = anchor_npu

    # Sequence anchors can strand a few repeated ops when one backend inserts
    # helper calls between otherwise equivalent occurrences. If the total
    # cardinality of a canonical op is identical on both sides, recover only
    # the still-unmatched occurrences in occurrence order.
    paired_gpu = {gpu_index for gpu_index, _, _ in paired_indices}
    paired_npu = {npu_index for _, npu_index, _ in paired_indices}
    gpu_name_counts = Counter(gpu_names)
    npu_name_counts = Counter(npu_names)
    unmatched_gpu_by_name: Dict[str, List[int]] = defaultdict(list)
    unmatched_npu_by_name: Dict[str, List[int]] = defaultdict(list)
    for index, name in enumerate(gpu_names):
        if index not in paired_gpu:
            unmatched_gpu_by_name[name].append(index)
    for index, name in enumerate(npu_names):
        if index not in paired_npu:
            unmatched_npu_by_name[name].append(index)
    for name in sorted(set(unmatched_gpu_by_name) & set(unmatched_npu_by_name)):
        gpu_remaining = unmatched_gpu_by_name[name]
        npu_remaining = unmatched_npu_by_name[name]
        if (
            gpu_name_counts[name] == npu_name_counts[name]
            and len(gpu_remaining) == len(npu_remaining)
        ):
            paired_indices.extend(
                (gpu_index, npu_index, "same_name_recovery")
                for gpu_index, npu_index in zip(gpu_remaining, npu_remaining)
            )

    if not op_map:
        paired_indices = _repair_logical_calls(gpu, npu, paired_indices)
        paired_indices = _recover_slice_copies(gpu, npu, paired_indices)
        paired_indices = recover_layernorm_blocks(gpu, npu, paired_indices, semantic_anchors)
    paired_indices.sort(key=lambda item: (item[0], item[1]))

    matches: List[OpMatch] = []
    custom_names = {"nestedconcatfunc", "position_concat_fwd_kernel",
                    "create_position_offset_fwd_kernel"}
    custom_positions = []
    strict_custom_positions = []
    for profile in (gpu, npu):
        groups = defaultdict(list)
        strict_groups = defaultdict(list)
        for index, op in enumerate(profile.ops):
            if op.canonical_name in custom_names:
                groups[(op.canonical_name, op.module_key)].append(index)
            if op.canonical_name in CUSTOM_COMPARABLE_OPS:
                strict_groups[(op.canonical_name, op.module_key)].append(index)
        custom_positions.append(groups)
        strict_custom_positions.append(strict_groups)
    used_gpu = set()
    used_npu = set()
    next_match_id = 1
    for gpu_index, npu_index, alignment_basis in paired_indices:
        score, confidence = match_score(
            gpu.ops[gpu_index],
            npu.ops[npu_index],
            gpu_index,
            npu_index,
            len(gpu.ops),
            len(npu.ops),
            op_map,
        )
        mapped_name = op_map.get(
            gpu.ops[gpu_index].canonical_name,
            gpu.ops[gpu_index].canonical_name,
        )
        if alignment_basis == "layernorm_block_metadata":
            score, confidence = max(score, 0.75), "high"
        custom_key = (mapped_name, gpu.ops[gpu_index].module_key)
        gpu_custom = custom_positions[0].get(custom_key, [])
        npu_custom = custom_positions[1].get(custom_key, [])
        if (mapped_name in custom_names
                and gpu.ops[gpu_index].module_key
                and gpu.ops[gpu_index].module_key == npu.ops[npu_index].module_key
                and len(gpu_custom) == len(npu_custom)
                and (gpu_index, npu_index) in zip(gpu_custom, npu_custom)
                and all(not left or not right or left == right for left, right in (
                    (gpu.ops[gpu_index].input_shapes, npu.ops[npu_index].input_shapes),
                    (gpu.ops[gpu_index].input_dtypes, npu.ops[npu_index].input_dtypes),
                ))
                and len(gpu.ops[gpu_index].kernel_indices) == 1
                and len(npu.ops[npu_index].kernel_indices) == 1
                and gpu.kernels[gpu.ops[gpu_index].kernel_indices[0]].event.name
                == npu.kernels[npu.ops[npu_index].kernel_indices[0]].event.name):
            score, confidence = max(score, 0.75), "high"
            alignment_basis = "custom_scope_same_kernel"
        elif mapped_name in custom_names:
            score, confidence = min(score, 0.54), "low"
        if mapped_name in CUSTOM_COMPARABLE_OPS:
            left = strict_custom_positions[0].get(custom_key, [])
            right = strict_custom_positions[1].get(custom_key, [])
            # Validate the entire occurrence sequence, not just one convenient
            # pair. Missing calls or conflicting metadata cannot be promoted.
            compatible = bool(left and len(left) == len(right)) and all(
                compare_metadata(npu.ops[n], gpu.ops[g]) == ([], [])
                for g, n in zip(left, right)
            )
            if compatible and (gpu_index, npu_index) in zip(left, right):
                score, confidence = max(score, 0.75), "high"
                alignment_basis = "custom_scope_metadata_sequence"
            else:
                score, confidence = min(score, 0.54), "low"
        if (
            confidence == "low"
            and mapped_name not in custom_names | CUSTOM_COMPARABLE_OPS
            and mapped_name == npu.ops[npu_index].canonical_name
            and alignment_basis == "name_sequence"
            and gpu.ops[gpu_index].module_key
            and gpu.ops[gpu_index].module_key == npu.ops[npu_index].module_key
            and gpu.ops[gpu_index].kernel_indices
            and npu.ops[npu_index].kernel_indices
        ):
            # Some exporters report scalar-filled tensor metadata differently
            # (for example FLOAT tensor vs BOOL result buffer). A direct
            # same-name sequence match, the same module and kernels on both
            # sides provide enough evidence for a medium-confidence comparison
            # even when one backend emits additional helper calls of that name.
            score = max(score, 0.55)
            confidence = "medium"
            alignment_basis = f"{alignment_basis}_module_metadata"
        matches.append(
            OpMatch(
                match_id=next_match_id,
                gpu_op_index=gpu_index,
                npu_op_index=npu_index,
                score=score,
                confidence=confidence,
                alignment_basis=alignment_basis,
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

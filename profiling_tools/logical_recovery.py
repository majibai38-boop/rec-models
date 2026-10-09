"""Conservative local recovery of complete LayerNorm primitive sequences.

Module numeric suffixes are capture-local. They are not discarded globally:
only unique, complete metadata sequences between the same trusted anchors
establish a correspondence. Repeated/partial sequences remain unresolved.
"""
import bisect
from collections import defaultdict
import json
import re

from .call_metadata import compare_metadata, dtypes, shapes


_PRIMITIVES = {"aten::mean", "aten::var", "aten::sub", "aten::add", "aten::div",
               "aten::mul", "aten::pow", "aten::sqrt", "aten::rsqrt"}


def _layernorm(op):
    return bool(re.fullmatch(r"layernorm_\d+", op.module_key))


def _blocks(profile, anchor_indices):
    blocks = []
    for i, op in enumerate(profile.ops):
        if not _layernorm(op):
            continue
        if not blocks or blocks[-1][-1] != i - 1 or profile.ops[blocks[-1][0]].module_key != op.module_key:
            blocks.append([])
        blocks[-1].append(i)
    result = defaultdict(list)
    for block in blocks:
        # Both ends must lie strictly inside the same paired-anchor interval.
        slot = bisect.bisect_left(anchor_indices, block[0])
        if slot == 0 or slot == len(anchor_indices) or bisect.bisect_left(anchor_indices, block[-1]) != slot:
            continue
        selected, fingerprint = [], []
        for i in block:
            op = profile.ops[i]
            if op.canonical_name not in _PRIMITIVES:
                continue
            parent = profile.ops[op.parent_index] if op.parent_index is not None else None
            if parent and parent.canonical_name == op.canonical_name:
                continue  # Nested overload is not an independent invocation.
            if compare_metadata(op, op) != ([], []):
                selected = []
                break
            selected.append(i)
            fingerprint.append((op.canonical_name,
                json.dumps(shapes(op.raw_input_shapes or op.input_shapes)),
                tuple(dtypes(op.raw_input_dtypes or op.input_dtypes))))
        names = {profile.ops[i].canonical_name for i in selected}
        if len(selected) >= 4 and len(names) >= 3 and {"aten::mean", "aten::var"} <= names:
            result[(slot, tuple(fingerprint))].append((block, selected))
    return result


def recover_layernorm_blocks(gpu, npu, pairs, anchors):
    anchors = [(g, n) for g, n in anchors if not _layernorm(gpu.ops[g]) and not _layernorm(npu.ops[n])]
    if len(anchors) < 2:
        return list(pairs)
    left = _blocks(gpu, [g for g, _ in anchors])
    right = _blocks(npu, [n for _, n in anchors])
    recovered, replaced_g, replaced_n = [], set(), set()
    for key in left.keys() & right.keys():
        if len(left[key]) != 1 or len(right[key]) != 1:
            continue
        gb, gs = left[key][0]
        nb, ns = right[key][0]
        proposed = set(zip(gs, ns))
        # Never take an endpoint away from an already metadata-consistent pair.
        if any((g in gb or n in nb) and (g, n) not in proposed
               and compare_metadata(npu.ops[n], gpu.ops[g]) == ([], [])
               and gpu.ops[g].canonical_name in _PRIMITIVES
               for g, n, _ in pairs):
            continue
        recovered.extend((g, n, "layernorm_block_metadata") for g, n in zip(gs, ns))
        replaced_g.update(i for i in gb if gpu.ops[i].canonical_name in _PRIMITIVES)
        replaced_n.update(i for i in nb if npu.ops[i].canonical_name in _PRIMITIVES)
    kept = []
    for g, n, basis in pairs:
        if g in replaced_g or n in replaced_n:
            continue
        # Reject clearly contradictory same-name fallback inside this family;
        # lack of a unique structural recovery is not permission to guess.
        if (basis == "same_name_recovery" and _layernorm(gpu.ops[g]) and _layernorm(npu.ops[n])
                and gpu.ops[g].canonical_name in _PRIMITIVES
                and compare_metadata(npu.ops[n], gpu.ops[g])[0]):
            continue
        kept.append((g, n, basis))
    return kept + recovered

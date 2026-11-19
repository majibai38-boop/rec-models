"""Additional report-only scopes under an established CPU pair."""
from collections import Counter, defaultdict

from .call_metadata import compare_metadata
from .model import OpMatch
from .whole_calls import build_whole_calls


def child_candidates(gpu, npu, matches, mapping_rows, op_map):
    """Pair a uniquely named immediate child on each side of an aligned parent.

    Existing CPU ownership is authoritative. Repeated siblings and incomplete
    outer metadata cannot establish an additional match by execution order.
    """
    def children(profile):
        result = defaultdict(lambda: defaultdict(list))
        for i, op in enumerate(profile.ops):
            if op.parent_index is not None:
                name = (op_map or {}).get(op.canonical_name, op.canonical_name) if profile is gpu else op.canonical_name
                result[op.parent_index][name].append(i)
        return result

    nc, gc = children(npu), children(gpu)
    used_n = Counter(m.npu_op_index for m in matches)
    used_g = Counter(m.gpu_op_index for m in matches)
    found, parents = [], {}
    for match in matches:
        ni, gi = match.npu_op_index, match.gpu_op_index
        if match.confidence not in {"high", "medium"} or used_n[ni] != 1 or used_g[gi] != 1:
            continue
        if not nc.get(ni) or not gc.get(gi) or any(compare_metadata(npu.ops[ni], gpu.ops[gi], op_map)):
            continue
        for name, indices in nc[ni].items():
            siblings = gc[gi].get(name, [])
            if len(indices) != 1 or len(siblings) != 1:
                continue
            n, g = indices[0], siblings[0]
            if n in used_n or g in used_g or any(compare_metadata(npu.ops[n], gpu.ops[g], op_map)):
                continue
            found.append(OpMatch(len(found), g, n, 1, "high", "unique_child_of_matched_parent"))
            parents[n] = {"match_id": match.match_id, "npu_cpu": ni, "gpu_cpu": gi}
    if not found:
        return []
    rows = build_whole_calls(gpu, npu, found, mapping_rows, [], [], op_map)
    for row in rows:
        row["Source"] = "unique_child_of_matched_parent"
        row["Parent Anchor"] = parents[row["NPU CPU OP Index"]]
        row["Module"] = npu.ops[row["NPU CPU OP Index"]].module_key
        for side, profile in (("NPU", npu), ("GPU", gpu)):
            op = profile.ops[row[side + " CPU OP Index"]]
            row[side + " Input Shapes"] = op.raw_input_shapes or op.input_shapes
            row[side + " Input DTypes"] = op.raw_input_dtypes or op.input_dtypes
    return rows

"""Align batched MatMul gradients across different autograd call boundaries."""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from math import prod

from .call_metadata import compare_metadata, dtypes, shapes
from .device_compare import scoped_indices
from .model import TraceEvent
from .parser import get_arg


BACKWARD_OP = "aten::matmul_backward"
BACKWARD_SOURCE = "matmul_backward_scope"
GPU_LINKS = {"external_id", "runtime_correlation"}
NPU_LINKS = GPU_LINKS | {"torch_to_npu", "host_to_device_queue", "host_to_device_thread"}
FLOAT_TYPES = {"float16", "bfloat16", "float32", "float64"}


def _dims(value):
    return bool(value) and all(type(n) is int and n > 0 for n in value)


def _gemm(input_shapes, input_types, output_shapes=""):
    """Record the effective batch, M/K/N and operand dtype for one gradient GEMM.

    A kernel detail with physical transpose layouts needs a recorded output
    shape to disambiguate its GEMM. Without it, use the logical operand order.
    """
    inputs = shapes(input_shapes)
    types = tuple(t.removeprefix("dt_") for t in dtypes(input_types))
    if (len(inputs) != 2 or len(types) != 2 or not set(types) <= FLOAT_TYPES
            or any(not isinstance(s, (list, tuple)) or len(s) < 3 or not _dims(s) for s in inputs)):
        return None
    left, right = inputs
    if left[:-2] != right[:-2]:
        return None
    batch = prod(left[:-2])
    output = shapes(output_shapes)
    if output and (len(output) != 1 or not isinstance(output[0], (list, tuple))
                   or len(output[0]) < 3 or not _dims(output[0]) or prod(output[0][:-2]) != batch):
        return None
    a, b = tuple(left[-2:]), tuple(right[-2:])
    candidates = set()
    for m, k in (a, a[::-1]) if output else (a,):
        for inner, n in (b, b[::-1]) if output else (b,):
            if k == inner and (not output or (m, n) == tuple(output[0][-2:])):
                candidates.add((batch, m, k, n, types))
    return next(iter(candidates)) if len(candidates) == 1 else None


def _sequence(event):
    seq = get_arg(event.args, "Sequence number", "Sequence nr")
    return str(seq) if str(seq).isdigit() else None


def _forward_index(profile, op_to_anchor, include_bmm=False):
    result = defaultdict(set)
    for i, op in enumerate(profile.ops):
        if i in op_to_anchor:
            anchor = op_to_anchor[i]
        elif include_bmm and op.canonical_name == "aten::bmm":
            parent = op.parent_index
            while parent is not None and profile.ops[parent].canonical_name != "aten::matmul":
                parent = profile.ops[parent].parent_index
            if parent not in op_to_anchor:
                continue
            anchor = op_to_anchor[parent]
        else:
            continue
        if (sequence := _sequence(op.event)) is not None:
            result[sequence].add((op.event.tid, anchor))
    return result


def _forward_anchor(events, fwd_index):
    anchors = set()
    for event in events:
        tid = str(get_arg(event.args, "Fwd thread id", "Forward thread id") or "")
        entries = fwd_index.get(_sequence(event), ())
        anchors.update(anchor for thread, anchor in entries if tid in {"", "0", "-1"} or tid == thread)
    return next(iter(anchors)) if len(anchors) == 1 else None


@dataclass
class _Scope:
    event: TraceEvent
    op_index: int | None
    kernels: list
    memory: list
    gemms: dict
    module: str
    forward: int | None

    @property
    def signature(self):
        return tuple(sorted(signature for signature, launches in self.gemms.items() for _ in launches))


@dataclass
class BackwardCall:
    npu: _Scope
    gpu: _Scope
    basis: str

    @property
    def call_id(self):
        return f"matmul-backward:npu:{self.npu.op_index}"


def _gpu_boundaries(gpu):
    by_thread = defaultdict(list)
    for event in gpu.backward_scopes:
        by_thread[event.tid].append(event)
    for events in by_thread.values():
        inner = sorted((e for e in events if e.name.strip().lower() == "bmmbackward0"),
                       key=lambda e: (e.timestamp_us, e.index))
        starts = [e.timestamp_us for e in inner]
        boundaries = {e.index: (e, [e]) for e in inner}
        for event in events:
            if event.index in boundaries:
                continue
            first = bisect_left(starts, event.timestamp_us)
            last = bisect_left(starts, event.end_us, lo=first)
            children = [e for e in inner[first:last] if e.end_us <= event.end_us]
            if len(children) == 1:
                boundaries[children[0].index][1].append(event)
            elif not children:
                boundaries[event.index] = (event, [event])
        # Discard intersecting boundaries rather than using a kernel twice.
        spans = sorted(boundaries.values(), key=lambda pair: (pair[0].timestamp_us, pair[0].index))
        max_end = float("-inf")
        for i, (event, wrappers) in enumerate(spans):
            separate = max_end <= event.timestamp_us and (
                i + 1 == len(spans) or event.end_us <= spans[i+1][0].timestamp_us)
            max_end = max(max_end, event.end_us)
            if separate:
                yield event, wrappers


def _gpu_scopes(gpu, fwd_index):
    ops_by_thread = defaultdict(list)
    for index, op in enumerate(gpu.ops):
        ops_by_thread[op.event.tid].append(index)
    starts = {tid: [gpu.ops[i].event.timestamp_us for i in indices]
              for tid, indices in ops_by_thread.items()}
    memory_by_owner = defaultdict(list)
    for index, activity in enumerate(gpu.memory_activities):
        if activity.owner_op_index is not None:
            memory_by_owner[activity.owner_op_index].append(index)
    result = []
    for event, wrappers in _gpu_boundaries(gpu):
        timestamps = starts.get(event.tid, [])
        first = bisect_left(timestamps, event.timestamp_us)
        last = bisect_left(timestamps, event.end_us, lo=first)
        kernels, memory, gemms, modules = [], [], defaultdict(list), set()
        for i in ops_by_thread[event.tid][first:last]:
            op = gpu.ops[i]
            if op.event.end_us > event.end_us:
                continue
            kernels.extend(op.kernel_indices)
            memory.extend(memory_by_owner[i])
            if op.canonical_name in {"aten::matmul", "aten::mm", "aten::addmm", "aten::baddbmm"}:
                break
            if op.canonical_name != "aten::bmm":
                continue
            signature = _gemm(op.input_shapes, op.input_dtypes)
            if signature is None or not op.kernel_indices:
                break
            gemms[signature].append(tuple(op.kernel_indices))
            if op.module_key:
                modules.add(op.module_key)
        else:
            if not gemms or any(gpu.kernels[k].link_method not in GPU_LINKS for k in kernels):
                continue
            if any(gpu.memory_activities[k].link_method not in GPU_LINKS for k in memory):
                continue
            if len({(gpu.kernels[k].event.pid, str(get_arg(gpu.kernels[k].event.args, "device")))
                    for k in kernels}) != 1 or len(modules) > 1:
                continue
            result.append(_Scope(event, None, kernels, memory, dict(gemms),
                                 next(iter(modules), ""), _forward_anchor(wrappers, fwd_index)))
    return result


def _npu_scopes(npu, matches, fwd_index):
    result = []
    existing = {m.npu_op_index for m in matches if m.confidence in {"high", "medium"}}
    kernels_by_op = scoped_indices(npu)
    for i, op in enumerate(npu.ops):
        if op.canonical_name != BACKWARD_OP or i in existing:
            continue
        parent = op.parent_index
        while parent is not None and npu.ops[parent].canonical_name != BACKWARD_OP:
            parent = npu.ops[parent].parent_index
        if parent is not None:
            continue
        kernels = kernels_by_op[i]
        if not kernels or any(npu.kernels[k].detail is None or npu.kernels[k].link_method not in NPU_LINKS
                              for k in kernels):
            continue
        if len({npu.kernels[k].detail.device_id for k in kernels}) != 1:
            continue
        gemms = defaultdict(list)
        for k in kernels:
            detail = npu.kernels[k].detail
            kind = detail.op_type.lower()
            if kind != "batchmatmulv3":
                if "matmul" in kind:
                    break
                continue
            if (signature := _gemm(detail.input_shapes, detail.input_dtypes, detail.output_shapes)) is None:
                break
            gemms[signature].append((k,))
        else:
            if gemms:
                result.append(_Scope(op.event, i, kernels, [], dict(gemms), op.module_key,
                                     _forward_anchor([op.event], fwd_index)))
    return result


def recover_backward_calls(gpu, npu, matches):
    if not gpu.backward_scopes or not any(op.canonical_name == BACKWARD_OP for op in npu.ops):
        return []
    # For a multi-step capture require a selected step on both sides. The
    # fallback must never trade a missing backward call for one in another step.
    if any(p.selected_step == "all" and len(p.step_names) > 1 for p in (gpu, npu)):
        return []
    anchors = {}
    for match in matches:
        g, n = match.gpu_op_index, match.npu_op_index
        gop, nop = gpu.ops[g], npu.ops[n]
        if (match.confidence in {"high", "medium"} and gop.canonical_name == nop.canonical_name == "aten::matmul"
                and gop.input_shapes and gop.input_dtypes
                and not compare_metadata(nop, gop)[0]):
            anchors[g] = n
    left = _npu_scopes(npu, matches, _forward_index(npu, {n: n for n in anchors.values()}))
    right = _gpu_scopes(gpu, _forward_index(gpu, anchors, include_bmm=True))
    calls, used_npu, used_gpu = [], set(), set()

    def pair(key, basis):
        groups = [defaultdict(list), defaultdict(list)]
        for side, scopes, used in ((0, left, used_npu), (1, right, used_gpu)):
            for scope in scopes:
                value = key(scope)
                if value is not None and scope.event.index not in used:
                    groups[side][(scope.signature, value)].append(scope)
        for group, nscopes in groups[0].items():
            gscopes = groups[1].get(group, [])
            if len(nscopes) != 1 or len(gscopes) != 1:
                continue
            nscope, gscope = nscopes[0], gscopes[0]
            if nscope.forward is not None and gscope.forward is not None and nscope.forward != gscope.forward:
                continue
            if (basis != "matmul_backward_forward_anchor" and nscope.module and gscope.module
                    and nscope.module != gscope.module):
                continue
            used_npu.add(nscope.event.index)
            used_gpu.add(gscope.event.index)
            calls.append(BackwardCall(nscope, gscope, basis))

    pair(lambda scope: scope.forward, "matmul_backward_forward_anchor")
    pair(lambda scope: scope.module or None, "matmul_backward_module_workload")
    pair(lambda scope: "", "matmul_backward_unique_workload")
    return sorted(calls, key=lambda call: call.npu.op_index)


def type_units(call):
    """Expose only uniquely matched gradient kernels inside a verified call."""
    for signature, npu_launches in call.npu.gemms.items():
        gpu_launches = call.gpu.gemms[signature]
        if len(npu_launches) == len(gpu_launches) == len(gpu_launches[0]) == 1:
            yield npu_launches[0][0], gpu_launches[0][0]

"""Non-Cast device-memory correspondences and explicitly bounded CPU bundles.

Memory activities live in a separate ID namespace. Bundle reports are an
alternative view of work already present in the per-type report, not additive.
"""
from collections import Counter, defaultdict
import json

from .model import bundle_metrics
from .parser import get_arg


def encoded(values):
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def scoped_indices(profile, memory=False):
    result = [[] for _ in profile.ops]
    records = profile.memory_activities if memory else profile.kernels
    for index, record in enumerate(records):
        owner = record.owner_op_index
        while owner is not None:
            result[owner].append(index)
            owner = profile.ops[owner].parent_index
    return result


def enrich_memory_mappings(gpu, npu, matches, rows):
    by_npu = {m.npu_op_index: m for m in matches}
    memory = scoped_indices(gpu, True)
    compute = scoped_indices(gpu)
    used = set()
    for row in rows:
        row.update({
            "GPU Activity Kind": "kernel" if row["Comparable"] else "",
            "GPU Memory Activity Count": 0,
            "GPU Memory Activity Indices": "[]",
            "GPU Memory Activity Names": "[]",
            "GPU Memory Activity Durations(us)": "[]",
            "GPU Kernel Time(us)": row["GPU Total Time(us)"],
            "GPU Memory Time(us)": 0.0,
        })
        index = row["NPU CPU OP Index"]
        match = by_npu.get(index)
        if match is None or match.confidence not in {"high", "medium"} or row["Comparable"]:
            continue
        op = npu.ops[index]
        if (row["OP Type"], op.canonical_name) not in {
            ("Tile", "aten::repeat"), ("TensorMove", "aten::cat"),
        }:
            continue
        # A complete, unambiguous single NPU computation against device copies.
        if len([k for k in op.kernel_indices if npu.kernels[k].detail is not None]) != 1:
            continue
        indices = memory[match.gpu_op_index]
        if compute[match.gpu_op_index] or not indices or used.intersection(indices):
            continue
        activities = [gpu.memory_activities[i] for i in indices]
        if any(a.event.category != "gpu_memcpy" or "DtoD" not in a.event.name for a in activities):
            continue
        metrics = bundle_metrics(activities)
        used.update(indices)
        row.update({
            "Mapping Mode": "gpu_device_memcpy", "Mapped": True,
            "Comparable": True, "Accounted": True, "API Helper Kernel": False,
            "GPU Kernel Source": "device_memory", "GPU Activity Kind": "memcpy",
            "GPU Memory Activity Count": len(indices),
            "GPU Memory Activity Indices": encoded(indices),
            "GPU Memory Activity Names": encoded(a.event.name for a in activities),
            "GPU Memory Activity Durations(us)": encoded(a.event.duration_us for a in activities),
            "GPU Memory Time(us)": round(metrics.sum_us, 6),
            "GPU Total Time(us)": round(metrics.sum_us, 6),
            "GPU Device Busy(us)": round(metrics.busy_us, 6),
            "GPU Bundle Span(us)": round(metrics.span_us, 6),
            "Performance Ratio(GPU/NPU)": round(metrics.sum_us / row["NPU Duration(us)"], 6)
            if row["NPU Duration(us)"] else "",
        })


def memory_diagnostic_rows(gpu):
    return [{
        "GPU Memory Activity Index": i, "Activity Kind": activity.event.category,
        "Name": activity.event.name, "Timestamp(us)": activity.event.timestamp_us,
        "Duration(us)": activity.event.duration_us,
        "Bytes": get_arg(activity.event.args, "bytes"),
        "Stream ID": activity.stream_id, "Correlation ID": activity.correlation_id,
        "Owner OP Index": activity.owner_op_index if activity.owner_op_index is not None else "",
        "Owner OP Name": gpu.ops[activity.owner_op_index].canonical_name
        if activity.owner_op_index is not None else "",
        "Link Method": activity.link_method,
    } for i, activity in enumerate(gpu.memory_activities)]


def _scalar_bool(op):
    try:
        shapes, dtypes = json.loads(op.input_shapes), json.loads(op.input_dtypes)
        return bool(shapes and dtypes and shapes[0] == [] and dtypes[0] == "bool")
    except (ValueError, TypeError):
        return False


def _preceding_scalar(profile, index, names):
    consumer = profile.ops[index]
    candidates = []
    for i in range(max(0, index - 8), index):
        op = profile.ops[i]
        if (op.canonical_name in names and _scalar_bool(op)
                and op.module_key == consumer.module_key and bool(op.module_key)
                and (op.event.pid, op.event.tid) == (consumer.event.pid, consumer.event.tid)
                and 0 <= consumer.event.timestamp_us - op.event.end_us <= 1000):
            # Do not cross a different logical index_put invocation.
            if not any(profile.ops[j].canonical_name == "aten::_index_put_impl_"
                       for j in range(i + 1, index)):
                candidates.append(i)
    return candidates[0] if len(candidates) == 1 else None


def build_cpu_bundle_rows(gpu, npu, matches, mapping_rows):
    """Expose mixed-type Stack and validated boolean scalar preparation calls."""
    compute, memory = scoped_indices(gpu), scoped_indices(gpu, True)
    npu_compute = scoped_indices(npu)
    used_gpu, used_memory, used_npu = set(), set(), set()
    main_gpu = {k for row in mapping_rows for k in json.loads(row["GPU Kernel Indices"])}
    rows = []
    for match in matches:
        if match.confidence not in {"high", "medium"}:
            continue
        op = npu.ops[match.npu_op_index]
        if not op.module_key or op.module_key != gpu.ops[match.gpu_op_index].module_key:
            continue
        if op.canonical_name not in {"aten::stack", "aten::_index_put_impl_"}:
            continue
        nk = [i for i in npu_compute[match.npu_op_index] if npu.kernels[i].detail is not None]
        types = Counter(npu.kernels[i].detail.op_type for i in nk)
        gk = compute[match.gpu_op_index]
        gm = memory[match.gpu_op_index]
        prep = None
        if op.canonical_name == "aten::stack":
            if set(types) != {"Pack", "ConcatD"} or not gk or main_gpu.intersection(gk) or gm:
                continue
            name, basis = "Stack", "trusted_cpu_call_mixed_types"
        else:
            if not types or "IndexPutV2" not in types or set(types) - {"IndexPutV2", "BroadcastTo"}:
                continue
            try:
                shapes, dtypes = json.loads(op.input_shapes), json.loads(op.input_dtypes)
                if shapes[2] != [] or dtypes[0] != "bool" or dtypes[2] != "bool":
                    continue
                gpu_op = gpu.ops[match.gpu_op_index]
                gpu_shapes, gpu_dtypes = json.loads(gpu_op.input_shapes), json.loads(gpu_op.input_dtypes)
                if gpu_shapes != shapes or [d or "none" for d in gpu_dtypes] != dtypes:
                    continue
            except (IndexError, ValueError, TypeError):
                continue
            prep = _preceding_scalar(npu, match.npu_op_index, {"aten::fill_"})
            lift = _preceding_scalar(gpu, match.gpu_op_index, {"aten::lift_fresh"})
            if prep is None or lift is None or not gk or len(gm) != 1:
                continue
            activity = gpu.memory_activities[gm[0]]
            if (activity.event.category != "gpu_memcpy" or "HtoD" not in activity.event.name
                    or get_arg(activity.event.args, "bytes") != 1):
                continue
            pk = npu.ops[prep].kernel_indices
            prep_parent = npu.ops[prep].parent_index
            if prep_parent is not None and npu.ops[prep_parent].canonical_name == "aten::scalar_tensor":
                pk = npu_compute[prep_parent]
            if len(pk) != 1 or npu.kernels[pk[0]].detail is None or npu.kernels[pk[0]].detail.op_type != "Fill":
                continue
            nk = pk + nk
            name, basis = "ScalarPrepare+IndexPut", "matched_consumer_bool_scalar_1byte_htod"
        if used_gpu.intersection(gk) or used_memory.intersection(gm) or used_npu.intersection(nk):
            continue
        used_gpu.update(gk); used_memory.update(gm); used_npu.update(nk)
        gpu_metrics = bundle_metrics([gpu.kernels[i] for i in gk] + [gpu.memory_activities[i] for i in gm])
        npu_metrics = bundle_metrics([npu.kernels[i] for i in nk])
        kernel_time = sum(gpu.kernels[i].event.duration_us for i in gk)
        memory_time = sum(gpu.memory_activities[i].event.duration_us for i in gm)
        rows.append({
            "CPU Bundle ID": f"cpu-bundle:{match.npu_op_index}", "CPU Bundle": name,
            "Module Key": op.module_key, "Alignment Basis": basis,
            "CPU Match ID": match.match_id, "NPU CPU OP Index": match.npu_op_index,
            "NPU Preparation OP Index": prep if prep is not None else "",
            "GPU CPU OP Index": match.gpu_op_index,
            "NPU Kernel Indices": encoded(nk), "GPU Kernel Indices": encoded(gk),
            "GPU Memory Activity Indices": encoded(gm),
            "NPU Types": " + ".join(sorted({npu.kernels[i].detail.op_type for i in nk})),
            "NPU Kernel Count": len(nk), "GPU Kernel Count": len(gk),
            "GPU Memory Activity Count": len(gm),
            "NPU Total Time(us)": round(npu_metrics.sum_us, 6),
            "GPU Kernel Time(us)": round(kernel_time, 6), "GPU Memory Time(us)": round(memory_time, 6),
            "GPU Total Time(us)": round(gpu_metrics.sum_us, 6),
            "Comparable Ratio(GPU/NPU)": round(gpu_metrics.sum_us / npu_metrics.sum_us, 6)
            if npu_metrics.sum_us else "",
            "Scope": "独立视图，勿与逐类型表相加",
        })
    by_kernel = {i: row for row in rows for i in json.loads(row["NPU Kernel Indices"])}
    for row in mapping_rows:
        bundle = by_kernel.get(row["NPU Kernel Index"])
        row["CPU Bundle ID"] = bundle["CPU Bundle ID"] if bundle else ""
        row["Comparable In CPU Bundle"] = bool(bundle)
        if bundle:
            row["Accounted"] = True
    return rows


def summarize_cpu_bundles(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["CPU Bundle"]].append(row)
    result = []
    for name, group in groups.items():
        row = {"CPU Bundle": name, "Call Count": len(group),
               "NPU Types": " + ".join(sorted({value.strip() for r in group
                                                for value in r["NPU Types"].split("+")}))}
        for key in ("NPU Kernel Count", "GPU Kernel Count", "GPU Memory Activity Count",
                    "NPU Total Time(us)", "GPU Kernel Time(us)", "GPU Memory Time(us)", "GPU Total Time(us)"):
            row[key] = round(sum(r[key] for r in group), 6)
        row["Comparable Ratio(GPU/NPU)"] = round(row["GPU Total Time(us)"] / row["NPU Total Time(us)"], 6) if row["NPU Total Time(us)"] else ""
        row["Scope"] = "独立视图，勿与逐类型表相加"
        result.append(row)
    return result

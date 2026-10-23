"""Single-side adapter over profiling_tools; never changes its ownership rules."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from profiling_tools import parser
from profiling_tools.core_breakdown import category
from profiling_tools.model import interval_union_us
from .features import normalize_dtypes_json

MAX_EXPANDED_BYTES = 4096 * 1024**2
MAX_EVENTS = 2_000_000
MAX_UNITS = 200_000


def materialize_trace(path: Path):
    limit = int(os.environ.get("WORKLOAD_MODELER_EXPANDED_MB", "4096"))*1024**2
    if path.suffix != ".gz":
        if path.stat().st_size > limit:
            raise ValueError(f"Trace 超过 {limit//1024**2} MiB 解析上限，可调整 --expanded-limit-mb")
        return path
    target = path.parent / "expanded.json"
    size = 0
    try:
        with gzip.open(path, "rb") as source, target.open("wb") as dest:
            while chunk := source.read(1024**2):
                size += len(chunk)
                if size > limit:
                    raise ValueError(f"Trace 解压大小超过 {limit//1024**2} MiB，可调整 --expanded-limit-mb")
                dest.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def paths(directory):
    directory = Path(directory)
    trace = next((p for p in (directory/"trace.json", directory/"trace.json.gz") if p.is_file()), None)
    detail = directory/"kernel_details.csv"
    return trace, detail if detail.is_file() else None


def trace_data(path):
    events, count = parser._load_trace_events(materialize_trace(path), strict=True)
    if count > MAX_EVENTS:
        raise ValueError("Trace 超过 200 万事件，请缩小采集范围")
    return events, count


def device_key(event, detail=None):
    if detail is not None and detail.device_id:
        return "device:" + detail.device_id
    device = parser.get_arg(event.args, "device", "Device ID", "device_id", "device index")
    return "device:" + str(device) if device is not None else "pid:" + event.pid


def load_details(path):
    if path is None:
        return []
    limit = int(os.environ.get("WORKLOAD_MODELER_EXPANDED_MB", "4096"))*1024**2
    if path.stat().st_size > limit:
        raise ValueError(f"CSV 超过 {limit//1024**2} MiB，可调整 --expanded-limit-mb")
    details = parser.load_kernel_details(str(path))
    if not details:
        raise ValueError("kernel_details.csv 为空，不能当作零工作量")
    if len(details) > MAX_UNITS:
        raise ValueError("CSV 超过20万行，请缩小采集范围")
    for d in details:
        if not d.name or not d.op_type or d.duration_us is None or d.duration_us < 0:
            raise ValueError(f"CSV 第{d.row_index}行缺少 Name/Type 或有效 Duration(us)")
        raw_start = parser.get_arg(d.raw, "Start Time(us)", "Start Time(μs)")
        if raw_start not in (None, "", "N/A") and d.start_us is None:
            raise ValueError(f"CSV 第{d.row_index}行开始时间无效")
    return details


def inventory(directory, side):
    trace, detail = paths(directory)
    details = load_details(detail) if side == "npu" else []
    if trace is None and not details:
        raise ValueError("GPU需要trace；NPU至少需要kernel_details.csv或带CPU调用的trace")
    info = {"side": side, "cpu_pids": [], "steps": [], "devices": [], "warnings": [],
            "trace_events": 0, "detail_rows": len(details), "scope_required": True}
    if trace:
        events, count = trace_data(trace)
        info["trace_events"] = count
        pids = Counter(e.pid for e in events if e.phase == "X" and e.name.startswith("aten::"))
        info["cpu_pids"] = [{"pid": p, "count": n} for p,n in sorted(pids.items())]
        info["steps"] = [{"pid": e.pid, "name": e.name, "start_us": e.timestamp_us, "duration_us": e.duration_us}
                         for e in events if e.phase == "X" and parser._is_step_event(e)]
        hw = parser._device_process_ids(events)
        info["devices"] = sorted({device_key(e) for e in events if e.phase == "X" and parser._is_kernel(e, False, hw)})
        if not pids:
            info["warnings"].append("Trace无aten CPU调用。请使用对应PyTorch trace；NPU可仅上传CSV分析。")
        if count != len(events):
            info["warnings"].append("部分非完整事件被解析器丢弃，关联证据可能不完整")
    else:
        info["devices"] = sorted({"device:"+d.device_id if d.device_id else "unknown" for d in details})
        info["steps"] = [{"pid":"", "name":s} for s in sorted({d.step_id for d in details if d.step_id})]
        info["warnings"].append("CSV-only：没有CPU Step墙钟边界，不能输出整步性能比")
    if details:
        info["detail_devices"] = sorted({"device:"+d.device_id if d.device_id else "unknown" for d in details})
    return info


ALIASES = {"MatMul": "aten::mm", "MatMulV2": "aten::mm", "MatMulV3": "aten::mm",
           "BatchMatMul": "aten::bmm", "BatchMatMulV2": "aten::bmm", "Add": "aten::add",
           "Sub": "aten::sub", "Mul": "aten::mul", "Relu": "aten::relu"}
MATRIX = {"aten::mm", "aten::matmul", "aten::bmm", "aten::addmm", "aten::linear", "aten::conv2d", "aten::convolution"}
VECTOR = {"aten::add", "aten::sub", "aten::mul", "aten::div", "aten::relu", "aten::gelu", "aten::sum",
          "aten::mean", "aten::index", "aten::index_add", "aten::index_add_", "aten::gather", "aten::scatter",
          "aten::cat", "aten::concat", "aten::copy_", "aten::to", "aten::_to_copy", "aten::slice", "aten::softmax"}


def signature(op, shapes, dtypes):
    payload = json.dumps([op, shapes, dtypes], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def analyze(directory, scope):
    trace, detail_path = paths(directory)
    details = load_details(detail_path) if scope.side == "npu" else []
    warnings = []
    if not scope.step and not scope.all_steps:
        raise ValueError("请选择明确Step，或确认使用全采集范围（含预热和未标记区间）")
    records, units = [], {}
    window = None
    if trace:
        data = trace_data(trace)
        events = data[0]
        pids = sorted({e.pid for e in events if e.phase == "X" and e.name.startswith("aten::")})
        if not pids:
            raise ValueError("Trace没有aten CPU调用，无法建立逻辑算子归属")
        if scope.pid and scope.pid not in pids:
            raise ValueError("所选CPU PID不存在")
        if not scope.pid and len(pids) != 1:
            raise ValueError("存在多个CPU PID，请明确选择")
        pid = scope.pid or pids[0]
        if scope.step:
            matches = [e for e in events if e.pid == pid and e.phase == "X" and parser._is_step_event(e)
                       and parser._step_matches(e.name, scope.step)]
            if len(matches) != 1 or matches[0].duration_us <= 0:
                raise ValueError("Step必须唯一且有正的墙钟时长")
            window = (matches[0].timestamp_us, matches[0].end_us)
        profile = parser.parse_trace(str(trace), scope.side, requested_pid=pid, requested_step=scope.step,
                                     trace_data=data, include_communication=False)
        if details:
            selected = []
            for d in details:
                if scope.step and d.step_id and not parser._step_matches(d.step_id, scope.step):
                    continue
                if scope.step and not d.step_id:
                    if d.start_us is None:
                        raise ValueError("CSV缺少Step ID及开始时间，无法安全限定Step")
                    if not window[0] <= d.start_us < window[1]:
                        continue
                selected.append(d)
            if not selected:
                raise ValueError("所选Step没有NPU明细")
            parser.attach_kernel_details(profile, selected)
        for idx, k in enumerate(profile.kernels):
            owner = profile.ops[k.owner_op_index] if k.owner_op_index is not None else None
            d = k.detail
            op = owner.canonical_name if owner else (ALIASES.get(d.op_type,d.op_type) if d else "UNKNOWN")
            shape = owner.input_shapes if owner else parser.normalize_input_shapes(d.input_shapes) if d else ""
            dtype = owner.input_dtypes if owner else parser.normalize_input_dtypes(d.input_dtypes) if d else ""
            unit = "op:"+str(k.owner_op_index) if owner else "kernel:"+str(idx)
            cls = category(d.aicore_time_us) if d else "Unknown"
            if scope.side == "gpu":
                cls = "Cube-like" if op in MATRIX else "Vector-like" if op in VECTOR else "Mixed" if "attention" in op.lower() else "Unknown"
            records.append(dict(id="kernel:"+str(k.event.index),unit=unit,op=op,shape=shape,dtype=dtype,
                class_name=cls,kind="compute",device=device_key(k.event,d),stream=k.stream_id or k.event.tid,
                start=k.event.timestamp_us,duration=k.event.duration_us,name=k.event.name,
                link=k.link_method,aic=d.aicore_time_us if d else None,aiv=d.aiv_vec_time_us if d else None,
                metadata_source="cpu_op" if owner else "npu_kernel" if d else "unknown",
                raw_shape=owner.raw_input_shapes if owner else d.input_shapes if d else "",
                raw_dtype=owner.raw_input_dtypes if owner else d.input_dtypes if d else "",
                output_shape=parser.normalize_input_shapes(d.output_shapes) if d and not owner else "",
                output_dtype=parser.normalize_input_dtypes(d.output_dtypes) if d and not owner else ""))
        for idx,k in enumerate(profile.memory_activities):
            records.append(dict(id="memory:"+str(k.event.index),unit="memory:"+str(idx),op="MEMORY_ACTIVITY",shape="",dtype="",
                class_name="Memory",kind="memory",device=device_key(k.event),stream=k.stream_id or k.event.tid,
                start=k.event.timestamp_us,duration=k.event.duration_us,name=k.event.name,link=k.link_method,aic=None,aiv=None))
        # Communication is accounted only for device events, never CPU NCCL wrappers.
        hw = parser._device_process_ids(events)
        for e in events:
            if e.phase != "X" or not parser._is_communication_event(e):
                continue
            if not ("kernel" in e.category.lower() or e.pid in hw or parser._task_type(e)):
                continue
            if window and not window[0] <= e.timestamp_us < window[1]:
                continue
            records.append(dict(id="communication:"+str(e.index),unit="communication:"+str(e.index),op="COMMUNICATION",shape="",dtype="",
                class_name="Communication",kind="communication",device=device_key(e),stream=str(parser.get_arg(e.args,"stream") or e.tid),
                start=e.timestamp_us,duration=e.duration_us,name=e.name,link="device_event",aic=None,aiv=None))
        if len(pids)>1:
            warnings.append("CPU PID不等于设备过滤；未归属事件可能来自其他CPU进程，请核验设备范围")
        if data[1] != len(events):
            warnings.append("存在解析器丢弃的非完整事件")
    else:
        if scope.side != "npu" or not details:
            raise ValueError("缺少输入")
        for d in details:
            if scope.step:
                if not d.step_id:
                    raise ValueError("CSV-only限定Step要求每行有Step ID")
                if not parser._step_matches(d.step_id,scope.step):
                    continue
            op=ALIASES.get(d.op_type,d.op_type)
            comm="communication" in d.core_type.lower() or "hccl" in d.name.lower()
            records.append(dict(id="detail:"+str(d.row_index),unit="detail:"+str(d.row_index),op=op,
                shape=parser.normalize_input_shapes(d.input_shapes),dtype=parser.normalize_input_dtypes(d.input_dtypes),
                metadata_source="npu_kernel",raw_shape=d.input_shapes,raw_dtype=d.input_dtypes,
                output_shape=parser.normalize_input_shapes(d.output_shapes),output_dtype=parser.normalize_input_dtypes(d.output_dtypes),
                class_name="Communication" if comm else category(d.aicore_time_us),
                kind="communication" if comm else "compute",device="device:"+d.device_id if d.device_id else "unknown",
                stream=d.stream_id,start=d.start_us,duration=d.duration_us,name=d.name,link="csv_detail",aic=d.aicore_time_us,aiv=d.aiv_vec_time_us))
        warnings.append("CSV-only：每行是独立kernel，不代表完整CPU调用；无法恢复Step墙钟时间")
    devices=sorted({r["device"] for r in records})
    if not devices:
        raise ValueError("选择范围内没有设备活动")
    if scope.device and scope.device not in devices:
        raise ValueError(f"所选设备不存在；可用设备：{devices}")
    if not scope.device and len(devices)>1:
        raise ValueError(f"存在多个设备，必须明确选择：{devices}")
    chosen=scope.device or devices[0]
    records=[r for r in records if r["device"]==chosen]
    if len(records)>MAX_UNITS:
        raise ValueError("所选范围超过20万设备活动，请缩小Step范围")
    if len({r['id'] for r in records})!=len(records):
        raise ValueError("设备事件ID重复，停止统计")
    for r in records:
        r['dtype'] = normalize_dtypes_json(r['dtype'])
        r['output_dtype'] = normalize_dtypes_json(r.get('output_dtype', ''))
        u=units.setdefault(r["unit"],dict(id=r["unit"],op=r["op"],input_shapes=r["shape"],input_dtypes=r["dtype"],
            signature=signature(r["op"],r["shape"],r["dtype"]),source_us=0.,kernel_count=0,events=[],classes=set(),kind=r["kind"],
            metadata_source=r.get("metadata_source","unknown"),raw_input_shapes=r.get("raw_shape",""),
            raw_input_dtypes=r.get("raw_dtype",""),output_shapes=r.get("output_shape",""),output_dtypes=r.get("output_dtype","")))
        u["source_us"]+=r["duration"]
        u["kernel_count"]+=1
        u["events"].append({k:r[k] for k in ("id","name","start","duration","stream","link","aic","aiv")})
        u["classes"].add(r["class_name"])
    for u in units.values():
        u["class_name"]=next(iter(u["classes"])) if len(u["classes"])==1 else "Mixed"
        del u["classes"]
    compute=[r for r in records if r["kind"]=="compute"]
    compute_us=math.fsum(r["duration"] for r in compute)
    breakdown=[]
    for cls in sorted({r["class_name"] for r in compute}):
        members=[r for r in compute if r["class_name"]==cls]
        dur=math.fsum(r["duration"] for r in members)
        breakdown.append(dict(category=cls,count=len(members),time_us=dur,time_pct=100*dur/compute_us if compute_us else None))
    complete=all(r["start"] is not None for r in records)
    # Subtract the origin before adding durations to avoid cancellation at large trace timestamps.
    origin=min((r["start"] for r in records if r["start"] is not None), default=0.)
    intervals=[(r["start"]-origin,r["start"]-origin+r["duration"]) for r in records if r["start"] is not None]
    busy=interval_union_us(intervals) if complete else None
    total=math.fsum(r["duration"] for r in records)
    bounds=(window[0]-origin,window[1]-origin) if window else None
    in_window=bounds is not None and complete and all(bounds[0]<=a and b<=bounds[1] for a,b in intervals)
    step_busy=interval_union_us([(max(a,bounds[0]),min(b,bounds[1])) for a,b in intervals
                                if a<bounds[1] and b>bounds[0]]) if bounds and complete else None
    if window and not in_window:
        warnings.append("存在异步活动越过CPU Step边界，不输出整步预测")
    streams={r["stream"] for r in records}
    serial=complete and len(streams)==1 and "" not in streams and math.isclose(busy,total,abs_tol=1e-6)
    if not serial:
        warnings.append("非单流串行活动或时间戳不完整：严格整步预测不可用；整体经验估算中的多流Step场景须另行检查边界与重叠比例假设")
    return dict(schema_version=1,side=scope.side,device=chosen,scope=scope.model_dump(),units=list(units.values()),
        summary=dict(kernel_count=len(compute),compute_sum_us=compute_us,device_sum_us=total,busy_us=busy,
            memory_sum_us=math.fsum(r["duration"] for r in records if r["kind"]=="memory"),
            communication_sum_us=math.fsum(r["duration"] for r in records if r["kind"]=="communication"),
            step_us=window[1]-window[0] if window else None,serial_step_eligible=serial and in_window,
            streams=sorted(streams),timestamps_complete=complete,activities_in_step=in_window,
            overlap_us=max(0.,total-busy) if busy is not None else None,step_busy_us=step_busy,
            compute_breakdown=breakdown,unit_count=len(units),
            classification_basis="NPU aicore_time: 0=Vector, >0=Cube, missing=Unknown" if scope.side=="npu" else "GPU逻辑算子语义推断，非NPU计数器或GPU硬件利用率",
            time_basis="Duration求和、区间并集与CPU Step墙钟分别记录；通信/内存不进入Cube/Vector分母"),warnings=warnings)

"""Source-anchored analytical scenarios, not cycle-accurate simulation."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

from .schema import AnalysisRequest
from .features import ModelIssue, extract_features, diagnostics
from .screening import assessment
from .empirical import estimate


def dtype_name(value):
    return {"float":"fp32","float32":"fp32","float16":"fp16","half":"fp16","bfloat16":"bf16",
            "fp32":"fp32","fp16":"fp16","bf16":"bf16"}.get(str(value).lower().replace("torch.",""))


def features(unit):
    """Compatibility wrapper for callers of the original feature helper."""
    try:
        return extract_features(unit), None
    except ModelIssue as exc:
        return None, str(exc)


def peak(hardware,key,assumptions,matrix=False):
    m=hardware["metrics"].get(key)
    if not m or m["value"] is None:
        raise ModelIssue("missing_spec",f"{hardware['name']}缺少{key}")
    if m["evidence"]=="estimated" and not assumptions.allow_estimated_specs:
        raise ModelIssue("estimated_spec",f"{key}是估算规格，需明确允许后才参与预测")
    if matrix and m.get("dense") is not True:
        raise ModelIssue("dense_basis",f"{key}未确认稠密口径，不能使用")
    return m["value"]


def analytical_time(f,hw,assumptions,side,kernels,factor=1.):
    mode=f["dtype"]
    if f["family"]=="matrix" and mode=="fp32":
        mode=getattr(assumptions,side+"_fp32_mode")
        if mode=="unknown":
            raise ModelIssue("precision","FP32矩阵计算模式未确认，不能自动按TF32/HF32折算")
    p=peak(hw,f["family"]+"_"+mode,assumptions,matrix=f["family"]=="matrix")
    b=peak(hw,"memory_bandwidth",assumptions)
    ce=min(1.,getattr(assumptions,side+"_compute_efficiency")*factor)
    be=min(1.,getattr(assumptions,side+"_memory_efficiency")*factor)
    compute_us=f["flops"]/(p*ce*1e6)
    memory_us=f["logical_bytes"]/(b*be*1e3)
    overhead=kernels*getattr(assumptions,side+"_launch_us")
    return overhead+max(compute_us,memory_us),dict(compute_us=compute_us,logical_memory_us=memory_us,
        launch_us=overhead,dominant="compute" if compute_us>=memory_us else "memory",
        compute_mode=mode,assumed_kernel_count=kernels)


def predict(profile,config):
    request=AnalysisRequest.model_validate(config["request"])
    a=request.assumptions
    source,target=config["source_hardware"],config["target_hardware"]
    calibration={c.signature:c for c in request.calibrations}
    known_signatures={u["signature"] for u in profile["units"]}
    if set(calibration)-known_signatures:
        raise ValueError("校准记录的签名不在当前范围中，请勿沿用其他任务的过期签名")
    rows=[]
    for unit in profile["units"]:
        row=dict(unit)
        row.update(predicted_us=None,ratio=None,method="unmodelled",reason="",reason_code="",features=None,scenario_us=None)
        cal=calibration.get(unit["signature"])
        if unit["source_us"]<=0:
            row["reason"]="零时长事件不能建立性能比"
            row["reason_code"]="zero_time"
        elif cal:
            t=unit["source_us"]*cal.target_us/cal.source_us
            row.update(predicted_us=t,ratio=cal.source_us/cal.target_us,method="user_calibration",
                       scenario_us=dict(conservative=t,baseline=t,optimistic=t),reason=cal.evidence)
        elif unit["kind"]!="compute":
            row["reason"]="内存活动/通信需要独立校准，不按显存带宽直接折算"
            row["reason_code"]="activity"
        else:
            try:
                f=extract_features(unit)
            except ModelIssue as exc:
                f=None
                row.update(reason=str(exc),reason_code=exc.code)
            row["features"]=f
            if f:
                try:
                    if f["family"]=="matrix" and f["dtype"]=="fp32" and a.source_fp32_mode!=a.target_fp32_mode and not a.precision_equivalence_confirmed:
                        raise ModelIssue("precision","两侧矩阵计算精度模式不同，需要确认精度策略可比较")
                    src,source_terms=analytical_time(f,source,a,"source",unit["kernel_count"])
                    scenarios={}
                    terms=None
                    for name,factor in [("conservative",.7),("baseline",1.),("optimistic",1.3)]:
                        dst,part=analytical_time(f,target,a,"target",unit["kernel_count"],factor)
                        scenarios[name]=unit["source_us"]*dst/src
                        if name=="baseline":
                            terms=part
                    row.update(predicted_us=scenarios["baseline"],ratio=unit["source_us"]/scenarios["baseline"],
                        method="source_anchored_roofline",scenario_us=scenarios,source_terms=source_terms,target_terms=terms,
                        reason="源耗时×两侧解析模型时间比；假设实现/逻辑流量/启动次数不变，非目标实测。效率系数是用户假设。")
                except ModelIssue as exc:
                    row["reason"]=str(exc)
                    row["reason_code"]=exc.code
        rows.append(row)
    total=profile["summary"]["device_sum_us"]
    modelled=[r for r in rows if r["predicted_us"] is not None]
    covered=math.fsum(r["source_us"] for r in modelled)
    predicted=math.fsum(r["predicted_us"] for r in modelled)
    full=len(modelled)==len(rows) and bool(rows) and total>0
    step=None
    if full and profile["summary"]["serial_step_eligible"] and a.fixed_gap_serial_step:
        wall=profile["summary"]["step_us"]
        gap=max(0.,wall-total)
        step={name:gap+math.fsum(r["scenario_us"][name] for r in modelled) for name in ("conservative","baseline","optimistic")}
        step.update(source_us=wall,fixed_gap_us=gap,ratio=wall/step["baseline"],
            ratio_low=wall/step["conservative"],ratio_high=wall/step["optimistic"],
            basis="单流无重叠、工作量完全建模、保留源端间隙不变的场景；不是完整依赖调度仿真")
    groups={}
    for r in rows:
        key=(r["op"],r["input_shapes"],r["input_dtypes"],r["class_name"])
        g=groups.setdefault(key,dict(op=r["op"],shape=r["input_shapes"],dtype=r["input_dtypes"],signature=r["signature"],
            class_name=r["class_name"],calls=0,kernel_count=0,source_us=0.,modelled_source_us=0.,predicted_us=0.,unit_ids=[],reasons=set()))
        g["calls"]+=1
        g["kernel_count"]+=r["kernel_count"]
        g["source_us"]+=r["source_us"]
        g["unit_ids"].append(r["id"])
        if r["predicted_us"] is not None:
            g["modelled_source_us"]+=r["source_us"]
            g["predicted_us"]+=r["predicted_us"]
        else:
            g["reasons"].add(r["reason"])
    for g in groups.values():
        g["coverage_pct"]=100*g["modelled_source_us"]/g["source_us"] if g["source_us"] else None
        g["ratio"]=g["modelled_source_us"]/g["predicted_us"] if g["predicted_us"] else None
        if not g["modelled_source_us"]:
            g["predicted_us"]=None
        g["reasons"]=sorted(g["reasons"])
    verdict="整步预测暂不可用；画像初筛与局部预测请分别查看"
    if step:
        verdict="场景假设下有优势，仍需实测" if step["ratio_low"]>1.05 else "场景假设下有性能风险" if step["ratio_high"]<.95 else "范围跨越或接近1，建议实测"
    risk=sorted([dict(op=r["op"],source_us=r["source_us"],reason=r["reason"],signature=r["signature"])
                 for r in rows if r["predicted_us"] is None],key=lambda r:r["source_us"],reverse=True)[:20]
    reason_groups=diagnostics(rows,total)
    blockers=[]
    if not full:
        blockers.append("设备活动未完全建模；局部倍率不能外推整步")
    if not profile["summary"]["serial_step_eligible"]:
        s=profile['summary']
        if 'streams' not in s:
            blockers.append("缺少单流、无重叠且活动完整落在唯一Step内的证据")
        else:
            if len(s['streams'])!=1 or '' in s['streams']:
                blockers.append("当前仅支持单流整步场景；本次stream："+", ".join(s['streams']))
            if not s['timestamps_complete']:
                blockers.append("存在缺少时间戳的设备活动")
            if s.get('overlap_us',0) and not math.isclose(s['busy_us'],s['device_sum_us'],abs_tol=1e-6):
                blockers.append(f"设备活动存在重叠：求和与并集差 {s['overlap_us']:.3f} μs")
            if not s['activities_in_step']:
                blockers.append("缺少唯一Step窗口，或部分设备活动越过Step边界")
    if not a.fixed_gap_serial_step:
        blockers.append("尚未确认源端间隙不变的Step场景假设")
    result=dict(schema_version=4,profile_summary=profile["summary"],warnings=profile["warnings"],units=rows,
        diagnostics=reason_groups,step_blockers=blockers,
        groups=sorted(groups.values(),key=lambda g:g["source_us"],reverse=True),risks=risk,step_prediction=step,
        prediction=dict(coverage_pct=100*covered/total if total else None,source_total_us=total,modelled_source_us=covered,
            unmodelled_source_us=total-covered,predicted_modelled_sum_us=predicted if modelled else None,
            modelled_device_ratio=covered/predicted if predicted else None,verdict=verdict,
            calibrated_source_us=math.fsum(r["source_us"] for r in modelled if r["method"]=="user_calibration"),
            ratio_basis="源/目标；设备倍率仅比较已建模子集，不代表整步速度；场景范围不是统计置信区间"),
        config=config,audit=dict(source_time_conserved=math.isclose(math.fsum(r["source_us"] for r in rows),total,abs_tol=1e-6),
            diagnostic_time_conserved=math.isclose(math.fsum(g["source_us"] for g in reason_groups),total-covered,abs_tol=1e-6),
            unique_activity_ids=len({e["id"] for r in rows for e in r["events"]})==sum(len(r["events"]) for r in rows)),
        model_version="source-anchored-v3")
    result['assessment']=assessment(result)
    result['empirical']=estimate(profile['summary'],rows,config,request.empirical.model_dump())
    return result

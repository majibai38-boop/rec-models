"""Coarse workload scenarios: explicit assumptions, disjoint time buckets, no precise-model gating."""
from collections import defaultdict
import math

from .schema import EmpiricalAssumptions

SCENARIOS = ('conservative', 'baseline', 'optimistic')
LABELS = {
    'matrix_compute':'矩阵计算部分', 'matrix_memory':'矩阵访存部分',
    'vector':'Vector（按访存折算）', 'fused':'FA/HSTU/混合融合路径',
    'short':'短kernel', 'transfer':'内存活动', 'communication':'通信活动',
    'unknown':'未知类别（独立假设）',
}


def estimate(summary, units, config, options=None):
    a = EmpiricalAssumptions.model_validate(options if options is not None else config.get('request', {}).get('empirical', {}))
    if not a.enabled:
        return dict(status='disabled', assumptions=a.model_dump(), conclusion='整体经验估算已关闭。')
    total = summary['device_sum_us']
    if total <= 0:
        return dict(status='empty', assumptions=a.model_dump(), conclusion='当前范围没有正的设备耗时。')
    source, target = config['source_hardware'], config['target_hardware']
    notices = [
        '所有倍率均为经验假设，未使用精细模型覆盖率作为门槛，也不等于目标设备实测。',
        f'短kernel先单独归类（小于{a.short_kernel_us:g}μs）；其余矩阵时间按假设拆为 '
        f'{a.cube_compute_fraction*100:g}%计算、{(1-a.cube_compute_fraction)*100:g}%访存，并非实测bound比例。',
        'Vector统一按带宽折算是经验假设；FA/HSTU、内存活动、通信和未知类别使用单独倍率。',
        f'非短矩阵统一采用源{a.source_matrix_mode.upper()}→目标{a.target_matrix_mode.upper()}规格档位，'
        '这是整体场景假设，不是逐调用dtype或精度等价结论。',
        '缺规格的基准1×、融合/未知/传输等默认基准1×及Step间隙默认不变，均为可编辑的中性假设。',
    ]

    def hardware_ratio(source_key, target_key, override):
        if override is not None:
            base, basis = override, '手动指定的经验倍率'
        else:
            sm = source.get('metrics', {}).get(source_key, {})
            tm = target.get('metrics', {}).get(target_key, {})
            sv, tv = sm.get('value'), tm.get('value')
            if not sv or not tv:
                return a.missing_spec_ratio.model_dump(), '缺少规格，采用显式“缺规格倍率”假设', False
            base = tv/sv
            basis = f'名义规格比 {target["name"]} {target_key} / {source["name"]} {source_key} = {base:.4f}，乘场景系数'
            for hw, key, metric in [(source,source_key,sm),(target,target_key,tm)]:
                if key.startswith('matrix_') and metric.get('dense') is not True:
                    notices.append(f'{hw["name"]} {key} 稠密口径未确认：仅在经验模式中按记录数值假设可比。')
                if metric.get('evidence') in {'estimated','unknown'}:
                    notices.append(f'{hw["name"]} {key} 的证据为 {metric.get("evidence")}，此处作为假设使用。')
        return {s:base*getattr(a.hardware_factors,s) for s in SCENARIOS}, basis, True

    matrix, matrix_basis, matrix_from_spec = hardware_ratio('matrix_'+a.source_matrix_mode, 'matrix_'+a.target_matrix_mode, a.matrix_ratio)
    bandwidth, bandwidth_basis, bandwidth_from_spec = hardware_ratio('memory_bandwidth','memory_bandwidth',a.bandwidth_ratio)
    rates = dict(matrix_compute=matrix, matrix_memory=bandwidth, vector=bandwidth,
        fused=a.fused_ratio.model_dump(), short=a.short_ratio.model_dump(), transfer=a.transfer_ratio.model_dump(),
        communication=a.communication_ratio.model_dump(), unknown=a.unknown_ratio.model_dump())
    bases = dict(matrix_compute=matrix_basis, matrix_memory=bandwidth_basis, vector=bandwidth_basis)
    times = defaultdict(list)
    for unit in units:
        # Use each recorded activity once. Cube compute/memory divide time, not duplicate it.
        events = unit.get('events') or [dict(duration=unit['source_us'])]
        if not math.isclose(math.fsum(e['duration'] for e in events),unit['source_us'],rel_tol=1e-8,abs_tol=1e-6):
            raise ValueError('经验估算调用与设备事件耗时不守恒')
        for event in events:
            duration = event['duration']
            if not math.isfinite(duration) or duration < 0:
                raise ValueError('经验估算不接受负数或非有限设备耗时')
            if unit['kind'] == 'memory':
                kind = 'transfer'
            elif unit['kind'] == 'communication':
                kind = 'communication'
            elif duration < a.short_kernel_us:
                kind = 'short'
            elif unit['class_name'] == 'Mixed' or any(k in unit['op'].lower() for k in ('flashattention','hstu')):
                kind = 'fused'
            elif unit['class_name'] in {'Cube','Cube-like'}:
                times['matrix_compute'].append(duration*a.cube_compute_fraction)
                times['matrix_memory'].append(duration*(1-a.cube_compute_fraction))
                continue
            elif unit['class_name'] in {'Vector','Vector-like'}:
                kind = 'vector'
            else:
                kind = 'unknown'
            times[kind].append(duration)
    buckets = []
    for kind, label in LABELS.items():
        duration = math.fsum(times[kind])
        if duration <= 0:
            continue
        ratios = rates[kind]
        buckets.append(dict(key=kind,label=label,source_us=duration,source_pct=100*duration/total,
            ratios=ratios,target_us={s:duration/ratios[s] for s in SCENARIOS},
            basis=bases.get(kind,'显式类别倍率假设；不是硬件实测或自动校准')))
    accounted = math.fsum(b['source_us'] for b in buckets)
    if not math.isclose(accounted,total,rel_tol=1e-8,abs_tol=1e-6):
        raise ValueError('经验估算分组耗时不守恒')
    device = {s:dict(target_us=math.fsum(b['target_us'][s] for b in buckets)) for s in SCENARIOS}
    for item in device.values():
        item['ratio'] = total/item['target_us']
        item['time_reduction_pct'] = 100*(1-item['target_us']/total)
    manual_time = math.fsum(b['source_us'] for b in buckets if b['key'] not in {'matrix_compute','matrix_memory','vector'}
                           or (b['key']=='matrix_compute' and not matrix_from_spec)
                           or (b['key'] in {'matrix_memory','vector'} and not bandwidth_from_spec))
    step = None
    step_reason = ''
    wall, busy = summary.get('step_us'), summary.get('step_busy_us')
    if not wall or busy is None:
        step_reason = '缺少Step墙钟或窗口内设备忙碌时间；可使用设备整体估算。'
    elif not summary.get('activities_in_step'):
        step_reason = '部分活动越过Step边界，无法把当前活动总量按比例折算到该Step。'
    elif not (0 < busy <= wall and busy <= total + max(1e-6,total*1e-8)):
        step_reason = 'Step或设备时间口径不一致，未生成整步场景。'
    else:
        gap = wall-busy
        step = dict(source_us=wall, source_busy_us=busy, source_gap_us=gap, scenarios={},
            basis='假设设备活动重叠比例不变：目标忙碌=源忙碌×目标活动求和/源活动求和；间隙按单独倍率折算。支持多流经验场景，不是依赖调度仿真。')
        for s in SCENARIOS:
            target_busy = busy*device[s]['target_us']/total
            target_gap = gap/getattr(a.gap_ratio,s)
            target_step = target_busy+target_gap
            step['scenarios'][s] = dict(target_us=target_step,busy_us=target_busy,gap_us=target_gap,
                ratio=wall/target_step,time_reduction_pct=100*(1-target_step/wall))
    d = device['baseline']['ratio']
    conclusion = f'整体经验估算：设备活动基准 {d:.3f}×（保守 {device["conservative"]["ratio"]:.3f}× / 乐观 {device["optimistic"]["ratio"]:.3f}×）。'
    if step:
        conclusion += f' 固定当前场景假设下整步基准 {step["scenarios"]["baseline"]["ratio"]:.4f}×。'
    conclusion += ' 倍率大于1表示目标更快；这是经验场景，不是精细预测或实测。'
    return dict(status='available',model='empirical-time-buckets-v1',conclusion=conclusion,
        assumptions=a.model_dump(),source_total_us=total,buckets=buckets,device=device,step=step,step_reason=step_reason,
        direct_assumption_time_pct=100*manual_time/total,
        notices=notices, audit=dict(source_time_conserved=True),
        formula='目标设备时间=Σ(源类别时间/类别源目标倍率)；整体倍率=源总时间/目标总时间；不对倍率做时间加权算术平均。')

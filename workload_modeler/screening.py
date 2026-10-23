"""Workload screening independent of prediction coverage and hardware fallbacks."""
from collections import defaultdict
import math


def percent(value, total):
    return 100 * value / total if total > 0 else None


def screen(summary, units, target_vendor):
    compute = summary['compute_sum_us']
    categories = defaultdict(float)
    for item in summary['compute_breakdown']:
        category = {'Cube-like':'Cube', 'Vector-like':'Vector'}.get(item['category'], item['category'])
        categories[category if category in {'Cube', 'Vector', 'Mixed'} else 'Unknown'] += item['time_us']
    cube, vector, mixed, unknown = (categories[k] for k in ('Cube', 'Vector', 'Mixed', 'Unknown'))
    cube_pct, vector_pct = percent(cube, compute), percent(vector, compute)
    cube_rows = [u for u in units if u['kind'] == 'compute' and u['class_name'] in {'Cube', 'Cube-like'}]
    # These are analytical-model terms, never counter-derived hardware bound labels.
    bound_rows = [u for u in cube_rows if u.get('target_terms')]
    bound_known = math.fsum(u['source_us'] for u in bound_rows)
    bound_memory = math.fsum(u['source_us'] for u in bound_rows if u['target_terms']['dominant'] == 'memory')
    memory_lower = percent(bound_memory, cube)
    memory_upper = percent(bound_memory + max(0., cube-bound_known), cube)
    kernels = [e for u in units if u['kind'] == 'compute' for e in u['events']]
    small = [e for e in kernels if e['duration'] < 5]
    fused = [u for u in units if u['kind'] == 'compute'
             and any(k in u['op'].lower() for k in ('flashattention', 'hstu'))]
    metrics = dict(cube_pct=cube_pct, vector_pct=vector_pct, mixed_pct=percent(mixed, compute),
        unknown_pct=percent(unknown, compute), classified_pct=percent(cube+vector+mixed, compute),
        small_kernel_count=len(small), small_kernel_time_pct=percent(math.fsum(e['duration'] for e in small), compute),
        fused_time_pct=percent(math.fsum(u['source_us'] for u in fused), compute),
        cube_bound_coverage_pct=percent(bound_known, cube), cube_memory_lower_pct=memory_lower,
        cube_memory_upper_pct=memory_upper, compute_us=compute)
    if not compute:
        status, conclusion = 'no_compute', '当前范围没有计算kernel；先核对采集范围与内存/通信活动。'
    elif vector_pct > 50:
        status, conclusion = 'vector_risk', 'Vector类工作量占主导：优先验证访存、索引及逐元素路径，矩阵峰值不足以解释整体性能。'
    elif cube_pct > 50 and memory_lower is not None and memory_lower > 70:
        status, conclusion = 'matrix_memory_risk', '矩阵占主导，但解析模型提示访存项占比较高：优先验证有效带宽、布局与数据复用。'
    elif unknown > compute / 2:
        status, conclusion = 'classification_gap', '多数计算时间尚未分类：已有热点可供排查，需先确认未知路径才能判断主要负载倾向。'
    elif cube_pct > 50:
        status, conclusion = 'matrix_candidate', '矩阵路径值得优先验证：Cube类工作量占主导，需结合精度、Shape和带宽确认能否兑现算力优势。'
    else:
        status, conclusion = 'mixed_workload', '混合负载：矩阵与Vector均需验证，优先检查高耗时矩阵、数据搬运和融合路径。'
    if target_vendor != 'Ascend':
        conclusion = '通用负载画像（目标非昇腾）。' + conclusion
    signals = []
    if compute:
        signals.append(dict(title='分类证据', detail=f'Cube {cube_pct:.2f}% / Vector {vector_pct:.2f}% / '
            f'混合 {metrics["mixed_pct"]:.2f}% / 未知 {metrics["unknown_pct"]:.2f}%；分母为计算kernel总时间。'))
        if mixed+unknown:
            signals.append(dict(title='未拆分时间的影响', detail=f'混合与未知合计 {percent(mixed+unknown,compute):.2f}%。'
                '未将这部分重新分配给Cube或Vector，可能改变负载倾向；分类覆盖率不等于预测覆盖率。'))
        signals.append(dict(title='短kernel', detail=f'{len(small)} 个kernel短于5μs，合计占计算时间 '
            f'{metrics["small_kernel_time_pct"]:.2f}%。建议检查启动、同步及融合机会；短耗时不等于小Shape或已确认启动受限。'))
        if fused:
            signals.append(dict(title='FA/HSTU融合路径', detail=f'按调用名识别的FA/HSTU占计算时间 {metrics["fused_time_pct"]:.2f}%。'
                '该指标与分类占比交叉，不能重复相加；需独立校准，未假定两侧等速。'))
    if cube:
        signals.append(dict(title='矩阵访存证据', detail=f'可判定目标解析模型计算/访存项的Cube时间覆盖率 '
            f'{metrics["cube_bound_coverage_pct"]:.2f}%。访存主导占全部Cube时间的可能范围 '
            f'{memory_lower:.2f}%–{memory_upper:.2f}%；范围来自未覆盖部分，不是统计置信区间或硬件实测。'))
    for key, label in [('memory_sum_us','内存活动'), ('communication_sum_us','通信活动')]:
        if summary.get(key, 0):
            signals.append(dict(title=label, detail=f'{summary[key]:.3f} μs，单独记账；不进入Cube/Vector分母，需按传输条件校准。'))
    busy = summary.get('step_busy_us')
    if summary.get('step_us') and busy is not None:
        outside = max(0., summary['step_us']-busy)
        signals.append(dict(title='Step内未被设备活动覆盖的时间', detail=f'{outside:.3f} μs '
            f'（{percent(outside,summary["step_us"]):.2f}%）。这是Step窗口减窗口内活动并集，不直接归因于CPU调度。'))
    priorities = defaultdict(lambda: dict(source_us=0., calls=0, reasons=set()))
    for u in units:
        p = priorities[u['op']]
        p['source_us'] += u['source_us']
        p['calls'] += 1
        if u.get('reason_code'):
            p['reasons'].add(u['reason_code'])
    suggestions = {
        'aten::addmm':'核对bias、alpha/beta和矩阵精度，补充addmm专用模型或同Shape校准。',
        'aten::cat':'核对TensorList、拼接维度及输出大小，再估计逻辑读写量。',
        'aten::copy_':'区分同精度复制、dtype转换和传输方向，分别校准。',
        'aten::bmm':'确认batch/MNK、稠密峰值和精度模式，检查有效带宽与矩阵效率。',
        'aten::mm':'确认MNK、稠密峰值和精度模式，检查有效带宽与矩阵效率。',
        'aten::native_layer_norm':'为归一化及其融合路径补充专用模型或实测校准。',
        'MEMORY_ACTIVITY':'核对传输方向、链路及是否与计算重叠，不直接套用显存带宽。',
    }
    top = []
    for op, p in sorted(priorities.items(), key=lambda item:item[1]['source_us'], reverse=True)[:5]:
        top.append(dict(op=op, source_us=p['source_us'], source_pct=percent(p['source_us'],summary['device_sum_us']),
            calls=p['calls'], reason_codes=sorted(p['reasons']), suggestion=suggestions.get(op,
                '核对输入Shape、精度、融合及实现路径，使用同签名实测校准验证。')))
    return dict(version=1, status=status, conclusion=conclusion, metrics=metrics, signals=signals, priorities=top,
        basis=summary['classification_basis'],
        limitation='画像初筛是排查方向，不代表已验证的昇腾亲和性或性能倍率。未知项保留，不沿用未知dtype默认TF32、缺规格默认等速的规则。',
        rules=dict(dominant_pct=50, cube_memory_pct=70, short_kernel_us=5,
                   note='经验阈值，仅用于初筛；分类占比分母为计算kernel时间，包含未知项。'))


def assessment(result, historical=False):
    p = result['prediction']
    ratio = p['modelled_device_ratio']
    coverage = p['coverage_pct']
    step = result.get('step_prediction')
    screening = screen(result['profile_summary'], result['units'], result['config']['target_hardware']['vendor'])
    partial_text = ('暂无可量化的局部倍率；查看未建模原因，画像初筛仍可使用。' if ratio is None else
        f'已覆盖源设备时间的 {coverage:.2f}%，该子集源/目标预测耗时比 {ratio:.3f}×；未覆盖项不按等速补齐，不能外推整步。')
    return dict(screening=screening,
        partial=dict(status='unavailable' if ratio is None else 'available', conclusion=partial_text,
            coverage_pct=coverage, ratio=ratio, covered_source_us=p['modelled_source_us'],
            uncovered_source_us=p['unmodelled_source_us'], predicted_us=p['predicted_modelled_sum_us']),
        step=dict(status='available' if step else 'unavailable',
            conclusion=p['verdict'] if step else '整步性能暂不可预测；这不否定已获得的画像与局部结论。',
            blockers=result.get('step_blockers', []), prediction=step),
        historical=historical,
        note='旧结果按已保存画像补充初筛展示，历史预测和导出不变；复用输入重新分析可应用新引擎。' if historical else '')

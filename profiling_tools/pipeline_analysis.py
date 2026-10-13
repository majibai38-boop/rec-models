"""Evidence-preserving PipeUtilization screening, not a Roofline diagnosis.

Ratios use their original core-cycle denominator. Never add pipeline durations,
infer bandwidth saturation, or use GPU alignment to decide an NPU bottleneck.
"""
from collections import defaultdict
import math

from .call_metadata import shapes, dtypes


THRESHOLD = .8
DOMINANCE_MARGIN = .15
LABELS = {
    'compute': '计算主导候选', 'memory': '搬运主导候选',
    'scalar': 'Scalar/Control 主导候选', 'mixed': '多流水/混合候选',
    'no_bound': 'No Bound 候选', 'unknown': 'Unknown',
}
ADVICE = {
    'compute': '补采 ArithmeticUtilization 或单算子指令分析；检查实际吞吐、指令依赖与资源冲突。',
    'memory': '补采 Memory/MemoryAccess/L2Cache；检查具体通路带宽、缓存命中及搬运粒度。',
    'scalar': '查看代码热点和指令流水；检查循环、地址计算、分支与指令下发。',
    'mixed': '查看 AIC/AIV 和单核流水时间线，确定重叠、同步及关键路径。',
    'no_bound': '查看流水空洞、同步依赖、核间负载与小任务固定开销；No Bound 不表示没有优化空间。',
    'unknown': '核对硬件/核类型和 PipeUtilization 采集配置，补齐有效指标后重试。',
}


def parse_ratio(value):
    """Only fractions or explicitly suffixed percentages; never guess a scale."""
    try:
        text = str(value).strip()
        number = float(text[:-1]) / 100 if text.endswith('%') else float(text)
        return number if math.isfinite(number) and 0 <= number <= 1 else None
    except (TypeError, ValueError):
        return None


def result(kind, reason, **extra):
    return {'category': kind, 'label': LABELS[kind], 'reason': reason,
            'confidence': 'insufficient' if kind == 'unknown' else 'heuristic',
            'next_check': ADVICE[kind], **extra}


def screen_side(raw, side):
    required = {'aiv': ('vec', 'scalar', 'mte2', 'mte3'),
                'aic': ('mac', 'scalar', 'mte1', 'mte2'),
                'legacy': ('vec', 'mac', 'scalar', 'mte1', 'mte2', 'mte3')}[side]
    prefix = '' if side == 'legacy' else side + '_'
    pipes = list(required)
    if side == 'aic':
        # Output path varies by hardware generation. Read every provided path.
        pipes += [p for p in ('fixpipe', 'mte3') if prefix + p + '_ratio' in raw]
    elif prefix + 'fixpipe_ratio' in raw:
        pipes.append('fixpipe')
    fields = {p: prefix + p + '_ratio' for p in pipes}
    values = {p: parse_ratio(raw.get(f)) for p, f in fields.items()}
    missing = [fields[p] for p, v in values.items() if v is None]
    if side == 'aic' and not any(p in values for p in ('fixpipe', 'mte3')):
        missing.append('aic_fixpipe_ratio 或 aic_mte3_ratio')
    observed = {p: v for p, v in values.items() if v is not None}
    highest = max(observed.values(), default=0)
    dominant = [fields[p] for p, v in observed.items() if v == highest] if highest else []
    evidence = {'side': side, 'ratios': {fields[p]: v for p, v in values.items()},
                'raw_values': {f: raw.get(f) for f in fields.values()},
                'missing_fields': missing, 'dominant_pipes': dominant,
                'dominant_ratio': highest if observed else None}
    if missing:
        return result('unknown', '适用指标缺失或无效：' + '、'.join(missing), **evidence)
    if not highest:
        return result('unknown', '适用流水指标全零，不能确认有效采集。', **evidence)
    if highest <= THRESHOLD:
        return result('no_bound', '所有适用流水占比均未超过 80%。', **evidence)
    families = {'compute': max((values.get(p, 0) for p in ('vec', 'mac')), default=0),
                'memory': max((values.get(p, 0) for p in ('mte1', 'mte2', 'mte3', 'fixpipe')), default=0),
                'scalar': values['scalar']}
    ordered = sorted(families, key=families.get, reverse=True)
    # Close competitors are retained rather than forced into one diagnosis.
    kind = 'mixed' if (families[ordered[1]] > THRESHOLD or
                       families[ordered[0]] - families[ordered[1]] < DOMINANCE_MARGIN) else ordered[0]
    answer = result(kind, '、'.join(dominant) + f' 最高，为 {highest:.1%}；仅为流水占用初筛。', **evidence)
    if kind == 'memory' and any('mte1' in p or 'fixpipe' in p for p in dominant):
        answer['next_check'] = '查看 L1/L0/Fixpipe 片上搬运与结果输出；结合单算子通路分析，不能直接归因于 HBM。'
    return answer


def classify(detail):
    base = {'method': 'pipe_utilization_v1', 'threshold': THRESHOLD,
            'dominance_margin': DOMINANCE_MARGIN, 'sides': []}
    if detail is None:
        return result('unknown', '缺少 kernel_details 数据。', **base)
    raw, core = detail.raw, detail.core_type.upper()
    if core in {'AI_VECTOR_CORE', 'AIV'}:
        sides = ['aiv']
    elif core in {'MIX_AIC', 'MIX_AIV', 'MIX'}:
        sides = ['aic', 'aiv']
    elif core in {'AI_CORE', 'AIC', 'AI_CUBE_CORE'}:
        sides = ['aic'] if any(k.startswith(('aic_', 'aiv_')) for k in raw) else ['legacy']
    else:
        return result('unknown', '该核类型尚不支持流水初筛：' + core, **base)
    # A prefix-free schema is supported only for a single core domain.
    if sides == ['aiv'] and not any(k.startswith('aiv_') for k in raw) and 'vec_ratio' in raw:
        raw = {'aiv_' + k: v for k, v in raw.items()}
    parts = [screen_side(raw, side) for side in sides]
    base['sides'] = parts
    if len(parts) == 1:
        return {**parts[0], **base}
    kinds = {p['category'] for p in parts}
    kind = 'unknown' if 'unknown' in kinds else next(iter(kinds)) if len(kinds) == 1 else 'mixed'
    return result(kind, '；'.join(p['side'].upper() + '：' + p['label'] for p in parts), **base)


def summarize(kernels):
    """Partition by per-kernel classification, using duration sums, not ratios."""
    buckets = {k: {'category': k, 'label': v, 'count': 0, 'time_us': 0.0} for k, v in LABELS.items()}
    for kernel in kernels:
        kind = kernel.get('pipeline', {}).get('category', 'unknown')
        bucket = buckets[kind]
        bucket['count'] += 1
        bucket['time_us'] += kernel['time_us']
    total = sum(b['time_us'] for b in buckets.values())
    for b in buckets.values():
        b['time_pct'] = 100 * b['time_us'] / total if total > 0 else None
    return {'count': sum(b['count'] for b in buckets.values()), 'time_us': total,
            'categories': [b for b in buckets.values() if b['count']]}


def build_pipeline_report(npu, rows):
    import json
    kernels, types, templates = [], defaultdict(list), defaultdict(list)
    seen = set()
    for row in rows:
        index = int(row['NPU Kernel Index'])
        if index in seen:
            raise ValueError('Duplicate NPU kernel in pipeline analysis')
        seen.add(index)
        detail = npu.kernels[index].detail
        item = {'index': index, 'name': npu.kernels[index].event.name,
                'op_type': row['OP Type'], 'device_id': row['Device_id'], 'core_type': row['Core Type'],
                'time_us': float(row['NPU Duration(us)']), 'pipeline': classify(detail)}
        item['metadata'] = {'input_shapes': shapes(detail.input_shapes), 'input_dtypes': dtypes(detail.input_dtypes),
                            'output_shapes': shapes(detail.output_shapes), 'output_dtypes': dtypes(detail.output_dtypes),
                            'input_formats': detail.raw.get('Input Formats', ''),
                            'output_formats': detail.raw.get('Output Formats', '')} if detail else {}
        kernels.append(item)
        key = (item['op_type'], item['device_id'], item['core_type'])
        types[key].append(item)
        templates[key + (json.dumps(item['metadata'], sort_keys=True),)].append(item)
    def group(items, include_shape=False):
        return {**{k: items[0][k] for k in ('op_type', 'device_id', 'core_type')},
                **({'metadata': items[0]['metadata']} if include_shape else {}), **summarize(items)}
    return {'version': 1, 'method': 'pipe_utilization_v1', 'threshold': THRESHOLD,
            'dominance_margin': DOMINANCE_MARGIN,
            'note': '流水初筛候选，非 Roofline 饱和结论；占比不可相加。分类耗时为 kernel duration 之和，非墙钟时间。',
            'summary': summarize(kernels), 'kernels': kernels,
            'types': [group(v) for v in types.values()],
            'shapes': [group(v, True) for v in templates.values()]}

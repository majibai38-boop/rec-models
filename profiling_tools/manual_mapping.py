"""Explicit, run-local human confirmations. Never changes CPU ownership."""
from collections import defaultdict
import json

from .call_metadata import compare_metadata, shapes, dtypes
from .device_compare import scoped_indices
from .model import bundle_metrics


def _metadata(profile, index):
    if index is None:
        return {"cpu_index": None, "cpu_name": "未知 owner", "module": "", "shapes": [], "dtypes": []}
    op = profile.ops[index]
    return {"cpu_index": index, "cpu_name": op.canonical_name, "module": op.module_key,
            "shapes": shapes(op.raw_input_shapes or op.input_shapes),
            "dtypes": dtypes(op.raw_input_dtypes or op.input_dtypes)}


def catalog(gpu, npu, matches, rows):
    """Small event-group inventory for indexed web browsing, not raw traces."""
    used = {i for r in rows for i in json.loads(r['GPU Kernel Indices'])}
    ns, gs, memory = scoped_indices(npu), scoped_indices(gpu), scoped_indices(gpu, True)
    by_npu = {m.npu_op_index: m for m in matches if m.confidence in {'high', 'medium'}}
    groups = defaultdict(list)
    for r in rows:
        if r['Comparable']:
            continue
        i = int(r['NPU Kernel Index'])
        owner = npu.kernels[i].owner_op_index
        groups[(r['Device_id'], r['OP Type'], r['Core Type'], owner if owner is not None else f'kernel:{i}')].append(i)
    targets = []
    candidate_ops = {k.owner_op_index for k in gpu.kernels if k.owner_op_index is not None}
    for key, indices in groups.items():
        owner = npu.kernels[indices[0]].owner_op_index
        aligned = []
        ancestor = owner
        while ancestor is not None:
            match = by_npu.get(ancestor)
            if match:
                candidate_ops.add(match.gpu_op_index)
                differences, missing = compare_metadata(npu.ops[ancestor], gpu.ops[match.gpu_op_index])
                # Strongest suggestion covers this exact NPU group, not sibling types/calls.
                if set(ns[ancestor]) == set(indices) and not differences and not missing:
                    aligned.append(f'op:{match.gpu_op_index}')
            ancestor = npu.ops[ancestor].parent_index
        targets.append({'id': f'npu:{indices[0]}', 'op_type': key[1], 'device_id': key[0], 'core_type': key[2],
                        'indices': indices, 'count': len(indices),
                        'time_us': sum(npu.kernels[i].event.duration_us for i in indices),
                        'names': sorted({npu.kernels[i].event.name for i in indices}),
                        'aligned': aligned,
                        'weak_owner': any(npu.kernels[i].link_method not in {'external_id','runtime_correlation','torch_to_npu','host_to_device_queue','host_to_device_thread'} for i in indices),
                        **_metadata(npu, owner)})
    candidates = []
    scopes = [(f'op:{i}', i, gs[i]) for i in sorted(candidate_ops)]
    scopes += [(f'kernel:{i}', None, [i]) for i,k in enumerate(gpu.kernels) if k.owner_op_index is None]
    for ident, owner, indices in scopes:
        if not indices or used.intersection(indices) or (owner is not None and memory[owner]):
            continue
        if len({(gpu.kernels[i].event.pid, str(gpu.kernels[i].event.args.get('device', ''))) for i in indices}) != 1:
            continue
        candidates.append({'id': ident, 'indices': indices, 'count': len(indices),
                           'time_us': sum(gpu.kernels[i].event.duration_us for i in indices),
                           'names': sorted({gpu.kernels[i].event.name for i in indices}),
                           'weak_owner': any(gpu.kernels[i].link_method not in {'external_id','runtime_correlation'} for i in indices),
                           **_metadata(gpu, owner)})
    return {'targets': targets, 'candidates': candidates}


def rank_candidate(target, candidate):
    reasons, warnings = [], []
    score = 0
    if candidate['id'] in target['aligned']:
        score = 100
        reasons.append('已对齐的 CPU 调用范围；完整输入元数据一致')
    else:
        for key, label, weight in [('module', '模块', 25), ('shapes', '输入 shape', 25), ('dtypes', '输入 dtype', 15)]:
            if not target[key] or not candidate[key]:
                warnings.append(label + '缺失')
            elif target[key] == candidate[key]:
                score += weight
                reasons.append(label + '一致')
            else:
                warnings.append(label + '不同')
        if target['cpu_name'] == candidate['cpu_name'] and target['cpu_index'] is not None:
            score += 20
            reasons.append('CPU op 名称一致')
    if not reasons:
        reasons.append('仅供人工检索，未发现可靠对应证据')
    if target.get('weak_owner') or candidate.get('weak_owner'):
        warnings.append('kernel owner 关联证据不足')
        score = min(score,60)
    return {**candidate, 'rank_score': score,
            'recommendation': '优先推荐' if score == 100 else '候选，需核实' if score >= 40 else '证据不足',
            'reasons': reasons, 'warnings': warnings}


def apply_links(gpu, npu, rows, links):
    by_index = {int(r['NPU Kernel Index']): r for r in rows}
    used_gpu = {i for r in rows for i in json.loads(r['GPU Kernel Indices'])}
    used_npu, link_ids = set(), set()
    for r in rows:
        r.update({'Manual Link ID': '', 'Manual Reason': ''})
    for link in links:
        ni, gi = link['npu_indices'], link['gpu_indices']
        if (not ni or not gi or len(set(ni)) != len(ni) or len(set(gi)) != len(gi)
                or any(type(i) is not int or i not in by_index for i in ni)
                or any(type(i) is not int or not 0 <= i < len(gpu.kernels) for i in gi)):
            raise ValueError('人工关联包含无效或重复的事件编号')
        if link['id'] in link_ids or not link['reason'].strip():
            raise ValueError('人工关联编号重复或缺少确认理由')
        selected = [by_index[i] for i in ni]
        if any(r['Comparable'] or json.loads(r['GPU Kernel Indices']) or json.loads(r['GPU Memory Activity Indices']) for r in selected):
            raise ValueError('NPU kernel 已有关联，不能覆盖')
        if used_npu.intersection(ni) or used_gpu.intersection(gi):
            raise ValueError('kernel 已被其他关联占用')
        if len({(r['Device_id'], r['OP Type'], r['Core Type']) for r in selected}) != 1:
            raise ValueError('仅支持单一 NPU 类型/设备/core 的人工关联')
        kernels = [gpu.kernels[i] for i in gi]
        if len({(k.event.pid, str(k.event.args.get('device', ''))) for k in kernels}) != 1:
            raise ValueError('不支持跨 GPU 设备关联')
        metrics = bundle_metrics(kernels)
        for r in selected:
            r.update({'Manual Link ID': link['id'], 'Manual Reason': link['reason'],
                      'Mapping Mode': 'manual_scope_shared_bundle', 'Mapped': True, 'Comparable': True,
                      'Accounted': True, 'Direct Kernel Pair': False, 'GPU Bundle Shared': True,
                      'GPU Kernel Source': 'manual_confirmation', 'GPU Kernel Count': len(gi),
                      'GPU Kernel Indices': json.dumps(gi), 'GPU Kernel Names': json.dumps([k.event.name for k in kernels]),
                      'GPU Kernel Owner OP Names': json.dumps([gpu.ops[k.owner_op_index].event.name if k.owner_op_index is not None else '' for k in kernels]),
                      'GPU Kernel Durations(us)': json.dumps([k.event.duration_us for k in kernels]),
                      'GPU Total Time(us)': round(metrics.sum_us, 6), 'GPU Kernel Time(us)': round(metrics.sum_us, 6),
                      'GPU Device Busy(us)': round(metrics.busy_us, 6), 'GPU Bundle Span(us)': round(metrics.span_us, 6),
                      'Performance Ratio(GPU/NPU)': '', 'GPU Activity Kind': 'kernel'})
        used_npu.update(ni)
        used_gpu.update(gi)
        link_ids.add(link['id'])

"""Explicit, run-local human confirmations. Never changes CPU ownership."""
from collections import Counter, defaultdict
import json

from .call_metadata import compare_metadata, shapes, dtypes
from .device_compare import scoped_indices
from .model import bundle_metrics


def _metadata(profile, index):
    if index is None:
        return {"cpu_index": None, "cpu_name": "未知 owner", "module": "", "shapes": [], "dtypes": []}
    op = profile.ops[index]
    path, ancestor = [], index
    while ancestor is not None:
        path.append(profile.ops[ancestor].canonical_name)
        ancestor = profile.ops[ancestor].parent_index
    differences, missing = compare_metadata(op, op)
    return {"cpu_index": index, "cpu_name": op.canonical_name, "module": op.module_key,
            "batch_context": {"path": list(reversed(path)), "pid": op.event.pid, "tid": op.event.tid,
                              "timestamp_us": op.event.timestamp_us, "metadata_complete": not differences and not missing},
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
                           'device_key': [gpu.kernels[indices[0]].event.pid, str(gpu.kernels[indices[0]].event.args.get('device', ''))],
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


def batch_plan(targets, candidates, target_id, candidate_id):
    """Expand a human-selected seed into independent call pairs, never a mega-bundle."""
    ts, cs = {r['id']: r for r in targets}, {r['id']: r for r in candidates}
    if target_id not in ts or candidate_id not in cs:
        raise ValueError('候选已不可用，请刷新后重试')
    seed_t, seed_c = ts[target_id], cs[candidate_id]
    if not seed_t.get('batch_context') or not seed_c.get('batch_context'):
        raise ValueError('旧报告缺少批量关联信息，请复用输入重新分析')

    def family(r, gpu=False):
        context = r.get('batch_context', {})
        # Thread and call path prevent crossing independent invocation sequences.
        return json.dumps([r['cpu_name'], r['module'], context.get('path'), context.get('pid'),
                           context.get('tid'), r['count'], r['names'] if gpu else
                           [r['op_type'], r['device_id'], r['core_type']],
                           r.get('device_key') if gpu else None], sort_keys=True)

    scoped_t = [r for r in targets if family(r) == family(seed_t)]
    scoped_c = [r for r in candidates if family(r, True) == family(seed_c, True)]
    reasons, pairs = {}, {}
    eligible_t = [r for r in scoped_t if not r.get('weak_owner') and r['batch_context']['metadata_complete']]
    eligible_ids = {r['id'] for r in eligible_t}
    eligible_c = {r['id']: r for r in scoped_c if not r.get('weak_owner') and r['batch_context']['metadata_complete']}
    for t in scoped_t:
        if t['id'] not in eligible_ids:
            reasons[t['id']] = 'owner 或输入元数据证据不足'
    # Only a unique, available aligned scope is automatically propagated.
    for t in eligible_t:
        options = [eligible_c[i] for i in t['aligned'] if i in eligible_c]
        if len(options) == 1:
            pairs[t['id']] = (t, options[0], 'aligned_cpu_scope')
        elif len(options) > 1:
            reasons[t['id']] = '存在多个已对齐 GPU 调用范围'
        else:
            reasons[t['id']] = '已对齐候选不可用或不符合模板' if t['aligned'] else '未建立唯一 CPU 对齐'

    def signature(r):
        return json.dumps([r['module'], r['shapes'], r['dtypes']], sort_keys=True)

    def order(r):
        return (r['batch_context']['timestamp_us'], r['cpu_index'])

    nparts, gparts = defaultdict(list), defaultdict(list)
    for r in eligible_t:
        nparts[signature(r)].append(r)
    for r in eligible_c.values():
        gparts[signature(r)].append(r)
    # A manual template can extend missing alignments only with equal complete
    # populations and order consistent with every existing alignment and the seed.
    for key, nt in nparts.items():
        gt = gparts[key]
        if len(nt) != len(gt):
            for t in nt:
                if not t['aligned']:
                    reasons[t['id']] = '同元数据调用数量不一致，不能按顺序批量配对'
            continue
        ordered = list(zip(sorted(nt, key=order), sorted(gt, key=order)))
        consistent = all((not t['aligned'] or g['id'] in t['aligned']) and
                         (t['id'] != target_id or g['id'] == candidate_id) for t, g in ordered)
        if not consistent:
            for t in nt:
                if not t['aligned']:
                    reasons[t['id']] = '调用顺序与已选配对或已有对齐冲突'
            continue
        for t, g in ordered:
            if not t['aligned']:
                pairs[t['id']] = (t, g, 'manual_template_order')
    # Remove all conflicting scopes rather than arbitrarily keeping the first.
    gpu_use = Counter(i for _, g, _ in pairs.values() for i in g['indices'])
    for ident, (_, g, _) in list(pairs.items()):
        if any(gpu_use[i] > 1 for i in g['indices']):
            reasons[ident] = 'GPU 调用范围重叠，不能重复分配'
            del pairs[ident]
    if target_id not in pairs or pairs[target_id][1]['id'] != candidate_id:
        raise ValueError('所选这一对不能安全扩展为批量模板；请核对推荐候选、完整元数据和调用顺序，或使用单组关联')
    selected = sorted(pairs.values(), key=lambda pair: order(pair[0]))
    if len(selected) > 5000:
        raise ValueError('本次模板超过 5000 对，请缩小调用范围后重新分析')
    result_pairs = [{'target_id': t['id'], 'candidate_id': g['id'], 'basis': basis,
                     'npu_indices': t['indices'], 'gpu_indices': g['indices'],
                     'npu_cpu_index': t['cpu_index'], 'gpu_cpu_index': g['cpu_index'],
                     'npu_time_us': t['time_us'], 'gpu_time_us': g['time_us']}
                    for t, g, basis in selected]
    excluded = [{'target_id': t['id'], 'cpu_index': t['cpu_index'], 'reason': reasons.get(t['id'], '不符合批量条件')}
                for t in scoped_t if t['id'] not in pairs]
    return {'pairs': result_pairs, 'scope_target_count': len(scoped_t), 'scope_candidate_count': len(scoped_c),
            'other_target_count': len(targets) - len(scoped_t), 'excluded_count': len(excluded),
            'excluded': excluded[:20], 'excluded_reasons': dict(Counter(r['reason'] for r in excluded)),
            'pair_count': len(selected), 'basis_counts': dict(Counter(p['basis'] for p in result_pairs)),
            'npu_kernel_count': sum(len(t['indices']) for t, _, _ in selected),
            'gpu_kernel_count': sum(len(g['indices']) for _, g, _ in selected),
            'npu_time_us': sum(t['time_us'] for t, _, _ in selected),
            'gpu_time_us': sum(g['time_us'] for _, g, _ in selected)}


def link_history(links):
    """Compact batch-level history; individual pairs stay in manual_links.json."""
    groups = {}
    for link in links:
        ident = link.get('batch_id') or link['id']
        if ident not in groups:
            groups[ident] = {**link, 'id': ident, 'pair_count': 0, 'npu_indices': [], 'gpu_indices': []}
        group = groups[ident]
        group['pair_count'] += 1
        group['npu_indices'].extend(link['npu_indices'])
        group['gpu_indices'].extend(link['gpu_indices'])
    return list(groups.values())


def apply_links(gpu, npu, rows, links):
    by_index = {int(r['NPU Kernel Index']): r for r in rows}
    used_gpu = {i for r in rows for i in json.loads(r['GPU Kernel Indices'])}
    used_npu, link_ids = set(), set()
    for r in rows:
        r.update({'Manual Link ID': '', 'Manual Reason': '', 'Manual Batch ID': '', 'Manual Pair Basis': ''})
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
                      'Manual Batch ID': link.get('batch_id', ''), 'Manual Pair Basis': link.get('basis', ''),
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

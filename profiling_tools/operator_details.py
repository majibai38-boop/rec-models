"""Read-only detail view of established mappings, independent of diagnostics.

This does not change alignment or type accounting. Shared GPU sets form one
comparison unit; overlapping sets or cross-type sharing have no standalone ratio.
"""
from collections import defaultdict
import hashlib
import json

from .call_metadata import shapes, dtypes
from .parser import get_arg


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def identity(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()[:24]


def ids(row, field):
    return tuple(sorted(set(json.loads(row.get(field, '[]')))))


def tensor_metadata(inputs=None, input_types=None, outputs=None, output_types=None):
    input_types = encoded(input_types) if isinstance(input_types, (list, tuple)) else input_types
    output_types = encoded(output_types) if isinstance(output_types, (list, tuple)) else output_types
    return {'input_shapes': shapes(inputs), 'input_dtypes': dtypes(input_types),
            'output_shapes': shapes(outputs), 'output_dtypes': dtypes(output_types)}


def cpu_metadata(profile, index):
    if index is None:
        return None
    op = profile.ops[index]
    path, current = [], index
    while current is not None:
        path.append(profile.ops[current].canonical_name)
        current = profile.ops[current].parent_index
    return {'index': index, 'name': op.event.name, 'module': op.module_key,
            'path': list(reversed(path)), 'pid': op.event.pid, 'tid': op.event.tid,
            'timestamp_us': op.event.timestamp_us, 'source': 'cpu_op',
            **tensor_metadata(op.raw_input_shapes or op.input_shapes, op.raw_input_dtypes or op.input_dtypes,
                              get_arg(op.event.args, 'Output Dims', 'Output Shapes', 'output_shapes'),
                              get_arg(op.event.args, 'Output type', 'Output Types', 'Output Data Types'))}


def kernel_metadata(profile, index, memory=False):
    k = (profile.memory_activities if memory else profile.kernels)[index]
    d = k.detail
    return {'index': index, 'name': k.event.name, 'kind': 'memory' if memory else 'kernel',
            'time_us': k.event.duration_us, 'timestamp_us': k.event.timestamp_us,
            'owner_index': k.owner_op_index, 'stream': k.stream_id,
            'pid': k.event.pid, 'device': d.device_id if d else str(k.event.args.get('device', '')),
            'source': 'kernel_details' if d else 'not_collected',
            **(tensor_metadata(d.input_shapes, d.input_dtypes, d.output_shapes, d.output_dtypes)
               if d else tensor_metadata())}


def signature(meta):
    # Exclude instance IDs and timestamps, never collapse tensor argument order.
    return {k: meta[k] for k in ('name', 'source', 'input_shapes', 'input_dtypes',
                                'output_shapes', 'output_dtypes', 'module', 'path', 'pid', 'tid', 'device') if k in meta}


def build_operator_details(gpu, npu, matches, rows):
    groups, users = defaultdict(list), defaultdict(set)
    match_by_id = {str(m.match_id): m for m in matches}
    for row in rows:
        gi, mi = ids(row, 'GPU Kernel Indices'), ids(row, 'GPU Memory Activity Indices')
        key = (row['OP Type'], row['Device_id'], row['Core Type'], gi, mi,
               None if gi or mi else int(row['NPU Kernel Index']))
        groups[key].append(row)
        for kind, indices in (('kernel', gi), ('memory', mi)):
            for i in indices:
                users[kind, i].add(key)
    units = []
    for key, members in groups.items():
        op_type, device, core, gi, mi, _ = key
        ni = sorted(int(r['NPU Kernel Index']) for r in members)
        nk = [kernel_metadata(npu, i) for i in ni]
        gk = [kernel_metadata(gpu, i) for i in gi] + [kernel_metadata(gpu, i, True) for i in mi]
        owners_n = sorted({k['owner_index'] for k in nk if k['owner_index'] is not None})
        owners_g = sorted({k['owner_index'] for k in gk if k['owner_index'] is not None})
        scopes = {(match_by_id[str(r['CPU Match ID'])].npu_op_index,
                   match_by_id[str(r['CPU Match ID'])].gpu_op_index)
                  for r in members if not r.get('Manual Link ID') and str(r.get('CPU Match ID')) in match_by_id}
        nscope = [cpu_metadata(npu, i) for i in sorted({a for a, _ in scopes})]
        gscope = [cpu_metadata(gpu, i) for i in sorted({b for _, b in scopes})]
        nowners = [cpu_metadata(npu, i) for i in owners_n]
        gowners = [cpu_metadata(gpu, i) for i in owners_g]
        shared = any(len(users[kind, i]) > 1 for kind, indices in (('kernel', gi), ('memory', mi)) for i in indices)
        comparable = bool(gk) and all(r['Comparable'] for r in members) and not shared
        ntime, gtime = sum(k['time_us'] for k in nk), sum(k['time_us'] for k in gk)
        status = '共享 GPU 活动，不能独立归因' if shared else '可比较' if comparable else '未独立对齐'
        unit = {'id': 'unit:' + str(ni[0]), 'op_type': op_type, 'device_id': device, 'core_type': core,
                'npu_kernels': nk, 'gpu_kernels': gk, 'npu_owners': nowners, 'gpu_owners': gowners,
                'npu_scopes': nscope, 'gpu_scopes': gscope,
                'npu_time_us': ntime, 'gpu_time_us': gtime if gk else None,
                'comparable': comparable, 'ratio': gtime / ntime if comparable and ntime > 0 else None,
                'status': status, 'granularity': 'kernel' if len(nk) == len(gk) == 1 else 'bundle',
                'basis': sorted({r['Mapping Mode'] for r in members}),
                'manual': any(r.get('Manual Link ID') for r in members),
                'manual_reasons': sorted({r['Manual Reason'] for r in members if r.get('Manual Reason')}),
                'timestamp_us': min(k['timestamp_us'] for k in nk)}
        template = {field: [signature(v) for v in unit[field]] for field in
                    ('npu_kernels', 'gpu_kernels', 'npu_owners', 'gpu_owners', 'npu_scopes', 'gpu_scopes')}
        unit['group_id'] = identity([op_type, device, core, unit['granularity'], template])
        units.append(unit)
    families = defaultdict(list)
    templates = defaultdict(list)
    for u in units:
        families[u['op_type'], u['device_id'], u['core_type']].append(u)
        templates[u['group_id']].append(u)
    for buckets, field in ((families, 'position'), (templates, 'template_position')):
        for items in buckets.values():
            items.sort(key=lambda u: (u['timestamp_us'], int(u['id'].split(':')[1])))
            for i, u in enumerate(items, 1):
                u[field] = {'index': i, 'total': len(items)}
    return {'version': 1, 'units': units}

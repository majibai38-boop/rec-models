"""Small user-facing per-device exports; internal comparison evidence stays intact."""
import csv
import math
import json
from collections import defaultdict
from pathlib import Path

from .parser import _find_selected_step, _step_matches, get_arg


NPU_STATS = ['Device_id', 'OP Type', 'Core Type', 'Count', 'Total Time(us)',
             'Min Time(us)', 'Avg Time(us)', 'Max Time(us)', 'Ratio(%)']
GPU_DETAILS = ['Name', 'Device_id', 'Stream ID', 'Start Time(us)', 'Duration(us)',
               'CPU Input Shapes', 'CPU Input Dtypes', 'Metadata Source']
GPU_KEYS = ['Name', 'Device_id', 'CPU Input Shapes', 'CPU Input Dtypes', 'Metadata Source']
GPU_STATS = GPU_KEYS + ['Count', 'Total Time(us)', 'Min Time(us)', 'Avg Time(us)', 'Max Time(us)', 'Ratio(%)']


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def statistics(rows, keys):
    groups, totals = defaultdict(list), defaultdict(list)
    for row in rows:
        value = row['Duration(us)']
        groups[tuple(row[key] for key in keys)].append(value)
        if value is not None:
            totals[row['Device_id']].append(value)
    denominators = {device: math.fsum(values) for device, values in totals.items()}
    incomplete_devices = {r['Device_id'] for r in rows if r['Duration(us)'] is None}
    result = []
    for key, values in groups.items():
        row = dict(zip(keys, key))
        valid = [v for v in values if v is not None]
        complete = len(valid) == len(values)
        total = math.fsum(valid) if complete else None
        denominator = 0 if row['Device_id'] in incomplete_devices else denominators.get(row['Device_id'], 0)
        row.update({'Count': len(values), 'Total Time(us)': round(total, 3) if complete else '',
                    'Min Time(us)': round(min(valid), 3) if complete else '',
                    'Avg Time(us)': round(total/len(values), 3) if complete else '',
                    'Max Time(us)': round(max(valid), 3) if complete else '',
                    'Ratio(%)': round(total/denominator*100, 3) if complete and denominator else ''})
        result.append(row)
    return sorted(result, key=lambda r: (str(r['Device_id']), -(r['Total Time(us)'] or 0), str(r[keys[0]])))


def write_download_csvs(output, gpu, npu, details, source_csv, npu_step):
    output = Path(output)
    window = _find_selected_step(npu.trace_events, npu.selected_pid, npu_step)[0] if npu_step else None
    chosen, unknown = [], 0
    for detail in details:
        if npu_step:
            if detail.step_id:
                if not _step_matches(detail.step_id, npu_step):
                    continue
            elif detail.start_us is None:
                unknown += 1
                continue
            elif not window[0] <= detail.start_us < window[1]:
                continue
        chosen.append(detail)
    with Path(source_csv).open(encoding='utf-8-sig', newline='') as stream:
        fields = next(csv.reader(stream))
    write_csv(output/'npu_kernel_details.csv', fields, [d.raw for d in chosen])
    npu_rows = [{'Device_id':d.device_id, 'OP Type':d.op_type, 'Core Type':d.core_type,
                 'Duration(us)':d.duration_us} for d in chosen]
    write_csv(output/'npu_op_statistic.csv', NPU_STATS, statistics(npu_rows, NPU_STATS[:3]))
    gpu_rows = []
    for kernel in gpu.kernels:
        owner = gpu.ops[kernel.owner_op_index] if kernel.owner_op_index is not None else None
        device = get_arg(kernel.event.args, 'device', 'Device ID', 'device_id')
        gpu_rows.append({'Name':kernel.event.name,
            'Device_id':str(device) if device is not None else 'pid:'+kernel.event.pid,
            'Stream ID':kernel.stream_id, 'Start Time(us)':kernel.event.timestamp_us,
            'Duration(us)':kernel.event.duration_us, 'CPU Input Shapes':owner.input_shapes if owner else '',
            'CPU Input Dtypes':owner.input_dtypes if owner else '', 'Metadata Source':'CPU owner' if owner else 'unknown'})
    write_csv(output/'gpu_kerenel_details.csv', GPU_DETAILS, gpu_rows)
    write_csv(output/'gpu_op_statistic.csv', GPU_STATS, statistics(gpu_rows, GPU_KEYS))
    return {'reports':['npu_kernel_details.csv','npu_op_statistic.csv','gpu_kerenel_details.csv','gpu_op_statistic.csv'],
            'unknown_step_rows':unknown}


def write_selected_trace(path, profile, requested_step):
    """Preserve original event fields and trace metadata in a scoped Trace JSON."""
    window = _find_selected_step(profile.trace_events, profile.selected_pid, requested_step)[0] if requested_step else None
    activities = profile.kernels + profile.memory_activities
    pids = {profile.selected_pid} | {k.event.pid for k in activities}
    explicit = {o.event.index for o in profile.ops} | {k.event.index for k in activities if k.event.index >= 0}
    selected = {e.index for e in profile.trace_events if e.index in explicit or
                (e.pid in pids and (e.phase.upper() == 'M' or window is None or window[0] <= e.timestamp_us < window[1]))}
    # Serialize incrementally to avoid creating a second encoded copy of the trace.
    with Path(profile.source_path).open(encoding='utf-8-sig') as stream:
        payload = json.load(stream)
    raw = payload.get('traceEvents', []) if isinstance(payload, dict) else payload
    with Path(path).open('w', encoding='utf-8') as stream:
        stream.write('{')
        if isinstance(payload, dict):
            for key, value in payload.items():
                if key != 'traceEvents':
                    stream.write(json.dumps(key)+':')
                    json.dump(value, stream, ensure_ascii=False)
                    stream.write(',')
        stream.write('"traceEvents":[')
        first = True
        for index, event in enumerate(raw):
            if index not in selected:
                continue
            if not first:
                stream.write(',')
            json.dump(event, stream, ensure_ascii=False)
            first = False
        stream.write(']}')

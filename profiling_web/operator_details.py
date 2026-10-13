"""Indexed shape groups and bounded detail queries; no trace IO on GET."""
from collections import defaultdict
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from profiling_tools.pipeline_analysis import summarize


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    at = (len(values) - 1) * p
    lo = int(at)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (at - lo)


def group_summary(units):
    first = units[0]
    pairs = [u for u in units if u['comparable']]
    nt = sum(u['npu_time_us'] for u in units)
    cn = sum(u['npu_time_us'] for u in pairs)
    cg = sum(u['gpu_time_us'] for u in pairs)
    nvalues, gvalues = [u['npu_time_us'] for u in pairs], [u['gpu_time_us'] for u in pairs]
    return {**{k: first[k] for k in ('group_id', 'op_type', 'device_id', 'core_type', 'granularity')},
            'pipeline': summarize(k for u in units for k in u['npu_kernels'])
                        if any('pipeline' in k for u in units for k in u['npu_kernels']) else None,
            'npu_metadata': [{k: v for k, v in r.items() if k in ('name', 'source', 'input_shapes', 'input_dtypes', 'output_shapes', 'output_dtypes')} for r in first['npu_kernels']],
            'gpu_names': [r['name'] for r in first['gpu_kernels']],
            'npu_cpu': first['npu_scopes'] or first['npu_owners'],
            'gpu_cpu': first['gpu_scopes'] or first['gpu_owners'],
            'count': len(units), 'comparable_count': len(pairs),
            'npu_kernel_count': sum(len(u['npu_kernels']) for u in units),
            'npu_time_us': nt, 'comparable_npu_time_us': cn, 'comparable_gpu_time_us': cg if pairs else None,
            'npu_mean_us': cn / len(pairs) if pairs else None,
            'gpu_mean_us': cg / len(pairs) if pairs else None,
            'npu_p50_us': percentile(nvalues, .5), 'gpu_p50_us': percentile(gvalues, .5),
            'npu_p95_us': percentile(nvalues, .95), 'gpu_p95_us': percentile(gvalues, .95),
            'ratio': cg / cn if cn > 0 else None, 'coverage_pct': 100 * cn / nt if nt > 0 else None,
            'statuses': sorted({u['status'] for u in units}), 'manual_count': sum(u['manual'] for u in units)}


def build_details_index(db, path):
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('version') != 1:
        raise ValueError('Unsupported operator detail version')
    db.executescript('''
        CREATE TABLE detail_units(id TEXT PRIMARY KEY, group_id TEXT, op_type TEXT, device_id TEXT,
            core_type TEXT, npu_time REAL, position INTEGER, payload TEXT);
        CREATE INDEX detail_unit_group ON detail_units(group_id,position);
        CREATE INDEX detail_unit_type ON detail_units(op_type,device_id,core_type,position);
        CREATE TABLE detail_groups(id TEXT PRIMARY KEY, op_type TEXT, device_id TEXT, core_type TEXT,
            npu_time REAL, ratio REAL, search TEXT, payload TEXT);
        CREATE INDEX detail_group_type ON detail_groups(op_type,device_id,core_type,npu_time);
    ''')
    groups = defaultdict(list)
    for u in report['units']:
        groups[u['group_id']].append(u)
        db.execute('INSERT INTO detail_units VALUES(?,?,?,?,?,?,?,?)',
                   (u['id'], u['group_id'], u['op_type'], u['device_id'], u['core_type'], u['npu_time_us'],
                    u['position']['index'], json.dumps(u, ensure_ascii=False)))
    for group_id, units in groups.items():
        s = group_summary(units)
        search = json.dumps([s['npu_metadata'], s['gpu_names'], s['npu_cpu'], s['gpu_cpu']], ensure_ascii=False)
        db.execute('INSERT INTO detail_groups VALUES(?,?,?,?,?,?,?,?)',
                   (group_id, s['op_type'], s['device_id'], s['core_type'], s['npu_time_us'], s['ratio'], search,
                    json.dumps(s, ensure_ascii=False)))


def query_details(path, *, op_type, device_id, core_type, group_id='', unit_id='',
                  search='', sort='npu_time', offset=0, limit=30, view='groups'):
    if offset < 0 or not 1 <= limit <= 200 or sort not in {'npu_time', 'ratio', 'position'} or view not in {'groups', 'units'}:
        raise ValueError('Invalid detail query')
    empty = {'supported': False, 'items': [], 'total': 0, 'has_more': False, 'offset': offset, 'limit': limit}
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='detail_units'").fetchone():
            return empty
        table = 'detail_groups' if view == 'groups' else 'detail_units'
        where = 'op_type=? AND device_id=? AND core_type=?'
        params = [op_type, device_id, core_type]
        if group_id:
            where += ' AND ' + ('id' if view == 'groups' else 'group_id') + '=?'
            params.append(group_id)
        if unit_id:
            if view != 'units':
                raise ValueError('Unit filter requires units view')
            where += ' AND id=?'
            params.append(unit_id)
        if search:
            where += " AND instr(lower(" + ('search' if view == 'groups' else 'payload') + "),lower(?))>0"
            params.append(search)
        order = 'npu_time DESC' if sort == 'npu_time' else ('ratio DESC' if view == 'groups' else "json_extract(payload,'$.ratio') DESC") if sort == 'ratio' else ('id' if view == 'groups' else 'position')
        count = db.execute(f'SELECT count(*) FROM {table} WHERE {where}', params).fetchone()[0]
        rows = db.execute(f'SELECT payload FROM {table} WHERE {where} ORDER BY {order},id LIMIT ? OFFSET ?', (*params, limit, offset))
        result = {**empty, 'supported': True, 'total': count, 'has_more': offset + limit < count,
                  'items': [json.loads(r[0]) for r in rows]}
        if view == 'groups':
            # Whole-filter metrics, not the visible page. Only independent pairs contribute GPU time.
            sums = db.execute(f'''SELECT sum(npu_time),sum(json_extract(payload,'$.comparable_npu_time_us')),
                sum(json_extract(payload,'$.comparable_gpu_time_us')),sum(json_extract(payload,'$.count')),
                sum(json_extract(payload,'$.comparable_count')) FROM {table} WHERE {where}''', params).fetchone()
            result['summary'] = dict(zip(('npu_time_us', 'comparable_npu_time_us', 'comparable_gpu_time_us', 'count', 'comparable_count'), sums))
            categories = db.execute(f'''SELECT json_extract(j.value,'$.category'),
                json_extract(j.value,'$.label'),sum(json_extract(j.value,'$.count')),
                sum(json_extract(j.value,'$.time_us'))
                FROM (SELECT payload FROM {table} WHERE {where}) AS filtered,
                json_each(filtered.payload,'$.pipeline.categories') AS j
                GROUP BY json_extract(j.value,'$.category')''', params).fetchall()
            total = sum(r[3] for r in categories)
            result['summary']['pipeline'] = {'count': sum(r[2] for r in categories), 'time_us': total,
                'categories': [dict(category=r[0], label=r[1], count=r[2], time_us=r[3],
                                    time_pct=100*r[3]/total if total else None) for r in categories]} if categories else None
        return result

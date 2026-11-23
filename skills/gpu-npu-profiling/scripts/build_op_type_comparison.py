"""Render a descriptive type/caller comparison from completed profiling results."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import re
import sys


# FAMILY CONFIG: recommendation-model conventions, not engine matching rules.
GEMM_CALLERS = {'aclnnMm', 'aclnnAddmm', 'aclnnMatmul'}
GEMM_GPU = ('aten::mm', 'aten::addmm', 'aten::bmm')
COMPONENTS = {
    'radix': 'cub::DeviceRadixSort', 'scan': 'cub::DeviceScan', 'select': 'cub::DeviceSelect',
    'linearize': 'fbgemm_gpu::linearize_index_wo_infos', 'unique': 'unique_indices_length',
    'delinearize': 'delinearize_unique_index', 'hash': 'cuco', 'optimizer': 'multi_tensor_apply',
    'index_add': 'indexFuncLargeIndex', 'gather': 'vectorized_gather_kernel',
}
DOMAIN_PATTERNS = {'fbgemm 域': 'fbgemm_gpu::'}  # domain evidence overlaps components; never sum into TOTAL
TYPE_RULES = {
    'MatMulV3': ('family', ''), 'BatchMatMulV3': ('family', ''),
    'InplaceIndexAdd': ('caller', 'aten::index_add_'), 'Sort': ('component', 'radix'),
    'JaggedUniqueIndicesLength': ('component', 'unique'), 'LinearizeIndex': ('component', 'linearize'),
    'DelinearizeUniqueIndex': ('component', 'delinearize'), 'GatherV2': ('route', 'gather'),
    'Addcdiv': ('route', 'optimizer'), **dict.fromkeys(('ForeachCopy', 'TensorMove', 'ViewCopy',
        'find_with_digest_kernel', 'insert_or_assign_kernel'), ('none', '无独立 GPU 对应')),
}
CUSTOM_PREFIX = '_ZN'
# label, NPU caller keys, GPU caller keys, interpretation. Unlisted callers pass through.
CALLER_ROWS = [
    ('index_add', ('aten::index_add', 'aten::index_add_'), ('aten::index_add_', 'aten::index_add'), '命名不对称'),
    ('bmm', ('aten::matmul', 'aten::matmul_backward'), ('aten::bmm',), '前向及反向汇总'),
    *((name, tuple('aten::'+s for s in keys), tuple('aten::'+s for s in keys), '调用方合并')
      for name, keys in [('add/add_', ('add', 'add_')), ('clamp/clamp_', ('clamp', 'clamp_')),
                         ('pow/log/rsub/le', ('pow', 'log', 'rsub', 'le'))]),
    ('t/view/transpose', ('aten::t', 'aten::view', 'aten::transpose'),
     ('aten::t', 'aten::view', 'aten::transpose'), '—'),
    ('multi_tensor_adam', ('aten::multi_tensor_adam', 'multi_tensor_adam'), (), '无独立 GPU 对应'),
    ('gather', ('aten::gather',), ('aten::gather',), '路线不同'),
    ('_foreach_copy_', ('aten::_foreach_copy_',), (), 'FSDP 拷贝；GPU DtoD 单列'),
]
CHAIN_GROUPS = [
    ('排序去重', ('Sort',), ('radix', 'scan', 'select'), '组件对照'),
    ('jagged 三件套', ('LinearizeIndex', 'JaggedUniqueIndicesLength', 'DelinearizeUniqueIndex'),
     ('linearize', 'unique', 'delinearize'), '组件对照'),
    ('哈希查表插入', ('find_with_digest_kernel', 'insert_or_assign_kernel'), ('hash',), '路线不同'),
    ('自定义 ascendc', ('custom_ascendc',), ('fbgemm 域',), '域级参考（无独立对应）'),
]
REG_ROWS = ('MatMulV3', 'BatchMatMulV3', 'InplaceIndexAdd', 'Sort', 'ForeachCopy', 'TensorMove',
            'JaggedUniqueIndicesLength', 'LinearizeIndex', 'DelinearizeUniqueIndex', 'GatherV2', 'Addcdiv')
COPY_TYPES = ('ForeachCopy', 'TensorMove', 'ViewCopy')
# END FAMILY CONFIG


def records(root, filename, fields):
    with (root / filename).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not set(fields) <= set(reader.fieldnames or ()):
            raise ValueError(f'{root / filename}: 缺少必需列 {set(fields) - set(reader.fieldnames or ())}')
        yield from reader


def number(value):
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f'非法计数或耗时: {value}')
    return result


def ms(us):
    return number(us) / 1000


def ratio(gpu, npu):
    return gpu / npu if gpu is not None and npu else None


def reconcile(name, actual, expected, count):
    # Engine CSVs round µs per row; allow at most half of a last-place unit per row.
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=(count+1)*0.000000000501):
        raise ValueError(f'{name} 无法与 summary.json 对账: {actual:g} vs {expected:g} ms')


@dataclass
class Work:
    count: int = 0
    raw: float = 0.0
    active: float = 0.0
    missing: int = 0
    callers: Counter = field(default_factory=Counter)

    @property
    def busy(self):
        return self.active if self.count and not self.missing else None

    def add(self, raw, busy, caller):
        self.count += 1
        self.raw += raw
        self.active += busy or 0
        self.missing += busy is None
        self.callers[caller] += 1


def pick(row, *names):
    return next((row[k].strip() for k in names if row.get(k, '').strip()), '')


def pmu(detail, key):
    v = pick(detail, *(key+s for s in ('(us)', '(μs)', '(µs)', '')))
    return None if v.lower() in ('', 'n/a', 'na', 'none', '--') else ms(v)


def npu_work(root, groups, total, gemm):
    mappings = records(root, 'npu_kernel_gpu_mapping.csv', ('OP Type', 'NPU Kernel Name', 'NPU Duration(us)'))
    details = records(root, 'npu_kernel_details.csv', ())
    for i, (m, d) in enumerate(zip_longest(mappings, details), 1):
        if m is None or d is None:
            raise ValueError(f'{root}: NPU mapping/details 行数不一致')
        kind, name = pick(d, 'Type', 'OP Type'), pick(d, 'Name', 'Kernel Name', 'OP Name')
        ids = [(m.get(a, ''), pick(d, *aliases)) for a, aliases in (
            ('Device_id', ('Device_id', 'Device ID')), ('NPU Task ID', ('Task ID', 'Task Id')),
            ('NPU Stream ID', ('Stream ID', 'Stream Id')))]
        # Trace and detail kernel names may differ. Require both stream and task in that case.
        if (kind != m['OP Type'] or any(x != y for x, y in ids if x and y)
                or (name != m['NPU Kernel Name'] and not all(x and y for x, y in ids[1:]))):
            raise ValueError(f'{root}: NPU mapping/details 行 {i} 身份不一致；不猜测重排')
        raw = ms(m['NPU Duration(us)'])
        vec, core = pmu(d, 'aiv_time'), pmu(d, 'aicore_time')
        busy = None if vec is None or core is None else max(vec, core)
        if busy == 0:
            busy = ms(pick(d, 'Duration(us)', 'Duration(μs)', 'Duration(µs)'))
        prefix = next((match.group() for text in (name, m['NPU Kernel Name'])
                       if (match := re.match(r'aclnn[A-Za-z0-9]+', text))), '')
        caller = prefix or m.get('NPU CPU OP Name') or '未知'
        label = 'custom_ascendc' if kind.startswith(CUSTOM_PREFIX) else kind
        groups[label].add(raw, busy, caller)
        total.add(raw, busy, caller)
        if prefix in GEMM_CALLERS:
            gemm.add(raw, busy, caller)


INPUT_CSVS = ('npu_kernel_gpu_mapping.csv', 'npu_kernel_details.csv', 'npu_gpu_op_statistic.csv',
              'aten_op_statistic.csv', 'gpu_kerenel_details.csv', 'gpu_memory_activity.csv', 'whole_call_mapping.csv')


def check_manifest(root, summary):
    manifest = json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig'))
    if manifest['run_id'] != summary['run_id']:
        raise ValueError(f'{root}: manifest 与 summary 的 run_id 不一致')
    for name in INPUT_CSVS:
        digest = hashlib.sha256()
        with (root/name).open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                digest.update(block)
        if digest.hexdigest() != manifest['artifacts'][name]['sha256']:
            raise ValueError(f'{root/name}: 引擎校验后源文件发生变化')
    return {k: manifest['inputs'][k]['sha256'] for k in ('gpu_trace', 'npu_trace', 'npu_kernel_details')}


def load(root, all_steps):
    root = Path(root).resolve()
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8-sig'))
    if summary.get('validation', {}).get('status') != 'valid' or summary.get('status') not in ('complete', 'complete_with_gaps'):
        raise ValueError(f'{root}: 需要已完成且通过引擎校验的 result')
    identities = check_manifest(root, summary)
    for side in ('npu', 'gpu'):
        step = summary[side].get('selected_step')
        if not step or (step == 'all') != all_steps:
            raise ValueError(f'{root}: {side} Step 范围与命令参数不一致 ({step})')
    audit = summary['gpu_accounting']
    types, total, gemm = defaultdict(Work), Work(), Work()
    npu_work(root, types, total, gemm)
    reconcile('NPU kernel', total.raw, ms(audit['npu_total_time_us']), total.count)
    if total.count != summary['comparison']['npu_first_kernel_count']:
        raise ValueError(f'{root}: NPU kernel 数量无法对账')
    alloc = Counter()
    for r in records(root, 'npu_gpu_op_statistic.csv', ('Row Kind', 'OP Type', 'GPU Total Time(us)')):
        if r['Row Kind'] == 'npu_type' and (int(r['GPU Kernel Count']) or int(r['GPU Memory Activity Count'])):
            alloc[r['OP Type']] += ms(r['GPU Total Time(us)'])
    aten = {}
    for r in records(root, 'aten_op_statistic.csv', ('OP Type',)):
        if r['OP Type'] in aten:
            raise ValueError(f'ATen 重复行: {r["OP Type"]}')
        aten[r['OP Type']] = {s: (int(r[s+' Raw Kernel Count']), ms(r[s+' Total Time(us)'])) for s in ('NPU','GPU')}
    components, memory = Counter(), Counter()
    for section, file in [('kernel', 'gpu_kerenel_details.csv'), ('memory', 'gpu_memory_activity.csv')]:
        count, elapsed = 0, 0.0
        for r in records(root, file, ('Name',)):
            t = ms(r['Duration(us)'])
            count += 1
            elapsed += t
            if section == 'memory':
                memory[r['Name']] += t
            else:
                hits = [k for k, token in COMPONENTS.items() if token in r['Name']]
                if len(hits) > 1:
                    raise ValueError(f'GPU kernel 命中多个组件族 {hits}，请复核 FAMILY CONFIG')
                if hits:
                    components[hits[0]] += t
                for domain, token in DOMAIN_PATTERNS.items():
                    if token in r['Name']:
                        components[domain] += t
        if count != audit[section]['total_count']:
            raise ValueError(f'{root}: GPU {section} 数量无法对账')
        reconcile(f'GPU {section}', elapsed, ms(audit[section]['total_time_us']), count)
    conflicts, missing, whole = Counter(), Counter(), 0
    for r in records(root, 'whole_call_mapping.csv', ('Whole OP',)):
        whole += 1
        for issue in r['Issues'].split('；'):
            if 'input_dtypes' in issue:
                if issue.startswith('冲突：'):
                    conflicts[r['Whole OP']] += 1
                elif issue.startswith('缺失：'):
                    missing[r['Whole OP']] += 1
    return Run(root, summary, identities, types, total, gemm, alloc, aten, components, memory, conflicts, missing, whole)


@dataclass
class Run:
    root: Path
    summary: dict
    identities: dict
    types: dict
    total: Work
    gemm: Work
    alloc: dict
    aten: dict
    components: dict
    memory: dict
    conflicts: Counter
    missing: Counter
    whole: int

    @property
    def gpu(self):
        return ms(self.summary['gpu_accounting']['kernel']['total_time_us'])

    def callers(self, side, names):
        vals = [self.aten[n][side] for n in names if n in self.aten and self.aten[n][side][0]]
        return (sum(c for c, _ in vals), sum(t for _, t in vals) if vals else None)

    def component(self, keys):
        vals = [self.components[k] for k in keys if k in self.components]
        return sum(vals) if vals else None


# Each section has one schema shared by Markdown and Excel. None is unknown/not applicable.
@dataclass
class Table:
    name: str
    columns: tuple
    rows: list = field(default_factory=list)


TYPE_COLUMNS = ('类型', 'NPU kernels', 'NPU raw (ms)', 'NPU 去扰 (ms)', 'GPU (ms)',
                'GPU/NPU', 'GPU/NPU 去扰', '对照来源', 'NPU 调用方（次数）')


def census(counts):
    return '、'.join(f'{k} ({v})' for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))) or '—'


def type_table(name, run, order):
    table = Table(name, TYPE_COLUMNS)

    def append(label, w, gpu, source):
        table.rows.append((label, w.count, w.raw if w.count else None, w.busy, gpu, ratio(gpu, w.raw),
                           ratio(gpu, w.busy), source, census(w.callers) if label != 'TOTAL' else '—'))

    for label in order:
        mode, key = TYPE_RULES.get(label, ('none', '无独立 GPU 对应') if label == 'custom_ascendc' else ('engine', ''))
        w = run.types.get(label, Work())
        if mode == 'family':
            gpu, source = None, 'GPU 仅列于 GEMM 家族'
        elif mode == 'caller':
            gpu, source = run.callers('GPU', [key])[1], '调用方：'+key
        elif mode in ('route', 'component'):
            gpu = run.component([key])
            source = ('路线不同：' if mode == 'route' else '组件：') + COMPONENTS[key]
        elif mode == 'none':
            gpu, source = None, key
        else:
            gpu, source = run.alloc.get(label), '引擎类型归属（可含显存）'
        append(label, w, gpu if w.count else None, source)
    if run.gemm.count or set(order).intersection(('MatMulV3', 'BatchMatMulV3')):
        append('GEMM 家族', run.gemm, run.callers('GPU', GEMM_GPU)[1], 'aclnn 调用方 ↔ mm+addmm+bmm；非加总行')
    append('TOTAL', run.total, run.gpu, 'summary：仅 GPU compute kernel；GPU 混源列不可相加')
    return table


def caller_table(run):
    table = Table('§3 调用方', ('调用方', 'NPU kernels', 'GPU kernels', 'NPU (ms)', 'GPU (ms)', 'GPU/NPU', '口径'))
    used_n, used_g = set(), set()

    def append(label, ns, gs, note):
        nc, n = run.callers('NPU', ns)
        gc, g = run.callers('GPU', gs)
        if nc or gc:
            table.rows.append((label, nc, gc, n, g, None if note == '—' else ratio(g, n), note))

    for label, ns, gs, note in CALLER_ROWS:
        append(label, ns, gs, note)
        used_n.update(ns)
        used_g.update(gs)
    for op in sorted(run.aten):
        append(op, () if op in used_n else (op,), () if op in used_g else (op,), '同名调用方')
    return table


def chain_table(run):
    table = Table('§3 排序与 jagged 链', ('链段', 'NPU 类型', 'GPU 族', 'NPU (ms)', 'GPU (ms)', 'GPU/NPU', '口径'))
    for label, types, families, note in CHAIN_GROUPS:
        vals = [run.types[t].raw for t in types if t in run.types]
        n, g = sum(vals) if vals else None, run.component(families)
        table.rows.append((label, '+'.join(types), '+'.join(families), n, g,
                           None if '无独立对应' in note else ratio(g, n), note))
    return table


def load_pair(step, whole):
    selected, full = load(step, False), load(whole, True)
    if selected.identities != full.identities:
        raise ValueError('Step 与 all 的原始输入指纹不同，不能组成同一配对')
    for side in ('npu', 'gpu'):
        a, b = selected.summary[side], full.summary[side]
        if a['selected_pid'] != b['selected_pid']:
            raise ValueError(f'{side}: Step 与 all 的 PID 不同，请确认属于同一配对')
    return selected, full


def baselines(current, baseline, args):
    comparison = Table('§4 跨口径同构对比', ('范围', '配对', 'NPU raw (ms)', 'NPU 去扰 (ms)', 'GPU kernel (ms)', 'GPU/NPU', 'GPU/NPU 去扰'))
    regression = Table('§5 NPU 两版类型变化', ('范围', '类型', '旧 NPU (ms)', '现 NPU (ms)', '现−旧 (ms)', '现/旧'))
    for scope, old, new in zip(('Step', 'all'), baseline, current):
        for label, run in ((args.baseline_label, old), ('当前', new)):
            comparison.rows.append((scope, label, run.total.raw, run.total.busy, run.gpu,
                                    ratio(run.gpu, run.total.raw), ratio(run.gpu, run.total.busy)))
        for kind in REG_ROWS:
            if kind in old.types or kind in new.types:
                a, b = (run.types[kind].raw if kind in run.types else None for run in (old, new))
                regression.rows.append((scope, kind, a, b, b-a if None not in (a, b) else None, ratio(b, a)))
    return comparison, regression


def build_report(args):
    step, full = load_pair(args.result_step10, args.result_all)
    base = load_pair(args.baseline_step10, args.baseline_all) if args.baseline_all else ()
    notes = Table('口径与证据', ('项目', '值'))
    notes.rows.extend([('设备', f'NPU：{args.npu_label}；GPU：{args.gpu_label}'),
        ('采集口径', args.caliber_note or '批次、精度、并行策略及 rank 是否等价未确认。'),
        ('读表', '跨批全 review 是严格门槛正常工作；这里只比较各侧观察到的累计设备耗时，不声称逐调用等价或端到端加速。'),
        ('单位与颜色', 'ms；比值 GPU/NPU，>1 绿色、<1 橙色、=1 中性；路线不同不着色。'),
        ('去扰', '逐 kernel max(aiv_time, aicore_time)，双零用 Duration；缺计数则该组不估。raw−去扰不能直接归因为排队。'),
        ('数据来源', '类型：mapping + details；引擎归属：npu_gpu_op_statistic；调用方：aten_op_statistic；组件：GPU kernel 全名。'),
        ('家族与合计', 'GEMM 按 aclnn 调用方聚合，可能含辅助类型；GPU 为 mm/addmm/bmm。类型、调用方及链段都是替代视图，不得相加。')])
    for tag, run in zip(('Step', 'all', '基线 Step', '基线 all'), (step, full, *base)):
        s, a = run.summary, run.summary['gpu_accounting']
        notes.rows.extend([(tag+' result', str(run.root)),
            (tag+' 引擎与校验', f"{s['producer']['version']}；{s['status']}；{s['validation']['status']}；run={s.get('run_id', '未知')}"),
            *((tag+' '+side, f"{s[side]['source']}；PID={s[side]['selected_pid']}；Step={s[side]['selected_step']}") for side in ('npu','gpu')),
            (tag+' 覆盖', f"NPU {run.total.count} kernels（{run.total.missing} 缺 busy）；GPU {a['kernel']['total_count']} kernels；显存 {a['memory']['total_count']} 项单列。"),
            (tag+' dtype 门禁', f'whole_call_mapping.csv {run.whole} 条：冲突 {sum(run.conflicts.values())}：{census(run.conflicts)}；缺失 {sum(run.missing.values())}：{census(run.missing)}。不代表其余调用已验证。')])
    order = sorted(step.types.keys() | full.types.keys(), key=lambda k: (-step.types.get(k, Work()).raw, k))
    callers = caller_table(full)
    tables = [notes, type_table('§1 Step 类型', step, order), type_table('§2 整体类型', full, order), callers, chain_table(full)]
    mem = Table('§3 GPU 显存（不入比值）', ('范围', 'GPU 显存活动', 'GPU (ms)'))
    for tag, run in [('Step', step), ('all', full)]:
        mem.rows.extend((tag, name, t) for name, t in sorted(run.memory.items()))
        mem.rows.append((tag, 'TOTAL', ms(run.summary['gpu_accounting']['memory']['total_time_us'])))
    tables.append(mem)
    if base:
        tables.extend(baselines((step, full), base, args))
    findings = Table('§6 结论', ('范围', '观察（不可解释为等价负载加速比）'))
    for tag, run in [('Step', step), ('all', full)]:
        n, b, g = run.total.raw, run.total.busy, run.gpu
        findings.rows.append((tag, f'NPU raw {fmt(n)} ms / 去扰 {fmt(b)} ms；GPU kernel {fmt(g)} ms；GPU/NPU {fmt(ratio(g,n),3)} → {fmt(ratio(g,b),3)}。'))
        share = (n-b)/n if b is not None and n and b <= n else None
        findings.rows.append((tag, 'raw 与 busy 的差额占比（排队线索，非排队测量）：'+(f'{share:.2%}' if share is not None else '—')))
    eligible = [(r[4]-r[3], r[0]) for r in callers.rows if r[5] is not None and r[-1] != '路线不同']
    for sign, desc in [(-1, 'NPU 累计耗时高于 GPU 最多'), (1, 'NPU 累计耗时低于 GPU 最多')]:
        differences = [(d*sign, k) for d,k in eligible if d*sign > 0]
        if differences:
            delta, label = max(differences)
            findings.rows.append(('all · 调用方', f'{desc}：{label}，差 {delta:.2f} ms；先核对调用数及工作量。'))
    copies = '、'.join(f'{k} {full.types[k].raw:.2f} ms' for k in COPY_TYPES if k in full.types) or '未观察到配置的拷贝类型'
    dtod = [t for k,t in full.memory.items() if re.search(r'dtod|d2d|device[ _-]*to[ _-]*device', k, re.I)]
    findings.rows.append(('all · FSDP / 拷贝', f'NPU：{copies}；GPU DtoD：{fmt(sum(dtod) if dtod else None)} ms。未观察到独立对应不等于零成本；不能直接归到 FSDP。'))
    if base:
        findings.rows.append(('跨版本', f'整体 GPU/NPU：{fmt(ratio(base[1].gpu,base[1].total.raw),3)} → {fmt(ratio(full.gpu,full.total.raw),3)}。两侧口径同时变化时不能归因为 NPU 单侧回归。'))
    return tables + [findings]


def fmt(x, digits=2):
    return '—' if x is None else f'{x:.{digits}f}' if isinstance(x, float) else str(x)


def markdown(title, tables):
    def escape(s):
        return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', r'\|').replace('\r', ' ').replace('\n', ' ')
    lines = ['# '+escape(title)]
    for t in tables:
        lines.extend(['', '## '+t.name, '', '| '+' | '.join(t.columns)+' |', '| '+' | '.join('---' for _ in t.columns)+' |'])
        for row in t.rows:
            lines.append('| '+' | '.join(escape(fmt(v, 3 if 'GPU/NPU' in col or col == '现/旧' else 2)) for col,v in zip(t.columns,row))+' |')
    return '\n'.join(lines)+'\n'


def check_reference(text, path, tables):
    """Compare ordered numeric rows in the same Markdown format, ignoring prose and path labels."""
    quantitative = {t.name: t.columns for t in tables if t.name.startswith('§') and t.name != '§6 结论'}
    def mechanical_rows(document):
        name = None
        for line in document.splitlines():
            if line.startswith('## '):
                name = line[3:]
                if name.startswith(('§1 ', '§2 ', '§3 ', '§4 ', '§5 ')):
                    yield name  # an extra/missing section is also a difference
            elif name in quantitative and line.startswith('| '):
                cells = line[2:-2].split(' | ')
                columns = quantitative[name]
                if tuple(cells) == columns or all(c == '---' for c in cells):
                    continue
                if len(cells) != len(columns):
                    raise ValueError(f'参考表 {name} 列数不匹配')
                yield tuple(v for k,v in zip(columns,cells) if k not in ('对照来源', '口径'))
    known = path.read_text(encoding='utf-8-sig')
    for i,(a,b) in enumerate(zip_longest(mechanical_rows(known), mechanical_rows(text)),1):
        if a != b:
            raise ValueError(f'已审核参考表的第 {i} 项不一致：{a!r} vs {b!r}；未写出结果')


def workbook(title, tables):
    from io import BytesIO
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    wb = Workbook()
    wb.remove(wb.active)
    wb.properties.title = title
    for t in tables:
        ws = wb.create_sheet(t.name)
        ws.freeze_panes = 'B2'
        for i, row in enumerate([t.columns, *t.rows], 1):
            for j, (col, v) in enumerate(zip(t.columns, row), 1):
                cell = ws.cell(i,j, v)
                if isinstance(v, str):
                    cell.data_type = 's'  # Trace names are text, even when they start with '='.
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
                if i == 1:
                    cell.font = Font(bold=True, color='FFFFFF')
                    cell.fill = PatternFill('solid', fgColor='264563')
                elif isinstance(v, float):
                    is_ratio = col.startswith('GPU/NPU')
                    cell.number_format = '0.000' if is_ratio or col == '现/旧' else '0.00'
                    if is_ratio and v != 1 and not any('路线不同' in str(x) for x in row):
                        cell.fill = PatternFill('solid', fgColor='D9F2DC' if v > 1 else 'FFE4C4')
        for j, col in enumerate(t.columns, 1):
            ws.column_dimensions[get_column_letter(j)].width = min(85, max(16, 2*len(col), max((len(str(r[j-1])) for r in t.rows), default=0)+2))
        ws.auto_filter.ref = ws.dimensions
    data = BytesIO()
    wb.save(data)
    return data.getvalue()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('result-step10','result-all','output-md','output-xlsx'):
        p.add_argument('--'+name, required=True, type=Path)
    for name in ('baseline-step10','baseline-all'):
        p.add_argument('--'+name, type=Path)
    p.add_argument('--reference-md', type=Path, help='复验已知配对时，对同格式已审核 Markdown 的机械行做精确比对')
    for name, default in [('title','NPU / GPU 类型与调用方对照'), ('npu-label','NPU'), ('gpu-label','GPU'),
                          ('baseline-label','上一配对'), ('caliber-note','')]:
        p.add_argument('--'+name, default=default)
    args = p.parse_args(argv)
    if bool(args.baseline_all) != bool(args.baseline_step10):
        p.error('基线须同时提供 --baseline-step10 与 --baseline-all')
    inputs = [x.resolve() for x in (args.result_step10, args.result_all, args.baseline_step10, args.baseline_all) if x]
    md, xlsx = [x.resolve() for x in (args.output_md, args.output_xlsx)]
    if md == xlsx or any(path.exists() or any(path == r or r in path.parents for r in inputs) for path in (md,xlsx)):
        p.error('输出必须是不同的新文件，且不在任何输入 result 目录内')
    try:
        tables = build_report(args)
        text = markdown(args.title, tables)
        if args.reference_md:
            check_reference(text, args.reference_md, tables)
        try:
            data = workbook(args.title, tables)
        except ModuleNotFoundError as exc:
            if exc.name != 'openpyxl':
                raise
            print('缺少 openpyxl：仅生成 Markdown，跳过 XLSX。', file=sys.stderr)
            data = None
        md.parent.mkdir(parents=True, exist_ok=True)
        with md.open('x', encoding='utf-8', newline='\n') as out:
            out.write(text)
        print(md)
        if data is not None:
            xlsx.parent.mkdir(parents=True, exist_ok=True)
            with xlsx.open('xb') as out:
                out.write(data)
            print(xlsx)
    except (OSError, KeyError, TypeError, ValueError, csv.Error) as exc:
        p.exit(2, f'op_type_comparison: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

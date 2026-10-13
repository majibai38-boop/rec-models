"""Presentation-only ordinals within the available manual mapping catalog."""
from collections import defaultdict
import json


def with_call_positions(records, *, gpu=False):
    """Number complete families before UI search/pagination, using batch scope keys."""
    families, templates = defaultdict(list), defaultdict(list)
    result = [{**r, 'call_position': None, 'template_position': None} for r in records]
    for r in result:
        context = r.get('batch_context') or {}
        if r.get('cpu_index') is None or context.get('timestamp_us') is None:
            continue
        family = json.dumps([r['cpu_name'], r['module'], context.get('path'), context.get('pid'),
                             context.get('tid'), r['count'], r['names'] if gpu else
                             [r['op_type'], r['device_id'], r['core_type']],
                             r.get('device_key') if gpu else None], sort_keys=True)
        template = json.dumps([r['shapes'], r['dtypes']], sort_keys=True)
        families[family].append(r)
        templates[family, template].append(r)
    for groups, field in ((families, 'call_position'), (templates, 'template_position')):
        for rows in groups.values():
            rows.sort(key=lambda r: (r['batch_context']['timestamp_us'], r['cpu_index']))
            for index, r in enumerate(rows, 1):
                r[field] = {'index': index, 'total': len(rows)}
    return result


def comparison_positions(targets, candidate):
    from profiling_tools.manual_mapping import _comparison_target
    return {r['id']: r for r in with_call_positions([_comparison_target(t, candidate) for t in targets])}

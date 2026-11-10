"""Private, disposable Web cache of immutable automatic analysis inputs.

JSON only: never deserialize executable Python objects from task storage. Full
trace events are intentionally omitted; scoped downloads must also be intact
before a cache hit can be used. Mapping-dependent reports are always rebuilt.
"""
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import shutil

from .model import ParsedProfile, TraceEvent, OpRecord, KernelRecord, KernelDetail, OpMatch
from .provenance import write_json
from .result_api import file_identity

EXPORT_NAMES = {'npu_kernel_details.csv', 'npu_op_statistic.csv',
                'gpu_kerenel_details.csv', 'gpu_op_statistic.csv',
                'gpu_trace.json', 'npu_trace.json'}


def cache_key(manifest):
    parameters = {k: v for k, v in manifest['parameters'].items()
                  if k not in {'output_dir', 'gpu_trace', 'npu_trace', 'npu_kernel_details', 'op_map'}}
    value = {'version': 1, 'producer': manifest['producer'], 'parameters': parameters,
             'inputs': {k: v['sha256'] for k, v in manifest['inputs'].items()}}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _encode(value):
    if is_dataclass(value):
        return {f.name: getattr(value, f.name) for f in fields(value)
                if not (isinstance(value, ParsedProfile) and f.name == 'trace_events')}
    raise TypeError(type(value).__name__)


def _profile(value):
    def kernel(row):
        return KernelRecord(**{**row, 'event': TraceEvent(**row['event']),
                               'detail': KernelDetail(**row['detail']) if row['detail'] else None})
    return ParsedProfile(**{**value,
        'ops': [OpRecord(**{**row, 'event': TraceEvent(**row['event'])}) for row in value['ops']],
        'kernels': [kernel(row) for row in value['kernels']],
        'memory_activities': [kernel(row) for row in value['memory_activities']]})


def load_cache(directory, manifest):
    if directory is None:
        return None, 'disabled'
    root = Path(directory)
    try:
        meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
        if meta['key'] != cache_key(manifest):
            return None, 'inputs_scope_or_engine_changed'
        if set(meta['exports']) != EXPORT_NAMES or set(meta['downloads']['reports']) != EXPORT_NAMES - {'gpu_trace.json', 'npu_trace.json'}:
            return None, 'invalid_export_manifest'
        if file_identity(root/'state.json')['sha256'] != meta['state_sha256']:
            return None, 'cache_checksum_mismatch'
        for name, identity in meta['exports'].items():
            path = Path(meta['reports_dir'])/name
            current = file_identity(path)
            if any(current[k] != identity[k] for k in ('sha256', 'size_bytes')):
                return None, 'export_checksum_mismatch'
        value = json.loads((root/'state.json').read_text(encoding='utf-8'))
        for side in ('gpu', 'npu'):
            value[side] = _profile(value[side])
            value[side].source_path = manifest['parameters'][side+'_trace']
        value['matches'] = [OpMatch(**row) for row in value['matches']]
        value['exports'] = meta
        return value, 'hit'
    except (OSError, ValueError, KeyError, TypeError):
        return None, 'missing_or_invalid_cache'


def save_cache(directory, manifest, state, output, downloads):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root/'state.tmp'
    try:
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(state, stream, default=_encode, ensure_ascii=False, separators=(',', ':'))
        temporary.replace(root/'state.json')
        write_json(root/'metadata.json', {
            'key': cache_key(manifest), 'state_sha256': file_identity(root/'state.json')['sha256'],
            'reports_dir': str(Path(output).resolve()), 'downloads': downloads,
            'exports': {name: file_identity(Path(output)/name)
                        for name in [*downloads['reports'], 'gpu_trace.json', 'npu_trace.json']},
        })
    finally:
        temporary.unlink(missing_ok=True)


def copy_exports(cached, output):
    meta = cached['exports']
    for name in meta['exports']:
        # Independent files preserve historical attempts if an artifact is edited.
        shutil.copyfile(Path(meta['reports_dir'])/name, Path(output)/name)
        if file_identity(Path(output)/name)['sha256'] != meta['exports'][name]['sha256']:
            raise ValueError('cached export changed while copying')
    return meta['downloads']

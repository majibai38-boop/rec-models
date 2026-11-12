"""Step Trace attachments and server-owned, version-bound CLI sessions."""
import hashlib
import json
import os
import re
from pathlib import Path

from .ai_backends import BackendError


def trace_limit():
    return max(1024, min(64 * 1024 * 1024, int(os.environ.get('PROFILING_AI_TRACE_MAX_BYTES', 2 * 1024 * 1024))))


def trace_files(root, job):
    paths = [root/'reports'/f'{side}_trace.json' for side in ('gpu', 'npu')]
    for path in paths:
        if path.name not in job['summary'].get('reports', []) or not path.is_file() or path.is_symlink():
            raise BackendError('结果缺少所选 Step 的完整 Trace，请复用输入重新分析')
    return paths


def trace_info(root, job, *, hashes=False):
    files = []
    for path in trace_files(root, job):
        row = {'name': path.name, 'bytes': path.stat().st_size}
        if hashes:
            with path.open('rb') as stream:
                row['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        files.append(row)
    total = sum(f['bytes'] for f in files)
    return {'scope': 'selected_steps', 'files': files, 'total_bytes': total,
            'max_bytes': trace_limit(), 'within_limit': total <= trace_limit(),
            'gpu_step': job['options'].get('gpu_step'), 'npu_step': job['options'].get('npu_step'),
            'all_steps': job['options'].get('all_steps', False)}


def session_path(job_path, key):
    if not re.fullmatch(r'[0-9a-f]{32}', key or ''):
        raise BackendError('Trace 会话记录无效，请重新附加 Trace')
    return job_path/'ai-sessions'/key


def read_session(job_path, key):
    path = session_path(job_path, key)/'state.json'
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        raise BackendError('Trace 会话已失效，请重新附加 Trace') from None


def write_session(path, state):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    pending = path/'state.tmp'
    pending.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
    pending.replace(path/'state.json')


def reusable_session(job_path, parent, provider, model, attempt):
    trace = (parent.get('evidence') or {}).get('trace', {})
    key = trace.get('session_key')
    state = read_session(job_path, key)
    if (not state.get('ready') or state.get('last_run') != parent['id'] or
            state.get('provider') != provider or state.get('model') != model or state.get('attempt') != attempt):
        raise BackendError('只能复用同一后端、模型和结果版本的最新成功会话；请选择最新记录或重新附加 Trace')
    if not (session_path(job_path, key)/'home').is_dir():
        raise BackendError('Trace 后端会话文件已清理，请重新附加 Trace')
    return key, state

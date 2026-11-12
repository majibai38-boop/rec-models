"""Server-owned CLI adapters. No browser-supplied commands, paths or credentials."""
from dataclasses import dataclass
from contextlib import ExitStack
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import re

from .ai_evidence import Analysis, prompt_for
from .worker import kill_process


class BackendError(ValueError):
    pass


@dataclass
class AIConfig:
    enabled: bool
    commands: dict
    models: dict
    timeout: int = 180

    @classmethod
    def from_env(cls):
        commands = {}
        for key, default in (('codex', 'codex'), ('deepseek', 'dsh')):
            value = os.environ.get(f'PROFILING_AI_{key.upper()}_COMMAND', json.dumps([default]))
            command = json.loads(value)
            if not isinstance(command, list) or not command or any(not isinstance(v, str) or not v for v in command):
                raise ValueError('AI COMMAND 必须是非空字符串组成的 JSON 数组')
            commands[key] = command
        return cls(os.environ.get('PROFILING_AI_ENABLED') == '1', commands,
                   {k: os.environ.get(f'PROFILING_AI_{k.upper()}_MODEL', '') for k in commands},
                   max(10, min(900, int(os.environ.get('PROFILING_AI_TIMEOUT', '180')))))

    def executable(self, provider):
        command = self.commands[provider]
        executable = shutil.which(command[0])
        if not executable or Path(executable).suffix.lower() in {'.cmd', '.bat', '.ps1'}:
            return None
        return [executable, *command[1:]]

    def auth_file(self):
        home = os.environ.get('PROFILING_AI_CODEX_AUTH_HOME')
        return Path(home).expanduser()/'auth.json' if home else None

    def availability(self):
        result = []
        for key, label in (('deepseek', 'DeepSeek Harness'), ('codex', 'Codex CLI')):
            reason = ''
            if not self.enabled:
                reason = '管理员尚未设置 PROFILING_AI_ENABLED=1'
            elif not self.executable(key):
                reason = '未找到可执行文件；请配置 COMMAND（Windows 使用 exe 或 node + CLI 脚本）'
            elif key == 'deepseek' and not os.environ.get('DEEPSEEK_API_KEY'):
                reason = '服务进程未配置 DEEPSEEK_API_KEY'
            elif key == 'codex' and not (os.environ.get('CODEX_API_KEY') or (self.auth_file() and self.auth_file().is_file())):
                reason = '请配置 CODEX_API_KEY 或专用的 PROFILING_AI_CODEX_AUTH_HOME 登录目录'
            result.append({'id': key, 'label': label, 'available': not reason, 'reason': reason,
                           'model': self.models[key] or 'CLI 默认模型'})
        return {'enabled': self.enabled, 'providers': result, 'timeout_seconds': self.timeout}


def child_env():
    # Do not pass Web access tokens, unrelated cloud credentials or agent settings.
    names = {'PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'COMSPEC', 'TEMP', 'TMP', 'HOME', 'USERPROFILE',
             'APPDATA', 'LOCALAPPDATA', 'LANG', 'LANGUAGE', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
             'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'ALL_PROXY', 'NODE_EXTRA_CA_CERTS'}
    return {k: v for k, v in os.environ.items() if k.upper() in names}


def deepseek_patch(model, persistent=False):
    # Official headless profile overlay. Guard installation is part of the required
    # headless runner's configuration: unsupported guard APIs fail before inference.
    disabled = ['tool-bash', 'tool-pwsh', 'tool-fs', 'tool-fs-search', 'tool-working-directory',
                'tool-web', 'tool-subagent', 'tool-subagent-fork', 'tool-subagent-control',
                'tool-agent-team', 'tool-workflow', 'tool-skill', 'tool-goal', 'tool-todo',
                'tool-jobs', 'agent-instructions', 'skill-filesystem', 'session-title-llm',
                'session-log-deepseek', 'session-telemetry-otel', 'plugin-package-inventory-deepseek']
    text = ''.join(f'- id: {row}\n  disabled: true\n' for row in disabled)
    text += '''- id: headless-runner
  inject: [headlessStartup, tools]
  config:
    task: !!js "(() => { ctx.tools.restrict({allow: []}); ctx.tools.guard(() => 'Profiling analysis has no tool permissions'); return ctx.headlessStartup.task; })()"
    json: false
'''
    if persistent:
        text = text.replace('    json: false', '    json: true\n    sessionId: !!js ctx.headlessStartup.sessionId')
    if model:
        text += '- id: agent-default-model\n  config:\n    provider: deepseek-official\n    model: ' + json.dumps(model) + '\n'
    return text


def prepare_command(config, provider, work, session=None):
    command = config.executable(provider)
    if not command:
        raise BackendError('AI 程序不可用，请检查服务端 COMMAND 配置')
    env = child_env()
    home = (session['path'] if session else work)/'home'
    home.mkdir(mode=0o700, exist_ok=True)
    if provider == 'deepseek':
        env.update(DSH_HOME=str(home), DEEPSEEK_API_KEY=os.environ.get('DEEPSEEK_API_KEY', ''),
                   DSH_TELEMETRY_DISABLED='1', DSH_TELEMETRY_MODE='DISABLED')
        patch = work/'analysis.patch.yml'
        patch.write_text(deepseek_patch(config.models[provider], bool(session)), encoding='utf-8')
        extra = ['--json'] if session else []
        if session and session.get('backend_id'):
            extra += ['--session-id', session['backend_id']]
        return [*command, '--profile', 'headless', '--patch', str(patch),
                '--permission-mode', 'read-only', '--approval-policy', 'never', *extra, '-'], env, None
    env['CODEX_HOME'] = str(home)
    if os.environ.get('CODEX_API_KEY'):
        env['CODEX_API_KEY'] = os.environ['CODEX_API_KEY']
    elif config.auth_file() and config.auth_file().is_file():
        shutil.copyfile(config.auth_file(), home/'auth.json')
        (home/'auth.json').chmod(0o600)
    schema, output = work/'schema.json', work/'answer.json'
    schema.write_text(json.dumps(Analysis.model_json_schema(), ensure_ascii=False), encoding='utf-8')
    # The private home has no plugins, MCP, skills or project configuration.
    options = ['approval_policy="never"', 'web_search="disabled"', 'features.shell_tool=false',
               'features.apps=false', 'features.hooks=false', 'features.multi_agent=false',
               'features.js_repl=false', 'features.apply_patch_freeform=false']
    args = [*command, 'exec', '--sandbox', 'read-only', '--color', 'never']
    if session and session.get('backend_id'):
        args += ['resume', session['backend_id']]
    args += ['--json'] if session else ['--ephemeral']
    args += ['--skip-git-repo-check',
            '--ignore-user-config', '--ignore-rules', '--output-schema', str(schema), '-o', str(output)]
    for option in options:
        args += ['-c', option]
    if config.models[provider]:
        args += ['--model', config.models[provider]]
    return [*args, '-'], env, output


def session_result(provider, stdout, answer, expected_id=None):
    """Accept only a completed run with an exact resumable session identity."""
    try:
        events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
        if provider == 'codex':
            ident = next(e['thread_id'] for e in events if e.get('type') == 'thread.started')
            complete = any(e.get('type') == 'turn.completed' for e in events)
            failed = any(e.get('type') in {'error', 'turn.failed'} for e in events)
            text = answer
        else:
            final = next(e for e in reversed(events) if e.get('type') == 'final')
            ident = final['sessionId']
            complete = (final.get('reason') or {}).get('kind') == 'completed'
            failed = any(e.get('type') == 'error' for e in events)
            text = final['text']
        if (not complete or failed or not isinstance(text, str) or
                not re.fullmatch(r'[a-zA-Z0-9_-]{1,128}', ident) or (expected_id and ident != expected_id)):
            raise ValueError()
        return {'text': text, 'backend_id': ident}
    except (ValueError, KeyError, StopIteration, TypeError, AttributeError):
        raise BackendError('AI 未返回可验证的成功会话，请检查 CLI 版本；不会自动重发 Trace') from None


def run_backend(config, provider, question, evidence, scratch, cancelled, session=None, attachments=()):
    """Isolated CLI; persistent homes are used only for explicitly shared Traces."""
    prompt = prompt_for(question, evidence).encode('utf-8')
    with tempfile.TemporaryDirectory(prefix='ai-', dir=scratch) as directory, ExitStack() as cleanup:
        work = Path(directory)
        if session:
            cleanup.callback((session['path']/'home'/'auth.json').unlink, missing_ok=True)
        command, env, answer = prepare_command(config, provider, work, session)
        with (work/'stdin').open('wb') as stream:
            stream.write(prompt)
            # Copy bytes exactly; never silently truncate or parse/re-encode Trace events.
            for path, meta in attachments:
                stream.write(('\nTRACE_ATTACHMENT_JSON ' + meta['name'] + ':\n').encode())
                digest, count = hashlib.sha256(), 0
                with path.open('rb') as source:
                    while chunk := source.read(65536):
                        if cancelled.is_set():
                            raise BackendError('AI 分析已取消')
                        count += len(chunk)
                        if count > meta['bytes']:
                            raise BackendError('Trace 文件已变化，请刷新后重试')
                        digest.update(chunk)
                        stream.write(chunk)
                if count != meta['bytes'] or digest.hexdigest() != meta['sha256']:
                    raise BackendError('Trace 文件已变化，请刷新后重试')
        with (work/'stdin').open('rb') as stdin, (work/'stdout').open('w+b') as stdout, (work/'stderr').open('w+b') as stderr:
            process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr, cwd=session['path'] if session else work, env=env,
                                       shell=False, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            started = time.monotonic()
            try:
                while True:
                    if cancelled.is_set():
                        raise BackendError('AI 分析已取消')
                    if time.monotonic() - started > config.timeout:
                        raise BackendError('AI 分析超时，请缩小问题范围或调整服务端超时')
                    if max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size,
                           answer.stat().st_size if answer and answer.exists() else 0) > 2*1024*1024:
                        raise BackendError('AI 输出超过 2 MB 上限')
                    if process.poll() is not None:
                        break
                    cancelled.wait(.1)
                if process.returncode:
                    # Only classify, never echo provider logs that may include secrets.
                    stderr.seek(0)
                    diagnostic = stderr.read(2*1024*1024).decode('utf-8', errors='replace').lower()
                    hint = '请检查 CLI 版本、登录/密钥、模型配置及网络'
                    if any(s in diagnostic for s in ('401', 'unauthorized', 'authentication', 'api key')):
                        hint = '认证失败，请检查服务进程的 API Key 或专用 Codex 登录状态'
                    elif any(s in diagnostic for s in ('429', 'quota', 'rate limit')):
                        hint = '模型额度或速率受限，请稍后重试或检查账户额度'
                    elif any(s in diagnostic for s in ('context length', 'context window', 'too many tokens', 'request too large', '413')):
                        hint = '输入超过模型上下文或请求容量，请缩小 Step 范围或仅发送统计证据；不会自动重发 Trace'
                    raise BackendError(f'AI 程序退出（{process.returncode}）；{hint}')
                if answer:
                    if not answer.is_file():
                        raise BackendError('Codex 未生成结构化结果，请检查 CLI 版本')
                    result = answer.read_text(encoding='utf-8')
                    if not session:
                        return result
                stdout.seek(0)
                output = stdout.read().decode('utf-8')
                return session_result(provider, output, result if answer else None, session.get('backend_id')) if session else output
            finally:
                if process.poll() is None:
                    kill_process(process)
                process.wait(timeout=10)

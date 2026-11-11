"""Server-owned CLI adapters. No browser-supplied commands, paths or credentials."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

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


def deepseek_patch(model):
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
    if model:
        text += '- id: agent-default-model\n  config:\n    provider: deepseek-official\n    model: ' + json.dumps(model) + '\n'
    return text


def prepare_command(config, provider, work):
    command = config.executable(provider)
    if not command:
        raise BackendError('AI 程序不可用，请检查服务端 COMMAND 配置')
    env = child_env()
    home = work/'home'
    home.mkdir(mode=0o700)
    if provider == 'deepseek':
        env.update(DSH_HOME=str(home), DEEPSEEK_API_KEY=os.environ.get('DEEPSEEK_API_KEY', ''),
                   DSH_TELEMETRY_DISABLED='1', DSH_TELEMETRY_MODE='DISABLED')
        patch = work/'analysis.patch.yml'
        patch.write_text(deepseek_patch(config.models[provider]), encoding='utf-8')
        return [*command, '--profile', 'headless', '--patch', str(patch),
                '--permission-mode', 'read-only', '--approval-policy', 'never', '-'], env, None
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
    args = [*command, 'exec', '--sandbox', 'read-only', '--skip-git-repo-check', '--ephemeral',
            '--ignore-user-config', '--ignore-rules', '--color', 'never', '--output-schema', str(schema), '-o', str(output)]
    for option in options:
        args += ['-c', option]
    if config.models[provider]:
        args += ['--model', config.models[provider]]
    return [*args, '-'], env, output


def run_backend(config, provider, question, evidence, scratch, cancelled):
    """A bounded isolated process. Raw stderr/reasoning is never published or kept."""
    prompt = prompt_for(question, evidence).encode('utf-8')
    with tempfile.TemporaryDirectory(prefix='ai-', dir=scratch) as directory:
        work = Path(directory)
        command, env, answer = prepare_command(config, provider, work)
        with (work/'stdin').open('wb') as stream:
            stream.write(prompt)
        with (work/'stdin').open('rb') as stdin, (work/'stdout').open('w+b') as stdout, (work/'stderr').open('w+b') as stderr:
            process = subprocess.Popen(command, stdin=stdin, stdout=stdout, stderr=stderr, cwd=work, env=env,
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
                    raise BackendError(f'AI 程序退出（{process.returncode}）；{hint}')
                if answer:
                    if not answer.is_file():
                        raise BackendError('Codex 未生成结构化结果，请检查 CLI 版本')
                    return answer.read_text(encoding='utf-8')
                stdout.seek(0)
                return stdout.read().decode('utf-8')
            finally:
                if process.poll() is None:
                    kill_process(process)
                process.wait(timeout=10)

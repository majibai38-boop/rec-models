# AI 辅助分析：DeepSeek Harness 与 Codex

这是 Profiling Web 的可选功能，默认关闭。原有解析、覆盖率、Excel 和人工关联不依赖模型服务。Workload Modeler 暂未接入。

## 网页使用

1. 完成 GPU/NPU 分析后，点击结果顶部「AI 辅助分析」，或类型行中的「AI 分析」。算子详情页也可以分析当前类型。
2. 选择 **DeepSeek Harness** 或 **Codex CLI**，输入问题，例如“检查 IndexAdd 覆盖率低是否与两侧 CPU 嵌套调用不同有关”。
3. 确认允许发送本次问题及统计、名称、模块、shape/dtype、调用样本后开始。不会发送完整 Trace；推理仍可能经互联网访问所选服务，并产生模型费用。
4. 查看结论、局限和引用的 `E1` 等证据。点击证据编号可检查模型实际收到的字段。支持分析历史、取消、下载分析 JSON 和删除记录。
5. 有关联建议时，点击「核对并预览关联」，进入原有人工关联窗口，已选中推荐配对并带入理由。核对后可单组确认，也可预览同模板/同模块批量范围，再一次确认。在该类型的「人工补充关联」中可撤销。

AI 不直接改映射。人工确认后的来源仍是 `manual`，不会改为 Direct；历史另外保留 `ai_run_id`、`ai_proposal_id`、`ai_provider`。修改推荐配对后不再沿用该提案来源。结果版本更新后，旧建议只供阅读，不能应用。

## 服务端配置

CLI、认证和模型配置全部在 **部署 Web 的主机** 上完成，其他电脑通过 IP 访问，不需要安装 CLI。不在浏览器中填写或返回密钥。共用工作区的访问者都能使用已启用的后端，认证沿用 Web 的令牌/免令牌方式。

| 环境变量 | 用途 |
| --- | --- |
| `PROFILING_AI_ENABLED=1` | 开启 AI，默认关闭 |
| `PROFILING_AI_DEEPSEEK_COMMAND` | JSON 参数数组，默认 `["dsh"]` |
| `DEEPSEEK_API_KEY` | Harness 使用的 DeepSeek API Key |
| `PROFILING_AI_DEEPSEEK_MODEL` | 可选，留空使用 Harness 默认模型 |
| `PROFILING_AI_CODEX_COMMAND` | JSON 参数数组，默认 `["codex"]` |
| `CODEX_API_KEY` | Codex API Key；与下项任选其一，API Key 优先 |
| `PROFILING_AI_CODEX_AUTH_HOME` | 专用于 Web 的 Codex 登录目录，应包含 `auth.json` |
| `PROFILING_AI_CODEX_MODEL` | 可选，留空使用 Codex 默认模型 |
| `PROFILING_AI_TIMEOUT` | 每次超时秒数，默认 180，范围 10–900 |

COMMAND 是管理员提供的可执行文件和固定参数数组，不是 shell 命令。推荐绝对路径。Windows 不接受 `.cmd`、`.bat`、`.ps1`；使用原生 `.exe`，或 `["node.exe", "CLI脚本绝对路径"]`。不要在 COMMAND 中放密钥或绕过审批参数。

### DeepSeek Harness

接入 [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness) 官方 `headless` 单任务入口，经 stdin 输入问题、stdout 返回最终 JSON。适配依据为 **0.2.2-rc.1 源码接口**，包括 `--patch`、`--approval-policy`、`tools.restrict` 和 `tools.guard`。这是预览接口，安装后需验证版本兼容性。

安装方法以[官方 README](https://github.com/deepseek-ai/deepseek-harness#readme)为准。Node 发行包的入口为 `@deepseek-ai/dsh/lib/bin.js`，该源码版本要求 Node >= 24.9；也可使用官方 `deepseek-harness-runtime-bin` 本平台发行包，其原生 `dsh` 不依赖本机 Node。此可选依赖不加入 Web 的基础 requirements。

PowerShell 示例，路径替换为实际安装位置，密钥通过服务管理器或私有环境变量注入：

```powershell
$env:PROFILING_AI_ENABLED = '1'
$env:PROFILING_AI_DEEPSEEK_COMMAND = '["C:/ai/harness/Scripts/dsh.exe"]'
# 已设置 DEEPSEEK_API_KEY；可选设置 PROFILING_AI_DEEPSEEK_MODEL
python -m profiling_web --port 8000
```

每次生成独立 `DSH_HOME`，不沿用日常 Harness 配置和 GUI 登录。自动挂载只读分析 overlay，关闭文件、shell、网络检索、子代理等工具，并在 required headless runner 初始化时安装禁止工具执行的 guard。不兼容该 API 时停止，不降级到无限制模式。禁用额外会话遥测及模型生成标题，运行结束删除临时会话和原始输出。

### Codex

安装 [Codex CLI](https://learn.chatgpt.com/docs/cli/reference)，需支持 `exec`、`--output-schema`、`--ephemeral`、`--ignore-user-config` 和 `--ignore-rules`。使用 [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)，不依赖桌面正在打开的聊天。

```powershell
$env:PROFILING_AI_ENABLED = '1'
$env:PROFILING_AI_CODEX_COMMAND = '["C:/ai/codex/codex.exe"]'
# 已设置 CODEX_API_KEY；可选设置 PROFILING_AI_CODEX_MODEL
python -m profiling_web --port 8000
```

也可先在独立终端为 Web 创建专用登录目录：

```powershell
$taskCodexAuth = 'C:/ai/profiling-codex-auth'
$env:CODEX_HOME = $taskCodexAuth
codex -c cli_auth_credentials_store='"file"' login
# 关闭该终端，在启动 Web 的终端设置：
# $env:PROFILING_AI_CODEX_AUTH_HOME = 'C:/ai/profiling-codex-auth'
```

Web 仅复制该目录的 `auth.json` 到单次私有目录，不沿用个人配置、MCP 和插件。设置 read-only、never approval，关闭 shell、apps、hooks、web search 等工具。凭据过期时需在专用目录重新登录，临时运行中的凭据刷新不回写。

## 数据范围与限制

- 读取现有 SQLite 索引，不重解析 Trace。全任务按 NPU 耗时取前 6 类型；每类型最多 4 条诊断、3 个 shape 组、1 个 NPU 待关联调用及前 3 GPU 候选。字段数组最多 20 项，记录预算约 96 KB，截取/省略会标记。具体类型入口更适合深入排查。
- 这是已有证据分析及候选推荐，不是无限翻查 Trace 的交互代理。未采样调用不会被模型判断，没有建议不代表没有可关联项。无 CPU 场景仍需人工确认两侧整组语义，数量或耗时相近不是等价证明。
- 响应需符合 JSON Schema，引用与提案 ID 需属于本次证据/候选集合，否则不产生可应用建议。结构校验不能证明模型文字判断正确。
- 独立队列单并发，最多 8 个排队/运行分析；每任务最多 1 个活动分析、50 条历史。CLI 单份输出上限 2 MB，超时/取消终止进程树。没有硬性计费 token 上限，费用由模型服务计量。
- 问题、状态、证据、结果持久化在任务数据库，刷新可继续查看。原始 stderr/推理过程不公开或长期保留。服务重启标记旧活动分析为中断，需重新发起。删除任务需先结束 AI 分析，并一并删除 AI 历史。
- 服务数据目录应保持私有。断电/强制终止可能留下 `jobs/<id>/ai-scratch/` 临时文件，包括短暂凭据副本；确认没有 CLI 运行后可由管理员清理。建议使用独立系统用户部署。
- 回归测试使用合成数据和模拟模型，不自动使用桌面登录，也不发送真实用户 profiling 数据。真实服务的认证、额度及网络需在部署环境验证。

## API

| 操作 | 路径 |
| --- | --- |
| 后端就绪状态，无密钥 | `GET /api/ai/config` |
| 发起分析 | `POST /api/jobs/{id}/ai-runs` |
| 历史 / 单次结果 | `GET /api/jobs/{id}/ai-runs[/{run_id}]` |
| 取消 | `POST /api/jobs/{id}/ai-runs/{run_id}/cancel` |
| 删除历史 | `DELETE /api/jobs/{id}/ai-runs/{run_id}` |

发起请求包括 `attempt`、`provider: deepseek|codex`、`question`、`share_evidence:true`，以及可选 `scope:{op_type,device_id,core_type}`。认证/Origin 校验与其余 API 相同。关联仍使用 `manual-batch/preview`、`manual-batch` 或 `manual-links`；额外传 `ai_run_id`、`ai_proposal_id` 时校验来源和版本，不新增绕过原校验的应用接口。

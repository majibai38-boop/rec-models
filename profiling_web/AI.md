# AI 辅助分析：DeepSeek Harness 与 Codex

这是 Profiling Web 的可选功能，默认关闭。原有解析、覆盖率、Excel 和人工关联不依赖模型服务。Workload Modeler 暂未接入。

## 网页使用

AI 现有独立任务页签。从类型行或算子详情点击「AI 分析」会切换到此页签并携带类型范围；可一键切换全任务。提供整体性能、覆盖率、shape/dtype 和关联四种问题模板，以及按问题自动选择证据的模式。查询为服务端受控规则检索，不是模型任意执行 SQL/脚本。

成功回答后可以勾选「继续当前会话」发送追问，服务器校验任务和结果版本，继承最近 4 轮问题、摘要及部分发现；旧回答视为待核实上下文，不能直接复用旧证据编号。「新建会话」不带入历史。服务重启后仍可从分析历史继续，删除父记录会截断后续上下文追溯。

新增完整范围统计：总 NPU 累计耗时、不可比耗时，以及精确排除指定 NPU 类型后的累计耗时。此汇总读取全部所选类型，不受前 6 类型的明细采样限制。排除列表通过独立字段填写，不能仅凭自然语言要求让程序自动扣减；共享 GPU 组不会被按 NPU 类型简单相减。明细优先选择问题提及的类型，覆盖率模式按不可比耗时排序，shape 模式最多取每类型 8 个 shape 组。所有模式仍受证据大小预算限制。

1. 完成 GPU/NPU 分析后，点击结果顶部「AI 辅助分析」，或类型行中的「AI 分析」。算子详情页也可以分析当前类型。
2. 选择 **DeepSeek Harness** 或 **Codex CLI**，输入问题，例如“检查 IndexAdd 覆盖率低是否与两侧 CPU 嵌套调用不同有关”。
3. 默认仅发送问题及统计、名称、模块、shape/dtype、调用样本。可在「完整 Trace」中主动选择附加本次选中 Step 的完整 GPU/NPU JSON；确认发送范围后开始。推理可能经互联网访问所选服务，并产生模型费用。
4. 查看结论、局限和引用的 `E1` 等证据。点击证据编号可检查模型实际收到的字段。支持分析历史、取消、下载分析 JSON 和删除记录。
5. 有关联建议时，点击「核对并预览关联」，进入原有人工关联窗口，已选中推荐配对并带入理由。核对后可单组确认，也可预览同模板/同模块批量范围，再一次确认。在该类型的「人工补充关联」中可撤销。

AI 不直接改映射。人工确认后的来源仍是 `manual`，不会改为 Direct；历史另外保留 `ai_run_id`、`ai_proposal_id`、`ai_provider`。修改推荐配对后不再沿用该提案来源。结果版本更新后，旧建议只供阅读，不能应用。

### 完整 Trace 与会话复用

- **不发送完整 Trace（默认）**：保持原有统计证据路径。即使上一轮附加过 Trace，本轮也不会恢复那个后端会话；勾选追问时仅带最近 4 轮问答摘要。已发送给模型服务的数据不能撤回。
- **首次附加 / 重新附加**：发送与「更多输出」及时间线相同的 `gpu_trace.json`、`npu_trace.json`，范围为本次选中 Step/PID（选择全量时即为该次全量导出）。不按当前算子类型或排除类型进一步裁剪；完整保留导出 JSON 的事件参数，可能包含路径等元数据。页面显示两份文件大小，服务端校验 SHA-256，不静默截断。
- **复用已发送的 Trace**：首次成功后默认选中。使用 Codex `exec resume` 或 Harness `--session-id` 继续服务器专属 CLI 会话；本轮 stdin 不再附加 Trace 正文，仅增加问题、最新统计证据和文件清单。**这不是模型服务的文件上传缓存承诺**，后端仍可能传输、计费历史 Token，或压缩旧上下文。模型无法找回具体事件时，需明确说明并由用户主动选择重新附加。
- 只允许同任务、同结果版本、同后端及配置模型、同文件哈希、同会话最新成功轮次复用。旧分支、更换模型、CLI 会话丢失、超时、取消或校验失败后不能继续复用；不会自动重发文件。重新附加会创建独立后端会话；勾选追问时仍保留最近 4 轮摘要。
- 附件合计默认上限 **2 MiB**，可通过 `PROFILING_AI_TRACE_MAX_BYTES` 调整（1 KiB–64 MiB）。这是应用的字节上限，不是模型上下文容量保证；还需为证据、历史和输出预留 Token。超限会在提交前提示，可缩小 Step 范围或仅使用统计。不要把大 Trace 直接塞进不支持相应上下文的模型。
- 本地 CLI 会话保存在任务的 `ai-sessions/`，可能含完整 Trace、对话和 CLI 持久化事件。网页/分析 JSON 只返回附件清单和哈希，不复制 Trace 正文或公开 CLI 推理事件。删除最后一条引用该会话的 AI 记录时清理本地会话；删除最新轮次后不能从旧轮次复用。删除任务也会清理会话；模型服务端留存策略由提供方决定。

会话协议参考 [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode) 和 [Harness headless](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/bundle/headless/README.md)。仅保存成功且通过结果校验的会话为可复用状态。

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
| `PROFILING_AI_TRACE_MAX_BYTES` | 完整 Step Trace 附件合计字节上限，默认 2097152，范围 1024–67108864 |

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

使用专属 `DSH_HOME`，不沿用日常 Harness 配置和 GUI 登录。统计模式每轮临时创建，完整 Trace 模式持久化以便复用。自动挂载只读分析 overlay，关闭文件、shell、网络检索、子代理等工具，并在 required headless runner 初始化时安装禁止工具执行的 guard。不兼容该 API 时停止，不降级到无限制模式。禁用额外会话遥测及模型生成标题，运行结束删除临时 stdin/stdout/stderr；完整 Trace 模式保留 CLI 会话。此模式额外要求支持 `--json`、`--session-id` 以及持久化会话查询服务。

### Codex

安装 [Codex CLI](https://learn.chatgpt.com/docs/cli/reference)，需支持 `exec`、`--output-schema`、`--ephemeral`、`--ignore-user-config` 和 `--ignore-rules`。完整 Trace 模式额外要求 `--json`、`exec resume`（含输出 Schema 参数），并取消 `--ephemeral` 以保留会话。使用 [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)，不依赖桌面正在打开的聊天。

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

Web 仅复制该目录的 `auth.json` 到私有运行目录（完整 Trace 模式为专属会话目录），正常退出后删除，不沿用个人配置、MCP 和插件。设置 read-only、never approval，关闭 shell、apps、hooks、web search 等工具。凭据过期时需在专用目录重新登录，临时运行中的凭据刷新不回写。

## 数据范围与限制

- 读取现有 SQLite 索引，不重解析 Trace。全任务汇总覆盖全部所选类型；明细按问题及模式选取前 6 类型，每类型最多 4 条诊断、3 个 shape 组（shape 模式 8 个）、1 个 NPU 待关联调用及前 3 GPU 候选。字段数组最多 20 项，记录预算约 96 KB，截取/省略会标记。具体类型入口更适合深入排查。
- 这是已有证据分析及候选推荐，不是无限翻查 Trace 的交互代理。统计模式未采样调用不会被模型判断；附加完整 Trace 后可解释其中事件，但关联提案仍只能来自当前受控候选集合。没有建议不代表没有可关联项。无 CPU 场景仍需人工确认两侧整组语义，数量或耗时相近不是等价证明。
- 响应需符合 JSON Schema，引用与提案 ID 需属于本次证据/候选集合，否则不产生可应用建议。结构校验不能证明模型文字判断正确。
- 独立队列单并发，最多 8 个排队/运行分析；每任务最多 1 个活动分析、50 条历史。CLI 单份输出上限 2 MB，超时/取消终止进程树。没有硬性计费 token 上限，费用由模型服务计量。
- 问题、状态、证据、结果持久化在任务数据库，刷新可继续查看。原始 stderr 不公开或长期保留；完整 Trace 模式的 CLI 会话可能持久化推理事件，但不通过网页返回。服务重启标记旧活动分析为中断，需重新发起。删除任务需先结束 AI 分析，并一并删除 AI 历史。
- 服务数据目录应保持私有。断电/强制终止可能留下 `jobs/<id>/ai-scratch/` 或 `ai-sessions/` 文件，包括短暂凭据副本；正常退出会删除复制的登录凭据，确认没有 CLI 运行后可由管理员清理中断残留。建议使用独立系统用户部署。
- 回归测试使用合成数据和模拟模型，不自动使用桌面登录，也不发送真实用户 profiling 数据。真实服务的认证、额度及网络需在部署环境验证。

## API

| 操作 | 路径 |
| --- | --- |
| 后端就绪状态，无密钥 | `GET /api/ai/config` |
| 完整 Step Trace 清单与字节上限，不读正文 | `GET /api/jobs/{id}/ai-trace?attempt={attempt}` |
| 发起分析 | `POST /api/jobs/{id}/ai-runs` |
| 历史 / 单次结果 | `GET /api/jobs/{id}/ai-runs[/{run_id}]` |
| 取消 | `POST /api/jobs/{id}/ai-runs/{run_id}/cancel` |
| 删除历史 | `DELETE /api/jobs/{id}/ai-runs/{run_id}` |

发起请求包括 `attempt`、`provider: deepseek|codex`、`question`、`share_evidence:true`，以及可选 `scope:{op_type,device_id,core_type}`、`parent_id`、`trace_mode:none|attach|reuse`（默认 none；reuse 必须有 parent_id）。结果中的 `trace_context.reusable` 表示该轮是否仍可作为复用起点；`evidence.trace` 记录发送范围、哈希和本轮新增附件字节数。认证/Origin 校验与其余 API 相同。关联仍使用 `manual-batch/preview`、`manual-batch` 或 `manual-links`；额外传 `ai_run_id`、`ai_proposal_id` 时校验来源和版本，不新增绕过原校验的应用接口。

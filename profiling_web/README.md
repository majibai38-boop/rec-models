# Profiling Workbench：单机 Web 工作台

复用 `profiling_tools` 的对齐与诊断引擎，提供上传、范围预检查、后台任务、图表、证据查询和 Excel 下载。不需要 GPU/NPU 硬件，不调用外部 AI 服务。

这是**单机、可信用户共享工作区**，不是多租户服务。本机免令牌模式下，本机其他进程及操作用户也能访问数据；令牌模式下，持有令牌的用户可以读取及删除所有任务。默认仅监听本机；不要直接暴露到公网。

## 本机启动

在仓库根目录执行。Web 版本验证环境为 Python 3.12、Node.js 22、pnpm 11.25.0；核心 CLI 的 Python 要求不变。

Windows PowerShell：

```powershell
python -m venv profiling_web/.venv
profiling_web/.venv/Scripts/python -m pip install -r profiling_web/requirements.lock.txt
pnpm --dir profiling_web/frontend install --frozen-lockfile
pnpm --dir profiling_web/frontend build
profiling_web/.venv/Scripts/python -m profiling_web
```

打开 http://127.0.0.1:8000 ，默认直接进入工作台，无需令牌。`--auth auto`（默认）在回环监听、未显式提供 `PROFILING_WEB_TOKEN` 时选择本机模式；已有的令牌文件不妨碍本机免登录。若需要本机也认证，可运行：

```powershell
profiling_web/.venv/Scripts/python -m profiling_web --auth token
# 另开终端读取令牌，再粘贴到登录页；不要分享终端输出
Get-Content profiling_web/data/access-token
```

Linux 使用相同流程，将 `profiling_web/.venv/Scripts/python` 换成 `profiling_web/.venv/bin/python`。也可以通过 `PROFILING_WEB_TOKEN` 提供至少 24 字符的私有随机令牌，此时不会生成令牌文件。数据目录应只对运行服务的系统用户开放，尤其是 Windows 上应检查目录 ACL。

`--auth local` 只允许监听 `127.0.0.1`、`localhost` 或 `::1`；同时检查实际客户端地址、Host 和写请求 Origin，拒绝转发头，不信任代理提供的客户端地址。不能通过把监听地址改成 `0.0.0.0` 来关闭认证。局域网、反向代理与 Docker Compose 仍使用令牌模式。

`all`（默认）启动一个 API 进程和一个独立 Worker，Ctrl+C 停止。可分别运行 `python -m profiling_web serve` 与 `python -m profiling_web worker`，但必须使用相同的数据目录与配置；每个数据目录只允许一个 API 和一个 Worker。

```powershell
profiling_web/.venv/Scripts/python -m profiling_web all --port 8000 --data-dir profiling_web/data
```

前端开发可运行 `pnpm --dir profiling_web/frontend dev`，Vite 将 `/api` 代理到本机 8000 端口。正式部署使用构建后的静态资源，无需运行 Node 服务。

## 使用流程

1. 新建任务，上传 GPU trace JSON、NPU trace JSON 和 NPU `kernel_details.csv`。这版要求三者齐全，不接受压缩包或服务器任意路径。
2. 预检查后明确选择两端 CPU PID、稳定 step，或显式勾选完整采集。相同 step 编号不保证输入、精度和工作量等价。
3. 提交分析。页面显示队列/阶段状态，刷新不丢任务；可取消运行。
4. 在五个页签依次查看类型对比、整体对比、覆盖率诊断、调用明细和耗时分布，默认打开类型对比。类型表可跳转关联的整体分组，再下钻到完整调用事件；大结果按页查询。
5. 下载多页签 Excel 或其他报告。删除任务需要确认，将永久删除输入、所有运行结果和日志，不可恢复。

### Cube / Vector 总览

结果顶部展示两端计算 kernel 的数量、数量占比、`Duration(us)` 耗时和时间占比。NPU 按 `kernel_details.csv` 的 `aicore_time(us)` 分类：等于 0 为 Vector，大于 0 为 Cube，缺失、负数或无效值为未知；使用了 Vector 的混合 NPU kernel 仍按此规则归入 Cube。另行汇总 `aicore_time(us)` 和 `aiv_vec_time(us)`，展示有效记录数；计数器不是 kernel 总耗时，不与 Duration 相加，也不作为硬件利用率。

GPU 类 Cube/Vector 仅参考已对齐的 NPU 分类，包括人工确认关联，每个 GPU 计算 kernel 只计一次。若整组关联同时涉及 Cube 和 Vector，归为混合；涉及未知 NPU 类别则归为未知；未分配事件单列。数量/时间占比分母均为各端所选范围内全部计算 kernel。GPU 内存活动不参与此处分类，页面单列其数量和耗时，原 TOTAL 仍包含它们。

相同统计写入 `summary.json` 的 `core_breakdown`、`cube_vector_statistic.csv` 及 Excel 的「Cube-Vector总览」页签；NPU 映射明细保留每条 kernel 的分类及原始计数器值。旧任务需「复用输入，新建分析」后生成这些信息。

### 人工补充 GPU 关联

在「类型对比」中，对可比时间覆盖率不足 100% 的 NPU 类型点击「补充关联」：

1. 选择待核实的 NPU 调用组，默认按耗时降序排列。
2. 查看前三个 GPU 候选及推荐理由；也可按 kernel 名称、CPU op 或模块搜索其他未分配候选。优先推荐已对齐的完整 CPU scope，再参考模块、shape 和 dtype；排序不是匹配概率，不使用耗时相近作为匹配证据。
3. 人工核对调用范围后选择候选，查看预计覆盖率及整组时间，填写确认理由，点击「仅确认当前组」。后台重新生成报告，完成后网页及 Excel 同步更新。
4. 点击「人工关联 ×N」查看理由、撤销关联；撤销同样会重新生成报告。

重复调用可批量确认：选好一对 NPU/GPU 候选后，点击「预览同类批量关联」。系统以这一对为模板，在当前 NPU 类型、模块、CPU 调用结构和线程范围内扩展；GPU 候选还限制设备及 kernel 组成。优先使用唯一的已有 CPU 对齐；没有对齐时，仅在两端完整 shape/dtype 分组数量相等、调用顺序与已选配对和已有对齐一致时，提供「人工模板 + 调用顺序」配对。缺失元数据、弱 owner、数量冲突和重叠候选会排除并显示原因。旧报告缺少批量上下文时，需要复用输入重新分析。

NPU 通过内层实现下发 kernel 时，优先使用与所选 GPU CPU op 同名、且恰好覆盖当前 NPU kernel 组的最近外层调用。例如 NPU `aten::index_add_ → aten::index_add → IndexAdd` 对 GPU `aten::index_add_ → kernel`，使用两侧 `index_add_` 的元数据；内层多参数或缺失元数据不会误阻止批量扩展。含其他类型或兄弟调用 kernel 的祖先不用于这种替换。候选与预览中展示比较使用的 CPU scope，真实 kernel owner 保留。

若两侧比较 scope 的参数数量、shape 或 dtype 仍不相同，人工核实这一对等价后，可勾选「按人工确认模板扩展」重新预览。此模式分别使用所选 NPU 和 GPU 各自的完整 shape/dtype 作为模板，只纳入各侧同模板的重复调用，要求调用数量相等。模板内已有完整、唯一的一一 CPU 对应时，优先采用它，允许两端实际调用顺序不同，结果标记 `manual_confirmed_cpu_scope`；其余场景按各端时间顺序配对，要求与已有 CPU 对齐一致，标记 `manual_confirmed_template_order`。两种依据均属于人工确认，可整批撤销。弱 owner、缺失 shape/dtype 或重复使用 GPU 事件不会被强行放行。其他 shape 的调用需要另选模板预览，不能只按 kernel 类型全量绑定。

NPU 列表默认按耗时排序，GPU 推荐列表按证据排序，因此各自列表第一项不一定对应同一次调用。如果选中了 NPU 第 180 次、GPU 第 1 次，预览会展示同模板内的实际序号，并提供「改选对应 GPU 调用并重新预览」按钮，自动定位该 NPU 对应的 GPU 候选（即使不在当前候选页），再生成整批预览。此按钮只改选和预览，不提交关联；仍需填写理由并点击一次确认。若冲突来自其他调用的 CPU 对齐，则展示冲突编号和已有候选，不将其误报为当前选择位置错误。

更新引擎后，请重启 API / Worker、刷新网页，并对旧任务「复用输入，新建分析」；这样才能取得新增的外层调用信息并通过引擎版本校验。

预览展示调用对数、两端 kernel 数、预计覆盖率、前 20 对样本及例外；填写一次理由，点击「一次确认 N 对并更新报告」。例如两端各 260 次 `index_add_`，NPU 内层 `index_add` 下发 kernel，可以一次确认 260 对，底层保留每对的独立事件集合。整批只运行一次分析，历史中显示一条批次，可「撤销整批」。完整逐对明细和批次编号保存在 `manual_links.json`，NPU 映射明细也保存批次编号与配对依据。

每批最多 5000 对，单任务最多 100 次人工确认、合计 10000 对；一批 260 对算一次确认。确认时核对预览内容和运行版本，过期预览必须刷新，不会将旧事件编号套入新结果。

人工确认会明确标记为「非自动验证」，提升 Comparable/Accounted 覆盖率，不冒充 Direct 一一映射，也不修改真实 CPU owner。一个关联只包含同设备、同类型、同 Core Type 的 NPU kernel，按整组比较，不虚构组内 GPU 耗时分摊。GPU 事件不能重复分配，TOTAL 两端总时间不变。当前不支持包含 GPU 内存活动、跨 GPU 设备或已部分分配的 GPU 调用组。

自动分析原始运行保留，每次修改生成独立运行；失败、取消或中断时恢复上一份有效结果。输入、分析范围或引擎源码变化会拒绝沿用旧事件关联，此时请复用输入重新分析。旧报告缺少候选索引时，弹窗提供「复用输入」入口，不会在浏览时自动解析 trace。更新代码后需重启 API 与 Worker，并刷新页面。

人工理由与事件索引保存在可下载的 `manual_links.json` 中；候选索引不作为公开下载件。人工修改作用于 NPU-first 类型比较，旧版 GPU-first 自动映射报告仍保留自动结果。

### 任务与视图操作

- 点击标题旁“修改名称”内联编辑，支持中文，去除首尾空格后须为 1–120 个字符。任务列表支持搜索、状态筛选、分页和归档；归档不删除数据，也不释放空间。
- “复用输入，新建分析”校验并复制被冻结的三份输入，保留原任务及结果，继承分析设置后重新预检查。独立副本会额外占用输入大小的磁盘空间；删除原任务不影响新任务。运行或上传中的任务不能复用。
- 类型表默认简洁模式，勾选“详细模式”查看 kernel 数量和三种数量/时间覆盖率。类型、覆盖率图表和证据应用相同搜索与热点筛选；整体调用页独立提供调用名、比较状态和排序。
- 面包屑返回下钻前的筛选与页码。地址栏记录任务、页签、筛选和事件定位；刷新可恢复，链接不包含令牌。链接仅对能访问同一服务的人有效，包含任务和算子标识，请勿公开分享敏感分析链接。
- “保存视图”将显示设置保存到当前浏览器，可删除；不保存文件、密码、具体运行事件或分析配置。更改 PID/step/诊断生成阈值需要新建运行，显示筛选不会补生成缺失诊断。
- “导出当前筛选 CSV”导出当前类型表或整体/调用/未映射事件列表，不限当前分页，并附筛选范围；超过 10 万行时提示缩小范围。完整 Excel/报告下载保持原始全范围。“覆盖率诊断”页此按钮导出上方类型表。
- 调用详情以结构化事件表和 CPU 元数据证据展示，JSON 收入折叠区域。按调用补充最多 200 个 NPU 映射记录的单事件耗时，并展示最多 5 条诊断样本；超出范围明确标记，未知耗时不均摊、不补零。
- 上传支持拖拽与批量选取文件，角色识别仅依据文件名，须检查确认；取消或失败后保留已上传成功文件，只重试剩余文件，不支持文件内断点续传。
- “服务状态”按需查看 Worker、队列、磁盘用量和资源上限；不是持续资源监控图。完成任务的后台刷新降为约 30 秒，运行中约 2.5 秒，隐藏页面暂停轮询。

两个性能比均为 GPU/NPU：Comparable Ratio 比较可比子集，Total Time Ratio 比较总体统计范围。`>= 2` 深绿色，`>= 1 且 < 2` 浅绿色，`< 1` 黄色。没有分母时显示空值，不补零。

Direct、Comparable、Accounted 均分别显示数量/时间覆盖率；Accounted 不等于性能可比。GPU 未映射计算 kernel 和内存活动保留独立记账，参与总体时间。事件时长求和不是墙钟时间，整体算子视图与类型视图重叠，不能相加。图表沿用当前类型筛选，最多显示 200 类（柱图 Top 12）；计时总卡片始终使用全范围统计。

整体对比统一展示已有 CPU bundle、FlashAttention 和通用匹配 CPU scope；Div/Sub 精度路径仍只在排查报告。元数据不完整或关联不足的记录显示为待核实，不计算可比比值。完整调用内包括辅助 kernel 与 GPU 内存活动，组内相同事件不重复计入；跨组、跨页签不做累加 TOTAL。Total Time 比值仅在两端全部同名逻辑调用能归入单一实现组合时提供，包含未配对调用；其余显示空值并说明范围未建立。旧任务仍可读，但需重新分析才有新增字段/页签，GET 不会重新解析 trace 或补建报告。

## Docker Compose

仓库包含构建和部署配置；当前开发环境没有 Docker，**容器路径尚未实际运行验证**。

创建本地 `profiling_web/.env`（已被 Git 忽略），设置 `PROFILING_WEB_TOKEN=<私有随机令牌>`。可用 Python `secrets.token_urlsafe(32)` 生成，不要使用固定示例密码或提交令牌文件。

```bash
docker compose --env-file profiling_web/.env -f profiling_web/compose.yaml up --build -d
docker compose --env-file profiling_web/.env -f profiling_web/compose.yaml logs -f
docker compose --env-file profiling_web/.env -f profiling_web/compose.yaml down
```

访问本机 8000 端口。API 与 Worker 共用命名数据卷，停止容器不会删除数据；不要使用 `down -v`，除非明确要永久清除数据。镜像使用非 root 用户，运行时文件系统只读，数据卷和临时目录可写。

## 配置与资源边界

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `PROFILING_WEB_DATA` | `profiling_web/data` | 数据目录，CLI `--data-dir` 可覆盖 |
| `PROFILING_WEB_TOKEN` | 自动生成 | 共享访问令牌，至少 24 字符 |
| `PROFILING_WEB_AUTH` | CLI 为 auto | local / token；CLI 还支持 auto。可用 `--auth` 覆盖；直接调用 Python app factory 默认为 token |
| `PROFILING_WEB_UPLOAD_MB` | 1024 | 单个文件上限，单位 MiB |
| `PROFILING_WEB_STORAGE_MB` | 20480 | 数据目录空间配额，单位 MiB |
| `PROFILING_WEB_MEMORY_MB` | 8192 | 分析进程树 RSS 总和软限制（含子进程，共享页可能重复计入），单位 MiB |
| `PROFILING_WEB_TIMEOUT` | 1800 | 单次分析/预检查超时，秒 |
| `PROFILING_WEB_HOSTS` | localhost,127.0.0.1,[::1] | 接受的 HTTP Host，逗号分隔 |
| `PROFILING_WEB_HTTPS` | 0 | HTTPS 反向代理下设 1，启用 Secure Cookie |
| `PROFILING_WEB_WORKER_MEMORY` | 10g | 仅 Compose：Worker 容器硬内存上限 |

并发固定为 1；每次分析使用独立子进程。RSS/目录配额是定期采样的保护，不是硬隔离，瞬时峰值可能超限。Docker 额外限制 API 内存 1 GiB、Worker 默认 10 GiB。解析器仍会完整加载 JSON，较大 trace 需要提高资源限制或缩小采集范围；预检查和正式分析会分别读取输入。

需要局域网访问时，必须显式设置令牌、监听地址和允许的 Host。建议通过 HTTPS 反向代理，正确保留原始 Host/协议并仅信任指定代理。当前没有多用户权限、登录限流和公网防护，不应将它视为可直接公网部署的产品。

## 数据、恢复与安全

`tasks.sqlite`（WAL）保存任务状态；`jobs/<随机ID>/` 保存输入及各次运行目录。报告校验成功后生成只读查询索引，后续浏览只查询索引，不再解析 trace。输入在预检查开始后冻结，修正输入需新建任务。

Worker 重启时将遗留运行标为 `interrupted`，不会静默重跑。失败/取消/中断可手动重试，每次使用新目录并保留旧结果；成功任务不原地覆盖。暂不自动过期清理，请定期在页面删除不需要的任务。备份前停止 API 与 Worker，或使用 SQLite 备份机制；不要只复制正在运行的主数据库而遗漏 WAL。

上传流式落盘，服务端固定文件路径并记录哈希；用户不能指定执行命令、配置文件路径或下载任意服务器文件。浏览器使用 HttpOnly、SameSite Cookie，写请求校验 Origin；API 客户端可用 Bearer 令牌。日志保留在本机，可能包含输入相关信息，应限制目录访问。

为防止表格公式注入，CSV 下载会转义公式样式的文本；磁盘中的原始报告不修改，因此下载 CSV 与原报告的字节哈希可能不同。只公开允许的报告，内部 manifest、服务器路径和原始 trace 不通过报告下载接口公开。

## API 与测试

API 路径均以 `/api` 开头。令牌模式除登录外需要 Cookie 或 `Authorization: Bearer <token>`；本机模式无需凭据，但写请求必须提供与服务一致的 `Origin`（如 `http://127.0.0.1:8000`）。本机模式不接受代理转发头：

| 操作 | 接口 |
| --- | --- |
| 登录 / 注销 | POST / DELETE `/session` |
| 配置与 Worker 状态 | GET `/config` |
| 创建 / 列表 | POST / GET `/jobs` |
| 名称 / 归档 | PATCH `/jobs/{id}`，`name` / `archived` |
| 复用输入 | POST `/jobs/{id}/clone` |
| 服务状态 | GET `/health` |
| 任务状态 | GET `/jobs/{id}` |
| 上传文件原始请求体 | PUT `/jobs/{id}/inputs/{gpu_trace,npu_trace,npu_kernel_details}` |
| 预检查 / 分析 / 取消 | POST `/jobs/{id}/{inspect,run,cancel}` |
| 分页结果 | GET `/jobs/{id}/results?kind=operators&limit=30&offset=0` |
| 调用事件证据 | GET `/jobs/{id}/call-evidence?call_id=…&offset=0&limit=200` |
| 下载 | GET `/jobs/{id}/artifacts/{name}` |
| 永久删除非运行任务 | DELETE `/jobs/{id}?confirm=true` |

`run` 请求只接受白名单选项，如 `gpu_pid`、`npu_pid`、`gpu_step`、`npu_step`、`all_steps` 和诊断阈值。结果种类包括 operators、mappings、diagnoses、reasons、calls、gpu_unmapped、whole_calls、call_details；支持类型、设备、核心和 kernel 编号过滤，单页最多 200 条。整体/调用查询支持 `search`、`sort=npu_time|ratio|name`、`comparison_status=comparable|pending`。任务列表支持 `search`、`status`、`archived`、`offset` 和 `limit`。

```powershell
profiling_web/.venv/Scripts/python -m pip install -r profiling_web/requirements-dev.txt
profiling_web/.venv/Scripts/python -m unittest discover -s profiling_web/tests -v
profiling_web/.venv/Scripts/python -m unittest discover -s profiling_tools/tests -v
pnpm --dir profiling_web/frontend test
```

可选真实 HTTP 回归（先启动服务，在客户端环境设置对应 `PROFILING_WEB_TOKEN`；会新建任务并保存输入）：

```powershell
profiling_web/.venv/Scripts/python -m profiling_web.smoke --url http://127.0.0.1:8000 --gpu-trace profiling_tools/example/demo1/gpu/trace.json --npu-trace profiling_tools/example/demo1/npu/trace_view.json --npu-kernel-details profiling_tools/example/demo1/npu/kernel_details.csv --gpu-step 15 --npu-step 15
```

已验证：核心 139 项测试（包括桌面 Excel 所要求的 OOXML 元素顺序）、Web 24 项测试、前端状态工具 4 项测试及前端生产构建。浏览器已验证本机免令牌访问、修改任务名、文件选择上传、预检查、指定 Step 分析、Excel 下载、页面内保存视图，以及真实报告的类型→整体→调用证据下钻和刷新恢复。标准 `ProfilerStep#1` 会规范为 CSV 使用的 Step `1`，自定义 Step 名称保持原样。此前真实 demo1 step 15 HTTP 回归结果保留原引擎口径（NPU 13987.133 μs、GPU 14242.9 μs）。

后续扩展边界：暂不支持断点上传、多用户隔离、跨任务缓存、跨轮对比和完整时间线查看。CLI/Skill 不依赖 Web 组件，原有调用方式不变。

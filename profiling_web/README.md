# Profiling Workbench：单机 Web 工作台

复用 `profiling_tools` 的对齐与诊断引擎，提供上传、范围预检查、后台任务、图表、证据查询和 Excel 下载。不需要 GPU/NPU 硬件，不调用外部 AI 服务。

这是**单机、可信用户共享工作区**，不是多租户服务：持有令牌的用户可以读取及删除所有任务。默认仅监听本机；不要直接暴露到公网。

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

打开 http://127.0.0.1:8000 。首次启动生成随机令牌，启动日志只显示令牌文件位置。另开终端读取并粘贴到登录页（不要分享终端输出）：

```powershell
Get-Content profiling_web/data/access-token
```

Linux 使用相同流程，将 `profiling_web/.venv/Scripts/python` 换成 `profiling_web/.venv/bin/python`。也可以通过 `PROFILING_WEB_TOKEN` 提供至少 24 字符的私有随机令牌，此时不会生成令牌文件。数据目录应只对运行服务的系统用户开放，尤其是 Windows 上应检查目录 ACL。

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

API 路径均以 `/api` 开头，除登录外需要 Cookie 或 `Authorization: Bearer <token>`：

| 操作 | 接口 |
| --- | --- |
| 登录 / 注销 | POST / DELETE `/session` |
| 配置与 Worker 状态 | GET `/config` |
| 创建 / 列表 | POST / GET `/jobs` |
| 任务状态 | GET `/jobs/{id}` |
| 上传文件原始请求体 | PUT `/jobs/{id}/inputs/{gpu_trace,npu_trace,npu_kernel_details}` |
| 预检查 / 分析 / 取消 | POST `/jobs/{id}/{inspect,run,cancel}` |
| 分页结果 | GET `/jobs/{id}/results?kind=operators&limit=30&offset=0` |
| 下载 | GET `/jobs/{id}/artifacts/{name}` |
| 永久删除非运行任务 | DELETE `/jobs/{id}?confirm=true` |

`run` 请求只接受白名单选项，如 `gpu_pid`、`npu_pid`、`gpu_step`、`npu_step`、`all_steps` 和诊断阈值。结果种类包括 operators、mappings、diagnoses、reasons、calls、gpu_unmapped；支持类型、设备、核心和 kernel 编号过滤，单页最多 200 条。

```powershell
profiling_web/.venv/Scripts/python -m pip install -r profiling_web/requirements-dev.txt
profiling_web/.venv/Scripts/python -m unittest discover -s profiling_web/tests -v
profiling_web/.venv/Scripts/python -m unittest discover -s profiling_tools/tests -v
```

可选真实 HTTP 回归（先启动服务，在客户端环境设置对应 `PROFILING_WEB_TOKEN`；会新建任务并保存输入）：

```powershell
profiling_web/.venv/Scripts/python -m profiling_web.smoke --url http://127.0.0.1:8000 --gpu-trace profiling_tools/example/demo1/gpu/trace.json --npu-trace profiling_tools/example/demo1/npu/trace_view.json --npu-kernel-details profiling_tools/example/demo1/npu/kernel_details.csv --gpu-step 15 --npu-step 15
```

已验证：核心 126 项测试（包括桌面 Excel 所要求的 OOXML 元素顺序）、Web 13 项测试、真实 demo1 step 15 HTTP 上传至 Excel 下载全流程，以及前端生产构建。真实数据结果保留原引擎口径（NPU 13987.133 μs、GPU 14242.9 μs）。浏览器登录和页面显示已检查；浏览器文件选择上传因自动审批额度限制未完成，不能视为浏览器端到端验证通过。

后续扩展边界：暂不支持断点上传、多用户隔离、跨任务缓存、跨轮对比和完整时间线查看。CLI/Skill 不依赖 Web 组件，原有调用方式不变。

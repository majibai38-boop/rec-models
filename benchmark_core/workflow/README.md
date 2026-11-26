# 执行合同：workflow、manifest 与设备锁

这是 P1-01 的软件合同层，只做**校验、解析和状态记录**，不执行模型、不启动容器、不连接远端。三个执行步骤（环境搭建、模型运行、性能采集）的脚本属于 P1-02～P1-04，尚未实现。

合同把用户配置拆成两层：

- `recbench.benchmark-spec/v1` 保持严格，只描述测什么（模型、数据、target、矩阵、测量协议、正确性）。
- `recbench.workflow/v1` 负责“在哪跑、怎么归档、是否开 profiling”，通过 `target_id` 选中一个 target，不往 BenchmarkSpec 里塞未知字段。

## 快速使用

```console
python -m benchmark_core.workflow check benchmark_catalog/workflows/dlrm-small-local.json
python -m benchmark_core.workflow resolve benchmark_catalog/workflows/dlrm-small-local.json --output benchmark-results/first-resolution
```

`resolve` 只接受**尚不存在**的输出目录，并写出五个文件：

| 文件 | 内容 |
|---|---|
| `request.json` | 原始 workflow、路径、入口工作区和 workflow hash |
| `resolved_workflow.json` | 绝对路径、选中 target、节点/配方身份、归档目的地、各类 hash |
| `resolved_spec.json` | 仅含所选 target 的有效 BenchmarkSpec，可直接通过 v1 校验 |
| `plan.json` | 复用 `benchmark_core.planning` 的离线计划与阻塞原因 |
| `admission.json` | 分层准入结果（见下节） |

示例的 `environment_recipe` 仍是 `draft`、也没有真实快照，因此 `resolve` 返回退出码 3、状态 `blocked`。这不是错误，而是信息不足时的预期结果；`null` 一律表示待办，不会被补成猜测值。

## 分层准入

单目标运行需要先满足环境条件，再产生模型状态 hash，最后才允许正式测量，避免“必须先有全部 hash 才能准备环境”的循环。

| 阶段 | 通过条件 | 未通过时的典型原因 |
|---|---|---|
| `structure` | workflow/ spec / 配方 / 节点配置互相引用正确 | 未知 target、配方与 target 环境不一致 |
| `environment` | 配方为 `pinned`；所选节点快照存在、未过期、身份与规则匹配 | `draft_environment`、`missing_inventory`、`inventory_expired`、`requirement_mismatch` |
| `workload_state` | 源码、数据、checkpoint、训练状态、正确性策略 hash 已冻结 | `missing_hash` |
| `pilot` | `measurement.measure_steps` 已由 pilot 选定 | `missing_measure_steps` |
| `measurement` | 以上三段全部通过 | `prerequisites_blocked` |

`execution_ready` 在所有输出中恒为 `false`：合同通过不等于设备已验证。

## 退出码

| 退出码 | 含义 |
|---|---|
| 0 | `check` 结构有效；或 `resolve` 各阶段均 eligible |
| 2 | 输入错误：非法 JSON、字段/引用/路径错误、输出目录已存在 |
| 3 | `resolve` 成功但存在阻塞阶段（计划已保存） |
| 4 | 设备锁被占用（P1-02 起由执行脚本使用） |
| 5 | 执行被中断（记录在 `logs/interruptions.jsonl`） |

## 运行状态与设备锁

`benchmark_core/workflow/state.py` 提供后续脚本共用的约定：

- attempt 目录：`<archive_root>/runs/<run_id>/<attempt_id>/`；新 attempt 必须新建目录，已存在的 attempt 绝不复用；重跑用新 attempt，归档重试沿用原 attempt。
- 设备锁：每台机器每个设备一个 `<state_root>/<backend>-<device>.lock`，用 `O_CREAT|O_EXCL` 原子创建，记录 pid、主机、设备和获取时间；被占用时立即失败或按超时等待。`break_stale` 只有在证明锁龄超过阈值后才删除。
- 中断记录：`record_interrupt` 追加写入 `logs/interruptions.jsonl`，不改写已有记录。

## Manifest / 指标 / 产物

`benchmark_core/workflow/manifest.py` 定义最小合同，不含任何成绩计算：

- `manifest.json`（`recbench.run-manifest/v1`）：run/attempt、来源 hash、执行设备、归档目的地、`prepare / run / measure / profile` 四个阶段、结果有效性。
- 阶段状态机：`pending → running → succeeded/failed/interrupted`；`pending → skipped/blocked`；`blocked → running`。成功必须有 log 或 detail 证据，失败/中断/阻塞必须写明原因。
- `metrics.json`（`recbench.metrics/v1`）：名称、单位、范围、聚合方式、值、样本数、有效性（`valid/invalid/unsupported/not_measured`）与原因。不支持或未测量的指标必须为 `null` 并写明原因，不得填 0，也不得生成伪造的 P95/P99。
- `artifacts.json`（`recbench.artifacts/v1`）：相对 POSIX 路径、大小、SHA256、媒体类型；禁止绝对路径、`..`、反斜杠和保留文件名；`verify_artifacts` 会重新计算并报告缺失、大小或 hash 不符。
- `write_json` 使用临时文件加替换，避免留下半写文件。

## 归档默认值

| 配置 | 默认与约束 |
|---|---|
| `execution.transport` | 第一阶段仅 `local`：在启动 Skill/CLI 的宿主机执行 |
| `archive.transport` | 默认 `local`，可选 `ssh` |
| `archive.host` | local 时为 `localhost`；ssh 时填写已有 SSH 配置别名 |
| `archive.root` | local 默认入口工作区下 `benchmark-results/archive`；ssh 必须填写目标机绝对 POSIX 路径 |
| `archive.web_base_url` | 可选；未配置时只返回归档位置，不编造网页链接 |

`--entry-workspace` 指定“启动流程的入口工作区”，默认当前目录；归档默认值以它为准，不随 workflow 文件位置改变。SSH 目标的 `staging_root` 是执行机上的本地暂存目录，P1-05 才会创建和传输。

## 测试

```console
python -m unittest benchmark_core.workflow.tests.test_workflow -v
```

测试覆盖合同校验、路径与归档解析、单目标准入（含只需要所选 target 快照）、阶段状态机、指标与产物合同、设备锁与中断记录。所有夹具都是合成的，**不构成 GPU/NPU 运行证据**。

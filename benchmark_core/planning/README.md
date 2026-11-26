# 环境配方与实验规划

这是已实现的离线配置层（原 M0，现由 P1-01～P1-04 复用）：将 `BenchmarkSpec`、环境配方、节点声明和设备快照展开为可审阅的执行计划。只依赖 Python 标准库，建议 Python 3.10+。不安装依赖、不连接远端、不启动容器。现阶段没有 `run` 命令。2026-11-25 的[第一阶段计划](../../docs/recommendation-benchmark-platform-roadmap.md)将 Skill 三步执行、本地/SSH 归档和只读 Web 列为下一阶段开发项；本文描述的现有 CLI 和 v1 合同没有因此改变。

## 快速使用

在仓库根目录运行，PowerShell 和 Linux 使用相同的 Python CLI：

```console
python -m benchmark_core.planning check spec benchmark_catalog/experiments/dlrm-small-pair.json
python -m benchmark_core.planning check recipe benchmark_catalog/environments/cuda.template.json
python -m benchmark_core.planning check recipe benchmark_catalog/environments/ascend.template.json
python -m benchmark_core.planning plan --spec benchmark_catalog/experiments/dlrm-small-pair.json --recipe benchmark_catalog/environments/cuda.template.json --recipe benchmark_catalog/environments/ascend.template.json --inventory-config benchmark_catalog/inventory/nodes.example.json --output benchmark-results/first-plan
```

示例生成 2 个 batch × 2 个 target = **4 个计划条目**，预期退出码 **3**、状态 `blocked`。报告会列出未采集节点、草稿环境、未冻结源码/数据/初始状态/正确性策略，以及待 pilot 确定的共同测量步数。`null` 是显式待办，不会被替换成猜测值。

完成真实设备[盘点](../inventory/README.md)后，使用自己的节点配置与实验配置，并重复传入 `--snapshot path/to/snapshot.json`。节点 ID、profile ID、device ID 必须对应实际采集结果；不能直接使用示例主机名。`--recipe` 可以指定仓库外自有配方；匹配使用配方中的 `id`，不是文件名。

```console
python -m benchmark_core.planning plan --spec experiments.local.json --recipe cuda.local.json --recipe ascend.local.json --inventory-config inventory.local.json --snapshot inventory-results/gpu/snapshot.json --snapshot inventory-results/npu/snapshot.json --max-age-hours 24 --output benchmark-results/pair-plan
```

输出目录必须不存在。`plan.json` 保存完整解析配置、逐项兼容性结果、阻塞原因、工作负载条件和指纹；`report.md` 提供可读摘要。规划结果不是 `RunManifest` 或性能成绩，重复执行必须选择新目录。

| 退出码 / 状态 | 含义 |
|---|---|
| 0 / `valid_structure` | `check` 仅检查文档结构，包括草稿；不检查运行条件 |
| 0 / `eligible_for_runtime_check` | `plan` 的离线条件已满足，可以进入后续运行前检查 |
| 3 / `blocked` | 计划已保存，有未解析内容、采集缺失或不兼容条件 |
| 2 / `error` | 非法 JSON、字段错误、引用错误、重复输入或文件错误 |

所有输出的 `execution_ready` 当前恒为 `false`。程序不会把 `pinned`、设备被发现或规则匹配解释为运行验证通过。

## 环境配方

格式为 `recbench.environment-recipe/v1`。CUDA 与 Ascend 使用同一结构，各自选择独立镜像与版本。采集命令继续由 inventory profile 管理；环境配方只存声明与约束，规则不会执行 shell、代码或网络请求。

| 字段 | 含义 |
|---|---|
| `id` / `backend` | 配方 ID；当前 backend 为 `cuda` 或 `ascend` |
| `state` | `draft` 表示待补齐；`pinned` 表示版本与兼容性依据已明确登记 |
| `image` | 不可变的 `repository@sha256:<64 位小写摘要>`；不接受只有 tag 的镜像 |
| `runtime` | 容器内预期的精确 Python、torch、toolkit 版本；toolkit 分别指 CUDA 或 CANN；Ascend 必须有 torch_npu |
| `requirements` | 从主机或指定设备的已观测字段求值，全部规则为 AND |
| `compatibility` | HTTPS 依据地址、审核日期及说明；它是人工登记证据，不是自动认证 |

`pinned` 必须填写镜像、必需 runtime 版本、兼容性来源/日期，并提供 OS、CPU 架构、设备型号、驱动规则；Ascend 还必须有固件规则。实际审核需核对镜像中 Python/PyTorch/toolkit 的组合，并按目标机型补充 OS 发行版、内存等限制。未知版本不得用一个宽泛下限“补齐”。

规则例子（仅展示语法，不代表厂商兼容性结论）：

```json
[
  {"field": "host.architecture", "op": "in", "value": ["x86_64", "aarch64"]},
  {"field": "device.model", "op": "eq", "value": "实际采集到的完整型号"},
  {"field": "device.driver_version", "op": "version_gte", "value": "550.9"},
  {"field": "device.memory_capacity_bytes", "op": "gte", "value": 42949672960}
]
```

- `eq`：精确相等；`in`：文本白名单。字符串区分大小写，不做模糊型号匹配。
- `gte`：仅用于字节容量，拒绝布尔值和字符串数字。
- `version_gte`：仅用于驱动/固件的纯数字点分版本，按数字比较，`550.10 > 550.9`；补零后 `1.2 == 1.2.0`。RC、厂商后缀或未知值返回 `unknown`，需用明确的 `eq` / `in` 规则，不猜测顺序。
- 可用字段：`host.os`、`host.architecture`、`host.os_distribution`、`host.memory_total_bytes`，以及 `device.vendor/kind/model/driver_version/firmware_version/memory_capacity_bytes/health`。新字段须在合同中注册并加对应采集证据；规则列表无 shell 或 eval 扩展点。
- `unknown` 与 `mismatch` 都阻塞。Ascend 内存目前保留厂商 HBM/DDR 单位，缺少规范化字节容量时不能靠相加或猜测通过内存规则。

配置化规则可支持不同型号、架构和精确版本组合；引入新的执行后端仍需注册对应 backend adapter，不能仅新增 JSON 就声称已支持执行。

模板参考 [NVIDIA CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)与 [Ascend PyTorch 兼容性表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)，没有替实际机器选择组合。NVIDIA 的版本兼容存在功能限制，单个驱动下限不构成工作负载验证。选定环境后应记录精确版本对应的固定文档/提交链接及说明。声明与采集事实分开管理的思路也参考了 [MLPerf sysinfo 配置](https://docs.mlcommons.org/mlperf-sysinfo/configuration/)；本格式不是 MLPerf 官方提交格式。

## 实验合同与测量口径

`recbench.benchmark-spec/v1` 严格校验顶层与固定结构字段，拒绝拼写错误、重复 JSON key、NaN/Infinity 和隐式类型转换。v1 范围是单卡、FP32、eager、common-portable、`train_step`；未知 schema version 不自动迁移。

模型参数保存在 `workload.parameters`，目前只验证它是有限 JSON 对象，语义与模型支持能力留给 P1-03 adapter 校验。小型 DLRM 示例是参数草案，还没有实现对应模型或生成训练数据。`source_sha256` 用于冻结模型实现内容；数据、checkpoint、完整训练状态与正确性策略各有独立 SHA256。当前只校验摘要格式，后续运行器还必须验证实际文件内容。

`training_state_sha256` 指向模型/optimizer/RNG/数据游标的恢复契约；每次重复恢复后执行相同 warmup 和输入序列。正确性策略应定义输出、loss、梯度或更新检查及容差，不接受将“合成数据 loss 有限”视作业务质量通过。

测量配置统一指定：

- 共同的预热步数、正式步数、独立重复次数；`measure_steps: null` 在 pilot 完成前阻塞。建议初始采用 50 warmup、5 repeats；单目标 pilot 选择 N，记录实际窗口时长；需要跨设备可比时再冻结双方共同 N，不能各自按时间提前停止。第一阶段允许只运行一个目标。
- `step_components` 固定为 zero_grad → forward → loss → backward → optimizer_step；数据加载和传输是否纳入由 `scope` 决定。`end_to_end` 包含获取已冻结输入及主机到设备传输；`device_only` 使用预先放在设备上的输入，加载/传输在窗口外。两种口径不能混排。
- `sync_policy: before_and_after_window` 表示正式墙钟窗口开始前、结束读时钟前都等待设备完成。指标是整窗吞吐，不由该设置推导逐 step 延迟或 P99。
- 状态恢复在正式窗口外；正式测量禁止 profiler，独立 profiling 重放将在 P1-04 增加。

这些是平台合同约束，尚未证明计时实现或模型在硬件上的行为；P1-04 要在实际运行器中验证后再冻结。

## 快照与指纹

规划检查节点声明/profile 的 hash、主机身份、盘点完整性与指定设备；默认只接受过去 24 小时以内的快照，可用 `--max-age-hours 1..8760` 调整。未来时间、无时区时间、未知设备、缺失节点都阻塞。快照是一次观测，不能替代后续设备租约或运行前重新探测。

NVIDIA ID 是采集到的物理设备 ID；Ascend ID 是 `NPU ID:Chip ID`。它们不自动等同于容器内 runtime ordinal。规划不会替换设备或生成未经验证的挂载/设备映射命令。

| 指纹 | 范围 |
|---|---|
| `spec_sha256` | 整份实验声明，包括目标选择 |
| `workload_sha256` | 模型、数据、状态、质量策略、实际 batch/precision/devices 和测量协议；不含目标名或环境 |
| `planned_environment_sha256` | 完整环境配方、观测主机/指定设备、版本探针和可见性上下文 |
| `snapshot_sha256` | 使用的完整快照，包括采集时间和原始证据 |
| `plan_key` | workload + 目标选择 + 计划环境；只是确定性规划键，不是运行/attempt ID |

时间戳和设备动态 health 不进入计划环境指纹；每次捕获的证据通过 snapshot hash 追溯。采集器 Python 中的 torch 包不是未来容器里的 torch，因此不用于判断 runtime 版本一致；容器内版本与 digest 必须另做运行验证。配方和证据变化会保守地产生新的环境指纹。

`RunManifest / Metric / Artifact`、产物真实性检查、环境构建安装、运行前 smoke、真实模型执行与比较资格仍待后续实现；本轮不生成占位运行成绩。

## 验证

```console
python -m unittest discover -s benchmark_core -v
```

新增 15 项离线规划测试，加上 28 项盘点测试共 43 项。覆盖结构错误、配方缺失、数值版本比较、快照缺失/过期/配置漂移、指定设备、指纹分离、序列化、防覆盖及 CLI 退出码。测试环境版本和设备名称都是合成 fixture，不表示硬件兼容认证。

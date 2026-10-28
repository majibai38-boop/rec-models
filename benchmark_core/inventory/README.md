# 设备盘点：节点配置、采集 Profile 与版本化快照

这是 M0-01 的工具实现。使用 Python 3.10+ 标准库，在目标节点本地运行，不需要安装 PyTorch、CUDA、CANN 或额外 Python 包。软件包版本通过当前 Python 环境的 metadata 查询，不导入或初始化加速器运行时。

**支持范围：** Linux / Windows 主机，NVIDIA GPU、Linux 昇腾 NPU；可通过 profile 接入新的采集工具。首版提供 CLI 和离线多节点汇总，尚未接入 Web、SSH 自动分发、Agent 或模型 smoke test。

## 为什么分开配置和采集

不同设备的命令不相同。同一厂商也可能因为 OS、驱动、安装路径和命令版本而不同，所以将流程拆成三层：

| 层 | 存储位置 | 内容 |
|---|---|---|
| 节点声明 | `inventory.local.json` 或自建 JSON 清单 | 节点 ID、预期 hostname/OS、所选 profiles、预期设备数量、访问方式、标签和备注 |
| 采集规则 | `benchmark_catalog/inventory/profiles/*.json` | argv 参数数组或候选文件路径、OS 范围、解析器、是否必需、是否逐设备执行 |
| 实测快照 | 每次采集的新目录 | `snapshot.json` 和 `report.md`，包含事实、声明、证据、失败状态、配置与版本指纹 |

JSON 配置便于版本控制，也可由 Web 表单生成；首版无需 YAML 依赖。节点声明不覆盖探测值。例如配置预期 8 张卡，实际发现 4 张时，输出仍为 4，同时标记不匹配。采集失败的数量是 `null`，不是 0。

参考 [xpu-perf 的后端注册](https://github.com/bytedance/xpu-perf/blob/main/projects/infer_perf/general_perf/core/configs/backend_store.py)对硬件差异分层，以及 [MLPerf sysinfo 配置](https://docs.mlcommons.org/mlperf-sysinfo/configuration/)对声明与探测信息的区分。这里是独立实现，没有复制其代码，也没有将本平台快照冒充 MLPerf 官方提交格式。

## 快速开始

以下命令均在仓库根目录运行。`python` 应替换为节点上真实可用的 Python 路径。

```bash
# 在目标 GPU 节点生成本机声明，hostname 自动填写。
python -m benchmark_core.inventory init --node gpu-01 --backend nvidia --expected-devices 8 --output inventory.local.json

# 在目标昇腾节点使用以下初始化方式；不要在同一路径重复覆盖。
python -m benchmark_core.inventory init --node npu-01 --backend ascend --expected-devices 8 --output inventory.local.json

# CPU / 控制节点只采集主机信息，使用 --backend none（默认值）。
python -m benchmark_core.inventory init --node control-01 --output inventory.local.json
```

选择上面适合当前节点的一条初始化命令，然后依次执行：

```bash
python -m benchmark_core.inventory check --config inventory.local.json
python -m benchmark_core.inventory plan --config inventory.local.json --node gpu-01
python -m benchmark_core.inventory collect --config inventory.local.json --node gpu-01 --output inventory-results/gpu-01-001
```

`check` 只校验配置，不探测主机可达性；`plan` 打印完整命令和逐设备模板，不执行命令；`collect` 在当前机器执行。节点名应替换为实际配置中的 ID。输出目录必须不存在，历史快照不会被覆盖。

程序要求当前 hostname/OS 与节点声明一致，防止在控制电脑上误采集远程节点。`access.method` 支持 `local / ssh / scheduler`，其中 SSH 别名、调度系统和备注目前仅作资产登记，**不会自动连接远程主机**。在每台节点上复制 `benchmark_core/` 和 `benchmark_catalog/`，通过已有登录/调度方式执行采集，再将结果取回汇总。无需复制现有 profiling Web 的依赖或数据。

## 节点配置

完整的双节点模板见 [nodes.example.json](../../benchmark_catalog/inventory/nodes.example.json)。编辑其中的实际 hostname、卡数和访问别名后使用。

```json
{
  "schema_version": "recbench.inventory-config/v1",
  "execution": {
    "timeout_seconds": 15,
    "max_output_bytes": 262144,
    "max_devices": 64
  },
  "profile_files": [],
  "nodes": [{
    "id": "gpu-01",
    "expected_hostname": "actual-hostname",
    "expected_os": "Linux",
    "profiles": ["linux-host", "nvidia"],
    "expected_devices": {"nvidia": 8},
    "access": {"method": "ssh", "target": "gpu-lab", "notes": "使用已有 SSH config 别名"},
    "labels": {"site": "lab", "role": "benchmark"}
  }]
}
```

- 每个节点选择一个 host profile，可附加不同厂商的 accelerator profile。一个厂商只选择一个发现 profile，避免重复统计。
- 配置未知键、重复节点、重复 JSON 键、无效解析器或 OS 组合会被拒绝。
- `expected_devices` 可以省略；它的计数对象是 profile 定义的设备单元。NVIDIA 是管理工具可见的物理 GPU；Ascend 是计算芯片，不等于板卡数量。
- `max_output_bytes` 限制每个命令的 stdout 和 stderr 各自的捕获大小；文件读取同样受限。超限、超时、权限不足或工具缺失都会保留为证据。
- 本机声明和默认快照目录已加入 Git 忽略规则；团队共享清单可另选路径提交。不在配置中保存密码或私钥。

## 不同后端如何采集

| Profile | 主要采集方式 | 差异与限制 |
|---|---|---|
| `linux-host` | `lscpu --json`、`/proc/meminfo`、`/etc/os-release` | CPU/内存是必需项；NUMA、网卡、PCI、DMI 和 cgroup 信息按工具/权限保留 |
| `windows-host` | PowerShell `Get-CimInstance`，输出 JSON | CPU、内存、系统型号、BIOS、OS；网卡独立可选采集 |
| `nvidia` | `nvidia-smi --query-gpu=... --format=csv,noheader,nounits` | 保存 UUID、管理索引、型号、PCI、内存、驱动；固件、拓扑、实例信息、nvcc 是独立采集项 |
| `ascend` | `npu-smi info -m`，随后按物理 NPU ID / Chip ID 查询 | 保存逻辑 ID 与物理/芯片 ID；逐芯片查 board、usages、health；驱动/CANN 文件和拓扑单独采集 |

当前 Ascend 映射解析器支持 `NPU ID / Chip ID / Chip Logic ID / Chip Name` 四列格式；其它格式会明确报告 `parse_error`，保留原文，不猜测设备。各型号 `-t` 子命令或 `-c` 支持情况不同，按节点版本配置 profile。NPU 型号不能用来推断服务器型号。

NVIDIA CSV 的内存单位按查询接口记为 MiB，并换算 bytes。Ascend 的 HBM/DDR 容量保留命令中的 MB 标签及 `vendor_reported`，不擅自解释为 MiB，也不将不同存储层相加。

Windows/Linux 的 NUMA、拓扑数据可能不可用；未知字段列在 `unknown_fields`。GPU 的管理索引不等于 `CUDA_VISIBLE_DEVICES` 重映射后的进程索引；MIG/vNPU 分区信息不直接转换成可调度资源数。

## 扩展或修改 Profile

将现有 JSON profile 复制到自己的配置目录，例如 `custom-nvidia.json`，保持 `id: nvidia`，按实际情况调整命令路径、采集项或必需性。在节点配置中增加：

```json
"profile_files": ["custom-nvidia.json"]
```

路径相对节点配置文件解析；自定义 profile 显式替换同 ID 的内置 profile，不做隐式深层合并。完整解析后的 profile 与 hash 会写入快照，因此修改规则后汇总能识别旧结果。

每个 probe 必须指定 `argv` 或 `paths` 之一。命令以参数数组执行，不作隐式 shell 插值；Windows 的 CIM 采集使用明确配置的 PowerShell 脚本。逐设备命令只替换完整的 `{device_id}`、`{chip_id}` 参数，不从输出生成命令文本。`paths` 按顺序读取首个可访问文件，不支持 shell/glob 展开。

接入新厂商时，可以让其只读工具输出标准 JSON 数组，再注册如下 profile，无需修改采集编排：

```json
{
  "schema_version": "recbench.inventory-profile/v1",
  "id": "example-xpu",
  "kind": "accelerator",
  "vendor": "ExampleVendor",
  "platforms": ["Linux"],
  "probes": [{
    "id": "devices",
    "argv": ["/opt/vendor/bin/inventory-json", "--list"],
    "parser": "json_devices",
    "required": true
  }]
}
```

工具输出合同：

```json
[{"id":"0","model":"Example XPU","memory_capacity_bytes":17179869184,"driver_version":"1.2"}]
```

`id`、`model` 必需，内存字段为 bytes 整数或 null；还可提供 `physical_id / chip_id / runtime_id / uuid / pci_bus_id / firmware_version / health`。上面仅是扩展示例，不代表已经支持或实测某个新厂商。原始格式需要专用解析时，在 `parsers.py` 增加解析器、注册名称并补充对应版本的输出测试。Profile 是受信任的运维代码配置，快照中的输出始终作为数据读取。

## 结果与离线汇总

快照包含：

- `declared`：人工填写的节点、预期数量和访问信息。
- `host / accelerators / software`：实际探测值；字段来源通过 `host_sources` 和设备 `sources` 关联到 evidence。
- `evidence`：命令或文件、状态、返回码、时间、受限原文、解析结果和错误。
- `config_sha256 / profiles_sha256 / observed_configuration_sha256`：声明、采集规则和观测配置的指纹。
- `runtime_context`：容器提示、设备可见性环境变量；这些只是上下文，不计算可调度资源预算。
- `benchmark_readiness: not_validated`：盘点成功不代表运行时、数值或模型执行已经验证。

`software` 的 Python 包版本仅代表运行采集器的 Python 环境；CANN、driver、nvcc 等版本文本保存在 `software.version_reports` 及对应 evidence，并参与观测配置指纹。`nvidia-smi` 能支持的 CUDA 版本不能当作已安装 toolkit 版本；本工具只用独立的 `nvcc --version` 记录该工具的信息，也不保证所有环境都安装了 nvcc。

```bash
python -m benchmark_core.inventory summary --config fleet.json inventory-results/gpu-01/snapshot.json inventory-results/npu-01/snapshot.json
```

带 `--config` 时，未返回的节点标记 `not_collected`，配置或采集规则变化的节点标记 `stale`；不能只因收到一个节点结果就声称全机群完成。不带配置时，只汇总明确提供的快照。每个节点只接受一份快照，需主动选取版本，不自动挑“最新”文件。

| 状态/退出码 | 含义 |
|---|---|
| `complete` / 0 | 该配置要求的采集成功；未采集字段仍可能 Unknown |
| `complete_with_warnings` / 0 | 必需项成功，可选项有问题 |
| `partial` / 3 | 必需采集失败、计数不符，或汇总时有缺失/过期节点；仍保存诊断快照 |
| `error` / 2 | 配置错误、机器身份不符、输出目录已存在或文件错误 |

## 验证与边界

```bash
python -m unittest discover -s benchmark_core/inventory/tests -v
```

测试使用明确的合成命令输出，覆盖双后端、多芯片、未知值、配置扩展、命令参数、超时、输出限制、重复/缺失/过期快照和 CLI。开发环境另做 Windows 主机真实采集；Linux、NVIDIA、Ascend 的端到端真实采集仍需目标服务器验证，不能用测试夹具代替 M0-01 的节点验收。

本模块不修改现有 Workload Modeler 数据库，不将其人工填写的峰值算力当成探测事实。后续运行 manifest 可引用快照路径/hash；环境兼容性和设备可运行性检查由 M0-03 / M1 单独实现。

## 一手参考

- [xpu-perf](https://github.com/bytedance/xpu-perf)：分硬件后端和配置组织方式；本次核对了当前 `projects/infer_perf/general_perf/core/configs/backend_store.py`。
- [MLPerf sysinfo](https://github.com/mlcommons/mlperf-sysinfo) 与[配置说明](https://docs.mlcommons.org/mlperf-sysinfo/configuration/)：声明、采集、输出和机群完整性分离。本工具不生成官方 `system_desc.json`。
- [NVIDIA SMI 文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)：CSV 查询与跨 OS 支持。官方指出 CLI 输出不保证跨版本兼容；本实现保留原文并拒绝未知格式，后续可增加 NVML 后端。
- [Ascend device queries](https://github.com/Ascend/agent-skills/blob/master/skills/npu-smi/references/device-queries.md)：设备/芯片标识、按硬件版本变化的查询能力。

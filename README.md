# 推荐模型 Benchmark 平台

第一阶段聚焦 **环境搭建 → 模型运行 → 性能采集**，由 `rec-benchmark` Skill 编排脚本执行。结果默认归档到启动 Skill/CLI 的本机，也可指定 SSH 归档机器；Web 负责只读浏览模型性能、日志和归档文件。

当前分支 `codex/rec-benchmark-platform-plan` 已实现设备盘点和离线实验规划。三步执行脚本、Skill、归档传输和 Web 尚待实现；短期不建设集中调度或常驻 Agent。详见[第一阶段方案](docs/recommendation-benchmark-platform-proposal.md)。

## 当前能力

| 模块 | 已实现 |
|---|---|
| [设备盘点](benchmark_core/inventory/README.md) | 按 OS/厂商配置采集命令、本地采集、原始证据、版本化快照、多节点离线汇总 |
| [环境与实验规划](benchmark_core/planning/README.md) | 环境配方、BenchmarkSpec、兼容性规则、快照校验、矩阵展开、指纹与阻塞报告 |
| [执行合同](benchmark_core/workflow/README.md) | workflow 校验与单目标解析、分层准入、manifest/指标/产物合同、设备锁与中断记录；不执行模型 |
| [rec-benchmark Skill](skills/rec-benchmark/SKILL.md) | 三步流程编排说明；当前仅 `check`/`resolve` 可用，执行脚本待 P1-02～P1-04 |
| [配置目录](benchmark_catalog/) | Linux/Windows 主机与 NVIDIA/Ascend profile、环境模板、小型 DLRM 实验草案与 workflow 示例 |

工具只依赖 Python 标准库，建议 Python 3.10+。结构校验成功不代表运行环境已通过验收。待定版本、镜像 digest、模型/数据/状态 hash 和测量步数会明确阻塞计划。

## 快速开始

在仓库根目录执行。先为当前机器生成配置，再预览并采集；目标 GPU/NPU 应在对应节点执行采集。

```console
python -m benchmark_core.inventory init --output inventory.local.json --node local-control --backend none
python -m benchmark_core.inventory check --config inventory.local.json
python -m benchmark_core.inventory plan --config inventory.local.json --node local-control
python -m benchmark_core.inventory collect --config inventory.local.json --node local-control --output inventory-results/local-control
```

已有配置应直接复用，输出目录应选择新名称。无硬件也可检查实验草案并生成离线计划：

```console
python -m benchmark_core.planning check spec benchmark_catalog/experiments/dlrm-small-pair.json
python -m benchmark_core.planning plan --spec benchmark_catalog/experiments/dlrm-small-pair.json --recipe benchmark_catalog/environments/cuda.template.json --recipe benchmark_catalog/environments/ascend.template.json --inventory-config benchmark_catalog/inventory/nodes.example.json --output benchmark-results/first-plan
```

示例生成 4 个计划条目，预期状态为 `blocked`、退出码为 3，原因保存在 `plan.json` 和 `report.md`。真实使用时填写实际节点配置，并用 `--snapshot` 传入采集结果。

执行合同把“测什么”与“在哪跑、怎么归档”分开。示例 workflow 只选一个 target，配方仍是 draft、没有快照，所以同样返回退出码 3：

```console
python -m benchmark_core.workflow check benchmark_catalog/workflows/dlrm-small-local.json
python -m benchmark_core.workflow resolve benchmark_catalog/workflows/dlrm-small-local.json --output benchmark-results/first-resolution
```

`resolve` 写出 `request.json`、`resolved_workflow.json`、`resolved_spec.json`、`plan.json` 和 `admission.json`；阻塞阶段与原因以 `admission.json` 为准。合同通过不等于设备已验证，`execution_ready` 恒为 `false`。

## 开发与计划

```console
python -m unittest discover -s benchmark_core -v
```

- [平台方案](docs/recommendation-benchmark-platform-proposal.md)：Skill 三步流程、归档配置、产物协议与只读 Web。
- [实施计划与验收](docs/recommendation-benchmark-platform-roadmap.md)：按 P1-01～P1-06 推进，真实硬件能力单独验收。
- [分支边界](docs/recommendation-benchmark-repository-audit.md)：保留模块及与历史工具的分离记录。

最小 workflow/归档合同与 Skill 流程骨架已定义（P1-01），环境准备、小型 DLRM 运行与性能采集脚本尚未实现。保留 GPU/NPU 后端与可选独立 profiling，先完成单目标流程，再逐端验收。上面的命令仅对应已有盘点/规划/合同工具，不会执行新计划中的三步流程。

## 许可证

[Apache License 2.0](LICENSE)。

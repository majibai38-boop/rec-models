# Benchmark 分支边界

更新日期：2026-11-25。分支：`codex/rec-benchmark-platform-plan`。本分支聚焦推荐模型 benchmark 的配置、执行、测量、追溯及未来的 Web / Skill 入口。

2026-10-29 的初始审计曾考虑复用 `auto_tools` 的分析与估算实现。根据新的分支范围要求，2026-11-25 移除 Profiling Workbench、Workload Modeler 及其专属实现，后续平台开发不再依赖这些模块。

| 当前保留 | 职责与边界 |
|---|---|
| `benchmark_core/inventory/` | 配置化本地采集、原始证据、版本化快照、离线汇总；不表示设备已能运行模型 |
| `benchmark_core/planning/` | 环境配方和实验配置校验、矩阵展开、兼容性规则、指纹与阻塞报告；不执行模型 |
| `benchmark_catalog/` | 采集 profile、节点示例、环境模板和实验草案 |
| `docs/` | 平台方案、实施计划与本边界记录 |

移除目录：`profiling_tools/`、`profiling_web/`、`workload_modeler/`、`web_shared/`、`skills/gpu-npu-profiling/`，以及专属演示 `docs/demo/`、`docs/media/`。相应启动说明、依赖安装入口、主题和截图一并移除。

移除时这些目录的完整本地内容（包括未提交改动及本地数据）已移至 `.local-archive/auto-tools-20261125/`，旧根 README 也在该目录备份。备份目录及 `.worktrees/` 均被 Git 忽略，不属于平台源码或交付内容。原 `auto_tools` 分支及其独立 worktree 保留；本次清理不改写 Git 历史，也不将尚未提交的本地改动自动同步到其他分支。

设备盘点和离线规划仅依赖 Python 标准库及 `benchmark_core` 内部模块。按 2026-11-25 调整后的[第一阶段方案](recommendation-benchmark-platform-proposal.md)，新增 `rec-benchmark` Skill、三步执行脚本、本地/SSH 归档和只读 `benchmark_web`。集中 `benchmark_service`、调度、Agent 与团队权限延后，不预设已有服务或 UI 可复用。平台仍需实现正式性能测量与可选独立 trace 采集；单侧估算不在当前范围。

清理验收：运行 `python -m unittest discover -s benchmark_core -v`，检查保留代码无被移除模块的 import，检查文档无指向移除文件的链接。原 M0-02 的分支清理结论保留；后续任务按 P1-01～P1-06 跟踪，硬件与运行验收状态不因计划调整而改变。

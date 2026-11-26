---
name: rec-benchmark
description: 推荐模型 benchmark 第一阶段入口。校验并解析 recbench.workflow/v1，按“环境搭建 → 模型运行 → 性能采集”三步调用统一脚本，归档到本机或指定 SSH 机器，并返回摘要与浏览链接。当前仅有合同校验与 resolve 可用，执行脚本待 P1-02～P1-04。
---

# rec-benchmark：Skill 主流程

本 Skill 负责**编排**，不自己计算结果：每一步的具体动作由 `python -m benchmark_core.workflow` 等脚本执行，Skill 读取脚本的退出码、阶段 JSON 和实际产物来判断成功，再从自然语言或日志里猜结论。

当前状态（2026-11-26）：**只有合同层可用**。`check` 与 `resolve` 已实现并有测试；环境准备、模型短跑、性能采集、归档传输和只读 Web 属于 P1-02～P1-06，尚未实现。不要用“合同通过”代替“设备已验证”。

## 何时使用

- 用户要为一个已声明的 BenchmarkSpec 目标跑单卡、FP32、eager、固定步数 train step。
- 用户需要把结果归档到本机或指定 SSH 机器，并拿到可浏览的位置。
- 用户要在运行前确认配置是否完整、缺什么。

不适用：多卡多机、服务推理、正式 MLPerf 成绩、模型移植咨询、跨设备倍率报表。这些属于后续阶段。

## 输入收集

在调用脚本前确认这些信息，缺一项就停下来问，不要自行猜：

1. **workflow 文件路径**（推荐从 `benchmark_catalog/workflows/dlrm-small-local.json` 复制后修改）。
2. **目标**：`target_id` 必须是 spec 里已声明的 target；一次只选一个。
3. **环境配方**：`environment_recipe` 的 `id` 必须等于所选 target 的 `environment`。
4. **归档目的地**：默认本机入口工作区下 `benchmark-results/archive`；只有用户明确要求时才改为 `transport=ssh`，并要求给出 SSH 别名与目标机绝对路径。
5. **profiling 开关**：默认关闭；正式测量期间不得开启。
6. **入口工作区**：启动本流程的宿主机目录，默认当前目录，可用 `--entry-workspace` 显式指定。

## 第一步：校验与解析（已实现）

```console
python -m benchmark_core.workflow check <workflow.json>
python -m benchmark_core.workflow resolve <workflow.json> --output <新目录> [--snapshot <快照.json>]... [--entry-workspace <目录>]
```

`resolve` 写出 `request.json`、`resolved_workflow.json`、`resolved_spec.json`、`plan.json`、`admission.json`。向用户报告 `admission.json` 中每个阶段的 `status` 与 `blockers`，不要改写或隐藏阻塞原因。

退出码：`0` 通过；`2` 输入错误；`3` 已保存但存在阻塞阶段。退出码 3 不是故障，是“还缺条件”，应逐条列出 `blockers`。

## 第二步：环境搭建（P1-02，待实现）

预期脚本职责：按 pinned 配方检查宿主机驱动与容器工具、拉取固定 digest 镜像、验证设备透传、核对容器内 Python/PyTorch/toolkit 版本、运行算子 smoke，并归档设备快照、版本、镜像 digest 与日志。

在该脚本可用之前：只做 `check` / `resolve`，把 `environment` 阶段的 `draft_environment`、`missing_inventory` 等阻塞原样报告给用户，并说明需要真实执行节点。不要用本机 Windows 盘点结果声称 GPU/NPU 环境已就绪。

## 第三步：模型运行（P1-03，待实现）

预期脚本职责：生成固定输入与初始训练状态并记录 hash、显式选择设备、验证 forward/loss/backward/optimizer update、拒绝静默 CPU fallback，并保存日志与正确性结果。

在该脚本可用之前：`workload_state` 阶段会一直阻塞（`missing_hash`）。不要手填 hash，也不要把合成数据的数值检查说成业务质量验收。

## 第四步：性能采集（P1-04，待实现）

预期脚本职责：独立 pilot 冻结 `measure_steps`、按重复次数恢复完整训练状态、窗口前后同步、输出吞吐/窗口均摊 step 时间/内存峰值/原始样本，可选独立 profiling 重放并将原始 trace 单独归档。

在该脚本可用之前：不要对外宣称任何吞吐、延迟或加速比；不要用 kernel 时长求和代替墙钟吞吐，也不要生成 P95/P99。

## 第五步：归档（P1-05，待实现）

预期行为：本地暂存 → 完整性清单 → 原子发布 `COMPLETE`；`ssh` 目标先上传临时目录、校验大小/hash 后再发布。相同 attempt 相同内容重试幂等，不同内容冲突不覆盖，失败保留本地证据并支持只重试归档。

在该步骤可用之前：只报告 `resolved_workflow.json` 里的归档配置，不要声称文件已经归档。

## 第六步：返回结果（P1-06，待实现）

- 只有 `archive.web_base_url` 已配置且服务确实可访问时，才返回浏览链接；否则返回归档位置。
- 返回内容应包含：run/attempt、target 与设备、各阶段状态、指标及其范围、缺失项与原因。
- 失败实验也可以有完整归档，但必须如实标注缺失阶段，不能说成成功。

## 失败与中断处理

- 任何阶段失败：保留证据，按退出码区分输入错误（2）、阻塞（3）、设备占用（4）、中断（5）。
- 中断记录写入 `logs/interruptions.jsonl`；恢复前先确认进程、互斥锁和已有产物，再决定继续同一 attempt 还是新建 attempt。
- 重新跑模型或测量用**新 attempt**；归档重试沿用**原 attempt**，不重跑 benchmark。
- 第一阶段不承诺对话结束后自动后台续跑。

## 必须遵守

- 不伪造 hash、不关闭校验、不把 `null` 补成猜测值。
- 不把合同校验、mock 测试或合成数据当成硬件验收或真实成绩。
- 不自行计算第二套统计口径；指标由脚本产生，Skill 只解释。
- 不改动已校验的归档内容；需要变更时新建 attempt。

# GPU/NPU Profiling Workbench

面向 PyTorch GPU 与 Ascend NPU profiling 数据的本地对齐、性能比较和覆盖率诊断工具。

项目以 NPU `kernel_details.csv` 中的 kernel 类型为统计主轴，同时结合两端 trace 中的 CPU `aten::` 调用关系，将 GPU kernel 映射到对应的 NPU 类型。对于一对多、多对一和融合算子，不强制伪造逐 kernel 一一对应，而是保留可比子集、整体调用和未映射活动三类证据。

## 目录结构

```text
.
├── profiling_tools/          # 无 Web 依赖的解析、对齐、诊断和报告引擎
│   ├── tests/                # 核心算法与报告回归测试
│   ├── pyproject.toml        # 独立 Python 包配置
│   └── README.md             # CLI、指标和输出文件说明
├── profiling_web/            # 单机 Web 工作台
│   ├── frontend/             # Vue + ECharts 前端
│   ├── tests/                # API、任务队列和安全边界测试
│   ├── Dockerfile
│   ├── compose.yaml
│   └── README.md             # 本机与容器部署指南
├── skills/gpu-npu-profiling/ # 可分发的 Codex Skill
├── LICENSE
└── README.md
```

三部分共享同一套 `profiling_tools` 分析语义：

```text
GPU trace ─┐
NPU trace ─┼─> inspect ─> scope selection ─> compare ─> validate ─> reports/index
NPU CSV ───┘                                      │
                                                  ├─ CLI
                                                  ├─ Web UI
                                                  └─ Codex Skill
```

## 输入

每次比较需要：

- GPU PyTorch profiler trace JSON；
- NPU `trace_view.json`；
- NPU `kernel_details.csv`。

预检查会列出两端 CPU PID 和可用 step。正式比较必须分别指定 GPU/NPU 的 PID 与 step，或显式选择完整采集范围。相同 step 编号并不能证明 batch、shape、精度和工作量相同。

## CLI 快速开始

核心工具仅依赖 Python 3.9+ 标准库。在仓库根目录执行：

```bash
python -m profiling_tools inspect \
  --gpu-trace gpu.json \
  --npu-trace trace_view.json \
  --npu-kernel-details kernel_details.csv

python -m profiling_tools compare \
  --gpu-trace gpu.json \
  --npu-trace trace_view.json \
  --npu-kernel-details kernel_details.csv \
  --gpu-step 15 \
  --npu-step 15 \
  --output-dir result \
  --json

python -m profiling_tools validate --result-dir result
python -m profiling_tools summary --result-dir result
python -m profiling_tools query --result-dir result --op-type Slice --limit 5
```

也可以独立安装核心包：

```bash
python -m pip install ./profiling_tools
```

完整参数、指标口径和输出文件见 [profiling_tools/README.md](profiling_tools/README.md)。

## Web 工作台

Web 版提供大文件流式上传、PID/step 选择、持久化任务队列、取消与恢复、分页查询、覆盖率图表和多页签 Excel 下载。

Windows PowerShell：

```powershell
python -m venv profiling_web/.venv
profiling_web/.venv/Scripts/python -m pip install -r profiling_web/requirements.lock.txt
pnpm --dir profiling_web/frontend install --frozen-lockfile
pnpm --dir profiling_web/frontend build
profiling_web/.venv/Scripts/python -m profiling_web
```

然后打开 `http://127.0.0.1:8000`。默认本机部署直接进入工作台，无需访问令牌；可用 `--auth token` 显式启用认证。

该服务定位为单机、可信用户共享工作区。本机免令牌模式只允许回环地址直接访问；局域网和容器部署保留令牌认证。支持任务重命名、归档、复用输入、保存视图和调用证据下钻。Docker Compose、资源限制、数据恢复及安全说明见 [profiling_web/README.md](profiling_web/README.md)。

## 主要输出

- `npu_gpu_op_statistic.xlsx`：类型对比、整体对比、覆盖率诊断、调用明细和耗时分布五个页签；
- `npu_gpu_op_statistic.csv`：完整类型级统计；
- `npu_kernel_gpu_mapping.csv`：NPU kernel 到 CPU op/GPU 活动的逐项证据；
- `coverage_diagnostic.md`：低覆盖率类型的通用排查报告；
- `gpu_unmapped_activity.csv`：未映射 GPU 计算与内存活动；
- `summary.json`、`validation.json`：机器可读摘要与一致性校验结果。

性能比统一使用 `GPU 时间 / NPU 时间`。Comparable Ratio 只比较可比子集，Total Time Ratio 比较所选范围内的总体统计。Direct、Comparable、Accounted 覆盖率含义不同，不能相互替代；整体算子视图与类型视图也不能累加。

## Codex Skill

`skills/gpu-npu-profiling/` 是仓库内可分发版本，封装了稳定命令入口、输入契约、指标定义和排查流程。它复用本仓库引擎，不维护另一套映射算法。

安装或更新 Skill 前可先运行：

```bash
python skills/gpu-npu-profiling/scripts/run_compare.py --version
```

## 验证

```bash
python -m unittest discover -s profiling_tools/tests -v

profiling_web/.venv/Scripts/python -m unittest discover \
  -s profiling_web/tests -v
```

当前回归范围包括 trace 解析、CPU/kernel 所有权、融合映射、覆盖率与时间记账、诊断报告、Excel 的桌面兼容性，以及 Web 上传、鉴权、任务恢复、资源限制和下载边界。

## 许可证

项目采用 [Apache License 2.0](LICENSE)。

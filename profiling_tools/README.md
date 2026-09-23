# GPU/NPU Profiling 对比工具

该工具从 GPU 与 NPU 的 PyTorch Profiler trace 中提取 `aten::` 调用，通过执行顺序进行对齐，并把每个 aten 实例直接下发的 device kernel 作为一个 **kernel bundle** 进行比较。NPU 的 `kernel_details.csv` 用于为匹配到的 GPU bundle 补充 NPU `Type`、`Core Type`、shape 和 dtype。

工具不会假设 GPU 与 NPU 的物理 kernel 数量相同：数量相等时会输出顺序映射，数量不等时保留 bundle 级映射，避免伪造逐 kernel 的一一关系。

## 环境

- Python 3.9+
- 运行比较功能只使用 Python 标准库，不需要额外安装依赖。
- 命令应在仓库根目录执行。

建议 GPU、NPU 使用相同输入，并只采集一个稳定 step。采集时建议启用 `record_shapes=True`；如需 Module/源码辅助对齐，再启用 `with_stack=True` 或在两边加入同名 `record_function` 标记。

## 基本用法

```bash
python -m profiling_tools \
  --gpu-trace /path/to/gpu.pt.trace.json \
  --npu-trace /path/to/ASCEND_PROFILER_OUTPUT/trace_view.json \
  --npu-kernel-details /path/to/ASCEND_PROFILER_OUTPUT/kernel_details.csv \
  --output-dir /path/to/compare_output
```

只比较指定 step：

```bash
python -m profiling_tools \
  --gpu-trace /path/to/gpu.pt.trace.json \
  --npu-trace /path/to/trace_view.json \
  --npu-kernel-details /path/to/kernel_details.csv \
  --gpu-step 3 \
  --npu-step 3 \
  --output-dir /path/to/compare_output
```

`--gpu-step 3` 可以匹配名为 `ProfilerStep#3` 的 trace 区间。若不指定 step，工具会处理 trace 中的全部 aten 事件，结果可能同时包含 warmup、稳定运行和抖动阶段。

## GPU/NPU aten 名称不一致

创建 JSON 映射文件，例如 `op_map.json`：

```json
{
  "aten::gpu_custom_op": "aten::npu_custom_op"
}
```

然后增加：

```bash
--op-map /path/to/op_map.json
```

默认只选择名称以 `aten::` 开头的事件。比较其他框架层事件时可以使用：

```bash
--op-prefix "my_namespace::"
```

多进程 trace 默认选择 aten 事件数量最多的 Host PID，也可以显式指定：

```bash
--gpu-pid 12345 --npu-pid 23456
```

通信事件默认排除；需要同时纳入 NCCL/HCCL 类事件时增加 `--include-communication`。

完整参数：

```bash
python -m profiling_tools --help
```

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 输入、PID/Step、关联覆盖率、匹配覆盖率和输出清单 |
| `matches.csv` | 每个 GPU/NPU aten 匹配及两边 kernel bundle 的时间比较 |
| `gpu_kernel_mapping.csv` | GPU kernel 到 NPU Type/shape 的映射；明确标记 `one_to_one`、`ordered_equal_count` 或 `bundle` |
| `aten_op_statistic.csv` | 参考 `op_statistic.csv`，按规范化后的 aten 类型汇总调用数、kernel 数和时间 |
| `kernel_type_statistic.csv` | 按 NPU Type/Core Type 汇总匹配后的 GPU/NPU bundle |
| `module_statistic.csv` | 按共同 `record_function`/Module 路径进行粗粒度汇总，可用于 aten 数量不一致时比较整体时间 |
| `unmatched_gpu.csv` | 未匹配的 GPU aten 及其 kernel bundle |
| `unmatched_npu.csv` | 未匹配的 NPU aten 及其 kernel bundle |

## 时间字段

- `Kernel Sum(us)`：bundle 中所有 kernel duration 之和；多 Stream 并发时可能重复计算并发区间。
- `Device Busy(us)`：所有 kernel 时间区间的并集，并发区间只计算一次。
- `Bundle Span(us)`：第一个 kernel 开始到最后一个 kernel 结束，包含 kernel 之间的空隙。

比较算子工作量时看 `Kernel Sum`；比较实际设备占用看 `Device Busy`；观察拆分 kernel 带来的间隙和完成延迟时看 `Bundle Span`。

## 匹配和限制

1. aten 实例首先按规范化名称和执行顺序对齐，shape、dtype、Module 路径、调用栈和相对位置用于计算置信度。
2. kernel 优先通过 `External id` 关联到 aten；其次使用 kernel/runtime correlation 和 trace flow。Ascend flow 允许 Host PID 与 `Ascend Hardware` PID 不同，按 TID/Stream 和时间戳继续关联。
3. NPU 设备任务同时支持 `cat=kernel` 和 `process_name=Ascend Hardware` + `Task Type=AI_CORE` 等格式。
4. `kernel_details.csv` 是 NPU kernel 的权威清单：优先用 `Task ID + Stream ID` 关联 trace kernel，缺失时按 Stream、名称、时间和 duration 降级匹配；trace 中完全缺失的 CSV 行会生成 synthetic kernel，再通过 flow 关联 aten。
5. `summary.json` 中的 `device_process_ids`、`task_type_counts`、`kernel_detail_matched_count` 和 `kernel_detail_synthetic_count` 可用于判断 NPU 数据采用了哪条提取路径。
6. `low` 置信度或低 kernel-duration coverage 的结果应人工检查，不应作为严格性能结论。
7. CUDA Graph、NPU 静态图或编译器大范围融合可能丢失逐 aten 的下发关系。这类数据应加入相同的 `record_function`/NVTX/MSTX 区域，主要查看 `module_statistic.csv`。
8. 当前实现会将 trace JSON 整体读入内存；超大 trace 建议先只导出一个 step。

## 测试

```bash
python -m unittest discover -s profiling_tools/tests -v
```

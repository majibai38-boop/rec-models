# 可复现的公开演示

这里的 GPU/NPU Trace 和 CSV 都是人工构造的演示输入，不是实测模型，不可用于硬件性能结论。截图来自真实 Web 工作台，结果由本仓库引擎运行生成。

## 运行

Web：上传 `gpu.json`、`npu.json`、`kernel_details.csv`，分别选择 CPU PID 1 和 3，两侧 Step 1，诊断阈值保持 90%。

CLI：在仓库根目录执行以下命令，输出目录须不存在或为空。

```bash
python -m profiling_tools inspect --gpu-trace docs/demo/gpu.json --npu-trace docs/demo/npu.json --npu-kernel-details docs/demo/kernel_details.csv
python -m profiling_tools compare --gpu-trace docs/demo/gpu.json --npu-trace docs/demo/npu.json --npu-kernel-details docs/demo/kernel_details.csv --gpu-pid 1 --npu-pid 3 --gpu-step 1 --npu-step 1 --output-dir result-demo --json
python -m profiling_tools validate --result-dir result-demo
python -m profiling_tools summary --result-dir result-demo
```

第一条 `inspect` 未指定 Step，返回 `needs_selection` 是预期提示；第二条显式选择两侧 PID 和 Step 后进行分析。不要把预检查提示理解为采集文件损坏。

## 预期结果

- 两端各 4 个计算 kernel，无内存活动；GPU 165 μs、NPU 100 μs。
- MatMulV3 / Add / Relu 的可比倍率分别为 2.4 / 0.8 / 1.5。
- Cast 的 NPU 15 μs 不可比，原因是构造的 NPU 独有 CPU 调用没有跨后端匹配；**不是 Div/Sub 缺陷示例**。
- GPU 独有调用的 10 μs 单列为未映射，并计入 GPU 总时间。
- 可比 NPU 时间为 85 μs，可比 GPU 时间为 155 μs；可比时间覆盖率 85%，数量覆盖率 75%。
- NPU Cube/Vector 时间各为 50 μs。GPU 按已映射 NPU 类型参考分类，未映射单列。
- 仅提供少量计数器，不足以判断流水瓶颈，Unknown 是预期结果。

2026-10-22 演示运行：引擎 0.2.0，`validate` 返回 valid，107 项检查通过、0 项失败。后续新增校验可能改变检查数量，应以当次 validate 为准。

完整报告由上述命令生成；不分发包含本机绝对路径的运行清单。公开素材只包括合成输入、经过检查的界面截图和本说明。

## 演示素材维护

GIF 是五张真实截图组成的分步演示，每帧停留 3 秒；它不是连续录屏或速度测试。原始 JPEG 同目录保留为静态阅读替代。更新界面后重新截图，再运行：

```bash
python -m pip install Pillow
python docs/media/build_demo_gif.py
```

截图顺序：选择输入 → 确认范围 → 总体结果 → 类型对比 → MatMul Shape 图表。使用隔离数据目录录制，只创建公开合成任务，不展示现有业务列表、令牌、用户路径或私有模型名。不要通过修改 DOM、伪造接口响应或手填表格来制造分析结果。

README 信息架构参考：

- [Streamlit](https://github.com/streamlit/streamlit)：短 GIF 紧邻安装和最小样例，展示用户可以完成的动作。
- [Polars](https://github.com/pola-rs/polars)：首屏定位与集中导航，简短示例后链接详细文档，高级安装折叠。
- [Kineto](https://github.com/pytorch/kineto)：明确 profiling 工具的职责边界和分层文档。该仓库现已归档，仅参考文档组织方式。

只参考组织方式，不复制其他项目的截图、Logo 或宣传指标。

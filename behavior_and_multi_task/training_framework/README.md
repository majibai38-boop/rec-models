# 公共训练框架

目录名使用 `training_framework`，避免遮蔽 CANN/NPU profiler 解析器使用的
顶层 `framework` 包。运行入口应从仓库根目录使用模块方式启动，不要把
`behavior_and_multi_task` 目录本身加入 `PYTHONPATH`。

该目录只负责模型无关的训练生命周期：

- `runner.py` 合并公共默认值、模型默认值和命令行参数，并接收数据依赖注入；
- `handler.py` 管理 rank-local device、DDP/TorchRec DMP、`torch.compile`、训练/验证、
  checkpoint、profiling 和精度对比；
- `config.py` 提供无第三方依赖的属性字典；
- `utils/` 保存日志和辅助实现。

数据集 adapter 由顶层 `main.py` 注入，当前实现位于
`../data_process/aliccp.py`。具体模型通过注册表提供
`model_factory(params, spec)`；框架不导入 ETA 或 ESMM。

模型接口：

```python
forward(features, mode) -> dict[str, Tensor]
loss(predictions, labels) -> scalar Tensor
```

TorchRec DMP 是该通用接口的分布式扩展：adapter 通过 `build_distributed` 返回 DMP
callable 和优化器，DMP callable 接收 `Pipelineable` batch 并返回
`(loss, output)`，以满足 TorchRec SparseDist pipeline 协议。

模型在进程根据 `LOCAL_RANK` 选定设备后才会创建。多卡时先包装 DDP，再对最终
callable 应用可选的 `torch.compile`。checkpoint 保存前会去掉 DDP/compile
wrapper，并兼容读取旧的 `module._orig_mod.` 键前缀。

`--distributed_strategy` 默认为 `auto`。普通模型按原流程使用 DDP；实现
`build_distributed(params, device, dense_dtype)` 协议的模型可选择 `torchrec_dmp`。
该分支不再执行整模 `.to()`、外层 DDP 或全参数 Adam，而由模型完成
meta EBC 分片并返回 DMP callable 与组合优化器。DMP 数据 adapter 返回实现
`Pipelineable` 的 batch，并在 DataLoader 侧构建 KJT；训练使用
`TrainPipelineSparseDist` 重叠下一批 KJT input distribution 与当前批计算；TorchRec
1.2 NPU 下的验证/测试在 `model.eval()` 模式复用同一 pipeline。Embedding shard
checkpoint 由模型 adapter 在 DMP 创建后通过 PyTorch Distributed Checkpoint
集体保存和加载。

NPU 适配版 TorchRec 1.2.0 的 FUSED EBC 仅支持 `row_wise`，因此该后端默认并强制
使用 row-wise 分片；标准 GPU TorchRec 1.2.0 仍可选择 `hybrid`、`auto` 或
`table_wise`。

`--model_type` 控制 Dense/Attention 等非 Embedding 参数及浮点 buffer 的 dtype，可选
`float32`（默认）和 `bfloat16`。`--embedding_type` 控制 `nn.Embedding` 和 TorchRec
EBC 内部 `nn.EmbeddingBag` 的参数 dtype，可选 `model`（默认，跟随 `model_type`）、
`float32` 和 `bfloat16`。类型转换发生在模型创建后、DDP 包装前，因此 DDP bucket、
梯度和优化器状态会跟随各自参数类型。加载不同 dtype 的 checkpoint 时，
`load_state_dict` 会将权重复制到当前配置的参数 dtype。

混合配置 `--model_type=float32 --embedding_type=bfloat16` 会让 Embedding 查询和池化
保持 BF16，并在进入 Dense/Attention/GRU 边界时显式转换为 FP32；这避免依赖不同设备
后端的隐式类型提升行为。当前 NPU 适配版 TorchRec 1.2.0 DMP 的 FUSED EBC 固定为
FP32，但 Dense tower 仍可通过 `--model_type` 独立选择 FP32 或 BF16。

`--load_checkpoint` 与 `--save_checkpoint` 默认均为 `false`。加载使用
`weights_only=True`，并通过临时 safe-globals 上下文允许旧 checkpoint 中的
`getattr`，不会把整个反序列化过程切换为不受限模式。

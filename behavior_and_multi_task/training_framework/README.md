# 公共训练框架

目录名使用 `training_framework`，避免遮蔽 CANN/NPU profiler 解析器使用的
顶层 `framework` 包。运行入口应从仓库根目录使用模块方式启动，不要把
`behavior_and_multi_task` 目录本身加入 `PYTHONPATH`。

该目录只负责模型无关的训练生命周期：

- `runner.py` 合并公共默认值、模型默认值和命令行参数，并接收数据依赖注入；
- `handler.py` 管理 rank-local device、DDP、`torch.compile`、训练/验证、
  checkpoint、profiling 和精度对比；
- `config.py` 提供无第三方依赖的属性字典；
- `utils/` 保存日志和辅助实现。

数据集 adapter 由顶层 `main.py` 注入，当前实现位于
`../data_process/aliccp.py`。具体模型通过注册表提供
`model_factory(params, spec)`；框架不导入 ETA 或 ESSM。

模型接口：

```python
forward(features, mode) -> dict[str, Tensor]
loss(predictions, labels) -> scalar Tensor
```

模型在进程根据 `LOCAL_RANK` 选定设备后才会创建。多卡时先包装 DDP，再对最终
callable 应用可选的 `torch.compile`。checkpoint 保存前会去掉 DDP/compile
wrapper，并兼容读取旧的 `module._orig_mod.` 键前缀。

`--load_checkpoint` 与 `--save_checkpoint` 默认均为 `false`。加载使用
`weights_only=True`，并通过临时 safe-globals 上下文允许旧 checkpoint 中的
`getattr`，不会把整个反序列化过程切换为不受限模式。

# Behavior and Multi-Task

`behavior_and_multi_task` 将行为序列模型 ETA、DIEN、特征交互模型 AutoInt 与多任务模型 ESSM 放在同一套
PyTorch 训练框架中。四个模型共用 Ali-CCP 的预处理结果、训练生命周期和命令行
参数，并分别通过独立脚本启动。

## 目录

```text
.
├── pyproject.toml             # Python 包元数据、依赖和命令行入口
├── requirements.txt           # 运行依赖清单
├── LICENSE                    # Apache-2.0
└── behavior_and_multi_task/
    ├── main.py                # 组合入口：选择模型并注入数据 adapter
    ├── __main__.py            # 支持 python -m behavior_and_multi_task
    ├── run_{eta,dien,essm,autoint}.sh
    ├── training_framework/    # 训练、评估、checkpoint、profiling
    ├── data_process/          # Ali-CCP 预处理和运行时 adapter
    ├── models/                # 模型实现和注册表
    └── tests/                 # 单元测试
```

依赖方向为：

```text
main.py
├── models/registry.py ──> models/{eta,dien,essm,autoint}.py
├── data_process/aliccp.py ──> training_framework/handler.py
└── training_framework/runner.py ──> training_framework/handler.py
```

`behavior_and_multi_task/main.py` 是 composition root。`training_framework` 不导入具体模型或数据集；模型通过
`factory(params, spec)` 延迟创建，确保每个 torchrun worker 先选定
`LOCAL_RANK` 对应设备。

公共框架目录有意命名为 `training_framework`，并通过模块入口启动，避免项目目录
遮蔽 CANN/NPU profiler 解析器依赖的顶层 `framework` 包。不要将
`behavior_and_multi_task` 目录本身加入 `PYTHONPATH`；脚本会自动将仓库根目录加入。

## 安装

建议在 Linux、Python 3.9+ 和 Bash 环境中使用。先创建虚拟环境，再安装项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

CPU/CUDA 环境请按 PyTorch 官方说明选择对应构建；昇腾 NPU 环境还需要安装与当前
PyTorch、CANN 版本匹配的 `torch-npu`，本项目不固定设备相关版本。仅使用遗留的
`models/layers/self_gru.py` 时，可安装可选依赖：

```bash
python -m pip install -e ".[legacy]"
```

## 1. 生成同一份 Ali-CCP 数据

将以下原始文件放到 `behavior_and_multi_task/data_process/`：

```text
common_features_test.csv
common_features_train.csv
sample_skeleton_test.csv
sample_skeleton_train.csv
```

然后执行：

```bash
cd behavior_and_multi_task/data_process
bash run.sh
```

输出位于 `behavior_and_multi_task/data_process/aliccp_out/`。ETA、DIEN、ESSM 和 AutoInt 都读取该目录。真实的
train/eval/test 必须使用这里生成的 `spec.json`；内置 schema 只供
`test_qps` 随机输入使用。

序列字段使用 `-1` 表示 padding，`0` 是可训练的 OOV ID。点击标签为
`labels["y"]`，点击且转化标签为 `labels["z"]`。

预处理参数可通过环境变量覆盖：

```bash
MAX_LENGTH=50 NUM_OF_PROC=10 CHUNK_SIZE_MB=128 PADDING=true bash run.sh
```

## 2. 分开运行 ETA、DIEN、ESSM 和 AutoInt

四个脚本都将仓库根目录作为包根目录，通过 `torchrun --module` 调用
同一个 `behavior_and_multi_task.main`，但显式选择不同模型，并使用
不同的默认端口：

```bash
# ETA，默认端口 12345
DEVICE=npu DEVICE_ID=0 bash behavior_and_multi_task/run_eta.sh

# ESSM，默认端口 12346
DEVICE=npu DEVICE_ID=0 bash behavior_and_multi_task/run_essm.sh

# DIEN，默认端口 12347
DEVICE=npu DEVICE_ID=0 bash behavior_and_multi_task/run_dien.sh

# AutoInt，默认端口 12348
DEVICE=npu DEVICE_ID=0 bash behavior_and_multi_task/run_autoint.sh
```

CUDA 示例：

```bash
DEVICE=cuda DEVICE_ID=0,1 bash behavior_and_multi_task/run_eta.sh
DEVICE=cuda DEVICE_ID=0,1 bash behavior_and_multi_task/run_essm.sh
DEVICE=cuda DEVICE_ID=0,1 bash behavior_and_multi_task/run_dien.sh
DEVICE=cuda DEVICE_ID=0,1 bash behavior_and_multi_task/run_autoint.sh
```

CPU 调试也通过脚本运行：

```bash
DEVICE=cpu DEVICE_ID=0 BATCH_SIZE=4 TRAIN_STOP_STEP=1 VAL_STOP_STEP=1 \
  bash behavior_and_multi_task/run_essm.sh \
  --hidden_dims=8,4 --embedding_size=4
```

DIEN 可通过独立脚本启动，也可以直接调用统一模块入口：

```bash
PYTHONPATH=. torchrun --standalone --nproc_per_node=1 --module \
  behavior_and_multi_task.main --model=dien --mode=test_qps \
  --device=cpu --compile=false --dnn_hidden_size=8,4 --att_hidden_size=8,4

# 独立脚本/NPU 场景
DEVICE=npu DEVICE_ID=0 \
  bash behavior_and_multi_task/run_dien.sh
```

DIEN 使用 `206/207/216` 目标特征和对应的三组行为序列；`-1` 按 padding 处理，
ID `0` 仍是有效 OOV。为对齐 DeepCTR，局部注意力分数默认使用 softmax 归一化，
即 `--dien_attention_weight_normalization=true`；如需消融实验可显式传入 `false`。

本实现仍保留 Ali-CCP benchmark 的数据与训练配置：数据集没有负历史输入，因此不启用
负采样辅助损失；CTR 继续使用来源 benchmark 的正样本加权损失（默认权重为
`(1 - 0.14) / 0.14`）；默认 prediction DNN 和 attention DNN 宽度也继续采用当前
benchmark 的规模，而不是重置为 DeepCTR 示例中的较小规模。这些参数均可通过
`--dien_positive_class_weight`、`--dnn_hidden_size` 和 `--att_hidden_size` 覆盖。
`run_dien.sh` 还支持通过 `GRU_TYPE`、`DIEN_DROPOUT`、
`DIEN_ATTENTION_WEIGHT_NORMALIZATION`、`DIEN_POSITIVE_CLASS_WEIGHT`、
`DNN_HIDDEN_SIZE` 和 `ATT_HIDDEN_SIZE` 环境变量配置这些参数；脚本末尾的命令行
参数优先级更高，但模型名固定为 `dien`。

AutoInt 将每个 Ali-CCP one-hot、multi-hot 和 special 字段池化为一个 field token，
再沿 field 维执行多头自注意力；`-1` 不参与池化，ID `0` 仍是有效 OOV。默认采用
benchmark 的 8 层、8 头 attention-only 配置，同时保留 DeepCTR 的一阶线性分支、
残差投影和预测偏置。可通过 `AUTOINT_DNN_HIDDEN_UNITS=256,128` 开启并行 DNN
分支，也可用 `ATTENTION_LAYERS`、`NUM_HEADS`、`AUTOINT_RESIDUAL`、
`AUTOINT_SCALING`、`AUTOINT_DROPOUT` 和 `AUTOINT_POSITIVE_CLASS_WEIGHT` 调整。
`EMBEDDING_SIZE` 必须能被 `NUM_HEADS` 整除，脚本末尾模型名固定为 `autoint`。

常用环境变量包括 `PREPROCESSED_DATASET`、`MODEL_DIR`、
`LOAD_CHECKPOINT`、`SAVE_CHECKPOINT`、`REPORT_DIR`、
`PROFILING_PATH`、`MODE`、`BATCH_SIZE`、`NUM_EPOCHS`、
`TRAIN_STOP_STEP` 和 `VAL_STOP_STEP`。脚本末尾的参数会继续传给模块入口。

checkpoint 加载和保存默认都关闭；仅传入 `MODEL_DIR` 不会读写权重，也不会创建
checkpoint 目录。需要时分别显式开启：

```bash
# 从头训练并保存验证集最优权重
bash behavior_and_multi_task/run_eta.sh \
  --save_checkpoint=true

# 加载已有权重但不继续保存
bash behavior_and_multi_task/run_eta.sh \
  --load_checkpoint=true

# 断点续训：加载并继续保存
bash behavior_and_multi_task/run_essm.sh \
  --load_checkpoint=true --save_checkpoint=true
```

`--load_weights`、`--save_weights` 是对应参数的兼容别名。加载始终使用
`weights_only=True`，并在受限安全上下文中只额外允许旧 checkpoint 可能引用的
内置 `getattr`；仍应只加载可信来源的 checkpoint。

查看统一入口参数：

```bash
python -m behavior_and_multi_task.main --model=eta --help
python -m behavior_and_multi_task.main --model=dien --help
python -m behavior_and_multi_task.main --model=essm --help
python -m behavior_and_multi_task.main --model=autoint --help
```

ETA legacy 版本保留了 LSH 修复前的行为，可复用现有 ETA 脚本切换，无需新增脚本：

```bash
MODEL=eta_legacy bash behavior_and_multi_task/run_eta.sh
# 或使用命令行参数覆盖
bash behavior_and_multi_task/run_eta.sh --model=eta_legacy
```

`eta` 使用修复后的最小汉明距离候选，`eta_legacy` 则有意保留原先的最大汉明距离
候选及旧 padding/hash Tensor 行为，仅用于结果回归和差异对比。

也可以在仓库根目录使用包入口查看参数：

```bash
python -m behavior_and_multi_task --model=essm --help
```

## 3. 模型契约

注册到框架的模型需要实现：

```python
forward(features: dict[str, Tensor], mode: str) -> dict[str, Tensor]
loss(predictions: dict[str, Tensor], labels: dict[str, Tensor]) -> Tensor
```

ETA、DIEN 和 AutoInt 输出 `ctr`。ESSM 输出 `ctr`、`cvr`、`ctcvr`，并满足
`ctcvr = ctr * cvr`；ESSM 同时监督点击和点击且转化两个目标。

新增模型时：

1. 在 `behavior_and_multi_task/models/` 新增模型和 `build_<name>(params, spec)`；
2. 在 `behavior_and_multi_task/models/registry.py` 注册 factory、默认参数和可选参数钩子；
3. 按需增加一个显式传递 `--model=<name>` 的运行脚本。

无需修改 `behavior_and_multi_task/training_framework/handler.py`。

## 4. 验证

从仓库根目录执行：

```bash
python -m unittest discover \
  -s behavior_and_multi_task/tests -v
```

测试覆盖 Ali-CCP 跨分片组 batch、ETA/DIEN/ESSM/AutoInt 公共接口、ESSM 概率约束、
padding/OOV 语义、loss backward 和真实模式下的 schema 校验。

## 许可证

项目采用 [Apache License 2.0](LICENSE)。

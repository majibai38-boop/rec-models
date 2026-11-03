# 内置设备来源与口径

核对日期：2026-10-26。`presets.py` 向已有数据库增量安装，不覆盖自定义版本。均为单设备理论峰值，不能当作模型实测性能。

| 设备 | 稠密矩阵FP16/BF16 TFLOPS | TF32/HF32档位 TFLOPS | 非矩阵FP32 TFLOPS | 显存带宽 GB/s | 显存 GB |
|---|---:|---:|---:|---:|---:|
| A100 40GB PCIe | 312 | 156 | 19.5 | 1555 | 40 |
| A100 40GB SXM | 312 | 156 | 19.5 | 1555 | 40 |
| A100 80GB PCIe | 312 | 156 | 19.5 | 1935 | 80 |
| A100 80GB SXM | 312 | 156 | 19.5 | 2039 | 80 |
| 950DT 28 Cube / 56 Vector | 378 | 189 | 23 | 4000 | 未确定 |
| 950DT 32 Cube / 64 Vector | 432 | 216 | 27 | 4000 | 未确定 |
| 950DT 36 Cube / 72 Vector | 486 | 243 | 30 | 4000 | 未确定 |

## NVIDIA A100

- [官方A100产品页](https://www.nvidia.com/en-us/data-center/a100/)：80GB PCIe/SXM、HBM带宽及稠密/稀疏算力。
- [官方A100数据表](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/a100-80gb-datasheet-update-nvidia-us-1521051-r2-web.pdf)：40GB PCIe/SXM参数。
- [NVIDIA Ampere架构说明](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)：非Tensor FP16为78 TFLOPS、BF16为39 TFLOPS、FP32为19.5 TFLOPS。这里的指令峰值不是所有Vector算子的有效吞吐。

采用312/156稠密矩阵值，而非624/312稀疏值。适用于完整GPU，不适用于MIG切片。FP32输入不意味着启用了TF32，实际数学模式及跨模式精度仍需确认。

原始工作簿A100列未明确40/80GB和PCIe/SXM，故不将其中CPU/内存配置自动绑定某一SKU。用户可新建平台版本补齐。

## 昇腾950DT

- [昇腾950 NPU架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/昇腾950%20NPU架构白皮书.pdf)，表3-1（PDF第13–14页）：核数及Cube、Vector独立算力、带宽。
- Vector FP16/BF16分别为47、54、60 TFLOPS；FP32分别为23、27、30 TFLOPS。不要把Cube+Vector合计425/486/547作为矩阵算力。
- 带宽4 TB/s按十进制转换为4000 GB/s，不是片间互联带宽。
- 白皮书列出144/96GB容量，但未给出各Cube档位一一对应关系，因此不自动填写容量。
- 白皮书称TF32，工具沿用现有`matrix_hf32`字段。需要结合CANN实际数学模式及精度验证，不能推断默认就启用了HF32。

950DT Server 28 Cube参考平台的主机信息来自用户工作簿《推荐硬件对比.xlsx》Sheet1!F2:F8：2颗Kunpeng 950 7592C、每颗64核、2300MHz、24条64GB内存、额定4800MT/s。原表378 TFLOPS档位对应28 Cube是参考配置，不代表所有950DT Server。`card_count=1`是本工具的单设备建模范围，不代表整机实际只装一颗NPU。原表Vector FP32为23.5，新增预设采用官方整数23并保留差异说明。

950DT 32/36 Cube的主机参数、实际内存速率、软件和拓扑继续保持未知；原工作簿没有被修改。

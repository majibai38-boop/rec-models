# 类型＋调用方聚合对照表

**选对流程。** 要判断“这次调用该不该关联”、修订决策或统计 Comparable，走 [严格调用评审](call-alignment-tables.md)。要回答“这两次采集里，哪些类型、调用方及组件占时不同”，走本页。跨批时严格表全部 `review` 可能是正确结果；本表不会让它们变成 verified，也不写回覆盖率。只有严格表把 GPU 显存活动算入设备 TOTAL；两种表的 TOTAL 比值不能直接替换。

## 建表前的门禁

1. 先对两侧配对的 Step result 和 all-steps result 分别查看 `summary`、运行 `validate`。二者须来自同一组原始输入、同一侧 PID。基线也独立满足这些条件。不要为了得到表格重跑 `compare` 或扩大采集范围。
2. 核对模型/版本、Step、batch size 与实际数据批、精度（如 FP16/FP32）、并行/分片、rank 数与设备。将已知差异和未知项写进 `--caliber-note`。同 Step 号不代表等工作量。不能确认时，只描述本次采集的累计时间，不作硬件优劣或回归归因。
3. 查 `whole_call_mapping.csv` 的 `Issues`：按 `Whole OP` 分别清点 `冲突：…input_dtypes` 和 `缺失：…input_dtypes`，不要用疑似 shape 冲突代替 dtype 冲突。报告给出的是该文件覆盖的调用记录数，不是全 trace 的冲突率；零也不是等价证明。
4. **验证优先**：在换模型家族、改配置或接受新配对的结论前，先用已知正确的旧配对复建到新目录。与已审核 Markdown 的机械表格行按字节 diff；只有事先说明的策展文案改动可豁免。至少核对 TOTAL、GEMM/bmm、index_add、每条组件链及显存单列。数字变化先解释、再交付。若缺旧输入或旧表，须标记“未完成该真实配对复验”，不可把 issue 中的数字当实测结果。仓库的合成黄金样例可先做回归：`python -m unittest profiling_tools.tests.test_build_op_type_comparison`。

```text
python <skill>/scripts/build_op_type_comparison.py --result-step10 <selected-result> --result-all <all-result> --output-md <new>/op_type_comparison.md --output-xlsx <new>/op_type_comparison.xlsx --title "训练配对 A" --npu-label "Ascend" --gpu-label "GPU" --caliber-note "同模型、不同数据批；rank 与精度……"
```

`--result-step10` 是兼容参数名，不限于第 10 步；报告记录真实 Step。复验旧配对时可加 `--reference-md <同格式已审核旧表>`，机械行、数值或次序不符即退出，不写文件；不同布局的历史表需先人工逐列核对。对比上一版时一起给 `--baseline-step10`、`--baseline-all`，可加 `--baseline-label`。不提供基线则省略 §4/§5。只接受已有、已校验且有指纹的引擎结果；脚本校验所读 CSV 的摘要，不打开 trace，不覆盖已有输出。标准库可生成 Markdown；安装了 openpyxl 的解释器同时生成 Excel，否则明确提示跳过，不自动安装。

## 如何复核一行

| 位置 | 数据来源与解释 |
| --- | --- |
| §1 所选 Step、§2 整体 | `npu_kernel_gpu_mapping.csv` 给出类型、raw、CPU owner；与 `npu_kernel_details.csv` 同行核对类型、名称或 stream/task 身份，再读 PMU。两表行序相同；仅一侧出现的类型在另一表留空。 |
| NPU 去扰 | 每个 kernel 取 `max(aiv_time, aicore_time)`，二者实测均为 0 时回退明细 Duration（含 AICPU）；缺测不是 0，该组去扰留空。`aiv_vec_time` 不是 `aiv_time` 的别名。去扰是活动时间代理，raw−busy 只是排队/干扰的线索，不能证明抢占或预测可省 wall time。 |
| GPU 类型列 | 默认取 `npu_gpu_op_statistic.csv` 的 `Row Kind=npu_type`、`GPU Total Time(us)`；显式类型规则可改用 ATen 调用方或全名 GPU 组件。无独立对应不补零；路线不同可给参考数但不着色。 |
| GEMM 家族 | NPU 取 aclnnMm/Addmm/Matmul 前缀的所有 kernel（可含非 MatMul 辅助类型）；GPU 取 `aten::mm/addmm/bmm`。保留单类型 NPU 数，但不在多个 MatMul 类型上反复分摊家族 GPU 时间。 |
| TOTAL | `summary.json → gpu_accounting` 中 `kernel.total_time_us / npu_total_time_us`；去扰分母是全部 detail-backed kernel 的 busy 之和。`memory.total_time_us` 单列。GPU 类型列混源、部分引擎归属可含显存，**列和不等于 TOTAL**。累计设备耗时可跨 stream 重叠，不是 Step 经过时间。 |
| §3 调用方 | 仅用 `aten_op_statistic.csv` 两侧的 Raw Kernel Count 与 Total Time；不用 `matches.csv` 求和。index_add/index_add_ 合并；bmm 只比较 NPU matmul + matmul_backward 与 GPU bmm；不取整个 GEMM 作分子。t/view/transpose 展示噪声归属但不算比值。 |
| §3 链段与显存 | 对 `gpu_kerenel_details.csv`（历史拼写）按完整名称子串归族，不截断模板名。排序看 radix+scan+select，jagged 看三个 fbgemm 组件，哈希看 cuco，自定义 ascendc 仅域级参考；链段不可叠加。`gpu_memory_activity.csv` 按原名拆分 DtoD/HtoD 等，不把 GPU 拷贝自动归到 FSDP。 |
| §4–§6 | 同时换 NPU/GPU 配对是跨口径同构对比；单独看 NPU 两次 run 的类型变化。结论区列 raw/去扰、调用方差额、拷贝路径和排队线索，不能从中推出等价工作负载的加速比。 |

CSV 的 µs 在载入时只除以 1000 一次。Excel 的 GPU/NPU 数字大于 1 用绿、小于 1 用橙；空值、等于 1、路线不同不着色。所有模型家族相关的别名、组件模式、链段及回归关注类型集中在脚本顶部 `FAMILY CONFIG`；迁移家族先核对调用关系，不要修改引擎 matcher 或用提高覆盖率作为验证标准。

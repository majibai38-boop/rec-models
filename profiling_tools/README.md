# GPU/NPU Profiling 对比工具

浏览器上传、任务队列、可视化和报告下载，见 [单机 Web 工作台部署指南](../profiling_web/README.md)。原 CLI 和 Skill 接口保持不变。

Excel/Web 的主要页签布局、整体比较口径和下钻方式见 [整体调用视图说明](docs/whole-call-views.md)。整体视图不改变类型表分配或 TOTAL，不可跨视图相加。

Cube/Vector 总览：`aicore_time(us)=0` 的 NPU kernel 归为 Vector，`>0` 归为 Cube，缺失/无效为未知；单独汇总 `aicore_time(us)`、`aiv_vec_time(us)` 和有效记录数。各类耗时使用 kernel `Duration(us)`，占比分母为各端全部计算 kernel。GPU 继承已映射 NPU 分类并按事件去重，跨类组记为混合、未映射单列，GPU 内存活动独立保留；GPU 类别仅是参考，非硬件利用率。输出包括 `cube_vector_statistic.csv`、`summary.json/core_breakdown` 和 Excel「Cube-Vector总览」页签。

该工具从 GPU 与 NPU 的 PyTorch Profiler trace 中提取 `aten::` 调用，先规范化 shape、dtype 和 Module，再使用语义锚点与局部执行顺序进行对齐，并把每个 aten 实例直接下发的 device kernel 作为一个 **kernel bundle** 进行比较。NPU 的 `kernel_details.csv` 用于为匹配到的 GPU bundle 补充 NPU `Type`、`Core Type`、shape 和 dtype。

工具不会假设 GPU 与 NPU 的物理 kernel 数量相同：数量相等时会输出顺序映射；数量不等但 bundle 中存在与 aten 同类型的唯一主 kernel 时，会输出 `semantic_primary` 映射并保留完整 bundle；其余情况保留 bundle 级映射，避免伪造逐 kernel 的一一关系。

## 环境

- Python 3.9+
- 运行比较功能只使用 Python 标准库，不需要额外安装依赖。
- 命令应在仓库根目录执行。

模块命令从仓库根目录使用；也可独立安装本目录（`python -m pip install ./profiling_tools`），或使用下述便携 zipapp，从任意目录运行。独立包名为 `gpu-npu-profiling`，不依赖训练框架及其 torch/pandas 等包，原仓库的打包配置不变。

## Skill 与稳定命令接口

推荐新流程为 `inspect → compare → validate → summary/query`。命令只进行本地文件分析，不上传 trace、不安装依赖、不提交代码。各接口支持 Python 3.9+；本次验证环境为 Windows 的 bundled Python，其他平台仍建议运行测试。

```bash
# 只读预检查：列出 PID、step、设备事件候选、元数据及警告
python -m profiling_tools inspect --gpu-trace gpu.json --npu-trace trace_view.json --npu-kernel-details kernel_details.csv

# 严格比较：必须明确每端 step，或明确选择 --all-steps
python -m profiling_tools compare --gpu-trace gpu.json --npu-trace trace_view.json --npu-kernel-details kernel_details.csv --gpu-step 15 --npu-step 15 --output-dir new_result --json

# 不读取原始 trace 的结果复用
python -m profiling_tools validate --result-dir new_result
python -m profiling_tools summary --result-dir new_result
python -m profiling_tools query --result-dir new_result --op-type Slice --limit 5
python -m profiling_tools query --result-dir new_result --npu-kernel-index 42
python -m profiling_tools query --result-dir new_result --gpu-kernel-index 17
python -m profiling_tools query --result-dir new_result --memory-activity-index 3
python -m profiling_tools query --result-dir new_result --side gpu --cpu-op-index 99
```

- `inspect` 默认不写文件。`compare` 在多 PID 未选择、step 范围未确认时返回 `needs_selection`，不会静默选最大的 PID 或全部 step。即使只有一个 step 标记，标记外也可能有 warmup；无标记时应确认完整采集范围后使用 `--all-steps`。相同 step 编号不证明相同输入或工作量。
- 多个 CPU PID 时增加 `--gpu-pid` / `--npu-pid`。CPU PID 选择不是设备过滤，多个 device-event PID 会提示人工核对，当前不自动推断 rank/device 对应关系。
- `inspect` 的 kernel 数是关联前、时间窗口内的候选数，不等于最终 kernel 数。元数据“存在”不代表完整或正确。缺失 detail 时间可以从匹配 trace 恢复，缺失输入/不支持格式不会自动安装依赖或生成伪结果。
- 输出目录必须是新的或空的，避免覆盖旧结果或混入旧页签。原来不带子命令的调用保留自动范围选择，但新输出会记录警告；skill 使用严格入口。
- 退出码：`0` 操作成功（可能仍有覆盖缺口），`2` 输入/命令错误，`3` 需要选择范围，`4` 结果校验失败。
- `query` 按类型、NPU/GPU kernel ID、内存活动 ID、CPU op ID 查询；可加 `--device-id` / `--core-type`。条件按 AND 组合，每个结果集合均有 `total/offset/limit/has_more/items`，使用 `--offset` 翻页，`--limit` 范围 1..200。GPU CPU 查询包含对应逻辑匹配及该 CPU 直接拥有的设备事件。编号只在特定运行与命名空间内有效。
- `summary.overall_statistics` 和 `query.type_statistics` 提供有类型的数量、耗时、两个性能比及拆分覆盖率；`query.reason_groups` 提供原因聚合和下一步排查。后两项是完整 Device/Type/Core 的上下文，不是单个 kernel 查询后的子集小计。其余 CSV 来源记录保留原始列名/字符串，布尔值须显式解析。
- 新的 `matches.csv` 含 GPU/NPU CPU op 索引。旧结果仍可做类型/kernel 查询和数值校验，但缺少这些索引时不能可靠地做 GPU CPU-index 查询。

### 运行记录与校验

`summary.json` 增加 schema、运行 ID、工具/规则/源码版本、状态和警告。`run_manifest.json` 记录输入 SHA-256、大小/mtime、所有参数和生成物哈希；不复制原始 trace。`input_inspection.json` 保存预检查结果，`validation.json` 保存完成时校验。状态分为 `complete`、`complete_with_gaps`、`invalid`，低覆盖率不是执行失败。

`validate` 会重新读取事件清单、映射、统计与 manifest，而不是信任保存的 PASS：检查 NPU detail 范围、GPU 计算/内存独立编号、映射与余项守恒、重复计时、总体比值分母、诊断守恒与文件哈希。结果目录不被修改；旧结果无 manifest 时明确提示无法验证来源。数值校验通过不等于每个语义映射已证明正确。

NPU 总体目前仅包含 **kernel_details-backed kernel**，不包括 trace-only 任务；GPU 总体包括选中计算和内存活动。验证报告列出 NPU 被排除的数量和耗时，这一范围差异必须保留在总体比值解释中。

### 构建与使用 skill

仓库中的 skill 源码位于 `skills/gpu-npu-profiling`，薄入口调用唯一一份核心实现。构建时自动打包核心，不手工复制或维护第二套算法：

```bash
python -m profiling_tools.build_bundle --output-dir dist/profiling-skill-0.2.0
python /absolute/path/dist/profiling-skill-0.2.0/profiling_tools.pyz --version
python /absolute/path/dist/profiling-skill-0.2.0/gpu-npu-profiling/scripts/run_compare.py summary --result-dir /absolute/path/results
```

构建目录必须不存在。产物包含独立 `.pyz`、可分发 skill 文件夹和 `bundle.json`，不会包含 example、测试、trace 或训练依赖。将构建后的 `gpu-npu-profiling` 文件夹放入自己的 skill 目录后，可用 `$gpu-npu-profiling` 发起请求。不要单独复制仓库内未打包的 skill 文件夹到其他机器：该源版本依赖相邻仓库；分发版本内置固定版本 zipapp。

示例请求：`使用 $gpu-npu-profiling 分析这个结果目录，解释 Slice 的低覆盖率，不重跑 trace。`

核心测试：`python -m unittest discover -s profiling_tools/tests`。skill 的行为验证用例在其 `references/evaluation.md`。缓存、流式读取、独立设备过滤和自动实验验证仍属于后续能力，不属于本版封装。

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

Ascend flow 终点与硬件任务时间戳偏差较大时，可调整回退容差：

```bash
--flow-tolerance-us 2000
```

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
| `coverage_diagnostics.md` | 通用低覆盖率排查入口：按未比较 NPU 时间排序，解释主要缺口、给出证据样本与下一步排查建议，不限制 kernel 类型 |
| `coverage_diagnostics.json` | 同一诊断的完整机器可读证据：所有命中类型、每个未比较 kernel 的唯一主要原因、辅助观察、CPU/GPU 候选、整体报告引用与建议 |
| `matches.csv` | 每个 GPU/NPU aten 匹配及两边 kernel bundle 的时间比较 |
| `gpu_kernel_mapping.csv` | GPU kernel 到 NPU Type/shape 的映射；明确标记 `one_to_one`、`ordered_equal_count`、`semantic_primary` 或 `bundle`，并保留 NPU bundle 的完整类型和名称 |
| `npu_kernel_gpu_mapping.csv` | NPU-first 明细：每个 NPU kernel 经由所属 CPU op 和 CPU op 匹配结果映射到 GPU kernel/bundle；记录 GPU kernel 的实际子 op、来源以及三种覆盖状态 |
| `npu_gpu_op_statistic.csv` | NPU-first 汇总：在 `OP Type` 左侧增加 `FUSED_OP_NPU_TYPE` / `FUSED_OP_GPU_TYPE`；分别按 kernel 数量和 NPU 时间统计 Direct、Comparable、Accounted 覆盖率。同一融合组连续排列；CSV 中组名逐行重复。普通非融合算子两列均回填原 `OP Type` |
| `npu_gpu_op_statistic.xlsx` | 五个页签：`类型对比`、`整体对比`、`覆盖率诊断`、`调用明细`、`耗时分布`。表格展示通用排查结论，Div/Sub 具体路径保留在通用排查报告 |
| `whole_call_statistic.csv` / `whole_call_mapping.csv` | 整体调用分组汇总 / 逐调用证据，包含可比状态、两端完整设备活动、分组和调用编号；独立视图，不与类型表相加 |
| `gpu_memory_activity.csv` | GPU memcpy/memset 设备事件清单：独立编号、方向、字节数、设备时间、correlation 和 CPU owner；不计入计算 kernel 数量 |
| `cpu_bundle_mapping.csv` | Stack、标量准备＋IndexPut 的逐调用整体比较；包含两侧 kernel ID、GPU memory activity ID 和验证依据，可独立审计 |
| `cpu_bundle_statistic.csv` | 上述 CPU 整体调用汇总；这是独立比较视图，不可与逐类型、FA 或 region 表直接相加 |
| `gpu_unmapped_activity.csv` | 未分配到 NPU 类型的 GPU 计算 kernel 与内存活动：独立编号、名称、耗时、CPU owner、关联方式、状态及已有整体比较引用 |
| `fused_op_mapping.csv` | 跨后端融合算子逐调用明细；一个 GPU 自定义 op bundle 可包含多个不同类型的 NPU 主/辅助 kernel，并分别输出 sum、busy、span 和整体性能比 |
| `fused_op_statistic.csv` | 跨后端融合算子汇总；按融合算子统计调用数、两侧 kernel 数量、主/辅助 NPU 时间和整体 `GPU/NPU` 性能比 |
| `aten_op_statistic.csv` | 参考 `op_statistic.csv`，按规范化后的 aten 类型汇总调用数、kernel 数和时间；NPU kernel 进一步拆为同类型主 kernel 与其他辅助 kernel |
| `kernel_type_statistic.csv` | 按 NPU Type/Core Type 汇总匹配后的 GPU/NPU bundle |
| `module_statistic.csv` | 按规范化 Module key 汇总；busy time 对模块内 kernel 区间统一求并集 |
| `region_matches.csv` | 两个可信语义 kernel 锚点之间的 GPU/NPU 粗粒度算子区域，用于多对多整体时间比较 |
| `kernel_inventory_statistic.csv` | GPU/NPU 原始 kernel 类型、数量、耗时分位数和关联数量 |
| `kernel_link_diagnostic.csv` | 每个 kernel 的 aten 归属、关联方法、回退层级与时间偏差 |
| `unmatched_gpu.csv` | 未匹配的 GPU aten 及其 kernel bundle |
| `unmatched_npu.csv` | 未匹配的 NPU aten 及其 kernel bundle |

### 统计工作簿页签

打开 `npu_gpu_op_statistic.xlsx` 后，先从 `类型对比` 按 NPU 时间占比识别热点，再进入 `整体对比` 查看完整调用，结合 Comparable/Accounted 时间覆盖率判断性能比是否有代表性。**普通 NPU 类型行**的两列比值分别是 `Comparable Ratio(GPU/NPU) = GPU Total Time / Comparable NPU Time` 和 `Total Time Ratio(GPU/NPU) = GPU Total Time / NPU Total Time`。这里的 `GPU Total Time` 只累计分配到该类型的 GPU kernel 与已映射内存活动；覆盖率不足时，第二列并不代表该类型的完整 GPU/NPU 总耗时比。没有可比 kernel 的类型行在展示中留空比值，原 CSV 数值保持兼容。比值大于 1 表示分子所用的 GPU 时间大于相应 NPU 时间。两列在 XLSX 中按原始数值着色：`>= 2.0` 深绿、`1.0 <= x < 2.0` 浅绿、`< 1.0` 黄色；空值不着色。

- `整体对比`：按逻辑调用和两端实现组合分组，展示可比调用的完整设备时间及比值。仅在完成两端同名调用全量盘点且组合一致时展示 Total Time Ratio；否则留空并注明原因。可点击分组进入调用明细。
- `类型对比`：融合类型、原始 `OP Type`、两端 kernel 数量、NPU 总耗时与占比、两种时间覆盖率、可比 NPU 耗时、GPU 总耗时及上述两种比值。新增整体比较状态与跳转入口，保留通用 `排查结论`。
- `覆盖率诊断`：Direct、Comparable、Accounted 的 kernel 数量/时间覆盖率、未映射数量、`Mapping Mode Counts`，以及通用 `排查结论` / `排查建议`。未触发阈值与全部纳入比较明确区分；结论不包含 Div/Sub 专项路径。
- `耗时分布`：两边 kernel 数量、匹配的 CPU 调用数、NPU/GPU 单 kernel 的 Min/Avg/Max、可比范围的 GPU-NPU 时间差。
- `调用明细`：逐调用展示 CPU 对齐状态、kernel 比较状态、两端 kernel/内存活动和证据。元数据不足的候选保留观察耗时，但不输出性能比。包含 FlashAttentionFunc、已有 CPU bundle 和满足证据要求的通用调用；Div/Sub 精度路径只通过排查报告提供。

`类型对比`、`覆盖率诊断`、`耗时分布` 和 `npu_gpu_op_statistic.csv` 最后增加以下行（CSV 用 `Row Kind` 区分）：

- `GPU_UNMAPPED_KERNELS` / `gpu_unmapped_kernel`：**没有分配到任何 NPU 类型**的 GPU kernel 合计，始终输出，包括全量映射时的真实 0。即使某个 kernel 已在整体算子中比较，仍属于逐类型表的余项，结论会单独说明已有整体比较的数量。
- `GPU_UNMAPPED_MEMORY` / `gpu_unmapped_memory`：未分配的 GPU memcpy/memset，存在时单独输出，不计入 kernel 数量。
- `TOTAL` / `total`：所选 step/PID 范围的总体，所有 GPU 计算和内存事件各计一次。`Total Time Ratio = 全量 GPU device 活动时间 / 关联 kernel_details 的 NPU kernel 时间`，包含未映射工作，但不包含未关联 detail 的 NPU trace-only 任务（校验会提示其数量和时间）。`Comparable Ratio` 仍为**去重后的可比 GPU 子集时间 / 对应可比 NPU 时间**，不能用 TOTAL 的全量 GPU 时间作为其分子。分子/分母审计值见 `summary.json -> gpu_accounting`，两边总体采集范围是否等价仍需确认。

未映射行 NPU 数量/时间为 0 仅表示无 NPU 类型归属，不表示 NPU 不需要这项工作；覆盖率、性能比留空。TOTAL 是独立汇总，不是额外工作，**不要再与明细行相加**。若 GPU 事件被多个 NPU 类型引用，表格与报告会警告类型行不可直接相加，总体仍按事件编号去重。不自动推断 GPU/NPU 多设备配对，汇总行的 Device_id 为 `ALL_SELECTED`。

总体时间是 device duration 的和（含并发流），不是 step 墙钟时间，也不证明两端工作完全等价。若只关注计算成本，查看 `耗时分布` 中 `GPU Kernel Time` 与 `GPU Memory Time` 分项。整体算子、FA、精度路径都是其他分组视图，不追加进 TOTAL。

### 通用低覆盖率诊断（推荐入口）

普通比较命令会同时生成 `coverage_diagnostics.md` 和 `coverage_diagnostics.json`。表格呈现通用结论，Markdown 提供排查优先级、建议、GPU 未映射工作与 Div/Sub 专项路径；通过 `NPU Kernel Index` 或 `Precision Path ID` 在 JSON 中查看完整证据。原类型映射与覆盖率不变，不再输出两个精度路径页签或 `precision_path_*.csv`。历史输出目录不删除旧文件，建议使用新的输出目录，实际本次文件清单以 `summary.json -> reports` 为准。

```bash
python -m profiling_tools --gpu-trace profiling_tools/example/demo1/gpu/trace.json --npu-trace profiling_tools/example/demo1/npu/trace_view.json --npu-kernel-details profiling_tools/example/demo1/npu/kernel_details.csv --gpu-step 15 --npu-step 15 --diagnostic-coverage-threshold 90 --output-dir profiling_tools/example/demo1/result_npu_first_v17
```

- `--diagnostic-coverage-threshold` 默认 `90`，范围 `0..100`。按 `Device_id + OP Type + Core Type` 分组，**Comparable 数量覆盖率或时间覆盖率任一严格低于阈值**即命中。设置 `100` 检查所有有未比较 kernel 的类型；`0` 不选择任何类型。零总时长时，时间覆盖率为不可用，仅检查数量覆盖率。
- 不限制算子名称、ACLNN API 或 NPU 时间占比。命中类型中的**全部未比较 kernel**都会得到诊断，低耗时类型也不会被过滤。优先级按未比较 NPU 时间降序，不按推测的“可优化收益”排序。
- 判断来自映射状态、CPU owner/匹配、元数据、父子 scope、GPU 计算/内存活动及已有 GPU 时间使用情况。API 名称只用于呈现与细分，不作为进入诊断的白名单。

诊断会区分：

| 情况 | 排查方向 |
| --- | --- |
| NPU owner 缺失、弱关联或 API/CPU 冲突 | 检查 Task/Stream、flow、队列、原始 owner 和采集范围 |
| CPU 无匹配、不唯一、置信度低或元数据不同 | 检查模块、父调用、局部顺序、overload、shape/dtype 规范化和真实输入 |
| GPU 仅有 memcpy/memset、无设备事件或工作在子调用中 | 检查 device 活动归属、view/空操作、融合以及父子调用语义 |
| 辅助工作伴随 dtype、shape、format 变化 | 检查类型提升、布局/广播/切片及中间张量物化，不依赖 kernel 必须叫 Cast/Transpose |
| GPU 工作已被使用、多对多 bundle 或已有整体比较 | 检查分组粒度和共享关系，优先做去重的完整调用比较，不重复分摊 GPU 时间 |
| 证据不足或未识别的新模式 | 保留为 `unclassified` 等记录，明确待补充的证据，不从缺口统计中删除 |

每个未比较 kernel **只计入一个主要原因**；dtype/shape 等辅助观察可能同时存在，不能再相加。每类的原因数量/耗时之和等于该类未比较数量/耗时。`Accounted=True` 不代表可以逐 kernel 比较，也不会让诊断忽略它。

元数据判断会区分数据张量与额外参数：例如 `[FP16, INT64, INT64] -> [FP16]` 可能是数据、索引和轴参数，不会因此标记为 dtype 转换。只有单输入/单输出 dtype 不同才产生 `dtypes_change` 线索；多输入混合类型单独记录。多输入的 shape/format 观察仅在首输入与单输出 dtype 相同时比较，并注明 `first_input_output_candidate`，不当作已证明的数据依赖。

精度路径、Stack/标量准备等 CPU 整体调用、FA 等融合调用是可选的专项证据提供方。已有整体比较时，报告引用原始记录并说明“整体可比较，但无法独立分摊”；不会为了提高类型覆盖率将同一个 GPU kernel 分配给多个 NPU 类型。Div/Sub 精度路径不再是诊断入口的限制。

JSON v2 包含 `selection`（筛选口径）、`summary`、`types`（类型与原因汇总）、`kernels`（逐 kernel 证据和建议），以及 `gpu_accounting`（全量/已映射/未映射计时）、`gpu_unmapped_activities`（GPU 余项）、`specialized_evidence.precision_paths`（完整精度路径证据）。GPU 余项和专项证据不受 NPU 低覆盖率阈值筛选。同模块同元数据 GPU CPU 候选最多展示 5 条，并给出完整候选数量与截断标记；没有已知模块时不会把全局同名调用当成可信候选。候选和最近匹配祖先只用于调查，不会自动成为映射。

这些建议是**确定性规则产生的待验证解释**，不是已证实的后端缺陷。未比较 NPU 时间不是可节省时间；没有独立 GPU kernel 也不等于 GPU 无此工作或耗时为零。验证实现问题仍需对应版本源码、相同输入/配置的最小复现和数值检查。通用规则集中在 `coverage_diagnostics.py`，扩展时应增加对应的合成边界测试，而不是继续堆叠 kernel 名称白名单。

### Div/Sub 精度路径诊断（专项证据）

正常运行比较命令即可在通用 Markdown 的“专项证据：完整精度路径”以及 JSON 的 `specialized_evidence.precision_paths` 查看，无需新增参数。Markdown 汇总路径与观测成本，JSON 保留每次调用的全部证据，不再作为表格页签或专项 CSV 输出。

第一版规则限定为 `aten::div/div_/true_divide + aclnnDivs_` 和 `aten::sub/sub_/subtract + aclnnSubs_`。Divs 主计算支持 `Muls`、`RealDiv`、`Div`；Subs 支持 `Sub`。规则集中在 `precision_analysis.py` 的 `RULES`，不依赖全局 kernel 数量一致或时间相邻。

| Pattern | 含义 |
| --- | --- |
| `round_trip_promotion` | FP16/BF16 输入经独立 Cast 升为 FP32，主计算后再转回原 dtype |
| `input_promotion` | 只有输入升精度，没有观测到回转 Cast；不推断回转缺失或冗余 |
| `output_conversion` | 只观测到主计算后的输出转换 |
| `native_fp16/native_bf16/native_fp32` | 主 kernel 的输入输出存储 dtype 一致，无独立 Cast；不代表内部指令精度 |
| `incomplete_metadata/ambiguous_path/other_precision_path` | 元数据不完整、跨 stream/混合 API/多个主计算或未识别路径，保留记录，不输出性能比 |

识别与计时约束：

- 按 CPU 调用及其子调用收集完整 NPU scope，包括原来的 Cast helper。只折叠主工作相同的同族嵌套 overload，不跨相邻调用补 Cast。一个 NPU kernel 在精度路径视图只出现一次。未归属的相关 API kernel 数量记入 `summary.json -> precision_analysis.unscoped_api_kernel_count`，不能理解为这些 kernel 不存在。
- Cast 必须保持 shape，前后 dtype/shape 必须与主 kernel 相容；路径必须在同一已知 stream/device，且没有明显执行重叠。使用原始 CSV 十进制时间排序，避免 epoch 时间戳转换为 float 后丢失顺序。
- 性能比较要求唯一且可信的 CPU 匹配、两侧完整 CPU 输入 shape/dtype 一致、NPU CPU 输入与设备输入一致，以及强 owner 关联（External ID、runtime correlation、torch_to_npu 或 host_to_device）。CPU 元数据中的标量 dtype 不等于输出 dtype，本工具不自行取“最大 dtype”推断类型提升。
- `GPU Total Time = GPU Kernel Time + GPU Memory Time`，仅包含对应调用的 device 活动，不包含 CPU runtime API 耗时。GPU 计算和内存活动分别去重；没有可比较数据时留空，不当成 0。汇总性能比为 `sum(GPU time) / sum(对应 NPU 完整调用 time)`，不是逐调用比值的平均值。
- 精度路径是原有工作的另一种分组视图，**不能与逐类型、整体算子或 FA 报表相加**。原有 Cast 的 Direct/Comparable/Accounted、GPU 时间分配不变。

`Candidate Cause=possible_unfused_conversion` 仅表示可信配对下，NPU 有前置/往返升精度，而 GPU 没有观测到独立转换迹象。GPU 子调用 `to/_to_copy/copy_/type_as` 或明确 Cast kernel 名称会单独标记，复制子调用不自动认定为 dtype 转换。GPU kernel 内部仍可能使用 FP32 opmath。

所有记录的 `Evidence Level=ordered_metadata_pattern`、`Cause Status=not_verified`：没有张量地址或完整数据依赖证明时，dtype/shape/顺序只能支持路径模式，不能自动确认为后端缺陷。`Cast Time` 是已观测成本，不是承诺可节省时间。确认根因仍需核对 SoC、CANN/torch_npu/PyTorch 版本、标量/零维 Tensor overload、autocast、布局及数值精度。分析过程不自动生成修复代码或更改 dtype；可选 skill 封装见本文开头，运行分析命令不会自动安装 skill。

只有一个 NPU 设备时，前三个页签隐藏冗余的 `Device_id` 列；多个设备时显示。前三页沿用 CSV 的融合组顺序及各 `OP Type` 明细行，融合组名称纵向合并；第四页使用独立的 CPU bundle 数据。

## 时间字段

- `Kernel Sum(us)`：bundle 中所有 kernel duration 之和；多 Stream 并发时可能重复计算并发区间。
- `Device Busy(us)`：所有 kernel 时间区间的并集，并发区间只计算一次。
- `Bundle Span(us)`：第一个 kernel 开始到最后一个 kernel 结束，包含 kernel 之间的空隙。

比较算子工作量时看 `Kernel Sum`；比较实际设备占用看 `Device Busy`；观察拆分 kernel 带来的间隙和完成延迟时看 `Bundle Span`。

## 匹配和限制

1. shape、dtype 和 Module 首先规范化；完整语义 token 用作高质量锚点，锚点之间再按名称和执行顺序对齐。shape、dtype、Module、调用栈和相对位置共同计算置信度。
2. `matches.csv` 保留所有序列匹配用于审计；只有 `medium/high` 且两边都包含 kernel 的匹配才进入 `kernel_type_statistic.csv` 等性能统计。零 kernel 的 `0 == 0` 不再视为 kernel 数量相等。
3. 可信语义 kernel 锚点之间的连续区间写入 `region_matches.csv`；它是逐 aten 统计的粗粒度替代视图，不能与逐 aten 时间相加。中高置信度且两边都有 kernel 的区域可以用于整体 device time 比较。
4. kernel 优先通过 `External id` 关联到 aten；其次使用 kernel/runtime correlation 和 trace flow。Ascend flow 按来源选取证据：`torch_to_npu` 通过起点所在的 CPU 线程定位原始调用；`HostToDevice` 通过 worker 上的 Dequeue、同进程/API 的 `correlation_id` 和 Enqueue 回溯到 CPU 调用，同步下发可按相同 TID 关联。两者都不允许跨 CPU 线程仅按时间寻找 owner。`async_task_queue` 只连接 CPU 队列，不参与 kernel 端点匹配。Flow ID 按 category 隔离，候选冲突时优先选择原始 `torch_to_npu`，相同强度但不同 owner 的冲突保留未关联。旧格式的通用 flow 仍允许时间回退。`kernel_link_diagnostic.csv` 的 `Link Method` 和 summary 的 `flow_selected_method_counts` 记录采用的来源。
5. NPU 设备任务同时支持 `cat=kernel` 和 `process_name=Ascend Hardware` + `Task Type=AI_CORE` 等格式。EVENT/PLACE_HOLDER/NOTIFY 等标记任务默认排除并单独计数。
6. `kernel_details.csv` 是 NPU 计算 kernel 的权威清单：优先用 `Task ID + Stream ID` 关联 trace kernel，缺失时按 Stream、名称、时间和 duration 降级匹配；summary 会分别统计 ID、顺序、模糊和 synthetic 关联数量。
7. 对缺失直接证据的旧格式 trace，通用 flow 起点若落在嵌套 helper aten 内，最短包含区间可能造成 kernel 误归属。当某个 aten 类型与 `kernel_details.csv` 的同名 Type 数量严格相等，且已有 flow 关系满足足够的顺序证据时，工具会按调用顺序校正归属。工具还会利用 ACLNN 名称族（如 `aclnnAddmm`、`aclnnClamp`）恢复真实 CPU op；名称族只接收对应主 `Type`，不会把同次调用产生的 Cast/Tile/Concat 等辅助 kernel 当作主 kernel。`aclnnStack_ConcatD` 若与 `aten::stack` 调用数严格相等、处于同一 Stream 且逐次时间偏差不超过 flow 容差，也允许在原 flow 全部错指向其他算子时按序校正。这些顺序规则不会覆盖 `torch_to_npu`、队列回溯、External ID 等更强的关联。校正前的 owner/link 信息保留在 `kernel_link_diagnostic.csv`，汇总写入 `summary.json` 的 `kernel_owner_reconciliation`。
8. NPU-first 路径以 `kernel_details.csv` 中的 NPU kernel 为起点，经 `NPU kernel -> NPU CPU op -> 匹配的 GPU CPU op -> GPU kernel` 生成映射。NPU 单 kernel 对应多个 GPU kernel 时比较 GPU bundle 总时间；数量相等时顺序配对；通过 aten/NPU Type 别名可识别唯一主 kernel 时只将 GPU kernel 计入主类型，辅助 kernel 不重复占用 GPU 时间。同一次 CPU op 的多个同类型、同 Core 主 NPU kernel 若无法逐一配对，则以 `npu_gpu_bundle_shared` 比较两边完整 bundle：逐 NPU kernel 行展示共享 GPU bundle，但单行性能比留空，汇总时 GPU kernel 只计一次。其余多对多关系标记为 `ambiguous_bundle_unmapped`。若 ACLNN API 的主 Type 与归属 CPU op 语义冲突，标记 `api_semantic_conflict_unmapped`，不参与性能比较。已识别的非 Cast API 下发的 Cast 辅助 kernel 标记 `api_helper_unmapped`，计入 Accounted 但不单独与 GPU 计算 kernel 比较。若匹配的 GPU 父 op 没有直接 kernel，且 NPU Type 与父 op 的语义一致，可使用其子 aten 的完整 kernel bundle（如 `relu -> clamp_min`、`embedding -> index_select`）；已分配给其他 NPU kernel 的 GPU kernel 不会重复使用。
9. 对同名、同 Module、两边都有 kernel 且由局部名称顺序直接配对的调用，部分后端元数据差异可提升为 `medium` 置信度。但 Fill、sum/arange 和 index_put 系列先检查父调用、完整输入 shape/dtype，明确冲突不允许提升。CUDA 可选参数的空 dtype 与 NPU `none` 在这一路径中等价。sum/arange 的同名嵌套 overload 不作为额外逻辑调用；同模块、同父调用类型、同完整元数据且数量一致时按组内顺序恢复，标记 `logical_call_metadata`，允许不同 dtype 的独立 index_put 调用改变顺序。Cast 的 copy_/to 规则不受此修复影响。
10. GPU `FlashAttentionFunc` 是非 `aten::` 的自定义 CPU op，工具会额外保留它。NPU 侧将 `_attn_fwd` 视为主计算 kernel，将 `gen_fa_mask_kernel` 视为同一次 FlashAttention 的辅助 kernel：逐 kernel 报告只把 GPU `fwd_kernel` 映射到 `_attn_fwd`，辅助 kernel 标记为 `fused_bundle_auxiliary`；`fused_op_mapping.csv` 再把两类 NPU kernel 与 GPU bundle 合为一次整体比较。只有 GPU 自定义 op、NPU 主 kernel 和每类辅助 kernel 的数量严格相等、执行顺序和 Stream 关系有效时才恢复；否则保持未映射。
11. `npu_gpu_op_statistic.csv` / `.xlsx` 中，`Direct` 表示一个 NPU kernel 与一个独占的 GPU kernel 配对；`Comparable` 既包括独占配对，也包括同次 CPU op 的共享整体 bundle 比较（后者不提供逐 kernel 性能比）；`Accounted` 还包含已知融合辅助 kernel、同一可信 CPU-op bundle 中无法单独分摊的辅助或多对多 kernel。每种覆盖率同时给出 kernel 数量占比与 NPU 时间占比，且 `Direct <= Comparable <= Accounted`。旧列 `Mapping Coverage(%)` 等于 `Comparable Coverage(%)`，用于兼容已有分析脚本。两个比值的 GPU 分子相同，均只覆盖已对齐 kernel；Comparable 分母是对应 NPU 时间，Total Time 分母是该类型全部 NPU 时间。低覆盖率时尤其不能把 Total Time 比值误读为完整算子的真实性能比。融合算子按 `FUSED_OP_NPU_TYPE` / `FUSED_OP_GPU_TYPE` 分组；例如 FlashAttention 的 NPU `_attn_fwd + gen_fa_mask_kernel` 对应 GPU `FlashAttentionFunc`，其 mask kernel 计入 Accounted 而不计入 Comparable。
12. `low` 置信度或低 kernel-duration coverage 的结果只用于定位，不进入可信性能统计。
13. CUDA Graph、NPU 静态图或编译器大范围融合可能丢失逐 aten 的下发关系。这类数据应加入相同的 `record_function`/NVTX/MSTX 区域，主要查看 `region_matches.csv` 和 `module_statistic.csv`。
14. 当前实现仍会将 trace JSON 整体读入内存；超大 trace 建议采集或预裁剪单个 step。

`cat/concat` 按调用实例关联，嵌套的 `aten::concat -> aten::cat` 不计作两次独立拼接。同次调用的多个同 Core ConcatD 即使与 GPU kernel 数量相等，也使用 `npu_gpu_bundle_shared` 整体比较，避免误认为输入分块方式相同。父 op 使用 GPU 子调用（如 `stack -> cat`）的同类型共享比较标记为 `gpu_child_shared_bundle`，同样对 GPU 时间去重。类型表中的 ConcatD 时间只包含 ConcatD；比较整个 Cat 调用（含 NPU Cast/Slice 等辅助 kernel）时，查看 `matches.csv` 的 `aten::cat` 行及两端 `Kernel Sum(us)`，不要将外层 concat 的时间再次相加。

## 测试

### 非 Cast kernel 的补充对齐

- `aten::mul -> Muls` 主类型别名将乘法计算与 Transpose 辅助任务分开；只恢复主计算的 GPU 映射。未配对的 `clone -> copy_` 仅在 NPU kernel 全部为 Slice、同模块/祖先类型/完整 shape/dtype、两侧剩余候选各唯一时恢复，不影响 Cast。
- GPU `gpu_memcpy`/`gpu_memset` 保存到独立活动表，通过唯一 External ID 或同 PID/TID 下的 runtime correlation 定位 CPU owner。重复/跨进程 correlation 或强证据冲突不匹配。设备拷贝不改变计算 kernel 的编号、数量及原有统计。
- 当前仅对无计算 kernel 的可信 `repeat -> Tile` 和 `cat -> TensorMove` 启用 DtoD 对齐，标记 `gpu_device_memcpy`。`GPU Kernel Count` 仍只计计算 kernel；新增 `GPU Memory Activity Count`、`GPU Kernel Time(us)`、`GPU Memory Time(us)`。NPU-first 的 `GPU Total Time(us)` 是两类已匹配设备时间之和，`GPU Activity Kind` 明示来源。内存对齐计入 Comparable，不计入 Direct Kernel Pair；原有 GPU Min/Avg/Max 仍仅统计计算 kernel。旧 GPU-first/aten/module 计算 kernel 报表保持原口径。
- 混合 `Pack + ConcatD` 的同次 Stack 仅进入整体比较，不向多个类型重复分摊 GPU 时间。标量准备＋IndexPut 仅接受同模块、完整消费者元数据一致、两侧附近唯一 bool 标量准备、GPU 消费者内唯一 1 字节 HtoD 的情况；限定同 CPU 线程、前 8 个 op、间隔不超过 1000 us，且不跨另一 IndexPut 调用。NPU scope 若包含其他类型（尤其 Cast）则不恢复。
- `CPU Bundle ID` / `Comparable In CPU Bundle` 在 NPU 明细中连接完整调用；Fill 可因此计入 Accounted，但不会伪装成独立 GPU Fill 的 Comparable。每个整体视图内部对 GPU/NPU 活动去重，整体视图与逐类型视图不可相加。

- GPU 子调用支持 `repeat -> copy_`（Tile）、`zero_ -> fill_`（ZerosLike）、`ones -> fill_`（OnesLike）、`rsub -> sub`（Sub）。子 kernel 已占用或没有设备工作时保留未映射。`count_nonzero` 仅选取 sum 子调用对应 ReduceSum，不重复使用 ne 或把 copy 当归约。
- 两侧额外保留 `NestedConcatFunc`。position concat/offset 使用 GPU 的 `position_concat.py: position_concat` / `position_offset_creator.py: create_position_offset` Python 作用域与 NPU 对应自定义 CPU op 对齐。只有同模块内调用数、调用顺序及单个直接 kernel 名称一致时，才提升为可信 `custom_scope_same_kernel`；不做全局 kernel 同名拉齐。
- 示例新结果可输出到独立目录，保留旧结果用于审计：

```bash
python -m profiling_tools --gpu-trace profiling_tools/example/demo1/gpu/trace.json --npu-trace profiling_tools/example/demo1/npu/trace_view.json --npu-kernel-details profiling_tools/example/demo1/npu/kernel_details.csv --gpu-step 15 --npu-step 15 --output-dir profiling_tools/example/demo1/result_npu_first_v14
```

### Rules 2：有证据约束的调用范围恢复

- 卷积支持 `_convolution / convolution -> Conv2DV2`，`index_select -> GatherV2` 区分主计算与 Transpose 辅助任务。卷积子调用与 GatherV2 混合 bundle 恢复要求完整 CPU 元数据一致。
- NPU kernel 位于嵌套 GELU overload 时，可以从可信父调用恢复单类型 scope。保留实际 NPU owner，并通过 `CPU Match ID` 指向比较父调用。`logical_scope_bundle` / `logical_scope_shared_bundle` 明示范围；后者只按 bundle 比较、不分摊单 kernel 比值。GPU 工作已占用、混合类型、跨设备、弱 owner、API 语义冲突、缺失元数据或 scope 含 trace-only 任务时不恢复。
- 两侧保留 `RopeFunc`、`customop::triton_target_attention_nobitwise_varlen`、`customop::triton_layernorm_forward`。通过已有 External ID / correlation / NPU flow 恢复 owner；只有同模块完整调用序列的数量、shape、dtype 一致才标记 `custom_scope_metadata_sequence`。不按 device kernel 名称相似或总数量直接配对。
- LayerNorm 模块编号仅在两个可信锚点之间，完整 mean/sub/var/div/add 等原语序列元数据一致且候选唯一时允许恢复，标记 `layernorm_block_metadata`。重复序列、真实 shape 冲突及缺失元数据保留未匹配；原模块名仍保留在报告中。跨模块且缺少完整块证据、或存在其他元数据冲突的 LayerNorm `same_name_recovery` 不再作为配对保留，可能降低局部小算子的覆盖率。
- 这些规则不把 Cast、Slice、Transpose 等辅助工作的 GPU 时间重复分配到多个类型；已在整体视图比较不等于逐类型独立可比。

无需新增命令行参数。运行中的 Web 服务需重启后重新分析，已有任务的历史报告不会自动更新；独立安装的 skill 分发包需要重新构建并更新才能使用新规则。

### 运行回归测试

```bash
python -m unittest discover -s profiling_tools/tests -v
```

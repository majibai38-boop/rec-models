# 归档协议（本地）

这是 P1-05 的**本地部分**：把一次 attempt 从 staging 校验后原子发布到归档目录，形成只读 Web（P1-06）可直接索引的目录结构。SSH 传输、远端预检与 Web 尚未实现。

## 目录协议

```text
<archive_root>/
  staging/<run_id>/<attempt_id>/     # 运行期写入，未发布
  runs/<run_id>/<attempt_id>/        # 已发布，只读
    artifacts.json                   # 相对路径、大小、SHA256、媒体类型
    COMPLETE                         # 发布标记，含 artifacts.json 的 SHA256
    ...                              # 实际产物（manifest、metrics、logs、profile 等）
```

staging 与最终目录使用同一结构，publish 只做"校验 + 目录改名"，因此不会出现半个归档。`COMPLETE` 只表示归档校验完成，**不表示 benchmark 成功**；失败实验也可以有完整的归档。

## 命令

```console
python -m benchmark_core.archive new --root benchmark-results/archive --run-id r1
python -m benchmark_core.archive publish --root benchmark-results/archive --run-id r1 --attempt-id attempt-1
python -m benchmark_core.archive verify benchmark-results/archive/runs/r1/attempt-1
```

| 命令 | 行为 |
|---|---|
| `new` | 新建 staging attempt；attempt 编号在 staging 与已发布目录中都不重复使用 |
| `publish` | 生成 `artifacts.json` → 写 `COMPLETE` → 重新校验 → 原子改名为 `runs/...` |
| `verify` | 重新计算清单中每个文件的大小与 SHA256，并检查缺失、多余、清单非法、标记不一致 |

退出码：`0` 成功；`2` 输入/状态错误（含内容冲突）；`3` `verify` 发现归档不完整或已被篡改。

## 幂等与冲突

- 同一 attempt 重试发布：内容完全相同 → 返回 `status=existing`，不覆盖、不修改已发布文件；
- 内容不同 → 报 `conflict`，绝不覆盖历史结果；
- 重新跑模型/测量应使用**新 attempt**；只有归档重试才复用原 attempt。

`publish` 会先写清单再校验，因此任何大小/hash 不符都会在改名之前失败，staging 保留供排查。

## 当前边界

- 只实现 `transport=local`；`archive.transport=ssh` 需要的预检、上传、远端校验与发布尚未实现。
- 不读取或改写 `manifest.json` 中的运行状态，也不计算成绩；这些由 P1-02～P1-04 的脚本负责。
- Web 索引属于 P1-06；本模块只保证目录协议与校验语义。

## 测试

```console
python -m unittest benchmark_core.archive.tests.test_archive -v
```

测试覆盖：attempt 编号、发布顺序、清单内容、幂等重试、内容冲突、篡改/缺失/多余文件、清单非法、路径穿越、失败实验归档与 CLI 退出码。

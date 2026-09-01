# ADEF Evaluation — publication entrypoint

从 Paper Protocol v2 起，**唯一推荐的论文评估入口是：**

```bash
python paper_evaluator.py --manifest MANIFEST.csv --method METHOD --output-dir OUT
```

完整指标定义、官方来源和表格口径见 [`README_metrics.md`](README_metrics.md)。

## 为什么不再用旧的逐视频 unified 流程？

旧流程把 FID/FVD 对每个视频对分别计算，再对这些 Fréchet distances 求均值。这不是标准 FID/FVD，数学上也不等价于完整数据集的 Fréchet distance。`unified_evaluator.py` 现在只保留单视频诊断兼容接口，并会明确拒绝单视频 FID/FVD。

## Manifest

CSV/TSV 至少包含：

```text
name,fake,gt
```

建议包含：

```text
name,fake,gt,emotion,image,audio
```

`emotion` 使用语义标签；脚本统一归一化 `angry→anger`、`happy→happiness`、`sad→sadness`、`surprised→surprise`、`disgusted→disgust`。

## 输出是否能直接填论文？

看 `paper_table.csv` 的 `Status`：

- `complete`：请求的指标均成功且样本覆盖满足协议，可作为该 evaluator 下的论文结果；
- `incomplete`：只能排错，禁止作为论文主表值。

同时保留 `paper_metrics.json` 和 manifest，便于审稿/复现实验时追踪具体 evaluator、覆盖率与样本。

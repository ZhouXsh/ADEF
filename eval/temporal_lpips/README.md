# Temporal LPIPS / 感知闪烁

该评估器使用 LPIPS 衡量相邻帧之间的感知变化。它可用于发现简单像素差异难以捕捉的感知闪烁。

## 安装准备

如果当前环境中没有 LPIPS，请安装：

```bash
pip install lpips
```

## 用法

```bash
python eval/temporal_lpips/eval_temporal_lpips.py \
  --video path/to/generated.mp4 \
  --out eval_results/temporal_lpips.json
```

批量模式：

```bash
python eval/temporal_lpips/eval_temporal_lpips.py \
  --manifest videos.txt \
  --out eval_results/temporal_lpips_batch.json
```

## 输出字段

- `temporal_lpips_mean`：相邻采样帧之间的平均 LPIPS。
- `temporal_lpips_std`：感知变化在时间维度上的不稳定程度。

通常数值越低表示感知闪烁越少，但过低也可能意味着视频过于静止。因此需要结合动作动态指标一起分析。
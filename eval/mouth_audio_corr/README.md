# 嘴部运动—音频相关性

这是一个轻量级唇音同步代理指标，不需要预训练 SyncNet checkpoint。

它会估计：

1. 由 MediaPipe FaceMesh 关键点得到的嘴部开合程度；
2. 由 RMS 和 MFCC-delta 得到的音频动态包络；
3. 嘴部运动与音频动态之间的零延迟相关性和最佳延迟相关性。

## 为什么使用它

当你想在开发阶段快速得到诊断信号时，可以使用该指标。它不能替代 SyncNet LSE-C/LSE-D，但在没有安装官方 SyncNet 推理代码时，适合用于比较 ADEF 不同版本之间的相对变化。

## 用法

```bash
python eval/mouth_audio_corr/eval_mouth_audio_corr.py \
  --video path/to/generated.mp4 \
  --out eval_results/mouth_audio.json
```

批量模式：

```bash
python eval/mouth_audio_corr/eval_mouth_audio_corr.py \
  --manifest videos.txt \
  --out eval_results/mouth_audio_batch.json
```

## 输出字段

- `mouth_audio_rms_corr`：嘴部开合程度与 RMS 音频包络之间的 Pearson 相关系数。
- `mouth_audio_mfcc_corr`：嘴部开合程度与 MFCC-delta 音频包络之间的 Pearson 相关系数。
- `best_lag_*`：在一个小的时间偏移窗口内搜索得到的最佳相关性。
- `mouth_opening_*`：嘴部开合程度的统计量。

通常相关性越高越好，但应在相同数据集和相同预处理流程内比较。
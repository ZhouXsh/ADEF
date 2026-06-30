# 无参考图像 / 视频质量代理指标

该评估器会在采样视频帧上快速计算一些诊断性画质代理指标：

- Laplacian 模糊度 / 清晰度；
- 亮度均值与标准差；
- 灰度熵；
- 基于 MediaPipe 的人脸检测比例；
- 人脸框面积占比；
- 帧间像素差异。

## 用法

```bash
python eval/no_reference_iqa/eval_iqa_basic.py \
  --video path/to/generated.mp4 \
  --out eval_results/iqa_basic.json
```

批量模式：

```bash
python eval/no_reference_iqa/eval_iqa_basic.py \
  --manifest videos.txt \
  --out eval_results/iqa_basic_batch.json
```

## 结果解释

这些指标不是最终感知质量指标。它们主要用于发现模糊帧、曝光问题、人脸丢失、严重闪烁或裁剪异常。若要做正式的图像分布质量评估，请使用 FID 或人工主观评价。
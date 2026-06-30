# 关键点动态指标

该指标族基于 MediaPipe FaceMesh 关键点估计视频的时间自然度。

它会输出：

- 全局关键点速度与加速度；
- 嘴部开合动态；
- 眉眼动态；
- 嘴部关键点抖动；
- 眉眼关键点抖动。

## 用法

```bash
python eval/landmark_dynamics/eval_landmark_dynamics.py \
  --video path/to/generated.mp4 \
  --out eval_results/landmark_dynamics.json
```

批量模式：

```bash
python eval/landmark_dynamics/eval_landmark_dynamics.py \
  --manifest videos.txt \
  --out eval_results/landmark_dynamics_batch.json
```

## 结果解释

- 嘴部或眉眼动态过低，可能说明人脸过于僵硬。
- 加速度或抖动过高，可能说明动画不稳定或存在伪影。
- 最好与同一分布下的真实视频或参考视频进行对比。
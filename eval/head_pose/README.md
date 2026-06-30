# 头部姿态自然度

该评估器基于 MediaPipe 关键点和 OpenCV `solvePnP` 近似估计 yaw、pitch 和 roll。

## 用法

```bash
python eval/head_pose/eval_head_pose.py \
  --video path/to/generated.mp4 \
  --out eval_results/head_pose.json
```

批量模式：

```bash
python eval/head_pose/eval_head_pose.py \
  --manifest videos.txt \
  --out eval_results/head_pose_batch.json
```

## 输出字段

- `yaw/pitch/roll`：均值、标准差、速度和加速度统计。
- `pose_jitter`：平均绝对角加速度。
- `frames_with_pose`：成功估计头部姿态的帧数。

## 注意事项

该指标是相对诊断指标。若用于正式汇报，请与一致的真实/参考集合进行对比，并确认当前管线中 yaw、pitch、roll 的定义约定。
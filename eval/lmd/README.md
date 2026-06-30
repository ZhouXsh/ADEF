# LMD / Landmark Distance 几何误差评估

LMD（Landmark Distance）用于评估生成视频与参考 / GT 视频之间的人脸关键点几何误差。它比 PSNR、SSIM 更适合分析说话人脸中的口型和面部运动误差，尤其适合检查音频驱动口型是否偏离真实轨迹。

## 适用场景

LMD 是 paired metric，需要生成视频和参考视频逐帧大致对齐。典型输入是：

```csv
generated,reference
/path/to/generated.mp4,/path/to/gt.mp4
```

如果生成视频没有逐帧对齐的 GT / reference，LMD 的解释力会下降。对于只给参考图的 talking-head 生成任务，LMD 更适合作为有 GT 子集上的辅助验证指标。

## 指标定义

脚本基于 MediaPipe FaceMesh 提取关键点，并计算：

```text
LMD = mean_t mean_i || p_gen(t,i) - p_ref(t,i) ||_2 / normalize_factor
```

默认输出：

- `full_lmd`：全脸 468 点平均距离；
- `mouth_lmd`：嘴部关键点平均距离；
- `outer_mouth_lmd`：外唇关键点平均距离；
- `inner_mouth_lmd`：内唇关键点平均距离；
- `brow_lmd`：眉部关键点平均距离；
- `eye_lmd`：眼部关键点平均距离。

其中 `mouth_lmd` 对 ADEF 最关键，因为它直接反映口型几何误差。

## 归一化方式

通过 `--normalize` 设置：

```text
face_width   默认。使用参考视频当前帧人脸宽度归一化。
interocular  使用参考视频双眼中心距离归一化。
none         不归一化，直接使用归一化坐标距离。
```

推荐默认使用 `face_width`。

## 用法

单对视频：

```bash
python eval/lmd/eval_lmd.py \
  --generated path/to/generated.mp4 \
  --reference path/to/gt.mp4 \
  --out eval_results/lmd.json
```

批量 CSV：

```bash
python eval/lmd/eval_lmd.py \
  --manifest generated.csv \
  --out eval_results/lmd_batch.json
```

其中 `generated.csv` 至少包含：

```csv
generated,reference
/path/to/gen_001.mp4,/path/to/gt_001.mp4
/path/to/gen_002.mp4,/path/to/gt_002.mp4
```

## 输出解释

- LMD 越低，说明生成视频与参考视频的关键点几何越接近。
- `mouth_lmd` 越低，通常说明口型几何越接近参考。
- 若 `mouth_lmd` 下降但 SyncNet LSE 没有改善，说明几何更接近参考，但音频同步未必更好。
- 若 `full_lmd` 很低但 `mouth_lmd` 很高，说明整体脸部稳定，但口型错误明显。

## 注意事项

- LMD 强依赖帧对齐、裁剪尺度和人脸检测稳定性。
- 建议同时汇报 SyncNet LSE-C/LSE-D、mouth-audio correlation 和 identity 指标。
- 本脚本使用的是 MediaPipe 468 点，不是 LivePortrait 的 21 个 expression keypoint。
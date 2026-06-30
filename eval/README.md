# ADEF 说话人脸视频评估工具包

本目录提供一套模块化的音频驱动说话人脸视频评估工具。每一类指标都放在独立子目录中，并配有对应的 README 和命令行入口。

本工具包不会把第三方研究代码直接拷贝进仓库，而是尽量提供轻量、可复现的封装；当某些指标需要官方仓库或预训练权重时，会在对应 README 中说明如何接入。

## 目录结构

```text
eval/
  common/                  公共视频、音频、人脸工具函数
  sync_lse/                SyncNet / LSE-C / LSE-D 封装
  mouth_audio_corr/         轻量级嘴部运动—音频相关性指标
  lmd/                      Landmark Distance 几何误差指标
  identity_arcface/         基于 InsightFace/ArcFace 的身份保持指标
  landmark_dynamics/        嘴部、眉眼、关键点速度与抖动指标
  head_pose/                基于 MediaPipe + solvePnP 的头部姿态自然度指标
  no_reference_iqa/         模糊度、亮度、熵、人脸区域等无参考画质代理指标
  fid_frame/                帧级 FID 封装
  fvd/                      视频级 FVD 封装
  temporal_lpips/           时间感知一致性 / 感知闪烁指标
  emotion_consistency/      情感标签一致性 / 置信度指标
  run_all.py                批量运行已安装指标的总控脚本
```

## 输入格式

大多数脚本支持单个视频：

```bash
--video path/to/video.mp4
```

也支持 txt 列表文件：

```text
/path/to/gen_001.mp4
/path/to/gen_002.mp4
```

需要参考视频、音频或情感标签的指标可以使用 CSV：

```csv
generated,reference,audio,label
/path/to/gen.mp4,/path/to/ref.mp4,/path/to/audio.wav,happy
```

## 快速开始

```bash
python eval/no_reference_iqa/eval_iqa_basic.py --video demo.mp4 --out demo_iqa.json
python eval/mouth_audio_corr/eval_mouth_audio_corr.py --video demo.mp4 --out demo_mouth_audio.json
python eval/landmark_dynamics/eval_landmark_dynamics.py --video demo.mp4 --out demo_landmarks.json
python eval/head_pose/eval_head_pose.py --video demo.mp4 --out demo_pose.json
```

如果有逐帧对齐的参考视频，可以运行 LMD：

```bash
python eval/lmd/eval_lmd.py --generated gen.mp4 --reference gt.mp4 --out demo_lmd.json
```

官方 SyncNet/LSE、FID、FVD、LPIPS 和情感一致性指标可能需要额外 checkpoint 或可选依赖，请先阅读各子目录的 README。

## 推荐的 ADEF 评估矩阵

建议每个生成视频至少汇报以下维度：

1. 唇音同步：优先使用 SyncNet LSE-C/LSE-D；没有官方 checkpoint 时可使用 mouth-audio correlation 作为轻量代理指标。
2. 几何误差：在有逐帧对齐参考视频时汇报 full LMD 和 mouth LMD。
3. 身份保持：使用源图/参考视频与生成视频帧之间的 ArcFace cosine similarity。
4. 动作自然度：嘴部、眉眼、头部的速度、加速度与抖动。
5. 图像质量：无参考画质代理指标；有真实/参考帧时可补充 FID。
6. 视频质量：样本数量足够时使用 FVD 做分布级视频质量评估。
7. 情感一致性：目标情感标签置信度、时间稳定性和不同情感之间的可分性。

## 重要注意事项

- LMD 是 paired metric，需要生成视频与参考视频逐帧大致对齐。
- FID 和 FVD 是分布级指标，不适合只评估单个视频或很小样本集。
- SyncNet/LSE 需要预训练 SyncNet checkpoint，本仓库只提供封装入口。
- MediaPipe 指标适合比较 ADEF 不同版本之间的相对变化，但不能替代人工主观评价。
- 无参考画质指标主要用于诊断问题，不应作为最终感知质量结论。
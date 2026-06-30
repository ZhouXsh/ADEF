# 评估指标参考与实现说明

本文档整理 `eval/` 中使用的指标类型，以及常见的外部实现和资源。

## 唇音同步

- Wav2Lip / SyncNet：LSE-D 和 LSE-C 是说话人脸领域常用的音视频同步指标。官方仓库：`https://github.com/Rudrabha/Wav2Lip`。
- 推荐本地封装：`eval/sync_lse/`。
- 轻量级替代指标：`eval/mouth_audio_corr/`。

## 几何误差

- LMD / Landmark Distance：说话人脸领域常用的关键点几何误差指标，适合在有逐帧对齐参考视频时评估面部运动和口型误差。
- 推荐重点关注 `mouth_lmd`，它直接反映嘴部几何轨迹与参考视频之间的偏差。
- 本地封装：`eval/lmd/`。

## 身份保持

- ArcFace / InsightFace 风格的人脸 embedding 常用于衡量生成视频的身份保持能力。
- ADEF 的 `requirements.txt` 中已经包含 `insightface`。
- 本地封装：`eval/identity_arcface/`。

## 分布级图像 / 视频质量

- FID：帧级图像分布指标。封装目录：`eval/fid_frame/`。
- FVD：视频分布指标，通常使用 I3D 特征。封装目录：`eval/fvd/`。
- 注意：二者都需要足够样本量，并且必须保持一致的预处理流程。

## 动作自然度与表情表现

- 关键点动态、嘴部运动、眉眼运动和头部运动可作为说话人脸生成的诊断性指标。
- 本地封装：`eval/landmark_dynamics/`、`eval/head_pose/`。

## 感知时间一致性

- 相邻帧 LPIPS 可作为感知闪烁的代理指标。
- 本地封装：`eval/temporal_lpips/`。

## 情感一致性

- 情感识别应在所有方法中使用同一个固定分类器。
- 本地封装：`eval/emotion_consistency/`，支持 HuggingFace 图像分类模型或自定义外部命令。

## 推荐汇报表

| 维度 | 指标 | 脚本 | 方向 |
|---|---|---|---|
| 唇音同步 | LSE-D | `sync_lse` | 越低越好 |
| 唇音同步 | LSE-C | `sync_lse` | 越高越好 |
| 唇音同步代理 | mouth-audio corr | `mouth_audio_corr` | 越高越好 |
| 几何误差 | full LMD | `lmd` | 越低越好 |
| 嘴部几何误差 | mouth LMD | `lmd` | 越低越好 |
| 身份保持 | ArcFace cosine | `identity_arcface` | 越高越好 |
| 图像质量 | FID | `fid_frame` | 越低越好 |
| 视频质量 | FVD | `fvd` | 越低越好 |
| 自然度 | mouth/head/eyebrow jitter | `landmark_dynamics`, `head_pose` | 与真实分布对齐 |
| 感知闪烁 | temporal LPIPS | `temporal_lpips` | 通常越低越稳定，但需避免过度静止 |
| 情感 | target confidence/top1 | `emotion_consistency` | 越高越好 |

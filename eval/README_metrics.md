# ADEF 论文评估协议（Paper Protocol v2）

> **论文实验请以 `paper_evaluator.py` 的 `paper_table.csv` 为唯一主表来源。**
> `Status=complete` 才表示该行满足完整样本覆盖；`incomplete` 只能用于排错，不能填论文表。

## 1. 论文主表指标

| 指标 | 方向 | 本仓库协议 | 官方来源 / 对齐依据 | 聚合层级 |
|---|---:|---|---|---|
| LSE-D ↓ / LSE-C ↑ | ↓ / ↑ | 25 fps → S3FD face track/crop → SyncNet-v2 | Wav2Lip `evaluation/scores_LSE` + `joonson/syncnet_python` | 每视频评分后，对完整测试集求均值 |
| FID ↓ | ↓ | 所有 GT 视频帧汇成一个 real set；所有生成帧汇成一个 fake set；调用 `pytorch-fid` 一次 | Wav2Lip evaluation README + `mseitzer/pytorch-fid` | **数据集级一次计算** |
| FVD ↓ | ↓ | 每视频均匀采样固定帧数 → Google I3D → 两个完整视频分布的 Fréchet distance | `google-research/google-research/frechet_video_distance` | **数据集级一次计算** |
| PSNR ↑ / SSIM ↑ | ↑ | 与 EAT 官方相同的 temporal linspace pairing 和 `utils_crop.crop_and_align` | `yuangan/evaluation_eat` | 有效对齐帧全局均值 |
| LPIPS ↓ | ↓ | 与 PSNR/SSIM 相同的 EAT 对齐帧对；官方 `lpips` AlexNet | `richzhang/PerceptualSimilarity` / `lpips` | 有效对齐帧全局均值 |
| M-LMD ↓ / F-LMD ↓ | ↓ | EAT 官方 dlib-68：嘴部 20 点 / 全脸 68 点，分别去中心后逐点 L2 | `yuangan/evaluation_eat/code/test_lmd.py` | 先每视频均值，再对视频均值 |
| EmotiEff-Acc ↑ | ↑ | MTCNN 检出人脸后，EmotiEffLib 8 类视频主导情感与目标标签比较 | `sb-ai-lab/EmotiEffLib` | 8 类视频 accuracy |
| DFER-CLIP-Acc ↑ | ↑ | 官方 DFEW fold-1 / ViT-B/32 / 16 segments / class descriptor | `zengqunzhao/DFER-CLIP` | 7 类视频 accuracy |

### DFER-CLIP 的类别边界

DFEW 只有：`happiness, sadness, neutral, anger, surprise, disgust, fear`，**没有 `contempt`**。
因此 MEAD 的 `contempt` 样本会输出 `label_supported=false`、`correct=null`，不进入 DFER accuracy 分母；`paper_table.csv` 同时给出 `DFER-N`。不要把该列称为“MEAD 8-class accuracy”，论文中建议写 **DFER-CLIP Acc. (7 cls.)**。

## 2. 已纠正的关键错误

1. **LSE 不再把整张视频帧直接 resize 到 224×224。** 默认先执行 SyncNet 官方 `run_pipeline.py`：转 25 fps、S3FD 检测/跟踪、224×224 face crop，再送入 SyncNet-v2。短 MEAD clip 仅将 upstream 的 track-duration gate 默认调为 5；用 `--lse-min-track 100` 可恢复 Wav2Lip 原 shell 的字面默认值。
2. **FID/FVD 不再逐视频计算后平均。** Fréchet distance 是分布级统计量；v2 每个方法在完整测试集上只产生一个 FID 和一个 FVD。
3. **FVD 的 batch size 16 只用于 I3D inference。** 最后不足 16 的 inference batch 可内部 padding，embedding 会在计算 Fréchet statistics 前裁回真实样本数。不会再通过“复制视频到 16 个”伪造统计样本。
4. **Emotion-FAN 不再使用随机 attention/classifier。** `Emotion-FAN/evaluate_emotion_fan.py` 必须显式提供训练完成的 AFEW FAN checkpoint，并且输入必须是官方 face-aligned frames；否则直接报错。
5. **EmotiEff 主表使用 correctness/accuracy，而不是 dominant fraction。** `dominant_fraction=1` 也可能是 100% 稳定地预测错情感，所以它只适合作诊断。
6. **NewEmo agreement 不再进入论文主表。** 两个分类器可以“一致地预测错”，agreement 不是 emotion correctness。
7. **所有论文行默认要求完整覆盖。** 任一必需指标或样本失败都会令 `Status=incomplete` 且进程返回非零。

## 3. 使用方式

### 已生成的一种方法 / ADEF checkpoint

准备 CSV：

```csv
name,fake,gt,emotion
0001,/abs/fake1.mp4,/abs/gt1.mp4,anger
0002,/abs/fake2.mp4,/abs/gt2.mp4,happiness
```

运行：

```bash
python eval/paper_evaluator.py \
  --manifest /path/to/manifest.csv \
  --method ADEF \
  --output-dir /path/to/eval/ADEF \
  --device cuda:0
```

输出：

- `paper_table.csv`：论文主表的一行；
- `paper_metrics.json`：完整协议、错误、原始聚合信息；
- `per_video.csv`：LSE / paired metrics / emotion prediction 的逐视频审计记录；
- `work/`：各 evaluator 的结构化中间报告。

### baseline 生成 + 统一评估

`final_evaluator.py` 的输入为：

```text
image,audio,gt_video[,emotion]
```

第 4 列 emotion 会逐样本生效；省略时从 MEAD GT 路径推断。运行后所有 baseline 都交给同一个 `paper_evaluator.py`，最终合并为：

```text
eval/RESULT/paper_table.csv
```

### ADEF 批量 checkpoint / 矩阵实验

```bash
python eval/ADEF_evaluator.py <exam_name> --pairs-file /path/to/pairs.txt
python eval/ADEF_all_evaluator.py --pairs-file /path/to/pairs.txt
```

`ADEF_all_evaluator.py` 只把当前 v2 协议下 `Status=complete` 的实验视为 done；旧 summary 或 incomplete 行会重新评估。

## 4. EAT / EmoNet 目录与运行前检查

当前仓库中的 `eval/evaluation_eat/` 和 `eval/emonet/` 是**直接 vendored 的普通目录**，不是 Git submodule。因此不需要、也不应该再执行 `git submodule update --init --recursive`。

`eval/evaluation_eat/code/utils_crop.py` 保持 EAT 官方的 crop/alignment 数学流程，并仅对资源路径做稳健化：`shape_predictor_68_face_landmarks.dat` 可以放在以下任一位置：

```text
eval/evaluation_eat/code/shape_predictor_68_face_landmarks.dat
eval/evaluation_eat/checkpoints/shape_predictor_68_face_landmarks.dat
```

模板文件必须存在：

```text
eval/evaluation_eat/code/base_68.npy
eval/evaluation_eat/code/base_68_close.npy
```

EmoNet 当前不是 `paper_evaluator.py` 默认论文主表指标。如果要单独运行 `eval/emonet/evaluate_emotion.py`，需要官方 8 类权重：

```text
eval/emonet/pretrained/emonet_8.pth
```

正式长实验前先运行：

```bash
python eval/check_paper_eval.py --deep-hash
```

如果还要使用 EmoNet，则运行：

```bash
python eval/check_paper_eval.py --deep-hash --with-emonet
```

各大模型权重应按各官方仓库许可证/说明下载；脚本不会在缺权重时生成占位结果。

DFER-CLIP 与 CLIP 权重放入 `eval/New_Emo/weights/`。OpenAI ViT-B/32 应使用官方 SHA256：

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

## 5. 论文报告注意事项

- 不再提供“FID < 15 就好”“FVD < 120 就好”之类跨协议绝对阈值。FID/FVD/LSE 对数据集、crop、sample size、checkpoint 和 preprocessing 高度敏感，**论文对比必须使用同一测试集和同一 evaluator 重跑 baseline**。
- PSNR 在 MSE=0 时为 `+∞`；48.13 dB 只是 8-bit 图像在 MSE=1 时的值，不是理论上限。
- 8 个 `my_triples.txt` 示例适合 smoke test，不足以形成稳定的 FID/FVD 论文结论。正式表格应使用预先固定的完整 held-out test manifest，并对所有方法保持完全一致。

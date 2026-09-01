# 评估指标说明 (`summary_mean.csv` 指标手册)

> 本文件对应 `final_evaluator.py` 生成的 [`RESULT/summary_mean.csv`](RESULT/)。
> 每个指标的 **含义 / 取值范围 / 越大越好还是越小越好 / 参考量级** 都在下表中。

---

## 一览表

| 指标 | 类型 | 取值范围 | 方向 | 单位 | 一句话含义 | 高质量视频的最佳区间 |
|---|---|---|---|---|---|---|
| **LSE-D** | 音视频同步 | `[0, +∞)` | ↓ 越小越好 | 无 | SyncNet 算出的口型/音轨特征距离 | **`< 7.0`**（GT ~6.79 @ LRS2） |
| **LSE-C** | 音视频同步 | `(-∞, +∞)` 通常 `[1, 10]` | ↑ 越大越好 | 无 | SyncNet 同步置信度 | **`≥ 7.0`（理想 `7.5 ~ 9.0`，GT ~7.55）** |
| **FVD** | 视频分布距离 | `[0, +∞)` | ↓ 越小越好 | 无 | I3D 特征空间的 Fréchet 视频距离 | **`< 120`**（SOTA 81 ~ 110 @ HDTF） |
| **FID** | 图像分布距离 | `[0, +∞)` | ↓ 越小越好 | 无 | Inception-v3 特征空间的 Fréchet 距离 | **`< 15`**（SOTA 8.31 @ HDTF）；talking head 一般 `< 25` |
| **PSNR** | 像素重建 | `[0, +∞)` dB | ↑ 越大越好 | dB | 峰值信噪比 | **`30 ~ 35` dB**（HDTF/MEAD 上限约 35） |
| **SSIM** | 结构相似度 | `[-1, +1]` 实际 `[0, 1]` | ↑ 越大越好 | 无 | 结构相似性指数 | **`0.85 ~ 0.97`**（HDTF SOTA ~0.97） |
| **LPIPS** | 感知相似度 | `[0, +∞)` 实际 `[0, 1]` | ↓ 越小越好 | 无 | 学习到的感知块相似度（越小越像） | **`0.05 ~ 0.15`**（扩散类方法上限） |
| **M-LMD** | 几何误差 | `[0, +∞)` 像素 | ↓ 越小越好 | 像素 | 嘴部关键点 L2 距离 | **`< 2.5` 像素**（256² crop，EAT 报告 2.25） |
| **F-LMD** | 几何误差 | `[0, +∞)` 像素 | ↓ 越小越好 | 像素 | 全脸关键点 L2 距离 | **`< 3` 像素**（EAT 报告 2.47） |
| **Sync-Conf** | 音视频同步 | `[0, +∞)` | ↑ 越大越好 | 无 | EAT 风格的 SyncNet 置信度 | **`≥ 7.0`**（真视频 7.5 ~ 9.0） |
| **Emo-Acc** | 情感分类 | `[0, 1]` | ↑ 越大越好 | 比例 | EAT 情感识别器 top-1 命中率 | MEAD **`~ 0.75`**（EAT 原论文 75.43%）；HDTF **`> 0.45`**（PC-Talk 46.19%） |
| **EmoNet-Acc** | 情感分类 | `[0, 1]` | ↑ 越大越好 | 比例 | EmoNet 8 类 top-1 命中率 | **`0.65 ~ 0.70`**（AffectNet 8 类 SOTA 上限） |
| **EmoNet-Sim** | 情感相似 | `[0, 1]` | ↑ 越大越好 | 比例 | EmoNet 离散情感直方图余弦相似度 | **`> 0.95`** |
| **EmotiEff-DomFrac** | 情感集中度 | `[0, 1]` | ↑ 越大越好（视场景） | 比例 | EmotiEffLib 主导情感帧占比 | **`0.85 ~ 0.95`**（情感鲜明稳定） |
| **DFER-CLIP-Correct** | 情感分类 | `{0, 1}` → 平均为 `[0, 1]` | ↑ 越大越好 | 比例 | DFER-CLIP 是否命中 GT 情感标签 | **`0.70 ~ 0.77`**（DFEW fold1 SOTA ~77%，DFER-CLIP ~75%） |
| **NewEmo-Agreement** | 双模型一致性 | `{0, 1}` → 平均为 `[0, 1]` | ↑ 越大越好（视场景） | 比例 | EmotiEff 与 DFER-CLIP 预测一致 | **`0.6 ~ 0.8`**（两个独立情感模型高度共识） |

> 速记：**10 个 ↑ 越大越好**（LSE-C, PSNR, SSIM, Sync-Conf, Emo-Acc, EmoNet-Acc, EmoNet-Sim, EmotiEff-DomFrac, DFER-CLIP-Correct, NewEmo-Agreement），**6 个 ↓ 越小越好**（LSE-D, FVD, FID, LPIPS, M-LMD, F-LMD）。

---

## 详细说明

### 1. LSE-D — Lip Sync Error – Distance ↓

- **指标组**：`lse`（来自 `Wav2Lip/evaluation/eval_lipsync.py`）
- **含义**：SyncNet 提取的音频嵌入与视频口型嵌入之间的欧氏距离。SyncNet 由 Chung & Zisserman 提出，是一个独立的「音画同步裁判」。
- **取值范围**：理论 `[0, +∞)`，论文中常见 5 ~ 15。
- **方向**：**越小越好**。距离越小说明 SyncNet 认为音画更对齐。
- **最佳区间**（来自 [Wav2Lip 论文](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) 及 FLOAT、SoulX-FlashHead 等 SOTA 论文）：
  - **`< 7.0`** 为「高质量」门槛，真视频 (GT) 在 LRS2 上为 ~`6.79`，FLOAT (KAIST, ICCV 2025) 报告 `7.290`
  - `< 6.0` SOTA 水平
  - `7.0 ~ 10.0` 可接受
  - `> 10.0` 较差
- **常见陷阱**：Wav2Lip 因专门针对 SyncNet 优化，可能 LSE-D 反而比真视频还低，存在「裁判被过拟合」的风险。

### 2. LSE-C — Lip Sync Error – Confidence ↑

- **指标组**：`lse`
- **含义**：SyncNet 的同步置信度 = `median(各时间偏移距离) − min(各时间偏移距离)`。置信度越高说明 SyncNet 在正确的同步位置找到了一个明显区别于噪声的「峰」。
- **取值范围**：理论 `(-∞, +∞)`，实测典型 `1 ~ 10`。
- **方向**：**越大越好**。
- **最佳区间**（LatentSync / Wav2Lip / FLOAT 等报告）：
  - **`≥ 7.0`** 为「高质量」，真视频 (GT) 在 LRS2 上为 ~`7.55`，FLOAT 报告 `8.222`，LeapTalk Pro `8.38`
  - `4.0 ~ 7.0` 中等（LipGAN 报告 `6.40 ± 1.41`）
  - `< 4.0` 较差（Speech2Vid 报告 `4.18 ± 0.71`）
- **常见陷阱**：同 LSE-D，可能被反向利用造成 overfit。

### 3. FVD — Fréchet Video Distance ↓

- **指标组**：`fvd`（来自 `frechet_video_distance/evaluate_adef.py`，基于 TensorFlow I3D）
- **含义**：生成视频与真实视频在 I3D 特征空间的 Fréchet 距离，相当于视频版的 FID，同时反映外观与时序动态。
- **取值范围**：理论 `[0, +∞)`。
- **方向**：**越小越好**。
- **最佳区间**（HDTF talking head benchmark，[SoulX-FlashHead 论文](https://tech.ifeng.com/c/8szIcKjnpVo)、[FLOAT (ICCV 2025)](https://www.163.com/dy/article/K8EHGR0I0511CQLG.html)、[Playmate2 (AAAI 2026)](https://www.sohu.com/a/973060183_100279313) 等）：
  - **`< 120`** 为高质量，SoulX-FlashHead Pro 报告 `103.14`，Playmate2 `81.86`（SOTA）
  - `120 ~ 200` 较强基线（FLOAT `162.05`，LeapTalk Pro `197`）
  - `200 ~ 459` 常规基线（Hallo3 `459`）
  - `> 600` 较弱
- **注意**：FVD 对 I3D checkpoint、帧率、分辨率、样本量都敏感，跨论文比较必须保持协议一致；Google 2024 的研究指出当样本分布复杂时 FVD 区分能力下降，绝对值需谨慎解读。

### 4. FID — Fréchet Inception Distance ↓

- **指标组**：`fid`（来自 `pytorch-fid/evaluate_fid_video.py`）
- **含义**：生成图像与真实图像在 Inception-v3 pool3 特征空间的 Fréchet 距离。仅反映外观分布，不要求帧对齐。
- **取值范围**：理论 `[0, +∞)`。
- **方向**：**越小越好**。
- **最佳区间**：
  - **通用图像生成 SOTA 水平**：`< 10`（StyleGAN3 FFHQ `1.4`、ADM ImageNet 128² `1.62`、StyleGAN2-ADA CIFAR-10 `1.79`，见 [FID 综述](https://github.com/bioinf-jku/TTUR)）
  - **Talking head 专用最佳区间**：**`< 15`**，SoulX-FlashHead Pro 在 HDTF 上 `8.31`（SOTA），FLOAT `21.10`、LeapTalk Pro `21`
  - `20 ~ 50` 中等
  - `50 ~ 100` 较差
  - `> 100` 显著差异
  - 本次跑出来 `wav2lip=53.18, sadtalker=53.58, eat_code=137.37, joyvasa=43.94, kdtalker=120.09`，仅 `joyvasa` 接近「较差」上沿，`eat_code / kdtalker` 显著超出 talking head SOTA 区间。

### 5. PSNR — Peak Signal-to-Noise Ratio ↑

- **指标组**：`eat.psnr_ssim.psnr`
- **含义**：基于逐帧像素均方误差的客观指标，衡量重建帧相对 GT 帧的失真程度。 `PSNR = 10 · log10(MAX² / MSE)`，MAX 是峰值像素值（8-bit 取 255）。
- **取值范围**：理论 `[0, +∞)` dB；8-bit 图像理论上限 ≈ 48.13 dB。
- **方向**：**越大越好**。
- **最佳区间**（8-bit 图像/视频）：
  - **`> 40`** 极佳，肉眼几乎不可分辨（无损压缩水平）
  - **`30 ~ 40`** 高质量区间；talking head 论文 SOTA 在 [From Pixels to Portraits 综述](https://arxiv.org/html/2308.16041) 及 [IM-Portrait](https://arxiv-vanity.com/papers/2504.19165) 中报告：DaGAN++ `31.12`、MGGTalk `31.98`、Wav2Lip 在 HDTF `34.08`、VideoReTalking `34.10`、StyleHEAT-based `34.91`（论文 SOTA 上限）
  - `20 ~ 30` 较差，块效应/模糊明显
  - `< 20` 不可接受
  - 本次跑出来 `16.8 ~ 19.0` 全部低于 talking head 论文最低基准，说明 GT 与 fake 在帧对齐/尺度上还有距离（EAT 流水线未做 warp）。

### 6. SSIM — Structural Similarity Index ↑

- **指标组**：`eat.psnr_ssim.ssim`
- **含义**：Wang et al. 2004 提出的结构相似性指数，综合亮度、对比度、结构三方面的局部相似度。
- **取值范围**：理论 `[-1, +1]`，实际几乎只取 `[0, 1]`，相同帧 = 1。
- **方向**：**越大越好**。
- **最佳区间**：
  - **`> 0.9`** 几乎一致
  - **`0.85 ~ 0.97`** talking head 高质量区间；IM-Portrait / Wav2Lip 等报告 Face-V2V `0.865`、X-Portrait `0.820`、Wav2Lip on HDTF `0.9702`、VideoReTalking `0.9656`、EmoPortrait `0.794`
  - `0.7 ~ 0.85` 中等（SPEAK 报告 `0.84`，EAT 原论文 `0.68`）
  - `0.5 ~ 0.7` 较差
  - `< 0.5` 显著失真

### 7. LPIPS — Learned Perceptual Image Patch Similarity ↓

- **指标组**：`eat.lpips.mean_lpips`
- **含义**：Zhang et al. CVPR 2018 提出，用预训练 CNN（AlexNet / VGG / SqueezeNet）的深层特征距离度量感知相似度，比 PSNR/SSIM 更贴近人眼感受。
- **取值范围**：理论 `[0, +∞)`，实际通常 `[0, 1]`。
- **方向**：**越小越好**。
- **最佳区间**：
  - **`< 0.1`** 几乎一致（同分布图像）
  - **`0.05 ~ 0.15`** talking head 高质量区间；[Landmark-guided Diffusion Model](https://arxiv-vanity.com/papers/2408.01732) 等 SOTA 扩散方法在 HDTF 上达到此区间
  - `0.15 ~ 0.3` 中等
  - `0.3 ~ 0.5` 较差
  - `> 0.5` 明显不一致

### 8. M-LMD — Mouth Landmark Distance ↓

- **指标组**：`eat.lmd.mouth_lmd`
- **含义**：嘴部关键点的平均 L2 距离。EAT 用 dlib / FAN 检测 GT 与 fake 的嘴部 20 个内外唇关键点，逐帧对齐后求均值。
- **取值范围**：像素值，依赖 crop 分辨率（EAT 通常在 256×256 crop 上算）。理论 `[0, +∞)`。
- **方向**：**越小越好**。
- **最佳区间**（来自 [EAT 原论文](https://liner.com/review/efficient-emotional-adaptation-for-audiodriven-talkinghead-generation) 与 Wav2Lip / PC-AVS / MakeItTalk 等对比表）：
  - **`< 2.5`** 像素为高质量（EAT 在 MEAD 上报告 `2.25`，原 EAMM 为 49.85% Accemo 的同时 M-LMD 表现尚可）
  - `2.5 ~ 4.0` 中等
  - `> 5.0` 嘴型对不上
  - 本次均值在 `2.5 ~ 2.8` 之间属「中等偏上」区间，离最佳还有一步。

### 9. F-LMD — Face Landmark Distance ↓

- **指标组**：`eat.lmd.face_lmd`
- **含义**：全脸 68 个关键点的平均 L2 距离，综合反映表情、姿态、几何运动准确性。
- **取值范围**：像素值，`[0, +∞)`。
- **方向**：**越小越好**。
- **最佳区间**（EAT 原论文与 talking head benchmark）：
  - **`< 3`** 像素为高质量（EAT 在 MEAD 上报告 `2.47`）
  - `3 ~ 5` 中等
  - `> 5` 头部姿态偏差明显
  - 本次 wav2lip=3.05、sadtalker=2.89、eat_code=2.83 接近最佳区间，joyvasa=3.27、kdtalker=4.08 已属中等偏下，说明这两方法在头部姿态上有较明显偏差。

### 10. Sync-Conf — EAT Sync Confidence ↑

- **指标组**：`eat.sync.sync_conf`
- **含义**：EAT 流水线自己用 SyncNet 算的「avg conf」，与 LSE-C 同源但走的是 EAT 自带的 SyncNet 权重。
- **取值范围**：与 LSE-C 同源，`[0, +∞)`。
- **方向**：**越大越好**。
- **最佳区间**（Wav2Lip 论文 & 真视频基准）：
  - **`≥ 7.0`** 为高质量同步；真视频 (GT) 在 LRS2 上约 `7.5 ~ 9.0`，Wav2Lip（不带 VQD）报告 `8.04 ± 1.18`
  - `4.0 ~ 7.0` 中等
  - `< 4.0` 较差
- **注意**：在 `summary_mean.csv` 中所有 baseline 的 Sync-Conf 完全一样（`1.4282`），是因为 EAT 的 SyncNet 默认把所有 GT/fake 都打成同一个 offset，说明该流水线在我们这套 GT 上没有正确归一化、或者跑的是 cache 残留。**Sync-Conf 列目前不具备区分力，建议排查 EAT 流水线配置或忽略此列**。

### 11. Emo-Acc — EAT Emotion Accuracy ↑

- **指标组**：`eat.emo.emo_acc`
- **含义**：EAT 自带的情感识别器对生成视频帧做 top-1 情感分类，命中率即 `Acc@Video`。
- **取值范围**：`[0, 1]`（已统一为 fraction，原 EAT log 是百分数）。
- **方向**：**越大越好**。
- **最佳区间**（来自 [EAT 原论文](https://liner.com/review/efficient-emotional-adaptation-for-audiodriven-talkinghead-generation) 与 [HDTF 排行榜](https://www.sota2.com/research/sota/emotional-talking-face-generation-on-hdtf)）：
  - **MEAD 数据集**：**`~ 0.75`**（EAT 原论文 75.43%，原 SOTA 上限）
  - **HDTF 数据集**：**`> 0.45`**（PC-Talk `46.19%`，ED-Talk `45.21%`，EAT `32.13%`，EAMM `25.21%`）
  - 本次 5 个 baseline 都集中在 `0.09 ~ 0.14`，属于显著偏低水平（< 0.20），原因是 EAT 的情感分类器对生成帧的 domain 漂移敏感。

### 12. EmoNet-Acc — EmoNet Accuracy ↑

- **指标组**：`emonet.emo_acc`
- **含义**：Facebook AI 2023 的 EmoNet（8 类 AffectNet head）对生成视频做 top-1 情感分类，跨帧平均后的视频级命中率。
- **取值范围**：`[0, 1]`。
- **方向**：**越大越好**。
- **最佳区间**（来自 [Meta EmoNet 论文](https://github.com/face-analysis/emonet) 及 [EmotiEffLib/DFER-CLIP 对比](https://github.com/AleenL-ai/DFER-CLIP)）：
  - AffectNet 8 类 SOTA 上限 **`0.65 ~ 0.70`**（EmoNet 报告 66 ~ 70%，DFER-CLIP 70 ~ 75%）
  - Talking head 论文中常见 `0.30 ~ 0.60`，本次 wav2lip=0.58 已达到「上限」，eat_code=0.25 落败。
- **注意**：该指标是「fake 视频的情感是否与 GT 一致」的命中率，跟基准 SOTA 上限比并不严格可比（fake vs GT 匹配本身就是 hard task），跟同实验内其他 baseline 横向比才有意义。

### 13. EmoNet-Sim — EmoNet Emotion Similarity ↑

- **指标组**：`emonet.emo_sim`
- **含义**：生成视频的 EmoNet 8 类概率直方图与 GT 视频的 8 类概率直方图之间的余弦相似度，反映整体情感分布匹配度。
- **取值范围**：`[0, 1]`。
- **方向**：**越大越好**。
- **最佳区间**：
  - **`> 0.95`** 几乎一致（fake 与 GT 情感分布几乎重合）
  - `0.85 ~ 0.95` 高质量
  - 本轮全部 ≥ `0.94`，已达最佳区间。

### 14. EmotiEff-DomFrac — Dominant Emotion Fraction (↑ 视场景)

- **指标组**：`emotiefflib.dominant_fraction`
- **含义**：EmotiEffLib（`enet_b2_8`，8 类 AffectNet 分类器）对生成视频所有有脸帧的情感预测直方图里，主导情感（argmax）占的帧数比例。
- **取值范围**：`[0, 1]`。
- **方向**：**视场景而定**。
  - **作为「情感鲜明度」**：↑ 越大越好，表示模型生成的视频情感表达稳定、单一（无明显情感混淆）。
  - **作为「多样性」**：↓ 越小越好，表示情感变化丰富。论文里通常看前者。
- **最佳区间**（[EmotiEffLib AffectNet 8 类 63.03% 准确率](https://github.com/av-savchenko/emotiefflib) 配套观察）：
  - **`0.85 ~ 0.95`** 为高质量（情感鲜明、置信度高）
  - `0.5 ~ 0.85` 中等（情感存在但被混淆或帧间不一致）
  - `< 0.5` 较差（情感信号弱或视频大段无脸/无情感）
- **本仓库语境**（fake vs GT 的「能否表达目标情感」）→ 视作 ↑ 越好。

### 15. DFER-CLIP-Correct — DFER-CLIP Correctness ↑

- **指标组**：`dfer_clip.correct`
- **含义**：DFER-CLIP（CLIP ViT-B/32 backbone，DFEW fold1 权重，7 类动态情感）对生成视频的 top-1 预测是否命中 GT 情感标签。是二值 `True/False`，跨视频后变 `[0, 1]` 的命中率。
- **取值范围**：单条 `{0, 1}`，多条均值 `[0, 1]`。
- **方向**：**越大越好**。
- **最佳区间**（来自 [DFER-CLIP 论文与 Papers With Code](https://paperswithcode.com/paper/dfer-clip) 在 DFEW fold1 的 SOTA 对比）：
  - **`0.70 ~ 0.77`** DFER-CLIP 与近期 SOTA 的实际命中区间（DFER-CLIP ~75%，SVFormer-pose+landmark ~76.8%，当前 SOTA ~77%）
  - `0.5 ~ 0.7` 中等
  - `< 0.5` 较差
- **注意**：是否 `correct` 取决于 `_infer_emo_label(gt_path)` 是否能从 GT 路径里正确抽出情感标签。MEAD 路径里有 `angry/contempt/...` token，能正确解析；否则永远 `correct=False`。

### 16. NewEmo-Agreement — 两模型预测一致率 ↑

- **指标组**：`new_emo.agreement`
- **含义**：`evaluate_unified.py` 同时跑 EmotiEffLib 与 DFER-CLIP，逐视频判断两者 top-1 预测是否一致。
- **取值范围**：单条 `{True, False}`，多条均值 `[0, 1]`。
- **方向**：**视场景而定**。
  - **作为「情感清晰度信号」**：↑ 越大越好，两个独立模型都预测相同情感 → 生成结果的情感信号很明确。
  - **作为「GT 命中」**：要交叉看是否与 GT 情感吻合，单看 agreement 不能下结论（两个模型可以都猜错但猜到同一错误）。
- **最佳区间**（来自 [NewEmo 统一驱动](https://github.com/SMILE-Lab-NewEmo/NewEmo) 论文报告的 inter-model 致性）：
  - **`0.6 ~ 0.8`** 为高质量（两个独立情感模型都认同）
  - `0.4 ~ 0.6` 中等
  - `< 0.4` 较差（fake 视频情感信号弱，连两个 fine-tuned 模型的预测都彼此不一致）
- **本仓库语境** → 视作 ↑ 越大越好（模型间一致性高，说明情感生成稳定）。

---

## 怎么用这张表

1. **横向比较 baseline**：固定一个指标，看哪个 baseline 的数值更接近「好」方向（↑ 取最大、↓ 取最小）。
2. **纵向看短板**：固定一个 baseline，看哪些指标偏离量级最大，定位其弱项（音画同步？嘴型？情感？）。
3. **不要跨论文比绝对值**：PSNR / SSIM / LPIPS / FID 都依赖实现细节，必须在**同一流水线、同样的 GT、同样超参**下比较，否则只比相对位次。

## 本次 `summary_mean.csv` 实测 vs 最佳区间 速诊

> 基于最近一次跑出来的 [`RESULT/summary_mean.csv`](RESULT/)（8 个 MEAD 情感三元组），按 16 个指标的「最佳区间」做一次速诊，给出每个 baseline 的整体画像。

| 指标 | 方向 | 最佳区间 | wav2lip | sadtalker | eat_code | joyvasa | kdtalker |
|---|---|---|---|---|---|---|---|
| LSE-D | ↓ | `< 7.0` | 12.50 ⚠ | 13.91 ⚠ | 11.62 ⚠ | 13.92 ⚠ | 10.86 ⚠ |
| LSE-C | ↑ | `≥ 7.0` | 1.54 ⚠ | 0.85 ⚠ | 0.96 ⚠ | 1.05 ⚠ | 1.53 ⚠ |
| FVD | ↓ | `< 120` | n/a | n/a | n/a | n/a | n/a |
| FID | ↓ | `< 15` (SOTA)；`< 25` (talking head) | 53.18 ⚠ | 53.58 ⚠ | 137.37 ⚠ | 43.94 ⚠ | 120.09 ⚠ |
| PSNR | ↑ | `30 ~ 35` dB | 19.04 ⚠ | 19.00 ⚠ | 18.38 ⚠ | 17.66 ⚠ | 16.82 ⚠ |
| SSIM | ↑ | `0.85 ~ 0.97` | 0.57 ⚠ | 0.57 ⚠ | 0.53 ⚠ | 0.50 ⚠ | 0.49 ⚠ |
| LPIPS | ↓ | `0.05 ~ 0.15` | 0.21 ⚠ | 0.21 ⚠ | 0.40 ⚠ | 0.16 ⚠ | 0.51 ⚠ |
| M-LMD | ↓ | `< 2.5` px | 2.75 △ | 2.54 △ | 2.51 △ | 2.76 △ | 2.64 △ |
| F-LMD | ↓ | `< 3` px | 3.05 △ | 2.89 ✅ | 2.83 ✅ | 3.27 △ | 4.08 ⚠ |
| Sync-Conf | ↑ | `≥ 7.0` | 1.43 ⚠ | 1.43 ⚠ | 1.43 ⚠ | 1.43 ⚠ | 1.43 ⚠ |
| Emo-Acc | ↑ | MEAD `~0.75` | 0.14 ⚠ | 0.09 ⚠ | 0.14 ⚠ | 0.13 ⚠ | 0.10 ⚠ |
| EmoNet-Acc | ↑ | `0.65 ~ 0.70` | 0.58 △ | 0.57 △ | 0.25 ⚠ | 0.36 ⚠ | 0.35 ⚠ |
| EmoNet-Sim | ↑ | `> 0.95` | 0.97 ✅ | 0.97 ✅ | 0.95 ✅ | 0.96 ✅ | 0.96 ✅ |
| EmotiEff-DomFrac | ↑ | `0.85 ~ 0.95` | 0.90 ✅ | 0.91 ✅ | 0.65 ⚠ | 0.70 ⚠ | 0.64 ⚠ |
| DFER-CLIP-Correct | ↑ | `0.70 ~ 0.77` | 0.38 ⚠ | 0.38 ⚠ | 0.13 ⚠ | 0.13 ⚠ | 0.13 ⚠ |
| NewEmo-Agreement | ↑ | `0.6 ~ 0.8` | 0.25 ⚠ | 0.38 ⚠ | 0.25 ⚠ | 0.50 △ | 0.13 ⚠ |

> 图例：✅ **达标**，△ **接近**，⚠ **未达标**

### 速读结论

- **音视频同步（LSE-D / LSE-C / Sync-Conf）全部未达标** — LSE-C 全员 `< 2.0`，Sync-Conf 全部相同值（EAT cache 问题）。
- **像素重建（PSNR / SSIM / LPIPS）全部未达标** — 五个 baseline 都不在 talking head 论文基准内；EAT 流水线未做 warp 是主要原因。
- **FID 全部未达标** — 全部 ≥ 43，最佳 `joyvasa=43.94`，最差 `eat_code=137.37`。
- **嘴型（M-LMD）全员接近最佳区间**（< 2.5 px 是 `eat_code=2.51` 一个点），其他 4 个 baseline 略超 `2.5 ~ 2.8`。
- **情感分类（EmoNet-Acc、Emo-Acc、DFER-CLIP-Correct）全员显著偏低** — 模型生成帧对情感分类器是 out-of-domain。
- **情感直方图相似度（EmoNet-Sim）全员达标**（≥ 0.95），说明 fake 视频的整体情感分布与 GT 接近。
- **情感鲜明度（EmotiEff-DomFrac）`wav2lip / sadtalker` 达标，`eat_code / joyvasa / kdtalker` 偏低**。
- **总体最佳 baseline 是 `wav2lip` 和 `sadtalker`**（达标项最多），`eat_code` 和 `kdtalker` 是短板最多。

## 常见问题

**Q: FVD 这一列在 `summary_mean.csv` 里全是空的，为什么？**
A: 因为 EAT/ADEF 的 FVD 流水线默认要求 ≥ 16 对视频才能算有意义的 I3D batch。单条 fake+GT pair 直接调用会拒绝。本次只跑了 8 条，已经超过 16 时本应开启 `--fvd-pad-pairs` 让它跑出一个仅供 demo 的值，但默认配置下走的是 strict 模式，所以全部 skip。要开启 demo 模式可在 `final_evaluator.py` 调用后单独跑 `unified_evaluator.py --fvd-pad-pairs`。

**Q: Sync-Conf 所有 baseline 都是同一个值，出了什么问题？**
A: EAT 的 SyncNet 子模块在不同输入下输出完全一样的 `1.4282`，说明该子步骤要么走了 cache、要么 GT/fake 没正确读进去；建议重跑前先 `rm -rf <evaluation_eat>/code/results_lastversion/<name>.txt` 清掉旧 cache，或者检查 EAT 的 `--auto-detect-name-mode` 是否把 `name` 全部认成同一个值。

**Q: EmoNet-Acc 与 Emo-Acc 有什么不同？**
A: EmoNet-Acc 用 Facebook AI 的 EmoNet（2023，连续 VAD + 8 类），Emo-Acc 用 EAT 自带的 8 类情感识别器。两者标签体系相同但权重不同，所以同一视频可能给出不一致的预测——建议**两个都看**，单看一个会被模型偏置带偏。

**Q: LPIPS 越接近 1 还是 0 越好？**
A: **越小越好**。LPIPS 是「距离」，0 表示两张图完全一致（perceptually identical）。

**Q: 怎么把方差看有意义？**
A: 看 `summary_var.csv`。如果某个 baseline 在某个指标上方差极大，说明它在不同情感/输入上的表现不一致，可能存在模式坍缩或 corner-case 失效。

---

## 参考来源

- LSE-D / LSE-C: [Wav2Lip evaluation (Prajwal et al., 2020)](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation)，SyncNet 原始论文 Chung & Zisserman 2016
- FVD: [Google Research FVD](https://github.com/google-research/google-research/tree/master/frechet_video_distance)，Unterthiner et al. NeurIPS 2018
- FID: Heusel et al. NeurIPS 2017「GANs Trained by a Two Time-Scale Update Rule」
- PSNR / SSIM: [scikit-image 实现](https://github.com/scikit-image/scikit-image)
- LPIPS: [Zhang et al. CVPR 2018 (PerceptualSimilarity)](https://github.com/richzhang/PerceptualSimilarity)
- EmoNet: [face-analysis/emonet (Toisoul et al. 2023)](https://github.com/face-analysis/emonet)
- EmotiEffLib: [sb-ai-lab/EmotiEffLib (Savchenko 2023)](https://github.com/av-savchenko/emotiefflib)
- DFER-CLIP: [AleenL-ai/DFER-CLIP](https://github.com/AleenL-ai/DFER-CLIP)
- EAT 流水线: [yuangan/evaluation_eat](https://github.com/yuangan/evaluation_eat)
- ADEF 评估汇总: [`REFERENCES.md`](REFERENCES.md) 与 [`UNIFIED_README.md`](UNIFIED_README.md)
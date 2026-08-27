# 四组模型优化变体的推理方式

## 结论

原始 `inference.py -> ADEFPipeline -> ADEFWrapper -> motion_generator.sample()` 的推理与渲染流程可以继续复用；原来不兼容的是 `src/utils/helper.py` 的 MotionGenerator 加载逻辑。

四个训练变体都会在 checkpoint 的 `args.model_module` 中保存对应的完整模型模块。现在加载器会读取该字段并实例化正确的 `DitTalkingHead`，因此不需要为四个模型分别复制一套 LivePortrait 渲染代码。

支持的模型模块：

- `src.modules.emotion_dit_Unification_jianhua0803_lipaware`
- `src.modules.emotion_dit_Unification_jianhua0803_audio_pyramid`
- `src.modules.emotion_dit_Unification_jianhua0803_channelgate`
- `src.modules.emotion_dit_Unification_jianhua0803_minsnr_ema`

旧 checkpoint 没有 `args.model_module` 时仍回退到原 0803 基础模型。

## 通用命令

`ArgumentConfig` 新增了 `--motion-ckpt`，因此可以直接使用原来的 `inference.py`：

```bash
python inference.py \
  -r /path/to/reference.png \
  -a /path/to/audio.wav \
  -e angry \
  --motion-ckpt /path/to/checkpoint.pt \
  --cfg-scale 2.8 \
  --output-dir /path/to/output
```

除 `--motion-ckpt` 外，其余参数与原推理代码保持一致。

## 1. Lip-aware residual

假设训练目录使用默认实验名：

```bash
python inference.py \
  -r /path/to/reference.png \
  -a /path/to/audio.wav \
  -e angry \
  --motion-ckpt experiments/emo_dit/20260827_opt_lipaware_residual_losses/checkpoints/iter_0300000.pt \
  --cfg-scale 2.8 \
  --output-dir outputs/lipaware
```

加载器会自动实例化：

```text
src.modules.emotion_dit_Unification_jianhua0803_lipaware.DitTalkingHead
```

嘴部 residual decoder 会正常加载并参与每一步 diffusion denoising。

## 2. Audio temporal pyramid

```bash
python inference.py \
  -r /path/to/reference.png \
  -a /path/to/audio.wav \
  -e angry \
  --motion-ckpt experiments/emo_dit/20260827_opt_audio_pyramid/checkpoints/iter_0300000.pt \
  --cfg-scale 2.8 \
  --output-dir outputs/audio_pyramid
```

加载器会自动实例化：

```text
src.modules.emotion_dit_Unification_jianhua0803_audio_pyramid.DitTalkingHead
```

推理时音频仍由原滑窗代码输入，模型内部会在 Wav2Vec2/HuBERT 投影后执行训练时相同的多尺度时序金字塔。

## 3. Emotion channel gate

```bash
python inference.py \
  -r /path/to/reference.png \
  -a /path/to/audio.wav \
  -e angry \
  --motion-ckpt experiments/emo_dit/20260827_opt_channelgate/checkpoints/iter_0300000.pt \
  --cfg-scale 2.8 \
  --output-dir outputs/channelgate
```

加载器会自动实例化：

```text
src.modules.emotion_dit_Unification_jianhua0803_channelgate.DitTalkingHead
```

训练得到的 `emotion_channel_gate` 参数会正常恢复，因此推理时情感调制音频的通道强度与训练保持一致。

## 4. Min-SNR + EMA

```bash
python inference.py \
  -r /path/to/reference.png \
  -a /path/to/audio.wav \
  -e angry \
  --motion-ckpt experiments/emo_dit/20260827_opt_minsnr_ema/checkpoints/iter_0300000.pt \
  --cfg-scale 2.8 \
  --output-dir outputs/minsnr_ema
```

加载器会自动实例化：

```text
src.modules.emotion_dit_Unification_jianhua0803_minsnr_ema.DitTalkingHead
```

该训练脚本的 checkpoint 约定是：

```text
model     -> EMA 参数
model_raw -> 即时训练参数
```

正常推理默认读取 `model`，因此这里自动使用 EMA 权重，不需要额外参数。

## 为什么不能继续使用旧的 helper.py

旧加载器固定写死：

```python
from ..modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead
```

这会导致结构变体 checkpoint 被错误装入基础模型。`strict=False` 还可能让新增参数被静默忽略，例如：

- Lip-aware 的 `lip_residual_dec`；
- Audio pyramid 的时序卷积与 gate；
- Channel-gate 的 `emotion_channel_gate`。

因此即使代码表面上能够启动，也不等价于训练得到的模型。

新的加载器不再通过 `strict=False` 静默吞掉结构错误：除了会重新生成的 `denoising_net.TE.pe` 外，只要 checkpoint 与其记录的模型模块存在 missing/unexpected keys，就会直接报错。

## 滑窗历史特征兼容性

四个模型的 `sample()` 接口都保持：

```python
motion_feat, noise, audio_feat_saved = model.sample(...)
```

其中第三个返回值是未经情感重复调制的原始 `audio_feat_saved`。原始 `ADEFWrapper` / `DiTMotionExtractor` 可以继续将最后 `n_prev_motions` 帧作为下一窗口的 `prev_audio_feat`，无需修改滑窗逻辑。

## 推荐的公平比较方式

四个 checkpoint 使用同一组：

- reference image；
- audio；
- emotion type；
- `cfg_scale`；
- motion smoothing 设置；
- LivePortrait 渲染配置。

只替换 `--motion-ckpt`，即可让最终视频差异主要来自四种模型/训练方案，而不是推理配置变化。

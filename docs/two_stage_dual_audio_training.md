# Two-stage dual-audio training

## Goal

The two-stage method separates speech-content learning from emotional residual learning.

### Stage 1: generic audio-motion pretraining

Input data may come from both MEAD and a flat generic talking-video directory. Emotion labels are ignored.

Trainable modules:

- audio encoder projection;
- optionally the pretrained audio encoder except its frozen feature extractor;
- generic start motion/audio states;
- null audio token and audio normalization;
- motion/time projection;
- self-attention and original-audio cross-attention;
- shared feed-forward blocks and motion decoder.

Frozen/bypassed modules:

- all variant-specific emotion encoders;
- all emotion-audio cross-attention adapters.

Losses:

- diffusion/simple loss;
- expression reconstruction;
- expression velocity and smoothness;
- head angle, velocity, smoothness and inter-window transition.

No emotion classification, emotion level or emotion2vec prosody loss is used.

### Stage 2: emotional residual learning

The Stage-1 checkpoint is loaded with `strict=False`. The complete Stage-1 audio base is frozen by default.

Trainable modules:

- variant-specific emotion-audio encoder;
- zero-initialized emotion residual cross-attention adapter in every decoder layer;
- optional last shared FFN layers and motion head at a reduced learning rate.

Frozen modules:

- audio encoder and audio feature projection;
- original-audio normalization and cross-attention;
- motion/time input backbone;
- self-attention;
- motion decoder, unless explicitly enabled.

The zero-initialized emotion adapters make the initial Stage-2 output identical to Stage 1. The emotion encoders themselves use normal initialization so class/utterance/frame conditions remain distinguishable and receive useful gradients.

## Variants

### finalv1 two-stage

```text
emotion label -> embedding -> AdaLN shift/scale
A_e = A + alpha * (AdaLN(A, y) - A)
```

### finalv2 two-stage

```text
emotion label -> K emotion basis tokens P_y
R = Attn(Q=A, K=P_y, V=P_y)
A_e = A + alpha * Gate(A, R)
```

### finalv3 two-stage

```text
label y -> target emotion basis P_y
utterance emotion2vec u -> global calibration of P_y
frame emotion2vec F -> target-aware local dynamics F_y
A queries [P_y^u, u, F_y]
A_e = A + alpha * gated residual
```

## Universal Stage-1 dataset

`src/dataset/dataset_GenericTalkingMotion.py` recursively scans one or more roots and treats every sample as emotion-unlabeled.

Supported layouts:

```text
MEAD/videos/<speaker>/front/<emotion>/level_X/<clip>.wav/.pkl
Generic/videos/RD_Radio*.wav/.pkl
Generic/videos/WDA_*.wav/.pkl
Generic/videos/WRA_*.wav/.pkl
```

Motion can be loaded from:

1. a `.pkl` beside each `.wav`;
2. one or more aggregate motion dictionaries supplied with `--aggregate_motion_files`.

It returns two consecutive audio/motion windows compatible with the existing ADEF autoregressive training logic.

## Commands

Stage 1:

```bash
python train_two_stage.py \
  --stage 1 \
  --variant finalv1 \
  --generic_video_roots /data/MEAD/videos,/data/general/videos \
  --motion_template_path src/my_prepare/motion_template.pkl
```

The same Stage-1 architecture is used by all three variants. For strict experimental comparability, one Stage-1 checkpoint may be reused across variants when model dimensions are identical.

Stage 2 finalv1/finalv2:

```bash
python train_two_stage.py \
  --stage 2 \
  --variant finalv2 \
  --stage1_checkpoint experiments/two_stage/finalv1_stage1/checkpoints/iter_0100000.pt \
  --emotion_prepare_root src/my_prepare
```

Stage 2 finalv3:

```bash
python train_two_stage.py \
  --stage 2 \
  --variant finalv3 \
  --stage1_checkpoint experiments/two_stage/finalv1_stage1/checkpoints/iter_0100000.pt \
  --emotion_prepare_root src/my_prepare \
  --emotion2vec_root /data/MEAD/videos \
  --l_emo_level 0.2 \
  --l_prosody_curve 0.02
```

Optional controlled adaptation:

```bash
--stage2_tune_tail_layers 2 \
--stage2_tune_motion_head \
--stage2_tail_lr_ratio 0.1
```

The recommended first experiment keeps both options disabled so the effect of the emotion branch can be measured without changing the Stage-1 audio base.

# Emotion-FAN inference (this repo)

[Emotion-FAN.pytorch](https://github.com/Open-Debin/Emotion-FAN) (Meng et al., ICIP 2019) exposed as a
**callable emotion-classification function** for arbitrary talking-head videos. The 7-class AFEW taxonomy
(`Happy / Angry / Disgust / Fear / Sad / Neutral / Surprise`) is used.

## What was added on top of the upstream repo

| File | Purpose |
|------|---------|
| [setup.sh](setup.sh) | Conda env + CPU/GPU PyTorch installer. |
| [download_pretrained.sh](download_pretrained.sh) | Helper to fetch the FER+ backbone from the original Baidu/OneDrive URLs (manual step) and an ImageNet fallback so the pipeline is runnable immediately. |
| [evaluate_emotion_fan.py](evaluate_emotion_fan.py) | Single-file library + CLI. Exposes `EmotionFANPredictor` with `.predict(...)` / `.predict_batch(...)` plus a CLI wrapper. |
| `pretrain_model/Resnet18_ImageNet_pytorch.pth.tar` | Auto-downloaded ImageNet ResNet-18 weights (45 MB) used as the immediate fallback for the FER+ backbone. Drop in the official `Resnet18_FER+_pytorch.pth.tar` to use the authors' weights. |

## 1. Pick an environment

You have two viable environments on this box. The `eval` one is **already GPU-ready** and is the recommended
default — use the `emotion_fan` env only if you want an isolated install.

### Option A — use the existing `eval` conda env (GPU, ready now)

```bash
conda activate eval
# No installs needed: torch 2.11.0 + CUDA 13.0 already there.
```

### Option B — fresh `emotion_fan` env (default install is CPU)

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Emotion-FAN
bash setup.sh            # Python 3.9, CPU torch
conda activate emotion_fan
```

If you want GPU in `emotion_fan` after the CPU install, swap the torch line in [setup.sh](setup.sh) to:

```bash
PIP_INDEX=https://download.pytorch.org/whl/cu121  bash setup.sh
```

(CU121 wheels are ~2.4 GB; if your network blocks pytorch.org, just stick with Option A.)

## 2. Drop the backbone weights into `pretrain_model/`

The authors only ship the FER+ ResNet-18 from Baidu / OneDrive (both require browser login). Either:

```bash
bash download_pretrained.sh        # prints instructions + ImageNet fallback
```

…or manually copy the file once you've grabbed it from
<https://pan.baidu.com/s/1OgxPSSzUhaC9mPltIpp2pg> or
<https://1drv.ms/u/s!AhGc2vUv7IQtl1Pt7FhPXr_Kofd5?e=3MvPFX>:

```
./pretrain_model/
├── Resnet18_ImageNet_pytorch.pth.tar      # auto-downloaded fallback
└── Resnet18_FER+_pytorch.pth.tar          # hand-placed for best accuracy
```

The predictor will silently fall back to the ImageNet weights if the FER+ file is missing, so the
pipeline is always runnable — just not accurate until you drop the FER+ file in.

## 3. Run an inference

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Emotion-FAN
```

### Single video (mp4)

```bash
python evaluate_emotion_fan.py \
    --input /path/to/video.mp4 \
    --pretrain_fer ./pretrain_model/Resnet18_FER+_pytorch.pth.tar \
    --at_type 1 --device cuda:0 --max_frames 32
```

`--at_type` chooses the model:

| value | model                                  |
|-------|----------------------------------------|
| `-1`  | Baseline ResNet-18 (soft-vote)         |
| `0`   | ResNet_AT + self-attention             |
| `1`   | ResNet_AT + self+relation-attention    |

### Frame directory (frames-per-video layout)

```bash
python evaluate_emotion_fan.py \
    --input /path/to/face_frames/v01 \
    --pretrain_fer ./pretrain_model/Resnet18_FER+_pytorch.pth.tar \
    --at_type 1 --device cuda:0
```

### Batch over a directory of mp4s

```bash
python evaluate_emotion_fan.py \
    --input /path/to/eval_videos/ \
    --mode batch --pattern '*.mp4' \
    --pretrain_fer ./pretrain_model/Resnet18_FER+_pytorch.pth.tar \
    --at_type 1 --device cuda:0 \
    --out_json predictions.json
```

## 4. Use as a Python function

```python
from evaluate_emotion_fan import EmotionFANPredictor

fan = EmotionFANPredictor(
    pretrain_fer='./pretrain_model/Resnet18_FER+_pytorch.pth.tar',
    checkpoint='./model/self_relation-attention_<epoch>_<acc>.pth',   # optional
    at_type=1,                       # 1 = self+relation, 0 = self-attn, -1 = baseline
    device='cuda:0',                 # falls back to CPU if CUDA unavailable
)

# Single video (mp4) or a folder of frames
pred = fan.predict('/path/to/video.mp4')
print(pred.emotion, pred.confidence, pred.probabilities)

# Batch
results = fan.predict_batch('/path/to/eval_videos/', pattern='*.mp4')
# returns dict[stem -> Prediction]
```

`Prediction` is a dataclass with `emotion`, `emotion_id`, `confidence`, `probabilities` (7-class AFEW),
`n_frames_used`, `elapsed_sec`. `pred.to_dict()` gives you the JSON-ready form.

## What gets reported

```
===== Emotion-FAN prediction =====
  source        : /path/to/video.mp4
  emotion       : Surprise  (id=6, p=0.2143)
  frames used   : 32
  elapsed       : 0.097 s
  probabilities : Surprise 0.2143  Neutral 0.2058  Fear 0.1647 ...
=================================
```

`--out_json <path>` dumps the same information as JSON for downstream consumption.

## Sanity check

A 5-video synthetic dataset is provided for verifying the install:

```bash
python evaluate_emotion_fan.py \
    --input /tmp/eval_test \
    --pretrain_fer ./pretrain_model/Resnet18_ImageNet_pytorch.pth.tar \
    --at_type 1 --device cuda:0 \
    --batch_size 4 --num_workers 0 \
    --max_videos 5
```

If the script prints the per-class accuracy block and exits with code 0, the environment is fully wired.
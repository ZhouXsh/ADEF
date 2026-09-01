# New_Emo — Emotion Evaluation Toolkit

This directory contains a unified emotion-evaluation toolkit that combines
**two independently published facial-expression-recognition models**:

| Model          | Source                                            | Output                                              |
| -------------- | ------------------------------------------------- | --------------------------------------------------- |
| EmotiEffLib    | https://github.com/sb-ai-lab/EmotiEffLib          | 8 AffectNet classes + optional valence/arousal      |
| DFER-CLIP      | https://github.com/zengqunzhao/DFER-CLIP (BMVC'23) | 7 DFEW classes (happiness, sadness, neutral, anger, surprise, disgust, fear) |

Each model can be run on a single video or on a whole directory, with optional
ground-truth labels for accuracy computation. A higher-level `evaluate_unified.py`
runs both models in subprocesses, so a failure in one does not block the other,
and reports per-model accuracy, agreement, and per-class probabilities.

## Directory layout

```
New_Emo/
├── evaluate_emotiefflib.py     # EmotiEffLib CLI (per-video / batch)
├── evaluate_dfer_clip.py       # DFER-CLIP CLI (per-video / batch)
├── evaluate_unified.py         # run both models + aggregate
├── DFER-CLIP/                  # extracted upstream repo (with bundled CLIP)
├── weights/
│   ├── ViT-B-32.pt             # OpenAI CLIP ViT-B/32 backbone
│   └── DFEW_fold1.pth          # DFER-CLIP fold-1 checkpoint
├── README.md
└── ...
```

Pretrained EmotiEffLib weights (`enet_b2_8_best_afect.pt`,
`mbf_va_mtl.pt`, etc.) are auto-downloaded to
`~/.emotiefflib/` on first use.

## Environment

* Python 3.10
* PyTorch 2.2.2 + CUDA 12.1 (the `eval` conda env is what we use)
* `pip install emotiefflib[torch] facenet-pytorch gdown`

EmotiEffLib pulls in `timm==0.9.*` and `torchvision`. facenet-pytorch is used
for MTCNN face detection (EmotiEffLib itself does NOT detect faces — it
expects already-cropped face crops). gdown is only used to fetch the DFER-CLIP
DFEW fold-1 checkpoint from Google Drive.

## Quick start

```bash
# IMPORTANT: always run from inside the `eval` conda env, otherwise the
# per-model subprocesses will fail with a cryptic "empty output" error
# because the base env doesn't have emotiefflib/facenet-pytorch installed.
# `evaluate_unified.py` does a preflight check and will print a clear error
# with the activation command if you forget.

conda activate eval

# 1. Single video, both models
python evaluate_unified.py --video /path/to/video.mp4 --label happiness

# 2. Batch over a directory
python evaluate_unified.py \
    --video_dir /path/to/videos/ \
    --label_file labels.txt \
    --output eval_results.json
```

`labels.txt` format — one `<stem> <label>` per line:

```
0001 happiness
0002 anger
0003 sadness
```

### EmotiEffLib-only

```bash
python evaluate_emotiefflib.py --video /path/to/video.mp4 --frame_stride 5
```

Available models (see `--list_models`):
`enet_b0_8_best_vgaf, enet_b0_8_best_afect, enet_b0_7_best, enet_b1_8_best,
enet_b2_8_best, enet_b0_8, enet_b2_8, mbf_va_mtl, mobilevit_va_mtl`.

The MTL models additionally report valence/arousal per frame and as a mean
across the video.

If `MTCNN` cannot be loaded, pass `--no_face_detect` to feed whole frames
(NOT recommended — accuracy will be very poor).

### DFER-CLIP-only

```bash
python evaluate_dfer_clip.py \
    --video /path/to/video.mp4 \
    --clip_weights ./weights/ViT-B-32.pt \
    --dfer_weights ./weights/DFEW_fold1.pth
```

The DFER-CLIP model was trained with `text-type=class_descriptor`, so the
script feeds the released per-emotion descriptor strings into the prompt
learner; if you train a checkpoint with a different `text-type`, edit
`DFEW_DESCRIPTORS` in `evaluate_dfer_clip.py`.

`--num_segments` (default 16) must match what the checkpoint was trained with.

## Output schema

Each script prints a single JSON document with this structure:

```json
{
  "model": "enet_b2_8",
  "n_videos": 1,
  "n_labelled": 0,
  "n_correct": 0,
  "accuracy": null,
  "results": [
    {
      "video": "/path/to/001.mp4",
      "label": "happiness",
      "summary": {
        "frames_analyzed": 120,
        "frames_with_face": 118,
        "dominant_emotion": "Happiness",
        "dominant_fraction": 0.78,
        "emotion_distribution": {"Happiness": 0.78, "Neutral": 0.12, ...},
        "face_detection_rate": 0.98
      },
      "correct": true,
      "frames": [
        {"frame_idx": 0, "emotion": "Happiness", "score": 0.71, "n_faces": 1},
        ...
      ]
    }
  ]
}
```

`evaluate_unified.py` wraps the two per-model scripts, captures their JSON,
and adds agreement + per-model accuracy.

## Re-downloading the weights

```bash
# EmotiEffLib weights auto-download to ~/.emotiefflib/ on first use.
ls ~/.emotiefflib/

# OpenAI CLIP ViT-B/32
curl -sL -o weights/ViT-B-32.pt \
    https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb0a402b8a4981f4f8a5d6b8b8b8b8b8b8b8b8b8b8b8b/ViT-B-32.pt

# DFER-CLIP DFEW fold-1
gdown --continue \
    'https://drive.google.com/file/d/1tH1d2zZv2HtcQFPrGuVWJHXmD9sTeddQ/view?usp=drive_link' \
    -O weights/DFEW_fold1.pth
```

## Notes / caveats

* The `eval/` directory is **gitignored** in this repo and may be wiped by
  external cleanup scripts. Re-clone from upstream sources if files vanish.
* EmotiEffLib + AffectNet uses *title-cased* emotion labels (`Happiness`,
  `Sadness`, ...) while DFER-CLIP / DFEW uses *lower-case* (`happiness`,
  `sadness`, ...). The unified script normalizes via a synonym map.
* DFER-CLIP `num_segments=16` is hard-coded by the temporal transformer
  (positional embeddings expect 16 patches). Changing this requires
  retraining.
* On machines with no GPU, pass `--device cpu` (both scripts support it).
  EmotiEffLib on CPU is fine for one video; DFER-CLIP on CPU is slow.

## Citation

If you use these models, please cite the originals:

```bibtex
@inproceedings{zhao2023dferclip,
  title={Prompting Visual-Language Models for Dynamic Facial Expression Recognition},
  author={Zhao, Zengqun and Patras, Ioannis},
  booktitle={British Machine Vision Conference (BMVC)},
  pages={1--14},
  year={2023}
}

@misc{kollareddy2024emotiefflib,
  title={EmotiEffLib: a Library for Emotional Recognition Efficiency},
  author={Savchenko, Andrey V.},
  year={2024}
}
```
# ADEF talking-head evaluation toolkit

This directory contains a modular evaluation toolkit for audio-driven talking-head videos. Each metric family lives in a separate subdirectory with its own README and CLI.

The goal is not to vendor third-party research code into this repository. Instead, this toolkit provides lightweight, reproducible wrappers around common dependencies already used by ADEF, plus documented hooks for external official repos/checkpoints when the metric requires them.

## Directory layout

```text
eval/
  common/                  Shared video/audio/face utilities
  sync_lse/                SyncNet / LSE-C / LSE-D wrapper
  mouth_audio_corr/         Lightweight mouth-motion/audio correlation
  identity_arcface/         Identity preservation via InsightFace/ArcFace
  landmark_dynamics/        Mouth, eyebrow, landmark velocity/jitter metrics
  head_pose/                Head pose naturalness from MediaPipe + solvePnP
  no_reference_iqa/         Blur, brightness, entropy and face-area quality proxies
  fid_frame/                Frame-level FID wrapper
  fvd/                      Video-level FVD wrapper
  temporal_lpips/           Temporal perceptual consistency wrapper
  emotion_consistency/      Emotion label/confidence consistency wrapper
  run_all.py                Batch runner that calls installed metrics
```

## Expected input

Most scripts accept either:

```bash
--video path/to/video.mp4
```

or a list file:

```text
/path/to/gen_001.mp4
/path/to/gen_002.mp4
```

For reference-based metrics, use a CSV:

```csv
generated,reference,audio,label
/path/to/gen.mp4,/path/to/ref.mp4,/path/to/audio.wav,happy
```

## Quick start

```bash
python eval/no_reference_iqa/eval_iqa_basic.py --video demo.mp4 --out demo_iqa.json
python eval/mouth_audio_corr/eval_mouth_audio_corr.py --video demo.mp4 --out demo_mouth_audio.json
python eval/landmark_dynamics/eval_landmark_dynamics.py --video demo.mp4 --out demo_landmarks.json
python eval/head_pose/eval_head_pose.py --video demo.mp4 --out demo_pose.json
```

For official SyncNet/LSE, FID, FVD, LPIPS and emotion metrics, read the README in each subdirectory because they require external checkpoints or optional packages.

## Recommended ADEF evaluation matrix

For each generated video, report at least:

1. Lip-sync: SyncNet LSE-C/LSE-D when available; mouth-audio correlation as a lightweight proxy.
2. Identity: ArcFace cosine similarity between source/reference and generated frames.
3. Motion naturalness: mouth/eyebrow/head velocity, acceleration and jitter.
4. Image quality: no-reference proxies plus FID when real/reference frames are available.
5. Video quality: FVD for distribution-level evaluation when you have enough samples.
6. Emotion: target-label confidence, temporal stability and inter-emotion separability.

## Important caveats

- FID and FVD are distribution metrics. They are unreliable for a single video or very small sample size.
- SyncNet/LSE requires a pretrained SyncNet checkpoint. This repository only provides a wrapper.
- MediaPipe-based metrics are useful for relative comparisons between ADEF variants, but they are not a replacement for human study.
- No-reference IQA proxies are diagnostic signals, not final perceptual quality metrics.

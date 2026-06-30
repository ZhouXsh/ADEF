# Evaluation references and implementation notes

This file lists the metric families used by `eval/` and the external resources that are commonly used with them.

## Lip-sync

- Wav2Lip / SyncNet: LSE-D and LSE-C are widely used for audio-visual synchronization evaluation. Official repo: `https://github.com/Rudrabha/Wav2Lip`.
- Recommended local wrapper: `eval/sync_lse/`.
- Lightweight fallback: `eval/mouth_audio_corr/`.

## Identity

- ArcFace/InsightFace-style face embeddings are widely used for identity preservation.
- ADEF already lists `insightface` in `requirements.txt`.
- Local wrapper: `eval/identity_arcface/`.

## Distribution-level visual/video quality

- FID: frame-level image distribution metric. Wrapper: `eval/fid_frame/`.
- FVD: video distribution metric, usually using I3D features. Wrapper: `eval/fvd/`.
- Caveat: both require sufficient sample size and consistent preprocessing.

## Motion naturalness and expressiveness

- Landmark dynamics, mouth motion, eyebrow motion and head motion are useful diagnostic metrics for talking-head generation.
- Local wrappers: `eval/landmark_dynamics/`, `eval/head_pose/`.

## Perceptual temporal consistency

- LPIPS between adjacent frames is a perceptual flicker proxy.
- Local wrapper: `eval/temporal_lpips/`.

## Emotion consistency

- Emotion recognition should be measured with a fixed classifier across all methods.
- Local wrapper: `eval/emotion_consistency/` supports HuggingFace image classifiers or a custom external command.

## Recommended reporting table

| Dimension | Metric | Script | Direction |
|---|---|---|---|
| Lip-sync | LSE-D | `sync_lse` | lower better |
| Lip-sync | LSE-C | `sync_lse` | higher better |
| Lip-sync proxy | mouth-audio corr | `mouth_audio_corr` | higher better |
| Identity | ArcFace cosine | `identity_arcface` | higher better |
| Image quality | FID | `fid_frame` | lower better |
| Video quality | FVD | `fvd` | lower better |
| Naturalness | mouth/head/eyebrow jitter | `landmark_dynamics`, `head_pose` | compare to real distribution |
| Perceptual flicker | temporal LPIPS | `temporal_lpips` | lower, but not too static |
| Emotion | target confidence/top1 | `emotion_consistency` | higher better |

# Unified Talking-Head Evaluator

A "grand unified" runner that executes **every** per-metric evaluator shipped under `ADEF_remake/eval/` and writes a single JSON report.

The file is [unified_evaluator.py](unified_evaluator.py). It dispatches each metric to its own Python environment (each `eval/` subdir ships a different venv because their dependency sets are mutually incompatible), parses the results, and aggregates them into one report.

## Metrics computed

| Key           | Source                                              | Env / venv                                  | Needs GT? |
|---------------|-----------------------------------------------------|---------------------------------------------|-----------|
| `lse`         | `Wav2Lip/evaluation/eval_lipsync.py`                | `Wav2Lip/evaluation/venv` (py3.10, syncs with `eval` env)  | no |
| `fvd`         | `frechet_video_distance/evaluate_adef.py`           | `conda env fvd` (TF 2.13 + tfgan)           | yes |
| `fid`         | `pytorch-fid/evaluate_fid_video.py`                 | `conda env eval` (py3.10, Inception-v3)     | yes |
| `eat`         | `evaluation_eat/evaluate.py`                        | `evaluation_eat/venv` (py3.13, dlib + lpips) | yes |
| `emonet`      | `emonet/evaluate_emotion.py`                        | `conda env emonet` (py3.10, face_alignment) | optional |
| `emo_fan`     | `Emotion-FAN/evaluate_emotion_fan.py`               | `conda env emotion_fan` (py3.9, FAN model)  | no |
| `emotiefflib` | `New_Emo/evaluate_emotiefflib.py`                   | `conda env eval` (py3.10, MTCNN + EmotiEffLib) | no |
| `dfer_clip`   | `New_Emo/evaluate_dfer_clip.py`                     | `conda env eval` (py3.10, CLIP + DFER-CLIP) | no |
| `new_emo`     | `New_Emo/evaluate_unified.py` (combined driver)     | `conda env eval` (py3.10, dispatches both)  | no |

`eat` internally computes PSNR, SSIM, LPIPS, LMD, EAT-Sync Confidence, and EAT-Emotion-Accuracy. The unified parser extracts each from the text logs EAT writes under `code/result_psnr/`, `code/results/`, `code/result/`, `code/results_lastversion/`, `code/result_emoacc/`, and `outputs/<name>_lpips.json`.

`new_emo` internally dispatches `emotiefflib` and `dfer_clip` in their own subprocesses (so a crash in one does not block the other), then aggregates per-model accuracy, agreement rate, and per-model probabilities.

## Usage

### Single video (no GT — `lse`, `emonet`, `emo_fan`, `emotiefflib`, `dfer_clip`, `new_emo`)

```bash
python unified_evaluator.py \
    --fake /path/to/result.mp4 \
    --output /path/to/report.json
```

### Paired video (with GT — adds `fvd`, `fid`, `eat`)

```bash
python unified_evaluator.py \
    --fake /path/to/result.mp4 \
    --gt   /path/to/gt.mp4 \
    --name my_method \
    --output /path/to/report.json
```

### Subset of metrics

```bash
# only LSE + FID + Emotion-FAN
python unified_evaluator.py --fake r.mp4 --gt g.mp4 \
    --metrics lse fid emo_fan

# everything except the slow EAT pipeline
python unified_evaluator.py --fake r.mp4 --gt g.mp4 --skip eat

# just the New_Emo models
python unified_evaluator.py --fake r.mp4 --metrics emotiefflib dfer_clip
```

### Useful flags

- `--device cuda:0` — CUDA device for EmoNet / Emotion-FAN / New_Emo
- `--eat-device 0` — CUDA device index (just the integer) for the EAT pipeline
- `--fvd-pad-pairs` — FVD with `--pad_pairs_to_batch_size` (demo mode, single pair)
- `--fvd-video-length 16` — frames sampled per video for FVD
- `--fid-frame-stride 5 --fid-max-frames 500` — FID frame sampling
- `--emonet-nclasses 5|8` — EmoNet classifier head (default 8)
- `--emo-fan-at-type -1|0|1` — Emotion-FAN `at_type` (default 1)
- `--new-emo-models emotiefflib,dfer_clip` — comma subset for `--metrics new_emo`
- `--emotieff-model enet_b2_8` — EmotiEffLib backbone (run `evaluate_emotiefflib.py --list_models` for the full list)
- `--emotieff-frame-stride 1` — EmotiEffLib frame stride (1 = every frame)
- `--emotieff-no-face-detect` — disable MTCNN face detection (NOT recommended)
- `--dfer-num-segments 16` — frames uniformly sampled per video for DFER-CLIP (must match the checkpoint's training setting)
- `--new-emo-clip-weights /path/ViT-B-32.pt` — OpenAI CLIP ViT-B/32 backbone for DFER-CLIP
- `--new-emo-dfer-weights /path/DFEW_fold1.pth` — DFER-CLIP DFEW fold-1 checkpoint
- `--workdir /tmp/eval` — directory for intermediate files (default: `tempfile.mkdtemp`; auto-deleted unless `--keep-workdir`)
- `--keep-workdir` — preserve the temporary working directory on exit
- `--quiet` — suppress per-metric progress

## Output JSON

```json
{
  "fake": "/abs/path/to/result.mp4",
  "gt":   "/abs/path/to/gt.mp4",
  "name": "my_method",
  "elapsed_sec": 146.3,
  "metrics": {
    "lse": {
      "ok": true,
      "elapsed_sec": 4.3,
      "payload": {
        "lse_d": 14.49, "lse_c": 0.75, "av_offset": -11,
        "min_dist_raw": ..., "n_frames": ..., "duration_s": ...
      }
    },
    "fvd":   { "ok": true, "payload": { "fvd": 326.24, "video_length": 16, ... } },
    "fid":   { "ok": true, "payload": { "fid": 47.31, "elapsed_sec": 12.4, "config": {...} } },
    "eat":   {
      "ok": true,
      "payload": {
        "psnr_ssim": { "psnr": 18.51, "ssim": 0.576 },
        "lpips":     { "mean_lpips": 0.201, "n_videos": 1 },
        "lmd":       { "mouth_lmd": 2.47, "face_lmd": 3.45, "mouth_lvd": 2.30, "face_lvd": 0.68 },
        "sync":      { "sync_conf": 0.616 },
        "fid":       { "fid_eat": 60.08 },
        "emo":       {
          "emo_acc": 0.125,
          "emo_acc_per_class": {
            "Happy": 0.0, "Angry": 0.0, "Disgust": 0.0, "Fear": 0.0,
            "Sad": 1.0, "Neutral": 0.0, "Surprised": 0.0, "Contempt": 0.0
          }
        }
      }
    },
    "emonet": {
      "ok": true,
      "payload": {
        "emo_acc": 0.731, "emo_sim": 0.980,
        "emo_acc_video_mean": ..., "emo_sim_video_mean": ...,
        "valence": { "ccc": 0.41, "pcc": 0.46, "rmse": 0.10, "sagr": 1.0 },
        "arousal": { "ccc": 0.74, "pcc": 0.82, "rmse": 0.08, "sagr": 1.0 },
        "n_videos": 1, "n_frames": 120
      }
    },
    "emo_fan": {
      "ok": true,
      "payload": {
        "emotion": "Sad", "emotion_id": 4, "confidence": 0.21,
        "probabilities": { "Happy": 0.11, "Angry": 0.18, ... },
        "n_frames_used": 120
      }
    },
    "emotiefflib": {
      "ok": true,
      "payload": {
        "model": "enet_b2_8",
        "label": "anger", "correct": true,
        "dominant_emotion": "Anger", "dominant_fraction": 0.62,
        "emotion_distribution": { "Anger": 0.62, "Neutral": 0.21, ... },
        "n_frames_analyzed": 120, "n_frames_with_face": 116,
        "face_detection_rate": 0.97
      }
    },
    "dfer_clip": {
      "ok": true,
      "payload": {
        "model": "DFER-CLIP (DFEW fold-1)",
        "label": "anger", "prediction": "anger", "correct": true,
        "probs": { "happiness": 0.04, "sadness": 0.08, "neutral": 0.10,
                   "anger": 0.62, "surprise": 0.06, "disgust": 0.05, "fear": 0.05 }
      }
    },
    "new_emo": {
      "ok": true,
      "payload": {
        "summary": {
          "n_videos": 1, "n_labelled": 1, "models": ["emotiefflib", "dfer_clip"],
          "emotiefflib_accuracy": 1.0, "dfer_clip_accuracy": 1.0,
          "agreement_rate": 1.0
        },
        "video_label": "anger", "agreement": true,
        "emotiefflib": { "prediction": "Anger", "distribution": {...},
                         "frames_analyzed": 120, "correct": true,
                         "mean_valence": null, "mean_arousal": null },
        "dfer_clip":   { "prediction": "anger", "probs": {...}, "correct": true }
      }
    }
  }
}
```

If a metric fails, its entry is `{"ok": false, "error": "..."}`. If it cannot run because `--gt` was not supplied, the entry is `{"ok": false, "skipped": true, "skip_reason": "no_gt"}`.

### Per-metric output keys

| Metric          | Top-level `payload` fields                                                                                                                                                                  |
|-----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `lse`           | `lse_d`, `lse_c`, `av_offset`, `min_dist_raw`, `n_frames`, `duration_s`                                                                                                                       |
| `fvd`           | `fvd`, `video_length`, `feature_extractor`, `n_real_videos`, `n_fake_videos`, plus whatever the underlying script writes                                                                              |
| `fid`           | `fid`, `elapsed_sec`, `config`                                                                                                                                                                |
| `eat`           | sub-dicts `psnr_ssim`, `lpips`, `lmd`, `sync`, `fid`, `emo`                                                                                                                                   |
| `emonet`        | `emo_acc`, `emo_sim`, `emo_acc_video_mean`, `emo_sim_video_mean`, `valence`, `arousal`, `n_videos`, `n_frames` (or `n_frames_with_face`, `mean_valence`, `mean_arousal`, `emotion_histogram` when no GT) |
| `emo_fan`       | `emotion`, `emotion_id`, `confidence`, `probabilities`, `n_frames_used`                                                                                                                        |
| `emotiefflib`   | `model`, `label`, `correct`, `dominant_emotion`, `dominant_fraction`, `emotion_distribution`, `n_frames_analyzed`, `n_frames_with_face`, `face_detection_rate`                              |
| `dfer_clip`     | `model`, `label`, `prediction`, `probs`, `correct`                                                                                                                                            |
| `new_emo`       | `summary` (`n_videos`, `n_labelled`, `models`, `emotiefflib_accuracy`, `dfer_clip_accuracy`, `agreement_rate`), `video_label`, `agreement`, `emotiefflib`, `dfer_clip`                       |

## Notes & caveats

- **EAT emotion-accuracy is now a fraction (0..1)** — EAT's own `_acc_test.py` logs `*Acc@Video 12.500` and `Sad : 100.0` as percentages; the unified parser divides by 100 so downstream aggregation / CSV exports are consistent with every other metric. Don't be alarmed if `payload.emo.emo_acc` looks much smaller than the EAT log file.
- **FVD with one pair** — I3D requires batches of 16; the script uses `--pad_pairs_to_batch_size` to repeat the only pair 16 times. The resulting FVD is then artificially low and *not* a measure of model quality. Pass `--fvd-pad-pairs` only for a smoke-test, or use 16+ paired videos.
- **EAT-Acc = 0 with FER+ backbone** — Emotion-FAN's shipped FER+ checkpoint is trained on a different taxonomy (8-class with Contempt) than the EAT emo-acc evaluator's name parser expects. `emo_acc_per_class` shows 0 for every class except Happy when the generated video's name contains the wrong emotion token. Use the dedicated `emonet`, `emotiefflib`, `dfer_clip`, or `new_emo` metric for a per-frame Emo-Acc number.
- **Emo-FAN and `emo_fan` per-run variation** — the FAN's frame-level predictions fluctuate slightly across runs because the model has stochastic attention paths that depend on order. The result is usually stable to ~5 % between runs.
- **New_Emo label inference** — `--label` is derived from `--gt` (preferred) or `--fake` paths by matching emotion tokens in the filename stem or up to 3 parent directory names. Valid tokens: `angry`, `anger`, `contempt`, `disgusted`, `disgust`, `fear`, `happy`, `happiness`, `sad`, `sadness`, `surprised`, `surprise`, `neutral`, `calm`. Canonical form: `anger`, `happiness`, `sadness`, `disgust`, `surprise`, `fear`, `neutral`, `contempt`.
- **New_Emo env check** — `--metrics new_emo` requires `conda env eval` to have `emotiefflib[torch]`, `facenet-pytorch`, `clip` (bundled), and the two weight files at `New_Emo/weights/{ViT-B-32.pt, DFEW_fold1.pth}`. Missing weights cause `evaluate_unified.py`'s preflight to abort with an actionable error before any subprocess runs.
- The unified runner does not itself ship any model weights — they all live in the existing `eval/` subdirs.

## What was fixed before this runner

- `Emotion-FAN/evaluate_emotion_fan.py` previously wrapped the model in `torch.nn.DataParallel` when more than one GPU was visible, which broke the upstream `ResNet_AT.forward()` signature `forward(self, x='', phrase='train', ...)` — the wrapped call reordered the keyword argument and produced `conv2d() got (str, Tensor, ...)` errors. The DataParallel wrap has been removed; the inference is single-GPU (which is what evaluation workloads want anyway).

## What was fixed in this runner

- **EAT emotion-accuracy** — `_parse_eat_text` now divides the per-class and overall `*Acc@Video` values by 100 because EAT's log file prints them as percentages. Without the fix, `payload.emo.emo_acc` was 100× too large and aggregations against other metrics (which all live on the 0..1 scale) were meaningless.
- **EAT one-line summary** — `short_keys` lookups now run on a flattened view of the EAT payload (`psnr_ssim`, `lpips`, `lmd`, `sync`, `emo`, `fid` sub-dicts are merged one level deep), so the `[unified] eat: ok ... → psnr=18.51, ssim=0.576, ...` print is no longer empty.
- **New_Emo label inference** — `_infer_emo_label()` derives the canonical emotion label from the GT (or fake) video path so `emotiefflib.correct`, `dfer_clip.correct`, and the per-model accuracy in `new_emo.summary` are populated without having to pass `--label` explicitly.
- **`--no_face_detect` parity** — `run_new_emo` now forwards `--emotieff-no-face-detect` through to `evaluate_unified.py` (which in turn forwards it to `evaluate_emotiefflib.py`), matching the standalone `run_emotiefflib` behaviour.

## Batch evaluation over `run_alone.py` outputs

[`batch_evaluator.py`](batch_evaluator.py) glues everything together for an entire `run_alone.py` output directory:

- It auto-detects each video's mode by parsing its filename:
  - `inT` (identity-preserving, e.g. `M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4`) → GT exists in MEAD11
  - `ouT` (cross-identity, e.g. `白人男_M003_front_angry_level_3_001_angry.mp4`) → no GT, only reference-free metrics
- For every video it spawns `unified_evaluator.py` as a subprocess (in the same way as the standalone CLI).
- Aggregates per-video metrics (mean / std / min / max) **overall**, **with GT only**, **without GT only**, and **per-emotion**.
- Writes a single `batch_eval_report.json` into the input directory.

```bash
python batch_evaluator.py /path/to/run_alone/output_dir/

# Quick smoke test (skip the slow EAT pipeline + restrict metrics)
python batch_evaluator.py /path/to/run_alone/output_dir/ \
    --skip eat --fvd-pad-pairs --limit 5

# Only the cheap New_Emo metrics
python batch_evaluator.py /path/to/run_alone/output_dir/ \
    --metrics lse emotiefflib dfer_clip --limit 10

# Custom MEAD root
python batch_evaluator.py /path/to/run_alone/output_dir/ \
    --mead-root /data/MEAD11/videos
```

The output JSON has shape:

```json
{
  "input_dir": "...",
  "mead_root": "...",
  "n_videos": 16, "n_with_gt": 8, "n_without_gt": 8,
  "videos": [ { "video": "...", "mode": "inT|ouT", "pid": "M003",
                "emo": "angry", "has_gt": true, "ok": true,
                "report": { ...per-metric payload... }, ... }, ... ],
  "summary": {
    "overall":    { "n_videos": ..., "metrics": { "lse": {...}, ... } },
    "with_gt":    { ... },
    "without_gt": { ... },
    "by_emotion": { "angry": {...}, "happy": {...}, ... }
  }
}
```

where each `metrics` block contains an `aggregate` field with `mean / std / min / max / n` for every numeric leaf (e.g. `lse_d`, `fvd`, `fid`, `psnr`, `mouth_lmd`, `emo_acc`, `confidence`, `dominant_fraction`, …).
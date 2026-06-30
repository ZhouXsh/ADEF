# Mouth-audio correlation

This is a lightweight lip-sync proxy that does not require a pretrained SyncNet checkpoint.

It estimates:

1. mouth opening from MediaPipe FaceMesh landmarks;
2. audio dynamics from RMS and MFCC-delta envelopes;
3. zero-lag and best-lag correlations between mouth movement and audio dynamics.

## Why use it

Use this when you want a quick diagnostic signal during development. It is not a replacement for SyncNet LSE-C/LSE-D, but it is useful for comparing ADEF variants when official SyncNet inference is not installed.

## Usage

```bash
python eval/mouth_audio_corr/eval_mouth_audio_corr.py \
  --video path/to/generated.mp4 \
  --out eval_results/mouth_audio.json
```

Batch mode:

```bash
python eval/mouth_audio_corr/eval_mouth_audio_corr.py \
  --manifest videos.txt \
  --out eval_results/mouth_audio_batch.json
```

## Output fields

- `mouth_audio_rms_corr`: Pearson correlation between mouth opening and RMS audio envelope.
- `mouth_audio_mfcc_corr`: Pearson correlation between mouth opening and MFCC-delta envelope.
- `best_lag_*`: best correlation within a small temporal lag window.
- `mouth_opening_*`: mouth opening statistics.

Higher correlation is usually better, but compare it within the same dataset and preprocessing pipeline.

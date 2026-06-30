# Temporal LPIPS / perceptual flicker

This evaluator measures frame-to-frame perceptual change using LPIPS. It is useful for detecting perceptual flicker that simple pixel differences may miss.

## Setup

Install LPIPS if it is not available:

```bash
pip install lpips
```

## Usage

```bash
python eval/temporal_lpips/eval_temporal_lpips.py \
  --video path/to/generated.mp4 \
  --out eval_results/temporal_lpips.json
```

Batch mode:

```bash
python eval/temporal_lpips/eval_temporal_lpips.py \
  --manifest videos.txt \
  --out eval_results/temporal_lpips_batch.json
```

## Output fields

- `temporal_lpips_mean`: mean LPIPS between adjacent sampled frames.
- `temporal_lpips_std`: temporal instability of perceptual changes.

Lower values generally mean less perceptual flicker, but very low values can also indicate overly static videos. Compare with motion metrics.

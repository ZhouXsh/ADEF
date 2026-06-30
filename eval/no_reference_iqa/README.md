# No-reference image/video quality proxies

This evaluator computes fast diagnostic quality proxies on sampled video frames:

- Laplacian blur / sharpness;
- brightness mean/std;
- grayscale entropy;
- face detection ratio with MediaPipe;
- face box area ratio;
- frame-to-frame pixel difference.

## Usage

```bash
python eval/no_reference_iqa/eval_iqa_basic.py \
  --video path/to/generated.mp4 \
  --out eval_results/iqa_basic.json
```

Batch mode:

```bash
python eval/no_reference_iqa/eval_iqa_basic.py \
  --manifest videos.txt \
  --out eval_results/iqa_basic_batch.json
```

## Interpretation

These are not final perceptual metrics. Use them to find blurry frames, exposure problems, missing faces, severe flicker, or bad crops. For publication-quality image distribution evaluation, use FID or a human study.

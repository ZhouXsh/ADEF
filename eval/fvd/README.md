# Frechet Video Distance / video distribution quality

FVD is a common distribution-level metric for generated videos. It usually uses I3D features and computes a Frechet distance between generated and real video feature distributions.

This directory provides a wrapper instead of vendoring an I3D implementation. Use it with a local FVD implementation such as a PyTorch/TensorFlow FVD repo, or install a package that exposes an FVD CLI.

## Usage with explicit external command

```bash
python eval/fvd/eval_fvd.py \
  --real_dir path/to/real_videos \
  --gen_dir path/to/generated_videos \
  --external_cmd "python third_party/fvd/eval.py --real {real_dir} --fake {gen_dir} --out {out}" \
  --out eval_results/fvd.json
```

The external command should write JSON or text containing a numeric `fvd` value.

## Why external

FVD depends strongly on the exact I3D checkpoint, preprocessing, clip length and frame rate. For reproducible experiments, it is safer to keep the official/selected implementation explicit and record its commit.

## Caveat

FVD is unreliable for very small sample sizes. Report the number of videos, frame sampling, resolution and I3D checkpoint.

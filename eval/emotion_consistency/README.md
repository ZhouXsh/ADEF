# Emotion consistency

This evaluator measures whether generated frames express the requested emotion label.

Because emotion recognition models vary a lot across datasets, this directory provides a generic wrapper around either:

1. a HuggingFace image-classification model; or
2. a custom external command.

For ADEF, if you have a project-specific emotion classifier that accepts cropped face frames, use `--external_cmd` and keep the model consistent across all experiments.

## Usage with HuggingFace image classifier

```bash
python eval/emotion_consistency/eval_emotion_consistency.py \
  --manifest generated.csv \
  --hf_model your/emotion-classifier \
  --out eval_results/emotion_consistency.json
```

`generated.csv`:

```csv
generated,label
/path/to/gen_happy.mp4,happy
```

## Usage with external command

```bash
python eval/emotion_consistency/eval_emotion_consistency.py \
  --manifest generated.csv \
  --external_cmd "python my_eval.py --video {video} --label {label} --out {out}" \
  --out eval_results/emotion_consistency.json
```

The external command should write JSON or text containing confidence/accuracy information.

## Output fields

- `target_confidence_mean`: average confidence for the target emotion.
- `target_top1_ratio`: fraction of sampled frames predicted as target emotion.
- `emotion_entropy_mean`: temporal uncertainty proxy.

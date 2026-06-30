# Identity preservation with ArcFace / InsightFace

This evaluator measures whether the generated talking-head video preserves identity.

It uses `insightface`, which is already listed in the ADEF requirements. For each video, it samples frames, extracts face embeddings, and compares them to either:

- a reference video/image from the manifest; or
- the first detected generated frame if no reference is provided.

## Usage

Reference-based:

```bash
python eval/identity_arcface/eval_identity_arcface.py \
  --manifest generated.csv \
  --out eval_results/identity_arcface.json
```

`generated.csv`:

```csv
generated,reference
/path/to/gen.mp4,/path/to/source.png
```

Single-video self-consistency:

```bash
python eval/identity_arcface/eval_identity_arcface.py \
  --video path/to/generated.mp4 \
  --out eval_results/identity_self.json
```

## Output fields

- `identity_cosine_mean`: mean cosine similarity between generated frames and reference embedding.
- `identity_cosine_std`: identity stability across frames.
- `detected_frames`: number of frames with a detected face.

Higher cosine similarity is better.

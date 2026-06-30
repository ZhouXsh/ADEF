# Landmark dynamics

This metric family estimates temporal naturalness from MediaPipe FaceMesh landmarks.

It reports:

- global landmark velocity and acceleration;
- mouth opening dynamics;
- eyebrow dynamics;
- mouth-specific landmark jitter;
- eyebrow-specific landmark jitter.

## Usage

```bash
python eval/landmark_dynamics/eval_landmark_dynamics.py \
  --video path/to/generated.mp4 \
  --out eval_results/landmark_dynamics.json
```

Batch mode:

```bash
python eval/landmark_dynamics/eval_landmark_dynamics.py \
  --manifest videos.txt \
  --out eval_results/landmark_dynamics_batch.json
```

## Interpretation

- Too-low mouth/eyebrow dynamics may indicate a stiff face.
- Too-high acceleration/jitter may indicate unstable animation or artifacts.
- Compare generated videos against real/reference videos from the same distribution whenever possible.

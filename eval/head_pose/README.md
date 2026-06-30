# Head pose naturalness

This evaluator estimates approximate yaw, pitch and roll from MediaPipe landmarks and OpenCV `solvePnP`.

## Usage

```bash
python eval/head_pose/eval_head_pose.py \
  --video path/to/generated.mp4 \
  --out eval_results/head_pose.json
```

Batch mode:

```bash
python eval/head_pose/eval_head_pose.py \
  --manifest videos.txt \
  --out eval_results/head_pose_batch.json
```

## Output fields

- `yaw/pitch/roll`: mean, std, velocity and acceleration statistics.
- `pose_jitter`: average absolute angular acceleration.
- `frames_with_pose`: number of frames with successful landmark detection.

## Caveat

This is a relative diagnostic metric. For publication-quality head-pose evaluation, compare against a consistent real/reference set and verify the convention of yaw/pitch/roll in your pipeline.

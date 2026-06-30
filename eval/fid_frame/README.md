# Frame-level FID

Frame-level FID compares generated frames with real/reference frames. It is useful for distribution-level visual quality, but it ignores temporal dynamics.

This wrapper supports two modes:

1. `pytorch-fid` CLI if installed;
2. a lightweight local implementation using `torchvision.models.inception_v3`.

## Prepare frame folders

You need two image folders:

```text
real_frames/
  000001.png
  ...
gen_frames/
  000001.png
  ...
```

You can extract frames with ffmpeg or your own scripts. Keep sampling consistent across methods.

## Usage

```bash
python eval/fid_frame/eval_fid_frame.py \
  --real_dir path/to/real_frames \
  --gen_dir path/to/gen_frames \
  --out eval_results/fid_frame.json
```

If `pytorch-fid` is installed, you can force external mode:

```bash
python eval/fid_frame/eval_fid_frame.py \
  --real_dir real_frames --gen_dir gen_frames \
  --use_pytorch_fid \
  --out fid.json
```

## Notes

FID is biased for small sample sizes. Use enough frames and report the sampling protocol.

# EAT Talking-Head Evaluation — Deployment

This directory hosts a self-contained deployment of the **EAT** (Efficient Emotional
Adaptation for Audio-Driven Talking-Head Generation) evaluation pipeline, plus a
unified Python wrapper (`evaluate.py`) that runs every common metric in a single
command.

## What is here

```
evaluation_eat/
├── checkpoints/                 # pre-trained model weights (auto-downloaded)
│   ├── syncnet_v2.model         # SyncNet v2 — audio-lip sync (LSE-C)
│   ├── shape_predictor_68_face_landmarks.dat  # dlib 68-pt landmark model (LMD)
│   └── Resnet18_FER+_pytorch.pth.tar        # Emotion-FAN (Accemo) — manual
├── code/                        # vendored EAT eval scripts (unchanged + minor patch)
│   ├── preprocess.py            # alignment + cropping
│   ├── test_psnr_ssim.py        # PSNR / SSIM
│   ├── test_fid.py              # FID (Inception-v3)
│   ├── test_lmd.py              # mouth / face landmark distance
│   ├── test_sync_conf.py        # SyncNet confidence (LSE-C)
│   └── test_emotion_acc.py      # Emotion-FAN accuracy
├── outputs/                     # JSON + text reports per run
├── logs/                        # raw per-metric logs
├── venv/                        # Python 3.13 virtual environment (CUDA 12.4 torch)
├── evaluate.py                  # UNIFIED entry point
├── smoke_test.py                # synthetic end-to-end test
└── README_DEPLOYMENT.md         # this file
```

## Virtual environment

Created with:

```bash
cd evaluation_eat
python3 -m venv venv --system-site-packages
./venv/bin/pip install --upgrade "setuptools<82" pip wheel
./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
./venv/bin/pip install opencv-python imageio numpy tqdm scipy scikit-image \
                    python_speech_features imutils dlib face-alignment lpips
```

Sanity check (CUDA-enabled torch, RTX 4090 should be visible):

```bash
./venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Pre-trained weights

| File | Auto? | Notes |
| --- | --- | --- |
| `checkpoints/syncnet_v2.model` (52 MB) | yes — VGG Oxford mirror | used by `test_sync_conf.py` |
| `checkpoints/shape_predictor_68_face_landmarks.dat` (95 MB) | yes — dlib.net | used by `test_lmd.py` & crop helpers |
| `checkpoints/Resnet18_FER+_pytorch.pth.tar` (≈43 MB) | **manual** — Baidu/OneDrive | fetch from `../Emotion-FAN/pretrain_model/readme.md` and drop into `checkpoints/` |

`code/shape_predictor_68_face_landmarks.dat` and `code/syncnet_v2.model` are symlinks
into `checkpoints/` so the original EAT scripts (which assume the weights live next
to them) work out of the box.

## Using the unified evaluator

```bash
./venv/bin/python evaluate.py \
    --fake "results/my_method/*.mp4" \
    --gt   "gt/MEAD/*.mp4" \
    --name my_method \
    --device 0 \
    --metrics all
```

Flags of interest:

* `--metrics` — comma-separated subset of `psnr_ssim,lpips,fid,lmd,sync,emo` or `all`.
* `--name-mode` (0..6) — filename parsing mode, default `4` (`{prefix}_{pid}_{emo}_{lev}_{vid}.mp4`).
* `--gt-name` — subfolder of the GT root (default `evp_gt`).
* `--only-pre-96-frames` — restrict FID/PSNR to the first 96 frames.
* `--no-overwrite` — skip metrics whose output already exists.
* `--skip-preprocess` / `--use-existing-preprocess` — bypass `code/preprocess.py`
  (useful when you've already aligned/cropped your videos).

The output is written to `outputs/{name}_report.json` and `outputs/{name}_report.txt`.

## Smoke test

A zero-data dependency test:

```bash
./venv/bin/python smoke_test.py
```

It generates four synthetic talking-head mp4s (with audio), runs the LPIPS branch,
and asserts the JSON report contains a numeric `mean_lpips`.  Use it to verify the
deployment after any change.

## Notes on the patched `code/preprocess.py`

The original script unconditionally imports `tkinter.tix.Tree`, which is not
shipped with the system Tk on Python 3.13.  The import is wrapped in a
`try/except ImportError` so headless servers can run the pipeline.  Nothing else
in the EAT scripts touches `Tree`, so this is safe.
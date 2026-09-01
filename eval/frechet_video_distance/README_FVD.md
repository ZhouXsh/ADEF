# ADEF Fréchet Video Distance (FVD) Evaluation

This directory extends the [Google Research FVD reference implementation](https://github.com/google-research/google-research/tree/master/frechet_video_distance) with an ADEF-specific evaluation pipeline. It computes the Fréchet Video Distance between two sets of talking-head videos — typically the ground-truth MEAD clips and the videos produced by ADEF — using the I3D model served through TensorFlow Hub.

## Files

| File | Purpose |
|------|---------|
| `frechet_video_distance.py` | Original Google reference implementation (I3D embedder + Frechet distance). |
| `evaluate_adef.py` | Memory-resident FVD evaluation: takes two MP4 directories, samples frames in RAM, returns FVD. |
| `evaluate_videos.py` | Disk-resident variant: takes video files (or directories) as input, writes per-frame JPEGs to a temp dir, computes FVD, and cleans up the temp dir on exit. |
| `setup_fvd_env.sh` | Bootstraps a self-contained `fvd` conda environment and prefetches the I3D model. |
| `requirements_fvd.txt` | Pinned dependency versions used by `setup_fvd_env.sh`. |
| `run_smoke_test.py` | Smoke test on 32 synthetic videos (low FVD expected). |
| `run_smoke_test2.py` | Smoke test on 48 synthetic videos (high FVD expected). |
| `run_smoke_test3.py` | Smoke test for `evaluate_videos.py` (auto-cleanup + `--keep_frames`). |

## Environment

The FVD reference code relies on `tensorflow.compat.v1` and the legacy `hub.Module` API, which were progressively removed in newer TF / TF-Hub releases. After testing several combinations, the following pinned versions are known to work together on the ADEF evaluation host (which ships with CUDA 13.1 — incompatible with TF 2.14+, so we run on CPU):

| Package | Version |
|---------|---------|
| Python | 3.10.x |
| tensorflow | 2.13.0 |
| tensorflow-hub | 0.13.0 (still ships `hub.Module`) |
| tensorflow-gan | 2.1.0 |
| tensorflow-probability | 0.20.1 |
| tensorflow-estimator | 2.13.0 |
| decord / opencv-python / imageio | latest stable |

The I3D model (`https://tfhub.dev/deepmind/i3d-kinetics-400/1`) is cached at `/home/Zhouxishi/tfhub_cache/` by default.

## Quick start

```bash
# 1. Bootstrap the environment (idempotent; re-runs are safe).
bash setup_fvd_env.sh
conda activate fvd

# 2. Sanity-check the FVD pipeline against the example in the original README.
python example.py
# Expected: "FVD is: ~131." (zero frames vs all-white frames)

# 3. Run the ADEF evaluation.
python evaluate_adef.py \
    --real_dir /path/to/MEAD_test_videos \
    --fake_dir /path/to/ADEF_outputs \
    --video_length 15 \
    --output_file /path/to/results.json
```

## CLI reference for `evaluate_videos.py`

`evaluate_videos.py` is the recommended entry point when you want to inspect the frames that go into I3D, or when you need to free memory between frame extraction and inference on very large evaluation sets. It accepts a single video file or a directory of videos on each side, writes per-pair JPEGs into a temp directory, and removes that directory on exit.

```bash
# Default — frames are extracted to a temp dir that is removed at the end.
python evaluate_videos.py \
    --real_dir /path/to/MEAD_test_videos \
    --fake_dir /path/to/ADEF_outputs \
    --video_length 15 \
    --output_file /path/to/results.json

# Keep the per-pair frame JPEGs after the run (e.g. for debugging).
python evaluate_videos.py \
    --real_dir /path/to/MEAD_test_videos \
    --fake_dir /path/to/ADEF_outputs \
    --work_dir /tmp/fvd_frames \
    --keep_frames \
    --output_file /path/to/results.json

# Pass a single video file as one side (mixed file/dir inputs are allowed).
python evaluate_videos.py \
    --real_dir /path/to/MEAD_test_videos/clip_001.mp4 \
    --fake_dir /path/to/ADEF_outputs/clip_001.mp4 \
    --video_length 15
```

| Flag | Description |
|------|-------------|
| `--real_dir` | Directory of ground-truth videos OR a single video file (mp4/mov/avi/mkv/webm/m4v). |
| `--fake_dir` | Directory of generated videos OR a single video file. File basenames (without extension) must match those on the other side. |
| `--video_length` | Frames to sample per video (default 15). |
| `--work_dir` | Where to write per-pair frame JPEGs. Defaults to a `tempfile.mkdtemp()` directory that is removed on exit. |
| `--keep_frames` | Do not delete the work_dir after evaluation. |
| `--output_file` | Write JSON results here; printed to stdout otherwise. |
| `--limit` | Optional cap on number of pairs (useful when iterating). |
| `--quiet` | Suppress per-video loading messages. |

Output JSON includes the `work_dir` path and a `work_dir_kept` boolean so you can see exactly where the frames landed.

## How many videos do I need? And what is a good FVD value?

**Minimum 16 per side** because the original I3D reference implementation hard-codes the batch dimension to 16 inside `create_id3_embedding()`. Practically though, with fewer than ~50 samples per side the covariance estimate becomes rank-deficient and Frechet distance is unreliable. Common choices in literature:

| Sample count per side | Notes |
|---|---|
| 16 | Bare minimum (I3D batch size). Padding to 16 via `--pad_pairs_to_batch_size` exists for sanity checks only. |
| 50 | Tight. FVD still fluctuates a lot from run to run if you change the sample. |
| 100–256 | Most common range for talking-head evaluation. |
| 2 048 | Default in the original FVD paper. Tighter estimate but much more compute. |

**Video pairing**: pairs of real/fake don't need to be temporally aligned — file basenames are just used to match. The 16 real videos can be a mix of any identities, emotions, or levels. As long as the *fake* side mirrors the *real* side set for set, the FVD is a fair comparison.

**Reasonable FVD ranges** for talking-head evaluation with 100–500 samples per side using the bundled I3D-Kinetics-400 implementation:

| FVD | Interpretation |
|---|---|
| ≈ 0 | Sanity-check: real vs real or replicated-pair demo. |
| < 50 | Excellent — generated motion features barely deviate from real. |
| 50 – 150 | Good — typical for state-of-the-art talking-head / video-generation models. |
| 150 – 300 | Acceptable — visible room for improvement. |
| > 300 | Poor — generated motion distribution is clearly off. |

**Reference FVD figures** (same I3D implementation, for context only):
- Wav2Lip on LRS2 ~ 30-90
- MakeItTalk / PC-AVS / SadTalker ~ 100-250
- DiffTalk / VASA-1 ~ 70-200

Cross-paper comparison is only fair when **sample counts and reference implementation match**.

### Running with fewer than 16 pairs (demo only)

The `--pad_pairs_to_batch_size` flag replicates existing pairs until we have at least 16, then computes FVD. The result is **biased low** because both distributions collapse onto themselves, but it is useful for verifying the end-to-end pipeline works.

```bash
python evaluate_adef.py \
    --real_dir /path/to/M003_front_angry_level_3_001.mp4 \
    --fake_dir /path/to/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --video_length 16 \
    --pad_pairs_to_batch_size \
    --output_file demo.json
```

For a real per-identity, per-emotion evaluation, drop multiple MEAD clips and their matching ADEF outputs into two directories:

```bash
ls /path/to/mead_test/M003/front/angry/level_3/
# M003_front_angry_level_3_001.mp4 ... 029.mp4

ls /path/to/adef_outputs/M003/front/angry/level_3/
# M003_front_angry_level_3_001_*.mp4 ... 029_*.mp4
```

The scripts pair them by basename (the `*_xxx_angry` suffix on the fake side gets stripped). If the fakes have extra metadata in their filenames, drop them in the same directory and rely on the basename prefix match, or pass individual file pairs.

## CLI reference for `evaluate_adef.py`

| Flag | Description |
|------|-------------|
| `--real_dir` | Directory of ground-truth videos, or a single video file. |
| `--fake_dir` | Directory of ADEF-generated videos, or a single video file. |
| `--video_length` | Number of frames uniformly sampled from each video (default `15`, matching the I3D example). For talking-head evaluation, 16 is a common choice. |
| `--output_file` | If set, results are written as JSON; otherwise printed to stdout. |
| `--limit` | Optional cap on number of pairs (useful when iterating). |
| `--pad_pairs_to_batch_size` | Demo-only: replicate pairs until ≥ 16 to satisfy I3D batch size. The FVD is biased low; only use for sanity checks. |
| `--quiet` | Suppress per-video loading messages. |

The script requires **at least 16** matched video pairs because the reference I3D implementation uses a hard-coded batch size of 16. Pairs that don't divide evenly into 16 are zero-padded internally with repeated frames — only this padding is reflected in the warning, the result still uses the original `N` activations. To run on fewer than 16 pairs, pass `--pad_pairs_to_batch_size` and read the "How many videos do I need?" section above first.

### Output format

```json
{
  "fvd": 28.1306,
  "num_pairs": 32,
  "video_length": 15,
  "real_dir": "/abs/path/to/real",
  "fake_dir": "/abs/path/to/fake"
}
```

## Implementation notes

1. **Graph construction ordering.** The FVD library constructs the I3D embedding op the first time it is called. New variables are added to the graph at that moment, so `tf.global_variables_initializer()` must run **after** the embedding op exists but **before** the first `sess.run(embedding_op, ...)` call. `evaluate_adef.py` enforces this ordering explicitly.

2. **Shared placeholder.** Real and fake videos share a single placeholder + embedding so the I3D module is loaded into the graph exactly once. The I3D batch dimension is fixed to 16 by the reference implementation, so both inputs are fed in batches of 16.

3. **Variable scope renaming.** The reference library exposes its embedding op only via a graph tensor lookup, so we feed different `tf.placeholder` tensors (named `real_emb_ph`/`fake_emb_ph`) into `tfgan.eval.frechet_classifier_distance_from_activations` to keep the two activations separate.

4. **Package import quirk.** `frechet_video_distance` is both the directory name and the name of the module file inside it. `evaluate_adef.py` adds the parent directory (`eval/`) to `sys.path` so `from frechet_video_distance import frechet_video_distance as fvd` resolves to the submodule.

5. **CPU runtime.** Even though the host has 3× RTX 4090 GPUs, the system's CUDA 13.1 driver is newer than any TF 2.x wheel supports, so TF 2.13 runs in CPU-only mode. FVD inference is bounded by the I3D forward pass (~50M params) and is fast enough on CPU for typical ADEF evaluation batches; the script accepts the overhead rather than fighting the CUDA toolkit versioning.

## Troubleshooting

- **`AttributeError: module 'tensorflow_hub' has no attribute 'Module'`** — you have a newer `tensorflow-hub`. Pin to `tensorflow-hub==0.13.0`.
- **`ImportError: This version of TensorFlow Probability requires TensorFlow version >= 2.18`** — your `tensorflow-probability` is too new. Pin to `tensorflow-probability==0.20.1`.
- **`Unable to register cuDNN factory` / `Cannot dlopen some GPU libraries`** — harmless when the TF build cannot find a matching CUDA toolkit; the script falls back to CPU. The I3D embedding is unaffected.
- **`FailedPreconditionError: variable … does not exist`** — `tf.global_variables_initializer()` ran before the I3D variables were added; re-check that the embedding op is created inside the `tf.Graph().as_default()` block and that initialization runs after it.
- **`Cannot import name 'frechet_video_distance' from 'frechet_video_distance'`** — you're running the script in a way that lets Python pick up the .py file before the package directory. Run it from `eval/` (one directory above), or set `PYTHONPATH` to `eval/`.

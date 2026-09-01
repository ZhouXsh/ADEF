# coding=utf-8
# Fréchet Video Distance evaluation that takes **video files** as input and
# extracts frames to a temporary directory on disk before computing FVD.
#
# Compared to `evaluate_adef.py` (which keeps frames in memory) this script is
# useful when you want to inspect the preprocessed frames, or when you need to
# free memory between frame extraction and I3D inference for very large
# evaluation sets. The frame extraction step writes `<name>_frame_NNN.jpg`
# files into a per-pair subdirectory under a temp directory, and the whole
# directory is removed on exit (unless `--keep_frames` is passed).
#
# Usage:
#   python evaluate_videos.py \
#       --real_dir path/to/real_videos \
#       --fake_dir path/to/generated_videos \
#       --video_length 15 \
#       --output_file results.json

"""FVD evaluation that materialises frames to a temp dir then cleans up."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
import warnings
from glob import glob

# Default TF Hub cache so the I3D model loads offline.
os.environ.setdefault("TFHUB_CACHE_DIR", "/home/Zhouxishi/tfhub_cache")

import numpy as np  # noqa: E402

# Fix the package import collision: the directory and module share the same
# name. Insert the parent of the package directory on sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from frechet_video_distance import frechet_video_distance as fvd  # noqa: E402

# The FVD reference code relies on TF1-style graph APIs.
import tensorflow.compat.v1 as tf  # noqa: E402
tf.disable_eager_execution()

import tensorflow_hub as hub  # noqa: E402
import tensorflow_gan as tfgan  # noqa: E402
import six  # noqa: E402

I3D_BATCH_SIZE = 16
EMBEDDING_DIM = 400
IMAGE_SIZE = 224


# -----------------------------------------------------------------------------
# Path collection
# -----------------------------------------------------------------------------


VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "mkv", "webm", "m4v")


def is_video_path(path):
    """True if *path* is a single video file (not a directory)."""
    if not os.path.isfile(path):
        return False
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    return ext in VIDEO_EXTENSIONS


def collect_inputs(path):
    """Resolve *path* (file or directory) into a list of video files.

    - If *path* is a single file: returns [path] (when extension matches).
    - If *path* is a directory: returns sorted video files inside it.
    - Otherwise: raises.
    """
    if os.path.isfile(path):
        if not is_video_path(path):
            raise ValueError(
                "Input %s is a file but its extension is not a known video "
                "format (%s)." % (path, ", ".join(VIDEO_EXTENSIONS)))
        return [path]
    if os.path.isdir(path):
        found = []
        for ext in VIDEO_EXTENSIONS:
            found.extend(glob(os.path.join(path, "*." + ext)))
            found.extend(glob(os.path.join(path, "*." + ext.upper())))
        found = sorted(set(found))
        if not found:
            raise ValueError("No video files found in directory %s" % path)
        return found
    raise ValueError("Path %s does not exist (or is neither file nor directory)." % path)


def pair_videos(real_paths, fake_paths):
    """Pair real and fake videos.

    Pairing rules, in order of preference:
      1. If both sides resolve to a single video, pair them directly
         (basename does not need to match — useful for ADEF where the fake
         filename may embed driver metadata like
         ``M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4``).
      2. Otherwise, pair by basename (filename without extension).
    """
    if len(real_paths) == 1 and len(fake_paths) == 1:
        # Both inputs are single files — pair them directly.
        warnings.warn(
            "Single-file pair mode: pairing %s with %s without checking basename."
            % (os.path.basename(real_paths[0]), os.path.basename(fake_paths[0])))
        return [(real_paths[0], fake_paths[0])]

    real_map = {os.path.splitext(os.path.basename(p))[0]: p for p in real_paths}
    fake_map = {os.path.splitext(os.path.basename(p))[0]: p for p in fake_paths}
    keys = sorted(set(real_map).intersection(fake_map))
    if not keys:
        raise RuntimeError(
            "No matching filenames between the real and fake sets. Cross-check "
            "that basenames (e.g. `clip_001.mp4`) coincide on both sides, or pass "
            "a single file on each side to bypass basename matching.")
    missing_real = sorted(set(fake_map) - set(real_map))
    missing_fake = sorted(set(real_map) - set(fake_map))
    if missing_real:
        warnings.warn("%d fake video(s) have no real counterpart: %s%s"
                      % (len(missing_real), missing_real[:3],
                         "" if len(missing_real) <= 3 else ", ..."))
    if missing_fake:
        warnings.warn("%d real video(s) have no fake counterpart: %s%s"
                      % (len(missing_fake), missing_fake[:3],
                         "" if len(missing_fake) <= 3 else ", ..."))
    return [(real_map[k], fake_map[k]) for k in keys]


# -----------------------------------------------------------------------------
# Frame extraction (writes JPEG files into a per-pair subdirectory)
# -----------------------------------------------------------------------------


def extract_frames(video_path, target_length, out_dir, image_size=IMAGE_SIZE):
    """Decode *video_path* into JPEG frames stored under *out_dir*.

    Returns the in-memory uint8 array [T, H, W, 3] that callers can feed to
    I3D directly. The JPEGs are kept under *out_dir* so the caller can inspect
    them later, and the directory itself is removed by the temp-dir cleanup
    handler registered at script startup.
    """
    os.makedirs(out_dir, exist_ok=True)
    frames = _decode_video(video_path)
    indices = list(range(len(frames)))
    if target_length is not None and len(frames) != target_length:
        indices = np.linspace(0, len(frames) - 1, target_length).round().astype(int).tolist()

    import cv2
    arr = np.empty((len(indices), image_size, image_size, 3), dtype=np.uint8)
    for i, idx in enumerate(indices):
        frame = frames[idx]
        resized = cv2.resize(frame, (image_size, image_size),
                             interpolation=cv2.INTER_LINEAR)
        arr[i] = resized
        if i == idx:  # only persist original-length frames? No — always persist the sampled ones.
            pass
        # Save the sampled-and-resized frame so users can inspect what I3D saw.
        out_path = os.path.join(out_dir, "frame_%03d.jpg" % i)
        cv2.imwrite(out_path, cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    return arr


def _decode_video(path):
    """Decode *path* into a list of RGB uint8 frames. Tries decord, falls back to OpenCV."""
    try:
        import decord  # noqa: WPS433
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, num_threads=1)
        total = len(vr)
        if total == 0:
            raise RuntimeError("Empty video: %s" % path)
        return [vr[i].asnumpy() for i in range(total)]
    except Exception:  # pragma: no cover
        import cv2
        cap = cv2.VideoCapture(path)
        raw = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not raw:
            raise RuntimeError("Unreadable video: %s" % path)
        return raw


# -----------------------------------------------------------------------------
# I3D embedding
# -----------------------------------------------------------------------------


def _pad_to_batch(arr, batch_size):
    pad = (-len(arr)) % batch_size
    if pad == 0:
        return arr, len(arr)
    extra = np.tile(arr[-1:], (pad, 1, 1, 1, 1))
    return np.concatenate([arr, extra], axis=0), len(arr)


def embed_videos(video_array, sess, videos_ph, embedding_op):
    arr, n = _pad_to_batch(video_array, I3D_BATCH_SIZE)
    out = np.zeros((arr.shape[0], EMBEDDING_DIM), dtype=np.float32)
    for start in range(0, arr.shape[0], I3D_BATCH_SIZE):
        batch = arr[start:start + I3D_BATCH_SIZE]
        out[start:start + I3D_BATCH_SIZE] = sess.run(
            embedding_op, feed_dict={videos_ph: batch})
    return out[:n]


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="FVD between real and fake video files (frames are materialised "
                    "in a temp directory).")
    parser.add_argument("--real_dir", required=True,
                        help="Directory of ground-truth videos OR a single video file.")
    parser.add_argument("--fake_dir", required=True,
                        help="Directory of generated videos OR a single video file.")
    parser.add_argument("--video_length", type=int, default=15,
                        help="Frames to sample per video (default 15).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Write JSON results here; printed to stdout otherwise.")
    parser.add_argument("--work_dir", type=str, default=None,
                        help="Where to write per-pair frame JPEGs. Defaults to a "
                             "temp directory that is removed on exit.")
    parser.add_argument("--keep_frames", action="store_true",
                        help="Do not delete the work_dir after evaluation.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on the number of matched video pairs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-video progress messages.")
    parser.add_argument("--pad_pairs_to_batch_size", action="store_true",
                        help="Demo-only: when the number of pairs is smaller than "
                             "I3D_BATCH_SIZE (16), repeat the existing pairs until "
                             "the total reaches 16. The reported FVD will be close "
                             "to 0 because both distributions are identical — this "
                             "verifies the pipeline end-to-end but does NOT measure "
                             "model quality.")
    return parser.parse_args()


def main():
    args = parse_args()

    real_paths = collect_inputs(args.real_dir)
    fake_paths = collect_inputs(args.fake_dir)
    pairs = pair_videos(real_paths, fake_paths)
    if args.limit:
        pairs = pairs[: args.limit]
    print("[FVD] Matched %d video pair(s)." % len(pairs))
    demo_mode = False
    num_pairs_original = len(pairs)
    if len(pairs) < I3D_BATCH_SIZE:
        if not args.pad_pairs_to_batch_size:
            raise RuntimeError(
                "Need at least %d matched video pairs to evaluate FVD (have %d). "
                "Pass --pad_pairs_to_batch_size to repeat the existing pairs and "
                "fill the I3D batch (DEMO ONLY — the resulting FVD ~0 says nothing "
                "about model quality)." % (I3D_BATCH_SIZE, len(pairs)))
        original = list(pairs)
        while len(pairs) < I3D_BATCH_SIZE:
            pairs.append(original[len(pairs) % len(original)])
        demo_mode = True
        warnings.warn(
            "[FVD] pad_pairs_to_batch_size: extended %d -> %d pairs by cycling. "
            "FVD result is statistically meaningless." % (len(original), len(pairs)))

    # Set up the work directory (per-pair frame JPEGs land here).
    if args.work_dir:
        work_root = os.path.abspath(args.work_dir)
        os.makedirs(work_root, exist_ok=True)
        cleanup_workdir = False
    else:
        work_root = tempfile.mkdtemp(prefix="fvd_frames_")
        cleanup_workdir = True

    if cleanup_workdir:
        def _cleanup():
            if os.path.isdir(work_root):
                shutil.rmtree(work_root, ignore_errors=True)
                print("[FVD] Cleaned up temp frame directory %s" % work_root)
        atexit.register(_cleanup)
    print("[FVD] Frames will be written under: %s%s"
          % (work_root, "" if cleanup_workdir else " (kept: --keep_frames)"))

    real_arr = []
    fake_arr = []
    for idx, (real_path, fake_path) in enumerate(pairs):
        if not args.quiet:
            print("[FVD] Pair %d/%d: %s <-> %s"
                  % (idx + 1, len(pairs),
                     os.path.basename(real_path), os.path.basename(fake_path)))
        stem = os.path.splitext(os.path.basename(real_path))[0]
        pair_dir = os.path.join(work_root, stem)
        real_dir = os.path.join(pair_dir, "real")
        fake_dir = os.path.join(pair_dir, "fake")
        real_arr.append(extract_frames(real_path, args.video_length, real_dir))
        fake_arr.append(extract_frames(fake_path, args.video_length, fake_dir))

    real_arr = np.stack(real_arr, axis=0)
    fake_arr = np.stack(fake_arr, axis=0)
    print("[FVD] Real videos shape: %s, dtype=%s" % (real_arr.shape, real_arr.dtype))
    print("[FVD] Fake videos shape: %s, dtype=%s" % (fake_arr.shape, fake_arr.dtype))

    T, H, W, C = real_arr.shape[1], real_arr.shape[2], real_arr.shape[3], real_arr.shape[4]

    with tf.Graph().as_default():
        videos_ph = tf.placeholder(tf.uint8, [I3D_BATCH_SIZE, T, H, W, C],
                                   name="fvd_eval_videos")
        processed = fvd.preprocess(videos_ph, (IMAGE_SIZE, IMAGE_SIZE))
        embedding_op = fvd.create_id3_embedding(processed)
        fvd_tensor = tfgan.eval.frechet_classifier_distance_from_activations(
            tf.placeholder(tf.float32, [None, EMBEDDING_DIM], name="real_emb_ph"),
            tf.placeholder(tf.float32, [None, EMBEDDING_DIM], name="fake_emb_ph"))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())

            print("[FVD] Extracting I3D embeddings for real videos ...")
            real_emb = embed_videos(real_arr, sess, videos_ph, embedding_op)

            print("[FVD] Extracting I3D embeddings for generated videos ...")
            fake_emb = embed_videos(fake_arr, sess, videos_ph, embedding_op)

            print("[FVD] Computing Frechet distance ...")
            fvd_value = sess.run(fvd_tensor, feed_dict={
                "real_emb_ph:0": real_emb,
                "fake_emb_ph:0": fake_emb,
            })

    print("[FVD] Fréchet Video Distance = %.4f" % fvd_value)

    result = {
        "fvd": float(fvd_value),
        "num_pairs": int(len(pairs)),
        "num_pairs_original": int(num_pairs_original),
        "video_length": int(args.video_length),
        "real_dir": os.path.abspath(args.real_dir),
        "fake_dir": os.path.abspath(args.fake_dir),
        "work_dir": work_root,
        "work_dir_kept": bool(args.keep_frames or args.work_dir),
        "demo_mode": bool(demo_mode),
    }
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("[FVD] Saved result to %s" % args.output_file)
    return result


if __name__ == "__main__":
    main()
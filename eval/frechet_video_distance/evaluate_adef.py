# coding=utf-8
# Fréchet Video Distance evaluation script for ADEF-generated talking videos.
#
# This script computes FVD between two sets of videos (typically ADEF-generated
# videos vs. ground-truth MEAD videos) using the I3D model distributed via
# TensorFlow Hub.
#
# Usage:
#   python evaluate_adef.py \
#       --real_dir path/to/real_videos \
#       --fake_dir path/to/generated_videos \
#       --video_length 15 \
#       --output_file results.json
#
# Inputs are two directories containing the same number of videos. Each video
# is uniformly sampled to a fixed number of frames, resized to 224x224, and
# passed through the I3D network. The FVD is computed from the resulting
# activations via the Frechet distance.

"""Compute Frechet Video Distance for ADEF-generated and real videos."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import sys
import warnings
from glob import glob

# Before importing TF / TF-Hub, set TFHUB_CACHE_DIR so the I3D module is
# loaded from the local cache (~49 MB) populated by setup_fvd_env.sh. This
# allows the script to run without re-downloading the model each time and
# keeps evaluation runs fully offline once the cache is in place.
os.environ.setdefault(
    "TFHUB_CACHE_DIR", "/home/Zhouxishi/tfhub_cache")

import numpy as np  # noqa: E402

# Fix the package-relative import: `frechet_video_distance` is a package whose
# submodule file shares the same name, which trips Python's name resolution.
# We add the package's parent directory to sys.path so the import resolves.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Import after sys.path adjustment. The I3D model file uses
# `frechet_video_distance` as both the package and module name.
from frechet_video_distance import frechet_video_distance as fvd  # noqa: E402

# TensorFlow must be imported with eager execution disabled because the FVD
# implementation uses TF1 graph APIs (tf.compat.v1 + hub.Module).
import tensorflow.compat.v1 as tf  # noqa: E402
tf.disable_eager_execution()

import tensorflow_hub as hub  # noqa: E402
import tensorflow_gan as tfgan  # noqa: E402
import six  # noqa: E402

# The I3D model is hard-coded for batch size 16 (see frechet_video_distance.py).
I3D_BATCH_SIZE = 16
EMBEDDING_DIM = 400  # I3D-Kinetics-400 Mean-pool output dimension.
IMAGE_SIZE = 224


# -----------------------------------------------------------------------------
# Video loading
# -----------------------------------------------------------------------------


def load_video(path, target_length=None, image_size=IMAGE_SIZE):
    """Reads a video file and returns an array [T, H, W, 3] in uint8.

    Uses decord for fast, deterministic frame reads, with a fallback chain
    through OpenCV so the script still works if decord is not available.
    """
    frames = _read_frames(path)
    if target_length is not None and len(frames) != target_length:
        indices = np.linspace(0, len(frames) - 1, target_length).round().astype(int)
        frames = [frames[i] for i in indices]
    import cv2
    resized = np.empty((len(frames), image_size, image_size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        resized[i] = cv2.resize(frame, (image_size, image_size),
                                interpolation=cv2.INTER_LINEAR)
    return resized


def _read_frames(path):
    """Read all frames as a list of RGB uint8 ndarrays. Tries decord first."""
    try:
        import decord  # noqa: WPS433
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, num_threads=1)
        total = len(vr)
        if total == 0:
            raise RuntimeError("Empty video: %s" % path)
        return [vr[i].asnumpy() for i in range(total)]
    except Exception:  # pragma: no cover - environment-specific fallback
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
            raise RuntimeError("Empty or unreadable video: %s" % path)
        return raw


def collect_videos(directory, extensions=("mp4", "mov", "avi", "mkv")):
    """Returns a sorted list of video paths in *directory* matching extensions."""
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(directory, "*." + ext)))
        files.extend(glob(os.path.join(directory, "*." + ext.upper())))
    return sorted(files)


def match_videos(real_dir, fake_dir):
    """Pair real and fake inputs (each may be a directory or a single file).

    Pairing rules, in order of preference:
      1. If both sides resolve to a single video, pair them directly without
         checking basename (useful when ADEF's fake filename embeds extra
         driver metadata).
      2. Otherwise, pair by basename.
    """
    real_paths = collect_inputs(real_dir)
    fake_paths = collect_inputs(fake_dir)
    if len(real_paths) == 1 and len(fake_paths) == 1:
        warnings.warn(
            "Single-file pair mode: pairing %s with %s without checking basename."
            % (os.path.basename(real_paths[0]), os.path.basename(fake_paths[0])))
        return [(real_paths[0], fake_paths[0])]
    real_map = {os.path.splitext(os.path.basename(p))[0]: p for p in real_paths}
    fake_map = {os.path.splitext(os.path.basename(p))[0]: p for p in fake_paths}
    keys = sorted(set(real_map).intersection(fake_map))
    if not keys:
        raise RuntimeError(
            "No matching filenames between %s and %s. "
            "Cross-check that both directories contain the same video names, "
            "or pass a single file on each side to bypass basename matching."
            % (real_dir, fake_dir))
    return [(real_map[k], fake_map[k]) for k in keys]


def collect_inputs(path):
    """Resolve *path* (file or directory) into a list of video file paths."""
    VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "mkv", "webm", "m4v")
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext not in VIDEO_EXTENSIONS:
            raise ValueError(
                "Input %s is a file but its extension is not a known video format (%s)."
                % (path, ", ".join(VIDEO_EXTENSIONS)))
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


# -----------------------------------------------------------------------------
# I3D embedding extraction
# -----------------------------------------------------------------------------


def _pad_to_batch(arr, batch_size):
    """Pad *arr* along axis 0 so its first dimension is a multiple of batch_size."""
    pad = (-len(arr)) % batch_size
    if pad == 0:
        return arr, len(arr)
    # Repeat the last video rather than zero-padding so we never feed
    # tensors outside the [-1, 1] range that I3D asserts on.
    extra = np.tile(arr[-1:], (pad, 1, 1, 1, 1))
    return np.concatenate([arr, extra], axis=0), len(arr)


def embed_videos(video_array, sess, videos_ph, embedding_op):
    """Compute I3D Mean-pool embeddings for a [N, T, H, W, 3] uint8 batch.

    Args:
        video_array: numpy array uint8 in [0, 255], shape [N, T, H, W, 3].
        sess: an active TF1 session with all variables initialised.
        videos_ph: tf.uint8 placeholder of shape [B, T, H, W, 3], B=I3D_BATCH_SIZE.
        embedding_op: tensor returned by fvd.create_id3_embedding.

    Returns:
        np.ndarray of shape [N, EMBEDDING_DIM], dtype float32.
    """
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
        description="Evaluate Fréchet Video Distance between ADEF-generated videos and ground truth.")
    parser.add_argument("--real_dir", required=True,
                        help="Directory of ground-truth / reference videos.")
    parser.add_argument("--fake_dir", required=True,
                        help="Directory of ADEF-generated videos.")
    parser.add_argument("--video_length", type=int, default=15,
                        help="Frames to sample per video (default 15).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Write JSON results here; printed to stdout otherwise.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on the number of matched video pairs.")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-video loading messages.")
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

    pairs = match_videos(args.real_dir, args.fake_dir)
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
        # Pad by cycling through the existing pairs until we reach batch size.
        original = list(pairs)
        while len(pairs) < I3D_BATCH_SIZE:
            pairs.append(original[len(pairs) % len(original)])
        demo_mode = True
        warnings.warn(
            "[FVD] pad_pairs_to_batch_size: extended %d -> %d pairs by cycling. "
            "FVD result is statistically meaningless." % (len(original), len(pairs)))

    real_arr = []
    fake_arr = []
    for idx, (real_path, fake_path) in enumerate(pairs):
        if not args.quiet:
            print("[FVD] Loading pair %d/%d: %s / %s"
                  % (idx + 1, len(pairs), os.path.basename(real_path),
                     os.path.basename(fake_path)))
        real_arr.append(load_video(real_path, target_length=args.video_length))
        fake_arr.append(load_video(fake_path, target_length=args.video_length))

    real_arr = np.stack(real_arr, axis=0)
    fake_arr = np.stack(fake_arr, axis=0)
    print("[FVD] Real videos shape: %s, dtype=%s" % (real_arr.shape, real_arr.dtype))
    print("[FVD] Fake videos shape: %s, dtype=%s" % (fake_arr.shape, fake_arr.dtype))

    # Per the official I3D implementation, videos in the
    # fvd.create_id3_embedding graph must have the same batch dim across both
    # feeds. We share one placeholder + I3D embedding for both real and fake
    # sets so the I3D module is loaded once into the graph.
    T, H, W, C = real_arr.shape[1], real_arr.shape[2], real_arr.shape[3], real_arr.shape[4]

    with tf.Graph().as_default():
        videos_ph = tf.placeholder(tf.uint8, [I3D_BATCH_SIZE, T, H, W, C],
                                   name="fvd_eval_videos")
        processed = fvd.preprocess(videos_ph, (IMAGE_SIZE, IMAGE_SIZE))
        embedding_op = fvd.create_id3_embedding(processed)
        fvd_tensor = tfgan.eval.frechet_classifier_distance_from_activations(
            tf.placeholder(tf.float32, [None, EMBEDDING_DIM], name="real_emb_ph"),
            tf.placeholder(tf.float32, [None, EMBEDDING_DIM], name="fake_emb_ph"),
        )

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
        "demo_mode": bool(demo_mode),
    }
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("[FVD] Saved result to %s" % args.output_file)
    return result


if __name__ == "__main__":
    main()

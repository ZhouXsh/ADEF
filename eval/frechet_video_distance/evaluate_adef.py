#!/usr/bin/env python3
# coding=utf-8
"""Dataset-level Fréchet Video Distance using the official Google I3D graph.

FVD is computed exactly once between the complete real-video embedding set and
the complete generated-video embedding set.  The I3D module has a fixed
inference batch size of 16; the final incomplete *inference batch* is padded
internally and trimmed again before the Fréchet statistics are computed.  This
batch constraint is not a requirement to have a multiple of 16 videos.
"""
from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys
import warnings
from glob import glob
from pathlib import Path

os.environ.setdefault("TFHUB_CACHE_DIR", "/home/Zhouxishi/tfhub_cache")

import numpy as np  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
from frechet_video_distance import frechet_video_distance as fvd  # noqa: E402

import tensorflow.compat.v1 as tf  # noqa: E402
tf.disable_eager_execution()

I3D_BATCH_SIZE = 16
EMBEDDING_DIM = 400
IMAGE_SIZE = 224
VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "mkv", "webm", "m4v")


def _read_list(path):
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    out = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if s and not s.startswith("#"):
            if not Path(s).is_file():
                raise FileNotFoundError(s)
            out.append(s)
    if not out:
        raise ValueError("empty video list: %s" % p)
    return out


def collect_inputs(path):
    if os.path.isfile(path):
        ext = os.path.splitext(path)[1].lstrip(".").lower()
        if ext not in VIDEO_EXTENSIONS:
            raise ValueError("not a supported video: %s" % path)
        return [path]
    if os.path.isdir(path):
        found = []
        for ext in VIDEO_EXTENSIONS:
            found.extend(glob(os.path.join(path, "*." + ext)))
            found.extend(glob(os.path.join(path, "*." + ext.upper())))
        if not found:
            # Dataset folders are often nested by speaker/emotion.
            for root, _, files in os.walk(path):
                for name in files:
                    if name.rsplit(".", 1)[-1].lower() in VIDEO_EXTENSIONS:
                        found.append(os.path.join(root, name))
        found = sorted(set(found))
        if not found:
            raise ValueError("no videos under %s" % path)
        return found
    raise ValueError("path does not exist: %s" % path)


def _read_frames(path):
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path, num_threads=1)
        if len(vr) == 0:
            raise RuntimeError("empty video")
        return [vr[i].asnumpy() for i in range(len(vr))]
    except Exception:
        import cv2
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError("empty/unreadable video: %s" % path)
        return frames


def load_video(path, target_length=16, image_size=IMAGE_SIZE):
    frames = _read_frames(path)
    if target_length is not None:
        indices = np.linspace(0, len(frames) - 1, target_length).round().astype(int)
        frames = [frames[i] for i in indices]
    import cv2
    arr = np.empty((len(frames), image_size, image_size, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        arr[i] = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return arr


def _pad_to_batch(arr, batch_size):
    n = len(arr)
    pad = (-n) % batch_size
    if pad == 0:
        return arr, n
    extra = np.tile(arr[-1:], (pad, 1, 1, 1, 1))
    return np.concatenate([arr, extra], axis=0), n


def embed_videos(video_array, sess, videos_ph, embedding_op):
    arr, n = _pad_to_batch(video_array, I3D_BATCH_SIZE)
    chunks = []
    for start in range(0, len(arr), I3D_BATCH_SIZE):
        chunks.append(sess.run(embedding_op, feed_dict={videos_ph: arr[start:start + I3D_BATCH_SIZE]}))
    return np.concatenate(chunks, axis=0)[:n]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--real_dir", help="Real video file/directory")
    p.add_argument("--fake_dir", help="Generated video file/directory")
    p.add_argument("--real_list", help="Text file: one real video per line")
    p.add_argument("--fake_list", help="Text file: one generated video per line")
    p.add_argument("--video_length", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_file")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if bool(args.real_list) != bool(args.fake_list):
        raise SystemExit("--real_list and --fake_list must be supplied together")
    if args.real_list:
        real_paths = _read_list(args.real_list)
        fake_paths = _read_list(args.fake_list)
    else:
        if not args.real_dir or not args.fake_dir:
            raise SystemExit("provide --real_dir/--fake_dir or --real_list/--fake_list")
        real_paths = collect_inputs(args.real_dir)
        fake_paths = collect_inputs(args.fake_dir)

    if len(real_paths) != len(fake_paths):
        raise RuntimeError("FVD protocol requires equal real/fake set sizes (%d vs %d)" %
                           (len(real_paths), len(fake_paths)))
    if args.limit:
        real_paths = real_paths[:args.limit]
        fake_paths = fake_paths[:args.limit]
    if len(real_paths) < 2:
        raise RuntimeError("FVD needs at least two videos in each set")
    if len(real_paths) < 16:
        warnings.warn("FVD with fewer than 16 videos has high estimator variance; the value is valid but unstable.")

    real_arr, fake_arr = [], []
    for i, (rp, fp) in enumerate(zip(real_paths, fake_paths), start=1):
        if not args.quiet:
            print("[FVD] %d/%d %s | %s" % (i, len(real_paths), os.path.basename(rp), os.path.basename(fp)))
        real_arr.append(load_video(rp, args.video_length))
        fake_arr.append(load_video(fp, args.video_length))
    real_arr = np.stack(real_arr, axis=0)
    fake_arr = np.stack(fake_arr, axis=0)

    T, H, W, C = real_arr.shape[1:]
    with tf.Graph().as_default():
        videos_ph = tf.placeholder(tf.uint8, [I3D_BATCH_SIZE, T, H, W, C], name="fvd_eval_videos")
        processed = fvd.preprocess(videos_ph, (IMAGE_SIZE, IMAGE_SIZE))
        embedding_op = fvd.create_id3_embedding(processed)
        real_emb_ph = tf.placeholder(tf.float32, [None, EMBEDDING_DIM], name="real_emb_ph")
        fake_emb_ph = tf.placeholder(tf.float32, [None, EMBEDDING_DIM], name="fake_emb_ph")
        fvd_tensor = fvd.calculate_fvd(real_emb_ph, fake_emb_ph)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            real_emb = embed_videos(real_arr, sess, videos_ph, embedding_op)
            fake_emb = embed_videos(fake_arr, sess, videos_ph, embedding_op)
            value = float(sess.run(fvd_tensor, feed_dict={real_emb_ph: real_emb, fake_emb_ph: fake_emb}))

    result = {
        "protocol": "official Google I3D dataset FVD",
        "fvd": value,
        "num_videos": len(real_paths),
        "video_length": args.video_length,
        "i3d_batch_size": I3D_BATCH_SIZE,
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()

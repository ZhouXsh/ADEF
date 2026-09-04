#!/usr/bin/env python3
# coding=utf-8
"""Dataset-level FVD with pairwise failure exclusion.

In paired-list mode, unreadable real/fake pairs are skipped together so the
Fréchet statistics use the same successful sample subset on both sides.
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
    out = [s for raw in p.read_text(encoding="utf-8").splitlines()
           if (s := raw.strip()) and not s.startswith("#")]
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
    if not Path(path).is_file():
        raise FileNotFoundError(path)
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
    p.add_argument("--real_dir")
    p.add_argument("--fake_dir")
    p.add_argument("--real_list")
    p.add_argument("--fake_list")
    p.add_argument("--video_length", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output_file")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _write_result(args, result):
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if not args.quiet:
        print(text)
    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_file).write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    result = {
        "protocol": "official Google I3D dataset FVD",
        "fvd": None,
        "video_length": args.video_length,
        "i3d_batch_size": I3D_BATCH_SIZE,
        "failures": [],
    }
    try:
        if bool(args.real_list) != bool(args.fake_list):
            raise ValueError("--real_list and --fake_list must be supplied together")
        if args.real_list:
            real_paths = _read_list(args.real_list)
            fake_paths = _read_list(args.fake_list)
        else:
            if not args.real_dir or not args.fake_dir:
                raise ValueError("provide --real_dir/--fake_dir or --real_list/--fake_list")
            real_paths = collect_inputs(args.real_dir)
            fake_paths = collect_inputs(args.fake_dir)
        if len(real_paths) != len(fake_paths):
            raise RuntimeError("FVD protocol requires equal real/fake set sizes (%d vs %d)" %
                               (len(real_paths), len(fake_paths)))
        if args.limit:
            real_paths = real_paths[:args.limit]
            fake_paths = fake_paths[:args.limit]

        result["n_total"] = len(real_paths)
        real_arr, fake_arr = [], []
        for i, (rp, fp) in enumerate(zip(real_paths, fake_paths)):
            if not args.quiet:
                print("[FVD] %d/%d %s | %s" % (i + 1, len(real_paths), os.path.basename(rp), os.path.basename(fp)))
            try:
                rr = load_video(rp, args.video_length)
                ff = load_video(fp, args.video_length)
            except Exception as exc:
                result["failures"].append({
                    "index": i, "real": rp, "fake": fp,
                    "error": "%s: %s" % (type(exc).__name__, exc),
                })
                continue
            real_arr.append(rr)
            fake_arr.append(ff)

        result["n_success"] = len(real_arr)
        result["num_videos"] = len(real_arr)
        if len(real_arr) < 2:
            raise RuntimeError("FVD needs at least two successful real/fake video pairs")
        if len(real_arr) < 16:
            warnings.warn("FVD with fewer than 16 videos has high estimator variance; the value is valid but unstable.")

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
                result["fvd"] = float(sess.run(fvd_tensor, feed_dict={real_emb_ph: real_emb, fake_emb_ph: fake_emb}))
        _write_result(args, result)
        return 0
    except Exception as exc:
        result["global_error"] = "%s: %s" % (type(exc).__name__, exc)
        _write_result(args, result)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# coding: utf-8
from __future__ import annotations

import argparse

from eval.common.face import (
    MP_LEFT_BROW,
    MP_MOUTH_INNER,
    MP_MOUTH_OUTER,
    MP_RIGHT_BROW,
    eyebrow_motion,
    extract_landmark_sequence,
    landmark_jitter,
    mouth_opening,
    sequence_dynamics,
)
from eval.common.io import iter_video_paths, summarize, write_json


def evaluate_video(video_path: str, stride: int = 1, max_frames: int = 0):
    seq = extract_landmark_sequence(video_path, stride=stride, max_frames=max_frames)
    mouth = mouth_opening(seq.landmarks)
    brow = eyebrow_motion(seq.landmarks)
    brow_indices = MP_LEFT_BROW + MP_RIGHT_BROW
    mouth_indices = sorted(set(MP_MOUTH_OUTER + MP_MOUTH_INNER))
    return {
        "video": video_path,
        "fps": float(seq.fps),
        "frames_with_face": int(seq.landmarks.shape[0]),
        "global_landmark_jitter": landmark_jitter(seq.landmarks, seq.fps),
        "mouth_landmark_jitter": landmark_jitter(seq.landmarks, seq.fps, mouth_indices),
        "eyebrow_landmark_jitter": landmark_jitter(seq.landmarks, seq.fps, brow_indices),
        "mouth_opening_dynamics": sequence_dynamics(mouth, seq.fps),
        "eyebrow_dynamics": sequence_dynamics(brow, seq.fps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for row in iter_video_paths(args.video or None, args.manifest or None):
        video = row.get("generated") or row.get("video")
        rows.append(evaluate_video(video, args.stride, args.max_frames))

    summary = {
        "global_acceleration_mean": summarize(r["global_landmark_jitter"]["acceleration_mean"] for r in rows),
        "mouth_acceleration_mean": summarize(r["mouth_landmark_jitter"]["acceleration_mean"] for r in rows),
        "eyebrow_acceleration_mean": summarize(r["eyebrow_landmark_jitter"]["acceleration_mean"] for r in rows),
        "mouth_opening_std": summarize(r["mouth_opening_dynamics"].get("std", float("nan")) for r in rows),
        "eyebrow_motion_std": summarize(r["eyebrow_dynamics"].get("std", float("nan")) for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

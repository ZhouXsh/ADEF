# coding: utf-8
from __future__ import annotations

import argparse
from typing import List

import cv2
import numpy as np

from eval.common.face import extract_landmark_sequence, sequence_dynamics
from eval.common.io import iter_video_paths, summarize, write_json

# Six MediaPipe landmarks commonly used for approximate head pose.
# nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
MP_POSE_IDXS = [1, 152, 33, 263, 61, 291]
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, -63.6, -12.5],
    [-43.3, 32.7, -26.0],
    [43.3, 32.7, -26.0],
    [-28.9, -28.9, -24.1],
    [28.9, -28.9, -24.1],
], dtype=np.float64)


def rotation_vector_to_euler(rvec: np.ndarray):
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0
    return np.degrees([x, y, z])  # pitch, yaw, roll approximately


def estimate_pose_sequence(video_path: str, stride: int = 1, max_frames: int = 0):
    seq = extract_landmark_sequence(video_path, stride=stride, max_frames=max_frames)
    width, height = seq.image_size
    focal = width
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array([[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    angles: List[np.ndarray] = []
    for pts in seq.landmarks:
        image_points = pts[MP_POSE_IDXS, :2].copy()
        image_points[:, 0] *= width
        image_points[:, 1] *= height
        ok, rvec, _ = cv2.solvePnP(MODEL_POINTS, image_points.astype(np.float64), camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            pitch, yaw, roll = rotation_vector_to_euler(rvec)
            angles.append(np.asarray([yaw, pitch, roll], dtype=np.float64))
    if not angles:
        raise RuntimeError(f"No head pose estimated in {video_path}")
    return np.stack(angles, axis=0), seq.fps


def evaluate_video(video_path: str, stride: int = 1, max_frames: int = 0):
    angles, fps = estimate_pose_sequence(video_path, stride=stride, max_frames=max_frames)
    yaw, pitch, roll = angles[:, 0], angles[:, 1], angles[:, 2]
    if len(angles) >= 3:
        acc = np.diff(np.diff(angles, axis=0) * fps, axis=0) * fps
        pose_jitter = float(np.mean(np.linalg.norm(acc, axis=-1)))
    else:
        pose_jitter = float("nan")
    return {
        "video": video_path,
        "fps": float(fps),
        "frames_with_pose": int(len(angles)),
        "yaw": sequence_dynamics(yaw, fps),
        "pitch": sequence_dynamics(pitch, fps),
        "roll": sequence_dynamics(roll, fps),
        "pose_jitter": pose_jitter,
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
        rows.append(evaluate_video(video, stride=args.stride, max_frames=args.max_frames))
    summary = {
        "pose_jitter": summarize(r["pose_jitter"] for r in rows),
        "yaw_std": summarize(r["yaw"].get("std", float("nan")) for r in rows),
        "pitch_std": summarize(r["pitch"].get("std", float("nan")) for r in rows),
        "roll_std": summarize(r["roll"].get("std", float("nan")) for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

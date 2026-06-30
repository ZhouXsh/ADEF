# coding: utf-8
from __future__ import annotations

import argparse

import cv2
import numpy as np

from eval.common.face import detect_landmarks_frame
from eval.common.io import iter_video_paths, read_video_frames, summarize, video_info, write_json


def frame_entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def frame_metrics(frame_rgb: np.ndarray):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_mean = float(gray.mean())
    brightness_std = float(gray.std())
    entropy = frame_entropy(gray)
    pts = detect_landmarks_frame(frame_rgb)
    if pts is None:
        face_detected = 0.0
        face_area_ratio = float("nan")
    else:
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        face_detected = 1.0
        face_area_ratio = float(max(0.0, x1 - x0) * max(0.0, y1 - y0))
    return {
        "laplacian_sharpness": blur,
        "brightness_mean": brightness_mean,
        "brightness_std": brightness_std,
        "entropy": entropy,
        "face_detected": face_detected,
        "face_area_ratio": face_area_ratio,
    }


def evaluate_video(video_path: str, num_frames: int = 32):
    frames = read_video_frames(video_path, max_frames=0, stride=max(1, int(max(1, video_info(video_path)["n_frames"]) / max(1, num_frames))))
    if len(frames) > num_frames:
        idx = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
        frames = [frames[i] for i in idx]
    metrics = [frame_metrics(f) for f in frames]
    flicker = []
    for a, b in zip(frames[:-1], frames[1:]):
        aa = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
        bb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
        flicker.append(float(np.mean(np.abs(aa - bb))))
    return {
        "video": video_path,
        "sampled_frames": len(frames),
        "laplacian_sharpness": summarize(m["laplacian_sharpness"] for m in metrics),
        "brightness_mean": summarize(m["brightness_mean"] for m in metrics),
        "brightness_std": summarize(m["brightness_std"] for m in metrics),
        "entropy": summarize(m["entropy"] for m in metrics),
        "face_detection_ratio": float(np.mean([m["face_detected"] for m in metrics])) if metrics else float("nan"),
        "face_area_ratio": summarize(m["face_area_ratio"] for m in metrics),
        "frame_difference": summarize(flicker),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for row in iter_video_paths(args.video or None, args.manifest or None):
        video = row.get("generated") or row.get("video")
        rows.append(evaluate_video(video, args.num_frames))
    summary = {
        "sharpness_mean": summarize(r["laplacian_sharpness"]["mean"] for r in rows),
        "face_detection_ratio": summarize(r["face_detection_ratio"] for r in rows),
        "frame_difference_mean": summarize(r["frame_difference"]["mean"] for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

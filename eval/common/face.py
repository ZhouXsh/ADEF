# coding: utf-8
"""基于 MediaPipe FaceMesh 的人脸关键点工具函数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# MediaPipe FaceMesh 的语义子集。这里的编号来自 MediaPipe 468 点网格，
# 不是 LivePortrait 的 21 个 expression keypoint。
MP_MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MP_MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MP_LEFT_BROW = [70, 63, 105, 66, 107]
MP_RIGHT_BROW = [336, 296, 334, 293, 300]


@dataclass
class LandmarkSequence:
    landmarks: np.ndarray  # [T, N, 3]，归一化 x/y/z 坐标
    frame_indices: List[int]
    fps: float
    image_size: Tuple[int, int]


def _load_facemesh(static_image_mode: bool = False, max_num_faces: int = 1):
    import mediapipe as mp

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=static_image_mode,
        max_num_faces=max_num_faces,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect_landmarks_frame(frame_rgb: np.ndarray, facemesh=None) -> Optional[np.ndarray]:
    close = False
    if facemesh is None:
        facemesh = _load_facemesh(static_image_mode=True)
        close = True
    try:
        result = facemesh.process(frame_rgb)
        if not result.multi_face_landmarks:
            return None
        pts = result.multi_face_landmarks[0].landmark
        arr = np.asarray([[p.x, p.y, p.z] for p in pts], dtype=np.float32)
        return arr
    finally:
        if close:
            facemesh.close()


def extract_landmark_sequence(video_path: str, stride: int = 1, max_frames: int = 0) -> LandmarkSequence:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) / max(1, stride)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    facemesh = _load_facemesh(static_image_mode=False)
    landmarks: List[np.ndarray] = []
    frame_indices: List[int] = []
    idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if idx % max(1, stride) == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pts = detect_landmarks_frame(frame_rgb, facemesh)
                if pts is not None:
                    landmarks.append(pts)
                    frame_indices.append(idx)
                    if max_frames and len(landmarks) >= max_frames:
                        break
            idx += 1
    finally:
        facemesh.close()
        cap.release()
    if not landmarks:
        raise RuntimeError(f"No face landmarks detected in {video_path}")
    return LandmarkSequence(np.stack(landmarks, axis=0), frame_indices, fps, (width, height))


def pairwise_distance(points: np.ndarray, a: int, b: int, xy_only: bool = True) -> np.ndarray:
    p1 = points[:, a, :2 if xy_only else 3]
    p2 = points[:, b, :2 if xy_only else 3]
    return np.linalg.norm(p1 - p2, axis=-1)


def mouth_opening(landmarks: np.ndarray) -> np.ndarray:
    """根据上唇 / 下唇距离近似估计嘴部开合程度。"""
    vertical_1 = pairwise_distance(landmarks, 13, 14)
    vertical_2 = pairwise_distance(landmarks, 82, 87)
    vertical_3 = pairwise_distance(landmarks, 312, 317)
    width = pairwise_distance(landmarks, 61, 291)
    opening = (vertical_1 + vertical_2 + vertical_3) / 3.0
    return opening / (width + 1e-6)


def eyebrow_motion(landmarks: np.ndarray) -> np.ndarray:
    left = landmarks[:, MP_LEFT_BROW, :2].mean(axis=1)
    right = landmarks[:, MP_RIGHT_BROW, :2].mean(axis=1)
    eye_l = landmarks[:, MP_LEFT_EYE, :2].mean(axis=1)
    eye_r = landmarks[:, MP_RIGHT_EYE, :2].mean(axis=1)
    return 0.5 * (np.linalg.norm(left - eye_l, axis=-1) + np.linalg.norm(right - eye_r, axis=-1))


def sequence_dynamics(signal: np.ndarray, fps: float) -> Dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    if signal.size < 3:
        return {"mean": float(np.mean(signal)) if signal.size else float("nan"), "std": float(np.std(signal)) if signal.size else float("nan")}
    vel = np.diff(signal) * fps
    acc = np.diff(vel) * fps
    return {
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "velocity_mean_abs": float(np.mean(np.abs(vel))),
        "velocity_std": float(np.std(vel)),
        "acceleration_mean_abs": float(np.mean(np.abs(acc))),
        "jitter": float(np.mean(np.abs(acc))),
    }


def landmark_jitter(landmarks: np.ndarray, fps: float, indices: Optional[Sequence[int]] = None) -> Dict[str, float]:
    pts = landmarks[:, indices, :2] if indices is not None else landmarks[:, :, :2]
    if pts.shape[0] < 3:
        return {"velocity_mean": float("nan"), "acceleration_mean": float("nan")}
    vel = np.diff(pts, axis=0) * fps
    acc = np.diff(vel, axis=0) * fps
    return {
        "velocity_mean": float(np.linalg.norm(vel, axis=-1).mean()),
        "velocity_std": float(np.linalg.norm(vel, axis=-1).std()),
        "acceleration_mean": float(np.linalg.norm(acc, axis=-1).mean()),
        "acceleration_std": float(np.linalg.norm(acc, axis=-1).std()),
    }

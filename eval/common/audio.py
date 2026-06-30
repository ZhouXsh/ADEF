# coding: utf-8
"""轻量级说话人脸评估使用的音频工具函数。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np

from .io import extract_audio


def load_audio_from_video_or_wav(path: str, sr: int = 16000) -> Tuple[np.ndarray, int, str]:
    path_obj = Path(path)
    if path_obj.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".aac"}:
        wav_path = str(path_obj)
    else:
        wav_path = extract_audio(path_obj, sr=sr)
    audio, sr = librosa.load(wav_path, sr=sr, mono=True)
    return audio.astype(np.float32), sr, wav_path


def rms_envelope(audio: np.ndarray, sr: int, fps: float) -> np.ndarray:
    hop = max(1, int(round(sr / fps)))
    frame_length = max(hop * 2, 512)
    rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop, center=True)[0]
    return rms.astype(np.float32)


def mfcc_energy_envelope(audio: np.ndarray, sr: int, fps: float, n_mfcc: int = 13) -> np.ndarray:
    hop = max(1, int(round(sr / fps)))
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop)
    # 语音动态代理特征：逐帧 MFCC 变化幅度。
    delta = np.diff(mfcc, axis=1, prepend=mfcc[:, :1])
    env = np.linalg.norm(delta, axis=0)
    return env.astype(np.float32)


def align_1d(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return np.asarray(a[:n], dtype=np.float64), np.asarray(b[:n], dtype=np.float64)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = align_1d(a, b)
    if len(a) < 3:
        return float("nan")
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 5) -> Dict[str, float]:
    """在整数帧偏移范围内返回最佳相关性。

    正 lag 表示 b 相对于 a 向后移动。
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    best = {"corr": float("nan"), "lag": 0}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa, bb = a[-lag:], b[:len(b) + lag]
        elif lag > 0:
            aa, bb = a[:len(a) - lag], b[lag:]
        else:
            aa, bb = a, b
        aa, bb = align_1d(aa, bb)
        corr = safe_corr(aa, bb)
        if np.isfinite(corr) and (not np.isfinite(best["corr"]) or corr > best["corr"]):
            best = {"corr": corr, "lag": lag}
    return {"best_corr": float(best["corr"]), "best_lag_frames": int(best["lag"])}

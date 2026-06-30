# coding: utf-8
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval.common.audio import lagged_corr, load_audio_from_video_or_wav, mfcc_energy_envelope, rms_envelope, safe_corr
from eval.common.face import extract_landmark_sequence, mouth_opening, sequence_dynamics
from eval.common.io import iter_video_paths, summarize, write_json


def evaluate_video(video_path: str, audio_path: str | None = None, max_lag: int = 5):
    seq = extract_landmark_sequence(video_path)
    mouth = mouth_opening(seq.landmarks)
    audio, sr, used_audio_path = load_audio_from_video_or_wav(audio_path or video_path, sr=16000)
    rms = rms_envelope(audio, sr, seq.fps)
    mfcc_env = mfcc_energy_envelope(audio, sr, seq.fps)

    n = min(len(mouth), len(rms), len(mfcc_env))
    mouth = mouth[:n]
    rms = rms[:n]
    mfcc_env = mfcc_env[:n]

    return {
        "video": video_path,
        "audio": used_audio_path,
        "frames_with_face": int(len(mouth)),
        "fps": float(seq.fps),
        "mouth_audio_rms_corr": safe_corr(mouth, rms),
        "mouth_audio_mfcc_corr": safe_corr(mouth, mfcc_env),
        "best_lag_rms": lagged_corr(mouth, rms, max_lag=max_lag),
        "best_lag_mfcc": lagged_corr(mouth, mfcc_env, max_lag=max_lag),
        "mouth_opening": sequence_dynamics(mouth, seq.fps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--audio", type=str, default="", help="optional external wav/audio path")
    parser.add_argument("--manifest", type=str, default="", help="txt or csv with generated[,audio] columns")
    parser.add_argument("--max_lag", type=int, default=5)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = []
    for row in iter_video_paths(args.video or None, args.manifest or None):
        video = row.get("generated") or row.get("video")
        audio = row.get("audio") or args.audio or None
        rows.append(evaluate_video(video, audio, max_lag=args.max_lag))

    summary = {
        "mouth_audio_rms_corr": summarize(r["mouth_audio_rms_corr"] for r in rows),
        "mouth_audio_mfcc_corr": summarize(r["mouth_audio_mfcc_corr"] for r in rows),
        "best_lag_rms_corr": summarize(r["best_lag_rms"]["best_corr"] for r in rows),
        "best_lag_mfcc_corr": summarize(r["best_lag_mfcc"]["best_corr"] for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

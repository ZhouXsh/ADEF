# coding: utf-8
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from eval.common.io import iter_video_paths, sample_frames, summarize, write_json


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def load_image_or_video_frames(path: str, num_frames: int):
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(path)
        return [cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)]
    return sample_frames(path, num_frames=num_frames, rgb=True)


def init_app(ctx_id: int):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def extract_embeddings(app, frames) -> List[np.ndarray]:
    embs = []
    for rgb in frames:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            continue
        faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        emb = faces[0].normed_embedding
        if emb is not None:
            embs.append(np.asarray(emb, dtype=np.float32))
    return embs


def evaluate_item(app, row, num_frames: int):
    video = row.get("generated") or row.get("video")
    reference = row.get("reference") or row.get("source") or ""
    gen_frames = load_image_or_video_frames(video, num_frames)
    gen_embs = extract_embeddings(app, gen_frames)
    if not gen_embs:
        return {"video": video, "reference": reference, "detected_frames": 0, "identity_cosine_mean": float("nan"), "identity_cosine_std": float("nan")}

    if reference:
        ref_frames = load_image_or_video_frames(reference, num_frames=8)
        ref_embs = extract_embeddings(app, ref_frames)
        if not ref_embs:
            ref_emb = gen_embs[0]
        else:
            ref_emb = np.mean(np.stack(ref_embs, axis=0), axis=0)
    else:
        ref_emb = gen_embs[0]

    scores = [cosine(ref_emb, emb) for emb in gen_embs]
    return {
        "video": video,
        "reference": reference,
        "detected_frames": len(gen_embs),
        "identity_cosine_mean": float(np.mean(scores)),
        "identity_cosine_std": float(np.std(scores)),
        "identity_cosine_min": float(np.min(scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--ctx_id", type=int, default=0, help="InsightFace ctx_id; use -1 for CPU")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    app = init_app(args.ctx_id)
    rows = [evaluate_item(app, row, args.num_frames) for row in iter_video_paths(args.video or None, args.manifest or None)]
    summary = {
        "identity_cosine_mean": summarize(r["identity_cosine_mean"] for r in rows),
        "identity_cosine_std": summarize(r["identity_cosine_std"] for r in rows),
        "detected_frames": summarize(r["detected_frames"] for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

# coding: utf-8
from __future__ import annotations

import argparse
from typing import Iterable, List

import numpy as np

from eval.common.face import (
    MP_LEFT_BROW,
    MP_LEFT_EYE,
    MP_MOUTH_INNER,
    MP_MOUTH_OUTER,
    MP_RIGHT_BROW,
    MP_RIGHT_EYE,
    extract_landmark_sequence,
)
from eval.common.io import read_manifest, summarize, write_json


def unique_ids(ids: Iterable[int]) -> List[int]:
    return sorted(set(int(x) for x in ids))


GROUPS = {
    "full_lmd": None,
    "mouth_lmd": unique_ids(MP_MOUTH_OUTER + MP_MOUTH_INNER),
    "outer_mouth_lmd": unique_ids(MP_MOUTH_OUTER),
    "inner_mouth_lmd": unique_ids(MP_MOUTH_INNER),
    "brow_lmd": unique_ids(MP_LEFT_BROW + MP_RIGHT_BROW),
    "eye_lmd": unique_ids(MP_LEFT_EYE + MP_RIGHT_EYE),
}


def get_normalizer(ref_pts: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones((ref_pts.shape[0],), dtype=np.float64)
    if mode == "face_width":
        x = ref_pts[:, :, 0]
        return np.maximum(x.max(axis=1) - x.min(axis=1), 1e-6)
    if mode == "interocular":
        left = ref_pts[:, MP_LEFT_EYE, :2].mean(axis=1)
        right = ref_pts[:, MP_RIGHT_EYE, :2].mean(axis=1)
        return np.maximum(np.linalg.norm(left - right, axis=-1), 1e-6)
    raise ValueError(f"未知 normalize 模式：{mode}")


def align_landmarks(gen_seq, ref_seq):
    """按原始帧号对齐；若无交集，则退化为顺序截断对齐。"""
    gen_map = {idx: i for i, idx in enumerate(gen_seq.frame_indices)}
    ref_map = {idx: i for i, idx in enumerate(ref_seq.frame_indices)}
    common = sorted(set(gen_map).intersection(ref_map))
    if common:
        gen = np.stack([gen_seq.landmarks[gen_map[i]] for i in common], axis=0)
        ref = np.stack([ref_seq.landmarks[ref_map[i]] for i in common], axis=0)
        return gen, ref, len(common), "frame_index"
    n = min(gen_seq.landmarks.shape[0], ref_seq.landmarks.shape[0])
    return gen_seq.landmarks[:n], ref_seq.landmarks[:n], n, "sequential_fallback"


def lmd_group(gen_pts: np.ndarray, ref_pts: np.ndarray, ids, norm: np.ndarray):
    if ids is None:
        gen_xy = gen_pts[:, :, :2]
        ref_xy = ref_pts[:, :, :2]
    else:
        gen_xy = gen_pts[:, ids, :2]
        ref_xy = ref_pts[:, ids, :2]
    dist = np.linalg.norm(gen_xy - ref_xy, axis=-1) / norm[:, None]
    per_frame = dist.mean(axis=1)
    return {
        "mean": float(per_frame.mean()),
        "std": float(per_frame.std()),
        "min": float(per_frame.min()),
        "max": float(per_frame.max()),
        "frames": int(per_frame.shape[0]),
    }


def evaluate_pair(generated: str, reference: str, normalize: str, stride: int, max_frames: int):
    gen_seq = extract_landmark_sequence(generated, stride=stride, max_frames=max_frames)
    ref_seq = extract_landmark_sequence(reference, stride=stride, max_frames=max_frames)
    gen_pts, ref_pts, n_aligned, align_mode = align_landmarks(gen_seq, ref_seq)
    if n_aligned <= 0:
        raise RuntimeError("没有可对齐的关键点帧")
    norm = get_normalizer(ref_pts, normalize)
    metrics = {name: lmd_group(gen_pts, ref_pts, ids, norm) for name, ids in GROUPS.items()}
    return {
        "generated": generated,
        "reference": reference,
        "normalize": normalize,
        "stride": int(stride),
        "align_mode": align_mode,
        "aligned_frames": int(n_aligned),
        "generated_detected_frames": int(gen_seq.landmarks.shape[0]),
        "reference_detected_frames": int(ref_seq.landmarks.shape[0]),
        **metrics,
    }


def collect_rows(args):
    rows = []
    if args.generated and args.reference:
        rows.append({"generated": args.generated, "reference": args.reference})
    if args.manifest:
        rows.extend(read_manifest(args.manifest))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=str, default="", help="单个生成视频路径")
    parser.add_argument("--reference", type=str, default="", help="单个参考或 GT 视频路径")
    parser.add_argument("--manifest", type=str, default="", help="包含 generated,reference 列的 CSV")
    parser.add_argument("--normalize", type=str, default="face_width", choices=["face_width", "interocular", "none"])
    parser.add_argument("--stride", type=int, default=1, help="按帧抽样间隔")
    parser.add_argument("--max_frames", type=int, default=0, help="最多处理多少个成功检测帧，0 表示不限制")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    items = []
    for row in collect_rows(args):
        generated = row.get("generated") or row.get("video")
        reference = row.get("reference") or row.get("gt")
        if not generated or not reference:
            items.append({"generated": generated, "reference": reference, "error": "缺少 generated 或 reference 字段"})
            continue
        try:
            items.append(evaluate_pair(generated, reference, args.normalize, args.stride, args.max_frames))
        except Exception as exc:
            items.append({"generated": generated, "reference": reference, "error": str(exc)})

    valid = [x for x in items if "error" not in x]
    summary = {name: summarize(x[name]["mean"] for x in valid) for name in GROUPS}
    summary["aligned_frames"] = summarize(x["aligned_frames"] for x in valid)
    write_json({"summary": summary, "items": items}, args.out)


if __name__ == "__main__":
    main()

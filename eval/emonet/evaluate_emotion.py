"""
Emotion evaluation for talking-head generation using EmoNet.

Each --gen / --gt path may be either a single video file or a folder of
videos. With both --gen and --gt supplied (and at least one folder), videos
are paired by filename stem and the script reports:

  * Emo-Acc  : agreement rate of the argmax discrete expression class
  * Valence / Arousal : CCC, PCC, RMSE, SAGR between gen and GT
  * Emo-SIM  : mean cosine similarity of the EmoNet emotion embedding

With only --gen (single-video mode), comparison metrics are skipped and only
the per-frame EmoNet outputs (discrete emotion, valence, arousal) are saved.

Usage:
  # folder vs folder (existing behavior)
  python evaluate_emotion.py --gen path/to/generated --gt path/to/gt

  # single video vs single GT video
  python evaluate_emotion.py --gen result.mp4 --gt reference.mp4

  # single generated video, no GT (just per-frame EmoNet outputs)
  python evaluate_emotion.py --gen result.mp4
"""

from pathlib import Path
from typing import Dict, List, Optional
import argparse
import json

import numpy as np
import torch
from torch import nn
import cv2

from face_alignment.detection.sfd.sfd_detector import SFDDetector
from emonet.models import EmoNet
from emonet.metrics import CCC, PCC, RMSE, SAGR


IMAGE_SIZE = 256
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
EMOTION_CLASSES = {
    0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
    4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt",
}


def load_emonet(n_expression: int, device: str) -> EmoNet:
    state_dict_path = Path(__file__).parent / "pretrained" / f"emonet_{n_expression}.pth"
    print(f"Loading EmoNet from {state_dict_path}")
    state_dict = torch.load(str(state_dict_path), map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=n_expression).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


def load_video_frames(video_path: Path) -> List[np.ndarray]:
    """Reads all frames of a video as RGB uint8 arrays."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


class EmoNetRunner:
    """Wraps EmoNet + SFD face detector and extracts per-frame predictions."""

    def __init__(self, net: EmoNet, detector: SFDDetector, device: str, batch_size: int = 32):
        self.net = net
        self.detector = detector
        self.device = device
        self.batch_size = batch_size
        # Capture the 256-d embedding fed into the final classifier as the
        # emotion descriptor used for Emo-SIM.
        self._embeddings: Optional[torch.Tensor] = None
        self.net.emo_fc_2.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self._embeddings = inputs[0].detach()

    @torch.no_grad()
    def _detect_face_crop(self, frame_rgb: np.ndarray) -> Optional[np.ndarray]:
        # SFD detector expects a BGR image.
        detected = self.detector.detect_from_image(frame_rgb[:, :, ::-1].copy())
        if len(detected) == 0:
            return None
        bbox = np.array(detected[0][:4]).astype(np.int32)
        x1, y1, x2, y2 = bbox
        x1, y1 = max(x1, 0), max(y1, 0)
        x2 = min(x2, frame_rgb.shape[1])
        y2 = min(y2, frame_rgb.shape[0])
        if x2 <= x1 or y2 <= y1:
            return None
        return frame_rgb[y1:y2, x1:x2, :]

    @torch.no_grad()
    def _run_batch(self, crops: List[np.ndarray]) -> Dict[str, np.ndarray]:
        tensors = []
        for crop in crops:
            resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            tensors.append(torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0)
        batch = torch.stack(tensors).to(self.device)
        out = self.net(batch)
        expr = torch.argmax(nn.functional.softmax(out["expression"], dim=1), dim=1)
        return {
            "expression": expr.cpu().numpy(),
            "valence": out["valence"].clamp(-1.0, 1.0).cpu().numpy(),
            "arousal": out["arousal"].clamp(-1.0, 1.0).cpu().numpy(),
            "embedding": self._embeddings.cpu().numpy(),
        }

    @torch.no_grad()
    def process_video(self, video_path: Path) -> Dict[int, Dict]:
        """Returns {frame_idx: {expression, valence, arousal, embedding}} for
        frames where a face was detected."""
        frames = load_video_frames(video_path)
        kept_idx, crops = [], []
        for idx, frame in enumerate(frames):
            crop = self._detect_face_crop(frame)
            if crop is not None:
                kept_idx.append(idx)
                crops.append(crop)

        results: Dict[int, Dict] = {}
        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start:start + self.batch_size]
            batch_idx = kept_idx[start:start + self.batch_size]
            out = self._run_batch(batch_crops)
            for i, fidx in enumerate(batch_idx):
                results[fidx] = {
                    "expression": int(out["expression"][i]),
                    "valence": float(out["valence"][i]),
                    "arousal": float(out["arousal"][i]),
                    "embedding": out["embedding"][i],
                }
        return results


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def compare_video(gen: Dict[int, Dict], gt: Dict[int, Dict]) -> Optional[Dict]:
    """Aligns gen/gt by shared frame indices where both have a detected face."""
    common = sorted(set(gen.keys()) & set(gt.keys()))
    if not common:
        return None

    gen_expr = np.array([gen[i]["expression"] for i in common])
    gt_expr = np.array([gt[i]["expression"] for i in common])
    gen_val = np.array([gen[i]["valence"] for i in common])
    gt_val = np.array([gt[i]["valence"] for i in common])
    gen_aro = np.array([gen[i]["arousal"] for i in common])
    gt_aro = np.array([gt[i]["arousal"] for i in common])
    emo_sims = [cosine_sim(gen[i]["embedding"], gt[i]["embedding"]) for i in common]

    return {
        "n_frames": len(common),
        "emo_acc": float(np.mean(gen_expr == gt_expr)),
        "emo_sim": float(np.mean(emo_sims)),
        "valence": _pair_metrics(gt_val, gen_val),
        "arousal": _pair_metrics(gt_aro, gen_aro),
        # raw arrays kept for global aggregation
        "_raw": {
            "gen_expr": gen_expr, "gt_expr": gt_expr,
            "gen_val": gen_val, "gt_val": gt_val,
            "gen_aro": gen_aro, "gt_aro": gt_aro,
            "emo_sims": emo_sims,
        },
    }


def _pair_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    out = {"rmse": float(RMSE(gt, pred)), "sagr": float(SAGR(gt, pred))}
    # CCC/PCC need variance in both signals.
    if len(gt) > 1 and np.std(gt) > 1e-6 and np.std(pred) > 1e-6:
        out["ccc"] = float(CCC(gt, pred))
        out["pcc"] = float(PCC(gt, pred))
    else:
        out["ccc"] = float("nan")
        out["pcc"] = float("nan")
    return out


def expand_video_inputs(paths: List[str]) -> List[Path]:
    """Accept a mix of file paths and directories; return sorted list of video files."""
    out: List[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(q for q in p.rglob("*") if q.suffix.lower() in VIDEO_EXTS))
        else:
            print(f"  WARNING: skipping unrecognized path: {raw}")
    return out


def find_pairs(gen_paths: List[str], gt_paths: List[str], mode: str = "positional") -> List[tuple]:
    """Pair gen videos with gt videos.

    mode = 'positional' : zip in input order (after expanding dirs).
                           One --gen and one --gt pair directly; multi-file
                           lists pair by index.
    mode = 'stem'       : match by filename stem (original behavior).

    Unmatched gen videos are returned with None as the GT.
    """
    gens = expand_video_inputs(gen_paths)
    gts = expand_video_inputs(gt_paths)

    if mode == "positional":
        if not gts:
            return [(g, None) for g in gens]
        if len(gens) != len(gts):
            print(f"  WARNING: positional mode expects equal counts; got "
                  f"{len(gens)} gen vs {len(gts)} gt. Falling back to min length.")
        return [(g, t) for g, t in zip(gens, gts)]
    elif mode == "stem":
        gt_map = {p.stem: p for p in gts}
        return [(g, gt_map.get(g.stem)) for g in gens]
    else:
        raise ValueError(f"Unknown pair mode: {mode}")


def summarize_single_video(gen_res: Dict[int, Dict]) -> Dict:
    """Per-frame EmoNet outputs for one video (no GT comparison)."""
    frames = sorted(gen_res.keys())
    return {
        "n_frames_with_face": len(frames),
        "frames": [
            {
                "frame_idx": i,
                "expression": gen_res[i]["expression"],
                "expression_name": EMOTION_CLASSES[gen_res[i]["expression"]],
                "valence": gen_res[i]["valence"],
                "arousal": gen_res[i]["arousal"],
            }
            for i in frames
        ],
        "mean_valence": float(np.mean([gen_res[i]["valence"] for i in frames])) if frames else None,
        "mean_arousal": float(np.mean([gen_res[i]["arousal"] for i in frames])) if frames else None,
        "emotion_histogram": _histogram([gen_res[i]["expression"] for i in frames]),
    }


def _histogram(expr_list: List[int]) -> Dict[str, int]:
    counts = {EMOTION_CLASSES[i]: 0 for i in EMOTION_CLASSES}
    for e in expr_list:
        counts[EMOTION_CLASSES[e]] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="EmoNet emotion evaluation (paired gen vs GT, or single video).")
    parser.add_argument("--gen", action="append", required=True,
                        help="Generated video file or folder. Repeat to pass multiple paths.")
    parser.add_argument("--gt", action="append", default=[],
                        help="Ground-truth video file or folder. Repeat to pass multiple paths. "
                             "Omit for single-video mode (per-frame outputs only).")
    parser.add_argument("--pair_mode", choices=["positional", "stem"], default="positional",
                        help="How to match gen videos with gt videos. "
                             "'positional' (default): pair in input order, so --gen v1.mp4 --gt v2.mp4 "
                             "compares them directly regardless of filename. "
                             "'stem': match by filename stem (original behavior).")
    parser.add_argument("--nclasses", type=int, default=8, choices=[5, 8])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="emonet_eval_results.json")
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    pairs = find_pairs(args.gen, args.gt, mode=args.pair_mode)
    if not pairs:
        raise SystemExit(f"No videos found among --gen inputs: {args.gen}")
    paired = [(g, t) for g, t in pairs if t is not None]
    unpaired = [g for g, t in pairs if t is None]
    use_single_mode = not args.gt
    if use_single_mode:
        print(f"Single-video mode: {len(pairs)} video(s), no GT comparison.")
    else:
        print(f"Found {len(paired)} paired videos ({len(unpaired)} unpaired, skipped).")
        for u in unpaired:
            print(f"  WARNING: no GT match for {u.name}, skipping.")
        if not paired:
            raise SystemExit("No gen/gt pairs matched. Pass matching filenames, "
                             "or omit --gt for single-video mode.")

    net = load_emonet(args.nclasses, args.device)
    detector = SFDDetector(args.device)
    runner = EmoNetRunner(net, detector, args.device, args.batch_size)

    per_video = {}
    agg = {"gen_expr": [], "gt_expr": [], "gen_val": [], "gt_val": [],
           "gen_aro": [], "gt_aro": [], "emo_sims": []}

    todo = [(g, None) for g, _ in pairs] if use_single_mode else paired
    for i, (gen_path, gt_path) in enumerate(todo):
        print(f"[{i + 1}/{len(todo)}] {gen_path.name}")
        gen_res = runner.process_video(gen_path)
        if gt_path is not None:
            gt_res = runner.process_video(gt_path)
            comp = compare_video(gen_res, gt_res)
            if comp is None:
                print(f"  WARNING: no aligned frames with faces, skipping.")
                continue
            raw = comp.pop("_raw")
            for k in agg:
                agg[k].append(raw[k])
            per_video[gen_path.name] = comp
            print(f"  frames={comp['n_frames']} emo_acc={comp['emo_acc']:.3f} "
                  f"emo_sim={comp['emo_sim']:.3f} val_ccc={comp['valence']['ccc']:.3f} "
                  f"aro_ccc={comp['arousal']['ccc']:.3f}")
        else:
            per_video[gen_path.name] = summarize_single_video(gen_res)
            print(f"  frames_with_face={per_video[gen_path.name]['n_frames_with_face']}")

    results = {"config": vars(args), "per_video": per_video}

    if agg["gen_expr"] and paired:
        gen_expr = np.concatenate(agg["gen_expr"])
        gt_expr = np.concatenate(agg["gt_expr"])
        gen_val = np.concatenate(agg["gen_val"])
        gt_val = np.concatenate(agg["gt_val"])
        gen_aro = np.concatenate(agg["gen_aro"])
        gt_aro = np.concatenate(agg["gt_aro"])
        emo_sims = np.concatenate([np.array(x) for x in agg["emo_sims"]])

        overall = {
            "n_videos": len(per_video),
            "n_frames": int(len(gen_expr)),
            "emo_acc": float(np.mean(gen_expr == gt_expr)),
            "emo_sim": float(np.mean(emo_sims)),
            "valence": _pair_metrics(gt_val, gen_val),
            "arousal": _pair_metrics(gt_aro, gen_aro),
            "emo_acc_video_mean": float(np.mean([v["emo_acc"] for v in per_video.values()])),
            "emo_sim_video_mean": float(np.mean([v["emo_sim"] for v in per_video.values()])),
        }
        results["overall"] = overall

        print("\n" + "=" * 60)
        print("OVERALL (pooled over all frames)")
        print("=" * 60)
        print(f"  videos           : {overall['n_videos']}")
        print(f"  frames           : {overall['n_frames']}")
        print(f"  Emo-Acc          : {overall['emo_acc']:.4f}")
        print(f"  Emo-SIM (cosine) : {overall['emo_sim']:.4f}")
        v, a = overall["valence"], overall["arousal"]
        print(f"  Valence  CCC={v['ccc']:.4f} PCC={v['pcc']:.4f} RMSE={v['rmse']:.4f} SAGR={v['sagr']:.4f}")
        print(f"  Arousal  CCC={a['ccc']:.4f} PCC={a['pcc']:.4f} RMSE={a['rmse']:.4f} SAGR={a['sagr']:.4f}")
    else:
        print("\nNo pairwise comparison (no GT provided or no aligned frames). "
              "Per-video EmoNet outputs saved to JSON.")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved detailed results to {args.output}")


if __name__ == "__main__":
    main()

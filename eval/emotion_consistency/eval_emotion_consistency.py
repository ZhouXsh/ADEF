# coding: utf-8
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from eval.common.io import iter_video_paths, sample_frames, summarize, write_json


def entropy_from_scores(scores):
    vals = np.asarray(list(scores), dtype=np.float64)
    vals = vals / (vals.sum() + 1e-12)
    vals = vals[vals > 0]
    return float(-(vals * np.log(vals + 1e-12)).sum())


def run_external(template: str, video: str, label: str):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        out = tmp.name
    cmd = template.format(video=video, label=label or "", out=out)
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = {}
    p = Path(out)
    if p.exists():
        try:
            result = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            result = {"raw_output": p.read_text(encoding="utf-8", errors="ignore")}
    result.update({"returncode": proc.returncode, "stdout_tail": proc.stdout[-1000:], "stderr_tail": proc.stderr[-1000:]})
    return result


def evaluate_with_hf(video: str, label: str, pipe, num_frames: int):
    frames = sample_frames(video, num_frames=num_frames, rgb=True)
    target_conf = []
    top1 = []
    entropies = []
    predictions = []
    for frame in frames:
        pred = pipe(Image.fromarray(frame))
        if isinstance(pred, dict):
            pred = [pred]
        pred = sorted(pred, key=lambda x: x.get("score", 0.0), reverse=True)
        score_map = {str(x["label"]).lower(): float(x["score"]) for x in pred}
        label_l = label.lower()
        conf = score_map.get(label_l, 0.0)
        if conf == 0.0:
            # 有些模型会输出 LABEL_0 这类类别名。即使标签无法精确匹配，
            # 也尽量让 top1 ratio 和 target confidence 的统计保持可用。
            for k, v in score_map.items():
                if label_l in k or k in label_l:
                    conf = max(conf, v)
        target_conf.append(conf)
        top1.append(1.0 if pred and label_l in str(pred[0]["label"]).lower() else 0.0)
        entropies.append(entropy_from_scores(score_map.values()))
        predictions.append(pred[:5])
    return {
        "video": video,
        "label": label,
        "sampled_frames": len(frames),
        "target_confidence_mean": float(np.mean(target_conf)) if target_conf else float("nan"),
        "target_top1_ratio": float(np.mean(top1)) if top1 else float("nan"),
        "emotion_entropy_mean": float(np.mean(entropies)) if entropies else float("nan"),
        "top_predictions_first_frame": predictions[0] if predictions else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--hf_model", type=str, default="")
    parser.add_argument("--external_cmd", type=str, default="", help="包含 {video},{label},{out} 的命令模板")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    pipe = None
    if args.hf_model:
        from transformers import pipeline
        pipe = pipeline("image-classification", model=args.hf_model, device=args.device)

    rows = []
    for row in iter_video_paths(args.video or None, args.manifest or None):
        video = row.get("generated") or row.get("video")
        label = row.get("label") or args.label
        if args.external_cmd:
            item = run_external(args.external_cmd, video, label)
            item.update({"video": video, "label": label})
        elif pipe is not None:
            item = evaluate_with_hf(video, label, pipe, args.num_frames)
        else:
            raise RuntimeError("请提供 --hf_model 或 --external_cmd")
        rows.append(item)

    summary = {
        "target_confidence_mean": summarize(r.get("target_confidence_mean", float("nan")) for r in rows),
        "target_top1_ratio": summarize(r.get("target_top1_ratio", float("nan")) for r in rows),
        "emotion_entropy_mean": summarize(r.get("emotion_entropy_mean", float("nan")) for r in rows),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

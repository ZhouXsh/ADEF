#!/usr/bin/env python3
"""
Unified emotion evaluation using both EmotiEffLib and DFER-CLIP.

For each video (or every video in a directory), runs both models and produces:
  - per-model dominant emotion
  - agreement flag (do both models agree?)
  - per-model probs/distribution
  - accuracy against GT labels when provided

Each model runs in its own subprocess so a crash in one (e.g. missing checkpoint)
does not take down the other. All output goes into a single JSON document.

Usage:
    # single video, both models
    python evaluate_unified.py --video /path/to/video.mp4 --label happiness

    # batch over a directory with a manifest
    python evaluate_unified.py --video_dir /path/to/videos/ --label_file labels.txt

    # only EmotiEffLib
    python evaluate_unified.py --video a.mp4 --models emotiefflib

    # only DFER-CLIP
    python evaluate_unified.py --video a.mp4 --models dfer_clip

    # explicitly point at weights
    python evaluate_unified.py --video a.mp4 \
        --clip_weights /path/to/ViT-B-32.pt \
        --dfer_weights /path/to/DFEW_fold1.pth

    # write JSON to disk
    python evaluate_unified.py --video_dir /path/to/videos/ --output eval_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent


def run_subprocess(script: Path, args: list[str]) -> dict[str, Any]:
    """Invoke one of the per-model scripts and parse its stdout JSON.

    The per-model scripts may print log lines (`[EmotiEffLib] loading ...`,
    `  -> dominant=...`, etc.) before the final JSON dump. We locate the JSON
    object by scanning for a line that starts with ``{`` (rather than ``naive
    find('{')`` which would catch any dict-repr in the log).
    """
    cmd = [sys.executable, str(script), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout.strip()
    err = proc.stderr.strip()
    if not out:
        return {"error": f"empty output (exit={proc.returncode})", "stderr": err}
    # Find the first '\n{' or a leading '{' that begins a JSON document.
    json_start = -1
    for candidate in range(len(out)):
        if out[candidate] == '{' and (candidate == 0 or out[candidate - 1] == '\n'):
            json_start = candidate
            break
    if json_start < 0:
        return {"error": "no JSON document found in stdout", "raw": out[-2000:], "stderr": err}
    payload = out[json_start:]
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return {"error": "could not parse JSON", "raw": payload[-2000:], "stderr": err}


def collect_videos(args) -> tuple[list[Path], dict[str, str]]:
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}
    if args.video:
        paths = [Path(args.video)]
        label_map = {paths[0].stem: args.label} if args.label else {}
        return paths, label_map
    if args.video_dir:
        root = Path(args.video_dir)
        paths = sorted(p for p in root.rglob("*")
                       if p.suffix.lower() in exts and p.is_file())
        label_map = {}
        if args.label_file:
            with open(args.label_file) as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        label_map[parts[0]] = " ".join(parts[1:])
        return paths, label_map
    raise SystemExit("Provide --video or --video_dir")


def build_per_model_args(model: str, vp: Path, lbl: str | None, args) -> list[str]:
    # Note: we deliberately do NOT pass --output to the per-model scripts.
    # They print their JSON to stdout, which we parse in this parent process.
    common = ["--video", str(vp), "--quiet"]
    if lbl is not None:
        common += ["--label", lbl]
    if model == "emotiefflib":
        ret = ["--model", args.emotieff_model,
               "--device", args.emotieff_device,
               "--frame_stride", str(args.frame_stride),
               *common]
        if getattr(args, "no_face_detect", False):
            ret.append("--no_face_detect")
        return ret
    if model == "dfer_clip":
        return ["--clip_weights", args.clip_weights,
                "--dfer_weights", args.dfer_weights,
                "--device", args.dfer_device,
                "--num_segments", str(args.num_segments),
                *common]
    raise ValueError(model)


def merge_emotion_to_canonical(emotion: str | None) -> str | None:
    """Map DFER-CLIP / EmotiEffLib class names onto a shared lowercase label."""
    if not emotion:
        return None
    e = emotion.strip().lower()
    synonyms = {
        "happy": "happiness",
        "happiness": "happiness",
        "sad": "sadness",
        "sadness": "sadness",
        "angry": "anger",
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "surprise": "surprise",
        "neutral": "neutral",
        "contempt": "contempt",
    }
    return synonyms.get(e, e)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=str, help="Single video path")
    p.add_argument("--video_dir", type=str, help="Directory of videos")
    p.add_argument("--label_file", type=str, help="Optional `<stem> <label>` per line")
    p.add_argument("--label", type=str, help="GT label for --video mode")
    p.add_argument("--models", type=str, default="emotiefflib,dfer_clip",
                   help="Comma-separated subset of {emotiefflib, dfer_clip}")
    p.add_argument("--emotieff_model", type=str, default="enet_b2_8",
                   help="EmotiEffLib model name")
    p.add_argument("--emotieff_device", type=str, default="cuda")
    p.add_argument("--frame_stride", type=int, default=1,
                   help="EmotiEffLib frame stride (1 = every frame)")
    p.add_argument("--no_face_detect", action="store_true",
                   help="Disable MTCNN face detection for EmotiEffLib "
                        "(NOT recommended — accuracy will be poor).")
    p.add_argument("--dfer_device", type=str, default="cuda")
    p.add_argument("--num_segments", type=int, default=16,
                   help="DFER-CLIP frames per video")
    p.add_argument("--clip_weights", type=str,
                   default=str(THIS_DIR / "weights" / "ViT-B-32.pt"))
    p.add_argument("--dfer_weights", type=str,
                   default=str(THIS_DIR / "weights" / "DFEW_fold1.pth"))
    p.add_argument("--output", type=str, default=None,
                   help="Write all results to this JSON file")
    args = p.parse_args()

    # Self-check: each requested model needs its own Python deps to be
    # importable in the *current* interpreter (the per-model scripts run as
    # subprocesses that inherit this interpreter). If something is missing we
    # raise an actionable error rather than the cryptic "empty output"
    # that would otherwise come back from the subprocess.
    preflight_errors: list[str] = []
    if "emotiefflib" in args.models:
        try:
            import emotiefflib.facial_analysis  # noqa: F401
            import facenet_pytorch  # noqa: F401
        except ImportError as exc:
            preflight_errors.append(
                f"emotiefflib requires emotiefflib[torch] + facenet-pytorch "
                f"(missing: {exc.name}). Activate the `eval` conda env, e.g.:\n"
                f"    conda activate eval\n"
                f"    pip install 'emotiefflib[torch]' facenet-pytorch"
            )
    if "dfer_clip" in args.models:
        # DFER-CLIP imports the bundled OpenAI-CLIP submodule by adding
        # DFER-CLIP/models to sys.path. Check that torch + clip.load works.
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            preflight_errors.append(
                f"dfer_clip requires PyTorch (missing: {exc.name}). Activate "
                f"the `eval` conda env, e.g.:\n    conda activate eval"
            )
        if not Path(args.clip_weights).is_file():
            preflight_errors.append(
                f"dfer_clip: CLIP weights not found at {args.clip_weights}"
            )
        if not Path(args.dfer_weights).is_file():
            preflight_errors.append(
                f"dfer_clip: DFEW fold-1 checkpoint not found at {args.dfer_weights}"
            )
    if preflight_errors:
        sys.stderr.write("Preflight check failed:\n")
        for err in preflight_errors:
            sys.stderr.write(f"  - {err}\n")
        sys.exit(2)

    video_paths, label_map = collect_videos(args)
    if not video_paths:
        sys.stderr.write("ERROR: no videos found\n")
        sys.exit(2)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    valid = {"emotiefflib", "dfer_clip"}
    unknown = [m for m in models if m not in valid]
    if unknown:
        sys.stderr.write(f"ERROR: unknown models: {unknown}. Choices: {sorted(valid)}\n")
        sys.exit(2)

    ee_script = THIS_DIR / "evaluate_emotiefflib.py"
    dc_script = THIS_DIR / "evaluate_dfer_clip.py"
    for m, s in [("emotiefflib", ee_script), ("dfer_clip", dc_script)]:
        if m in models and not s.is_file():
            sys.stderr.write(f"ERROR: missing script {s}\n")
            sys.exit(2)

    overall_results: list[dict] = []
    for vp in video_paths:
        lbl = label_map.get(vp.stem)
        record: dict[str, Any] = {"video": str(vp), "label": lbl, "models": {}}
        print(f"\n=== {vp.name} (label={lbl}) ===", flush=True)

        if "emotiefflib" in models:
            print("  [emotiefflib] running...", flush=True)
            res = run_subprocess(ee_script, build_per_model_args("emotiefflib", vp, lbl, args))
            summary = (res.get("results") or [{}])[0].get("summary") or {}
            dom = summary.get("dominant_emotion")
            record["models"]["emotiefflib"] = {
                "prediction": dom,
                "distribution": summary.get("emotion_distribution"),
                "frames_analyzed": summary.get("frames_analyzed"),
                "mean_valence": summary.get("mean_valence"),
                "mean_arousal": summary.get("mean_arousal"),
                "correct": (res.get("results") or [{}])[0].get("correct"),
                "error": res.get("error"),
            }
            print(f"    -> dominant={dom}", flush=True)

        if "dfer_clip" in models:
            print("  [dfer_clip] running...", flush=True)
            res = run_subprocess(dc_script, build_per_model_args("dfer_clip", vp, lbl, args))
            inner = (res.get("results") or [{}])[0]
            record["models"]["dfer_clip"] = {
                "prediction": inner.get("prediction"),
                "probs": inner.get("probs"),
                "correct": inner.get("correct"),
                "error": res.get("error"),
            }
            print(f"    -> prediction={inner.get('prediction')}", flush=True)

        ee_dom = merge_emotion_to_canonical(
            (record["models"].get("emotiefflib") or {}).get("prediction"))
        dc_dom = merge_emotion_to_canonical(
            (record["models"].get("dfer_clip") or {}).get("prediction"))
        if ee_dom and dc_dom:
            record["agreement"] = (ee_dom == dc_dom)
        else:
            record["agreement"] = None
        overall_results.append(record)

    # Aggregate stats
    n = len(overall_results)
    n_labelled = sum(1 for r in overall_results if r.get("label"))
    n_agree = sum(1 for r in overall_results if r.get("agreement") is True)
    summary = {
        "n_videos": n,
        "n_labelled": n_labelled,
        "models": models,
    }
    for m in models:
        ok = sum(1 for r in overall_results
                 if (r["models"].get(m) or {}).get("correct") is True)
        summary[f"{m}_accuracy"] = round(ok / n_labelled, 4) if n_labelled else None
    summary["agreement_rate"] = round(n_agree / n, 4) if n else None

    out = {"summary": summary, "results": overall_results}
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n[unified] wrote {args.output}", flush=True)
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

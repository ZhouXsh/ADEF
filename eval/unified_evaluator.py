#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified talking-head evaluation runner
======================================

This script runs **all** the per-metric evaluators that live in
``ADEF_remake/eval/`` against an input video (and, when available, its
ground-truth counterpart) and writes a single JSON report.

Per-metric evaluators are launched as ``subprocess`` calls in their own
Python environments (conda envs / venvs) so a single command-line
invocation aggregates results from every model the ADEF eval suite
supports:

* **LSE-D / LSE-C** (Wav2Lip SyncNet)              — single video
* **Sync Confidence** (EAT-style SyncNet)          — needs GT pair
* **FVD**      (I3D frechet_video_distance)        — needs GT pair
* **FID**      (pytorch-fid)                       — needs GT pair
* **PSNR / SSIM / LPIPS / LMD**  (EAT)             — needs GT pair
* **Emotion-Acc (EAT)**                            — single video
* **EmoNet** (continuous valence / arousal + discrete) — single video
  * when ``--gt`` is provided, also returns Emo-Acc / valence / arousal
    agreement with GT
* **Emotion-FAN** (7-class AFEW classifier)        — single video
* **EmotiEffLib** (8-class AffectNet, MTCNN faces) — single video
* **DFER-CLIP**   (7-class DFEW, CLIP-ViT-B/32)    — single video
* **New_Emo**     (combined driver of EmotiEffLib + DFER-CLIP, plus
  agreement / per-model accuracy / per-model probabilities)

Usage
-----
Single video (no GT):
    python unified_evaluator.py \
        --fake /path/to/result.mp4 \
        --output /path/to/report.json

Paired video (with GT — runs the GT-requiring metrics too):
    python unified_evaluator.py \
        --fake /path/to/result.mp4 \
        --gt   /path/to/gt.mp4 \
        --name my_run \
        --output /path/to/report.json

Skip a metric (multiple ``--skip`` are allowed):
    python unified_evaluator.py --fake r.mp4 --gt g.mp4 \
        --skip fvd --skip eat

Restrict to a subset of metrics:
    python unified_evaluator.py --fake r.mp4 --gt g.mp4 \
        --metrics lse eat fid emo_fan

Output
------
The output JSON has the following shape::

    {
      "fake": "...",
      "gt":   "...",
      "name": "...",
      "elapsed_sec": 123.4,
      "metrics": {
        "lse":          { "ok": true, "elapsed_sec": 4.3,
                          "payload": { "lse_d": ..., "lse_c": ..., "av_offset": ..., ... } },
        "fvd":          { "ok": true, "payload": { "fvd": ..., "video_length": ..., ... } },
        "fid":          { "ok": true, "payload": { "fid": ..., "elapsed_sec": ... } },
        "eat":          { "ok": true, "payload": {
                            "psnr_ssim": { "psnr": ..., "ssim": ... },
                            "lpips":     { "mean_lpips": ..., "n_videos": ... },
                            "lmd":       { "mouth_lmd": ..., "face_lmd": ..., ... },
                            "sync":      { "sync_conf": ... },
                            "fid":       { "fid_eat": ... },
                            "emo":       { "emo_acc": 0.125, "emo_acc_per_class": {...} } } },
        "emonet":       { "ok": true, "payload": {
                            "emo_acc": ..., "emo_sim": ..., "valence": {...}, "arousal": {...},
                            "n_videos": ..., "n_frames": ... } },
        "emo_fan":      { "ok": true, "payload": {
                            "emotion": ..., "emotion_id": ..., "confidence": ...,
                            "probabilities": {...}, "n_frames_used": ... } },
        "emotiefflib":  { "ok": true, "payload": {
                            "model": ..., "label": ..., "correct": ...,
                            "dominant_emotion": ..., "dominant_fraction": ...,
                            "emotion_distribution": {...},
                            "n_frames_analyzed": ..., "n_frames_with_face": ...,
                            "face_detection_rate": ... } },
        "dfer_clip":    { "ok": true, "payload": {
                            "model": ..., "label": ..., "prediction": ...,
                            "probs": {...}, "correct": ... } },
        "new_emo":      { "ok": true, "payload": {
                            "summary": { "n_videos": ..., "n_labelled": ...,
                                         "models": [...],
                                         "emotiefflib_accuracy": ...,
                                         "dfer_clip_accuracy": ...,
                                         "agreement_rate": ... },
                            "video_label": ..., "agreement": ...,
                            "emotiefflib": {...}, "dfer_clip": {...} } }
      }
    }

If a metric fails, its entry is ``{"ok": false, "error": "..."}``.
If it cannot run because ``--gt`` was not supplied, the entry is
``{"ok": false, "skipped": true, "skip_reason": "no_gt"}``.

Notes
-----
* The script is self-contained — no per-metric environment is required to
  import it, only to run the corresponding subprocess.  It can be invoked
  from any Python 3.8+ interpreter.
* For metrics that need GT (FVD, FID, EAT's PSNR/SSIM/LPIPS/LMD, and
  EmoNet's paired-mode comparison), if ``--gt`` is not given the metric
  is skipped and a ``"skipped": "no_gt"`` reason is recorded in the
  report.
* ``eat`` aggregates PSNR/SSIM/LPIPS/LMD/Sync/Emotion-Accuracy; the
  Emotion-Accuracy (``payload.emo.emo_acc``) is converted to a fraction
  (0..1) to match every other metric in this file — EAT's own log file
  prints it as a percentage.
* ``new_emo`` runs the New_Emo unified driver which itself dispatches
  EmotiEffLib and DFER-CLIP in subprocesses, so a crash in one model does
  not take down the other.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent                 # .../ADEF_remake/eval
WAV2LIP_DIR = EVAL_ROOT / "Wav2Lip" / "evaluation"
EAT_DIR = EVAL_ROOT / "evaluation_eat"
FVD_DIR = EVAL_ROOT / "frechet_video_distance"
PYTORCH_FID_DIR = EVAL_ROOT / "pytorch-fid"
SYNCNET_DIR = EVAL_ROOT / "syncnet_python"
EMONET_DIR = EVAL_ROOT / "emonet"
EMOTION_FAN_DIR = EVAL_ROOT / "Emotion-FAN"
NEW_EMO_DIR = EVAL_ROOT / "New_Emo"

# ---------------------------------------------------------------------------
# Resolve the python interpreter for each evaluator.
#
# The ADEF eval/ subdirs each ship their own virtualenv (some conda, some
# pure venv) because they have incompatible dependency sets (e.g. tensorflow
# 1.x for FVD vs. torch 2.x for the rest).  We hard-code the absolute paths
# here so the unified runner doesn't have to know about conda.
# ---------------------------------------------------------------------------
EVAL_ENV_PY = "/home/Zhouxishi/miniconda3/envs/eval/bin/python"
EAT_VENV_PY = str(EAT_DIR / "venv" / "bin" / "python")
FVD_ENV_PY = "/home/Zhouxishi/miniconda3/envs/fvd/bin/python"
EMONET_ENV_PY = "/home/Zhouxishi/miniconda3/envs/emonet/bin/python"
EMOTION_FAN_ENV_PY = "/home/Zhouxishi/miniconda3/envs/emotion_fan/bin/python"
SYNCNET_VENV_PY = str(SYNCNET_DIR / "syncnet_venv" / "bin" / "python")
WAV2LIP_VENV_PY = str(WAV2LIP_DIR / "venv" / "bin" / "python")


# ---------------------------------------------------------------------------
# Per-metric helpers
# ---------------------------------------------------------------------------
@dataclass
class MetricResult:
    name: str
    ok: bool
    elapsed_sec: float = 0.0
    error: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    skipped: bool = False
    skip_reason: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop the uninteresting skipped=False case to keep JSON tidy.
        if not d.pop("skipped"):
            d.pop("skip_reason", None)
        return d


def _run_subprocess(cmd: List[str], cwd: Optional[Path] = None,
                    timeout: Optional[int] = None,
                    env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Run *cmd* to completion, returning stdout / stderr / rc."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return {
        "rc": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            with path.open() as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _tail_lines(text: str, n: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


# ---------------------------------------------------------------------------
# 1) LSE-D / LSE-C (Wav2Lip SyncNet) — single video
# ---------------------------------------------------------------------------
def run_lse_d_lse_c(fake: str, workdir: Path, *, timeout: int = 600) -> MetricResult:
    name = "lse_d_lse_c"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "lse_d_lse_c.json"
    cmd = [
        WAV2LIP_VENV_PY, "eval_lipsync.py",
        "--video", fake,
        "--output_json", str(out_json),
    ]
    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=WAV2LIP_DIR, timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = f"rc={proc['rc']}: {proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        return res
    payload = _safe_read_json(out_json)
    if not payload or not payload.get("results"):
        res.error = "no JSON output"
        return res
    # Flatten the per-video result for a single input video.
    r = payload["results"][0]
    res.payload = {
        "lse_d": r.get("lse_d"),
        "lse_c": r.get("lse_c"),
        "av_offset": r.get("av_offset"),
        "min_dist_raw": r.get("min_dist_raw"),
        "n_frames": r.get("n_frames"),
        "duration_s": r.get("duration_s"),
    }
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 2) FVD — needs pair
# ---------------------------------------------------------------------------
def run_fvd(fake: str, gt: str, workdir: Path, *,
            video_length: int = 16, pad: bool = False,
            timeout: int = 600) -> MetricResult:
    name = "fvd"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "fvd.json"
    cmd = [
        FVD_ENV_PY, "evaluate_adef.py",
        "--real_dir", gt,
        "--fake_dir", fake,
        "--video_length", str(video_length),
        "--output_file", str(out_json),
    ]
    if pad:
        cmd.append("--pad_pairs_to_batch_size")
    env = os.environ.copy()
    env.setdefault("TFHUB_CACHE_DIR", "/home/Zhouxishi/tfhub_cache")
    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=FVD_DIR, timeout=timeout, env=env)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = f"rc={proc['rc']}: {proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        return res
    payload = _safe_read_json(out_json)
    if not payload or "fvd" not in payload:
        res.error = "no JSON output"
        return res
    res.payload = payload
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 3) FID — needs pair (pytorch-fid)
# ---------------------------------------------------------------------------
def run_fid(fake: str, gt: str, workdir: Path, *,
            frame_stride: int = 1, max_frames: Optional[int] = None,
            timeout: int = 600) -> MetricResult:
    name = "fid"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "fid.json"
    cmd = [
        EVAL_ENV_PY, "evaluate_fid_video.py",
        "--path1", gt,
        "--path2", fake,
        "--output-json", str(out_json),
    ]
    if frame_stride and frame_stride > 1:
        cmd.extend(["--frame-stride", str(frame_stride)])
    if max_frames:
        cmd.extend(["--max-frames", str(max_frames)])
    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=PYTORCH_FID_DIR, timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = f"rc={proc['rc']}: {proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        return res
    payload = _safe_read_json(out_json)
    if not payload or "fid" not in payload:
        res.error = "no JSON output"
        return res
    res.payload = {
        "fid": payload["fid"],
        "elapsed_sec": payload.get("elapsed_sec"),
        "config": payload.get("config"),
    }
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 4) EAT pipeline (PSNR / SSIM / LPIPS / LMD / Sync / Emo-Acc) — needs pair
# ---------------------------------------------------------------------------
def _parse_eat_text(path: Path) -> Dict[str, Any]:
    """Best-effort parser for the text logs EAT writes per-metric."""
    out: Dict[str, Any] = {}
    if not path.is_file():
        return out
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return out
    # PSNR / SSIM: "total: | psnr: 18.51 | ssim: 0.576"
    m = re.search(r"total:\s*\|\s*psnr:\s*([\d.e+-]+)\s*\|\s*ssim:\s*([\d.e+-]+)", text)
    if m:
        out["psnr"] = float(m.group(1))
        out["ssim"] = float(m.group(2))
    # FID: "fid: 60.08..."
    m = re.search(r"fid:\s*([\d.e+-]+)", text)
    if m:
        out["fid_eat"] = float(m.group(1))
    # LMD: "mouth lmd:2.47" / "face lmd:3.45" / "mouth lvd:2.30" / "face lvd:0.68"
    for key, pat in [
        ("mouth_lmd", r"mouth\s+lmd:\s*([\d.e+-]+)"),
        ("face_lmd", r"face\s+lmd:\s*([\d.e+-]+)"),
        ("mouth_lvd", r"mouth\s+lvd:\s*([\d.e+-]+)"),
        ("face_lvd", r"face\s+lvd:\s*([\d.e+-]+)"),
    ]:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            out[key] = float(m.group(1))
    # Sync: "avg conf : 0.6159"
    m = re.search(r"avg\s+conf\s*:\s*([\d.e+-]+)", text)
    if m:
        out["sync_conf"] = float(m.group(1))
    # Emo-Acc: "*Acc@Video 0.000" / "Happy : 0.0" ...
    # NOTE: EAT's _acc_test.py prints both the overall Acc@Video and the
    # per-class accuracies as PERCENTAGES (multiplied by 100, formatted with
    # "{:.3f}").  We convert to fractions (0..1) here so downstream
    # aggregation / CSV exports match the convention used by every other
    # metric in this file (and the rest of the ADEF eval pipeline).
    m = re.search(r"\*Acc@Video\s+([\d.e+-]+)", text)
    if m:
        out["emo_acc"] = float(m.group(1)) / 100.0
    # Per-class accuracies (also percentage in EAT's log)
    per_class = {}
    for cls in ("Happy", "Angry", "Disgust", "Fear", "Sad", "Neutral",
                "Surprised", "Surprise", "Contempt"):
        m = re.search(rf"{re.escape(cls)}\s*:\s*([\d.e+-]+)", text)
        if m:
            per_class[cls] = float(m.group(1)) / 100.0
    if per_class:
        out["emo_acc_per_class"] = per_class
    return out


def run_eat(fake: str, gt: str, name: str, workdir: Path, *,
            device: str = "0", metrics: Optional[List[str]] = None,
            timeout: int = 1800) -> MetricResult:
    """Run the EAT unified evaluator in its own venv and return parsed
    per-metric values."""
    mres = MetricResult(name="eat_pipeline", ok=False)
    save_name = name or "unified_eval"
    cmd = [
        EAT_VENV_PY, "evaluate.py",
        "--fake", fake,
        "--gt", gt,
        "--name", save_name,
        "--device", device,
        "--auto-detect-name-mode",
        "--allow-all-pids",
    ]
    if metrics:
        cmd.extend(["--metrics", ",".join(metrics)])
    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=EAT_DIR, timeout=timeout)
    mres.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        mres.error = f"rc={proc['rc']}: {proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        return mres
    # The EAT pipeline writes per-metric text logs; we parse them.
    code_dir = EAT_DIR / "code"
    paths = {
        "psnr_ssim": code_dir / "result_psnr" / f"{save_name}.txt",
        "fid":       code_dir / "results" / f"{save_name}.txt",
        "lmd":       code_dir / "result" / f"{save_name}.txt",
        "sync":      code_dir / "results_lastversion" / f"{save_name}.txt",
        "emo":       code_dir / "result_emoacc" / f"{save_name}.txt",
    }
    parsed: Dict[str, Any] = {}
    for key, p in paths.items():
        sub = _parse_eat_text(p)
        if sub:
            parsed[key] = sub
    # LPIPS is its own json file under outputs/
    lpips_json = EAT_DIR / "outputs" / f"{save_name}_lpips.json"
    lpips_data = _safe_read_json(lpips_json)
    if lpips_data:
        parsed["lpips"] = {
            "mean_lpips": lpips_data.get("mean_lpips"),
            "n_videos": len(lpips_data.get("per_video", [])),
        }
    mres.payload = parsed
    mres.ok = bool(parsed)
    if not mres.ok:
        mres.error = "no EAT log files parsed"
    return mres


# ---------------------------------------------------------------------------
# 5) EmoNet — single video (or paired with --gt)
# ---------------------------------------------------------------------------
def run_emonet(fake: str, gt: Optional[str], workdir: Path, *,
               nclasses: int = 8, device: str = "cuda:0",
               timeout: int = 1200) -> MetricResult:
    name = "emonet"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "emonet.json"
    cmd = [
        EMONET_ENV_PY, "evaluate_emotion.py",
        "--gen", fake,
        "--output", str(out_json),
        "--nclasses", str(nclasses),
        "--device", device,
    ]
    if gt:
        cmd.extend(["--gt", gt])
    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=EMONET_DIR, timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = f"rc={proc['rc']}: {proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        return res
    payload = _safe_read_json(out_json)
    if not payload:
        res.error = "no JSON output"
        return res
    summary: Dict[str, Any] = {}
    overall = payload.get("overall")
    if overall:
        summary["emo_acc"] = overall.get("emo_acc")
        summary["emo_sim"] = overall.get("emo_sim")
        summary["emo_acc_video_mean"] = overall.get("emo_acc_video_mean")
        summary["emo_sim_video_mean"] = overall.get("emo_sim_video_mean")
        if "valence" in overall:
            summary["valence"] = overall["valence"]
        if "arousal" in overall:
            summary["arousal"] = overall["arousal"]
        summary["n_videos"] = overall.get("n_videos")
        summary["n_frames"] = overall.get("n_frames")
    if not summary:
        # Single-video mode — pull per-video stats from the first entry.
        for v in (payload.get("per_video") or {}).values():
            summary["n_frames_with_face"] = v.get("n_frames_with_face")
            summary["mean_valence"] = v.get("mean_valence")
            summary["mean_arousal"] = v.get("mean_arousal")
            summary["emotion_histogram"] = v.get("emotion_histogram")
            break
    res.payload = summary
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 6) Emotion-FAN — single video
# ---------------------------------------------------------------------------
def run_emotion_fan(fake: str, workdir: Path, *,
                    pretrain: Optional[Path] = None,
                    at_type: int = 1, device: str = "cuda:0",
                    timeout: int = 600) -> MetricResult:
    name = "emotion_fan"
    res = MetricResult(name=name, ok=False)
    if pretrain is None:
        pretrain = EMOTION_FAN_DIR / "pretrain_model" / "Resnet18_FER+_pytorch.pth.tar"
    out_json = workdir / "emotion_fan.json"
    cmd = [
        EMOTION_FAN_ENV_PY, "evaluate_emotion_fan.py",
        "--input", fake,
        "--pretrain_fer", str(pretrain),
        "--at_type", str(at_type),
        "--device", device,
        "--out_json", str(out_json),
    ]
    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=EMOTION_FAN_DIR, timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = f"rc={proc['rc']}: {proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        return res
    payload = _safe_read_json(out_json)
    if not payload:
        res.error = "no JSON output"
        return res
    # The CLI writes {source: prediction_dict}; we have a single video.
    pred = next(iter(payload.values()), None)
    if pred is None:
        res.error = "empty Emo-FAN JSON"
        return res
    res.payload = {
        "emotion": pred.get("emotion"),
        "emotion_id": pred.get("emotion_id"),
        "confidence": pred.get("confidence"),
        "probabilities": pred.get("probabilities"),
        "n_frames_used": pred.get("n_frames_used"),
    }
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# 7) New_Emo — EmotiEffLib (8-class AffectNet) and/or DFER-CLIP (7 DFEW
#    classes).  Both models live under New_Emo/ and are dispatched as
#    **subprocesses** from the `eval` conda env (the same env that hosts
#    pytorch-fid) because pulling emotiefflib + facenet-pytorch into the
#    unified runner's interpreter would create dependency clashes with the
#    other per-metric environments.  Each model can be run standalone, or
#    both together via New_Emo/evaluate_unified.py.
#
#    Both models accept an optional `--label <emotion>` so that when a GT
#    video is supplied (or the emotion can be inferred from the file path /
#    parent directory), the per-model `correct` flag is computed and surfaced
#    in the unified JSON.  We use the helper below to derive the label from
#    either the GT path (preferred — semantically correct) or, failing that,
#    the fake path.
# ---------------------------------------------------------------------------
_EMO_TOKENS = {
    "angry", "anger", "contempt", "disgusted", "disgust",
    "fear", "happy", "happiness", "sad", "sadness",
    "surprised", "surprise", "neutral", "calm",
}

# Map every synonym to the canonical lowercase label the per-model scripts
# understand.  EmotiEffLib expects Title-case AffectNet names ("Anger",
# "Happiness"), but its CLI accepts lower-case via `--label` and does its own
# normalisation (see evaluate_emotiefflib.normalize_label_to_canonical).  We
# still feed it a canonical label so downstream JSON keys are consistent.
_EMO_CANON = {
    "angry":      "anger",
    "anger":      "anger",
    "happy":      "happiness",
    "happiness":  "happiness",
    "sad":        "sadness",
    "sadness":    "sadness",
    "disgusted":  "disgust",
    "disgust":    "disgust",
    "surprised":  "surprise",
    "surprise":   "surprise",
    "fear":       "fear",
    "neutral":    "neutral",
    "contempt":   "contempt",
    "calm":       "neutral",
}


def _infer_emo_label(path: Optional[str]) -> Optional[str]:
    """Best-effort extraction of the GT emotion label from a video path.

    Looks at (in order):
      1. The filename stem: any underscore/dash/dot-separated token that
         matches a known emotion word.
      2. Up to 3 parent directory names that match a known emotion word.

    Returns the canonical lower-case label or ``None`` if nothing matches.
    """
    if not path:
        return None
    p = Path(path)
    # 1) filename tokens
    parts = re.split(r"[_\-\s\.]+", p.stem.lower())
    for tok in parts:
        if tok in _EMO_TOKENS:
            return _EMO_CANON[tok]
    # 2) parent directory names
    for parent in (p.parent, p.parent.parent, p.parent.parent.parent):
        if parent is None or str(parent) == "":
            break
        tok = parent.name.lower()
        if tok in _EMO_TOKENS:
            return _EMO_CANON[tok]
    return None


def run_emotiefflib(fake: str, gt: Optional[str], workdir: Path, *,
                    model: str = "enet_b2_8", device: str = "cuda:0",
                    frame_stride: int = 1, no_face_detect: bool = False,
                    timeout: int = 1800) -> MetricResult:
    """Run EmotiEffLib on a single video and parse the JSON dump.

    ``gt`` is used only to derive the canonical emotion label so the
    ``correct`` field is populated; predictions run regardless.
    """
    name = "emotiefflib"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "emotiefflib.json"
    script = NEW_EMO_DIR / "evaluate_emotiefflib.py"
    label = _infer_emo_label(gt) or _infer_emo_label(fake)

    cmd = [
        EVAL_ENV_PY, str(script),
        "--video", fake,
        "--model", model,
        "--device", device,
        "--frame_stride", str(frame_stride),
        "--quiet",
        "--output", str(out_json),
    ]
    if no_face_detect:
        cmd.append("--no_face_detect")
    if label is not None:
        cmd.extend(["--label", label])

    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=str(NEW_EMO_DIR), timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = (
            f"rc={proc['rc']}: "
            f"{proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        )
        return res
    payload = _safe_read_json(out_json)
    if not payload:
        res.error = "no JSON output from evaluate_emotiefflib.py"
        return res
    results = payload.get("results") or []
    if not results:
        res.error = "empty results[] in JSON"
        return res
    r0 = results[0]
    summary = r0.get("summary") or {}
    res.payload = {
        "model":              payload.get("model"),
        "label":              r0.get("label"),
        "correct":            r0.get("correct"),
        "dominant_emotion":   summary.get("dominant_emotion"),
        "dominant_fraction":  summary.get("dominant_fraction"),
        "emotion_distribution": summary.get("emotion_distribution"),
        "n_frames_analyzed":  summary.get("frames_analyzed"),
        "n_frames_with_face": summary.get("frames_with_face"),
        "face_detection_rate": summary.get("face_detection_rate"),
    }
    res.ok = True
    return res


def run_dfer_clip(fake: str, gt: Optional[str], workdir: Path, *,
                  clip_weights: Optional[Path] = None,
                  dfer_weights: Optional[Path] = None,
                  device: str = "cuda:0", num_segments: int = 16,
                  timeout: int = 1800) -> MetricResult:
    """Run DFER-CLIP on a single video and parse the JSON dump."""
    name = "dfer_clip"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "dfer_clip.json"
    script = NEW_EMO_DIR / "evaluate_dfer_clip.py"
    if clip_weights is None:
        clip_weights = NEW_EMO_DIR / "weights" / "ViT-B-32.pt"
    if dfer_weights is None:
        dfer_weights = NEW_EMO_DIR / "weights" / "DFEW_fold1.pth"
    label = _infer_emo_label(gt) or _infer_emo_label(fake)

    cmd = [
        EVAL_ENV_PY, str(script),
        "--video", fake,
        "--clip_weights", str(clip_weights),
        "--dfer_weights", str(dfer_weights),
        "--device", device,
        "--num_segments", str(num_segments),
        "--quiet",
        "--output", str(out_json),
    ]
    if label is not None:
        cmd.extend(["--label", label])

    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=str(NEW_EMO_DIR), timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = (
            f"rc={proc['rc']}: "
            f"{proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        )
        return res
    payload = _safe_read_json(out_json)
    if not payload:
        res.error = "no JSON output from evaluate_dfer_clip.py"
        return res
    results = payload.get("results") or []
    if not results:
        res.error = "empty results[] in JSON"
        return res
    r0 = results[0]
    res.payload = {
        "model":      payload.get("model"),
        "label":      r0.get("label"),
        "prediction": r0.get("prediction"),
        "probs":      r0.get("probs"),
        "correct":    r0.get("correct"),
    }
    res.ok = True
    return res


def run_new_emo(fake: str, gt: Optional[str], workdir: Path, *,
                models: str = "emotiefflib,dfer_clip",
                emotieff_model: str = "enet_b2_8",
                emotieff_device: str = "cuda:0",
                emotieff_frame_stride: int = 1,
                emotieff_no_face_detect: bool = False,
                dfer_device: str = "cuda:0",
                dfer_num_segments: int = 16,
                clip_weights: Optional[Path] = None,
                dfer_weights: Optional[Path] = None,
                timeout: int = 1800) -> MetricResult:
    """Run New_Emo/evaluate_unified.py — both EmotiEffLib and DFER-CLIP.

    The unified driver writes its own aggregate ``summary`` (per-model
    accuracy, agreement rate) plus a per-video ``results`` array; we surface
    both here.
    """
    name = "new_emo"
    res = MetricResult(name=name, ok=False)
    out_json = workdir / "new_emo.json"
    script = NEW_EMO_DIR / "evaluate_unified.py"
    if clip_weights is None:
        clip_weights = NEW_EMO_DIR / "weights" / "ViT-B-32.pt"
    if dfer_weights is None:
        dfer_weights = NEW_EMO_DIR / "weights" / "DFEW_fold1.pth"
    label = _infer_emo_label(gt) or _infer_emo_label(fake)

    cmd = [
        EVAL_ENV_PY, str(script),
        "--video", fake,
        "--models", models,
        "--emotieff_model", emotieff_model,
        "--emotieff_device", emotieff_device,
        "--frame_stride", str(emotieff_frame_stride),
        "--dfer_device", dfer_device,
        "--num_segments", str(dfer_num_segments),
        "--clip_weights", str(clip_weights),
        "--dfer_weights", str(dfer_weights),
        "--output", str(out_json),
    ]
    if emotieff_no_face_detect:
        cmd.append("--no_face_detect")
    if label is not None:
        cmd.extend(["--label", label])

    t0 = time.time()
    proc = _run_subprocess(cmd, cwd=str(NEW_EMO_DIR), timeout=timeout)
    res.elapsed_sec = time.time() - t0
    if proc["rc"] != 0:
        res.error = (
            f"rc={proc['rc']}: "
            f"{proc['stderr'].splitlines()[-1] if proc['stderr'] else ''}"
        )
        return res
    payload = _safe_read_json(out_json)
    if not payload:
        res.error = "no JSON output from evaluate_unified.py"
        return res
    summary = payload.get("summary") or {}
    results = payload.get("results") or []
    inner = results[0] if results else {}
    inner_models = inner.get("models") or {}
    res.payload = {
        "summary": summary,
        "video_label": inner.get("label"),
        "agreement": inner.get("agreement"),
        "emotiefflib": inner_models.get("emotiefflib"),
        "dfer_clip":   inner_models.get("dfer_clip"),
    }
    res.ok = True
    return res


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
METRIC_REGISTRY = {
    "lse":            ("run_lse_d_lse_c",      False),
    "fvd":            ("run_fvd",              True),
    "fid":            ("run_fid",              True),
    "eat":            ("run_eat",              True),
    "emonet":         ("run_emonet",           False),
    "emo_fan":        ("run_emotion_fan",      False),
    "emotiefflib":    ("run_emotiefflib",      False),
    "dfer_clip":      ("run_dfer_clip",        False),
    "new_emo":        ("run_new_emo",          False),
}


def _resolve_metrics(selected: Optional[List[str]]) -> List[str]:
    if not selected or "all" in selected:
        return list(METRIC_REGISTRY.keys())
    out = []
    for m in selected:
        if m not in METRIC_REGISTRY:
            print(f"[warn] unknown metric '{m}', skipping")
            continue
        out.append(m)
    return out


def _flatten_for_summary(metric_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a flat dict for the per-metric one-line summary print.

    Most metrics store their numeric leaves at the top level of
    ``payload``.  The ``eat`` metric, however, nests them under
    ``psnr_ssim / lpips / lmd / sync / emo / fid`` sub-dicts, so without
    flattening the summary would print nothing for EAT.  This helper
    promotes those leaves one level so ``short_keys`` (``psnr``,
    ``mean_lpips``, ``emo_acc``, ``sync_conf`` …) can find them.
    """
    if metric_key == "eat" and isinstance(payload, dict):
        flat: Dict[str, Any] = {}
        for sub_key in ("psnr_ssim", "lpips", "lmd", "sync", "emo", "fid"):
            sub = payload.get(sub_key)
            if isinstance(sub, dict):
                flat.update(sub)
        return flat
    return payload


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Unified talking-head evaluation runner — runs every "
                    "available metric and writes a single JSON report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fake", required=True,
                   help="Generated/result video file to evaluate.")
    p.add_argument("--gt", default=None,
                   help="Optional ground-truth video for paired metrics "
                        "(FVD/FID/PSNR/SSIM/LPIPS/LMD/EmoNet-GT).")
    p.add_argument("--name", default="unified_eval",
                   help="Identifier used for EAT sub-outputs.")
    p.add_argument("--output", "-o", default=None,
                   help="Output JSON path. Default: <workdir>/report.json.")
    p.add_argument("--workdir", default=None,
                   help="Working directory for intermediate files. Default: "
                        "system temp dir; deleted at exit unless --keep-workdir.")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Do not delete the temporary working directory on exit.")
    p.add_argument("--metrics", nargs="+", default=["all"],
                   help="Subset of metrics to run, space-separated. Choices: "
                        "lse fvd fid eat emonet emo_fan emotiefflib dfer_clip "
                        "new_emo (or 'all').")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Metrics to skip (subtractive filter on top of --metrics).")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device for EmoNet / Emotion-FAN (e.g. cuda:0, cpu).")
    p.add_argument("--eat-device", default="0",
                   help="CUDA device index for EAT (just the number).")
    p.add_argument("--fvd-video-length", type=int, default=16,
                   help="Frames sampled per video for FVD.")
    p.add_argument("--fvd-pad-pairs", action="store_true",
                   help="Pass --pad_pairs_to_batch_size to FVD "
                        "(demo mode — single-pair results are NOT meaningful).")
    p.add_argument("--fid-frame-stride", type=int, default=1)
    p.add_argument("--fid-max-frames", type=int, default=None)
    p.add_argument("--emonet-nclasses", type=int, default=8, choices=[5, 8])
    p.add_argument("--emo-fan-at-type", type=int, default=1, choices=[-1, 0, 1])
    # New_Emo (EmotiEffLib + DFER-CLIP) options
    p.add_argument("--new-emo-models", default="emotiefflib,dfer_clip",
                   help="Comma subset of {emotiefflib, dfer_clip} for "
                        "the combined --metrics new_emo metric.")
    p.add_argument("--emotieff-model", default="enet_b2_8",
                   help="EmotiEffLib backbone (run with --list_models to list).")
    p.add_argument("--emotieff-frame-stride", type=int, default=1,
                   help="EmotiEffLib frame stride (1 = every frame).")
    p.add_argument("--emotieff-no-face-detect", action="store_true",
                   help="Disable MTCNN face detection for EmotiEffLib "
                        "(NOT recommended — accuracy will be poor).")
    p.add_argument("--dfer-num-segments", type=int, default=16,
                   help="Frames uniformly sampled per video for DFER-CLIP "
                        "(must match the checkpoint's training setting).")
    p.add_argument("--new-emo-clip-weights",
                   default=str(NEW_EMO_DIR / "weights" / "ViT-B-32.pt"),
                   help="Path to the OpenAI CLIP ViT-B/32 backbone for DFER-CLIP.")
    p.add_argument("--new-emo-dfer-weights",
                   default=str(NEW_EMO_DIR / "weights" / "DFEW_fold1.pth"),
                   help="Path to the DFER-CLIP DFEW fold-1 checkpoint.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-metric progress prints.")
    args = p.parse_args(argv)

    if not Path(args.fake).is_file():
        p.error(f"--fake not found: {args.fake}")
    if args.gt and not Path(args.gt).is_file():
        p.error(f"--gt not found: {args.gt}")

    selected = _resolve_metrics(args.metrics)
    for skip in args.skip:
        if skip in selected:
            selected.remove(skip)
            print(f"[skip] {skip} (removed by --skip)")

    # Working directory for intermediate files.
    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        keep = True
    else:
        workdir = Path(tempfile.mkdtemp(prefix="unified_eval_"))
        keep = args.keep_workdir

    if not args.quiet:
        print(f"[unified] workdir = {workdir}")
        print(f"[unified] fake  = {args.fake}")
        if args.gt:
            print(f"[unified] gt    = {args.gt}")
        print(f"[unified] metrics = {selected}")

    results: Dict[str, MetricResult] = {}
    overall_t0 = time.time()

    def _dispatch(key: str) -> Optional[MetricResult]:
        if key == "lse":
            return run_lse_d_lse_c(args.fake, workdir)
        if key == "fvd":
            if not args.gt:
                return MetricResult(name="fvd", ok=False, skipped=True,
                                    skip_reason="no_gt")
            return run_fvd(args.fake, args.gt, workdir,
                           video_length=args.fvd_video_length,
                           pad=args.fvd_pad_pairs)
        if key == "fid":
            if not args.gt:
                return MetricResult(name="fid", ok=False, skipped=True,
                                    skip_reason="no_gt")
            return run_fid(args.fake, args.gt, workdir,
                           frame_stride=args.fid_frame_stride,
                           max_frames=args.fid_max_frames)
        if key == "eat":
            if not args.gt:
                return MetricResult(name="eat", ok=False, skipped=True,
                                    skip_reason="no_gt")
            return run_eat(args.fake, args.gt, args.name, workdir,
                           device=args.eat_device)
        if key == "emonet":
            return run_emonet(args.fake, args.gt, workdir,
                              nclasses=args.emonet_nclasses,
                              device=args.device)
        if key == "emo_fan":
            return run_emotion_fan(args.fake, workdir,
                                   at_type=args.emo_fan_at_type,
                                   device=args.device)
        if key == "emotiefflib":
            return run_emotiefflib(args.fake, args.gt, workdir,
                                   model=args.emotieff_model,
                                   device=args.device,
                                   frame_stride=args.emotieff_frame_stride,
                                   no_face_detect=args.emotieff_no_face_detect)
        if key == "dfer_clip":
            return run_dfer_clip(args.fake, args.gt, workdir,
                                 clip_weights=Path(args.new_emo_clip_weights),
                                 dfer_weights=Path(args.new_emo_dfer_weights),
                                 device=args.device,
                                 num_segments=args.dfer_num_segments)
        if key == "new_emo":
            return run_new_emo(args.fake, args.gt, workdir,
                               models=args.new_emo_models,
                               emotieff_model=args.emotieff_model,
                               emotieff_device=args.device,
                               emotieff_frame_stride=args.emotieff_frame_stride,
                               emotieff_no_face_detect=args.emotieff_no_face_detect,
                               dfer_device=args.device,
                               dfer_num_segments=args.dfer_num_segments,
                               clip_weights=Path(args.new_emo_clip_weights),
                               dfer_weights=Path(args.new_emo_dfer_weights))
        return None

    for key in selected:
        if not args.quiet:
            print(f"\n[unified] === {key} ===")
        try:
            r = _dispatch(key)
        except Exception as exc:  # noqa: BLE001
            r = MetricResult(name=key, ok=False,
                             error=f"{type(exc).__name__}: {exc}")
            if not args.quiet:
                traceback.print_exc()
        if r is None:
            continue
        results[key] = r
        if not args.quiet:
            status = "ok" if r.ok else ("skipped" if r.skipped else "FAIL")
            extra = ""
            if r.ok and r.payload:
                # Try to print a one-line summary.  We flatten the payload
                # first so metrics like `eat` (whose numeric leaves are
                # nested under psnr_ssim / lpips / lmd / sync / emo / fid)
                # still produce a non-empty summary line.
                view = _flatten_for_summary(key, r.payload)
                short_keys = ("lse_d", "fid", "fvd", "psnr", "ssim", "mean_lpips",
                              "mouth_lmd", "face_lmd", "emo_acc", "emotion",
                              "dominant_emotion", "prediction", "sync_conf")
                parts = []
                for k in short_keys:
                    if k not in view:
                        continue
                    v = view[k]
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.4f}")
                    else:
                        parts.append(f"{k}={v}")
                if parts:
                    extra = f"  → {', '.join(parts)}"
            print(f"[unified] {key}: {status} ({r.elapsed_sec:.1f}s){extra}"
                  + (f"  ERR={r.error}" if r.error else ""))

    elapsed = time.time() - overall_t0

    report: Dict[str, Any] = {
        "fake": str(Path(args.fake).resolve()),
        "gt": str(Path(args.gt).resolve()) if args.gt else None,
        "name": args.name,
        "elapsed_sec": elapsed,
        "metrics": {k: v.to_dict() for k, v in results.items()},
    }

    out_path = Path(args.output) if args.output else workdir / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print("\n[unified] ============= SUMMARY =============")
        for k, r in results.items():
            if r.skipped:
                print(f"  {k:<12s}  skipped ({r.skip_reason})")
            elif r.ok:
                print(f"  {k:<12s}  ok ({r.elapsed_sec:.1f}s)")
            else:
                print(f"  {k:<12s}  FAIL: {r.error}")
        print(f"[unified] total: {elapsed:.1f}s")
        print(f"[unified] report -> {out_path}")

    if not keep:
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    return 0 if all(r.ok or r.skipped for r in results.values()) else 2


if __name__ == "__main__":
    sys.exit(main())

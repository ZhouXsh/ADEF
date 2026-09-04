#!/usr/bin/env python3
"""Authoritative paper-table evaluator for ADEF and all baselines.

One invocation evaluates one complete method result set described by an
explicit manifest.  Distribution metrics (FID/FVD) are computed once over the
whole set; per-video metrics are aggregated only after coverage is verified.
The generated ``paper_table.csv`` row is intended to be copied directly into
a paper table when ``Status == complete``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import (  # noqa: E402
    PROTOCOL_VERSION, Sample, manifest_fingerprint, read_manifest,
)

LSE_SCRIPT = THIS_DIR / "Wav2Lip" / "evaluation" / "eval_lipsync.py"
FID_SCRIPT = THIS_DIR / "pytorch-fid" / "evaluate_fid_video.py"
FVD_SCRIPT = THIS_DIR / "frechet_video_distance" / "evaluate_adef.py"
PAIRWISE_SCRIPT = THIS_DIR / "pairwise_metrics.py"
EMOTIEFF_SCRIPT = THIS_DIR / "New_Emo" / "evaluate_emotiefflib.py"
DFER_SCRIPT = THIS_DIR / "New_Emo" / "evaluate_dfer_clip.py"

DEFAULT_EVAL_PY = Path("/home/Zhouxishi/miniconda3/envs/eval/bin/python")
DEFAULT_FVD_PY = Path("/home/Zhouxishi/miniconda3/envs/fvd/bin/python")
DEFAULT_LSE_PY = THIS_DIR / "Wav2Lip" / "evaluation" / "venv" / "bin" / "python"
DEFAULT_PAIRWISE_PY = THIS_DIR / "evaluation_eat" / "venv" / "bin" / "python"

PAPER_METRICS = ("lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip")
TABLE_COLUMNS = [
    "Method", "Status", "N",
    "LSE-D", "LSE-C", "FID", "FVD",
    "PSNR", "SSIM", "LPIPS", "M-LMD", "F-LMD",
    "EmotiEff-Acc", "DFER-CLIP-Acc", "DFER-N",
    "Protocol", "Manifest-SHA256",
]


def _python(preferred: Path) -> str:
    return str(preferred) if preferred.is_file() else sys.executable


def _run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 7200) -> dict[str, Any]:
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True,
                           text=True, timeout=timeout)
        return {"rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr,
                "elapsed_sec": time.time() - t0, "cmd": cmd}
    except subprocess.TimeoutExpired as exc:
        return {"rc": 124, "stdout": exc.stdout or "", "stderr": f"timeout after {timeout}s",
                "elapsed_sec": time.time() - t0, "cmd": cmd}


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"expected JSON was not produced: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_list(path: Path, values: list[str]) -> Path:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path


def _safe_stem(name: str, index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "sample"
    return f"{index:05d}_{safe}"


def _stage_emotion_inputs(samples: list[Sample], root: Path):
    """Build a clean, manifest-exact input directory for emotion evaluators.

    ``work/emotion_inputs`` is disposable staging state.  Rebuilding it on
    every invocation makes reruns idempotent and prevents files from an older
    manifest from being silently included in EmotiEffLib/DFER-CLIP.
    """
    if root.is_symlink() or root.is_file():
        root.unlink()
    elif root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)

    labels = root / "labels.txt"
    stem_to_sample: dict[str, Sample] = {}
    with labels.open("w", encoding="utf-8") as lf:
        for i, s in enumerate(samples):
            stem = _safe_stem(s.name, i)
            dst = root / f"{stem}{Path(s.fake).suffix.lower() or '.mp4'}"
            src = Path(s.fake).resolve()
            try:
                os.symlink(src, dst)
            except OSError:
                # Some filesystems/containers disallow symlinks.  Since the
                # staging directory was just rebuilt, copying cannot collide
                # with a stale link from a previous evaluation.
                shutil.copy2(src, dst)
            stem_to_sample[dst.stem] = s
            if s.emotion:
                lf.write(f"{dst.stem} {s.emotion}\n")
    return labels, stem_to_sample


def _metric_error(proc: dict, json_path: Path | None = None) -> str | None:
    if proc["rc"] != 0:
        tail = (proc.get("stderr") or proc.get("stdout") or "").splitlines()[-4:]
        return f"rc={proc['rc']}: {' | '.join(tail)}"
    if json_path is not None and not json_path.is_file():
        return f"output missing: {json_path}"
    return None


def _mean(payload: dict, key: str):
    v = payload.get(key)
    return v.get("mean") if isinstance(v, dict) else None


def evaluate(samples: list[Sample], method: str, outdir: Path, metrics: list[str], args) -> tuple[dict, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)
    details: dict[str, Any] = {}
    errors: list[str] = []
    per_video: dict[str, dict[str, Any]] = {s.name: {"name": s.name, "emotion": s.emotion} for s in samples}

    fake_list = _write_list(work / "fake.txt", [s.fake for s in samples])
    gt_list = _write_list(work / "gt.txt", [s.gt for s in samples])

    if "lse" in metrics:
        out = work / "lse.json"
        cmd = [args.lse_python, str(LSE_SCRIPT), "--filelist", str(fake_list),
               "--output_json", str(out), "--device", args.device,
               "--min-track", str(args.lse_min_track)]
        proc = _run(cmd, cwd=LSE_SCRIPT.parent, timeout=args.timeout)
        err = _metric_error(proc, out)
        if err:
            errors.append("LSE: " + err); details["lse"] = {"error": err, "process": proc}
        else:
            data = _load_json(out); details["lse"] = data
            if data.get("n_success") != len(samples):
                errors.append(f"LSE coverage {data.get('n_success')}/{len(samples)}")
            # eval_lipsync preserves file-list order.
            for s, row in zip(samples, data.get("results", [])):
                per_video[s.name].update({"LSE-D": row.get("lse_d"), "LSE-C": row.get("lse_c")})

    if "fid" in metrics:
        out = work / "fid.json"
        cmd = [args.eval_python, str(FID_SCRIPT), "--list1", str(gt_list), "--list2", str(fake_list),
               "--output-json", str(out), "--device", args.device]
        if args.fid_frame_stride != 1:
            cmd += ["--frame-stride", str(args.fid_frame_stride)]
        proc = _run(cmd, cwd=FID_SCRIPT.parent, timeout=args.timeout)
        err = _metric_error(proc, out)
        if err:
            errors.append("FID: " + err); details["fid"] = {"error": err, "process": proc}
        else:
            data = _load_json(out); details["fid"] = data
            if data.get("fid") is None:
                errors.append("FID output has no finite value")

    if "fvd" in metrics:
        out = work / "fvd.json"
        cmd = [args.fvd_python, str(FVD_SCRIPT), "--real_list", str(gt_list), "--fake_list", str(fake_list),
               "--video_length", str(args.fvd_video_length), "--output_file", str(out)]
        proc = _run(cmd, cwd=FVD_SCRIPT.parent, timeout=args.timeout)
        err = _metric_error(proc, out)
        if err:
            errors.append("FVD: " + err); details["fvd"] = {"error": err, "process": proc}
        else:
            data = _load_json(out); details["fvd"] = data
            if data.get("fvd") is None:
                errors.append("FVD output has no finite value")

    if "pairwise" in metrics:
        manifest = work / "pairwise_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["name", "fake", "gt", "emotion"])
            for s in samples: w.writerow([s.name, s.fake, s.gt, s.emotion or ""])
        out = work / "pairwise.json"
        cmd = [args.pairwise_python, str(PAIRWISE_SCRIPT), "--manifest", str(manifest),
               "--output", str(out), "--device", args.device]
        if args.allow_partial:
            cmd.append("--allow-partial")
        proc = _run(cmd, cwd=THIS_DIR, timeout=args.timeout)
        err = _metric_error(proc, out)
        if err:
            errors.append("Pairwise: " + err); details["pairwise"] = {"error": err, "process": proc}
        else:
            data = _load_json(out); details["pairwise"] = data
            if data.get("n_success") != len(samples):
                errors.append(f"pairwise coverage {data.get('n_success')}/{len(samples)}")
            for row in data.get("per_video", []):
                if row.get("name") in per_video:
                    per_video[row["name"]].update({
                        "PSNR": row.get("psnr"), "SSIM": row.get("ssim"), "LPIPS": row.get("lpips"),
                        "M-LMD": row.get("mouth_lmd"), "F-LMD": row.get("face_lmd"),
                    })

    emotion_stage = None
    label_file = None
    stem_map = None
    if "emotiefflib" in metrics or "dfer_clip" in metrics:
        emotion_stage = work / "emotion_inputs"
        label_file, stem_map = _stage_emotion_inputs(samples, emotion_stage)
        if any(s.emotion is None for s in samples):
            errors.append("emotion labels missing for one or more manifest samples")

    if "emotiefflib" in metrics and emotion_stage is not None:
        out = work / "emotiefflib.json"
        cmd = [args.eval_python, str(EMOTIEFF_SCRIPT), "--video_dir", str(emotion_stage),
               "--label_file", str(label_file), "--model", args.emotieff_model,
               "--device", args.device, "--quiet", "--output", str(out)]
        proc = _run(cmd, cwd=EMOTIEFF_SCRIPT.parent, timeout=args.timeout)
        err = _metric_error(proc, out)
        if err:
            errors.append("EmotiEffLib: " + err); details["emotiefflib"] = {"error": err, "process": proc}
        else:
            data = _load_json(out); details["emotiefflib"] = data
            if data.get("n_videos") != len(samples) or data.get("n_labelled") != len(samples):
                errors.append(f"EmotiEff coverage videos={data.get('n_videos')} labelled={data.get('n_labelled')} expected={len(samples)}")
            if data.get("accuracy") is None:
                errors.append("EmotiEff accuracy is unavailable")
            failed_emo = [r.get("video") for r in data.get("results", []) if r.get("error")]
            if failed_emo:
                errors.append(f"EmotiEff failed on {len(failed_emo)} video(s)")
            for row in data.get("results", []):
                stem = Path(row.get("video", "")).stem
                s = stem_map.get(stem) if stem_map else None
                if s:
                    per_video[s.name]["EmotiEff-Correct"] = row.get("correct")
                    per_video[s.name]["EmotiEff-Pred"] = (row.get("summary") or {}).get("dominant_emotion")

    if "dfer_clip" in metrics and emotion_stage is not None:
        out = work / "dfer_clip.json"
        cmd = [args.eval_python, str(DFER_SCRIPT), "--video_dir", str(emotion_stage),
               "--label_file", str(label_file), "--device", args.device,
               "--num_segments", str(args.dfer_segments), "--quiet", "--output", str(out)]
        proc = _run(cmd, cwd=DFER_SCRIPT.parent, timeout=args.timeout)
        err = _metric_error(proc, out)
        if err:
            errors.append("DFER-CLIP: " + err); details["dfer_clip"] = {"error": err, "process": proc}
        else:
            data = _load_json(out); details["dfer_clip"] = data
            if data.get("n_videos") != len(samples):
                errors.append(f"DFER video coverage {data.get('n_videos')}/{len(samples)}")
            if not data.get("n_labelled_supported"):
                errors.append("DFER-CLIP has no supported labelled samples")
            for row in data.get("results", []):
                stem = Path(row.get("video", "")).stem
                s = stem_map.get(stem) if stem_map else None
                if s:
                    per_video[s.name]["DFER-Correct"] = row.get("correct")
                    per_video[s.name]["DFER-Pred"] = row.get("prediction")
                    per_video[s.name]["DFER-Label-Supported"] = row.get("label_supported")

    # Extract the single paper row. No mean of per-video FID/FVD exists here.
    lse = details.get("lse", {})
    pair = details.get("pairwise", {}).get("aggregate", {}) if isinstance(details.get("pairwise"), dict) else {}
    emot = details.get("emotiefflib", {})
    dfer = details.get("dfer_clip", {})
    row = {
        "Method": method,
        "Status": "complete" if not errors else "incomplete",
        "N": len(samples),
        "LSE-D": (((lse.get("aggregate") or {}).get("lse_d") or {}).get("mean")) if lse else None,
        "LSE-C": (((lse.get("aggregate") or {}).get("lse_c") or {}).get("mean")) if lse else None,
        "FID": details.get("fid", {}).get("fid") if isinstance(details.get("fid"), dict) else None,
        "FVD": details.get("fvd", {}).get("fvd") if isinstance(details.get("fvd"), dict) else None,
        "PSNR": _mean(pair, "psnr"),
        "SSIM": _mean(pair, "ssim"),
        "LPIPS": _mean(pair, "lpips"),
        "M-LMD": _mean(pair, "mouth_lmd"),
        "F-LMD": _mean(pair, "face_lmd"),
        "EmotiEff-Acc": emot.get("accuracy") if isinstance(emot, dict) else None,
        "DFER-CLIP-Acc": dfer.get("accuracy") if isinstance(dfer, dict) else None,
        "DFER-N": dfer.get("n_labelled_supported") if isinstance(dfer, dict) else None,
        "Protocol": PROTOCOL_VERSION,
        "Manifest-SHA256": manifest_fingerprint(samples, metrics),
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "status": row["Status"],
        "errors": errors,
        "metrics_requested": metrics,
        "table_row": row,
        "details": details,
        "per_video": list(per_video.values()),
    }
    return row, report


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen: seen.add(k); columns.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metrics", nargs="+", default=list(PAPER_METRICS), choices=PAPER_METRICS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-python", default=_python(DEFAULT_EVAL_PY))
    p.add_argument("--fvd-python", default=_python(DEFAULT_FVD_PY))
    p.add_argument("--lse-python", default=_python(DEFAULT_LSE_PY))
    p.add_argument("--pairwise-python", default=_python(DEFAULT_PAIRWISE_PY) if DEFAULT_PAIRWISE_PY.is_file() else _python(DEFAULT_EVAL_PY))
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--lse-min-track", type=int, default=5)
    p.add_argument("--fid-frame-stride", type=int, default=1)
    p.add_argument("--fvd-video-length", type=int, default=16)
    p.add_argument("--emotieff-model", default="enet_b2_8")
    p.add_argument("--dfer-segments", type=int, default=16)
    p.add_argument("--allow-partial", action="store_true",
                   help="Diagnostic only: write incomplete rows without failing process.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest, require_files=True)
    outdir = Path(args.output_dir).resolve()
    t0 = time.time()
    row, report = evaluate(samples, args.method, outdir, list(args.metrics), args)
    report["elapsed_sec"] = time.time() - t0
    (outdir / "paper_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(outdir / "paper_table.csv", [row], TABLE_COLUMNS)
    _write_csv(outdir / "per_video.csv", report["per_video"])
    print(f"[paper-eval] method={args.method} status={row['Status']} N={len(samples)}")
    print(f"[paper-eval] table: {outdir / 'paper_table.csv'}")
    if report["errors"]:
        for e in report["errors"]:
            print(f"  ERROR: {e}", file=sys.stderr)
    if row["Status"] != "complete" and not args.allow_partial:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

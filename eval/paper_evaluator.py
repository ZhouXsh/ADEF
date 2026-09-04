#!/usr/bin/env python3
"""Authoritative paper-table evaluator for ADEF and all baselines.

Protocol v3 keeps usable results when individual samples fail. Each metric is
aggregated over its own successful sample subset and exposes an explicit
coverage count in the paper table. All sample-level failures are written to
``failed_samples.csv`` and printed to stderr.

Status meanings:
- complete: every eligible sample succeeded for every requested metric.
- partial: every requested metric has a usable aggregate, but some samples
  failed or were unavailable upstream.
- failed: at least one requested metric has no usable aggregate at all.
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
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import (  # noqa: E402
    DFER_CLIP_EMOTIONS,
    PROTOCOL_VERSION,
    Sample,
    manifest_fingerprint,
    read_manifest,
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
    "Method", "Status", "N", "Evaluated-N",
    "LSE-D", "LSE-C", "LSE-N",
    "FID", "FID-N",
    "FVD", "FVD-N",
    "PSNR", "PSNR-N",
    "SSIM", "SSIM-N",
    "LPIPS", "LPIPS-N",
    "M-LMD", "F-LMD", "LMD-N",
    "EmotiEff-Acc", "EmotiEff-N",
    "DFER-CLIP-Acc", "DFER-N",
    "Protocol", "Manifest-SHA256",
]
FAILURE_COLUMNS = ["metric", "name", "fake", "gt", "error"]


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
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_if_present(path: Path) -> tuple[dict | None, str | None]:
    if not path.is_file():
        return None, f"output missing: {path}"
    try:
        return _load_json(path), None
    except Exception as exc:
        return None, f"invalid JSON {path}: {type(exc).__name__}: {exc}"


def _proc_tail(proc: dict) -> str:
    lines = (proc.get("stderr") or proc.get("stdout") or "").splitlines()
    return " | ".join(lines[-6:])


def _write_list(path: Path, values: list[str]) -> Path:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")
    return path


def _safe_stem(name: str, index: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or "sample"
    return f"{index:05d}_{safe}"


def _stage_emotion_inputs(samples: list[Sample], root: Path):
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
                shutil.copy2(src, dst)
            stem_to_sample[dst.stem] = s
            if s.emotion:
                lf.write(f"{dst.stem} {s.emotion}\n")
    return labels, stem_to_sample


def _mean(payload: dict, key: str):
    v = payload.get(key)
    return v.get("mean") if isinstance(v, dict) else None


def _failure(metric: str, sample: Sample | None, error: str, **extra) -> dict[str, Any]:
    fallback_name = extra.pop("name", "")
    fallback_fake = extra.pop("fake", "")
    fallback_gt = extra.pop("gt", "")
    row = {
        "metric": metric,
        "name": sample.name if sample else fallback_name,
        "fake": sample.fake if sample else fallback_fake,
        "gt": sample.gt if sample else fallback_gt,
        "error": error,
    }
    row.update(extra)
    return row


def _read_upstream_failures(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path)
    if not p.is_file():
        return [_failure("input", None, f"upstream failure file missing: {p}")]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return [_failure("input", None, f"cannot read upstream failures: {type(exc).__name__}: {exc}")]
    rows = data.get("failures", []) if isinstance(data, dict) else data
    out = []
    for raw in rows if isinstance(rows, list) else []:
        if not isinstance(raw, dict):
            continue
        out.append({
            "metric": str(raw.get("metric") or raw.get("stage") or "input"),
            "name": str(raw.get("name") or ""),
            "fake": str(raw.get("fake") or ""),
            "gt": str(raw.get("gt") or ""),
            "error": str(raw.get("error") or "upstream sample failure"),
        })
    return out


def _sample_by_index(samples: list[Sample], index: Any) -> Sample | None:
    try:
        i = int(index)
    except (TypeError, ValueError):
        return None
    return samples[i] if 0 <= i < len(samples) else None


def _add_global_failure(failures: list[dict], metric: str, error: str) -> None:
    failures.append(_failure(metric, None, error))


def evaluate(samples: list[Sample], method: str, outdir: Path, metrics: list[str], args) -> tuple[dict, dict]:
    outdir.mkdir(parents=True, exist_ok=True)
    work = outdir / "work"
    work.mkdir(parents=True, exist_ok=True)

    expected_n = args.expected_n if args.expected_n is not None else len(samples)
    if expected_n < len(samples):
        raise ValueError(f"--expected-n {expected_n} is smaller than manifest size {len(samples)}")

    details: dict[str, Any] = {}
    failures: list[dict[str, Any]] = _read_upstream_failures(args.upstream_failures)
    hard_errors: list[str] = []
    coverage: dict[str, int | None] = {}
    per_video: dict[str, dict[str, Any]] = {
        s.name: {"name": s.name, "fake": s.fake, "gt": s.gt, "emotion": s.emotion}
        for s in samples
    }

    fake_list = _write_list(work / "fake.txt", [s.fake for s in samples])
    gt_list = _write_list(work / "gt.txt", [s.gt for s in samples])

    if "lse" in metrics:
        out = work / "lse.json"
        out.unlink(missing_ok=True)
        cmd = [args.lse_python, str(LSE_SCRIPT), "--filelist", str(fake_list),
               "--output_json", str(out), "--device", args.device,
               "--min-track", str(args.lse_min_track)]
        proc = _run(cmd, cwd=LSE_SCRIPT.parent, timeout=args.timeout)
        data, load_err = _load_json_if_present(out)
        if data is None:
            msg = f"LSE unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "LSE", msg)
            details["lse"] = {"process": proc, "error": msg}
            coverage["lse"] = 0
        else:
            data["process"] = proc; details["lse"] = data
            results = data.get("results", [])
            n_ok = 0
            for i, s in enumerate(samples):
                r = results[i] if i < len(results) else {"error": "missing LSE result row"}
                per_video[s.name].update({"LSE-D": r.get("lse_d"), "LSE-C": r.get("lse_c")})
                if r.get("error") or r.get("lse_d") is None or r.get("lse_c") is None:
                    failures.append(_failure("LSE", s, str(r.get("error") or "missing LSE score")))
                else:
                    n_ok += 1
            coverage["lse"] = n_ok
            if n_ok == 0:
                msg = "LSE has zero successful samples"
                hard_errors.append(msg); _add_global_failure(failures, "LSE", msg)

    if "fid" in metrics:
        out = work / "fid.json"
        out.unlink(missing_ok=True)
        cmd = [args.eval_python, str(FID_SCRIPT), "--list1", str(gt_list), "--list2", str(fake_list),
               "--output-json", str(out), "--device", args.device]
        if args.fid_frame_stride != 1:
            cmd += ["--frame-stride", str(args.fid_frame_stride)]
        proc = _run(cmd, cwd=FID_SCRIPT.parent, timeout=args.timeout)
        data, load_err = _load_json_if_present(out)
        if data is None:
            msg = f"FID unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "FID", msg)
            details["fid"] = {"process": proc, "error": msg}; coverage["fid"] = 0
        else:
            data["process"] = proc; details["fid"] = data
            coverage["fid"] = int(data.get("n_success") if data.get("n_success") is not None else len(samples))
            for f in data.get("failures", []):
                s = _sample_by_index(samples, f.get("index"))
                failures.append(_failure("FID", s, str(f.get("error") or "FID pair failed"),
                                         name=f.get("fake", "") if s is None else ""))
            if data.get("fid") is None:
                msg = str(data.get("global_error") or "FID has no usable value")
                hard_errors.append(msg); _add_global_failure(failures, "FID", msg)

    if "fvd" in metrics:
        out = work / "fvd.json"
        out.unlink(missing_ok=True)
        cmd = [args.fvd_python, str(FVD_SCRIPT), "--real_list", str(gt_list), "--fake_list", str(fake_list),
               "--video_length", str(args.fvd_video_length), "--output_file", str(out)]
        proc = _run(cmd, cwd=FVD_SCRIPT.parent, timeout=args.timeout)
        data, load_err = _load_json_if_present(out)
        if data is None:
            msg = f"FVD unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "FVD", msg)
            details["fvd"] = {"process": proc, "error": msg}; coverage["fvd"] = 0
        else:
            data["process"] = proc; details["fvd"] = data
            coverage["fvd"] = int(data.get("n_success") if data.get("n_success") is not None else data.get("num_videos", 0))
            for f in data.get("failures", []):
                s = _sample_by_index(samples, f.get("index"))
                failures.append(_failure("FVD", s, str(f.get("error") or "FVD pair failed")))
            if data.get("fvd") is None:
                msg = str(data.get("global_error") or "FVD has no usable value")
                hard_errors.append(msg); _add_global_failure(failures, "FVD", msg)

    if "pairwise" in metrics:
        manifest = work / "pairwise_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["name", "fake", "gt", "emotion"])
            for s in samples:
                w.writerow([s.name, s.fake, s.gt, s.emotion or ""])
        out = work / "pairwise.json"
        out.unlink(missing_ok=True)
        cmd = [args.pairwise_python, str(PAIRWISE_SCRIPT), "--manifest", str(manifest),
               "--output", str(out), "--device", args.device]
        proc = _run(cmd, cwd=THIS_DIR, timeout=args.timeout)
        data, load_err = _load_json_if_present(out)
        if data is None:
            msg = f"Pairwise unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "pairwise", msg)
            details["pairwise"] = {"process": proc, "error": msg}
            for k in ("psnr", "ssim", "lpips", "lmd"):
                coverage[k] = 0
        else:
            data["process"] = proc; details["pairwise"] = data
            cov = data.get("coverage", {})
            for k in ("psnr", "ssim", "lpips", "lmd"):
                coverage[k] = int(cov.get(k) or 0)
            for f in data.get("failures", []):
                name = str(f.get("name") or "")
                s = next((x for x in samples if x.name == name), None)
                metric = str(f.get("metric") or "pairwise")
                failures.append(_failure(metric.upper() if metric != "lmd" else "LMD", s,
                                         str(f.get("error") or "pairwise sample failed"),
                                         name=name if s is None else ""))
            for r in data.get("per_video", []):
                if r.get("name") in per_video:
                    per_video[r["name"]].update({
                        "PSNR": r.get("psnr"), "SSIM": r.get("ssim"), "LPIPS": r.get("lpips"),
                        "M-LMD": r.get("mouth_lmd"), "F-LMD": r.get("face_lmd"),
                    })
            agg = data.get("aggregate", {})
            required_pairs = {
                "psnr": _mean(agg, "psnr"),
                "ssim": _mean(agg, "ssim"),
                "lpips": _mean(agg, "lpips"),
                "lmd": _mean(agg, "mouth_lmd"),
            }
            for key, value in required_pairs.items():
                if value is None:
                    msg = f"pairwise metric {key} has no usable value"
                    hard_errors.append(msg); _add_global_failure(failures, key.upper(), msg)

    emotion_stage = None
    label_file = None
    stem_map = None
    if "emotiefflib" in metrics or "dfer_clip" in metrics:
        emotion_stage = work / "emotion_inputs"
        label_file, stem_map = _stage_emotion_inputs(samples, emotion_stage)

    if "emotiefflib" in metrics and emotion_stage is not None:
        out = work / "emotiefflib.json"
        out.unlink(missing_ok=True)
        cmd = [args.eval_python, str(EMOTIEFF_SCRIPT), "--video_dir", str(emotion_stage),
               "--label_file", str(label_file), "--model", args.emotieff_model,
               "--device", args.device, "--quiet", "--output", str(out)]
        proc = _run(cmd, cwd=EMOTIEFF_SCRIPT.parent, timeout=args.timeout)
        data, load_err = _load_json_if_present(out)
        if data is None:
            msg = f"EmotiEff unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "EmotiEff", msg)
            details["emotiefflib"] = {"process": proc, "error": msg}; coverage["emotieff"] = 0
        else:
            successful_labelled = 0
            correct = 0
            for r in data.get("results", []):
                stem = Path(r.get("video", "")).stem
                s = stem_map.get(stem) if stem_map else None
                pred = (r.get("summary") or {}).get("dominant_emotion")
                label = s.emotion if s else r.get("label")
                error = r.get("error")
                if error or pred is None or label is None:
                    reason = str(error or ("no valid dominant emotion" if pred is None else "missing target emotion label"))
                    failures.append(_failure("EmotiEff", s, reason, name=stem if s is None else ""))
                else:
                    successful_labelled += 1
                    is_correct = str(pred).lower() == str(label).lower()
                    correct += int(is_correct)
                    r["correct_v3"] = is_correct
                if s:
                    per_video[s.name]["EmotiEff-Correct"] = r.get("correct_v3")
                    per_video[s.name]["EmotiEff-Pred"] = pred
            data["accuracy_v3"] = (correct / successful_labelled) if successful_labelled else None
            data["n_success_v3"] = successful_labelled
            data["process"] = proc
            details["emotiefflib"] = data
            coverage["emotieff"] = successful_labelled
            if successful_labelled == 0:
                msg = "EmotiEff has zero successfully labelled samples"
                hard_errors.append(msg); _add_global_failure(failures, "EmotiEff", msg)

    if "dfer_clip" in metrics and emotion_stage is not None:
        out = work / "dfer_clip.json"
        out.unlink(missing_ok=True)
        cmd = [args.eval_python, str(DFER_SCRIPT), "--video_dir", str(emotion_stage),
               "--label_file", str(label_file), "--device", args.device,
               "--num_segments", str(args.dfer_segments), "--quiet", "--output", str(out)]
        proc = _run(cmd, cwd=DFER_SCRIPT.parent, timeout=args.timeout)
        data, load_err = _load_json_if_present(out)
        if data is None:
            msg = f"DFER-CLIP unavailable: {load_err}; rc={proc['rc']} {_proc_tail(proc)}"
            hard_errors.append(msg); _add_global_failure(failures, "DFER-CLIP", msg)
            details["dfer_clip"] = {"process": proc, "error": msg}; coverage["dfer"] = 0
        else:
            successful_supported = 0
            correct = 0
            for r in data.get("results", []):
                stem = Path(r.get("video", "")).stem
                s = stem_map.get(stem) if stem_map else None
                label = s.emotion if s else r.get("label")
                supported = label in DFER_CLIP_EMOTIONS if label is not None else False
                pred = r.get("prediction")
                if label is None:
                    failures.append(_failure("DFER-CLIP", s, "missing target emotion label",
                                             name=stem if s is None else ""))
                    if s:
                        per_video[s.name]["DFER-Label-Supported"] = False
                    continue
                if not supported:
                    if s:
                        per_video[s.name]["DFER-Label-Supported"] = False
                    continue
                if r.get("error") or pred is None:
                    failures.append(_failure("DFER-CLIP", s, str(r.get("error") or "no prediction"),
                                             name=stem if s is None else ""))
                else:
                    successful_supported += 1
                    is_correct = str(pred).lower() == str(label).lower()
                    correct += int(is_correct)
                    r["correct_v3"] = is_correct
                if s:
                    per_video[s.name]["DFER-Correct"] = r.get("correct_v3")
                    per_video[s.name]["DFER-Pred"] = pred
                    per_video[s.name]["DFER-Label-Supported"] = True
            data["accuracy_v3"] = (correct / successful_supported) if successful_supported else None
            data["n_success_supported_v3"] = successful_supported
            data["process"] = proc
            details["dfer_clip"] = data
            coverage["dfer"] = successful_supported
            eligible = sum(s.emotion in DFER_CLIP_EMOTIONS for s in samples)
            if eligible > 0 and successful_supported == 0:
                msg = "DFER-CLIP has zero successful supported-label samples"
                hard_errors.append(msg); _add_global_failure(failures, "DFER-CLIP", msg)

    lse = details.get("lse", {})
    pair = details.get("pairwise", {}).get("aggregate", {}) if isinstance(details.get("pairwise"), dict) else {}
    emot = details.get("emotiefflib", {})
    dfer = details.get("dfer_clip", {})

    if hard_errors:
        status = "failed"
    elif failures or expected_n != len(samples):
        status = "partial"
    else:
        status = "complete"

    upstream_for_hash = _read_upstream_failures(args.upstream_failures)
    row = {
        "Method": method,
        "Status": status,
        "N": expected_n,
        "Evaluated-N": len(samples),
        "LSE-D": (((lse.get("aggregate") or {}).get("lse_d") or {}).get("mean")) if lse else None,
        "LSE-C": (((lse.get("aggregate") or {}).get("lse_c") or {}).get("mean")) if lse else None,
        "LSE-N": coverage.get("lse"),
        "FID": details.get("fid", {}).get("fid") if isinstance(details.get("fid"), dict) else None,
        "FID-N": coverage.get("fid"),
        "FVD": details.get("fvd", {}).get("fvd") if isinstance(details.get("fvd"), dict) else None,
        "FVD-N": coverage.get("fvd"),
        "PSNR": _mean(pair, "psnr"), "PSNR-N": coverage.get("psnr"),
        "SSIM": _mean(pair, "ssim"), "SSIM-N": coverage.get("ssim"),
        "LPIPS": _mean(pair, "lpips"), "LPIPS-N": coverage.get("lpips"),
        "M-LMD": _mean(pair, "mouth_lmd"), "F-LMD": _mean(pair, "face_lmd"),
        "LMD-N": coverage.get("lmd"),
        "EmotiEff-Acc": emot.get("accuracy_v3") if isinstance(emot, dict) else None,
        "EmotiEff-N": coverage.get("emotieff"),
        "DFER-CLIP-Acc": dfer.get("accuracy_v3") if isinstance(dfer, dict) else None,
        "DFER-N": coverage.get("dfer"),
        "Protocol": PROTOCOL_VERSION,
        "Manifest-SHA256": manifest_fingerprint(
            samples, metrics,
            context={"expected_n": expected_n, "upstream_failures": upstream_for_hash},
        ),
    }
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "status": status,
        "expected_n": expected_n,
        "evaluated_n": len(samples),
        "hard_errors": hard_errors,
        "errors": hard_errors,
        "failures": failures,
        "coverage": coverage,
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
                if k not in seen:
                    seen.add(k); columns.append(k)
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
    p.add_argument("--expected-n", type=int, default=None,
                   help="Original requested sample count before upstream generation/input failures.")
    p.add_argument("--upstream-failures", default=None,
                   help="JSON list/dict of samples unavailable before metric evaluation.")
    p.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    samples = read_manifest(args.manifest, require_files=True)
    outdir = Path(args.output_dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    row, report = evaluate(samples, args.method, outdir, list(args.metrics), args)
    report["elapsed_sec"] = time.time() - t0
    (outdir / "paper_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(outdir / "paper_table.csv", [row], TABLE_COLUMNS)
    _write_csv(outdir / "per_video.csv", report["per_video"])
    _write_csv(outdir / "failed_samples.csv", report["failures"], FAILURE_COLUMNS)

    print(f"[paper-eval] method={args.method} status={row['Status']} N={row['N']} evaluated={row['Evaluated-N']}")
    print(f"[paper-eval] table: {outdir / 'paper_table.csv'}")
    if report["failures"]:
        print(f"[paper-eval] failed samples: {outdir / 'failed_samples.csv'}", file=sys.stderr)
        for f in report["failures"]:
            label = f.get("name") or f.get("fake") or "<global>"
            print(f"  FAIL [{f.get('metric')}] {label}: {f.get('error')}", file=sys.stderr)
    if report["hard_errors"]:
        for e in report["hard_errors"]:
            print(f"  ERROR: {e}", file=sys.stderr)
    return 2 if row["Status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())

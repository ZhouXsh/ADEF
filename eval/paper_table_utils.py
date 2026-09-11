#!/usr/bin/env python3
"""Paper-table finalization and unified per-video bookkeeping.

Paper protocol v4 follows a record-first rule: retain one row per video with
frame counts, video-level metric values and participation flags, then derive
paper-facing means from those recorded video-level values. FID/FVD remain
standard dataset-level Frechet metrics and are never averaged per video.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

PAPER_TABLE_COLUMNS = [
    "Method",
    "LSE-D", "LSE-C",
    "FID", "FVD",
    "PSNR", "SSIM", "LPIPS",
    "M-LMD", "F-LMD",
    "EmotiEff-Acc", "DFER-CLIP-Acc",
]


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _frame_count(path: str | None) -> int | None:
    if not path:
        return None
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n > 0:
            cap.release()
            return n
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        return count or None
    except Exception:
        return None


def _finite_mean(values) -> float | None:
    xs = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            xs.append(x)
    return sum(xs) / len(xs) if xs else None


def _psnr_mean(values) -> float | None:
    xs = []
    for value in values:
        try:
            x = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(x):
            continue
        xs.append(x)
    if not xs:
        return None
    if any(math.isinf(x) and x > 0 for x in xs):
        return float("inf")
    finite = [x for x in xs if math.isfinite(x)]
    return sum(finite) / len(finite) if finite else None


def _bool_mean(values) -> float | None:
    xs = [int(v) for v in values if isinstance(v, bool)]
    return sum(xs) / len(xs) if xs else None


def _staged_index(video: str | None) -> int | None:
    stem = Path(video or "").stem
    head = stem.split("_", 1)[0]
    try:
        return int(head)
    except ValueError:
        return None


def _unified_per_video(outdir: Path, report: dict) -> list[dict[str, Any]]:
    raw_rows = report.get("per_video", [])
    rows = [dict(r) for r in raw_rows if isinstance(r, dict)]
    by_name = {str(r.get("name")): r for r in rows if r.get("name")}
    ordered_names = [str(r.get("name")) for r in rows]

    # Always record source video frame counts, independent of any metric.
    for row in rows:
        row["Fake-Frames"] = _frame_count(row.get("fake"))
        row["GT-Frames"] = _frame_count(row.get("gt"))

    work = outdir / "work"

    lse = _load_json(work / "lse.json")
    if lse:
        results = lse.get("results", [])
        for i, name in enumerate(ordered_names):
            if i >= len(results) or not isinstance(results[i], dict):
                continue
            r = results[i]
            row = by_name[name]
            row["LSE-D"] = r.get("lse_d")
            row["LSE-C"] = r.get("lse_c")
            row["LSE-OK"] = not bool(r.get("error")) and r.get("lse_d") is not None and r.get("lse_c") is not None
            row["LSE-SyncNet-Frames"] = r.get("n_frames")
            row["LSE-Tracks"] = r.get("n_tracks")
            row["LSE-Error"] = r.get("error")

    # Only the completed final pairwise JSON is eligible for paper aggregation.
    # The checkpoint may be incomplete and is intentionally not promoted to a
    # final paper value, although pairwise_per_video.csv remains available for
    # progress/resume inspection.
    pair = _load_json(work / "pairwise.json")
    if pair and pair.get("complete", True):
        for r in pair.get("per_video", []):
            if not isinstance(r, dict) or r.get("name") not in by_name:
                continue
            row = by_name[str(r["name"])]
            row.update({
                "Pairwise-Fake-Frames": r.get("frames_fake"),
                "Pairwise-GT-Frames": r.get("frames_gt"),
                "Pairwise-Paired-Frames": r.get("frames_paired"),
                "Pixel-Aligned-Frames": r.get("frames_pixel_aligned"),
                "LMD-Aligned-Frames": r.get("frames_lmd_aligned"),
                "LMD-Valid-Frames": r.get("frames_lmd_valid"),
                "PSNR": r.get("psnr"),
                "SSIM": r.get("ssim"),
                "LPIPS": r.get("lpips"),
                "M-LMD": r.get("mouth_lmd"),
                "F-LMD": r.get("face_lmd"),
                "PSNR-OK": r.get("psnr_ok"),
                "SSIM-OK": r.get("ssim_ok"),
                "LPIPS-OK": r.get("lpips_ok"),
                "LMD-OK": r.get("lmd_ok"),
                "Pairwise-Error": r.get("error"),
            })

    emot = _load_json(work / "emotiefflib.json")
    if emot:
        for r in emot.get("results", []):
            if not isinstance(r, dict):
                continue
            idx = _staged_index(r.get("video"))
            if idx is None or idx < 0 or idx >= len(ordered_names):
                continue
            row = by_name[ordered_names[idx]]
            summary = r.get("summary") if isinstance(r.get("summary"), dict) else {}
            pred = summary.get("dominant_emotion")
            label = row.get("emotion") or r.get("label")
            ok = not bool(r.get("error")) and pred is not None and label is not None
            correct = (str(pred).lower() == str(label).lower()) if ok else None
            row.update({
                "EmotiEff-Pred": pred,
                "EmotiEff-Correct": correct,
                "EmotiEff-OK": ok,
                "EmotiEff-Frames": summary.get("frames_total"),
                "EmotiEff-Frames-Analyzed": summary.get("frames_analyzed"),
                "EmotiEff-Frames-With-Face": summary.get("frames_with_face"),
                "EmotiEff-Face-Detection-Rate": summary.get("face_detection_rate"),
                "EmotiEff-Error": r.get("error"),
            })

    dfer = _load_json(work / "dfer_clip.json")
    if dfer:
        nseg = dfer.get("num_segments")
        for r in dfer.get("results", []):
            if not isinstance(r, dict):
                continue
            idx = _staged_index(r.get("video"))
            if idx is None or idx < 0 or idx >= len(ordered_names):
                continue
            row = by_name[ordered_names[idx]]
            supported = bool(r.get("label_supported")) and row.get("emotion") != "contempt"
            pred = r.get("prediction")
            ok = supported and not bool(r.get("error")) and pred is not None and row.get("emotion") is not None
            correct = (str(pred).lower() == str(row.get("emotion")).lower()) if ok else None
            row.update({
                "DFER-Pred": pred,
                "DFER-Correct": correct,
                "DFER-Label-Supported": supported,
                "DFER-OK": ok,
                "DFER-Sampled-Frames": nseg if ok else None,
                "DFER-Target-Probability": r.get("target_probability"),
                "DFER-Error": r.get("error"),
            })

    fid = _load_json(work / "fid.json")
    fid_failed = set()
    if fid:
        for f in fid.get("failures", []):
            try:
                fid_failed.add(int(f.get("index")))
            except (TypeError, ValueError):
                pass
        per_fid = fid.get("per_video", [])
        for i, name in enumerate(ordered_names):
            row = by_name[name]
            row["FID-Included"] = i not in fid_failed and fid.get("fid") is not None
            if i < len(per_fid) and isinstance(per_fid[i], dict):
                r = per_fid[i]
                row["FID-Real-Frames"] = r.get("real_frames")
                row["FID-Fake-Frames"] = r.get("fake_frames")
                if r.get("included") is not None:
                    row["FID-Included"] = bool(r.get("included")) and fid.get("fid") is not None
                row["FID-Error"] = r.get("error")

    fvd = _load_json(work / "fvd.json")
    fvd_failed = set()
    if fvd:
        for f in fvd.get("failures", []):
            try:
                fvd_failed.add(int(f.get("index")))
            except (TypeError, ValueError):
                pass
        per_fvd = fvd.get("per_video", [])
        for i, name in enumerate(ordered_names):
            row = by_name[name]
            row["FVD-Included"] = i not in fvd_failed and fvd.get("fvd") is not None
            row["FVD-Sampled-Frames"] = fvd.get("video_length") if row["FVD-Included"] else None
            if i < len(per_fvd) and isinstance(per_fvd[i], dict):
                r = per_fvd[i]
                row["FVD-Real-Frames"] = r.get("real_frames")
                row["FVD-Fake-Frames"] = r.get("fake_frames")
                row["FVD-Sampled-Frames"] = r.get("sampled_frames")
                if r.get("included") is not None:
                    row["FVD-Included"] = bool(r.get("included")) and fvd.get("fvd") is not None
                row["FVD-Error"] = r.get("error")

    return rows


def _reaggregate(rows: list[dict[str, Any]], report: dict) -> dict[str, Any]:
    values: dict[str, Any] = {}
    values["LSE-D"] = _finite_mean(r.get("LSE-D") for r in rows if r.get("LSE-OK") is True)
    values["LSE-C"] = _finite_mean(r.get("LSE-C") for r in rows if r.get("LSE-OK") is True)
    values["PSNR"] = _psnr_mean(r.get("PSNR") for r in rows if r.get("PSNR-OK") is True)
    values["SSIM"] = _finite_mean(r.get("SSIM") for r in rows if r.get("SSIM-OK") is True)
    values["LPIPS"] = _finite_mean(r.get("LPIPS") for r in rows if r.get("LPIPS-OK") is True)
    values["M-LMD"] = _finite_mean(r.get("M-LMD") for r in rows if r.get("LMD-OK") is True)
    values["F-LMD"] = _finite_mean(r.get("F-LMD") for r in rows if r.get("LMD-OK") is True)
    values["EmotiEff-Acc"] = _bool_mean(r.get("EmotiEff-Correct") for r in rows if r.get("EmotiEff-OK") is True)
    values["DFER-CLIP-Acc"] = _bool_mean(r.get("DFER-Correct") for r in rows if r.get("DFER-OK") is True)

    details = report.get("details", {}) if isinstance(report.get("details"), dict) else {}
    fid = details.get("fid") if isinstance(details.get("fid"), dict) else {}
    fvd = details.get("fvd") if isinstance(details.get("fvd"), dict) else {}
    values["FID"] = fid.get("fid")
    values["FVD"] = fvd.get("fvd")
    return values


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trim_paper_table(path: str | Path) -> bool:
    """Finalize unified per-video records and rewrite the paper-facing table."""
    p = Path(path)
    if not p.is_file():
        return False
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False

    outdir = p.parent
    metrics_path = outdir / "paper_metrics.json"
    report = _load_json(metrics_path)
    if report:
        unified = _unified_per_video(outdir, report)
        if unified:
            report["per_video"] = unified
            final_values = _reaggregate(unified, report)
            report["paper_aggregation"] = {
                "rule": "record per-video values first; paper means are equal-weight means over successful videos",
                "dataset_level_exceptions": ["FID", "FVD"],
                "final_values": final_values,
            }
            table_row = report.get("table_row") if isinstance(report.get("table_row"), dict) else {}
            table_row.update(final_values)
            report["table_row"] = table_row
            rows[0].update(final_values)
            metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            _write_csv(outdir / "per_video.csv", unified)

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True

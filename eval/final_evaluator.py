#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
final_evaluator.py
==================

End-to-end orchestrator that ties together the BASELINE generator
(``run_baselines.py``) and the unified metric runner
(``unified_evaluator.py``).

Given an ordered list of ``(reference_image, audio, gt_video)`` triples
the script will:

1. For every triple and every requested baseline (wav2lip / sadtalker /
   eat_code / joyvasa / kdtalker by default), generate a fake talking-head
   video and place it under
   ``<RESULT>/<baseline>/<pair_name>.mp4``.
2. For every produced fake video, invoke ``unified_evaluator.py`` against
   the corresponding GT video and write a per-video JSON report at
   ``<RESULT>/<baseline>/<pair_name>.json``.
3. After generation + evaluation finish, walk the ``<RESULT>`` tree,
   collect every per-video metric value, and produce a 2-D paper-style
   CSV: rows = evaluation metric, columns = baseline × {mean, var, n}.

Usage
-----
The triple list can be supplied two ways::

    # Inline, positional arg quoting: "img.png,aud.wav,gt.mp4" (whitespace-separated pairs)
    python final_evaluator.py \
        --triples "img1.png,a1.wav,gt1.mp4" "img2.png,a2.wav,gt2.mp4" ...

    # Or a text file, one triple per line:
    python final_evaluator.py --triples-file triples.txt

Other useful flags::

    --baselines wav2lip sadtalker eat_code joyvasa kdtalker
    --metrics  lse fid eat emonet new_emo ...
    --output-root /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/RESULT
    --device cuda:0 --eat-device 0
    --timeout 1800
    --keep-workdir
    --dry-run   # parse triples + show plan, do not actually run

Output layout
-------------
::

    <RESULT>/
        wav2lip/
            pair_0000.mp4
            pair_0000.json
            pair_0001.mp4
            ...
        sadtalker/
            ...
        ...
        summary_mean.csv     # 2-D: rows=metric, cols=baseline_mean
        summary_var.csv      # 2-D: rows=metric, cols=baseline_var
        summary_long.csv     # long-format (baseline, metric, mean, var, std, n)
        per_video_detail.csv # one row per (baseline, pair)

The full per-video raw reports are kept on disk so any downstream tool
(``batch_eval_paired.py`` style, plot scripts, …) can re-ingest them
without re-running the heavy generation / evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent
UNIFIED_EVAL = EVAL_ROOT / "unified_evaluator.py"

# run_baselines.py lives outside the eval tree
BASELINE_RUNNER = Path("/home/Zhouxishi/VirtualMan_proj/BASELINE/run_baselines.py")
DEFAULT_RESULT_ROOT = EVAL_ROOT / "RESULT"

# Methods supported by run_baselines.METHODS
BASELINE_METHODS = ["wav2lip", "sadtalker", "eat_code", "joyvasa", "kdtalker"]

# Metrics supported by unified_evaluator.METRIC_REGISTRY
METRIC_CHOICES = [
    "lse", "fvd", "fid", "eat", "emonet",
    "emo_fan", "emotiefflib", "dfer_clip", "new_emo",
]


# ---------------------------------------------------------------------------
# Headline metric extraction
#
# Each tuple is (metric_group, dotted_payload_path, csv_column_name).  These
# are the leaf values that get aggregated into the final 2-D table.  Any
# value missing from a JSON report is silently skipped (NaN-style).
# ---------------------------------------------------------------------------
HEADLINE_METRICS: List[Tuple[str, str, str]] = [
    # Audio-visual sync
    ("lse", "lse_d",       "LSE-D"),
    ("lse", "lse_c",       "LSE-C"),
    # Distribution distances
    ("fvd", "fvd",         "FVD"),
    ("fid", "fid",         "FID"),
    # Pixel / landmark (EAT pipeline)
    ("eat", "psnr_ssim.psnr",         "PSNR"),
    ("eat", "psnr_ssim.ssim",         "SSIM"),
    ("eat", "lpips.mean_lpips",       "LPIPS"),
    ("eat", "lmd.mouth_lmd",          "M-LMD"),
    ("eat", "lmd.face_lmd",           "F-LMD"),
    ("eat", "sync.sync_conf",         "Sync-Conf"),
    ("eat", "emo.emo_acc",            "Emo-Acc"),
    # EmoNet
    ("emonet", "emo_acc",             "EmoNet-Acc"),
    ("emonet", "emo_sim",             "EmoNet-Sim"),
    # Reference-free emotion models (categorical — we'll record dominant_emotion
    # fraction and DFER-CLIP's `correct` flag as a soft numeric signal).
    ("emotiefflib", "dominant_fraction", "EmotiEff-DomFrac"),
    ("dfer_clip",   "correct",            "DFER-CLIP-Correct"),
    ("new_emo",     "agreement",          "NewEmo-Agreement"),
]


# ============================================================
# Utility helpers
# ============================================================

def log_block(title: str) -> None:
    print(f'\n{"=" * 70}\n  {title}\n{"=" * 70}', flush=True)


def log_sub(title: str) -> None:
    print(f'\n--- {title} ---', flush=True)


def pair_name(idx: int, image: Optional[str] = None, audio: Optional[str] = None,
              driving_video: Optional[str] = None, scenario: str = "final_eval") -> str:
    """Stable identifier matching ``run_baselines.get_pair_name`` exactly.

    Critically, this **delegates** to ``run_baselines.get_pair_name`` so the
    on-disk filenames always match what ``run_baselines.METHODS[…]`` writes —
    otherwise the "already produced" cache check below misses every file and we
    re-run generation unnecessarily (or worse, fail to find files that DO exist).
    """
    run_baselines = _import_run_baselines()
    return run_baselines.get_pair_name(
        image=image, audio=audio, driving_video=driving_video,
        scenario=scenario, idx=idx,
    )


def parse_triple_string(s: str) -> Optional[Dict[str, str]]:
    """Parse ``"image,audio,gt"`` → dict (None on failure)."""
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 3:
        print(f"[triples] WARN: 跳过 {s!r}（需要 image,audio,gt_video 三段）")
        return None
    img, aud, gt = parts[0], parts[1], parts[2]
    if not Path(img).is_file():
        print(f"[triples] WARN: 跳过 {s!r}（image 不存在: {img}）")
        return None
    if not Path(aud).is_file():
        print(f"[triples] WARN: 跳过 {s!r}（audio 不存在: {aud}）")
        return None
    if not Path(gt).is_file():
        print(f"[triples] WARN: 跳过 {s!r}（gt_video 不存在: {gt}）")
        return None
    return {"image": img, "audio": aud, "gt": gt}


def read_triples_file(p: Path) -> List[Dict[str, str]]:
    """Read triples from a text file: one ``image,audio,gt`` triple per
    line; ``#`` comments / blank lines are stripped."""
    out: List[Dict[str, str]] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        d = parse_triple_string(s)
        if d:
            out.append(d)
    return out


def resolve_triples(args) -> List[Dict[str, str]]:
    if args.triples_file:
        triples = read_triples_file(Path(args.triples_file))
    else:
        triples = []
        for raw in args.triples or []:
            d = parse_triple_string(raw)
            if d:
                triples.append(d)
    return triples


# ============================================================
# Phase 1 — baseline generation
# ============================================================

# We import lazily so --help and --dry-run stay snappy.
def _import_run_baselines():
    """Add /VirtualMan_proj/BASELINE to sys.path and import the module."""
    sys.path.insert(0, str(BASELINE_RUNNER.parent))
    import run_baselines  # type: ignore
    return run_baselines


def run_baseline_for_triple(
    method: str,
    triple: Dict[str, str],
    out_dir: Path,
    scenario: str = "final_eval",
    emo: str = "neutral",
    idx: int = 0,
    timeout: Optional[int] = None,
) -> Path:
    """Invoke one baseline on one triple; return the produced mp4 path.

    Raises ``RuntimeError`` on non-zero return code (caller is expected to
    log + skip evaluation).
    """
    run_baselines = _import_run_baselines()
    fn = run_baselines.METHODS[method]
    out_dir.mkdir(parents=True, exist_ok=True)

    # EAT_code is the only baseline whose on-disk filename includes the
    # driving video stem — ``run_baselines.run_eat_code`` calls
    # ``get_pair_name(image=…, audio=…, driving_video=…, …)`` whereas the
    # other four call it WITHOUT ``driving_video``.  Mirror that here so
    # our "already produced?" cache check actually finds the file.
    if method == "eat_code":
        name_for_lookup = pair_name(
            idx=idx, image=triple["image"], audio=triple["audio"],
            driving_video=triple["gt"], scenario=scenario,
        )
    else:
        name_for_lookup = pair_name(
            idx=idx, image=triple["image"], audio=triple["audio"],
            driving_video=None, scenario=scenario,
        )
    target = out_dir / f"{name_for_lookup}.mp4"

    # EAT_code requires a driving video (we pass the GT as the motion source)
    # AND its ``emo`` arg expects the short EAT label (e.g. "neu" not
    # "neutral").  Map it through run_baselines.EAT_EMO_MAP.
    if method == "eat_code":
        emo_mapped = run_baselines.EAT_EMO_MAP.get(emo, "neu")
        rc = fn(
            image=triple["image"],
            audio=triple["audio"],
            output_dir=str(out_dir),
            driving_video=triple["gt"],
            emo=emo_mapped,
            scenario=scenario,
            idx=idx,
        )
    else:
        rc = fn(
            image=triple["image"],
            audio=triple["audio"],
            output_dir=str(out_dir),
            driving_video=triple["gt"],
            scenario=scenario,
            idx=idx,
            emo=emo,
        )
    if rc != 0:
        raise RuntimeError(f"{method} returned rc={rc}")
    if not target.is_file():
        raise RuntimeError(f"{method} did not produce {target}")
    return target


# ============================================================
# Phase 2 — unified_evaluator invocation
# ============================================================

def evaluate_one(
    fake: Path,
    gt: Path,
    out_json: Path,
    name: str,
    metrics: Sequence[str],
    skip: Sequence[str],
    device: str,
    eat_device: str,
    timeout: int,
    workdir: Optional[Path] = None,
) -> int:
    """Call unified_evaluator.py as a subprocess; returns its return code.

    Writes the per-video report to ``out_json``.  Raises
    ``subprocess.TimeoutExpired`` on timeout (caller catches).
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if workdir is None:
        workdir = out_json.parent / "_work"
    workdir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable, str(UNIFIED_EVAL),
        "--fake", str(fake),
        "--gt",   str(gt),
        "--name", name,
        "--output", str(out_json),
        "--workdir", str(workdir),
        "--device", device,
        "--eat-device", eat_device,
        "--quiet",
    ]
    if metrics and list(metrics) != ["all"]:
        cmd.extend(["--metrics", *metrics])
    if skip:
        cmd.extend(["--skip", *skip])

    proc = subprocess.run(
        cmd,
        cwd=str(EVAL_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode


# ============================================================
# Phase 3 — aggregation / CSV
# ============================================================

def _safe_num(x: Any) -> Optional[float]:
    # Treat booleans as 0/1 so categorical fields like `agreement` /
    # `correct` are aggregated as fractions.
    if isinstance(x, bool):
        return float(int(x))
    if isinstance(x, (int, float)):
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    return None


def _dotted_get(d: Dict[str, Any], dotted_key: str) -> Optional[float]:
    cur: Any = d
    for seg in dotted_key.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return _safe_num(cur)


def collect_metric_values(
    result_root: Path,
    baselines: Sequence[str],
    triples: Sequence[Dict[str, str]],
) -> Tuple[
    Dict[Tuple[str, str], List[Optional[float]]],  # per-leaf values
    Dict[Tuple[str, str], List[Optional[Dict[str, Any]]]],  # raw payloads (debug)
]:
    """Walk ``<baseline>/<pair>.json`` for every (baseline, triple), pull
    the headline metric values.  Returns two dicts keyed by
    ``(metric_col, baseline)``:
      * the numeric value list (length == #triples, with None for failures)
      * the raw payloads (for downstream debugging / re-aggregation)
    """
    values: Dict[Tuple[str, str], List[Optional[float]]] = {}
    payloads: Dict[Tuple[str, str], List[Optional[Dict[str, Any]]]] = {}

    n = len(triples)
    for method in baselines:
        method_dir = result_root / method
        for metric_group, leaf_key, csv_col in HEADLINE_METRICS:
            values[(csv_col, method)] = [None] * n
            payloads[(csv_col, method)] = [None] * n
        for i, triple in enumerate(triples):
            json_path = method_dir / f"{pair_name(i, triple['image'], triple['audio'])}.json"
            if not json_path.is_file():
                # Either generation failed (logged earlier) or evaluation failed.
                continue
            try:
                report = json.loads(json_path.read_text())
            except Exception:
                continue
            metrics_block = report.get("metrics", {})
            for metric_group, leaf_key, csv_col in HEADLINE_METRICS:
                bucket = metrics_block.get(metric_group, {})
                if not bucket.get("ok"):
                    continue
                payload = bucket.get("payload") or {}
                v = _dotted_get(payload, leaf_key)
                values[(csv_col, method)][i] = v
                payloads[(csv_col, method)][i] = payload
    return values, payloads


def aggregate(values: List[Optional[float]]) -> Optional[Dict[str, float]]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    if len(nums) == 1:
        var = 0.0
    else:
        var = statistics.pvariance(nums)
    return {
        "n":   len(nums),
        "mean": float(statistics.mean(nums)),
        "var":  float(var),
        "std":  float(statistics.pstdev(nums)),
        "min":  float(min(nums)),
        "max":  float(max(nums)),
    }


def build_summary_csvs(
    result_root: Path,
    baselines: Sequence[str],
    triples: Sequence[Dict[str, str]],
) -> Dict[str, Path]:
    """Write three CSV files under ``result_root``:
      * ``summary_mean.csv`` — rows = metric, cols = baseline (mean)
      * ``summary_var.csv``  — rows = metric, cols = baseline (variance)
      * ``summary_long.csv`` — long format (baseline, metric, n, mean, var, std, min, max)
      * ``per_video_detail.csv`` — one row per (baseline, pair)

    Returns a dict mapping logical name → written file path.
    """
    values, _payloads = collect_metric_values(result_root, baselines, triples)

    written: Dict[str, Path] = {}

    # ---- wide-format mean / variance CSVs ----
    rows_mean = [["metric", *baselines]]
    rows_var  = [["metric", *baselines]]
    for _grp, _leaf, csv_col in HEADLINE_METRICS:
        line_m = [csv_col]
        line_v = [csv_col]
        for method in baselines:
            stats = aggregate(values[(csv_col, method)])
            if stats is None:
                line_m.append("")
                line_v.append("")
            else:
                line_m.append(f"{stats['mean']:.4f}")
                line_v.append(f"{stats['var']:.4f}")
        rows_mean.append(line_m)
        rows_var.append(line_v)

    mean_csv = result_root / "summary_mean.csv"
    with mean_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_mean)
    written["summary_mean"] = mean_csv

    var_csv = result_root / "summary_var.csv"
    with var_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows_var)
    written["summary_var"] = var_csv

    # ---- long-format CSV ----
    long_csv = result_root / "summary_long.csv"
    with long_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["baseline", "metric", "n", "mean", "var", "std", "min", "max"])
        for method in baselines:
            for _grp, _leaf, csv_col in HEADLINE_METRICS:
                stats = aggregate(values[(csv_col, method)])
                if stats is None:
                    writer.writerow([method, csv_col, 0, "", "", "", "", ""])
                else:
                    writer.writerow([
                        method, csv_col, stats["n"],
                        f"{stats['mean']:.4f}", f"{stats['var']:.4f}",
                        f"{stats['std']:.4f}",
                        f"{stats['min']:.4f}", f"{stats['max']:.4f}",
                    ])
    written["summary_long"] = long_csv

    # ---- per-video detail CSV ----
    detail_csv = result_root / "per_video_detail.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["baseline", "pair", "image", "audio", "gt",
                  "fake_mp4", "report_json", "ok"]
        for _grp, _leaf, csv_col in HEADLINE_METRICS:
            header.append(csv_col)
        writer.writerow(header)
        for i, triple in enumerate(triples):
            p = pair_name(i, triple["image"], triple["audio"])
            for method in baselines:
                fake_mp4 = result_root / method / f"{p}.mp4"
                json_path = result_root / method / f"{p}.json"
                row: List[Any] = [
                    method, p, triple["image"], triple["audio"], triple["gt"],
                    str(fake_mp4) if fake_mp4.is_file() else "",
                    str(json_path) if json_path.is_file() else "",
                    int(json_path.is_file()),
                ]
                for metric_group, leaf_key, csv_col in HEADLINE_METRICS:
                    v = values.get((csv_col, method), [None] * len(triples))[i]
                    row.append("" if v is None else f"{v:.4f}")
                writer.writerow(row)
    written["per_video_detail"] = detail_csv
    return written


# ============================================================
# Main
# ============================================================

@dataclass
class TripleRecord:
    triple: Dict[str, str]
    idx: int
    name: str
    fakes: Dict[str, Path] = field(default_factory=dict)
    jsons: Dict[str, Path] = field(default_factory=dict)
    failed_baselines: List[str] = field(default_factory=list)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="End-to-end pipeline: BASELINE generation + unified "
                    "evaluation + 2-D CSV aggregation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--triples", nargs="+",
                     help="Inline triple list: \"image,audio,gt_video\" "
                          "(whitespace-separated multiple triples).")
    src.add_argument("--triples-file", type=str,
                     help="Text file with one \"image,audio,gt_video\" per line.")

    p.add_argument("--baselines", nargs="+", default=BASELINE_METHODS,
                   choices=BASELINE_METHODS,
                   help="Which BASELINE methods to run.")
    p.add_argument("--metrics", nargs="+", default=["all"],
                   help="Subset of unified_evaluator metrics to run. Choices: "
                        + " ".join(METRIC_CHOICES) + " (or 'all').")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Metrics to skip (subtractive filter on top of --metrics).")
    p.add_argument("--output-root", type=str, default=str(DEFAULT_RESULT_ROOT),
                   help="Root directory for fake videos / JSON / CSV outputs.")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device passed to unified_evaluator.")
    p.add_argument("--eat-device", default="0",
                   help="CUDA device index (just the integer) passed to EAT.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-video subprocess timeout (seconds).")
    p.add_argument("--scenario", default="final_eval",
                   help="scenario tag passed to run_baselines (purely cosmetic "
                        "for pair-name construction).")
    p.add_argument("--emo", default="neutral",
                   help="Default emotion label forwarded to EAT_code.  If you "
                        "want per-triple emo, edit --triples-file with a "
                        "4-column format (see README in this file).")
    p.add_argument("--keep-workdir", action="store_true",
                   help="Preserve per-video _work/ subfolders after evaluation.")
    p.add_argument("--no-aggregate", action="store_true",
                   help="Skip the final CSV aggregation (useful when only the "
                        "fake videos / per-video JSONs are wanted).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned schedule and exit without "
                        "generating or evaluating anything.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    triples = resolve_triples(args)
    if not triples:
        print("ERROR: 解析后没有可用 triple", file=sys.stderr)
        return 2
    print(f"[plan] triples = {len(triples)}")
    print(f"[plan] baselines = {args.baselines}")
    print(f"[plan] metrics   = {args.metrics} (skip={args.skip})")
    print(f"[plan] output_root = {args.output_root}")
    for i, t in enumerate(triples[:3]):
        print(f"[plan]   [{i}] image={t['image']}\n[plan]       audio={t['audio']}\n[plan]       gt={t['gt']}")
    if len(triples) > 3:
        print(f"[plan]   ... ({len(triples) - 3} more)")

    if args.dry_run:
        print("[dry-run] 仅打印计划，未启动任何子进程。")
        return 0

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records: List[TripleRecord] = []
    overall_t0 = time.time()

    # ---------------- Phase 1: generation ----------------
    log_block("Phase 1: baseline generation")
    for method in args.baselines:
        method_dir = output_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        log_sub(f"baseline: {method}  →  {method_dir}")
        for i, triple in enumerate(triples):
            # Per-method naming: only EAT_code includes driving_video in its
            # filename (see run_baseline_for_triple for the rationale).
            if method == "eat_code":
                name = pair_name(
                    idx=i, image=triple["image"], audio=triple["audio"],
                    driving_video=triple["gt"], scenario=args.scenario,
                )
            else:
                name = pair_name(
                    idx=i, image=triple["image"], audio=triple["audio"],
                    driving_video=None, scenario=args.scenario,
                )
            target = method_dir / f"{name}.mp4"
            rec = next((r for r in records if r.idx == i), None)
            if rec is None:
                rec = TripleRecord(triple=triple, idx=i, name=name)
                records.append(rec)
            # If a different method produced a fake with the same ``name``
            # earlier, this method's record still gets the right fake (each
            # method has its own sub-directory).

            if target.is_file():
                if not args.quiet:
                    print(f"  [{method} {i+1}/{len(triples)}] 已存在 {target.name}")
                rec.fakes[method] = target
                continue

            t0 = time.time()
            try:
                out_path = run_baseline_for_triple(
                    method=method,
                    triple=triple,
                    out_dir=method_dir,
                    scenario=args.scenario,
                    emo=args.emo,
                    idx=i,
                    timeout=args.timeout,
                )
                rec.fakes[method] = out_path
                print(f"  [{method} {i+1}/{len(triples)}] OK → {out_path.name} ({time.time()-t0:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                rec.failed_baselines.append(method)
                print(f"  [{method} {i+1}/{len(triples)}] FAIL: {exc}")
                if not args.quiet:
                    traceback.print_exc()

    # ---------------- Phase 2: evaluation ----------------
    log_block("Phase 2: unified evaluation")
    eval_t0 = time.time()
    for i, rec in enumerate(records):
        for method in args.baselines:
            if method not in rec.fakes:
                continue
            fake = rec.fakes[method]
            method_dir = output_root / method
            json_path = method_dir / f"{rec.name}.json"
            # ``rec.name`` was already computed in Phase 1 via pair_name() with
            # the same (image, audio, driving_video, scenario, idx) — both
            # phases agree on the filename.
            if json_path.is_file():
                if not args.quiet:
                    print(f"  [{method} {i+1}/{len(records)}] 已存在 {json_path.name}（跳过）")
                rec.jsons[method] = json_path
                continue
            t0 = time.time()
            try:
                rc = evaluate_one(
                    fake=fake,
                    gt=Path(rec.triple["gt"]),
                    out_json=json_path,
                    name=rec.name,
                    metrics=args.metrics,
                    skip=args.skip,
                    device=args.device,
                    eat_device=args.eat_device,
                    timeout=args.timeout,
                    workdir=method_dir / "_work" / rec.name,
                )
                elapsed = time.time() - t0
                if rc != 0 or not json_path.is_file():
                    print(f"  [{method} {i+1}/{len(records)}] FAIL: rc={rc}, "
                          f"json exists={json_path.is_file()} ({elapsed:.1f}s)")
                else:
                    rec.jsons[method] = json_path
                    print(f"  [{method} {i+1}/{len(records)}] OK ({elapsed:.1f}s)")
            except subprocess.TimeoutExpired:
                print(f"  [{method} {i+1}/{len(records)}] TIMEOUT ({time.time()-t0:.1f}s)")
            except Exception as exc:  # noqa: BLE001
                print(f"  [{method} {i+1}/{len(records)}] ERROR: {exc}")
                if not args.quiet:
                    traceback.print_exc()

    eval_elapsed = time.time() - eval_t0

    # ---------------- Phase 3: aggregation ----------------
    if args.no_aggregate:
        log_block("Phase 3: aggregation — skipped (--no-aggregate)")
    else:
        log_block("Phase 3: aggregation")
        written = build_summary_csvs(output_root, args.baselines, triples)
        for k, pth in written.items():
            print(f"  → {k}: {pth}")
        # Pretty-print the mean table to stdout
        try:
            rows = list(csv.reader(open(written["summary_mean"])))
            print("\n  === summary mean (rows = metric, cols = baseline) ===")
            col_w = [max(len(r[c]) for r in rows) for c in range(len(rows[0]))]
            for r in rows:
                print("    " + "  ".join(c.ljust(w) for c, w in zip(r, col_w)))
        except Exception:
            pass

    elapsed = time.time() - overall_t0
    print("\n" + "=" * 70)
    print("  FINAL")
    print("=" * 70)
    print(f"  triples           : {len(records)}")
    print(f"  baselines         : {args.baselines}")
    print(f"  total elapsed      : {elapsed:.1f}s  (eval sub-phase = {eval_elapsed:.1f}s)")
    n_done = sum(1 for r in records if r.jsons)
    n_fail = sum(1 for r in records if r.failed_baselines)
    print(f"  ok generations    : {sum(len(r.fakes) for r in records)} "
          f"/ {len(records) * len(args.baselines)}")
    print(f"  ok evaluations    : {sum(len(r.jsons) for r in records)} "
          f"/ {sum(len(r.fakes) for r in records)}")
    print(f"  output root        : {output_root}")

    if not args.keep_workdir:
        # Clean up the per-pair _work dirs created by unified_evaluator
        cleanup_count = 0
        for method in args.baselines:
            wd_root = output_root / method / "_work"
            if wd_root.is_dir():
                shutil.rmtree(wd_root, ignore_errors=True)
                cleanup_count += 1
        if cleanup_count:
            print(f"  cleaned _work/ dirs for {cleanup_count} baseline(s) "
                  f"(pass --keep-workdir to keep)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
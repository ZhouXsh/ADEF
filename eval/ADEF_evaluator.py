#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADEF_evaluator.py
=================

Per-``exam_name`` batch driver around ``unified_evaluator.py``.

Given a single ``exam_name`` (the output directory used by
``run_alone.py``'s ``inT()`` — every ``exam_name`` writes a fresh batch
of fake videos into ``<FATHER>/<exam_name>/``), this script:

1. Reads ``my_double.txt`` — one ``fake_filename,gt_fullpath`` pair per
   line.  ``fake_filename`` is **relative** to
   ``<FATHER>/<exam_name>/``; ``gt_fullpath`` is an **absolute** path.
2. For every pair, builds the full paths and invokes
   ``unified_evaluator.py`` as a subprocess.  The per-pair JSON report
   is written to ``<EVAL>/eval_reports/<exam_name>/<pair_name>.json``
   (under the eval tree, not under the visual tree).
3. After every pair has been evaluated, walks the per-pair JSONs and
   aggregates each headline metric's value across all videos (mean &
   variance).
4. **Incrementally** writes a one-row-per-metric CSV at
   ``<FATHER>/summary.csv`` with ``exam_name`` on the horizontal axis.
   On a fresh file the script writes headers + the new ``exam_name``
   mean/var columns.  On subsequent runs the script appends two new
   columns (``<exam_name>_mean``, ``<exam_name>_var``) to the right of
   the existing ones — existing exam rows are left untouched.  If the
   same ``exam_name`` is re-run, its previous columns are replaced in
   place (idempotent overwrite).

Usage
-----
::

    python ADEF_evaluator.py <exam_name>

    # or with options:
    python ADEF_evaluator.py <exam_name> \
        --pairs-file /path/to/my_double.txt \
        --father-dir /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake \
        --metrics lse fid eat emonet \
        --device cuda:0 \
        --eat-device 0 \
        --timeout 1800 \
        --resume        # skip pairs whose JSON already exists

Output
------
* Per-pair JSON reports under
  ``<EVAL>/eval_reports/<exam_name>/<pair_name>.json``.
* ``<FATHER>/summary.csv`` — one row per headline metric, one pair of
  columns (mean, var) per exam name.

CSV layout
~~~~~~~~~~

``exam_name`` is on the **vertical** axis (rows), metric on the
**horizontal** axis (columns).  Every new ``exam_name`` adds **two
rows** at the bottom — one for the mean, one for the variance.

::

    exam_name,stat,LSE-D,LSE-C,FVD,FID,PSNR,SSIM,LPIPS,M-LMD,F-LMD,Sync-Conf,Emo-Acc,EmoNet-Acc,EmoNet-Sim,EmotiEff-DomFrac,DFER-CLIP-Correct,NewEmo-Agreement
    examA,mean,12.5035,1.5351,...
    examA,var,1.6328,1.4140,...
    examB,mean,13.9068,0.8467,...
    examB,var,0.3492,0.1060,...
    ...

If the same ``exam_name`` is re-run, its previous two rows are
**replaced in place** (no row duplication; other exam rows are left
untouched).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent                          # .../ADEF_remake/eval
UNIFIED_EVAL = EVAL_ROOT / "unified_evaluator.py"
DEFAULT_PAIRS_FILE = EVAL_ROOT / "my_double.txt"

DEFAULT_FATHER_DIR = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY_CSV = DEFAULT_FATHER_DIR / "summary.csv"
# Per-pair JSON reports default to sitting **inside the exam_name
# directory** alongside the fake videos (so each ``<pair_name>.mp4`` has
# a sibling ``<pair_name>.json``).  Set ``--reports-dirname`` to override.
DEFAULT_REPORTS_DIRNAME = "."

# ---------------------------------------------------------------------------
# Headline metrics — same definition as in ``final_evaluator.py``.  These are
# the (metric_group, dotted_payload_path, csv_row_label) triples that drive
# both the per-pair JSON parsing and the summary CSV rows.
# ---------------------------------------------------------------------------
HEADLINE_METRICS: List[Tuple[str, str, str]] = [
    # Audio-visual sync
    ("lse",      "lse_d",                "LSE-D"),
    ("lse",      "lse_c",                "LSE-C"),
    # Distribution distances
    ("fvd",      "fvd",                  "FVD"),
    ("fid",      "fid",                  "FID"),
    # Pixel / landmark (EAT pipeline)
    ("eat",      "psnr_ssim.psnr",       "PSNR"),
    ("eat",      "psnr_ssim.ssim",       "SSIM"),
    ("eat",      "lpips.mean_lpips",     "LPIPS"),
    ("eat",      "lmd.mouth_lmd",        "M-LMD"),
    ("eat",      "lmd.face_lmd",         "F-LMD"),
    ("eat",      "sync.sync_conf",       "Sync-Conf"),
    ("eat",      "emo.emo_acc",          "Emo-Acc"),
    # EmoNet
    ("emonet",   "emo_acc",              "EmoNet-Acc"),
    ("emonet",   "emo_sim",              "EmoNet-Sim"),
    # Reference-free emotion models
    ("emotiefflib", "dominant_fraction", "EmotiEff-DomFrac"),
    ("dfer_clip",   "correct",           "DFER-CLIP-Correct"),
    ("new_emo",     "agreement",         "NewEmo-Agreement"),
]

# Ordered list of row labels — used to keep the summary CSV deterministic.
ROW_LABELS: List[str] = [csv_col for _grp, _leaf, csv_col in HEADLINE_METRICS]


# ============================================================
# Logging helpers
# ============================================================
def log_block(title: str) -> None:
    print(f'\n{"=" * 70}\n  {title}\n{"=" * 70}', flush=True)


def log_sub(title: str) -> None:
    print(f'\n--- {title} ---', flush=True)


# ============================================================
# Pair-list parsing
# ============================================================
def read_pairs_file(p: Path) -> List[Dict[str, str]]:
    """Read ``fake_filename,gt_fullpath`` lines from a text file.

    The fake filename is *relative* to ``<father>/<exam_name>/`` (the
    caller will join it later).  The GT path is *absolute* and used
    verbatim.

    Returns a list of dicts ``{"fake", "gt", "name"}`` where ``name``
    is the fake filename's stem (used for the per-pair JSON file).
    """
    out: List[Dict[str, str]] = []
    for lineno, raw in enumerate(p.read_text().splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # ``maxsplit=1`` so a stray comma inside a filename doesn't
        # silently break the GT lookup (we still emit a warning though).
        parts = [x.strip() for x in s.split(",", maxsplit=1)]
        if len(parts) != 2:
            print(f"[pairs] WARN: line {lineno} 跳过（需要 fake,gt 两段）: {s!r}")
            continue
        fake, gt = parts
        out.append({
            "fake": fake,
            "gt":   gt,
            "name": Path(fake).stem,
        })
    return out


# ============================================================
# unified_evaluator subprocess driver
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
    keep_workdir: bool = False,
) -> int:
    """Call ``unified_evaluator.py`` as a subprocess; return its return code.

    Stdout/stderr are **not** captured to disk — ``unified_evaluator.py``
    already prints concise status lines, and the per-pair JSON report
    carries every metric's structured payload.  Pass ``--quiet`` (which
    we do) so the parent log stays clean.

    The intermediate ``workdir`` (which holds each metric's
    per-video JSON dump) is deleted on exit unless ``keep_workdir`` is
    set — the consolidated per-pair ``out_json`` already contains
    everything needed downstream.
    """
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if workdir is None:
        workdir = out_json.parent / "_work"
    workdir.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [
        sys.executable, str(UNIFIED_EVAL),
        "--fake",   str(fake),
        "--gt",     str(gt),
        "--name",   name,
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

    rc = 0
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(EVAL_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124
    finally:
        if not keep_workdir and workdir.is_dir():
            shutil.rmtree(workdir, ignore_errors=True)

    return rc


# ============================================================
# Per-pair JSON → headline-metric values
# ============================================================
def _safe_num(x: Any) -> Optional[float]:
    """Coerce a JSON leaf to ``float``; return ``None`` if not numeric."""
    if isinstance(x, bool):
        # Treat booleans as 0/1 so categorical fields like
        # ``agreement`` / ``correct`` aggregate as fractions.
        return float(int(x))
    if isinstance(x, (int, float)):
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    return None


def _dotted_get(d: Any, dotted_key: str) -> Optional[float]:
    cur: Any = d
    for seg in dotted_key.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return _safe_num(cur)


# Registry of metric "names" accepted by ``--metrics`` — mirrors
# ``unified_evaluator.METRIC_REGISTRY``.  Kept inline so this module
# has no hard import dependency on unified_evaluator.
METRIC_KEYS = [
    "lse", "fvd", "fid", "eat", "emonet",
    "emo_fan", "emotiefflib", "dfer_clip", "new_emo",
]


def cached_report_has_metrics(
    report: Dict[str, Any],
    metrics: Sequence[str],
    skip: Sequence[str],
) -> bool:
    """Return True iff a cached ``unified_evaluator`` report covers every
    requested metric (after applying ``skip``).

    Used by the ``--resume`` path: blindly trusting an old report when
    the requested ``--metrics`` subset has changed would silently drop
    values.  We treat "covered" as ``metrics[name].ok == True`` for
    every name in the resolved metric list.
    """
    if not metrics or list(metrics) == ["all"]:
        requested = list(METRIC_KEYS)
    else:
        requested = [m for m in metrics if m in METRIC_KEYS]
    skip_set = {m for m in skip if m in METRIC_KEYS}
    requested = [m for m in requested if m not in skip_set]
    if not requested:
        return True
    metrics_block = report.get("metrics") or {}
    for name in requested:
        bucket = metrics_block.get(name) or {}
        if not bucket.get("ok"):
            return False
    return True


def extract_metric_values(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Given a single per-pair unified_evaluator JSON report, return
    ``{csv_row_label: numeric_value_or_None}`` for every headline
    metric.  A missing / non-ok metric becomes ``None`` (which the
    aggregator drops from mean/var)."""
    out: Dict[str, Optional[float]] = {label: None for _g, _l, label in HEADLINE_METRICS}
    metrics_block = report.get("metrics") or {}
    for metric_group, leaf_key, csv_col in HEADLINE_METRICS:
        bucket = metrics_block.get(metric_group) or {}
        if not bucket.get("ok"):
            continue
        payload = bucket.get("payload") or {}
        out[csv_col] = _dotted_get(payload, leaf_key)
    return out


# ============================================================
# Aggregation
# ============================================================
def aggregate_values(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """Compute (mean, population_variance, n_valid) over a list of values.

    Returns ``(None, None, 0)`` if no value is numeric.
    Population variance (i.e. ``statistics.pvariance``) is used so a
    single sample yields ``var=0`` rather than NaN.
    """
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None, 0
    if len(nums) == 1:
        var = 0.0
    else:
        var = statistics.pvariance(nums)
    return float(statistics.mean(nums)), float(var), len(nums)


# ============================================================
# Incremental summary CSV write
# ============================================================
def _read_existing_summary(summary_csv: Path) -> Tuple[List[str], Dict[Tuple[str, str], List[str]]]:
    """Return ``(header_row, {(exam_name, stat): [metric_values]})``
    from an existing summary CSV (axes-swapped layout).

    The header is the first non-empty line; rows are indexed by
    ``(exam_name, stat)`` where ``stat`` is ``"mean"`` or ``"var"``.
    Unknown keys yield ``[]`` and are padded by the writer.
    """
    if not summary_csv.is_file():
        return [], {}
    header: List[str] = []
    rows: Dict[Tuple[str, str], List[str]] = {}
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if not row:
                continue
            if i == 0:
                header = row
                continue
            if len(row) < 3:
                # Malformed row — skip silently.
                continue
            exam, stat = row[0], row[1]
            rows[(exam, stat)] = row[2:]
    return header, rows


def update_summary_csv(
    summary_csv: Path,
    exam_name: str,
    metric_means: Dict[str, Optional[float]],
    metric_vars: Dict[str, Optional[float]],
) -> Dict[str, Path]:
    """Write the summary CSV with a fresh two-row block for ``exam_name``.

    Layout (axes swapped — ``exam_name`` is on the vertical axis)::

        exam_name,stat,LSE-D,LSE-C,FVD,FID,PSNR,SSIM,LPIPS,M-LMD,F-LMD,Sync-Conf,Emo-Acc,EmoNet-Acc,EmoNet-Sim,EmotiEff-DomFrac,DFER-CLIP-Correct,NewEmo-Agreement
        examA,mean,12.5035,1.5351,...
        examA,var,1.6328,1.4140,...
        examB,mean,13.9068,0.8467,...
        examB,var,0.3492,0.1060,...
        ...

    Behaviour:
      * If ``summary_csv`` doesn't exist, write the header + the two
        new rows for ``exam_name``.
      * If it exists, append two rows (mean / var) at the bottom.
      * If the same ``exam_name`` is already present, its previous two
        rows are **replaced in place** (rows for other exams are
        preserved verbatim).  No duplicate rows are produced.

    Returns ``{"summary_csv": <path>}`` on success.
    """
    header, existing_rows = _read_existing_summary(summary_csv)

    # ----- 1. Build the new header ----------------------------------
    # New metric columns = every metric we know about + any legacy
    # metrics that appear in the existing CSV (so unknown columns are
    # preserved as best-effort empty cells).
    new_metrics: List[str] = list(ROW_LABELS)
    if header:
        for h in header[2:]:                  # skip leading "exam_name","stat"
            if h not in new_metrics:
                new_metrics.append(h)

    if header:
        # Keep the existing header verbatim if it's already correct,
        # otherwise replace it with our canonical metric order.
        canonical_header = ["exam_name", "stat", *new_metrics]
        if header[2:] == new_metrics:
            new_header = list(header)
        else:
            new_header = canonical_header
    else:
        new_header = ["exam_name", "stat", *new_metrics]

    metric_count = len(new_metrics)

    # ----- 2. Drop any existing rows for this exam_name --------------
    # (idempotent re-run — replace, don't duplicate)
    preserved_rows: List[Tuple[str, str, List[str]]] = []
    for (exam, stat), values in existing_rows.items():
        if exam == exam_name:
            continue
        # Pad / truncate previous values to the current metric_count.
        if len(values) < metric_count:
            values = values + [""] * (metric_count - len(values))
        elif len(values) > metric_count:
            values = values[:metric_count]
        preserved_rows.append((exam, stat, values))

    # ----- 3. Build the new (exam_name, mean) and (exam_name, var) rows
    def _row_for(stat: str, src: Dict[str, Optional[float]]) -> List[str]:
        out = [exam_name, stat]
        for m in new_metrics:
            v = src.get(m)
            out.append(f"{v:.4f}" if v is not None else "")
        return out

    new_mean_row = _row_for("mean", metric_means)
    new_var_row  = _row_for("var",  metric_vars)

    # ----- 4. Write the CSV ------------------------------------------
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_header)
        for exam, stat, values in preserved_rows:
            writer.writerow([exam, stat, *values])
        writer.writerow(new_mean_row)
        writer.writerow(new_var_row)

    return {"summary_csv": summary_csv}


# ============================================================
# Phase orchestration
# ============================================================
def evaluate_exam(
    exam_name: str,
    father_dir: Path,
    pairs: List[Dict[str, str]],
    metrics: Sequence[str],
    skip: Sequence[str],
    device: str,
    eat_device: str,
    timeout: int,
    reports_dir: Path,
    resume: bool,
    summary_csv: Path,
) -> int:
    """End-to-end driver: evaluate every pair, aggregate, append to CSV.

    Returns the script's process exit code (0 = success).
    """
    outdir = father_dir / exam_name     # where the fake videos live
    log_block(f"ADEF evaluator — exam: {exam_name}")
    print(f"[ADEF_eval] fake-video dir  : {outdir}")
    print(f"[ADEF_eval] pairs file      : ({len(pairs)} pairs)")
    print(f"[ADEF_eval] per-pair reports: {reports_dir}")
    print(f"[ADEF_eval] summary CSV     : {summary_csv}")
    print(f"[ADEF_eval] metrics         : {list(metrics) if metrics else ['all']}")
    if skip:
        print(f"[ADEF_eval] skip            : {list(skip)}")

    if not outdir.is_dir():
        print(f"[ADEF_eval] ERROR: fake-video dir not found: {outdir}")
        return 2

    reports_dir.mkdir(parents=True, exist_ok=True)

    # metric -> per-pair list of values (None = failed/skipped)
    metric_values: Dict[str, List[Optional[float]]] = {
        label: [] for _g, _l, label in HEADLINE_METRICS
    }

    pair_rcs: List[int] = []
    for i, pair in enumerate(pairs):
        fake_path = outdir / pair["fake"]
        gt_path   = Path(pair["gt"])
        name      = pair["name"]

        out_json = reports_dir / f"{name}.json"
        workdir  = reports_dir / "_work" / name

        log_sub(f"[{i+1}/{len(pairs)}] {name}")

        if not fake_path.is_file():
            print(f"  SKIP — fake not found: {fake_path}")
            for label in metric_values:
                metric_values[label].append(None)
            pair_rcs.append(1)
            continue
        if not gt_path.is_file():
            print(f"  SKIP — gt not found:   {gt_path}")
            for label in metric_values:
                metric_values[label].append(None)
            pair_rcs.append(1)
            continue

        if resume and out_json.is_file():
            # Validate the cached report covers the requested metrics.
            # If not (e.g. the previous run used a smaller ``--metrics``
            # subset, or a metric failed), force a fresh re-evaluation
            # — otherwise the missing metrics silently disappear.
            try:
                cached_report = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                cached_report = None
            if cached_report and cached_report_has_metrics(
                cached_report, metrics, skip,
            ):
                print(f"  RESUME — using cached report: {out_json}")
                rc = 0
            else:
                print(
                    f"  RESUME — cached report missing requested metrics, "
                    f"re-running: {out_json}"
                )
                cached_report = None
                rc = None  # fall through to the fresh evaluation below
        else:
            cached_report = None
            rc = None

        if cached_report is None and rc is None:
            print(f"  fake = {fake_path}")
            print(f"  gt   = {gt_path}")
            t0 = time.time()
            rc = evaluate_one(
                fake=fake_path,
                gt=gt_path,
                out_json=out_json,
                name=name,
                metrics=metrics,
                skip=skip,
                device=device,
                eat_device=eat_device,
                timeout=timeout,
                workdir=workdir,
            )
            elapsed = time.time() - t0
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"
            print(f"  -> {status} ({elapsed:.1f}s)  json={out_json}")

        pair_rcs.append(rc)
        if rc != 0 or not out_json.is_file():
            for label in metric_values:
                metric_values[label].append(None)
            continue

        # ---- Parse the per-pair JSON -------------------------------
        try:
            report = json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as exc:                            # noqa: BLE001
            print(f"  WARN: failed to parse {out_json}: {exc!r}")
            for label in metric_values:
                metric_values[label].append(None)
            continue

        extracted = extract_metric_values(report)
        for label in metric_values:
            metric_values[label].append(extracted.get(label))

    # ---- Aggregate -------------------------------------------------
    log_block(f"Aggregation — exam: {exam_name}")
    metric_means: Dict[str, Optional[float]] = {}
    metric_vars:  Dict[str, Optional[float]] = {}
    for label in ROW_LABELS:
        mean_v, var_v, n = aggregate_values(metric_values[label])
        metric_means[label] = mean_v
        metric_vars[label]  = var_v
        if mean_v is None:
            print(f"  {label:<18s}  (no valid values)")
        else:
            print(f"  {label:<18s}  mean={mean_v:9.4f}  var={var_v:9.4f}  n={n}")

    # ---- Incremental summary CSV ----------------------------------
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    update_summary_csv(summary_csv, exam_name, metric_means, metric_vars)
    print(f"\n[ADEF_eval] summary CSV updated -> {summary_csv}")

    # ---- Exit -------------------------------------------------------
    failed_pairs = sum(1 for rc in pair_rcs if rc != 0)
    if failed_pairs:
        print(f"\n[ADEF_eval] {failed_pairs}/{len(pairs)} pair(s) failed")
        # Still return 0 — partial results are useful; surface non-zero
        # only if the caller wants strict gating.
        return 0
    return 0


# ============================================================
# CLI
# ============================================================
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-exam_name batch driver for ADEF unified evaluation. "
                    "Reads fake/gt pairs from my_double.txt, runs "
                    "unified_evaluator.py on each, then incrementally "
                    "appends exam_name's mean/var to a summary CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("exam_name",
                   help="Output directory name (the same exam_name passed "
                        "to run_alone.py).")
    p.add_argument("--pairs-file", default=str(DEFAULT_PAIRS_FILE),
                   help="Path to the fake/gt pair list (one "
                        "'fake_filename,gt_fullpath' per line).")
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER_DIR),
                   help="Root directory under which <exam_name>/ holds "
                        "the fake videos.")
    p.add_argument("--reports-dirname", default=DEFAULT_REPORTS_DIRNAME,
                   help="Where per-pair JSON reports are written. "
                        "Default '.' places them inside <father>/<exam_name>/ "
                        "next to each fake mp4. Any other value puts them "
                        "under <EVAL>/<reports_dirname>/<exam_name>/.")
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV),
                   help="Path to the summary CSV (incremental).")
    p.add_argument("--metrics", nargs="+", default=["all"],
                   help="Subset of metrics to run, space-separated "
                        "(passed to unified_evaluator.py).")
    p.add_argument("--skip", nargs="+", default=[],
                   help="Metrics to skip (passed to unified_evaluator.py).")
    p.add_argument("--device", default="cuda:0",
                   help="CUDA device for EmoNet / Emotion-FAN / New_Emo.")
    p.add_argument("--eat-device", default="0",
                   help="CUDA device index for the EAT pipeline.")
    p.add_argument("--timeout", type=int, default=1800,
                   help="Per-pair subprocess timeout (seconds).")
    p.add_argument("--resume", action="store_true",
                   help="Skip pairs whose per-pair JSON already exists.")
    args = p.parse_args(argv)

    pairs_file = Path(args.pairs_file)
    if not pairs_file.is_file():
        print(f"[ADEF_eval] ERROR: pairs file not found: {pairs_file}")
        return 2

    pairs = read_pairs_file(pairs_file)
    if not pairs:
        print(f"[ADEF_eval] ERROR: no usable pairs in {pairs_file}")
        return 2

    # Resolve the per-pair JSON directory.  The default (``.``) puts
    # the JSONs inside ``<father>/<exam_name>/`` next to each fake mp4.
    # Any other value puts them under ``<EVAL>/<reports_dirname>/<exam_name>/``
    # (useful when you want to keep eval artefacts out of the visual tree).
    rdn = args.reports_dirname
    if rdn in (".", "", None):
        reports_dir = Path(args.father_dir) / args.exam_name
    else:
        reports_dir = EVAL_ROOT / rdn / args.exam_name

    try:
        return evaluate_exam(
            exam_name=args.exam_name,
            father_dir=Path(args.father_dir),
            pairs=pairs,
            metrics=args.metrics,
            skip=args.skip,
            device=args.device,
            eat_device=args.eat_device,
            timeout=args.timeout,
            reports_dir=reports_dir,
            resume=args.resume,
            summary_csv=Path(args.summary_csv),
        )
    except Exception:                                       # noqa: BLE001
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_summary.py
==================

Post-process the per-pair JSON reports produced by
``unified_evaluator.py`` and rebuild ``summary.csv`` from scratch.

Why this exists
---------------
``ADEF_evaluator.py`` increments ``summary.csv`` after each exam, but if
its aggregation step is wrong (or was never run for most exams) the CSV
ends up with rows of empty cells.  This script ignores the existing
``summary.csv`` and recomputes mean/var for every exam by walking the
per-pair JSONs that already live inside ``<father>/<exam>/``.

For each immediate subdirectory of ``--father-dir``:

* load every ``*.json`` in it,
* extract the same headline metrics that ``ADEF_evaluator.HEADLINE_METRICS``
  defines (using the same dotted payload paths),
* drop metrics whose source is ``ok=False`` or whose leaf value is
  non-numeric (matches ``ADEF_evaluator._safe_num`` behaviour),
* aggregate the surviving values with ``statistics.mean`` /
  ``statistics.pvariance`` (population variance, so a single sample
  yields ``var=0`` instead of NaN — same as ADEF_evaluator).

It then writes a clean ``summary.csv`` with the same header layout as
``ADEF_evaluator.update_summary_csv``:

::

    exam_name,stat,LSE-D,LSE-C,FVD,FID,PSNR,SSIM,LPIPS,M-LMD,F-LMD,Sync-Conf,Emo-Acc,EmoNet-Acc,EmoNet-Sim,EmotiEff-DomFrac,DFER-CLIP-Correct,NewEmo-Agreement
    examA,mean,12.5035,...
    examA,var,1.6328,...
    examB,mean,...

Usage
-----
::

    # defaults — father dir is the ADEFv4 visual tree
    python rebuild_summary.py

    # only print, don't write the CSV
    python rebuild_summary.py --dry-run

    # write somewhere else
    python rebuild_summary.py --summary-csv /tmp/test_summary.csv

    # only one exam (debugging)
    python rebuild_summary.py --only 20260616_emotion_dit_chatgpt

    # skip the per-pair JSONs that don't match the schema (default true)
    python rebuild_summary.py --no-skip-malformed

    # also include exams that have zero JSONs (rendered as empty rows)
    python rebuild_summary.py --include-empty
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent
DEFAULT_FATHER_DIR = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY_CSV = DEFAULT_FATHER_DIR / "summary.csv"

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:                                          # noqa: BLE001
    _HAS_TQDM = False


# ============================================================
# Headline metrics — kept in lock-step with ADEF_evaluator.py
# (metric_group, dotted_payload_path, csv_row_label)
# ============================================================
HEADLINE_METRICS: List[Tuple[str, str, str]] = [
    ("lse",         "lse_d",                "LSE-D"),
    ("lse",         "lse_c",                "LSE-C"),
    ("fvd",         "fvd",                  "FVD"),
    ("fid",         "fid",                  "FID"),
    ("eat",         "psnr_ssim.psnr",       "PSNR"),
    ("eat",         "psnr_ssim.ssim",       "SSIM"),
    ("eat",         "lpips.mean_lpips",     "LPIPS"),
    ("eat",         "lmd.mouth_lmd",        "M-LMD"),
    ("eat",         "lmd.face_lmd",         "F-LMD"),
    ("eat",         "sync.sync_conf",       "Sync-Conf"),
    ("eat",         "emo.emo_acc",          "Emo-Acc"),
    ("emonet",      "emo_acc",              "EmoNet-Acc"),
    ("emonet",      "emo_sim",              "EmoNet-Sim"),
    ("emotiefflib", "dominant_fraction",    "EmotiEff-DomFrac"),
    ("dfer_clip",   "correct",              "DFER-CLIP-Correct"),
    ("new_emo",     "agreement",            "NewEmo-Agreement"),
]
ROW_LABELS: List[str] = [csv_col for _g, _l, csv_col in HEADLINE_METRICS]
HEADER: List[str] = ["exam_name", "stat", *ROW_LABELS]


# ============================================================
# JSON helpers (mirrors ADEF_evaluator._safe_num / _dotted_get)
# ============================================================
def _safe_num(x: Any) -> Optional[float]:
    """Coerce a JSON leaf to float; return None if not numeric."""
    if isinstance(x, bool):
        # Booleans become 0/1 so categorical fields like
        # ``agreement`` / ``correct`` aggregate as fractions.
        return float(int(x))
    if isinstance(x, (int, float)):
        v = float(x)
        if v != v:                                        # NaN
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


# ============================================================
# Per-JSON validation + extraction
# ============================================================
def looks_like_pair_report(j: Any) -> bool:
    """True iff ``j`` has the unified_evaluator per-pair schema."""
    if not isinstance(j, dict):
        return False
    if not isinstance(j.get("metrics"), dict):
        return False
    # At least one metric group should be a dict with ``ok`` key.
    for v in j["metrics"].values():
        if isinstance(v, dict) and "ok" in v:
            return True
    return False


def extract_values(j: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Return ``{csv_label: float or None}`` for every headline metric."""
    out: Dict[str, Optional[float]] = {label: None for _g, _l, label in HEADLINE_METRICS}
    metrics_block = j.get("metrics") or {}
    for group, leaf, csv_col in HEADLINE_METRICS:
        bucket = metrics_block.get(group) or {}
        if not isinstance(bucket, dict) or not bucket.get("ok"):
            continue
        payload = bucket.get("payload") or {}
        out[csv_col] = _dotted_get(payload, leaf)
    return out


# ============================================================
# Aggregation
# ============================================================
def aggregate(values: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    """(mean, population_var, n_valid) — mirrors ADEF_evaluator.aggregate_values."""
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None, 0
    if len(nums) == 1:
        var = 0.0
    else:
        var = statistics.pvariance(nums)
    return float(statistics.mean(nums)), float(var), len(nums)


# ============================================================
# Discovery + per-exam processing
# ============================================================
def discover_exams(father_dir: Path) -> List[Path]:
    """Return immediate-subdirectory paths under ``father_dir``."""
    if not father_dir.is_dir():
        return []
    return sorted(p for p in father_dir.iterdir() if p.is_dir())


def process_exam(exam_dir: Path, skip_malformed: bool) -> Tuple[
    Dict[str, Optional[float]],                          # means
    Dict[str, Optional[float]],                          # vars
    int,                                                  # n_jsons_kept
    int,                                                  # n_jsons_skipped
]:
    metric_values: Dict[str, List[Optional[float]]] = {
        label: [] for _g, _l, label in HEADLINE_METRICS
    }
    n_kept = 0
    n_skipped = 0
    for jp in sorted(exam_dir.glob("*.json")):
        try:
            j = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:                                 # noqa: BLE001
            n_skipped += 1
            continue
        if not looks_like_pair_report(j):
            if skip_malformed:
                n_skipped += 1
                continue
            # When malformed files are kept, treat every metric as None
            # (so the exam's row reflects "0 valid pairs" rather than
            # silently using 0.0 from a default value).
            extracted = {label: None for _g, _l, label in HEADLINE_METRICS}
        else:
            extracted = extract_values(j)
            n_kept += 1
        for label, v in extracted.items():
            metric_values[label].append(v)

    means: Dict[str, Optional[float]] = {}
    vars_: Dict[str, Optional[float]] = {}
    for label in ROW_LABELS:
        m, v, _ = aggregate(metric_values[label])
        means[label] = m
        vars_[label] = v
    return means, vars_, n_kept, n_skipped


# ============================================================
# CSV writing
# ============================================================
def write_summary_csv(
    summary_csv: Path,
    rows: Sequence[Tuple[str, Dict[str, Optional[float]], Dict[str, Optional[float]]]],
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for exam_name, means, vars_ in rows:
            w.writerow(_fmt_row(exam_name, "mean", means))
            w.writerow(_fmt_row(exam_name, "var",  vars_))


def _fmt_row(exam_name: str, stat: str,
             values: Dict[str, Optional[float]]) -> List[str]:
    row = [exam_name, stat]
    for label in ROW_LABELS:
        v = values.get(label)
        row.append(f"{v:.4f}" if v is not None else "")
    return row


# ============================================================
# Pretty-print (for --dry-run / stdout)
# ============================================================
def _fmt_cell(v: Optional[float]) -> str:
    if v is None:
        return "    -    "
    return f"{v:9.4f}"


def print_table(rows: Sequence[Tuple[str, Dict[str, Optional[float]], Dict[str, Optional[float]]]]) -> None:
    """Print a compact, human-readable table to stdout."""
    label_w = max(len(l) for l in ROW_LABELS)
    for exam_name, means, vars_ in rows:
        print(f"\n--- {exam_name} ---")
        for label in ROW_LABELS:
            m = means.get(label)
            v = vars_.get(label)
            if m is None and v is None:
                print(f"  {label:<{label_w}}  (no valid values)")
            else:
                print(f"  {label:<{label_w}}  mean={_fmt_cell(m)}  var={_fmt_cell(v)}")


# ============================================================
# CLI
# ============================================================
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Recompute summary.csv from per-pair JSON reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER_DIR),
                   help="Root directory whose immediate subdirectories are exams.")
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV),
                   help="Output CSV path (overwritten).")
    p.add_argument("--dry-run", action="store_true",
                   help="Process every exam and print a table, but don't "
                        "write summary.csv.")
    p.add_argument("--include-empty", action="store_true",
                   help="Include exams with zero valid JSONs as empty rows "
                        "(default: omit them).")
    p.add_argument("--no-skip-malformed", dest="skip_malformed",
                   action="store_false",
                   help="Keep malformed JSONs (treated as all-None values) "
                        "instead of skipping them.")
    p.add_argument("--only", action="append", default=None,
                   help="Only process the listed exam name(s). May be repeated.")
    p.set_defaults(skip_malformed=True)
    args = p.parse_args(argv)

    father_dir = Path(args.father_dir)
    summary_csv = Path(args.summary_csv)

    if not father_dir.is_dir():
        print(f"[rebuild] ERROR: father dir not found: {father_dir}",
              file=sys.stderr)
        return 2

    all_exams = discover_exams(father_dir)
    if args.only:
        wanted = set(args.only)
        exams = [d for d in all_exams if d.name in wanted]
        missing = wanted - {d.name for d in exams}
        if missing:
            print(f"[rebuild] WARN: --only names not found: {sorted(missing)}",
                  file=sys.stderr)
    else:
        exams = all_exams

    print(f"[rebuild] father_dir : {father_dir}")
    print(f"[rebuild] summary_csv: {summary_csv}")
    print(f"[rebuild] tqdm       : {'yes' if _HAS_TQDM else 'no (no bar)'}")
    print(f"[rebuild] exams found: {len(exams)}")

    rows: List[Tuple[str, Dict[str, Optional[float]], Dict[str, Optional[float]]]] = []
    iterator = tqdm(exams, desc="rebuild", unit="exam",
                    dynamic_ncols=True) if _HAS_TQDM else exams
    for exam_dir in iterator:
        if _HAS_TQDM:
            iterator.set_postfix_str(exam_dir.name[:48])
        means, vars_, n_kept, n_skipped = process_exam(
            exam_dir, skip_malformed=args.skip_malformed,
        )
        if n_kept == 0 and not args.include_empty:
            continue
        rows.append((exam_dir.name, means, vars_))

    print(f"[rebuild] exams written: {len(rows)}")

    if args.dry_run:
        print_table(rows)
        print("\n[rebuild] dry-run: summary.csv not written.")
        return 0

    write_summary_csv(summary_csv, rows)
    print(f"[rebuild] wrote: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

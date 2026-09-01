#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADEF_all_evaluator.py
=====================

Batch driver: iterate over **every** subdirectory of the ADEF visual
tree and run ``ADEF_evaluator.py`` once per exam.  Exams already
present in ``summary.csv`` are skipped (idempotent re-run).

Usage
-----
::

    # process every subdir, skip ones already in summary.csv
    python ADEF_all_evaluator.py

    # common extras (forwarded verbatim to ADEF_evaluator.py)
    python ADEF_all_evaluator.py --resume --device cuda:0 --timeout 1800

    # debugging helpers
    python ADEF_all_evaluator.py --limit 3                # only first 3 to-do
    python ADEF_all_evaluator.py --include-done          # force re-run
    python ADEF_all_evaluator.py --dry-run               # print plan, don't run

A ``tqdm`` progress bar is shown when tqdm is importable; otherwise
the script falls back to a plain ``[i/N]`` line reporter.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent
ADEF_EVALUATOR = EVAL_ROOT / "ADEF_evaluator.py"
DEFAULT_FATHER_DIR = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY_CSV = DEFAULT_FATHER_DIR / "summary.csv"

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:                                          # noqa: BLE001
    _HAS_TQDM = False


# ============================================================
# Discovery
# ============================================================
def list_exams(father_dir: Path) -> List[str]:
    """Return sorted names of immediate subdirectories under ``father_dir``."""
    if not father_dir.is_dir():
        return []
    return sorted(p.name for p in father_dir.iterdir() if p.is_dir())


def read_done_exams(summary_csv: Path) -> set[str]:
    """Return the set of exam_name already written to ``summary_csv``.

    The CSV is laid out as ``exam_name,stat,metric,...`` so each exam
    occupies two rows (mean + var).  We collect column-0 across every
    data row; the set naturally deduplicates.
    """
    done: set[str] = set()
    if not summary_csv.is_file():
        return done
    try:
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)                              # skip header
            for row in reader:
                if row:
                    done.add(row[0])
    except Exception as exc:                                # noqa: BLE001
        print(f"[ADEF_all_eval] WARN: failed to parse {summary_csv}: {exc!r}",
              file=sys.stderr)
    return done


# ============================================================
# Progress bar (tqdm with graceful fallback)
# ============================================================
class _PlainBar:
    """Tiny drop-in replacement for ``tqdm`` when tqdm isn't installed."""

    def __init__(self, items: Sequence, desc: str, unit: str) -> None:
        self.items = list(items)
        self.desc = desc
        self.unit = unit
        self.n = len(self.items)
        print(f"[ADEF_all_eval] {desc} ({self.n} {unit})")

    def __iter__(self):
        for i, x in enumerate(self.items, 1):
            self._postfix = x
            yield x
            print(f"[ADEF_all_eval]   [{i}/{self.n}] {x} done")

    def set_postfix_str(self, s: str) -> None:
        self._postfix = s


def progress_iter(items: Sequence, desc: str, unit: str = "exam"):
    if _HAS_TQDM:
        return tqdm(items, desc=desc, unit=unit, dynamic_ncols=True)
    return _PlainBar(items, desc, unit)


# ============================================================
# Subprocess argv builder
# ============================================================
def build_forward_argv(exam_name: str, args, extra: Iterable[str]) -> List[str]:
    forward: List[str] = [sys.executable, str(ADEF_EVALUATOR), exam_name]
    if args.pairs_file is not None:
        forward += ["--pairs-file", args.pairs_file]
    if args.reports_dirname is not None:
        forward += ["--reports-dirname", args.reports_dirname]
    if args.metrics is not None:
        forward += ["--metrics", *args.metrics]
    if args.skip is not None:
        forward += ["--skip", *args.skip]
    if args.device is not None:
        forward += ["--device", args.device]
    if args.eat_device is not None:
        forward += ["--eat-device", args.eat_device]
    if args.timeout is not None:
        forward += ["--timeout", str(args.timeout)]
    if args.resume:
        forward += ["--resume"]
    forward += list(extra)
    return forward


# ============================================================
# CLI
# ============================================================
def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Run ADEF_evaluator.py over every exam directory under "
                    "--father-dir, skipping exams already in --summary-csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER_DIR),
                   help="Root directory whose subdirectories define the "
                        "exam set.")
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV),
                   help="CSV used as the source-of-truth for 'already done'.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N remaining exams (debugging).")
    p.add_argument("--include-done", action="store_true",
                   help="Re-run exams already present in summary.csv.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned invocations without executing.")
    # ---- passthrough args (forwarded verbatim to ADEF_evaluator.py) ----
    p.add_argument("--pairs-file", default=None)
    p.add_argument("--reports-dirname", default=None)
    p.add_argument("--metrics", nargs="+", default=None)
    p.add_argument("--skip", nargs="+", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--eat-device", default=None)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--resume", action="store_true")

    # Stop at the first unknown arg so we can collect extras.
    args, extra = p.parse_known_args(argv)

    father_dir = Path(args.father_dir)
    summary_csv = Path(args.summary_csv)

    if not ADEF_EVALUATOR.is_file():
        print(f"[ADEF_all_eval] ERROR: ADEF_evaluator.py not found at "
              f"{ADEF_EVALUATOR}", file=sys.stderr)
        return 3

    exams = list_exams(father_dir)
    if not exams:
        print(f"[ADEF_all_eval] ERROR: no subdirectories under {father_dir}",
              file=sys.stderr)
        return 2

    done = read_done_exams(summary_csv)
    if args.include_done:
        todo = list(exams)
    else:
        todo = [e for e in exams if e not in done]
    skipped = sorted(set(exams) - set(todo))
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]

    print(f"[ADEF_all_eval] father_dir : {father_dir}")
    print(f"[ADEF_all_eval] summary_csv: {summary_csv}")
    print(f"[ADEF_all_eval] tqdm       : {'yes' if _HAS_TQDM else 'no (fallback)'}")
    print(f"[ADEF_all_eval] total exams: {len(exams)}")
    print(f"[ADEF_all_eval] already done: {len(done)}")
    print(f"[ADEF_all_eval] to run      : {len(todo)}")
    if skipped:
        print(f"[ADEF_all_eval] skipping ({len(skipped)}):")
        for name in skipped:
            print(f"  - {name}")

    if not todo:
        print("[ADEF_all_eval] nothing to do.")
        return 0

    rc_total = 0
    bar = progress_iter(todo, desc="ADEF eval", unit="exam")
    for i, exam_name in enumerate(bar, 1):
        if _HAS_TQDM:
            bar.set_postfix_str(exam_name[:48])
        forward = build_forward_argv(exam_name, args, extra)
        print(f"\n[ADEF_all_eval] [{i}/{len(todo)}] {exam_name}")
        print(f"  $ {' '.join(forward)}")
        if args.dry_run:
            print("  -- dry-run: skipping execution")
            continue
        rc = subprocess.call(forward, cwd=str(EVAL_ROOT))
        status = "OK" if rc == 0 else f"FAIL (rc={rc})"
        print(f"  -> {status}")
        rc_total = rc_total or rc

    print(f"\n[ADEF_all_eval] done. {len(todo)} exam(s) processed.")
    return rc_total


if __name__ == "__main__":
    sys.exit(main())

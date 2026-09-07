#!/usr/bin/env python3
"""Helpers for the presentation-only paper table.

The paper CSV is intentionally kept free of execution metadata. Coverage,
status, protocol version and manifest fingerprints remain in paper_metrics.json
(and ADEF summary.csv where applicable).
"""
from __future__ import annotations

import csv
from pathlib import Path

PAPER_TABLE_COLUMNS = [
    "Method",
    "LSE-D", "LSE-C",
    "FID", "FVD",
    "PSNR", "SSIM", "LPIPS",
    "M-LMD", "F-LMD",
    "EmotiEff-Acc", "DFER-CLIP-Acc",
]


def trim_paper_table(path: str | Path) -> bool:
    """Rewrite an existing paper_table.csv with paper-facing columns only."""
    p = Path(path)
    if not p.is_file():
        return False
    with p.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPER_TABLE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return True

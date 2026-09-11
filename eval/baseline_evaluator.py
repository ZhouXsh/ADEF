#!/usr/bin/env python3
"""Evaluate one already-generated baseline with paper protocol v3.

This is the evaluation-only companion to ``baseline_generator.py``. It never
runs baseline inference. Instead, it resolves already generated fake videos
under ``RESULT/<method>/videos`` (preferring the manifest produced by the
generator), builds an evaluation manifest, propagates missing outputs as
upstream failures, and invokes the existing ``paper_evaluator.py``.

Run one process per baseline if desired. Each process writes only inside its
own ``RESULT/<method>/`` directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from final_evaluator import (  # noqa: E402
    BASELINE_METHODS,
    DEFAULT_RESULT_ROOT,
    _import_baselines,
    _pair_name,
    _read_method_status,
    _run_paper,
    _sample_name,
    read_triples,
)
from paper_protocol import Sample, read_manifest, write_manifest  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--method", required=True, choices=BASELINE_METHODS,
                   help="Evaluate exactly one already-generated baseline.")
    p.add_argument("--triples-file", required=True,
                   help="Text file with image,audio,gt_video[,emotion] per line.")
    p.add_argument("--output-root", default=str(DEFAULT_RESULT_ROOT),
                   help="Root directory containing <method>/videos/.")
    p.add_argument("--metrics", nargs="+", default=None,
                   choices=["lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=43200,
                   help="Per-metric subprocess timeout passed to paper_evaluator.py.")
    p.add_argument("--scenario", default="paper_eval",
                   help="Scenario used when reconstructing expected baseline filenames.")
    p.add_argument("--tmp-dir", default=None,
                   help="Optional large temporary directory exposed as TMPDIR to metric subprocesses.")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve generated videos and print coverage without running metrics.")
    return p.parse_args()


def _load_existing_manifest(path: Path) -> dict[str, Sample]:
    if not path.is_file():
        return {}
    try:
        return {s.name: s for s in read_manifest(path, require_files=False)}
    except Exception as exc:
        print(f"[baseline-eval] warning: cannot reuse {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return {}


def _load_generation_failures(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("failures", []) if isinstance(payload, dict) else []
    out: dict[str, dict] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = row
    return out


def _existing_fake_from_manifest(existing: dict[str, Sample], name: str) -> Path | None:
    sample = existing.get(name)
    if sample is None:
        return None
    path = Path(sample.fake)
    return path if path.is_file() else None


def _resolve_samples(args, indexed_triples, input_failures, method_dir: Path):
    videos_dir = method_dir / "videos"
    existing = _load_existing_manifest(method_dir / "manifest.csv")
    generation_failures = _load_generation_failures(method_dir / "generation_status.json")

    samples: list[Sample] = []
    failures: list[dict] = list(input_failures)

    # Import the existing baseline adapter only for filename reconstruction.
    # No inference function is called by this evaluator.
    rb = None

    for i, t in indexed_triples:
        name = _sample_name(t, i)
        fake = _existing_fake_from_manifest(existing, name)

        if fake is None:
            if rb is None:
                try:
                    rb = _import_baselines()
                except Exception as exc:
                    rb = exc
            if isinstance(rb, Exception):
                expected = None
            else:
                try:
                    expected_name = _pair_name(rb, args.method, t, i, args.scenario)
                    expected = videos_dir / f"{expected_name}.mp4"
                except Exception:
                    expected = None
            if expected is not None and expected.is_file():
                fake = expected

        if fake is None:
            prior = generation_failures.get(name)
            reason = None
            if prior:
                reason = prior.get("error")
            if not reason:
                reason = f"generated fake video not found under {videos_dir}"
            failures.append({
                "stage": "generation",
                "name": name,
                "fake": "",
                "gt": t.gt,
                "error": str(reason),
            })
            continue

        samples.append(Sample(
            name=name,
            fake=str(fake),
            gt=t.gt,
            emotion=t.emotion,
            image=t.image,
            audio=t.audio,
        ))

    return samples, failures


def _write_upstream(path: Path, failures: list[dict]) -> Path:
    path.write_text(json.dumps({"failures": failures}, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    indexed_triples, input_failures, expected_n = read_triples(args)

    root = Path(args.output_root).resolve()
    method_dir = root / args.method
    videos_dir = method_dir / "videos"
    if not method_dir.is_dir():
        print(f"ERROR: method result directory not found: {method_dir}", file=sys.stderr)
        return 2
    if not videos_dir.is_dir() and not (method_dir / "manifest.csv").is_file():
        print(f"ERROR: no generated videos/manifest found for {args.method}: {method_dir}", file=sys.stderr)
        return 2

    if args.tmp_dir:
        tmp_dir = Path(args.tmp_dir).expanduser().resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_dir)
        print(f"[baseline-eval] TMPDIR={tmp_dir}")

    samples, failures = _resolve_samples(
        args, indexed_triples, input_failures, method_dir
    )

    print(
        f"[baseline-eval] method={args.method} expected={expected_n} "
        f"resolved={len(samples)} missing_or_invalid={len(failures)}"
    )

    if args.dry_run:
        for s in samples:
            print(f"OK {s.name} -> {s.fake}")
        for f in failures:
            print(f"FAIL {f.get('name', '<unknown>')}: {f.get('error', '')}", file=sys.stderr)
        return 0 if samples else 2

    if not samples:
        print("ERROR: no generated samples are available for evaluation", file=sys.stderr)
        return 2

    manifest = write_manifest(method_dir / "evaluation_manifest.csv", samples)
    upstream = _write_upstream(method_dir / "evaluation_upstream_failures.json", failures)

    print(f"[baseline-eval] manifest: {manifest}")
    rc = _run_paper(manifest, args.method, method_dir, expected_n, upstream, args)

    status, n_fail = _read_method_status(method_dir)
    if status == "complete":
        print(f"[{args.method}] COMPLETE")
    elif status == "partial":
        print(f"[{args.method}] PARTIAL failures={n_fail}; table uses successful samples only",
              file=sys.stderr)
    elif status == "failed" or rc != 0:
        print(f"[{args.method}] FAILED failures={n_fail}", file=sys.stderr)

    table = method_dir / "paper_table.csv"
    if table.is_file():
        print(f"[baseline-eval] paper table: {table}")
    report = method_dir / "failed_samples.csv"
    if report.is_file() and n_fail:
        print(f"[baseline-eval] failed sample report: {report}", file=sys.stderr)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate baselines and evaluate them with the paper-grade ADEF protocol.

Input rows are ``image,audio,gt_video[,emotion]``.  Emotion is per-sample;
when omitted it is inferred from the MEAD GT path.  Every baseline is evaluated
on the same explicit manifest by ``paper_evaluator.py``.  FID/FVD are therefore
computed once per baseline dataset, never per video.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import MEAD_SHORT, Sample, canonical_emotion, infer_emotion, write_manifest  # noqa: E402

BASELINE_RUNNER = Path("/home/Zhouxishi/VirtualMan_proj/BASELINE/run_baselines.py")
PAPER_EVALUATOR = THIS_DIR / "paper_evaluator.py"
DEFAULT_RESULT_ROOT = THIS_DIR / "RESULT"
BASELINE_METHODS = ["wav2lip", "sadtalker", "eat_code", "joyvasa", "kdtalker"]
EAT_FULL = {
    "anger": "angry", "contempt": "contempt", "disgust": "disgusted",
    "fear": "fear", "happiness": "happy", "neutral": "neutral",
    "sadness": "sad", "surprise": "surprised",
}


@dataclass
class Triple:
    image: str
    audio: str
    gt: str
    emotion: Optional[str]


def _import_baselines():
    if not BASELINE_RUNNER.is_file():
        raise FileNotFoundError(f"baseline runner not found: {BASELINE_RUNNER}")
    sys.path.insert(0, str(BASELINE_RUNNER.parent))
    try:
        return importlib.import_module("run_baselines")
    finally:
        try: sys.path.remove(str(BASELINE_RUNNER.parent))
        except ValueError: pass


def parse_row(raw: str) -> Optional[Triple]:
    s = raw.strip()
    if not s or s.startswith("#"):
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError(f"expected image,audio,gt[,emotion], got {len(parts)} columns: {raw!r}")
    image, audio, gt = parts[:3]
    for label, p in (("image", image), ("audio", audio), ("gt", gt)):
        if not Path(p).is_file():
            raise FileNotFoundError(f"{label} not found: {p}")
    emo = canonical_emotion(parts[3]) if len(parts) == 4 and parts[3] else infer_emotion(gt)
    return Triple(image=image, audio=audio, gt=gt, emotion=emo)


def read_triples(args) -> list[Triple]:
    raws = Path(args.triples_file).read_text(encoding="utf-8").splitlines() if args.triples_file else args.triples
    out = [x for raw in raws if (x := parse_row(raw)) is not None]
    if not out:
        raise ValueError("no usable triples")
    return out


def _pair_name(rb, method: str, t: Triple, idx: int, scenario: str) -> str:
    kwargs = dict(image=t.image, audio=t.audio, scenario=scenario, idx=idx)
    # Only EAT's official generation route uses GT-derived pose/driving motion.
    kwargs["driving_video"] = t.gt if method == "eat_code" else None
    return rb.get_pair_name(**kwargs)


def generate_one(rb, method: str, t: Triple, idx: int, outdir: Path, scenario: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    name = _pair_name(rb, method, t, idx, scenario)
    target = outdir / f"{name}.mp4"
    if target.is_file():
        return target
    fn = rb.METHODS[method]
    if method == "eat_code":
        if t.emotion is None:
            raise ValueError("EAT requires an emotion label; add manifest emotion or use a MEAD-like GT path")
        full = EAT_FULL.get(t.emotion, t.emotion)
        mapped = getattr(rb, "EAT_EMO_MAP", {}).get(full, MEAD_SHORT.get(t.emotion, full))
        rc = fn(image=t.image, audio=t.audio, output_dir=str(outdir),
                driving_video=t.gt, emo=mapped, scenario=scenario, idx=idx)
    else:
        # Explicitly withhold GT motion from ordinary audio-driven baselines.
        rc = fn(image=t.image, audio=t.audio, output_dir=str(outdir),
                driving_video=None, scenario=scenario, idx=idx)
    if rc not in (0, None):
        raise RuntimeError(f"{method} returned rc={rc}")
    if not target.is_file():
        # Some wrappers return/rename output after generation. Resolve by name first,
        # then accept a unique newly produced mp4 as a compatibility fallback.
        candidates = sorted(outdir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        exact = [p for p in candidates if p.stem == name]
        if exact:
            target = exact[0]
        elif len(candidates) == 1:
            target = candidates[0]
        else:
            raise RuntimeError(f"generation completed but expected output not found: {target}")
    return target


def _run_paper(manifest: Path, method: str, outdir: Path, args) -> int:
    cmd = [sys.executable, str(PAPER_EVALUATOR), "--manifest", str(manifest),
           "--method", method, "--output-dir", str(outdir),
           "--device", args.device, "--timeout", str(args.timeout)]
    if args.metrics:
        cmd += ["--metrics", *args.metrics]
    if args.allow_partial:
        cmd.append("--allow-partial")
    return subprocess.call(cmd, cwd=str(THIS_DIR))


def _combine_tables(root: Path, methods: list[str]) -> Path:
    rows = []
    header = None
    for method in methods:
        p = root / method / "paper_table.csv"
        if not p.is_file():
            continue
        with p.open(newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            header = r.fieldnames
            rows.extend(r)
    out = root / "paper_table.csv"
    if header:
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader(); w.writerows(rows)
    return out


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--triples", nargs="+")
    src.add_argument("--triples-file")
    p.add_argument("--baselines", nargs="+", default=BASELINE_METHODS, choices=BASELINE_METHODS)
    p.add_argument("--metrics", nargs="+", default=None,
                   choices=["lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip"])
    p.add_argument("--output-root", default=str(DEFAULT_RESULT_ROOT))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=7200)
    p.add_argument("--scenario", default="paper_eval")
    p.add_argument("--generation-only", action="store_true")
    p.add_argument("--allow-partial", action="store_true", help="diagnostic only; incomplete rows are not paper-ready")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    triples = read_triples(args)
    root = Path(args.output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    print(f"[final-eval] samples={len(triples)} baselines={args.baselines}")
    print("[final-eval] EAT alone receives GT-derived driving pose, matching its official method; other baselines do not.")
    if args.dry_run:
        for i, t in enumerate(triples):
            print(i, t)
        return 0

    rb = _import_baselines()
    overall_rc = 0
    for method in args.baselines:
        method_dir = root / method
        generated: list[tuple[int, Triple, Path]] = []
        failures = []
        for i, t in enumerate(triples):
            try:
                t0 = time.time()
                fake = generate_one(rb, method, t, i, method_dir / "videos", args.scenario)
                generated.append((i, t, fake))
                print(f"[{method}] [{i+1}/{len(triples)}] OK {fake.name} ({time.time()-t0:.1f}s)")
            except Exception as exc:
                failures.append({"index": i, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{method}] [{i+1}/{len(triples)}] FAIL {exc}", file=sys.stderr)
        (method_dir / "generation_status.json").write_text(json.dumps({
            "expected": len(triples), "generated": len(generated), "failures": failures,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        if failures and not args.allow_partial:
            overall_rc = overall_rc or 2
            print(f"[{method}] evaluation skipped: generation coverage {len(generated)}/{len(triples)}", file=sys.stderr)
            continue
        samples = [Sample(
            name=f"{i:04d}_{Path(t.gt).stem}", fake=str(fake), gt=t.gt,
            emotion=t.emotion, image=t.image, audio=t.audio,
        ) for i, t, fake in generated]
        manifest = write_manifest(method_dir / "manifest.csv", samples)
        if not args.generation_only:
            rc = _run_paper(manifest, method, method_dir, args)
            overall_rc = overall_rc or rc
            # If this is a diagnostic partial run, ensure the row cannot be
            # mistaken for a complete benchmark.
            if failures and (method_dir / "paper_table.csv").is_file():
                rows = list(csv.DictReader((method_dir / "paper_table.csv").open(encoding="utf-8")))
                if rows:
                    rows[0]["Status"] = "incomplete"
                    with (method_dir / "paper_table.csv").open("w", newline="", encoding="utf-8") as f:
                        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    if not args.generation_only:
        table = _combine_tables(root, args.baselines)
        print(f"[final-eval] combined paper table: {table}")
    return overall_rc


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Evaluate one ADEF experiment directory with paper protocol v3.

Two input formats are supported:

1. Baseline-style triples: ``image,audio,gt_video[,emotion]``. ADEF videos
   already exist under ``<father-dir>/<exam_name>``; this evaluator does not
   regenerate them. It resolves each fake using ADEF's inference naming rule
   ``<image_stem>_<audio_stem>_<emotype>.mp4`` and compares it with gt_video.
2. Legacy ADEF pairs: ``fake_filename,gt_video[,emotion]``.

Missing individual files are recorded as upstream sample failures and the
remaining valid samples are evaluated.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from paper_protocol import Sample, canonical_emotion, infer_emotion, write_manifest  # noqa: E402
from paper_table_utils import trim_paper_table  # noqa: E402

PAPER_EVALUATOR = THIS_DIR / "paper_evaluator.py"
DEFAULT_FATHER = Path("/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake")
DEFAULT_SUMMARY = DEFAULT_FATHER / "summary.csv"

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}
ADEF_EMOTYPE = {
    "anger": "angry",
    "contempt": "contempt",
    "disgust": "disgusted",
    "fear": "fear",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprised",
}


def _is_audio(path: str) -> bool:
    return Path(path).suffix.lower() in AUDIO_EXTS


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _detect_row_mode(parts: list[str], path: Path, lineno: int) -> str:
    """Distinguish triples from legacy pairs without guessing by column count alone."""
    if len(parts) == 2:
        return "pairs"
    if len(parts) == 4:
        if not _is_audio(parts[1]) or not _is_video(parts[2]):
            raise ValueError(
                f"{path}:{lineno}: four-column input must be image,audio,gt_video,emotion"
            )
        return "triples"
    if len(parts) != 3:
        raise ValueError(
            f"{path}:{lineno}: expected image,audio,gt_video[,emotion] or "
            "fake_filename,gt_video[,emotion]"
        )

    # Three columns are ambiguous by count. Resolve by media types:
    # image,audio,gt_video vs fake_video,gt_video,emotion.
    if _is_audio(parts[1]) and _is_video(parts[2]):
        return "triples"
    if _is_video(parts[1]):
        return "pairs"
    raise ValueError(
        f"{path}:{lineno}: cannot identify input format. Column 2 is neither "
        f"an audio file (triples) nor a video file (legacy pairs): {parts[1]}"
    )


def _resolve_adef_fake(fake_root: Path, image: Path, audio: Path,
                       emotion: str | None) -> tuple[Path, str | None]:
    """Resolve an existing ADEF output using the pipeline's filename convention."""
    image_stem = image.stem
    audio_stem = audio.stem
    prefix = f"{image_stem}_{audio_stem}_"

    emotype = ADEF_EMOTYPE.get(emotion or "")
    if emotype:
        exact = fake_root / f"{prefix}{emotype}.mp4"
        if exact.is_file():
            return exact, None

    # Compatibility fallback for experiments generated with a custom/legacy
    # emotion spelling. It is safe only when the image+audio prefix identifies
    # exactly one final mp4. Temporary *_temp.mp4 files are excluded.
    candidates = sorted(
        p for p in fake_root.glob(f"{prefix}*.mp4")
        if p.is_file() and not p.name.endswith("_temp.mp4")
    )
    if len(candidates) == 1:
        return candidates[0], None

    if not candidates:
        expected = f"{prefix}{emotype or '<emotion>'}.mp4"
        return fake_root / expected, (
            "ADEF fake not found; expected pipeline output matching "
            f"{expected} under {fake_root}"
        )

    return fake_root / f"{prefix}{emotype or '<emotion>'}.mp4", (
        "ambiguous ADEF outputs for image/audio pair: "
        + ", ".join(p.name for p in candidates)
    )


def read_inputs(path: Path, fake_root: Path) -> tuple[list[Sample], list[dict], int, str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    samples: list[Sample] = []
    failures: list[dict] = []
    expected = 0
    seen_names: set[str] = set()
    detected_mode: str | None = None

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        expected += 1
        parts = [x.strip() for x in text.split(",")]
        mode = _detect_row_mode(parts, path, lineno)
        if detected_mode is None:
            detected_mode = mode
        elif detected_mode != mode:
            raise ValueError(
                f"{path}:{lineno}: mixed input formats are not allowed "
                f"({detected_mode} then {mode})"
            )

        if mode == "triples":
            image = Path(parts[0])
            audio = Path(parts[1])
            gt = Path(parts[2])
            emo = (
                canonical_emotion(parts[3])
                if len(parts) == 4 and parts[3]
                else infer_emotion(gt)
            )
            name = gt.stem
            missing = []
            if not image.is_file():
                missing.append(f"image not found: {image}")
            if not audio.is_file():
                missing.append(f"audio not found: {audio}")
            if not gt.is_file():
                missing.append(f"gt video not found: {gt}")
            fake, fake_error = _resolve_adef_fake(fake_root, image, audio, emo)
            if fake_error:
                missing.append(fake_error)

            if name in seen_names:
                raise ValueError(f"{path}:{lineno}: duplicate sample name: {name}")
            seen_names.add(name)

            if missing:
                failures.append({
                    "stage": "input", "name": name, "fake": str(fake), "gt": str(gt),
                    "error": "; ".join(missing),
                })
                continue
            samples.append(Sample(
                name=name, fake=str(fake), gt=str(gt), emotion=emo,
                image=str(image), audio=str(audio),
            ))
            continue

        # Legacy format: fake_filename,gt_video[,emotion]
        fake = fake_root / parts[0]
        gt = Path(parts[1])
        name = Path(parts[0]).stem
        emo = canonical_emotion(parts[2]) if len(parts) == 3 and parts[2] else infer_emotion(gt)
        if name in seen_names:
            raise ValueError(f"{path}:{lineno}: duplicate fake/sample name: {name}")
        seen_names.add(name)
        missing = []
        if not fake.is_file():
            missing.append(f"fake not found: {fake}")
        if not gt.is_file():
            missing.append(f"gt video not found: {gt}")
        if missing:
            failures.append({
                "stage": "input", "name": name, "fake": str(fake), "gt": str(gt),
                "error": "; ".join(missing),
            })
            continue
        samples.append(Sample(name=name, fake=str(fake), gt=str(gt), emotion=emo))

    if expected == 0:
        raise ValueError(f"no usable rows in {path}")
    return samples, failures, expected, detected_mode or "unknown"


def update_summary(summary: Path, row: dict):
    """Persist the full internal row; paper_table.csv is presentation-only."""
    rows = []
    fieldnames = list(row.keys())
    if summary.is_file():
        try:
            with summary.open(newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "Protocol" in reader.fieldnames and "Status" in reader.fieldnames:
                    rows = [r for r in reader if r.get("Method") != row.get("Method")]
                    fieldnames = list(dict.fromkeys([*reader.fieldnames, *fieldnames]))
        except Exception:
            rows = []
    rows.append(row)
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _clear_outputs(eval_dir: Path) -> None:
    for name in ("paper_table.csv", "paper_metrics.json", "per_video.csv", "failed_samples.csv"):
        p = eval_dir / name
        if p.is_file() or p.is_symlink():
            p.unlink()


def _load_report(eval_dir: Path) -> dict | None:
    path = eval_dir / "paper_metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("exam_name")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--triples-file",
        help="image,audio,gt_video[,emotion]; ADEF fake is resolved from the experiment directory",
    )
    src.add_argument(
        "--pairs-file",
        help="Legacy fake,gt[,emotion]. For compatibility, triples are auto-detected too.",
    )
    p.add_argument("--father-dir", default=str(DEFAULT_FATHER))
    p.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY))
    p.add_argument("--metrics", nargs="+", default=None,
                   choices=["lse", "fid", "fvd", "pairwise", "emotiefflib", "dfer_clip"])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--timeout", type=int, default=43200)
    p.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--eat-device", default=None, help=argparse.SUPPRESS)
    p.add_argument("--reports-dirname", default=None, help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fake_root = Path(args.father_dir) / args.exam_name
    if not fake_root.is_dir():
        print(f"ERROR: experiment directory not found: {fake_root}", file=sys.stderr)
        return 2

    eval_dir = fake_root / "paper_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(eval_dir)
    input_file = Path(args.triples_file or args.pairs_file)
    try:
        samples, upstream_failures, expected_n, input_mode = read_inputs(input_file, fake_root)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(
        f"[ADEF-eval] input-format={input_mode} requested={expected_n} "
        f"resolved={len(samples)}"
    )
    upstream_path = eval_dir / "upstream_failures.json"
    upstream_path.write_text(json.dumps({"failures": upstream_failures}, indent=2, ensure_ascii=False), encoding="utf-8")
    if not samples:
        print(f"ERROR: all {expected_n} requested samples are unavailable; see {upstream_path}", file=sys.stderr)
        return 2

    manifest = write_manifest(eval_dir / "manifest.csv", samples)
    cmd = [sys.executable, str(PAPER_EVALUATOR), "--manifest", str(manifest),
           "--method", args.exam_name, "--output-dir", str(eval_dir),
           "--device", args.device, "--timeout", str(args.timeout),
           "--expected-n", str(expected_n), "--upstream-failures", str(upstream_path)]
    if args.metrics:
        cmd += ["--metrics", *args.metrics]
    rc = subprocess.call(cmd, cwd=str(THIS_DIR))

    # paper_table.csv is deliberately presentation-only. Runtime metadata used
    # for resume/status decisions stays in paper_metrics.json and summary.csv.
    trim_paper_table(eval_dir / "paper_table.csv")
    report = _load_report(eval_dir)
    internal_row = report.get("table_row") if isinstance(report, dict) else None
    if isinstance(internal_row, dict):
        update_summary(Path(args.summary_csv), internal_row)
        print(f"[ADEF-eval] summary updated: {args.summary_csv}")
        print(
            f"[ADEF-eval] status={report.get('status')} "
            f"evaluated={report.get('evaluated_n')}/{report.get('expected_n')}"
        )

    failed = eval_dir / "failed_samples.csv"
    if failed.is_file() and failed.stat().st_size > 0:
        print(f"[ADEF-eval] failed sample report: {failed}", file=sys.stderr)
    if rc != 0:
        print("[ADEF-eval] evaluation failed: at least one requested metric has no usable aggregate", file=sys.stderr)
    elif isinstance(report, dict) and report.get("status") == "partial":
        print("[ADEF-eval] partial evaluation: table values use successful samples only", file=sys.stderr)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

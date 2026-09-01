#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSE-D / LSE-C (Wav2Lip SyncNet) batch evaluation over MEAD11
=============================================================

Computes the Wav2Lip SyncNet lip-sync metrics (LSE-D, LSE-C, AV-offset,
…) for **every** ``.mp4`` under
``/home/Zhouxishi/VirtualMan_proj/dataset/MEAD11/videos`` and writes:

* ``ADEF_remake/eval/LSE_MEAD.json``  — small aggregate report (no
  per-video rows).  Re-loaded cheaply for dashboards / summaries.
* ``ADEF_remake/eval/LSE_MEAD.jsonl`` — one JSON object per video per
  line.  Stream-friendly, append-only, easy to filter with
  ``jq -c 'select(.emotion == "angry")'``.

The dataset has ~30k videos across 40 speaker sub-directories (M003,
M005, …, W040) × 8 emotions × 3 levels.  Running SyncNet over the whole
tree in a single pass is brittle, so we process **per speaker
sub-directory**, delegating each chunk to the existing
``eval_lipsync.py`` CLI which already does recursive globs and per-
frame scoring.  Each chunk's per-video rows are appended to the JSONL
and the aggregate JSON is refreshed (cheap — ~30k records).

Why per-speaker chunks?
-----------------------
* SyncNet's weights are loaded by ``eval_lipsync.py`` *once per chunk*,
  i.e. 40 model-loads for ~30k videos vs. 30k with the naive
  ``--video <file>``-per-call approach.  This is the dominant cost.
* If a chunk crashes (OOM, GPU hang, …) we can resume from the next
  speaker — see ``--resume``.
* Per-speaker / per-emotion aggregates are produced for free, which is
  useful when comparing cross-speaker sync behaviour.

Usage
-----
Default (processes everything, resumable):

    python /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/LSE_MEAD.py

Skip speakers already present in the output file (default behaviour):

    python .../LSE_MEAD.py --resume

Force re-process a specific speaker:

    python .../LSE_MEAD.py --speakers M003

Force re-process everything (ignore ``--resume``):

    python .../LSE_MEAD.py --no-resume

Output files
------------
``LSE_MEAD.json`` (aggregate, small)::

    {
      "dataset":            "/abs/.../MEAD11/videos",
      "generated_at":       "2026-08-25T...",
      "elapsed_sec":        12345.6,
      "completed_speakers": ["M003", "M005", ...],
      "speakers":           ["M003", "M005", ...],
      "aggregate":          { "n_total": ..., "n_success": ...,
                              "lse_d_mean": ..., "lse_c_mean": ..., ... },
      "per_speaker":        { "M003": {...}, ... },
      "per_emotion":        { "angry": {...}, ... }
    }

``LSE_MEAD.jsonl`` (per-video, one record per line)::

    {"video":"/abs/.../M003/front/angry/level_3/M003_front_angry_level_3_021.mp4","speaker":"M003","emotion":"angry","level":3,"lse_d":7.234,"lse_c":1.987,"av_offset":0,"min_dist_raw":...,"n_frames":75,"duration_s":3.0,"elapsed_s":4.2,"error":null}
    {"video":..., ...}
    ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

# ---------------------------------------------------------------------------
# Paths.  These are intentionally absolute so the script can be invoked
# from anywhere (cron, another cwd, …).
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent                                  # .../ADEF_remake/eval
WAV2LIP_DIR = EVAL_ROOT / "Wav2Lip" / "evaluation"
EVAL_LIPSYNC_PY = WAV2LIP_DIR / "eval_lipsync.py"
WAV2LIP_VENV_PY = WAV2LIP_DIR / "venv" / "bin" / "python"

DATASET_ROOT = Path("/home/Zhouxishi/VirtualMan_proj/dataset/MEAD11/videos")
OUTPUT_JSON = EVAL_ROOT / "LSE_MEAD.json"
OUTPUT_JSONL = EVAL_ROOT / "LSE_MEAD.jsonl"
# eval_lipsync.py writes its per-chunk JSON here; overwritten by every
# chunk.  We read it immediately after each subprocess completes, so a
# stale file from a crashed run is harmless.
CHUNK_JSON = EVAL_ROOT / "LSE_MEAD_chunk.json"

# Sanity-check the static paths early so a typo is caught immediately
# rather than mid-run.
if not EVAL_LIPSYNC_PY.is_file():
    raise RuntimeError(f"eval_lipsync.py not found at {EVAL_LIPSYNC_PY}")
if not WAV2LIP_VENV_PY.is_file():
    raise RuntimeError(f"Wav2Lip venv python not found at {WAV2LIP_VENV_PY}")
if not DATASET_ROOT.is_dir():
    raise RuntimeError(f"MEAD11 videos directory not found at {DATASET_ROOT}")


# ---------------------------------------------------------------------------
# JSON-sanitisation helpers.  eval_lipsync.py writes ``NaN`` for failed
# videos — that is not valid JSON (jq, pandas, etc. choke on it).
# Convert NaN / ±Inf to None on the way out.
# ---------------------------------------------------------------------------
import math


def _sanitize_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _sanitize_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Emotion / level extraction.  MEAD layout is:
#     <root>/<speaker>/front/<emotion>/level_<N>/<speaker>_front_<emotion>_level_<N>_<idx>.mp4
# ---------------------------------------------------------------------------
_EMOTION_TOKENS = {
    "angry", "anger", "contempt", "disgusted", "disgust",
    "fear", "happy", "happiness", "sad", "sadness",
    "surprised", "surprise", "neutral", "calm",
}
_LEVEL_RE = re.compile(r"level[_\-]?(\d+)", flags=re.IGNORECASE)


def _extract_meta(video_path: Path, dataset_root: Path) -> Dict[str, Any]:
    """Pull speaker / emotion / level out of the MEAD directory layout."""
    try:
        rel = video_path.relative_to(dataset_root)
    except ValueError:
        rel = Path(video_path.name)

    parts = rel.parts  # e.g. ("M003", "front", "angry", "level_3", "M003_...mp4")
    speaker = parts[0] if parts else None
    emotion: Optional[str] = None
    level: Optional[int] = None

    if len(parts) >= 4 and parts[1].lower() == "front":
        emo = parts[2].lower()
        if emo in _EMOTION_TOKENS:
            emotion = emo
        if len(parts) >= 4:
            m = _LEVEL_RE.search(parts[3])
            if m:
                level = int(m.group(1))
        if level is None:
            m = _LEVEL_RE.search(video_path.stem)
            if m:
                level = int(m.group(1))

    return {
        "speaker": speaker,
        "emotion": emotion,
        "level": level,
        "relpath": str(rel),
    }


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------
def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return None
    return None


def _read_jsonl(path: Path) -> List[dict]:
    """Load every JSON line.  Returns an empty list if the file is
    missing or unparseable."""
    out: List[dict] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[LSE_MEAD] WARNING: malformed JSONL line {ln}: {exc}",
                      file=sys.stderr)
    return out


def _aggregate(results: List[dict]) -> Dict[str, Any]:
    """Mean / std / min / max over the *successful* results.

    Successful means ``error`` is None / falsy AND the numeric fields
    are actually present (failed chunks may write sentinel entries with
    ``lse_d=None``).
    """
    def _ok(r: dict) -> bool:
        if r.get("error"):
            return False
        if r.get("lse_d") is None or r.get("lse_c") is None:
            return False
        return True

    ok = [r for r in results if _ok(r)]
    if not ok:
        return {
            "n_total": len(results),
            "n_success": 0,
            "n_failed": len(results),
        }
    n = len(ok)
    lse_d_vals = [float(r["lse_d"]) for r in ok]
    lse_c_vals = [float(r["lse_c"]) for r in ok]
    offset_vals = [int(r["av_offset"]) for r in ok
                   if r.get("av_offset") is not None]
    lse_d_mean = sum(lse_d_vals) / n
    lse_c_mean = sum(lse_c_vals) / n
    lse_d_var = sum((v - lse_d_mean) ** 2 for v in lse_d_vals) / n
    lse_c_var = sum((v - lse_c_mean) ** 2 for v in lse_c_vals) / n
    return {
        "n_total": len(results),
        "n_success": n,
        "n_failed": len(results) - n,
        "lse_d_mean": lse_d_mean,
        "lse_d_std": lse_d_var ** 0.5,
        "lse_c_mean": lse_c_mean,
        "lse_c_std": lse_c_var ** 0.5,
        "lse_c_min": min(lse_c_vals),
        "lse_c_max": max(lse_c_vals),
        "av_offset_mean": (sum(offset_vals) / len(offset_vals)
                           if offset_vals else None),
    }


def _bucket_aggregate(results: List[dict], key: str) -> Dict[str, Dict[str, Any]]:
    """Group results by *key* (e.g. ``"speaker"`` / ``"emotion"``) and
    aggregate LSE-D / LSE-C within each bucket."""
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        bucket = r.get(key) or "unknown"
        buckets[bucket].append(r)
    return {name: _aggregate(items) for name, items in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Per-speaker driver — invokes eval_lipsync.py once per speaker subdir.
# ---------------------------------------------------------------------------
def _make_sentinel(video_path: Path, speaker: str, dataset_root: Path,
                   error_msg: str, elapsed: float) -> dict:
    """Build one per-video failure row from a real ``video_path``.

    Always emits ``lse_d=None`` / ``lse_c=None`` so downstream
    aggregations can tell successes apart from failures without a
    separate "failed" boolean.
    """
    meta = _extract_meta(video_path, dataset_root)
    return {
        "video":        str(video_path),
        "speaker":      meta.get("speaker") or speaker,
        "emotion":      meta.get("emotion"),
        "level":        meta.get("level"),
        "relpath":      meta.get("relpath"),
        "lse_d":        None,
        "lse_c":        None,
        "av_offset":    None,
        "min_dist_raw": None,
        "n_frames":     0,
        "duration_s":   0.0,
        "elapsed_s":    elapsed,
        "error":        error_msg,
    }


def _run_speaker_chunk(
    speaker: str,
    dataset_root: Path,
    *,
    timeout: int = 7200,
    device: Optional[str] = None,
    quiet: bool = False,
) -> List[dict]:
    """Run ``eval_lipsync.py --video_dir <dataset_root>/<speaker>`` and
    return one row per ``.mp4`` under the speaker dir (sentinel rows
    are emitted for videos we couldn't score).
    """
    speaker_dir = dataset_root / speaker
    if not speaker_dir.is_dir():
        print(f"[LSE_MEAD] {speaker}: directory missing, skipping",
              file=sys.stderr)
        return []

    # Pre-enumerate every video in this speaker so we can emit a
    # per-video sentinel if the subprocess fails or produces no JSON.
    all_videos: List[Path] = sorted(p for p in speaker_dir.rglob("*.mp4"))
    n_mp4 = len(all_videos)
    if not quiet:
        print(f"[LSE_MEAD] {speaker}: launching eval_lipsync.py "
              f"({n_mp4} mp4 found)")

    cmd = [
        str(WAV2LIP_VENV_PY),
        str(EVAL_LIPSYNC_PY),
        "--video_dir", str(speaker_dir),
        "--output_json", str(CHUNK_JSON),
    ]
    if device:
        cmd.extend(["--device", device])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WAV2LIP_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - t0
        print(f"[LSE_MEAD] {speaker}: TIMEOUT after {timeout}s",
              file=sys.stderr)
        return [_make_sentinel(v, speaker, dataset_root,
                               f"TimeoutExpired: {exc}", elapsed)
                for v in all_videos]
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        print(f"[LSE_MEAD] {speaker}: uncaught exception {exc!r}",
              file=sys.stderr)
        return [_make_sentinel(v, speaker, dataset_root,
                               f"{type(exc).__name__}: {exc}", elapsed)
                for v in all_videos]

    elapsed = time.time() - t0

    # Even on non-zero exit, eval_lipsync.py may have written a JSON
    # with per-video rows (one per ffmpeg/SyncNet failure + the
    # successes).  Try to read it first; only fall back to per-video
    # sentinels when the JSON is missing/unreadable.
    payload = _safe_read_json(CHUNK_JSON)
    if payload and "results" in payload:
        if proc.returncode != 0:
            last_err = (proc.stderr or "").splitlines()
            last_err = last_err[-1] if last_err else "(no stderr)"
            print(f"[LSE_MEAD] {speaker}: rc={proc.returncode} but partial "
                  f"JSON present ({len(payload['results'])} rows) — "
                  f"using eval_lipsync output.  last_err: {last_err}",
                  file=sys.stderr)
        # Fall through to the success path below.
    elif proc.returncode != 0:
        last_err = (proc.stderr or "").splitlines()
        last_err = last_err[-1] if last_err else "(no stderr)"
        print(f"[LSE_MEAD] {speaker}: rc={proc.returncode} after {elapsed:.1f}s "
              f"— {last_err}", file=sys.stderr)
        return [_make_sentinel(v, speaker, dataset_root,
                               f"rc={proc.returncode}: {last_err}", elapsed)
                for v in all_videos]
    else:
        print(f"[LSE_MEAD] {speaker}: no JSON output (elapsed {elapsed:.1f}s)",
              file=sys.stderr)
        return [_make_sentinel(v, speaker, dataset_root,
                               "no_json_output", elapsed)
                for v in all_videos]

    raw_results = payload.get("results") or []
    enriched: List[dict] = []
    for r in raw_results:
        vpath = Path(r.get("video") or "")
        meta = _extract_meta(vpath, dataset_root) if vpath else {}
        enriched.append({
            "video":        r.get("video"),
            "speaker":      meta.get("speaker"),
            "emotion":      meta.get("emotion"),
            "level":        meta.get("level"),
            "relpath":      meta.get("relpath"),
            "lse_d":        _sanitize_float(r.get("lse_d")),
            "lse_c":        _sanitize_float(r.get("lse_c")),
            "av_offset":    _sanitize_int(r.get("av_offset")),
            "min_dist_raw": _sanitize_float(r.get("min_dist_raw")),
            "n_frames":     _sanitize_int(r.get("n_frames")),
            "duration_s":   _sanitize_float(r.get("duration_s")),
            "elapsed_s":    _sanitize_float(r.get("elapsed_s")),
            "error":        r.get("error"),
        })

    # If eval_lipsync.py wrote fewer rows than we expected (e.g. it
    # silently dropped unreadable inputs), backfill sentinels for the
    # missing videos so the per-speaker video count matches the file
    # tree exactly.
    seen = {e.get("video") for e in enriched if e.get("video")}
    missing = [v for v in all_videos if str(v) not in seen]
    if missing:
        if not quiet:
            print(f"[LSE_MEAD] {speaker}: backfilling {len(missing)} missing "
                  f"rows as sentinels")
        for v in missing:
            enriched.append(_make_sentinel(
                v, speaker, dataset_root, "not_in_eval_lipsync_output", elapsed))

    n_ok = sum(1 for e in enriched if not e.get("error"))
    if not quiet:
        print(f"[LSE_MEAD] {speaker}: {n_ok}/{len(enriched)} ok "
              f"({elapsed:.1f}s)")

    return enriched


# ---------------------------------------------------------------------------
# Resumability helpers.
# ---------------------------------------------------------------------------
def _load_existing_aggregate(agg_path: Path) -> Dict[str, Any]:
    report = _safe_read_json(agg_path) or {}
    return report


def _completed_speakers(report: Dict[str, Any]) -> Set[str]:
    """Return the set of speaker ids already covered in the report.

    The aggregate JSONL keeps a ``completed_speakers`` list (alphabet-
    ically sorted) so resume is O(1) — no need to scan the jsonl.
    """
    out: Set[str] = set()
    for s in report.get("completed_speakers") or []:
        if isinstance(s, str):
            out.add(s)
    return out


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Append *rows* to *path* as JSON-lines.  Returns the count
    actually written (skips rows with no ``video`` field)."""
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            if not row.get("video"):
                continue
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _write_aggregate_report(
    path: Path,
    *,
    dataset: Path,
    speakers: List[str],
    completed_speakers: List[str],
    elapsed_sec: float,
    records: List[dict],
) -> None:
    report = {
        "dataset":            str(dataset),
        "generated_at":       datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec":        elapsed_sec,
        "speakers":           speakers,
        "completed_speakers": completed_speakers,
        "aggregate":          _aggregate(records),
        "per_speaker":        _bucket_aggregate(records, "speaker"),
        "per_emotion":        _bucket_aggregate(records, "emotion"),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Batch LSE-D / LSE-C evaluation over MEAD11 videos "
                    "using Wav2Lip SyncNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default=str(DATASET_ROOT),
                   help="Root directory containing per-speaker subdirs.")
    p.add_argument("--output", "-o", default=str(OUTPUT_JSON),
                   help="Path to the aggregate JSON (the .jsonl sibling "
                        "is written next to it).")
    p.add_argument("--speakers", nargs="+", default=None,
                   help="Restrict to a subset of speaker ids (default: all).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip speakers already present in the output file "
                        "(default: ON).")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Re-process every speaker, ignoring the output file.")
    p.add_argument("--device", default="cuda:7",
                   help="Override SyncNet device (cuda / cuda:0 / cpu).")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Per-speaker timeout in seconds (default: 2 hours).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-speaker progress prints.")
    p.add_argument("--keep-aggregate-on-resume", action="store_true",
                   help="When --no-resume is set, do not overwrite the "
                        "existing aggregate JSONL before re-processing. "
                        "Useful when you want to compare old vs. new runs.")
    args = p.parse_args(list(argv) if argv is not None else None)

    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")

    if not dataset_root.is_dir():
        p.error(f"--dataset-root not found: {dataset_root}")

    # Discover speaker sub-directories.
    if args.speakers:
        speakers = list(args.speakers)
    else:
        speakers = sorted(
            d.name for d in dataset_root.iterdir() if d.is_dir()
        )

    # Resumability.
    existing_report = (_load_existing_aggregate(output_path)
                       if args.resume else None)
    if existing_report and not args.quiet:
        n_existing = (sum(1 for _ in jsonl_path.open(encoding="utf-8"))
                      if jsonl_path.is_file() else 0)
        print(f"[LSE_MEAD] resuming — {len(existing_report.get('completed_speakers', []))} "
              f"speakers already done; {n_existing} jsonl rows on disk")

    done = _completed_speakers(existing_report) if existing_report else set()
    todo = [s for s in speakers if s not in done]
    if not todo:
        print(f"[LSE_MEAD] nothing to do — all {len(speakers)} speakers "
              f"already present")
        return 0
    if not args.quiet:
        print(f"[LSE_MEAD] speakers to process: {len(todo)}/{len(speakers)}")

    # When --no-resume is passed we wipe the existing jsonl/aggregate so
    # the run is a clean slate (unless the user opted to keep the
    # aggregate on resume, which is the comparison-run use case).
    if not args.resume and not args.keep_aggregate_on_resume:
        for f in (output_path, jsonl_path):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        if not args.quiet:
            print(f"[LSE_MEAD] --no-resume: cleared {output_path} and {jsonl_path}")

    completed_speakers = sorted(done)
    overall_t0 = time.time()

    for i, speaker in enumerate(todo, start=1):
        if not args.quiet:
            print(f"\n[LSE_MEAD] === [{i}/{len(todo)}] {speaker} ===")
        try:
            chunk = _run_speaker_chunk(
                speaker,
                dataset_root,
                timeout=args.timeout,
                device=args.device,
                quiet=args.quiet,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[LSE_MEAD] {speaker}: uncaught exception {exc!r}",
                  file=sys.stderr)
            # Build per-video sentinels for every mp4 under this speaker.
            speaker_dir = dataset_root / speaker
            chunk = [_make_sentinel(v, speaker, dataset_root,
                                    f"{type(exc).__name__}: {exc}", 0.0)
                     for v in sorted(speaker_dir.rglob("*.mp4"))]

        n_written = _append_jsonl(jsonl_path, chunk)
        if speaker not in completed_speakers:
            completed_speakers.append(speaker)
            completed_speakers.sort()
        if not args.quiet:
            print(f"[LSE_MEAD] {speaker}: wrote {n_written} rows -> {jsonl_path}")

        # Refresh aggregate report (cheap — 30k records is <1s).
        all_records = _read_jsonl(jsonl_path)
        _write_aggregate_report(
            output_path,
            dataset=dataset_root,
            speakers=speakers,
            completed_speakers=completed_speakers,
            elapsed_sec=time.time() - overall_t0,
            records=all_records,
        )

    elapsed = time.time() - overall_t0

    # Final summary.
    all_records = _read_jsonl(jsonl_path)
    agg = _aggregate(all_records)
    print("\n[LSE_MEAD] ============= SUMMARY =============")
    print(f"  dataset    : {dataset_root}")
    print(f"  jsonl      : {jsonl_path}")
    print(f"  aggregate  : {output_path}")
    print(f"  speakers   : {len(completed_speakers)}/{len(speakers)}")
    print(f"  results    : {agg.get('n_total', 0)} total, "
          f"{agg.get('n_success', 0)} ok, {agg.get('n_failed', 0)} failed")
    if agg.get("n_success", 0):
        print(f"  LSE-D mean : {agg['lse_d_mean']:.3f} "
              f"± {agg['lse_d_std']:.3f}")
        print(f"  LSE-C mean : {agg['lse_c_mean']:.3f} "
              f"± {agg['lse_c_std']:.3f}")
        print(f"  LSE-C range: [{agg['lse_c_min']:.3f}, {agg['lse_c_max']:.3f}]")
        if agg.get("av_offset_mean") is not None:
            print(f"  AV offset  : {agg['av_offset_mean']:+.2f} frames")
    print(f"  elapsed    : {elapsed:.1f}s")
    print("[LSE_MEAD] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

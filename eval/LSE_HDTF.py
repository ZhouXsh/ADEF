#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSE-D / LSE-C (Wav2Lip SyncNet) batch evaluation over HDTF_Processed
=====================================================================

Computes the Wav2Lip SyncNet lip-sync metrics (LSE-D, LSE-C, AV-offset,
…) for **every** ``.mp4`` under
``/home/Zhouxishi/VirtualMan_proj/dataset/HDTF_Processed/videos`` and
writes:

* ``ADEF_remake/eval/LSE_HDTF.json``  — small aggregate report.
* ``ADEF_remake/eval/LSE_HDTF.jsonl`` — one JSON object per video per
  line (append-only, stream-friendly).

This is the HDTF counterpart of ``LSE_MEAD.py`` — same code shape, same
output schema, only the input layout differs.  Where MEAD11 has the
``<speaker>/front/<emotion>/level_<N>/...`` tree, HDTF_Processed is a
flat directory of ``{speaker}_{idx}.mp4`` files (each sitting next to
its ``.pkl`` and ``.wav`` twins).  Speaker ids are therefore parsed out
of the filename, not out of the directory layout.

Layout
------
::

    HDTF_Processed/videos/
      RD_Radio10_000.mp4   <- speaker=RD_Radio10, utt=000
      RD_Radio10_000.pkl
      RD_Radio10_000.wav
      RD_Radio11_000.mp4
      ...
      WRA_ErikPaulsen_000.mp4
      WRA_ErikPaulsen_001.mp4
      ...

With ~355 videos across ~296 speakers, per-speaker chunks are small
(usually 1–2 videos each, max ~10).  SyncNet model-load time dominates
over per-video inference, so the per-speaker chunking inherited from
``LSE_MEAD.py`` is wasteful here.  Two practical options:

* Just run the script as-is (default).  ~296 model loads × ~5s + the
  actual scoring → roughly 25–30 min wall-time on a single GPU.
* Skip per-speaker chunking entirely by passing ``--batch-size 20``:
  videos are alphabetised then grouped into batches of 20 (~18 batches
  → ~13 min wall-time).  Per-speaker aggregates still work because
  speakers are still parsed from filenames.

Usage
-----
Default (per-speaker chunks, resumable):

    python /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/LSE_HDTF.py

Per-batch (faster — recommended for HDTF):

    python .../LSE_HDTF.py --batch-size 20

Resume / no-resume / specific speaker: same flags as LSE_MEAD.py.

Output schema (``LSE_HDTF.json``)
---------------------------------
Same as ``LSE_MEAD.json`` — see that file's docstring.  The
``per_emotion`` bucket is meaningless for HDTF (no emotion in the
filenames) and will collapse to a single ``"unknown"`` bucket.
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
# Paths.
# ---------------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
EVAL_ROOT = THIS_FILE.parent                                  # .../ADEF_remake/eval
WAV2LIP_DIR = EVAL_ROOT / "Wav2Lip" / "evaluation"
EVAL_LIPSYNC_PY = WAV2LIP_DIR / "eval_lipsync.py"
WAV2LIP_VENV_PY = WAV2LIP_DIR / "venv" / "bin" / "python"

DATASET_ROOT = Path("/home/Zhouxishi/VirtualMan_proj/dataset/HDTF_Processed/videos")
OUTPUT_JSON = EVAL_ROOT / "LSE_HDTF.json"
OUTPUT_JSONL = EVAL_ROOT / "LSE_HDTF.jsonl"
# eval_lipsync.py writes its per-chunk JSON here; overwritten by every
# chunk.  We read it immediately after each subprocess completes.
CHUNK_JSON = EVAL_ROOT / "LSE_HDTF_chunk.json"

if not EVAL_LIPSYNC_PY.is_file():
    raise RuntimeError(f"eval_lipsync.py not found at {EVAL_LIPSYNC_PY}")
if not WAV2LIP_VENV_PY.is_file():
    raise RuntimeError(f"Wav2Lip venv python not found at {WAV2LIP_VENV_PY}")
if not DATASET_ROOT.is_dir():
    raise RuntimeError(f"HDTF_Processed directory not found at {DATASET_ROOT}")


# ---------------------------------------------------------------------------
# JSON-sanitisation helpers.  eval_lipsync.py writes ``NaN`` for failed
# videos — that is not valid JSON (jq, pandas, etc. choke on it).
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
# Speaker / utterance extraction.  HDTF layout is flat — speaker and
# utterance index live in the filename.  Pattern observed:
#     RD_Radio10_000.mp4   ->  speaker=RD_Radio10, utt=000
#     WRA_ErikPaulsen_007.mp4  -> speaker=WRA_ErikPaulsen, utt=007
# So speaker = everything before the last ``_<digits>``.
# ---------------------------------------------------------------------------
_HDTF_UTT_RE = re.compile(r"^(?P<speaker>.+?)_(?P<utt>\d+)$")


def _extract_meta(video_path: Path, dataset_root: Path) -> Dict[str, Any]:
    """Pull speaker / utterance out of the HDTF filename layout.

    Always returns ``relpath`` (str relative to *dataset_root*).
    Speaker is parsed from the filename; ``emotion`` / ``level`` are
    ``None`` because HDTF has no such annotations.  If the filename
    doesn't match the ``<name>_<digits>`` pattern (data error /
    non-HDTF file), ``speaker`` falls back to the file stem so we never
    end up with a ``None`` group label that crashes bucketing.
    """
    try:
        rel = video_path.relative_to(dataset_root)
    except ValueError:
        rel = Path(video_path.name)

    speaker: Optional[str] = None
    utterance: Optional[int] = None
    m = _HDTF_UTT_RE.match(video_path.stem)
    if m:
        speaker = m.group("speaker")
        try:
            utterance = int(m.group("utt"))
        except (TypeError, ValueError):
            utterance = None
    if speaker is None:
        # Fallback: use the file stem as the speaker label.  This keeps
        # the per-speaker aggregate well-defined even for stray files
        # that don't follow the HDTF naming convention.
        speaker = video_path.stem

    return {
        "speaker": speaker,
        "emotion": None,
        "level": None,
        "utterance": utterance,
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
                print(f"[LSE_HDTF] WARNING: malformed JSONL line {ln}: {exc}",
                      file=sys.stderr)
    return out


def _aggregate(results: List[dict]) -> Dict[str, Any]:
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
    """Group results by *key* and aggregate.  Bucket keys are coerced
    to ``str`` so a mix of ``int`` (e.g. utterance index) and ``None``
    (sentinel rows) does not crash ``sorted``.
    """
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        bucket = r.get(key)
        bucket_key = "unknown" if bucket is None else str(bucket)
        buckets[bucket_key].append(r)
    return {name: _aggregate(items) for name, items in sorted(buckets.items())}


# ---------------------------------------------------------------------------
# Speaker discovery + per-speaker / per-batch grouping.
# ---------------------------------------------------------------------------
def _discover_speakers_flat(dataset_root: Path) -> Dict[str, List[Path]]:
    """Group every ``.mp4`` under *dataset_root* (recursive) by speaker
    id parsed from the filename.  Returns ``{speaker: [video_path, …]}``
    sorted by speaker name and by filename within each speaker.
    """
    by_speaker: Dict[str, List[Path]] = defaultdict(list)
    for mp4 in sorted(dataset_root.rglob("*.mp4")):
        meta = _extract_meta(mp4, dataset_root)
        speaker = meta.get("speaker") or mp4.stem
        by_speaker[speaker].append(mp4)
    return {sp: sorted(vs) for sp, vs in by_speaker.items()}


# ---------------------------------------------------------------------------
# Per-chunk driver — invokes eval_lipsync.py once per chunk.
# ---------------------------------------------------------------------------
def _make_sentinel(video_path: Path, speaker: str, dataset_root: Path,
                   error_msg: str, elapsed: float) -> dict:
    meta = _extract_meta(video_path, dataset_root)
    return {
        "video":        str(video_path),
        "speaker":      meta.get("speaker") or speaker,
        "emotion":      meta.get("emotion"),
        "level":        meta.get("level"),
        "utterance":    meta.get("utterance"),
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


def _write_chunk_videos(videos: List[Path]) -> Path:
    """Stage *videos* into a fresh tmpdir and return the dir.

    eval_lipsync.py takes a *directory* (recursive glob); HDTF videos
    sit alongside their ``.pkl`` / ``.wav`` siblings, which we don't want
    to accidentally glob into.  We symlink each target mp4 into a
    staging dir so only the chosen videos are seen.
    """
    import tempfile
    tmpdir = Path(tempfile.mkdtemp(prefix="lse_hdtf_chunk_"))
    for v in videos:
        link = tmpdir / v.name
        try:
            link.symlink_to(v.resolve())
        except OSError:
            # Fallback: hard-link or copy.
            import shutil
            shutil.copy2(v, link)
    return tmpdir


def _run_chunk(
    chunk_label: str,
    videos: List[Path],
    dataset_root: Path,
    *,
    timeout: int = 7200,
    device: Optional[str] = None,
    quiet: bool = False,
) -> List[dict]:
    """Score a batch of *videos* via eval_lipsync.py.

    *chunk_label* is just for log lines (speaker id or ``"batch_007"``).
    """
    if not videos:
        return []

    stage_dir = _write_chunk_videos(videos)
    n_mp4 = len(videos)
    if not quiet:
        print(f"[LSE_HDTF] {chunk_label}: launching eval_lipsync.py "
              f"({n_mp4} mp4 in staged dir)")

    cmd = [
        str(WAV2LIP_VENV_PY),
        str(EVAL_LIPSYNC_PY),
        "--video_dir", str(stage_dir),
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
        print(f"[LSE_HDTF] {chunk_label}: TIMEOUT after {timeout}s",
              file=sys.stderr)
        return [_make_sentinel(v, chunk_label, dataset_root,
                               f"TimeoutExpired: {exc}", elapsed)
                for v in videos]
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - t0
        print(f"[LSE_HDTF] {chunk_label}: uncaught exception {exc!r}",
              file=sys.stderr)
        return [_make_sentinel(v, chunk_label, dataset_root,
                               f"{type(exc).__name__}: {exc}", elapsed)
                for v in videos]
    finally:
        # Staging dir is no longer needed (eval_lipsync.py wrote JSON
        # to CHUNK_JSON in EVAL_ROOT, not in stage_dir).
        import shutil
        shutil.rmtree(stage_dir, ignore_errors=True)

    elapsed = time.time() - t0

    # Try to read the JSON regardless of return code — partial failures
    # still produce a usable per-video JSON.
    payload = _safe_read_json(CHUNK_JSON)
    if payload and "results" in payload:
        if proc.returncode != 0:
            last_err = (proc.stderr or "").splitlines()
            last_err = last_err[-1] if last_err else "(no stderr)"
            print(f"[LSE_HDTF] {chunk_label}: rc={proc.returncode} but partial "
                  f"JSON present ({len(payload['results'])} rows) — "
                  f"using eval_lipsync output.  last_err: {last_err}",
                  file=sys.stderr)
        # Fall through to success path below.
    elif proc.returncode != 0:
        last_err = (proc.stderr or "").splitlines()
        last_err = last_err[-1] if last_err else "(no stderr)"
        print(f"[LSE_HDTF] {chunk_label}: rc={proc.returncode} after {elapsed:.1f}s "
              f"— {last_err}", file=sys.stderr)
        return [_make_sentinel(v, chunk_label, dataset_root,
                               f"rc={proc.returncode}: {last_err}", elapsed)
                for v in videos]
    else:
        print(f"[LSE_HDTF] {chunk_label}: no JSON output (elapsed {elapsed:.1f}s)",
              file=sys.stderr)
        return [_make_sentinel(v, chunk_label, dataset_root,
                               "no_json_output", elapsed)
                for v in videos]

    raw_results = payload.get("results") or []
    # Map eval_lipsync's staged paths back to source paths so the
    # backfill comparison (and downstream consumers) see real locations.
    by_basename: Dict[str, Path] = {v.name: v for v in videos}

    enriched: List[dict] = []
    for r in raw_results:
        staged_path = Path(r.get("video") or "")
        source_path = by_basename.get(staged_path.name, staged_path)
        meta = _extract_meta(source_path, dataset_root) if source_path else {}
        enriched.append({
            "video":        str(source_path),
            "speaker":      meta.get("speaker"),
            "emotion":      meta.get("emotion"),
            "level":        meta.get("level"),
            "utterance":    meta.get("utterance"),
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

    # Backfill sentinels for any video eval_lipsync.py silently dropped
    # (e.g. ffmpeg decode failure that wasn't surfaced in the JSON).
    seen = {e.get("video") for e in enriched if e.get("video")}
    missing = [v for v in videos if str(v) not in seen]
    if missing:
        if not quiet:
            print(f"[LSE_HDTF] {chunk_label}: backfilling {len(missing)} "
                  f"missing rows as sentinels")
        for v in missing:
            enriched.append(_make_sentinel(
                v, chunk_label, dataset_root, "not_in_eval_lipsync_output",
                elapsed))

    n_ok = sum(1 for e in enriched if not e.get("error"))
    if not quiet:
        print(f"[LSE_HDTF] {chunk_label}: {n_ok}/{len(enriched)} ok "
              f"({elapsed:.1f}s)")
    return enriched


# ---------------------------------------------------------------------------
# Resumability helpers.
# ---------------------------------------------------------------------------
def _load_existing_aggregate(agg_path: Path) -> Dict[str, Any]:
    return _safe_read_json(agg_path) or {}


def _completed_chunks(report: Dict[str, Any]) -> Set[str]:
    out: Set[str] = set()
    for s in report.get("completed_chunks") or []:
        if isinstance(s, str):
            out.add(s)
    return out


def _append_jsonl(path: Path, rows: Iterable[dict]) -> int:
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
    chunks: List[str],
    completed_chunks: List[str],
    elapsed_sec: float,
    records: List[dict],
) -> None:
    report = {
        "dataset":          str(dataset),
        "generated_at":     datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec":      elapsed_sec,
        "chunks":           chunks,
        "completed_chunks": completed_chunks,
        "aggregate":        _aggregate(records),
        "per_speaker":      _bucket_aggregate(records, "speaker"),
        "per_utterance":    _bucket_aggregate(records, "utterance"),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Chunk planning.  Returns ``[(label, [video_path, …]), …]``.
# ---------------------------------------------------------------------------
def _plan_chunks(
    dataset_root: Path,
    *,
    batch_size: int,
    only_speakers: Optional[List[str]] = None,
) -> List[tuple]:
    """Decide how to slice the dataset into SyncNet eval chunks.

    * If *batch_size* > 0: alphabetise all mp4s, group into batches of
      that size.  Each chunk is labelled ``"batch_NNN"``.  This is the
      recommended mode for HDTF — fewer model loads.
    * Else: one chunk per speaker id, labelled with the speaker name
      (matches the per-subdir behaviour of LSE_MEAD.py).
    """
    by_speaker = _discover_speakers_flat(dataset_root)
    if only_speakers:
        unknown = set(only_speakers) - set(by_speaker)
        if unknown:
            print(f"[LSE_HDTF] WARNING: requested speakers not in dataset: "
                  f"{sorted(unknown)}", file=sys.stderr)
        by_speaker = {sp: vs for sp, vs in by_speaker.items()
                      if sp in only_speakers}

    if batch_size and batch_size > 0:
        # Flat alphabetical batching.
        flat = [v for sp in sorted(by_speaker) for v in by_speaker[sp]]
        chunks: List[tuple] = []
        for i in range(0, len(flat), batch_size):
            chunk_videos = flat[i:i + batch_size]
            label = f"batch_{i // batch_size:03d}"
            chunks.append((label, chunk_videos))
        return chunks

    # Per-speaker (one chunk per speaker).
    return [(sp, vs) for sp, vs in sorted(by_speaker.items())]


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Batch LSE-D / LSE-C evaluation over HDTF_Processed "
                    "videos using Wav2Lip SyncNet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default=str(DATASET_ROOT),
                   help="Root directory containing the HDTF mp4s.")
    p.add_argument("--output", "-o", default=str(OUTPUT_JSON),
                   help="Path to the aggregate JSON (the .jsonl sibling "
                        "is written next to it).")
    p.add_argument("--speakers", nargs="+", default=None,
                   help="Restrict to a subset of speaker ids.")
    p.add_argument("--batch-size", type=int, default=0,
                   help="If > 0, alphabetise all mp4s and group them "
                        "into batches of this many for SyncNet eval. "
                        "Recommended for HDTF (default 0 = per-speaker, "
                        "matches LSE_MEAD.py).")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip chunks already present in the output file "
                        "(default: ON).")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Re-process every chunk, ignoring the output file.")
    p.add_argument("--device", default=None,
                   help="Override SyncNet device (cuda / cuda:0 / cpu).")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Per-chunk timeout in seconds (default: 2 hours).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-chunk progress prints.")
    args = p.parse_args(list(argv) if argv is not None else None)

    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")

    if not dataset_root.is_dir():
        p.error(f"--dataset-root not found: {dataset_root}")

    chunks = _plan_chunks(
        dataset_root,
        batch_size=args.batch_size,
        only_speakers=args.speakers,
    )
    chunk_labels = [c[0] for c in chunks]
    if not chunks:
        print("[LSE_HDTF] no videos found — nothing to do")
        return 0

    existing_report = (_load_existing_aggregate(output_path)
                       if args.resume else None)
    if existing_report and not args.quiet:
        n_existing = (sum(1 for _ in jsonl_path.open(encoding="utf-8"))
                      if jsonl_path.is_file() else 0)
        print(f"[LSE_HDTF] resuming — {len(existing_report.get('completed_chunks', []))} "
              f"chunks already done; {n_existing} jsonl rows on disk")

    done = _completed_chunks(existing_report) if existing_report else set()
    todo = [(lab, vs) for lab, vs in chunks if lab not in done]
    if not todo:
        print(f"[LSE_HDTF] nothing to do — all {len(chunks)} chunks "
              f"already present")
        return 0
    if not args.quiet:
        total_vids = sum(len(vs) for _, vs in todo)
        print(f"[LSE_HDTF] chunks to process: {len(todo)}/{len(chunks)} "
              f"({total_vids} videos)")

    if not args.resume:
        for f in (output_path, jsonl_path):
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        if not args.quiet:
            print(f"[LSE_HDTF] --no-resume: cleared {output_path} and {jsonl_path}")

    completed_chunks = sorted(done)
    overall_t0 = time.time()

    for i, (label, videos) in enumerate(todo, start=1):
        if not args.quiet:
            print(f"\n[LSE_HDTF] === [{i}/{len(todo)}] {label} "
                  f"({len(videos)} videos) ===")
        try:
            chunk = _run_chunk(
                label, videos, dataset_root,
                timeout=args.timeout,
                device=args.device,
                quiet=args.quiet,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[LSE_HDTF] {label}: uncaught exception {exc!r}",
                  file=sys.stderr)
            chunk = [_make_sentinel(v, label, dataset_root,
                                    f"{type(exc).__name__}: {exc}", 0.0)
                     for v in videos]

        n_written = _append_jsonl(jsonl_path, chunk)
        if label not in completed_chunks:
            completed_chunks.append(label)
            completed_chunks.sort()
        if not args.quiet:
            print(f"[LSE_HDTF] {label}: wrote {n_written} rows -> {jsonl_path}")

        all_records = _read_jsonl(jsonl_path)
        _write_aggregate_report(
            output_path,
            dataset=dataset_root,
            chunks=chunk_labels,
            completed_chunks=completed_chunks,
            elapsed_sec=time.time() - overall_t0,
            records=all_records,
        )

    elapsed = time.time() - overall_t0

    all_records = _read_jsonl(jsonl_path)
    agg = _aggregate(all_records)
    print("\n[LSE_HDTF] ============= SUMMARY =============")
    print(f"  dataset    : {dataset_root}")
    print(f"  jsonl      : {jsonl_path}")
    print(f"  aggregate  : {output_path}")
    print(f"  chunks     : {len(completed_chunks)}/{len(chunks)}")
    print(f"  speakers   : {len({r['speaker'] for r in all_records if r.get('speaker')})}")
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
    print("[LSE_HDTF] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
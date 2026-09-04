#!/usr/bin/env python3
"""Shared protocol helpers for paper-grade ADEF evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

PROTOCOL_VERSION = "ADEF-paper-eval-v3"

EMOTION_ALIASES = {
    "ang": "anger", "angry": "anger", "anger": "anger",
    "con": "contempt", "contempt": "contempt",
    "dis": "disgust", "disgust": "disgust", "disgusted": "disgust",
    "fea": "fear", "fear": "fear",
    "hap": "happiness", "happy": "happiness", "happiness": "happiness",
    "neu": "neutral", "neutral": "neutral", "calm": "neutral",
    "sad": "sadness", "sadness": "sadness",
    "sur": "surprise", "surprise": "surprise", "surprised": "surprise",
}
MEAD_EMOTIONS = (
    "anger", "contempt", "disgust", "fear", "happiness", "neutral",
    "sadness", "surprise",
)
DFER_CLIP_EMOTIONS = tuple(x for x in MEAD_EMOTIONS if x != "contempt")

MEAD_SHORT = {
    "anger": "ang", "contempt": "con", "disgust": "dis", "fear": "fea",
    "happiness": "hap", "neutral": "neu", "sadness": "sad", "surprise": "sur",
}

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


@dataclass(frozen=True)
class Sample:
    name: str
    fake: str
    gt: str
    emotion: Optional[str] = None
    image: Optional[str] = None
    audio: Optional[str] = None

    def normalized(self) -> "Sample":
        emo = canonical_emotion(self.emotion) if self.emotion else infer_emotion(self.gt)
        return Sample(
            name=self.name,
            fake=str(Path(self.fake).expanduser()),
            gt=str(Path(self.gt).expanduser()),
            emotion=emo,
            image=self.image,
            audio=self.audio,
        )


def canonical_emotion(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    key = label.strip().lower()
    if not key:
        return None
    return EMOTION_ALIASES.get(key, key)


def infer_emotion(path: str | Path) -> Optional[str]:
    p = Path(path)
    tokens = re.split(r"[_\-\s.]+", p.stem.lower())
    tokens.extend(parent.name.lower() for parent in list(p.parents)[:4])
    for token in tokens:
        emo = canonical_emotion(token)
        if emo in MEAD_EMOTIONS:
            return emo
    return None


def parse_mead_meta(path: str | Path, fallback_emotion: Optional[str] = None):
    p = Path(path)
    stem = p.stem
    parts = re.split(r"[_\-]+", stem)
    speaker = next((x for x in parts if re.fullmatch(r"[MW]\d{3}", x, re.I)), None)
    level = None
    utterance = None
    m = re.search(r"level[_-]?(\d+)", stem, re.I)
    if m:
        level = m.group(1)
    else:
        for i, x in enumerate(parts):
            if speaker and x.lower() == speaker.lower() and i + 3 < len(parts):
                if parts[i + 2].isdigit():
                    level = parts[i + 2]
                    utterance = parts[i + 3]
                    break
    if utterance is None:
        m = re.search(r"(?:level[_-]?\d+[_-])?(\d{3,})$", stem, re.I)
        if m:
            utterance = m.group(1)
    emotion = canonical_emotion(fallback_emotion) or infer_emotion(path)
    short = MEAD_SHORT.get(emotion or "")
    if speaker and short and level and utterance:
        return speaker.upper(), short, str(int(level)), utterance
    return None


def canonical_eat_filename(sample: Sample, index: int = 0) -> str:
    meta = parse_mead_meta(sample.gt, sample.emotion)
    if not meta:
        raise ValueError(f"Cannot derive MEAD metadata from GT path: {sample.gt}")
    speaker, emo, level, utterance = meta
    return f"{index:04d}_{speaker}_{emo}_{level}_{utterance}.mp4"


def _validate_sample(s: Sample, require_files: bool = True) -> Sample:
    s = s.normalized()
    if not s.name:
        raise ValueError("sample name is empty")
    if require_files:
        for field in ("fake", "gt"):
            p = Path(getattr(s, field))
            if not p.is_file():
                raise FileNotFoundError(f"{field} not found for {s.name}: {p}")
    return s


def read_manifest(path: str | Path, require_files: bool = True) -> list[Sample]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    text = p.read_text(encoding="utf-8-sig")
    delimiter = "\t" if p.suffix.lower() == ".tsv" else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    required = {"name", "fake", "gt"}
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise ValueError(f"manifest must contain columns {sorted(required)}; got {reader.fieldnames}")
    out: list[Sample] = []
    seen = set()
    for lineno, row in enumerate(reader, start=2):
        if not any((v or "").strip() for v in row.values()):
            continue
        s = Sample(
            name=(row.get("name") or "").strip(),
            fake=(row.get("fake") or "").strip(),
            gt=(row.get("gt") or "").strip(),
            emotion=(row.get("emotion") or "").strip() or None,
            image=(row.get("image") or "").strip() or None,
            audio=(row.get("audio") or "").strip() or None,
        )
        try:
            s = _validate_sample(s, require_files=require_files)
        except Exception as exc:
            raise type(exc)(f"manifest line {lineno}: {exc}") from exc
        if s.name in seen:
            raise ValueError(f"duplicate sample name in manifest: {s.name}")
        seen.add(s.name)
        out.append(s)
    if not out:
        raise ValueError(f"manifest has no usable samples: {p}")
    return out


def write_manifest(path: str | Path, samples: Sequence[Sample]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "fake", "gt", "emotion", "image", "audio"])
        w.writeheader()
        for sample in samples:
            w.writerow(asdict(sample.normalized()))
    return p


def manifest_fingerprint(
    samples: Sequence[Sample], metrics: Iterable[str], context: Any | None = None
) -> str:
    payload = {
        "protocol": PROTOCOL_VERSION,
        "metrics": sorted(set(metrics)),
        "samples": [asdict(s.normalized()) for s in samples],
        "context": context,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def finite_values(values: Iterable[object]) -> list[float]:
    out: list[float] = []
    for v in values:
        if isinstance(v, bool):
            out.append(float(v))
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out.append(f)
    return out


def summarize(values: Iterable[object]) -> dict:
    vals = finite_values(values)
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(vals) / len(vals)
    var = sum((x - mean) ** 2 for x in vals) / len(vals)
    return {"n": len(vals), "mean": mean, "std": math.sqrt(var)}

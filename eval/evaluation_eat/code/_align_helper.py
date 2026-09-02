"""Stage GT videos into the EAT layout.

This helper is invoked as ``python _align_helper.py`` from the ``code/``
subdirectory so the dlib / SyncNet weights that some EAT scripts open
relatively resolve correctly.

Usage::

    python _align_helper.py <src_mp4> <primary_name> <alias1> <alias2> ...

It writes ``<primary_name>.mp4`` as an aligned/cropped 128-px mp4 (the same
transformation ``preprocess.align_crop_vids`` produces) inside the current
working directory, then symlinks the alternative names to it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio

from utils_crop import crop_and_align  # noqa: E402  (must be imported from code/)


def align_one(src: Path, dst_primary: Path) -> bool:
    """Align+src, write as 128-px mp4 at dst_primary, return True on success."""
    reader = imageio.get_reader(str(src))
    fps_meta = reader.get_meta_data().get("fps")
    fps = fps_meta if fps_meta else 25.0
    frames = []
    for idx, fr in enumerate(reader):
        if idx >= 96:
            break
        cropped, ret = crop_and_align(fr)
        if not ret:
            continue
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        frames.append(cropped)
    reader.close()
    if not frames:
        return False
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst_primary), fourcc, float(fps), (128, 128))
    for f in frames:
        writer.write(f)
    writer.release()
    return True


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: _align_helper.py <src_mp4> <primary_name> [alias1] [alias2] ...")
        return 1
    src = Path(sys.argv[1]).resolve()
    primary_name = sys.argv[2]
    aliases = sys.argv[3:]

    primary = Path(primary_name).resolve()
    primary.parent.mkdir(parents=True, exist_ok=True)

    if primary.exists() or primary.is_symlink():
        try:
            primary.unlink()
        except Exception:
            pass

    if not src.exists():
        print(f"NO SRC {src}", flush=True)
        return 1

    ok = align_one(src, primary)
    if ok:
        print(f"ALIGNED {src} -> {primary}", flush=True)
    else:
        # Fall back to a plain symlink when face detection fails for every frame.
        os.symlink(str(src), str(primary))
        print(f"SYMLINK {src} -> {primary}", flush=True)

    for alias in aliases:
        a = Path(alias).resolve()
        a.parent.mkdir(parents=True, exist_ok=True)
        if a.exists() or a.is_symlink():
            try:
                a.unlink()
            except Exception:
                pass
        os.symlink(str(primary), str(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
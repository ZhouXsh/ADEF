"""Shared filename parser used by every EAT-style test script.

Each ``test_*.py`` and ``preprocess.py`` used to inline its own ``name_mode``
if/elif ladder.  This module replaces those ladders with a single robust
implementation.  See ``evaluate.py`` docs for the conventions.
"""
from __future__ import annotations

import os


def split_parts(f: str) -> list[str]:
    return os.path.splitext(os.path.split(f)[1])[0].split('_')


def process_name(f: str, name_mode: int) -> tuple[str, str, str, str]:
    """Return ``(pid, emo, lev, vid)`` from a video filename.

    Modes added/relaxed for MEAD-style flat filenames:

    * ``4`` — originally ``{prefix}_{pid}_{emo}_{lev}_{vid}``.
      Now accepts extra trailing tokens by reading the LAST FOUR underscore-
      separated tokens as ``pid/emo/lev/vid``.
    * ``7`` — MEAD flat ``{pid}_front_{emo}_level_{lev}_{vid}[_extra]…``.
      Locates the literal ``"front"`` token and extracts the surrounding
      elements.
    """
    parts = split_parts(f)

    if name_mode == 0:
        # EAMM (dirs)
        pid, emo_lev_vid = f.split('/')[-2], f.split('/')[-1]
        emo, _, lev, vid = emo_lev_vid.split('_')
        emo = emo[:3]
        return pid, emo, lev, vid

    if name_mode == 1:
        # makeittalk: {pid}_{emo}_{lev}_{vid}_{extra}
        pid, emo, lev, vid, _ = parts[:5]
        return pid, emo, lev, vid

    if name_mode == 2:
        # ATVG gt — strict 4 tokens
        pid, emo, lev, vid = parts[:4]
        return pid, emo, lev, vid

    if name_mode == 3:
        # dash-separated
        pid, emo, lev, vid = f.split('-')
        return pid, emo, lev, vid

    if name_mode == 4:
        # {prefix}_{pid}_{emo}_{lev}_{vid}[_extra…]
        if len(parts) >= 5:
            _, pid, emo, lev, vid = parts[0], *parts[-4:]
            return pid, emo, lev, vid
        _, pid, emo, lev, vid = parts
        return pid, emo, lev, vid

    if name_mode == 5:
        # EAMM (processed vid)
        pid, emo, _, lev, vid = parts[:5]
        emo = emo[:3]
        return pid, emo, lev, vid

    if name_mode == 6:
        # PCAVS — relies on directory layout, not filename.
        pid, emo, lev, vid = f.split('/')[-2].split('_audio_')[1].split('_')
        return pid, emo, lev, vid

    if name_mode == 7:
        # MEAD flat: …{pid}_front_{emo}_level_{lev}_{vid}[_extra…]
        try:
            idx = parts.index('front')
            if 0 < idx and idx + 4 < len(parts):
                pid = parts[idx - 1]
                emo = parts[idx + 1]
                lev = parts[idx + 3]
                vid = parts[idx + 4]
                return pid, emo, lev, vid
        except ValueError:
            pass
        # Fallback: legacy mode-4 behaviour.
        if len(parts) >= 5:
            _, pid, emo, lev, vid = parts[0], *parts[-4:]
            return pid, emo, lev, vid
        if len(parts) == 4:
            pid, emo, lev, vid = parts
            return pid, emo, lev, vid
        return parts[0], 'unk', '1', '000'

    # Unknown mode — fall back to mode-4.
    if len(parts) >= 5:
        _, pid, emo, lev, vid = parts[0], *parts[-4:]
        return pid, emo, lev, vid
    if len(parts) == 4:
        pid, emo, lev, vid = parts
        return pid, emo, lev, vid
    return parts[0], 'unk', '1', '000'


def build_gt_candidates(pid: str, emo: str, lev: str, vid: str, gtname: str = 'evp_gt',
                         eat_root: str = '../talking_head_testing/25fps_video/align_crop') -> list[str]:
    """Return ordered list of candidate GT paths for the given (pid, emo, lev, vid).

    The original EAT scripts only try the strict ``{pid}_{emo}_{lev}_{vid}``
    form.  MEAD-style raw videos use ``{pid}_front_{emo}_level_{lev}_{vid}.mp4``
    so we also try that variant before giving up.
    """
    base = '{root}/{gtname}/{name}.mp4'.format(
        root=eat_root, gtname=gtname, name='{pid}_{emo}_{lev}_{vid}'.format(
            pid=pid, emo=emo, lev=lev, vid=vid,
        ),
    )
    mead = '{root}/{gtname}/{pid}_front_{emo}_level_{lev}_{vid}.mp4'.format(
        root=eat_root, gtname=gtname, pid=pid, emo=emo, lev=lev, vid=vid,
    )
    mead_short = '{root}/{gtname}/{pid}_front_{emo}_{lev}_{vid}.mp4'.format(
        root=eat_root, gtname=gtname, pid=pid, emo=emo, lev=lev, vid=vid,
    )
    # De-duplicate while preserving order.
    seen, ordered = set(), []
    for p in (base, mead, mead_short):
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    return ordered
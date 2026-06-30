# coding: utf-8
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import tempfile
from pathlib import Path

from eval.common.io import iter_video_paths, summarize, write_json


def run_external_template(template: str, video: str, audio: str, checkpoint: str, out_path: str):
    cmd = template.format(video=video, audio=audio or "", checkpoint=checkpoint or "", out=out_path)
    proc = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def parse_metric_file(path: str):
    p = Path(path)
    if not p.exists():
        return {}
    text = p.read_text(encoding="utf-8", errors="ignore")
    try:
        return json.loads(text)
    except Exception:
        pass
    metrics = {}
    for line in text.splitlines():
        low = line.lower()
        parts = line.replace(":", " ").replace(",", " ").split()
        for key in ["lse-d", "lsed", "lse_d", "distance", "lse-c", "lsec", "lse_c", "confidence"]:
            if key in low:
                for token in reversed(parts):
                    try:
                        value = float(token)
                        if key in {"lse-d", "lsed", "lse_d", "distance"}:
                            metrics["lse_d"] = value
                        else:
                            metrics["lse_c"] = value
                        break
                    except ValueError:
                        continue
    return metrics


def evaluate_item(row, args):
    video = row.get("generated") or row.get("video")
    audio = row.get("audio", "") or args.audio
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        metric_path = tmp.name
    if args.external_cmd:
        code, stdout, stderr = run_external_template(args.external_cmd, video, audio, args.syncnet_checkpoint, metric_path)
    else:
        raise RuntimeError(
            "Please provide --external_cmd for your installed SyncNet/Wav2Lip fork. "
            "Example: --external_cmd 'python third_party/Wav2Lip/evaluation/scores_LSE.py --data_root {video} --checkpoint_path {checkpoint} --out {out}'"
        )
    metrics = parse_metric_file(metric_path)
    return {
        "video": video,
        "audio": audio,
        "returncode": code,
        "lse_d": metrics.get("lse_d"),
        "lse_c": metrics.get("lse_c"),
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-1000:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--audio", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--wav2lip_root", type=str, default="", help="kept for documentation; use --external_cmd for exact fork")
    parser.add_argument("--syncnet_checkpoint", type=str, default="")
    parser.add_argument("--external_cmd", type=str, default="", help="command template with {video},{audio},{checkpoint},{out}")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    rows = [evaluate_item(row, args) for row in iter_video_paths(args.video or None, args.manifest or None)]
    summary = {
        "lse_d": summarize(r["lse_d"] for r in rows if r.get("lse_d") is not None),
        "lse_c": summarize(r["lse_c"] for r in rows if r.get("lse_c") is not None),
    }
    write_json({"summary": summary, "items": rows}, args.out)


if __name__ == "__main__":
    main()

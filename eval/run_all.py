# coding: utf-8
"""ADEF 评估指标批量运行脚本。

该脚本只会运行当前环境中依赖可用且输入参数齐全的指标。它主要作为便利入口；
如果需要严格记录实验，建议仍然分别运行每个子指标，并保存各自 JSON 输出。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from eval.common.io import write_json


def run_cmd(name: str, cmd, allow_fail: bool = True):
    print("[eval]", name, " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    result = {"name": name, "returncode": proc.returncode, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(f"{name} failed: {proc.stderr}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--skip_optional", action="store_true", help="跳过需要可选依赖或外部 checkpoint 的指标")
    parser.add_argument("--sync_external_cmd", type=str, default="")
    parser.add_argument("--sync_checkpoint", type=str, default="")
    parser.add_argument("--emotion_hf_model", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_arg = ["--manifest", args.manifest] if args.manifest else ["--video", args.video]

    jobs = [
        ("no_reference_iqa", [sys.executable, "eval/no_reference_iqa/eval_iqa_basic.py", *source_arg, "--out", str(out_dir / "no_reference_iqa.json")]),
        ("mouth_audio_corr", [sys.executable, "eval/mouth_audio_corr/eval_mouth_audio_corr.py", *source_arg, "--out", str(out_dir / "mouth_audio_corr.json")]),
        ("landmark_dynamics", [sys.executable, "eval/landmark_dynamics/eval_landmark_dynamics.py", *source_arg, "--out", str(out_dir / "landmark_dynamics.json")]),
        ("head_pose", [sys.executable, "eval/head_pose/eval_head_pose.py", *source_arg, "--out", str(out_dir / "head_pose.json")]),
    ]

    # LMD 需要 generated/reference 成对输入，因此只在 manifest 模式下自动运行。
    if args.manifest:
        jobs.append(("lmd", [sys.executable, "eval/lmd/eval_lmd.py", "--manifest", args.manifest, "--out", str(out_dir / "lmd.json")]))

    if not args.skip_optional:
        jobs.append(("identity_arcface", [sys.executable, "eval/identity_arcface/eval_identity_arcface.py", *source_arg, "--out", str(out_dir / "identity_arcface.json")]))
        jobs.append(("temporal_lpips", [sys.executable, "eval/temporal_lpips/eval_temporal_lpips.py", *source_arg, "--out", str(out_dir / "temporal_lpips.json")]))
        if args.sync_external_cmd:
            jobs.append(("sync_lse", [sys.executable, "eval/sync_lse/eval_sync_lse.py", *source_arg, "--external_cmd", args.sync_external_cmd, "--syncnet_checkpoint", args.sync_checkpoint, "--out", str(out_dir / "sync_lse.json")]))
        if args.emotion_hf_model:
            jobs.append(("emotion_consistency", [sys.executable, "eval/emotion_consistency/eval_emotion_consistency.py", *source_arg, "--hf_model", args.emotion_hf_model, "--out", str(out_dir / "emotion_consistency.json")]))

    runs = [run_cmd(name, cmd, allow_fail=True) for name, cmd in jobs]
    write_json({"runs": runs}, out_dir / "run_all_status.json")


if __name__ == "__main__":
    main()

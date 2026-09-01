"""Smoke test for evaluate_videos.py — verifies frame extraction + cleanup."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def make_synthetic_video(path, num_frames=15, height=224, width=224, color=(0, 0, 0)):
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = list(color)
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


def main():
    tmp = tempfile.mkdtemp(prefix="fvd_smoke3_")
    real_dir = os.path.join(tmp, "real")
    fake_dir = os.path.join(tmp, "fake")
    os.makedirs(real_dir)
    os.makedirs(fake_dir)
    print("[smoke] Generating synthetic videos in", tmp)
    rng = np.random.RandomState(7)
    for i in range(16):  # exact I3D batch size for simplicity
        cr = tuple(int(x) for x in rng.randint(0, 256, size=3))
        cf = tuple(int(min(255, max(0, c + rng.randint(-20, 20)))) for c in cr)
        make_synthetic_video(os.path.join(real_dir, "clip_%03d.mp4" % i), color=cr)
        make_synthetic_video(os.path.join(fake_dir, "clip_%03d.mp4" % i), color=cf)

    # Test 1: default behaviour — temp work_dir is cleaned up at exit.
    out_path = os.path.join(tmp, "result_default.json")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_videos.py")
    cmd = [sys.executable, script,
           "--real_dir", real_dir,
           "--fake_dir", fake_dir,
           "--video_length", "15",
           "--output_file", out_path,
           "--quiet"]
    print("\n[smoke] Run 1: default (auto-cleanup)")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("[smoke] STDOUT tail:")
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print("[smoke] STDERR:")
        print(proc.stderr[-2000:])
        sys.exit(proc.returncode)
    if os.path.exists(out_path):
        result = json.load(open(out_path))
        print("[smoke] FVD result:", json.dumps(result, indent=2))
        assert "work_dir" in result
        assert result["work_dir_kept"] is False
        # Verify cleanup happened.
        if os.path.isdir(result["work_dir"]):
            print("[smoke] WARNING: work_dir still exists after exit:", result["work_dir"])
            shutil.rmtree(result["work_dir"])
        else:
            print("[smoke] OK — temp work_dir was cleaned up")

    # Test 2: --keep_frames — frames remain after exit.
    out_path2 = os.path.join(tmp, "result_keep.json")
    work_dir = os.path.join(tmp, "kept_frames")
    cmd2 = list(cmd)
    cmd2[cmd2.index("--output_file") + 1] = out_path2
    cmd2 += ["--work_dir", work_dir, "--keep_frames"]
    print("\n[smoke] Run 2: --keep_frames --work_dir")
    proc = subprocess.run(cmd2, capture_output=True, text=True)
    print("[smoke] STDOUT tail:")
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print("[smoke] STDERR:")
        print(proc.stderr[-2000:])
        sys.exit(proc.returncode)
    if os.path.exists(out_path2):
        result = json.load(open(out_path2))
        print("[smoke] FVD result:", json.dumps(result, indent=2))
        assert result["work_dir_kept"] is True
        assert os.path.isdir(work_dir), "keep_frames mode should leave dir intact"
        # Spot-check: list a few extracted frames.
        pair_subdirs = sorted(os.listdir(work_dir))
        print("[smoke] Pair subdirs:", pair_subdirs[:3])
        first_pair = os.path.join(work_dir, pair_subdirs[0])
        if os.path.isdir(first_pair):
            real_frames = sorted(os.listdir(os.path.join(first_pair, "real"))) if os.path.isdir(os.path.join(first_pair, "real")) else []
            fake_frames = sorted(os.listdir(os.path.join(first_pair, "fake"))) if os.path.isdir(os.path.join(first_pair, "fake")) else []
            print("[smoke] First pair real frames (first 3):", real_frames[:3])
            print("[smoke] First pair fake frames (first 3):", fake_frames[:3])
            print("[smoke] Frame count real/fake: %d / %d"
                  % (len(real_frames), len(fake_frames)))

    shutil.rmtree(tmp)
    print("\n[smoke] All assertions passed.")


if __name__ == "__main__":
    main()
"""Generate synthetic videos and run the FVD evaluation end-to-end."""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def make_synthetic_video(path, num_frames=15, height=224, width=224, color=(0, 0, 0)):
    """Write a synthetic MP4 with a constant colour frame stream via OpenCV."""
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 25.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not open VideoWriter for %s" % path)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    for _ in range(num_frames):
        writer.write(frame)
    writer.release()


def main():
    tmp = tempfile.mkdtemp(prefix="fvd_smoke_")
    real_dir = os.path.join(tmp, "real")
    fake_dir = os.path.join(tmp, "fake")
    os.makedirs(real_dir)
    os.makedirs(fake_dir)

    print("[smoke] Generating synthetic videos in", tmp)
    rng = np.random.RandomState(0)
    for i in range(32):
        # 'real' videos: bright colours
        cr = tuple(int(x) for x in rng.randint(0, 256, size=3))
        cf = tuple(int(x) for x in rng.randint(0, 256, size=3))
        make_synthetic_video(os.path.join(real_dir, "clip_%03d.mp4" % i), color=cr)
        # 'fake' videos: close to real (low FVD expected)
        cf2 = tuple(int(min(255, c + rng.randint(-15, 15))) for c in cr)
        make_synthetic_video(os.path.join(fake_dir, "clip_%03d.mp4" % i), color=cf2)

    out_path = os.path.join(tmp, "result.json")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_adef.py")
    cmd = [sys.executable, script,
           "--real_dir", real_dir,
           "--fake_dir", fake_dir,
           "--video_length", "15",
           "--output_file", out_path,
           "--quiet"]
    print("[smoke] Running:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print("[smoke] STDOUT:\n" + proc.stdout[-2000:])
    if proc.returncode != 0:
        print("[smoke] STDERR:\n" + proc.stderr[-2000:])
        sys.exit(proc.returncode)
    if os.path.exists(out_path):
        print("[smoke] Result file contents:")
        print(json.dumps(json.load(open(out_path)), indent=2))
    shutil.rmtree(tmp)
    print("[smoke] OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
#
# Bootstrap a self-contained conda environment for ADEF FVD evaluation.
#
# Usage:
#   bash setup_fvd_env.sh
#
# Creates (or updates) the `fvd` conda env, installs TensorFlow + TF Hub +
# TF-GAN + six + a small set of video I/O libraries used by the FVD
# evaluation script, and prefetches the I3D model from tensorflow_hub so the
# network only needs to be reachable on this first run.
set -euo pipefail

ENV_NAME=${ENV_NAME:-fvd}
PY_VERSION=${PY_VERSION:-3.10.16}
TFHUB_CACHE_DIR=${TFHUB_CACHE_DIR:-/home/Zhouxishi/tfhub_cache}

echo "[setup] Conda env: $ENV_NAME (Python $PY_VERSION)"

# Reuse the existing env if present, otherwise create it.
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[setup] Reusing existing env $ENV_NAME"
else
  conda create -n "$ENV_NAME" python="$PY_VERSION" -y
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[setup] Upgrading pip / setuptools"
pip install --upgrade "pip<25" "setuptools<81"

echo "[setup] Installing TensorFlow + TF-Hub + TF-GAN"
pip install --upgrade \
  "tensorflow==2.13.0" \
  "tensorflow-hub==0.13.0" \
  "tensorflow-gan==2.1.0" \
  "tensorflow-probability==0.20.1" \
  "tensorflow-estimator==2.13.0" \
  "six==1.16.0"

echo "[setup] Installing video I/O dependencies"
pip install --upgrade \
  "decord==0.6.0" \
  "av==12.1.0" \
  "imageio==2.34.2" \
  "imageio-ffmpeg==0.5.1" \
  "opencv-python==4.10.0.84" \
  "pillow==10.3.0" \
  "tqdm==4.66.4"

echo "[setup] Prefetching I3D model into $TFHUB_CACHE_DIR"
mkdir -p "$TFHUB_CACHE_DIR"
TFHUB_CACHE_DIR="$TFHUB_CACHE_DIR" python - <<'PY'
import os
os.environ.setdefault("TFHUB_CACHE_DIR", os.environ.get("TFHUB_CACHE_DIR", "/home/Zhouxishi/tfhub_cache"))
import tensorflow_hub as hub
hub.resolve("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
print("[setup] I3D-Kinetics-400 cached at", os.environ["TFHUB_CACHE_DIR"])
PY

echo "[setup] Sanity check:"
python -c "import tensorflow as tf, tensorflow_hub, tensorflow_gan, six; print('TF', tf.__version__)"
echo "[setup] Done. Activate with: conda activate $ENV_NAME"

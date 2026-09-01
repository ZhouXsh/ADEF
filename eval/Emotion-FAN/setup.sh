#!/usr/bin/env bash
# =============================================================================
# setup.sh — Create the Emotion-FAN conda environment and install deps.
#
# Usage:
#     bash setup.sh
#
# Notes:
#   * The original Emotion-FAN repo pins `torch==1.3 / torchvision==0.4` (Aug
#     2019). Those wheels require Python 3.7 and an old CUDA runtime that no
#     longer plays well with RTX 4090 / driver 590.  We instead create a
#     Python 3.9 environment with a recent CPU-only PyTorch (clean inference
#     path, no CUDA needed for evaluation).
#   * If you do have GPU access and want to use the CUDA build, edit the
#     `PIP_INDEX` line below to point at the right CUDA channel, e.g.
#         PIP_INDEX=https://download.pytorch.org/whl/cu121
# =============================================================================
set -e

ENV_NAME=${ENV_NAME:-emotion_fan}
PY_VERSION=${PY_VERSION:-3.9}
PIP_INDEX=${PIP_INDEX:-https://download.pytorch.org/whl/cpu}
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[setup] environment: $ENV_NAME  python: $PY_VERSION"
conda create -n "$ENV_NAME" "python=$PY_VERSION" -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[setup] installing numpy / pillow / opencv-python / scipy"
pip install -i https://pypi.org/simple \
    "numpy" \
    "pillow" \
    "opencv-python" \
    "scipy"

echo "[setup] installing torch (CPU) from $PIP_INDEX"
pip install -i "$PIP_INDEX" "torch" "torchvision"

echo "[setup] verifying environment"
python - <<'PY'
import torch, torchvision, PIL, cv2, numpy, scipy
print(f"  torch       : {torch.__version__}")
print(f"  torchvision : {torchvision.__version__}")
print(f"  pillow      : {PIL.__version__}")
print(f"  opencv      : {cv2.__version__}")
print(f"  numpy       : {numpy.__version__}")
print(f"  scipy       : {scipy.__version__}")
print(f"  cuda avail. : {torch.cuda.is_available()}")
PY

echo
echo "[setup] done. Activate with:  conda activate $ENV_NAME"
echo "[setup] Next step — download the FER+ pretrained backbone into:"
echo "         $HERE/pretrain_model/Resnet18_FER+_pytorch.pth.tar"

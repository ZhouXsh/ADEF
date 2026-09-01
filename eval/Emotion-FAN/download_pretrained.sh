#!/usr/bin/env bash
# =============================================================================
# download_pretrained.sh
#
# Downloads the Emotion-FAN backbone weights into ./pretrain_model/.
#
# The authors' pretrained FER+ ResNet-18 lives on Baidu / OneDrive only, both
# of which require manual authentication from a browser.  This script:
#
#   * Tells you exactly where to download it manually
#       (./pretrain_model/Resnet18_FER+_pytorch.pth.tar)
#   * Falls back to torchvision's ImageNet-pretrained ResNet-18 so that the
#     evaluation script can run end-to-end without waiting.  Conv layers are
#     compatible; only the fc head differs and is deliberately skipped by the
#     loader in `evaluate_emotion_fan.py:load_model`.
# =============================================================================
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
PRETRAIN_DIR="$HERE/pretrain_model"
mkdir -p "$PRETRAIN_DIR"

FER_TARBALL="$PRETRAIN_DIR/Resnet18_FER+_pytorch.pth.tar"
IMAGENET_TARBALL="$PRETRAIN_DIR/Resnet18_ImageNet_pytorch.pth.tar"

if [[ -f "$FER_TARBALL" ]]; then
    echo "[download_pretrained] FER+ weights already present at $FER_TARBALL"
else
    echo "[download_pretrained] ----------------------------------------------------"
    echo "[download_pretrained] MANUAL STEP REQUIRED for the official FER+ backbone."
    echo "[download_pretrained] ----------------------------------------------------"
    echo "  1. Open one of the following URLs in your browser:"
    echo "       https://pan.baidu.com/s/1OgxPSSzUhaC9mPltIpp2pg"
    echo "       https://1drv.ms/u/s!AhGc2vUv7IQtl1Pt7FhPXr_Kofd5?e=3MvPFX"
    echo "  2. Download 'Resnet18_FER+_pytorch.pth.tar'."
    echo "  3. Save it to:"
    echo "       $FER_TARBALL"
    echo
    echo "  Reference: pretrain_model/readme.md"
fi

if [[ ! -f "$IMAGENET_TARBALL" ]]; then
    echo "[download_pretrained] fetching torchvision ImageNet ResNet-18 as fallback"
    python - <<'PY'
import torch, torchvision.models as tvm
m = tvm.resnet18(weights='IMAGENET1K_V1')
torch.save({'state_dict': m.state_dict()}, "$IMAGENET_TARBALL")
print('Saved to $IMAGENET_TARBALL')
PY
else
    echo "[download_pretrained] ImageNet fallback already present at $IMAGENET_TARBALL"
fi

ls -lh "$PRETRAIN_DIR"

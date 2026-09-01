"""
Emotion-FAN inference as a callable evaluation function
=======================================================

This module wraps the Frame Attention Network (Meng et al., ICIP 2019) so it
can be used as a drop-in emotion classifier for arbitrary talking-head videos,
exactly like any other metric in the ADEF eval/ folder.

Typical use
-----------
    from evaluate_emotion_fan import EmotionFANPredictor

    fan = EmotionFANPredictor(
        pretrain_fer='./pretrain_model/Resnet18_FER+_pytorch.pth.tar',
        checkpoint='./model/self_relation-attention_<epoch>_<acc>.pth',
        at_type=1,
        device='cuda:0',
    )

    # 1) single video
    pred = fan.predict('/path/to/video.mp4')          # or a folder of frames
    print(pred.emotion, pred.probabilities)

    # 2) batch over a directory of generated videos
    results = fan.predict_batch('/path/to/eval_videos/', pattern='*.mp4')
    # results is a dict {video_stem: Prediction}

    # 3) plain CLI
    python evaluate_emotion_fan.py --input /path/to/video.mp4

The predictor auto-detects whether the input is an mp4 (decoded with OpenCV)
or a folder of image frames.  Outputs follow the 7-class AFEW taxonomy:
    0:Happy, 1:Angry, 2:Disgust, 3:Fear, 4:Sad, 5:Neutral, 6:Surprise
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

# Make the upstream `basic_code/` package importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from basic_code import networks as fan_nets  # noqa: E402


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------
# 7-class AFEW taxonomy used by Emotion-FAN's AFEW experiments.
AFEW_ID2NAME: Dict[int, str] = {
    0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear',
    4: 'Sad', 5: 'Neutral', 6: 'Surprise',
}
AFEW_NAME2ID: Dict[str, int] = {v: k for k, v in AFEW_ID2NAME.items()}

# 8-class FER+ taxonomy used by the original FER+ pretrained checkpoint.
# Order matches Microsoft's canonical FERPlus repo
# (https://github.com/microsoft/FERPlus):
#     neutral=0, happiness=1, surprise=2, sadness=3,
#     anger=4, disgust=5, fear=6, contempt=7.
FER_ID2NAME: Dict[int, str] = {
    0: 'Neutral', 1: 'Happy', 2: 'Surprise', 3: 'Sad',
    4: 'Angry', 5: 'Disgust', 6: 'Fear', 7: 'Contempt',
}
# Re-index `logits_8` (FER+ order, length 8) into AFEW order (length 7),
# dropping Contempt along the way.  Entry i holds the FER+ column index that
# should land in AFEW slot i.
AFEW_FROM_FER: List[int] = [
    FER_ID2NAME[i] for i in range(8) if FER_ID2NAME[i] != 'Contempt'
]
# Inverse: `AFEW_TO_FER[afew_id] = fer_id`.
AFEW_TO_FER: Dict[int, int] = {
    AFEW_NAME2ID[name]: fer_id for fer_id, name in FER_ID2NAME.items()
    if name != 'Contempt'
}


@dataclass
class Prediction:
    """Single-video output of the emotion classifier."""
    source: str                     # path that produced this prediction
    emotion: str                    # predicted class name
    emotion_id: int                 # 0..6
    confidence: float               # softmax probability of the predicted class
    probabilities: Dict[str, float] = field(default_factory=dict)
    n_frames_used: int = 0
    elapsed_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def _iter_video_frames(video_path: str,
                       max_frames: Optional[int] = None,
                       frame_stride: int = 1) -> Iterable[np.ndarray]:
    """Yield BGR frames from an mp4 / avi / etc.

    `frame_stride=N` returns every Nth frame; `max_frames` caps the total.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise IOError(f'Could not open video: {video_path}')
    idx = 0
    yielded = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if idx % frame_stride == 0:
            yield cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
        idx += 1
    cap.release()


def _iter_dir_frames(frame_dir: str) -> Iterable[np.ndarray]:
    """Yield RGB arrays for every image inside `frame_dir` (sorted)."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = [os.path.join(frame_dir, f) for f in sorted(os.listdir(frame_dir))
             if f.lower().endswith(exts)]
    for fp in files:
        img = Image.open(fp).convert('RGB')
        yield np.asarray(img)


def _is_video_file(path: str) -> bool:
    return os.path.isfile(path) and path.lower().endswith(
        ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv'))


# ---------------------------------------------------------------------------
# Model loader (mirrors upstream's `load.model_parameters` logic)
# ---------------------------------------------------------------------------
def _safe_load_state(model: torch.nn.Module, ckpt_path: str,
                     strict: bool = False, prefix: str = 'module.') -> None:
    """Load a state-dict into `model`, stripping `prefix` (e.g. `module.`) and
    skipping keys whose shape doesn't match (e.g. a 1000-class fc head).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model_state = model.state_dict()
    new_state = {}
    skipped = []
    for k, v in ckpt.items():
        short = k.replace(prefix, '')
        if short in model_state and model_state[short].shape == v.shape:
            new_state[short] = v
        else:
            skipped.append((k, short, getattr(model_state.get(short, None),
                                             'shape', None)))
    missing, unexpected = model.load_state_dict(new_state, strict=strict)
    if missing:
        print(f'[load_model] missing keys: {len(missing)} (showing 5) '
              f'{missing[:5]}')
    if unexpected:
        print(f'[load_model] unexpected keys: {len(unexpected)} '
              f'(showing 5) {unexpected[:5]}')


def _build_model(at_type: int, num_classes: int = 7) -> torch.nn.Module:
    """at_type: -1 baseline, 0 self-attn, 1 self+relation-attn.

    `num_classes` only matters for the baseline (`at_type == -1`) path —
    pass 8 if you intend to load a 8-class FER+ fc head.
    """
    if at_type == -1:
        return tv_models.resnet18(num_classes=num_classes)
    at_name = ['self-attention', 'self_relation-attention'][at_type]
    return fan_nets.resnet18_at(at_type=at_name)


def _fer_fc_dims(ckpt_path: str) -> Optional[int]:
    """Return the FER+ `fc` output dimension from a checkpoint, or None."""
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt = ckpt['state_dict']
        for k in ('fc.weight', 'module.fc.weight'):
            if k in ckpt:
                return int(ckpt[k].shape[0])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public predictor
# ---------------------------------------------------------------------------
class EmotionFANPredictor:
    """Stateless wrapper around the FAN model that turns a video into an
    emotion label.

    Parameters
    ----------
    pretrain_fer : str
        Path to the FER+ ResNet-18 backbone (.pth.tar). This populates the
        convolutional features before any fine-tuning overrides them.
    checkpoint : str
        Optional fine-tuned checkpoint. If given, its weights override the
        FER+ ones (useful when evaluating a model trained on AFEW/CK+).
    at_type : {-1, 0, 1}
        -1: baseline ResNet-18 + soft-vote
         0: ResNet_AT + self-attention
         1: ResNet_AT + self+relation attention
    device : str
        'cuda:0' / 'cpu'. Falls back to CPU if CUDA is unavailable.
    max_frames : int | None
        Optional cap on the number of frames read per video. Default: no cap.
    frame_stride : int
        Read every Nth frame (default: every frame).
    batch_size : int
        Forward batch size for the per-frame ResNet pass.
    """

    def __init__(self,
                 pretrain_fer: str,
                 checkpoint: str = '',
                 at_type: int = 1,
                 device: Optional[str] = None,
                 max_frames: Optional[int] = None,
                 frame_stride: int = 1,
                 batch_size: int = 64) -> None:
        if at_type not in (-1, 0, 1):
            raise ValueError(f'at_type must be -1, 0 or 1, got {at_type}')
        self.at_type = at_type
        # Pick a device that the current torch build actually supports.
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        requested = torch.device(device)
        if requested.type == 'cuda' and not torch.cuda.is_available():
            print(f'[EmotionFANPredictor] {device} requested but this '
                  f'torch build has no CUDA; falling back to CPU.')
            self.device = torch.device('cpu')
        else:
            self.device = requested
        self.max_frames = max_frames
        self.frame_stride = frame_stride
        self.batch_size = batch_size
        self._post_aggregate_hook = None

        # Resolve the backbone weights: prefer `pretrain_fer` if it exists,
        # otherwise transparently fall back to the bundled ImageNet ResNet-18
        # so the evaluator is still runnable.  A warning is printed in either
        # case so the user knows which weights are actually being used.
        _FER_NAMES = ('Resnet18_FER+_pytorch.pth.tar',
                      'Resnet18_FER+_pytorch.pth')  # also accept .pth (no .tar)
        backbone_path = None
        candidates = [pretrain_fer]
        candidates += [os.path.join(_THIS_DIR, 'pretrain_model', n)
                       for n in _FER_NAMES]
        candidates.append(os.path.join(_THIS_DIR, 'pretrain_model',
                                       'Resnet18_ImageNet_pytorch.pth.tar'))
        for cand in candidates:
            if cand and os.path.isfile(cand):
                backbone_path = cand
                break
        if backbone_path is None:
            raise FileNotFoundError(
                f'No backbone weights found. Tried: '
                f'`{pretrain_fer}` and the bundled fallback '
                f'`{os.path.join(_THIS_DIR, "pretrain_model")}/*.pth*`. '
                f'Run `bash download_pretrained.sh` first.')
        if backbone_path != pretrain_fer:
            print(f'[EmotionFANPredictor] {pretrain_fer} not found; '
                  f'falling back to {backbone_path}.')

        # Detect how many classes the FER+ backbone was trained on (8 for the
        # canonical FERPlus taxonomy with Contempt, 7 if someone already
        # trimmed it).  For the baseline path (`at_type == -1`) we rebuild
        # the ResNet with the matching `fc` width so we can absorb the FER+
        # classifier directly instead of leaving it as random noise.
        fer_nc = _fer_fc_dims(backbone_path)
        if at_type == -1 and fer_nc is not None and fer_nc != 7:
            print(f'[EmotionFANPredictor] FER+ backbone has '
                  f'{fer_nc}-class fc (FERPlus w/ Contempt); using it as the '
                  f'classifier and remapping logits to 7-class AFEW.')
            model = _build_model(at_type, num_classes=fer_nc)
            self._post_aggregate_hook = self._remap_fer_to_afew
        else:
            model = _build_model(at_type)
            self._post_aggregate_hook = None

        _safe_load_state(model, backbone_path)
        if checkpoint and os.path.isfile(checkpoint):
            _safe_load_state(model, checkpoint)
            self._post_aggregate_hook = None  # checkpoint already targets 7 classes
        elif checkpoint:
            print(f'[EmotionFANPredictor] checkpoint {checkpoint} not found, '
                  f'using backbone weights only.')
        model = model.to(self.device).eval()
        # NOTE: we intentionally do NOT wrap with DataParallel here. The
        # upstream ResNet_AT.forward() has the unusual signature
        # `forward(self, x='', phrase='train', AT_level='first_level', ...)`
        # which trips DataParallel's tensor-scatter logic and produces
        # `conv2d() got (str, Tensor, ...)` errors. Single-GPU inference
        # is also the expected deployment scenario for evaluation.
        self.model = model

    # ------------------------------------------------------------------
    # Frame-level forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _forward_frames(self, frames_rgb: List[np.ndarray]) -> Tuple[
            torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run the first-level backbone on a list of RGB frames.

        Returns
        -------
        baseline_logits : [N, 7]    (only when at_type == -1)
        feats           : [N, 512]  or [N, 1024]  (only when at_type != -1)
        alphas          : [N, 1]    (only when at_type != -1)
        """
        if not frames_rgb:
            raise ValueError('No frames to evaluate.')

        if self.at_type == -1:
            logits = []
            for i in range(0, len(frames_rgb), self.batch_size):
                batch = frames_rgb[i:i + self.batch_size]
                tensors = torch.stack([_TRANSFORM(Image.fromarray(f))
                                       for f in batch]).to(self.device)
                logits.append(self.model(tensors))
            # NOTE: do NOT remap here; the remap is done once on the aggregated
            # (mean-pooled) video logit inside _aggregate() to avoid a second
            # index_select on already-7-dim tensors.
            return torch.cat(logits, dim=0), None, None

        feats, alphas = [], []
        for i in range(0, len(frames_rgb), self.batch_size):
            batch = frames_rgb[i:i + self.batch_size]
            tensors = torch.stack([_TRANSFORM(Image.fromarray(f))
                                   for f in batch]).to(self.device)
            f, a = self.model(tensors, phrase='eval')
            feats.append(f)
            alphas.append(a)
        return None, torch.cat(feats, dim=0), torch.cat(alphas, dim=0)

    # ------------------------------------------------------------------
    # Frame -> video aggregation
    # ------------------------------------------------------------------
    def _aggregate(self,
                   baseline_logits: Optional[torch.Tensor],
                   feats: Optional[torch.Tensor],
                   alphas: Optional[torch.Tensor],
                   n_frames: int = 1) -> torch.Tensor:
        """Aggregate per-frame outputs into a single 7-way logit vector.

        For the FAN model the upstream `val()` runs the alpha-weighted mean of
        frame features back through the second-level prediction head
        (`pred_fc2` for self+relation attention, `pred_fc1` for self only).
        We replicate that here so the output dimensions match the 7-class
        taxonomy (not 512/1024 raw features).
        """
        if self.at_type == -1:
            video_logit = baseline_logits.mean(dim=0)        # soft-vote
        else:
            # Build a fake single-row index matrix of shape [1, N] (one video
            # with N frames) — this is the upstream `index_matrix` collapsed
            # to the single-video case.
            device = feats.device
            N = feats.size(0)
            index_matrix = torch.ones(1, N, device=device)
            weighted = feats.mul(alphas)
            sum_alpha = index_matrix.mm(alphas)                       # [1, 1]
            vm = index_matrix.mm(weighted).div(sum_alpha + 1e-8)      # [1, C]

            # DataParallel wraps the model — unwrap for the keyword arg call.
            inner = self.model.module if isinstance(self.model,
                                                    torch.nn.DataParallel) \
                else self.model
            with torch.no_grad():
                if self.at_type == 0:    # self-attention: pred_fc1
                    logits = inner(vm=vm, phrase='eval', AT_level='pred')
                else:                    # self+relation: pred_fc2
                    logits = inner(vectors=feats, vm=vm, alphas_from1=alphas,
                                   index_matrix=index_matrix, phrase='eval',
                                   AT_level='second_level')
            video_logit = logits.squeeze(0)

        # If the backbone is the 8-class FERPlus fc, reorder+drop the 8th
        # logit so the caller sees a 7-class AFEW tensor.
        if self._post_aggregate_hook is not None:
            video_logit = self._post_aggregate_hook(video_logit)
        return video_logit

    @staticmethod
    def _remap_fer_to_afew(logits: torch.Tensor) -> torch.Tensor:
        """Reorder a FERPlus-8 logit vector into AFEW-7 (drop Contempt)."""
        idx = torch.as_tensor([AFEW_TO_FER[i] for i in range(7)],
                              device=logits.device, dtype=torch.long)
        return logits.index_select(0, idx)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict(self, source: Union[str, os.PathLike],
                max_frames: Optional[int] = None,
                frame_stride: Optional[int] = None) -> Prediction:
        """Run inference on a single video file OR a directory of frames.

        Parameters
        ----------
        source : path to an mp4 OR to a folder of jpg/png frames.
        max_frames, frame_stride : optional overrides of the constructor
            defaults.
        """
        source = str(source)
        max_frames = self.max_frames if max_frames is None else max_frames
        frame_stride = self.frame_stride if frame_stride is None else frame_stride

        t0 = time.time()
        if _is_video_file(source):
            frames = list(_iter_video_frames(source, max_frames, frame_stride))
        elif os.path.isdir(source):
            frames = list(_iter_dir_frames(source))
            if max_frames is not None:
                frames = frames[:max_frames]
        else:
            raise FileNotFoundError(source)
        if not frames:
            raise RuntimeError(f'No frames could be read from {source}')

        base_logits, feats, alphas = self._forward_frames(frames)
        video_logit = self._aggregate(base_logits, feats, alphas,
                                      n_frames=len(frames))
        probs = F.softmax(video_logit, dim=0)
        conf, idx = probs.max(dim=0)
        elapsed = time.time() - t0

        return Prediction(
            source=source,
            emotion=AFEW_ID2NAME[int(idx.item())],
            emotion_id=int(idx.item()),
            confidence=float(conf.item()),
            probabilities={AFEW_ID2NAME[i]: float(probs[i].item())
                           for i in range(len(probs))},
            n_frames_used=len(frames),
            elapsed_sec=elapsed,
        )

    def predict_batch(self, root: str, pattern: str = '*.mp4',
                      recursive: bool = True,
                      progress: bool = True) -> Dict[str, Prediction]:
        """Run `predict` on every video under `root` matching `pattern`.

        Returns
        -------
        dict[stem -> Prediction]
        """
        if recursive:
            files = sorted(glob.glob(os.path.join(root, '**', pattern),
                                     recursive=True))
        else:
            files = sorted(glob.glob(os.path.join(root, pattern)))
        if not files:
            raise FileNotFoundError(
                f'No files matching {pattern} under {root}')
        results: Dict[str, Prediction] = {}
        for i, fp in enumerate(files):
            try:
                results[os.path.splitext(os.path.basename(fp))[0]] = \
                    self.predict(fp)
            except Exception as exc:
                print(f'  [skip] {fp}: {exc}')
            if progress and (i + 1) % 10 == 0:
                print(f'  ...processed {i + 1}/{len(files)} videos')
        return results


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------
def _print_prediction(p: Prediction) -> None:
    print('\n===== Emotion-FAN prediction =====')
    print(f'  source        : {p.source}')
    print(f'  emotion       : {p.emotion}  (id={p.emotion_id}, '
          f'p={p.confidence:.4f})')
    print(f'  frames used   : {p.n_frames_used}')
    print(f'  elapsed       : {p.elapsed_sec:.3f} s')
    print('  probabilities :')
    for name in sorted(p.probabilities,
                       key=lambda k: p.probabilities[k], reverse=True):
        print(f'      {name:<10s} {p.probabilities[name]:.4f}')
    print('=================================\n')


def _cli() -> None:
    p = argparse.ArgumentParser(
        description='Run Emotion-FAN as an inference function on a single '
                    'video or a directory of videos.')
    p.add_argument('--pretrain_fer',
                   default=os.path.join(_THIS_DIR,
                                        'pretrain_model',
                                        'Resnet18_FER+_pytorch.pth.tar'),
                   help='Path to FER+ backbone weights.')
    p.add_argument('--checkpoint', default='',
                   help='Optional fine-tuned checkpoint to override the '
                        'FER+ backbone weights.')
    p.add_argument('--at_type', type=int, default=1,
                   help='-1 baseline / 0 self-attn / 1 self+relation.')
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--input', required=True,
                   help='Path to a single .mp4 video OR a folder of frames '
                        'OR a folder of .mp4 files (when --mode batch).')
    p.add_argument('--mode', choices=['single', 'batch'], default='single',
                   help='single: predict on one video. batch: predict on '
                        'every video under --input matching --pattern.')
    p.add_argument('--pattern', default='*.mp4',
                   help='Glob pattern when --mode=batch.')
    p.add_argument('--out_json', default='',
                   help='Optional path to dump predictions as JSON.')
    p.add_argument('--max_frames', type=int, default=0,
                   help='If >0, cap frames per video.')
    p.add_argument('--frame_stride', type=int, default=1,
                   help='Read every Nth frame.')
    p.add_argument('--batch_size', type=int, default=64)
    args = p.parse_args()

    fan = EmotionFANPredictor(
        pretrain_fer=args.pretrain_fer,
        checkpoint=args.checkpoint,
        at_type=args.at_type,
        device=args.device,
        max_frames=args.max_frames or None,
        frame_stride=args.frame_stride,
        batch_size=args.batch_size,
    )

    if args.mode == 'single':
        predictions = [fan.predict(args.input)]
    else:
        predictions = list(fan.predict_batch(args.input,
                                             pattern=args.pattern).values())

    for pred in predictions:
        _print_prediction(pred)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or '.',
                    exist_ok=True)
        payload = {p.source: p.to_dict() for p in predictions}
        with open(args.out_json, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f'[evaluate] wrote {args.out_json}')


if __name__ == '__main__':
    _cli()
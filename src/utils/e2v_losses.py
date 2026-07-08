"""Loss helpers for emotion2vec-guided emotional motion generation.

These helpers are intentionally placed in a new file instead of modifying
``src/utils/common.py``.  They can be imported by the emotion2vec training copy
without changing the original ADEF training path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _valid_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None, dim: int = 1, keepdim: bool = True) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    mask_f = mask.float()
    while mask_f.ndim < x.ndim:
        mask_f = mask_f.unsqueeze(-1)
    denom = mask_f.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return (x * mask_f).sum(dim=dim, keepdim=keepdim) / denom


def zscore_curve(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """Normalize a temporal curve per sample.

    Args:
        x: [B, L] or [B, L, 1].
        mask: optional [B, L] bool tensor. True means valid frame.
    """
    if x.ndim == 3:
        x = x.squeeze(-1)
    mean = _valid_mean(x, mask=mask, dim=1, keepdim=True)
    if mask is None:
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
    else:
        mask_f = mask.float()
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        var = ((x - mean) ** 2 * mask_f).sum(dim=1, keepdim=True) / denom
    return (x - mean) / torch.sqrt(var + eps)


def compute_motion_intensity_curve(
    motion_coef: torch.Tensor,
    rot_repr: str = "aa",
    use_velocity: bool = True,
    vel_weight: float = 0.5,
) -> torch.Tensor:
    """Build a visual affect-intensity curve from generated motion.

    The current ADEF motion layout is [exp(63), pose(7)] for rot_repr='aa'.
    We estimate affect intensity by expression amplitude and optional expression
    velocity.  This deliberately avoids rendering frames during training.
    """
    if rot_repr == "aa":
        exp = motion_coef[..., :63]
    elif rot_repr == "emo":
        exp = torch.cat([motion_coef[..., :63], motion_coef[..., -3:]], dim=-1)
    else:
        raise ValueError(f"Unknown rotation representation {rot_repr}")

    exp_center = exp.mean(dim=1, keepdim=True).detach()
    amp = torch.norm(exp - exp_center, p=2, dim=-1)
    if not use_velocity:
        return amp

    vel = torch.zeros_like(amp)
    vel[:, 1:] = torch.norm(exp[:, 1:] - exp[:, :-1], p=2, dim=-1)
    return amp + float(vel_weight) * vel


def compute_e2v_intensity_curve(
    emo_frame_feat: torch.Tensor,
    use_velocity: bool = True,
    vel_weight: float = 0.5,
) -> torch.Tensor:
    """Build an audio affect/prosody curve from emotion2vec frame features.

    Args:
        emo_frame_feat: [B, L, D_e2v], already aligned to motion length L.
    """
    center = emo_frame_feat.mean(dim=1, keepdim=True).detach()
    amp = torch.norm(emo_frame_feat - center, p=2, dim=-1)
    if not use_velocity:
        return amp

    vel = torch.zeros_like(amp)
    vel[:, 1:] = torch.norm(emo_frame_feat[:, 1:] - emo_frame_feat[:, :-1], p=2, dim=-1)
    return amp + float(vel_weight) * vel


def compute_prosody_curve_loss(
    motion_pred: torch.Tensor,
    emo_frame_feat: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    rot_repr: str = "aa",
    use_velocity: bool = True,
    vel_weight_motion: float = 0.5,
    vel_weight_audio: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Temporal correlation loss between audio affect and visual motion affect.

    This loss compares the *trend* of two curves rather than their absolute
    magnitudes, which is safer because emotion2vec feature norms and expression
    coefficient norms live in different spaces.
    """
    visual_curve = compute_motion_intensity_curve(
        motion_pred,
        rot_repr=rot_repr,
        use_velocity=use_velocity,
        vel_weight=vel_weight_motion,
    )
    audio_curve = compute_e2v_intensity_curve(
        emo_frame_feat,
        use_velocity=use_velocity,
        vel_weight=vel_weight_audio,
    )

    visual_curve = zscore_curve(visual_curve, mask=mask, eps=eps)
    audio_curve = zscore_curve(audio_curve, mask=mask, eps=eps)
    if mask is not None:
        visual_curve = visual_curve * mask.float()
        audio_curve = audio_curve * mask.float()

    return 1.0 - F.cosine_similarity(visual_curve, audio_curve, dim=1, eps=eps).mean()


def compute_level_curve_from_classifier(
    emotion_classifier,
    motion_pred: torch.Tensor,
    window: int = 16,
    stride: int = 4,
) -> torch.Tensor:
    """Optional heavier visual-affect curve using the frozen motion classifier.

    The classifier in ``src/modules/emotion_level_classifier.py`` returns a
    3-way level prediction.  This helper runs it on sliding windows and maps
    low/mid/high probabilities to a scalar curve in [0, 1].  It is not used by
    default because it is slower than expression-energy curves.
    """
    B, L, D = motion_pred.shape
    scores = []
    centers = []
    for center in range(0, L, stride):
        left = max(0, center - window // 2)
        right = min(L, left + window)
        left = max(0, right - window)
        clip = motion_pred[:, left:right]
        if clip.shape[1] < window:
            pad = clip[:, -1:].expand(B, window - clip.shape[1], D)
            clip = torch.cat([clip, pad], dim=1)
        _, level_logits = emotion_classifier(clip[..., :63])
        level_prob = F.softmax(level_logits, dim=-1)
        score = 0.0 * level_prob[:, 0] + 0.5 * level_prob[:, 1] + 1.0 * level_prob[:, 2]
        scores.append(score)
        centers.append(center)

    sparse = torch.stack(scores, dim=1).unsqueeze(1)  # [B, 1, T]
    dense = F.interpolate(sparse, size=L, mode="linear", align_corners=False).squeeze(1)
    return dense

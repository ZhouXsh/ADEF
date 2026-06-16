# coding: utf-8
"""
Decoupled emotion adapter for ADEF/JoyVASA-style DiT motion generation.

This file is intentionally additive: it does not modify the existing
``emotion_dit.py`` or ``emotion_dit_prev_modi.py`` files.  Import the exported
``DitTalkingHead`` from this file in a copied training/inference script when you
want to test the decoupled emotion design.

Core idea
---------
Keep the high-density audio condition out of the emotion modulation path.  The
base DiT is run as audio-only, then a zero-initialized motion-space residual
adapter adds the emotion component.  This makes the first training step behave
like the audio-only model and gives you a separate knob for emotion strength.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .emotion_dit_prev_modi import DitTalkingHead as _AudioBaseDitTalkingHead


def _parse_int_list(value: Optional[Union[str, Iterable[int]]]) -> Tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return tuple()
        return tuple(int(v.strip()) for v in value.split(",") if v.strip())
    return tuple(int(v) for v in value)


def build_emotion_dim_mask(
    motion_feat_dim: int,
    expression_dim: int = 63,
    pose_weight: float = 0.15,
    protected_dims: Optional[Union[str, Iterable[int]]] = None,
    protected_kp_indices: Optional[Union[str, Iterable[int]]] = None,
    protected_weight: float = 0.0,
) -> torch.Tensor:
    """Return a [motion_feat_dim] mask for the emotion residual.

    ``protected_dims`` is the safest option when you know which coefficients
    dominate lip-sync.  ``protected_kp_indices`` is a convenience for LivePortrait
    expression vectors arranged as 21 keypoints x xyz; each keypoint index maps
    to three expression dimensions.
    """
    mask = torch.zeros(motion_feat_dim, dtype=torch.float32)
    exp_end = min(expression_dim, motion_feat_dim)
    mask[:exp_end] = 1.0
    if motion_feat_dim > expression_dim:
        mask[expression_dim:] = pose_weight

    for dim in _parse_int_list(protected_dims):
        if 0 <= dim < motion_feat_dim:
            mask[dim] = protected_weight

    for kp in _parse_int_list(protected_kp_indices):
        start = kp * 3
        end = start + 3
        if start < exp_end:
            mask[start:min(end, exp_end)] = protected_weight
    return mask


class MotionSpaceEmotionAdapter(nn.Module):
    """Small zero-initialized residual adapter in motion-coefficient space.

    The adapter sees the base motion prediction plus aligned audio features, but
    emotion is injected only into this residual branch.  The base audio memory
    used by the DiT cross-attention is therefore not rewritten by emotion labels.
    """

    def __init__(
        self,
        motion_feat_dim: int,
        audio_feat_dim: int,
        emotion_dim: int,
        hidden_dim: int = 512,
        residual_scale: float = 0.35,
        pose_weight: float = 0.15,
        protected_dims: Optional[Union[str, Iterable[int]]] = None,
        protected_kp_indices: Optional[Union[str, Iterable[int]]] = None,
        protected_weight: float = 0.0,
    ):
        super().__init__()
        self.motion_norm = nn.LayerNorm(motion_feat_dim)
        self.audio_norm = nn.LayerNorm(audio_feat_dim)
        self.motion_proj = nn.Linear(motion_feat_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_dim)
        self.emo_to_film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emotion_dim, hidden_dim * 2),
        )
        self.delta = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motion_feat_dim),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)

        self.residual_scale = residual_scale
        mask = build_emotion_dim_mask(
            motion_feat_dim=motion_feat_dim,
            pose_weight=pose_weight,
            protected_dims=protected_dims,
            protected_kp_indices=protected_kp_indices,
            protected_weight=protected_weight,
        )
        self.register_buffer("dim_mask", mask.view(1, 1, -1))

    @staticmethod
    def _align_audio(audio_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad/crop audio features to match a motion sequence length."""
        if audio_feat.shape[1] == target_len:
            return audio_feat
        if audio_feat.shape[1] > target_len:
            return audio_feat[:, -target_len:]
        pad_len = target_len - audio_feat.shape[1]
        pad = audio_feat[:, :1].expand(-1, pad_len, -1)
        return torch.cat([pad, audio_feat], dim=1)

    def forward(
        self,
        base_motion: torch.Tensor,
        audio_feat: torch.Tensor,
        emotion_feat: torch.Tensor,
        strength: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        audio_feat = self._align_audio(audio_feat, base_motion.shape[1])
        h = self.motion_proj(self.motion_norm(base_motion)) + self.audio_proj(self.audio_norm(audio_feat))
        if emotion_feat.ndim == 2:
            emotion_feat = emotion_feat.unsqueeze(1)
        shift, scale = self.emo_to_film(emotion_feat).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        delta = self.delta(h)

        if not torch.is_tensor(strength):
            strength = torch.tensor(float(strength), device=base_motion.device, dtype=base_motion.dtype)
        while strength.ndim < 3:
            strength = strength.unsqueeze(-1)
        return base_motion + self.residual_scale * strength * self.dim_mask * delta


class DitTalkingHead(_AudioBaseDitTalkingHead):
    """Audio-preserving emotion DiT wrapper.

    Constructor arguments are kept close to ``emotion_dit_prev_modi.DitTalkingHead``.
    The inherited base model is forced to use audio-only guidance internally;
    emotion is handled by ``MotionSpaceEmotionAdapter`` after the base prediction.
    """

    def __init__(
        self,
        device: str = "cuda",
        target: str = "sample",
        architecture: str = "decoder",
        motion_feat_dim: int = 70,
        fps: int = 25,
        n_motions: int = 100,
        n_prev_motions: int = 10,
        audio_model: str = "hubert",
        feature_dim: int = 512,
        n_diff_steps: int = 500,
        diff_schedule: str = "cosine",
        cfg_mode: str = "incremental",
        guiding_conditions: str = "audio,emotion",
        emo_classes: int = 8,
        condition_dropout_prob: float = 0.1,
        emotion_dropout_prob: float = 0.15,
        emotion_residual_scale: float = 0.35,
        emotion_hidden_dim: int = 512,
        emotion_pose_weight: float = 0.15,
        emotion_protected_dims: Optional[Union[str, Iterable[int]]] = None,
        emotion_protected_kp_indices: Optional[Union[str, Iterable[int]]] = None,
        emotion_protected_weight: float = 0.0,
    ):
        # Keep the original requested conditions for logging/inference API, but
        # do not let the base DiT modulate audio with emotion.
        self.requested_guiding_conditions = guiding_conditions
        super().__init__(
            device=device,
            target=target,
            architecture=architecture,
            motion_feat_dim=motion_feat_dim,
            fps=fps,
            n_motions=n_motions,
            n_prev_motions=n_prev_motions,
            audio_model=audio_model,
            feature_dim=feature_dim,
            n_diff_steps=n_diff_steps,
            diff_schedule=diff_schedule,
            cfg_mode=cfg_mode,
            guiding_conditions="audio",
            emo_classes=emo_classes,
        )
        self.guiding_conditions = ["audio", "emotion"]
        self.emotion_dropout_prob = emotion_dropout_prob
        self.decoupled_null_emotion = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.decoupled_emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.decoupled_emo_norm = nn.LayerNorm(feature_dim)
        self.emotion_adapter = MotionSpaceEmotionAdapter(
            motion_feat_dim=motion_feat_dim,
            audio_feat_dim=feature_dim,
            emotion_dim=feature_dim,
            hidden_dim=emotion_hidden_dim,
            residual_scale=emotion_residual_scale,
            pose_weight=emotion_pose_weight,
            protected_dims=emotion_protected_dims,
            protected_kp_indices=emotion_protected_kp_indices,
            protected_weight=emotion_protected_weight,
        )
        self.to(device)

    def set_base_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze inherited audio DiT parameters, leaving adapter trainable."""
        adapter_prefixes = ("decoupled_", "emotion_adapter")
        for name, param in self.named_parameters():
            param.requires_grad = trainable or name.startswith(adapter_prefixes)

    def _emotion_feature(self, emo_index: Optional[torch.Tensor], batch_size: int,
                         drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if emo_index is None:
            emo_index = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        emo_feat = self.decoupled_emo_embed(emo_index).unsqueeze(1)
        if drop_mask is not None:
            emo_feat = torch.where(
                drop_mask.view(-1, 1, 1),
                self.decoupled_null_emotion.expand(batch_size, 1, -1),
                emo_feat,
            )
        return self.decoupled_emo_norm(emo_feat)

    def _apply_emotion(self, base_motion: torch.Tensor, audio_feat: torch.Tensor,
                       emo_index: Optional[torch.Tensor], strength: Union[float, torch.Tensor]) -> torch.Tensor:
        emo_feat = self._emotion_feature(emo_index, base_motion.shape[0])
        return self.emotion_adapter(base_motion, audio_feat, emo_feat, strength=strength)

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
                time_step=None, indicator=None, emo_index=None, emotion_strength=None):
        eps, base_target, prev_motion, audio_feat = super().forward(
            motion_feat,
            audio_or_feat,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            time_step=time_step,
            indicator=indicator,
            emo_index=emo_index,
        )
        batch_size = motion_feat.shape[0]
        if emotion_strength is None:
            if self.training and self.emotion_dropout_prob > 0:
                keep = torch.rand(batch_size, device=self.device) >= self.emotion_dropout_prob
                emotion_strength = keep.float()
            else:
                emotion_strength = 1.0
        target = self._apply_emotion(base_target, audio_feat, emo_index, emotion_strength)
        return eps, target, prev_motion, audio_feat

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None, emotion_strength: float = 1.0):
        # Run the diffusion process with audio CFG only.  The emotion delta is
        # added after denoising, avoiding emotion/audio CFG interference.
        if isinstance(cfg_scale, (list, tuple)):
            base_scale = cfg_scale[0] if len(cfg_scale) > 0 else 1.15
        else:
            base_scale = cfg_scale
        motion, noise, audio_feat = super().sample(
            audio_or_feat,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            motion_at_T=motion_at_T,
            indicator=indicator,
            cfg_mode=cfg_mode,
            cfg_cond=["audio"],
            cfg_scale=base_scale,
            flexibility=flexibility,
            dynamic_threshold=dynamic_threshold,
            ret_traj=ret_traj,
            emo_index=emo_index,
        )
        if ret_traj:
            for key in list(motion.keys()):
                if torch.is_tensor(motion[key]):
                    motion[key] = self._apply_emotion(motion[key].to(self.device), audio_feat, emo_index, emotion_strength)
            return motion, noise, audio_feat
        motion = self._apply_emotion(motion, audio_feat, emo_index, emotion_strength)
        return motion, noise, audio_feat

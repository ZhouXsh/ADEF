"""Final v2: dual-branch DiT with emotion-bank modulated audio.

This file builds on ``emotion_dit_clean_encoding.py`` and the final-v1
双分支 backbone.  The DiT still has only two condition branches:

    1. original audio branch for lip-sync;
    2. emotion-audio branch for emotional expression.

Different from final-v1, the emotion-audio feature is not produced by a single
emotion embedding.  It is produced by a class-specific emotion basis bank
``[B, K, D]``.  The audio feature queries this bank and obtains an emotion
residual on the audio time axis:

    A_e = A + alpha * Gate(A, Attn(Q=A, K=P_y, V=P_y))

Thus the emotion bank does not directly act as a separate DiT condition; it first
recalibrates audio into an emotion-fused audio representation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .emotion_dit_clean_encoding import EmotionBasisEncoder
from .emotion_dit_finalv1 import DitTalkingHead as _FinalV1DitTalkingHead


class EmotionBankAudioModulator(nn.Module):
    """Use a class-specific emotion basis bank to recalibrate audio features."""

    def __init__(
        self,
        feature_dim: int,
        emo_classes: int = 8,
        num_basis_tokens: int = 8,
        n_heads: int = 8,
        residual_init: float = 0.05,
    ):
        super().__init__()
        self.basis_encoder = EmotionBasisEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            num_basis_tokens=num_basis_tokens,
        )
        self.audio_to_emotion_attn = nn.MultiheadAttention(
            feature_dim, n_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        drop_emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        basis = self.basis_encoder(emo_index, drop_mask=drop_emotion)
        residual = self.audio_to_emotion_attn(
            query=audio_feat,
            key=basis,
            value=basis,
            need_weights=False,
        )[0]
        gate = self.gate(torch.cat([audio_feat, residual], dim=-1))
        residual = self.out_proj(gate * residual)
        return audio_feat + self.residual_scale.tanh() * residual


class DitTalkingHead(_FinalV1DitTalkingHead):
    """Final v2: original audio branch + emotion-bank modulated audio branch."""

    def __init__(
        self,
        device="cuda",
        target="sample",
        architecture="decoder",
        motion_feat_dim=70,
        fps=25,
        n_motions=100,
        n_prev_motions=10,
        audio_model="hubert",
        feature_dim=512,
        n_diff_steps=500,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,emotion",
        emo_classes=8,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 1,
        decoder_dropout: float = 0.1,
        audio_scale: float = 0.5,
        emotion_audio_scale: float = 0.5,
        num_emotion_basis_tokens: int = 8,
        emotion_audio_residual_init: float = 0.05,
    ):
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
            guiding_conditions=guiding_conditions,
            emo_classes=emo_classes,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            decoder_dropout=decoder_dropout,
            audio_scale=audio_scale,
            emotion_audio_scale=emotion_audio_scale,
        )
        self.emotion_bank_audio_modulator = EmotionBankAudioModulator(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            num_basis_tokens=num_emotion_basis_tokens,
            n_heads=n_heads,
            residual_init=emotion_audio_residual_init,
        ).to(self.device)

    def _build_emotion_audio(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        drop_emotion: Optional[torch.Tensor] = None,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio_feat = self.audio_norm(audio_feat)
        return self.emotion_bank_audio_modulator(
            audio_feat,
            emo_index=emo_index,
            drop_emotion=drop_emotion,
        )

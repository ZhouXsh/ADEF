"""Two-stage final-v2: emotion-bank recalibrated audio diffusion model.

Method
------
Stage 1 uses generic unlabeled talking videos and trains only the emotion-free
base: audio feature mapping, motion/time backbone, original-audio
cross-attention, autoregressive start states and motion decoder.  Emotion-bank
parameters and emotion cross-attention adapters are frozen and bypassed.

Stage 2 freezes the complete Stage-1 audio base and learns:

1. a class-specific [B, K, D] emotion basis bank;
2. audio-to-emotion-bank attention and a gated residual audio modulator;
3. zero-initialized per-layer emotion-audio residual cross-attention adapters;
4. optionally the last shared FFN layers / motion head with a smaller LR.

For each audio frame A_t, the emotion bank P_y is queried to obtain a
content-aware emotion residual.  The resulting emotion audio is

    A_e = A + alpha * Gate(A, Attn(Q=A, K=P_y, V=P_y)).

The original audio branch remains frozen in Stage 2, protecting lip-sync while
the learned emotion branch contributes only residual emotional motion.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .emotion_dit_two_stage_base import TwoStageDitTalkingHead


class EmotionBasisEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        emo_classes: int,
        num_basis_tokens: int = 8,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.emotion_basis = nn.Parameter(
            torch.randn(
                emo_classes, num_basis_tokens, feature_dim
            ) * init_std
        )
        self.null_basis = nn.Parameter(
            torch.zeros(1, num_basis_tokens, feature_dim)
        )
        self.token_position = nn.Parameter(
            torch.randn(1, num_basis_tokens, feature_dim) * init_std
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(
        self,
        emo_index: torch.Tensor,
        drop_emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = emo_index.shape[0]
        basis = self.token_projection(
            self.emotion_basis[emo_index] + self.token_position
        )
        if drop_emotion is not None:
            basis = torch.where(
                drop_emotion.to(basis.device).view(batch_size, 1, 1),
                self.null_basis.expand(batch_size, -1, -1),
                basis,
            )
        return basis


class EmotionBankAudioEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        emo_classes: int,
        num_basis_tokens: int,
        n_heads: int,
        residual_init: float = 0.05,
    ):
        super().__init__()
        self.basis_encoder = EmotionBasisEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            num_basis_tokens=num_basis_tokens,
        )
        self.audio_to_basis_attention = nn.MultiheadAttention(
            feature_dim, n_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        nn.init.zeros_(self.output_projection[-1].weight)
        nn.init.zeros_(self.output_projection[-1].bias)
        self.residual_scale = nn.Parameter(
            torch.tensor(float(residual_init))
        )

    def forward(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        drop_emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del emo_utt_feat, emo_frame_feat
        batch_size = audio_feat.shape[0]
        basis = self.basis_encoder(emo_index, drop_emotion)
        residual = self.audio_to_basis_attention(
            query=audio_feat,
            key=basis,
            value=basis,
            need_weights=False,
        )[0]
        residual = self.output_projection(
            self.gate(torch.cat([audio_feat, residual], dim=-1)) * residual
        )
        if drop_emotion is not None:
            residual = residual * (~drop_emotion).to(
                device=residual.device,
                dtype=residual.dtype,
            ).view(batch_size, 1, 1)
        return audio_feat + self.residual_scale.tanh() * residual


class DitTalkingHead(TwoStageDitTalkingHead):
    """Two-stage final-v2 model with emotion-bank audio modulation."""

    def _create_emotion_audio_encoder(
        self,
        feature_dim: int,
        emo_classes: int,
        e2v_dim: int,
        num_emotion_tokens: int,
        n_heads: int,
        residual_init: float,
    ) -> nn.Module:
        del e2v_dim
        return EmotionBankAudioEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            num_basis_tokens=num_emotion_tokens,
            n_heads=n_heads,
            residual_init=residual_init,
        )

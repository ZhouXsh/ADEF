"""Two-stage final-v1: label-AdaLN emotion-audio diffusion model.

Method
------
Stage 1 learns a generic audio-to-motion prior from unlabeled audiovisual data.
Only the audio encoder projection, motion/time backbone, original-audio
cross-attention and motion decoder are trainable.  The entire emotion path is
frozen and absent from the forward pass.

Stage 2 loads the Stage-1 checkpoint and freezes the learned audio base.  It
trains only:

1. the label embedding and AdaLN emotion-audio encoder;
2. the per-layer emotion-audio residual cross-attention adapters;
3. optionally a small shared FFN tail / motion head at a lower learning rate.

The Stage-2 DiT therefore has two semantically distinct paths:

    original audio -> frozen audio branch -> lip-sync/content motion
    label-modulated audio -> emotion adapter -> emotional residual motion

The emotion adapter output projection is zero-initialized, so Stage 2 starts
exactly from the Stage-1 behavior instead of immediately disturbing lip-sync.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .emotion_dit_two_stage_base import TwoStageDitTalkingHead


class LabelAdaLNAudioEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        emo_classes: int,
        residual_init: float = 0.05,
    ):
        super().__init__()
        self.label_embedding = nn.Embedding(emo_classes, feature_dim)
        self.null_label = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)
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
        batch_size = audio_feat.shape[0]
        label = self.label_embedding(emo_index).unsqueeze(1)
        if drop_emotion is not None:
            label = torch.where(
                drop_emotion.to(audio_feat.device).view(batch_size, 1, 1),
                self.null_label.expand(batch_size, -1, -1),
                label,
            )
        shift, scale = self.modulation(label).chunk(2, dim=-1)
        modulated = audio_feat * (1.0 + scale) + shift
        return audio_feat + self.residual_scale.tanh() * (
            modulated - audio_feat
        )


class DitTalkingHead(TwoStageDitTalkingHead):
    """Two-stage final-v1 model with label-AdaLN emotion audio."""

    def _create_emotion_audio_encoder(
        self,
        feature_dim: int,
        emo_classes: int,
        e2v_dim: int,
        num_emotion_tokens: int,
        n_heads: int,
        residual_init: float,
    ) -> nn.Module:
        del e2v_dim, num_emotion_tokens, n_heads
        return LabelAdaLNAudioEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            residual_init=residual_init,
        )

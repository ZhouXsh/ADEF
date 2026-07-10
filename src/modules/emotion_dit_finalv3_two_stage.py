"""Two-stage final-v3: hierarchical label/utterance/frame emotion-audio DiT.

Method
------
Stage 1 is emotion-agnostic.  It trains the generic audio-to-motion base on
unlabeled talking videos and completely bypasses the emotion branch.  The
learned original-audio cross-attention is therefore optimized only for speech
content, temporal alignment and lip synchronization.

Stage 2 loads the Stage-1 checkpoint, freezes the complete audio base, and
trains a target-anchored hierarchical emotion encoder plus zero-initialized
emotion residual adapters:

    label y -> class-specific emotion basis P_y
    utterance emotion2vec u -> global modulation of P_y
    frame emotion2vec F -> target-aware temporal emotion dynamics F_y
    audio A queries [P_y^u, u, F_y]
    A_e = A + alpha * gated emotion residual

The DiT retains two branches:

    frozen original audio branch -> content/lip synchronization
    trainable hierarchical emotion-audio branch -> emotional residual motion

By default Stage 2 does not update the audio encoder, audio feature projection,
original-audio attention, motion/time backbone or motion decoder.  Optional
low-rate tuning of the last shared FFN layers and motion head is supported by
the training script.  Both the emotion-audio output projection and per-layer
emotion adapters start from zero, so Stage 2 initially reproduces Stage 1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_dit_two_stage_base import TwoStageDitTalkingHead


class HierarchicalEmotionAudioEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        emo_classes: int,
        e2v_dim: int,
        num_label_tokens: int,
        n_heads: int,
        residual_init: float = 0.05,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.e2v_dim = e2v_dim

        self.label_basis = nn.Parameter(
            torch.randn(
                emo_classes, num_label_tokens, feature_dim
            ) * init_std
        )
        self.label_position = nn.Parameter(
            torch.randn(1, num_label_tokens, feature_dim) * init_std
        )
        self.null_label = nn.Parameter(
            torch.zeros(1, num_label_tokens, feature_dim)
        )
        self.label_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.utterance_projection = nn.Sequential(
            nn.LayerNorm(e2v_dim),
            nn.Linear(e2v_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.null_utterance = nn.Parameter(
            torch.zeros(1, 1, feature_dim)
        )
        self.label_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )

        self.frame_projection = nn.Sequential(
            nn.LayerNorm(e2v_dim),
            nn.Linear(e2v_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.null_frame = nn.Parameter(
            torch.zeros(1, 1, feature_dim)
        )
        self.frame_norm = nn.LayerNorm(feature_dim)
        self.frame_modulation = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2 * feature_dim),
        )

        self.audio_to_emotion_attention = nn.MultiheadAttention(
            feature_dim, n_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * feature_dim),
            nn.Linear(3 * feature_dim, feature_dim),
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

    @staticmethod
    def _resize_temporal(
        feature: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        if feature.shape[1] == target_length:
            return feature
        feature = F.interpolate(
            feature.transpose(1, 2),
            size=target_length,
            mode="linear",
            align_corners=False,
        )
        return feature.transpose(1, 2).contiguous()

    @staticmethod
    def _drop(
        feature: torch.Tensor,
        null_feature: torch.Tensor,
        drop_emotion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if drop_emotion is None:
            return feature
        batch_size = feature.shape[0]
        return torch.where(
            drop_emotion.to(feature.device).view(batch_size, 1, 1),
            null_feature.expand(batch_size, *null_feature.shape[1:]),
            feature,
        )

    def _encode_utterance(
        self,
        emo_utt_feat: Optional[torch.Tensor],
        batch_size: int,
        device,
        drop_emotion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if emo_utt_feat is None:
            utterance = self.null_utterance.expand(batch_size, -1, -1)
        else:
            utterance = emo_utt_feat.to(device).float()
            if utterance.ndim == 3:
                utterance = utterance.mean(dim=1)
            if utterance.shape[-1] == self.feature_dim:
                utterance = utterance.unsqueeze(1)
            else:
                utterance = self.utterance_projection(utterance).unsqueeze(1)
        return self._drop(
            utterance,
            self.null_utterance,
            drop_emotion,
        )

    def _encode_frame(
        self,
        emo_frame_feat: Optional[torch.Tensor],
        batch_size: int,
        target_length: int,
        device,
        drop_emotion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if emo_frame_feat is None:
            frame = self.null_frame.expand(
                batch_size, target_length, -1
            )
        else:
            frame = self._resize_temporal(
                emo_frame_feat.to(device).float(),
                target_length,
            )
            if frame.shape[-1] != self.feature_dim:
                frame = self.frame_projection(frame)
        null_frame = self.null_frame.expand(
            batch_size, target_length, -1
        )
        return self._drop(frame, null_frame, drop_emotion)

    def forward(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        drop_emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, length, _ = audio_feat.shape

        label = self.label_projection(
            self.label_basis[emo_index] + self.label_position
        )
        label = self._drop(label, self.null_label, drop_emotion)

        utterance = self._encode_utterance(
            emo_utt_feat,
            batch_size,
            audio_feat.device,
            drop_emotion,
        )
        label_scale, label_shift = self.label_modulation(
            utterance
        ).chunk(2, dim=-1)
        label = label * (1.0 + label_scale) + label_shift

        frame = self._encode_frame(
            emo_frame_feat,
            batch_size,
            length,
            audio_feat.device,
            drop_emotion,
        )
        target_context = label.mean(dim=1) + utterance.squeeze(1)
        frame_scale, frame_shift = self.frame_modulation(
            target_context
        ).chunk(2, dim=-1)
        frame = self.frame_norm(frame) * (
            1.0 + frame_scale.unsqueeze(1)
        ) + frame_shift.unsqueeze(1)
        frame = self._drop(
            frame,
            self.null_frame.expand(batch_size, length, -1),
            drop_emotion,
        )

        emotion_memory = torch.cat([label, utterance, frame], dim=1)
        residual = self.audio_to_emotion_attention(
            query=audio_feat,
            key=emotion_memory,
            value=emotion_memory,
            need_weights=False,
        )[0]
        residual = self.output_projection(
            self.gate(torch.cat([audio_feat, residual, frame], dim=-1))
            * residual
        )
        if drop_emotion is not None:
            residual = residual * (~drop_emotion).to(
                device=residual.device,
                dtype=residual.dtype,
            ).view(batch_size, 1, 1)
        return audio_feat + self.residual_scale.tanh() * residual


class DitTalkingHead(TwoStageDitTalkingHead):
    """Two-stage final-v3 hierarchical emotion-audio model."""

    def _create_emotion_audio_encoder(
        self,
        feature_dim: int,
        emo_classes: int,
        e2v_dim: int,
        num_emotion_tokens: int,
        n_heads: int,
        residual_init: float,
    ) -> nn.Module:
        return HierarchicalEmotionAudioEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            e2v_dim=e2v_dim,
            num_label_tokens=num_emotion_tokens,
            n_heads=n_heads,
            residual_init=residual_init,
        )

"""Shared stage-isolated decoder layers for two-stage dual-audio DiT.

This module is shared intentionally so finalv1/finalv2/finalv3 use exactly the
same Stage-1 audio base, freezing boundary and Stage-2 emotion residual adapter.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask


class TwoStageDualAudioDecoderLayer(nn.Module):
    """Frozen-capable audio base with parallel dual-audio conditioning.

    Both the original-audio branch and the emotion-audio branch query their
    respective time-aligned memories from the same post-self-attention motion
    representation. Their condition updates are fused before the shared FFN.
    The emotion branch remains an isolated, zero-initialized residual adapter,
    so Stage 2 starts from exactly the Stage-1 function.
    """

    def __init__(
        self,
        feature_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale_init: float = 0.10,
    ):
        super().__init__()
        self.audio_scale = float(audio_scale)

        self.self_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.audio_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.linear1 = nn.Linear(feature_dim, mlp_ratio * feature_dim)
        self.linear2 = nn.Linear(mlp_ratio * feature_dim, feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.ff_dropout = nn.Dropout(dropout)

        self.emotion_query_norm = nn.LayerNorm(feature_dim)
        self.emotion_audio_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.emotion_out_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.emotion_out_proj.weight)
        nn.init.zeros_(self.emotion_out_proj.bias)
        self.emotion_dropout = nn.Dropout(dropout)
        self.emotion_strength = nn.Parameter(
            torch.tensor(float(emotion_scale_init))
        )

    @staticmethod
    def _requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def enable_stage1_base(self) -> None:
        for module in (
            self.self_attn,
            self.audio_attn,
            self.norm1,
            self.norm2,
            self.norm3,
            self.linear1,
            self.linear2,
        ):
            self._requires_grad(module, True)

    def enable_stage2_emotion(self) -> None:
        for module in (
            self.emotion_query_norm,
            self.emotion_audio_attn,
            self.emotion_out_proj,
        ):
            self._requires_grad(module, True)
        self.emotion_strength.requires_grad_(True)

    def enable_shared_tail_finetune(self) -> None:
        """Low-rate Stage-2 adaptation without unfreezing audio attention."""
        for module in (self.norm3, self.linear1, self.linear2):
            self._requires_grad(module, True)

    def enforce_frozen_base_eval(self) -> None:
        """Disable stochastic behavior in the frozen Stage-1 base."""
        for module in (
            self.self_attn,
            self.audio_attn,
            self.norm1,
            self.norm2,
            self.norm3,
            self.linear1,
            self.linear2,
            self.dropout1,
            self.dropout2,
            self.dropout3,
            self.ff_dropout,
        ):
            module.eval()

    def forward(
        self,
        hidden: torch.Tensor,
        audio_memory: torch.Tensor,
        emotion_audio_memory: Optional[torch.Tensor] = None,
        alignment_mask: Optional[torch.Tensor] = None,
        emotion_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Motion self-attention first establishes the common query used by both
        # time-aligned audio condition branches.
        self_update = self.self_attn(
            hidden, hidden, hidden, need_weights=False
        )[0]
        hidden = self.norm1(hidden + self.dropout1(self_update))
        condition_query = hidden

        # Content/original-audio branch.
        audio_update = self.audio_attn(
            query=condition_query,
            key=audio_memory,
            value=audio_memory,
            attn_mask=alignment_mask,
            need_weights=False,
        )[0]
        condition_update = self.audio_scale * audio_update

        # Emotion-audio branch. It uses the same post-self-attention motion
        # representation as the content branch and is fused before the FFN.
        if emotion_audio_memory is not None:
            emotion_update = self.emotion_audio_attn(
                query=self.emotion_query_norm(condition_query),
                key=emotion_audio_memory,
                value=emotion_audio_memory,
                attn_mask=alignment_mask,
                need_weights=False,
            )[0]
            emotion_update = self.emotion_out_proj(emotion_update)
            if emotion_present is not None:
                emotion_update = emotion_update * emotion_present.to(
                    dtype=emotion_update.dtype,
                    device=emotion_update.device,
                ).view(-1, 1, 1)
            condition_update = condition_update + (
                self.emotion_strength.tanh()
                * self.emotion_dropout(emotion_update)
            )

        # Fuse both condition branches before the shared feed-forward block.
        hidden = self.norm2(
            hidden + self.dropout2(condition_update)
        )
        ff = self.linear2(
            self.ff_dropout(F.gelu(self.linear1(hidden)))
        )
        hidden = self.norm3(hidden + self.dropout3(ff))
        return hidden


class TwoStageDualAudioDenoisingNetwork(nn.Module):
    def __init__(
        self,
        device: str = "cuda",
        motion_feat_dim: int = 70,
        feature_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 3,
        use_indicator: bool = False,
        use_learnable_pe: bool = False,
        n_prev_motions: int = 25,
        n_motions: int = 100,
        n_diff_steps: int = 50,
        dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale_init: float = 0.10,
    ):
        super().__init__()
        self.motion_feat_dim = motion_feat_dim
        self.feature_dim = feature_dim
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions
        self.use_indicator = bool(use_indicator)
        self.use_learnable_pe = bool(use_learnable_pe)

        self.time_encoding = PositionalEncoding(
            feature_dim, max_len=n_diff_steps + 1
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.motion_proj = nn.Linear(
            motion_feat_dim + (1 if self.use_indicator else 0),
            feature_dim,
        )

        if self.use_learnable_pe:
            self.position = nn.Parameter(
                torch.randn(
                    1, n_prev_motions + n_motions, feature_dim
                ) * 0.02
            )
        else:
            self.position = PositionalEncoding(feature_dim, dropout=dropout)

        self.layers = nn.ModuleList([
            TwoStageDualAudioDecoderLayer(
                feature_dim=feature_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                audio_scale=audio_scale,
                emotion_scale_init=emotion_scale_init,
            )
            for _ in range(n_layers)
        ])

        if align_mask_width > 0:
            total_len = n_prev_motions + n_motions
            mask = enc_dec_mask(
                total_len,
                total_len,
                frame_width=1,
                expansion=align_mask_width - 1,
                device=device,
            )
            self.register_buffer("alignment_mask", mask)
        else:
            self.alignment_mask = None

        self.motion_dec = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, motion_feat_dim),
        )
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def enable_stage1_base(self) -> None:
        self._requires_grad(self.time_mlp, True)
        self._requires_grad(self.motion_proj, True)
        if isinstance(self.position, nn.Parameter):
            self.position.requires_grad_(True)
        for layer in self.layers:
            layer.enable_stage1_base()
        self._requires_grad(self.motion_dec, True)

    def enable_stage2_emotion(
        self,
        tune_tail_layers: int = 0,
        tune_motion_head: bool = False,
    ) -> None:
        for layer in self.layers:
            layer.enable_stage2_emotion()
        if tune_tail_layers > 0:
            for layer in self.layers[-tune_tail_layers:]:
                layer.enable_shared_tail_finetune()
        if tune_motion_head:
            self._requires_grad(self.motion_dec, True)

    def enforce_stage_mode(self, stage: int) -> None:
        if stage == 2:
            for layer in self.layers:
                layer.enforce_frozen_base_eval()

    def forward(
        self,
        motion_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        prev_motion_feat: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        step: torch.Tensor,
        indicator: Optional[torch.Tensor] = None,
        emotion_audio_feat: Optional[torch.Tensor] = None,
        prev_emotion_audio_feat: Optional[torch.Tensor] = None,
        emotion_present: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.tensor(step, device=self.device, dtype=torch.long)
        step = step.to(self.device).long()
        time_embedding = self.time_mlp(
            self.time_encoding.pe[0, step]
        ).unsqueeze(1)

        motion_input = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            if indicator is None:
                indicator = torch.ones(
                    motion_feat.shape[0],
                    self.n_motions,
                    dtype=motion_feat.dtype,
                    device=motion_feat.device,
                )
            indicator = torch.cat([
                torch.zeros(
                    indicator.shape[0],
                    self.n_prev_motions,
                    dtype=indicator.dtype,
                    device=indicator.device,
                ),
                indicator,
            ], dim=1).unsqueeze(-1)
            motion_input = torch.cat([motion_input, indicator], dim=-1)

        hidden = self.motion_proj(motion_input)
        if isinstance(self.position, nn.Parameter):
            hidden = hidden + self.position[:, :hidden.shape[1]]
        else:
            hidden = self.position(hidden)
        hidden = hidden + time_embedding

        audio_memory = torch.cat([prev_audio_feat, audio_feat], dim=1)
        emotion_audio_memory = None
        if emotion_audio_feat is not None:
            if prev_emotion_audio_feat is None:
                prev_emotion_audio_feat = prev_audio_feat
            emotion_audio_memory = torch.cat(
                [prev_emotion_audio_feat, emotion_audio_feat], dim=1
            )

        for layer in self.layers:
            hidden = layer(
                hidden,
                audio_memory=audio_memory,
                emotion_audio_memory=emotion_audio_memory,
                alignment_mask=self.alignment_mask,
                emotion_present=emotion_present,
            )

        return self.motion_dec(hidden)

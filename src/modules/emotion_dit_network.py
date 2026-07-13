from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .common import PositionalEncoding, enc_dec_mask


class DenoisingNetwork(nn.Module):
    def __init__(
        self,
        device: str | torch.device = "cuda",
        motion_feat_dim: int = 70,
        use_indicator: bool = False,
        architecture: str = "decoder",
        feature_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 1,
        no_use_learnable_pe: bool = True,
        n_prev_motions: int = 10,
        n_motions: int = 100,
        n_diff_steps: int = 500,
    ) -> None:
        super().__init__()
        if architecture != "decoder":
            raise ValueError(f"Unknown architecture: {architecture}")

        self.motion_feat_dim = int(motion_feat_dim)
        self.use_indicator = bool(use_indicator)
        self.architecture = architecture
        self.feature_dim = int(feature_dim)
        self.n_prev_motions = int(n_prev_motions)
        self.n_motions = int(n_motions)
        self.use_learnable_pe = not no_use_learnable_pe

        self.TE = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        sequence_length = n_prev_motions + n_motions
        if self.use_learnable_pe:
            self.PE = nn.Parameter(
                torch.randn(1, sequence_length, feature_dim) * 0.02
            )
        else:
            self.PE = PositionalEncoding(feature_dim)

        input_dim = motion_feat_dim + (1 if self.use_indicator else 0)
        self.feature_proj = nn.Linear(input_dim, feature_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=n_heads,
            dim_feedforward=mlp_ratio * feature_dim,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        if align_mask_width > 0:
            alignment_mask = enc_dec_mask(
                sequence_length,
                sequence_length,
                frame_width=1,
                expansion=align_mask_width - 1,
            )
            self.register_buffer("alignment_mask", alignment_mask)
        else:
            self.alignment_mask = None

        self.motion_dec = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, motion_feat_dim),
        )
        self.to(device)

    def forward(
        self,
        motion_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        prev_motion_feat: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        step: torch.Tensor,
        indicator: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        features = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            if indicator is None:
                indicator = torch.ones(
                    motion_feat.shape[0],
                    self.n_motions,
                    device=motion_feat.device,
                    dtype=motion_feat.dtype,
                )
            previous_indicator = torch.ones(
                indicator.shape[0],
                self.n_prev_motions,
                device=indicator.device,
                dtype=indicator.dtype,
            )
            full_indicator = torch.cat([previous_indicator, indicator], dim=1)
            features = torch.cat([features, full_indicator.unsqueeze(-1)], dim=-1)

        features = self.feature_proj(features)
        if self.use_learnable_pe:
            features = features + self.PE + step_embedding
        else:
            features = self.PE(features) + step_embedding

        audio_memory = torch.cat([prev_audio_feat, audio_feat], dim=1)
        output = self.transformer(
            features, audio_memory, memory_mask=self.alignment_mask
        )
        return self.motion_dec(output)

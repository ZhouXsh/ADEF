"""v2: DICE/IP-Adapter dual cross-attention with phase-aware emotion tokens."""

from typing import Optional

import torch
import torch.nn as nn

from .common import PositionalEncoding
from .emotion_dit_v1 import DitTalkingHead as _V1DitTalkingHead


class PhaseAwareEmotionEncoder(nn.Module):
    def __init__(self, feature_dim: int, emo_classes: int, n_diff_steps: int,
                 k_coarse: int = 2, k_region: int = 4, k_detail: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_diff_steps = n_diff_steps
        self.k_coarse = k_coarse
        self.k_region = k_region
        self.k_detail = k_detail
        self.emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.step_pe = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.step_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.shared = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.coarse_proj = nn.Linear(feature_dim, k_coarse * feature_dim)
        self.region_proj = nn.Linear(feature_dim, k_region * feature_dim)
        self.detail_proj = nn.Linear(feature_dim, k_detail * feature_dim)
        self.learned_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 3),
        )
        self.null_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def forward(self, emo_index: torch.Tensor, step: torch.Tensor,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emo_index.shape[0]
        C = self.feature_dim
        step = step.to(emo_index.device).long()
        if drop_mask is not None and bool(drop_mask.all()):
            return self.null_token.expand(B, self.k_coarse + self.k_region + self.k_detail, -1)

        emo = self.emo_embed(emo_index)
        step_emb = self.step_mlp(self.step_pe.pe[0, step].to(emo.device))
        cond = self.shared(torch.cat([emo, step_emb], dim=-1))
        coarse = self.coarse_proj(cond).view(B, self.k_coarse, C)
        region = self.region_proj(cond).view(B, self.k_region, C)
        detail = self.detail_proj(cond).view(B, self.k_detail, C)

        rho = (step.float() / float(self.n_diff_steps)).clamp(0, 1)
        mid = (1.0 - torch.abs(rho - 0.5) * 2.0).clamp(0, 1)
        phase_prior = torch.stack([
            0.35 + 0.65 * rho,
            0.35 + 0.65 * mid,
            0.35 + 0.65 * (1.0 - rho),
        ], dim=-1)
        phase_learned = torch.sigmoid(self.learned_gate(step_emb))
        gate = phase_prior * phase_learned
        coarse = coarse * gate[:, 0].view(B, 1, 1)
        region = region * gate[:, 1].view(B, 1, 1)
        detail = detail * gate[:, 2].view(B, 1, 1)
        tokens = torch.cat([coarse, region, detail], dim=1)

        if drop_mask is not None:
            null = self.null_token.expand(B, tokens.shape[1], -1)
            tokens = torch.where(drop_mask.view(B, 1, 1), null, tokens)
        return tokens


class DitTalkingHead(_V1DitTalkingHead):
    min_audio_cfg = 1.0
    min_emotion_cfg = 0.15

    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=10,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental",
                 guiding_conditions="audio,emotion", emo_classes=8):
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
        )
        self.emotion_encoder = PhaseAwareEmotionEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            n_diff_steps=n_diff_steps,
            k_coarse=2,
            k_region=4,
            k_detail=2,
        ).to(self.device)

    def encode_emotion(self, emo_index, step=None, audio_feat=None, drop_mask=None):
        if step is None:
            step = torch.full((emo_index.shape[0],), self.diffusion_sched.num_steps,
                              device=self.device, dtype=torch.long)
        return self.emotion_encoder(emo_index, step, drop_mask=drop_mask)

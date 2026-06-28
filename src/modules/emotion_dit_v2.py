"""Phase-aware step emotion tokens for implicit-keypoint motion diffusion.

Version v2 keeps the same safe topology as v1 but upgrades the emotion code:
- shared emotion semantics;
- three implicit phase token groups: coarse, dynamic, detail;
- diffusion-step gates make the groups dominate at early/middle/late denoising.

The groups are implicit subspaces, not explicit facial parts.
"""

import torch
import torch.nn as nn

from .common import PositionalEncoding
from .emotion_dit_v1 import DitTalkingHead as _V1DitTalkingHead
from .emotion_dit_v1 import DenoisingNetworkV1


class PhaseAwareEmotionEncoder(nn.Module):
    """Coarse/dynamic/detail emotion tokens controlled by diffusion timestep.

    Output shape: [B, K_total, C].
    K_total = k_coarse + k_dynamic + k_detail.
    """

    def __init__(self, feature_dim: int, emo_classes: int, n_diff_steps: int,
                 k_coarse: int = 2, k_dynamic: int = 4, k_detail: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_diff_steps = n_diff_steps
        self.k_coarse = k_coarse
        self.k_dynamic = k_dynamic
        self.k_detail = k_detail
        self.k_total = k_coarse + k_dynamic + k_detail

        self.emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.null_emo = nn.Parameter(torch.zeros(1, feature_dim))
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
        self.dynamic_proj = nn.Linear(feature_dim, k_dynamic * feature_dim)
        self.detail_proj = nn.Linear(feature_dim, k_detail * feature_dim)
        self.learned_phase_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 3),
        )

    def forward(self, emo_index: torch.Tensor, step: torch.Tensor, drop_mask: torch.Tensor | None = None):
        B = emo_index.shape[0]
        C = self.feature_dim
        step = step.to(emo_index.device).long()
        emo = self.emo_embed(emo_index)
        if drop_mask is not None:
            emo = torch.where(drop_mask.view(B, 1), self.null_emo.expand(B, -1), emo)

        step_emb = self.step_mlp(self.step_pe.pe[0, step].to(emo.device))
        cond = self.shared(torch.cat([emo, step_emb], dim=-1))

        coarse = self.coarse_proj(cond).view(B, self.k_coarse, C)
        dynamic = self.dynamic_proj(cond).view(B, self.k_dynamic, C)
        detail = self.detail_proj(cond).view(B, self.k_detail, C)

        rho = (step.float() / float(self.n_diff_steps)).clamp(0, 1)
        # sampling goes T -> 1, so high rho means early denoising.
        g_coarse = rho
        g_dynamic = (1.0 - torch.abs(rho - 0.5) * 2.0).clamp(0, 1)
        g_detail = 1.0 - rho
        phase_prior = torch.stack([g_coarse, g_dynamic, g_detail], dim=-1)
        phase_learned = torch.sigmoid(self.learned_phase_gate(step_emb))
        phase_gate = phase_prior * phase_learned

        coarse = coarse * phase_gate[:, 0].view(B, 1, 1)
        dynamic = dynamic * phase_gate[:, 1].view(B, 1, 1)
        detail = detail * phase_gate[:, 2].view(B, 1, 1)
        return torch.cat([coarse, dynamic, detail], dim=1)


class DitTalkingHead(_V1DitTalkingHead):
    """v2: v1 topology + coarse/dynamic/detail phase-aware emotion tokens."""

    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=10,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental", guiding_conditions="audio,emotion", emo_classes=8):
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
            k_dynamic=4,
            k_detail=2,
        )
        # Keep post-transformer adapter from v1; only the tokenization changes.
        self.denoising_net = DenoisingNetworkV1(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
        )
        self.to(device)

"""v6: Stochastic Affective Motion Prior.

This version extends v5 with a lightweight stochastic emotion prior, inspired by
FlowVQTalker's one-to-many nonverbal motion modeling:

1. The deterministic emotion basis bank from v5 is kept.
2. A Gaussian affective latent prior predicts mu/logvar from emotion id,
   audio summary, and diffusion timestep.
3. Reparameterized latent tokens are appended to the emotion memory during
   training; random samples are used during inference. This encourages the same
   audio + emotion pair to support diverse but emotion-consistent nonverbal
   motion, such as head tendency, eyebrow/cheek dynamics, and micro expression.
4. The denoiser still uses local/global audio attention plus orthogonalized
   emotion attention, keeping lip synchronization anchored by the local audio path.

The prior is intentionally small and self-contained; it does not require an
external AU/DECA pipeline and is suitable for 3x80G A100 training.
"""

from typing import Optional

import torch
import torch.nn as nn

from .common import PositionalEncoding
from .emotion_dit_v5 import DitTalkingHead as _V5DitTalkingHead
from .emotion_dit_v5 import EmotionMotionBasisBank
from .emotion_dit_v4 import DecoupledAudioEmotionDenoisingNetwork


class StochasticAffectivePrior(nn.Module):
    """Small Gaussian latent prior for emotion-consistent motion diversity."""

    def __init__(self, feature_dim: int, emo_classes: int, n_diff_steps: int,
                 n_latent_tokens: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_latent_tokens = n_latent_tokens
        self.emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.step_pe = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.prior = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * n_latent_tokens * feature_dim),
        )
        self.null_tokens = nn.Parameter(torch.zeros(1, n_latent_tokens, feature_dim))
        self.latent_scale = nn.Parameter(torch.tensor(-2.5))
        self.last_kl = None

    def forward(self, emo_index: torch.Tensor, step: torch.Tensor,
                audio_feat: Optional[torch.Tensor] = None,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emo_index.shape[0]
        C = self.feature_dim
        step = step.to(emo_index.device).long()
        if drop_mask is not None and bool(drop_mask.all()):
            self.last_kl = None
            return self.null_tokens.expand(B, -1, -1)

        emo = self.emo_embed(emo_index)
        step_emb = self.step_pe.pe[0, step].to(emo.device)
        if audio_feat is None:
            audio_summary = torch.zeros_like(emo)
        else:
            audio_summary = self.audio_proj(audio_feat.mean(dim=1))
        cond = torch.cat([emo, step_emb, audio_summary], dim=-1)
        mu_logvar = self.prior(cond).view(B, self.n_latent_tokens, 2, C)
        mu = mu_logvar[:, :, 0]
        logvar = mu_logvar[:, :, 1].clamp(-8.0, 4.0)

        if self.training:
            eps = torch.randn_like(mu)
            z = mu + torch.exp(0.5 * logvar) * eps
            self.last_kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        else:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            self.last_kl = None

        z = z * torch.sigmoid(self.latent_scale)
        if drop_mask is not None:
            null = self.null_tokens.expand(B, -1, -1)
            z = torch.where(drop_mask.view(B, 1, 1), null, z)
        return z


class StochasticEmotionMemory(nn.Module):
    """Combine deterministic emotion basis tokens with stochastic affective tokens."""

    def __init__(self, feature_dim: int, emo_classes: int, n_diff_steps: int):
        super().__init__()
        self.basis = EmotionMotionBasisBank(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            n_diff_steps=n_diff_steps,
            k_coarse=2,
            k_dynamic=4,
            k_detail=2,
        )
        self.prior = StochasticAffectivePrior(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            n_diff_steps=n_diff_steps,
            n_latent_tokens=2,
        )

    @property
    def last_kl(self):
        return self.prior.last_kl

    def forward(self, emo_index: torch.Tensor, step: torch.Tensor,
                audio_feat: Optional[torch.Tensor] = None,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        basis_tokens = self.basis(emo_index, step, audio_feat=audio_feat, drop_mask=drop_mask)
        latent_tokens = self.prior(emo_index, step, audio_feat=audio_feat, drop_mask=drop_mask)
        return torch.cat([basis_tokens, latent_tokens], dim=1)


class DitTalkingHead(_V5DitTalkingHead):
    """v6 model: v5 basis bank + stochastic affective latent tokens."""

    min_audio_cfg = 1.0
    min_emotion_cfg = 0.05
    default_cfg_schedule = "bell"

    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=25,
                 audio_model="hubert", feature_dim=512, n_diff_steps=50,
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
        self.emotion_memory = StochasticEmotionMemory(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            n_diff_steps=n_diff_steps,
        )
        # Slightly lower emotion scale than v5 because stochastic tokens add capacity.
        self.denoising_net = DecoupledAudioEmotionDenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
            n_diff_steps=n_diff_steps,
            use_indicator=True,
            use_global_audio=True,
            audio_global_scale=0.20,
            emotion_scale=0.30,
            orthogonalize_emotion=True,
        )
        self.to(device)

    def encode_emotion(self, emo_index, step=None, audio_feat=None, drop_mask=None):
        if step is None:
            step = torch.full((emo_index.shape[0],), self.diffusion_sched.num_steps,
                              device=self.device, dtype=torch.long)
        return self.emotion_memory(emo_index, step, audio_feat=audio_feat, drop_mask=drop_mask)

    def get_aux_loss(self):
        """Optional KL term for a custom trainer; safe to ignore in train_v6.py."""
        kl = self.emotion_memory.last_kl
        if kl is None:
            return torch.tensor(0.0, device=self.device)
        return 1e-4 * kl

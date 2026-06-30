"""v5: Orthogonal Emotion Motion Basis Bank.

This version builds on v4 and adds a stronger, paper-oriented emotion condition:

1. Emotion is represented by a learnable class-specific motion basis bank
   [num_emotions, K, C], inspired by EDTalk's component basis banks and
   DICE-Talk's emotion-bank retrieval.
2. The bank is step-aware: diffusion timestep gates control how strongly the
   coarse / dynamic / detail groups are exposed.
3. The bank is audio-aware: a summary of the current audio queries the emotion
   basis, making the same emotion adapt to different prosody.
4. The denoiser uses local audio cross-attention, global audio cross-attention,
   and emotion cross-attention. Emotion updates are orthogonalized against the
   local audio update in hidden space, preventing emotion from following the
   audio-sensitive direction that controls lip synchronization.

No explicit lip/non-lip keypoint split is used; decoupling is done in hidden
motion space and attention-update directions.
"""

from typing import Optional

import torch
import torch.nn as nn

from .common import PositionalEncoding
from .emotion_dit_v4 import DitTalkingHead as _V4DitTalkingHead
from .emotion_dit_v4 import DecoupledAudioEmotionDenoisingNetwork


class EmotionMotionBasisBank(nn.Module):
    """Class-specific emotion motion basis bank with step/audio-aware retrieval."""

    def __init__(self, feature_dim: int, emo_classes: int, n_diff_steps: int,
                 k_coarse: int = 2, k_dynamic: int = 4, k_detail: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.emo_classes = emo_classes
        self.n_diff_steps = n_diff_steps
        self.k_coarse = k_coarse
        self.k_dynamic = k_dynamic
        self.k_detail = k_detail
        self.k_total = k_coarse + k_dynamic + k_detail

        self.label_embed = nn.Embedding(emo_classes, feature_dim)
        self.basis = nn.Parameter(torch.randn(emo_classes, self.k_total, feature_dim) * 0.02)
        self.null_tokens = nn.Parameter(torch.zeros(1, self.k_total + 2, feature_dim))

        self.step_pe = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.step_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.query_label = nn.Linear(feature_dim, feature_dim)
        self.query_audio = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.phase_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 3),
        )
        self.audio_mix = nn.Parameter(torch.tensor(0.0))
        self.intensity = nn.Parameter(torch.tensor(0.0))

    def _attend(self, query: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        key = self.key_proj(tokens)
        value = self.value_proj(tokens)
        attn = torch.softmax(torch.matmul(query, key.transpose(-1, -2)) / (query.shape[-1] ** 0.5), dim=-1)
        return torch.matmul(attn, value)

    def _apply_phase_gate(self, tokens: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        step_emb = self.step_mlp(self.step_pe.pe[0, step].to(tokens.device))
        rho = (step.float() / float(self.n_diff_steps)).clamp(0, 1)
        mid = (1.0 - torch.abs(rho - 0.5) * 2.0).clamp(0, 1)
        # Conservative non-zero prior to avoid the motion collapse seen in the old v2.
        prior = torch.stack([
            0.45 + 0.55 * rho,
            0.45 + 0.55 * mid,
            0.45 + 0.55 * (1.0 - rho),
        ], dim=-1)
        learned = torch.sigmoid(self.phase_gate(step_emb))
        gate = prior * learned
        coarse = tokens[:, :self.k_coarse] * gate[:, 0].view(B, 1, 1)
        dynamic = tokens[:, self.k_coarse:self.k_coarse + self.k_dynamic] * gate[:, 1].view(B, 1, 1)
        detail = tokens[:, self.k_coarse + self.k_dynamic:] * gate[:, 2].view(B, 1, 1)
        return torch.cat([coarse, dynamic, detail], dim=1)

    def forward(self, emo_index: torch.Tensor, step: torch.Tensor,
                audio_feat: Optional[torch.Tensor] = None,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emo_index.shape[0]
        step = step.to(emo_index.device).long()
        if drop_mask is not None and bool(drop_mask.all()):
            return self.null_tokens.expand(B, -1, -1)

        raw_basis = self.basis[emo_index]
        basis_tokens = self.token_proj(raw_basis)
        basis_tokens = self._apply_phase_gate(basis_tokens, step)

        label = self.label_embed(emo_index).unsqueeze(1)
        label_query = self.query_label(label)
        label_retrieval = self._attend(label_query, basis_tokens)

        if audio_feat is not None:
            audio_summary = audio_feat.mean(dim=1, keepdim=True)
            audio_query = self.query_audio(audio_summary)
            audio_retrieval = self._attend(audio_query, basis_tokens)
            retrieval = label_retrieval + torch.tanh(self.audio_mix) * audio_retrieval
        else:
            retrieval = label_retrieval

        # A compact global style token helps the attention layer use the basis tokens.
        style = label + torch.tanh(self.intensity) * retrieval
        emotion_memory = torch.cat([style, retrieval, basis_tokens], dim=1)

        if drop_mask is not None:
            null = self.null_tokens.expand(B, emotion_memory.shape[1], -1)
            emotion_memory = torch.where(drop_mask.view(B, 1, 1), null, emotion_memory)
        return emotion_memory


class DitTalkingHead(_V4DitTalkingHead):
    """v5 model: emotion motion basis bank + hidden-direction orthogonalization."""

    min_audio_cfg = 1.0
    min_emotion_cfg = 0.10
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
        self.emotion_basis_bank = EmotionMotionBasisBank(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            n_diff_steps=n_diff_steps,
            k_coarse=2,
            k_dynamic=4,
            k_detail=2,
        )
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
            emotion_scale=0.35,
            orthogonalize_emotion=True,
        )
        self.to(device)

    def encode_emotion(self, emo_index, step=None, audio_feat=None, drop_mask=None):
        if step is None:
            step = torch.full((emo_index.shape[0],), self.diffusion_sched.num_steps,
                              device=self.device, dtype=torch.long)
        return self.emotion_basis_bank(emo_index, step, audio_feat=audio_feat, drop_mask=drop_mask)

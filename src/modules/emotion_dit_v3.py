"""v3: DICE-style emotion bank + audio-aware retrieval.

This variant keeps the v1 DICE/IP-Adapter dual cross-attention denoiser, but
replaces the single label embedding with a lightweight emotion code bank:

- each emotion class owns several learnable prototype tokens;
- a label query retrieves a class-specific emotion prototype;
- an optional audio query retrieves a prosody-aware prototype from the same bank;
- the final emotion memory is a small token set used by the emotion cross-attn.

This is the closest ADEF-side analogue of DICE-Talk's emotion-bank retrieval,
while remaining lightweight enough for motion diffusion.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_dit_v1 import DitTalkingHead as _V1DitTalkingHead


class EmotionBankEncoder(nn.Module):
    def __init__(self, feature_dim: int, emo_classes: int, num_codes: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.emo_classes = emo_classes
        self.num_codes = num_codes
        self.label_embed = nn.Embedding(emo_classes, feature_dim)
        self.codebook = nn.Parameter(torch.randn(emo_classes, num_codes, feature_dim) * 0.02)

        self.q_label = nn.Linear(feature_dim, feature_dim)
        self.q_audio = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.token_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.audio_mix = nn.Parameter(torch.tensor(0.0))
        self.null_token = nn.Parameter(torch.zeros(1, 1, feature_dim))

    def _attend(self, query: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        q = query
        k = self.k_proj(codes)
        v = self.v_proj(codes)
        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1)
        return torch.matmul(attn, v)

    def forward(self, emo_index: torch.Tensor, audio_feat: Optional[torch.Tensor] = None,
                drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emo_index.shape[0]
        if drop_mask is not None and bool(drop_mask.all()):
            return self.null_token.expand(B, self.num_codes + 1, -1)

        label = self.label_embed(emo_index).unsqueeze(1)
        codes = self.codebook[emo_index]
        label_query = self.q_label(label)
        label_retrieval = self._attend(label_query, codes)

        if audio_feat is not None:
            audio_summary = audio_feat.mean(dim=1, keepdim=True)
            audio_query = self.q_audio(audio_summary)
            audio_retrieval = self._attend(audio_query, codes)
            retrieval = label_retrieval + torch.tanh(self.audio_mix) * audio_retrieval
        else:
            retrieval = label_retrieval

        retrieval = self.out(retrieval)
        tokens = self.token_proj(codes)
        emotion_memory = torch.cat([retrieval, tokens], dim=1)

        if drop_mask is not None:
            null = self.null_token.expand(B, emotion_memory.shape[1], -1)
            emotion_memory = torch.where(drop_mask.view(B, 1, 1), null, emotion_memory)
        return emotion_memory


class DitTalkingHead(_V1DitTalkingHead):
    """v3: emotion-bank encoding + v1 dual cross-attention denoiser."""

    min_audio_cfg = 1.0
    min_emotion_cfg = 0.10

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
        self.emotion_bank = EmotionBankEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            num_codes=8,
        ).to(self.device)

    def encode_emotion(self, emo_index, step=None, audio_feat=None, drop_mask=None):
        return self.emotion_bank(emo_index, audio_feat=audio_feat, drop_mask=drop_mask)

"""
2026年7月2日16:14:48
基于emotion_dit_clean，ChatGPT改的

Clean dual-attention sanity model with emotion basis-token encoding.

This file extends src/modules/emotion_dit_clean.py and changes only the emotion
encoding, keeping the clean dual-attention denoising backbone untouched.

Motivation
----------
The original clean model still uses a weak emotion condition:

    emo_index -> nn.Embedding -> [B, 1, C]

For decoupled audio/emotion cross-attention, this single token is usually too
low-capacity. It cannot represent multiple implicit emotional motion directions
such as global tone, expression tendency, head-motion bias, or micro-expression
style. Following the design intuition of emotion banks / motion basis tokens in
recent emotional talking-head works, this version uses:

    emo_index -> class-specific emotion basis bank -> [B, K, C]

where K learnable tokens are used as emotion memory by the existing emotion
cross-attention in emotion_dit_clean.py.

Important constraints
---------------------
1. This is still a sanity version, not the full v5/v6 method.
2. No audio-emotion interaction, stochastic prior, AU/DECA supervision, or
   explicit lip/non-lip split is introduced here.
3. CFG dropout remains clean: when emotion is dropped, the emotion memory becomes
   a learned null basis of shape [B, K, C].
4. The denoising network, audio mask, sample CFG, start features, training return
   values, and all public method signatures are inherited from emotion_dit_clean.

Suggested first comparison
--------------------------
Compare against emotion_dit_clean.py with the same training settings. If this
file improves emotion strength without hurting simple_loss/lipsync, then the
multi-token basis memory is useful. If not, the issue is likely the dual-attn
backbone or training objective rather than emotion-token capacity.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .emotion_dit_clean import DitTalkingHead as _CleanDitTalkingHead


class EmotionBasisEncoder(nn.Module):
    """Class-specific multi-token emotion basis encoder.

    Parameters
    ----------
    feature_dim:
        Hidden feature dimension C used by the motion/audio/emotion transformer.
    emo_classes:
        Number of discrete emotion labels.
    num_basis_tokens:
        Number of learnable emotion memory tokens K for each emotion class.

    Input
    -----
    emo_index: [B]
    drop_mask: optional [B] bool

    Output
    ------
    emotion_memory: [B, K, C]

    Unlike the original `nn.Embedding(emo_classes, C)` which produces [B, 1, C],
    this module gives each emotion a small basis-token set. These tokens are
    implicit motion/emotion basis vectors, not explicit face-part tokens.
    """

    def __init__(
        self,
        feature_dim: int,
        emo_classes: int,
        num_basis_tokens: int = 8,
        init_std: float = 0.02,
        use_token_mlp: bool = True,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.emo_classes = emo_classes
        self.num_basis_tokens = num_basis_tokens

        self.emotion_basis = nn.Parameter(
            torch.randn(emo_classes, num_basis_tokens, feature_dim) * init_std
        )
        self.null_basis = nn.Parameter(torch.zeros(1, num_basis_tokens, feature_dim))
        self.token_pos = nn.Parameter(torch.randn(1, num_basis_tokens, feature_dim) * init_std)

        if use_token_mlp:
            self.token_proj = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.SiLU(),
                nn.Linear(feature_dim, feature_dim),
            )
        else:
            self.token_proj = nn.Identity()

    def forward(self, emo_index: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emo_index.shape[0]
        tokens = self.emotion_basis[emo_index] + self.token_pos
        tokens = self.token_proj(tokens)

        if drop_mask is not None:
            null_tokens = self.null_basis.expand(B, -1, -1)
            tokens = torch.where(drop_mask.view(B, 1, 1), null_tokens, tokens)
        return tokens


class DitTalkingHead(_CleanDitTalkingHead):
    """Clean dual-attn model with [B, K, C] emotion basis encoding.

    This class keeps all behavior from emotion_dit_clean.DitTalkingHead except
    `_encode_emotion`, which now returns a multi-token class-specific emotion
    memory instead of a single embedding token.
    """

    def __init__(
        self,
        device="cuda",
        target="sample",
        architecture="decoder",
        motion_feat_dim=70,
        fps=25,
        n_motions=100,
        n_prev_motions=10,
        audio_model="hubert",
        feature_dim=512,
        n_diff_steps=500,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,emotion",
        emo_classes=8,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 3,
        decoder_dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale: float = 0.5,
        num_emotion_basis_tokens: int = 8,
    ):
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
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            decoder_dropout=decoder_dropout,
            audio_scale=audio_scale,
            emotion_scale=emotion_scale,
        )
        self.emotion_basis_encoder = EmotionBasisEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            num_basis_tokens=num_emotion_basis_tokens,
        ).to(self.device)

    def _encode_emotion(self, emo_index: torch.Tensor, drop_mask: Optional[torch.Tensor] = None):
        return self.emotion_basis_encoder(emo_index, drop_mask=drop_mask)

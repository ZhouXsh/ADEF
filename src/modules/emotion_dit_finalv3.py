"""Final v3: hierarchical label/utterance/frame emotion-audio DiT.

This file combines the clean dual-branch DiT idea with the emotion2vec path.
The DiT still has only two cross-attention branches:

    1. original audio branch: preserves speech content and lip synchronization;
    2. emotion-audio branch: attends to audio that has been recalibrated by a
       hierarchical emotion encoder.

The hierarchical emotion encoder is the key module:

    label y -> emotion prototype bank P_y
    utterance emotion2vec u -> global calibration of P_y
    frame emotion2vec F -> target-aware temporal affect dynamics F_y
    audio A queries [P_y^u, u, F_y] -> emotion residual on the audio timeline
    A_e = A + alpha * gated_residual

This makes the three emotion features act on audio before DiT attention.  The
model story becomes "audio branch for lip-sync, emotion-audio branch for affect"
rather than four independent condition branches.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .emotion_dit_finalv1 import DitTalkingHead as _FinalV1DitTalkingHead


class HierarchicalEmotionAudioEncoder(nn.Module):
    """Target-anchored hierarchical emotion-audio encoder.

    Inputs:
        audio_feat: [B, L, C], normalized audio feature.
        emo_index: [B], target emotion label.
        emo_utt_feat: [B, D_e2v] or [B, 1, D_e2v], utterance-level emotion2vec.
        emo_frame_feat: [B, T, D_e2v], frame-level emotion2vec.

    Output:
        emotion_audio_feat: [B, L, C], audio feature recalibrated by target emotion.
    """

    def __init__(
        self,
        feature_dim: int,
        emo_classes: int = 8,
        e2v_dim: int = 1024,
        num_label_tokens: int = 8,
        n_heads: int = 8,
        residual_init: float = 0.05,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.emo_classes = emo_classes
        self.e2v_dim = e2v_dim
        self.num_label_tokens = num_label_tokens

        self.label_basis = nn.Parameter(
            torch.randn(emo_classes, num_label_tokens, feature_dim) * init_std
        )
        self.null_label_basis = nn.Parameter(torch.zeros(1, num_label_tokens, feature_dim))
        self.label_pos = nn.Parameter(torch.randn(1, num_label_tokens, feature_dim) * init_std)
        self.label_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.utt_proj = nn.Sequential(
            nn.LayerNorm(e2v_dim),
            nn.Linear(e2v_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.null_utt = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.global_label_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 2 * feature_dim),
        )

        self.frame_proj = nn.Sequential(
            nn.LayerNorm(e2v_dim),
            nn.Linear(e2v_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.null_frame = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.frame_norm = nn.LayerNorm(feature_dim)
        self.target_frame_mod = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2 * feature_dim),
        )

        self.audio_to_emotion_attn = nn.MultiheadAttention(
            feature_dim, n_heads, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * feature_dim),
            nn.Linear(3 * feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    @staticmethod
    def _resize_temporal(feat: torch.Tensor, target_len: int) -> torch.Tensor:
        if feat.shape[1] == target_len:
            return feat
        feat = feat.transpose(1, 2)
        feat = F.interpolate(feat, size=target_len, mode="linear", align_corners=False)
        return feat.transpose(1, 2).contiguous()

    @staticmethod
    def _apply_drop(x: torch.Tensor, null_x: torch.Tensor, drop_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if drop_mask is None:
            return x
        while null_x.shape[0] != x.shape[0]:
            null_x = null_x.expand(x.shape[0], -1, -1)
        return torch.where(drop_mask.view(x.shape[0], 1, 1), null_x, x)

    def _encode_label(self, emo_index: torch.Tensor, drop_mask: Optional[torch.Tensor]) -> torch.Tensor:
        B = emo_index.shape[0]
        label_tokens = self.label_basis[emo_index] + self.label_pos
        label_tokens = self.label_proj(label_tokens)
        null_tokens = self.null_label_basis.expand(B, -1, -1)
        return self._apply_drop(label_tokens, null_tokens, drop_mask)

    def _encode_utterance(
        self,
        emo_utt_feat: Optional[torch.Tensor],
        batch_size: int,
        device,
        drop_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if emo_utt_feat is None:
            utt_token = self.null_utt.expand(batch_size, -1, -1)
        else:
            emo_utt_feat = emo_utt_feat.to(device)
            if emo_utt_feat.ndim == 3:
                emo_utt_feat = emo_utt_feat.squeeze(1)
            if emo_utt_feat.shape[-1] == self.feature_dim:
                utt_token = emo_utt_feat.unsqueeze(1)
            else:
                utt_token = self.utt_proj(emo_utt_feat).unsqueeze(1)
        null_utt = self.null_utt.expand(batch_size, -1, -1)
        return self._apply_drop(utt_token, null_utt, drop_mask)

    def _encode_frame(
        self,
        emo_frame_feat: Optional[torch.Tensor],
        batch_size: int,
        target_len: int,
        device,
        drop_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if emo_frame_feat is None:
            frame_tokens = self.null_frame.expand(batch_size, target_len, -1)
        else:
            emo_frame_feat = emo_frame_feat.to(device)
            emo_frame_feat = self._resize_temporal(emo_frame_feat, target_len)
            if emo_frame_feat.shape[-1] == self.feature_dim:
                frame_tokens = emo_frame_feat
            else:
                frame_tokens = self.frame_proj(emo_frame_feat)
        null_frame = self.null_frame.expand(batch_size, target_len, -1)
        return self._apply_drop(frame_tokens, null_frame, drop_mask)

    def forward(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        drop_emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, C = audio_feat.shape
        device = audio_feat.device
        if drop_emotion is not None:
            drop_emotion = drop_emotion.to(device=device, dtype=torch.bool).view(B)

        # 1) label gives target emotion direction.
        label_tokens = self._encode_label(emo_index, drop_emotion)

        # 2) utterance-level emotion2vec calibrates global intensity/style.
        utt_token = self._encode_utterance(emo_utt_feat, B, device, drop_emotion)
        gamma_u, beta_u = self.global_label_mod(utt_token).chunk(2, dim=-1)
        label_tokens = label_tokens * (1 + gamma_u) + beta_u

        # 3) frame-level emotion2vec gives local affect dynamics, redirected by label+utterance.
        frame_tokens = self._encode_frame(emo_frame_feat, B, L, device, drop_emotion)
        target_context = label_tokens.mean(dim=1) + utt_token.squeeze(1)
        gamma_f, beta_f = self.target_frame_mod(target_context).chunk(2, dim=-1)
        frame_tokens = self.frame_norm(frame_tokens) * (1 + gamma_f.unsqueeze(1)) + beta_f.unsqueeze(1)
        null_frame = self.null_frame.expand(B, L, -1)
        frame_tokens = self._apply_drop(frame_tokens, null_frame, drop_emotion)

        # 4) audio queries target-aware emotion memory; output remains audio-time aligned.
        emotion_memory = torch.cat([label_tokens, utt_token, frame_tokens], dim=1)
        residual = self.audio_to_emotion_attn(
            query=audio_feat,
            key=emotion_memory,
            value=emotion_memory,
            need_weights=False,
        )[0]
        gate = self.gate(torch.cat([audio_feat, residual, frame_tokens], dim=-1))
        residual = self.out_proj(gate * residual)
        return audio_feat + self.residual_scale.tanh() * residual


class DitTalkingHead(_FinalV1DitTalkingHead):
    """Final v3: original audio branch + hierarchical emotion2vec-audio branch."""

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
        emotion_audio_scale: float = 0.5,
        e2v_dim: int = 1024,
        num_label_tokens: int = 8,
        emotion_audio_residual_init: float = 0.05,
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
            emotion_audio_scale=emotion_audio_scale,
        )
        self.hierarchical_emotion_audio_encoder = HierarchicalEmotionAudioEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            e2v_dim=e2v_dim,
            num_label_tokens=num_label_tokens,
            n_heads=n_heads,
            residual_init=emotion_audio_residual_init,
        ).to(self.device)

    def _build_emotion_audio(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        drop_emotion: Optional[torch.Tensor] = None,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        audio_feat = self.audio_norm(audio_feat)
        return self.hierarchical_emotion_audio_encoder(
            audio_feat=audio_feat,
            emo_index=emo_index,
            emo_utt_feat=emo_utt_feat,
            emo_frame_feat=emo_frame_feat,
            drop_emotion=drop_emotion,
        )

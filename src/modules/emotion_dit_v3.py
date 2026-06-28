"""Layer-wise phase-aware emotion adapters for implicit-keypoint motion diffusion.

Version v3 is the strongest variant:
- same phase-aware emotion tokens as v2;
- replaces the monolithic TransformerDecoder with an equivalent layer list;
- inserts zero-init emotion hidden adapters after the last several decoder layers.

This explores whether emotion should adapt only the final hidden representation (v1/v2)
or progressively refine the high-level denoising representation in late decoder layers.
"""

import torch
import torch.nn as nn

from .emotion_dit import DenoisingNetwork as _BaseDenoisingNetwork
from .emotion_dit_v1 import EmotionHiddenAdapter
from .emotion_dit_v2 import PhaseAwareEmotionEncoder, DitTalkingHead as _V2DitTalkingHead


class DenoisingNetworkV3(_BaseDenoisingNetwork):
    def __init__(self, *args, emotion_adapter_layers: int = 2, emotion_adapter_scale: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion_adapter_layers = max(1, int(emotion_adapter_layers))
        self.emotion_adapter_start = max(0, self.n_layers - self.emotion_adapter_layers)

        # Replace nn.TransformerDecoder by an explicit list so that we can insert
        # emotion adapters after selected high-level decoder layers.
        self.transformer_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=self.feature_dim,
                nhead=self.n_heads,
                dim_feedforward=self.mlp_ratio * self.feature_dim,
                activation='gelu',
                batch_first=True,
            )
            for _ in range(self.n_layers)
        ])
        self.transformer = None
        self.emotion_adapter = EmotionHiddenAdapter(
            feature_dim=self.feature_dim,
            n_heads=self.n_heads,
            mlp_ratio=self.mlp_ratio,
            adapter_scale=emotion_adapter_scale,
        )
        self.last_emo_residual_norm = None

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None, emo_tokens=None):
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                indicator,
            ], dim=1)
            indicator = indicator.unsqueeze(-1)

        if self.architecture == 'decoder':
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        feat_out = self.feature_proj(feats_in)
        if self.use_learnable_pe:
            feat_out = feat_out + self.PE + diff_step_embedding
        else:
            feat_out = self.PE(feat_out) + diff_step_embedding

        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
        step_emb = diff_step_embedding.squeeze(1)
        residual_norms = []
        for layer_idx, layer in enumerate(self.transformer_layers):
            feat_out = layer(feat_out, audio_feat_in, memory_mask=self.alignment_mask)
            if emo_tokens is not None and layer_idx >= self.emotion_adapter_start:
                feat_out, emo_residual = self.emotion_adapter(feat_out, emo_tokens, step_emb)
                residual_norms.append(emo_residual.detach().pow(2).mean())

        if residual_norms:
            self.last_emo_residual_norm = torch.stack(residual_norms).mean()
        else:
            self.last_emo_residual_norm = None

        motion_feat_target = self.motion_dec(feat_out)
        return motion_feat_target


class DitTalkingHead(_V2DitTalkingHead):
    """v3: phase-aware tokens + adapters inserted after the last decoder layers."""

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
        self.denoising_net = DenoisingNetworkV3(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
            emotion_adapter_layers=2,
            emotion_adapter_scale=0.5,
        )
        self.to(device)

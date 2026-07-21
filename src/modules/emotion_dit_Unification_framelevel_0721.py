import torch
import torch.nn as nn

from .emotion_dit_Unification import (
    DiffusionSchedule,
    DitTalkingHead as BaseDitTalkingHead,
)
from .emotion_dit_timestep_0714 import (
    DenoisingNetwork as BaseDenoisingNetwork,
    DiTDecoderLayer as BaseDiTDecoderLayer,
    modulate,
)


class FrameLevelDiTDecoderLayer(BaseDiTDecoderLayer):
    """Inject emotion2vec frame features before self-attention in every layer."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__(d_model, nhead, dim_feedforward, dropout=dropout)
        self.framelevel_modulation = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )

    def forward(self, tgt, memory, t_emb, frame_cond,
                memory_mask=None, tgt_mask=None):
        # Keep the original timestep adaLN-Zero gates and the original
        # cross-attention / FFN modulation paths unchanged.
        (_, _, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = self.adaLN_modulation(t_emb).chunk(9, dim=-1)

        # frame_cond: [emotion2vec_frame(512), timestep(512)] -> 1024
        # MLP: 1024 -> 1024 -> [scale(512), shift(512)].
        scale_sa, shift_sa = self.framelevel_modulation(frame_cond).chunk(2, dim=-1)

        # Frame-wise FiLM modulation before noisy-motion self-attention.
        h = modulate(self.norm1(tgt), shift_sa, scale_sa)
        sa = self.self_attn(h, h, h, attn_mask=tgt_mask, need_weights=False)[0]
        tgt = tgt + gate_sa * sa

        # Original audio cross-attention.
        h = modulate(self.norm2(tgt), shift_ca, scale_ca)
        ca = self.cross_attn(h, memory, memory,
                             attn_mask=memory_mask, need_weights=False)[0]
        tgt = tgt + gate_ca * ca

        # Original feed-forward network.
        h = modulate(self.norm3(tgt), shift_ff, scale_ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(h))))
        tgt = tgt + gate_ff * ff
        return tgt


class FrameLevelDiTDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FrameLevelDiTDecoderLayer(
                d_model, nhead, dim_feedforward, dropout=dropout
            ) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, t_emb, frame_cond,
                memory_mask=None, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, t_emb, frame_cond,
                        memory_mask=memory_mask, tgt_mask=tgt_mask)
        return tgt


class DenoisingNetwork(BaseDenoisingNetwork):
    """Minimal extension of the original denoiser for frame-level FiLM."""

    def __init__(self, *args, emotion2vec_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion2vec_dim = emotion2vec_dim
        self.framelevel_feature_map = nn.Linear(emotion2vec_dim, self.feature_dim)
        self.transformer = FrameLevelDiTDecoder(
            d_model=self.feature_dim,
            nhead=self.n_heads,
            dim_feedforward=self.mlp_ratio * self.feature_dim,
            num_layers=self.n_layers,
        )
        self._frame_level_feat = None
        self._prev_frame_level_feat = None
        self.to(self.device)

    def set_frame_level_condition(self, frame_level_feat,
                                  prev_frame_level_feat=None):
        self._frame_level_feat = frame_level_feat
        self._prev_frame_level_feat = prev_frame_level_feat

    def clear_frame_level_condition(self):
        self._frame_level_feat = None
        self._prev_frame_level_feat = None

    def _get_frame_condition(self, batch_size, seq_len, dtype, device):
        frame = self._frame_level_feat
        prev = self._prev_frame_level_feat
        if frame is None:
            raise ValueError('frame_level_feat is required for the frame-level model')

        if prev is None:
            prev = torch.zeros(
                frame.shape[0], self.n_prev_motions, self.emotion2vec_dim,
                dtype=frame.dtype, device=frame.device,
            )

        expected_frame = (frame.shape[0], self.n_motions, self.emotion2vec_dim)
        expected_prev = (frame.shape[0], self.n_prev_motions, self.emotion2vec_dim)
        if tuple(frame.shape) != expected_frame:
            raise ValueError(
                f'frame_level_feat must be {expected_frame}, got {tuple(frame.shape)}'
            )
        if tuple(prev.shape) != expected_prev:
            raise ValueError(
                f'prev_frame_level_feat must be {expected_prev}, got {tuple(prev.shape)}'
            )

        # During joint CFG sampling the base model doubles the batch as
        # [unconditional, conditional]. Use zeros for the unconditional branch.
        if batch_size == 2 * frame.shape[0]:
            frame = torch.cat([torch.zeros_like(frame), frame], dim=0)
            prev = torch.cat([torch.zeros_like(prev), prev], dim=0)
        elif batch_size != frame.shape[0]:
            raise ValueError(
                f'Condition batch {frame.shape[0]} does not match denoiser batch {batch_size}'
            )

        frame = torch.cat([prev, frame], dim=1)
        if frame.shape[1] != seq_len:
            raise ValueError(
                f'Frame condition length {frame.shape[1]} != motion length {seq_len}'
            )
        return self.framelevel_feature_map(frame.to(device=device, dtype=dtype))

    def forward(self, motion_feat, audio_feat, prev_motion_feat,
                prev_audio_feat, step, indicator=None):
        # Timestep: (B,) -> (B, 1, 512).
        t_emb = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions),
                            device=indicator.device),
                indicator,
            ], dim=1).unsqueeze(-1)

        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        feats_in = self.feature_proj(feats_in)
        feats_in = feats_in + self.PE if self.use_learnable_pe else self.PE(feats_in)

        # emotion2vec: (B, 125, 1024) -> (B, 125, 512).
        frame_feat = self._get_frame_condition(
            feats_in.shape[0], feats_in.shape[1], feats_in.dtype, feats_in.device
        )
        # Timestep broadcast: (B, 1, 512) -> (B, 125, 512).
        t_per_frame = t_emb.expand(-1, feats_in.shape[1], -1)
        # Concatenate on feature dimension: (B, 125, 1024).
        frame_cond = torch.cat([frame_feat, t_per_frame], dim=-1)

        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
        feat_out = self.transformer(
            feats_in, audio_feat_in, t_emb, frame_cond,
            memory_mask=self.alignment_mask,
        )
        return self.motion_dec(feat_out)


class DitTalkingHead(BaseDitTalkingHead):
    """Unification model with emotion2vec frame-level modulation."""

    def __init__(self, *args, emotion2vec_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion2vec_dim = emotion2vec_dim
        self.denoising_net = DenoisingNetwork(
            device=self.device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            n_diff_steps=self.diffusion_sched.num_steps,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=self.feature_dim,
            emotion2vec_dim=emotion2vec_dim,
        )
        self.to(self.device)

    def _set_frame_condition(self, frame_level_feat,
                             prev_frame_level_feat=None, training=False):
        if frame_level_feat is None:
            raise ValueError('frame_level_feat is required')
        expected = (frame_level_feat.shape[0], self.n_motions, self.emotion2vec_dim)
        if tuple(frame_level_feat.shape) != expected:
            raise ValueError(
                f'frame_level_feat must be {expected}, got {tuple(frame_level_feat.shape)}'
            )

        # Match the base joint-condition dropout approximately while keeping
        # the original Unification implementation untouched.
        if training:
            drop = (torch.rand(frame_level_feat.shape[0], device=frame_level_feat.device) < 0.1)
            frame_level_feat = torch.where(
                drop[:, None, None], torch.zeros_like(frame_level_feat), frame_level_feat
            )
            if prev_frame_level_feat is not None:
                prev_frame_level_feat = torch.where(
                    drop[:, None, None],
                    torch.zeros_like(prev_frame_level_feat),
                    prev_frame_level_feat,
                )

        self.denoising_net.set_frame_level_condition(
            frame_level_feat, prev_frame_level_feat
        )

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None,
                prev_audio_feat=None, time_step=None, indicator=None,
                emo_index=None, frame_level_feat=None,
                prev_frame_level_feat=None):
        self._set_frame_condition(
            frame_level_feat, prev_frame_level_feat, training=self.training
        )
        try:
            return super().forward(
                motion_feat, audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                time_step=time_step,
                indicator=indicator,
                emo_index=emo_index,
            )
        finally:
            self.denoising_net.clear_frame_level_condition()

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None,
               prev_audio_feat=None, motion_at_T=None, indicator=None,
               cfg_mode=None, cfg_cond=None, cfg_scale=1.15,
               flexibility=0, dynamic_threshold=None, ret_traj=False,
               emo_index=None, frame_level_feat=None,
               prev_frame_level_feat=None):
        self._set_frame_condition(
            frame_level_feat, prev_frame_level_feat, training=False
        )
        try:
            return super().sample(
                audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                motion_at_T=motion_at_T,
                indicator=indicator,
                cfg_mode=cfg_mode,
                cfg_cond=cfg_cond,
                cfg_scale=cfg_scale,
                flexibility=flexibility,
                dynamic_threshold=dynamic_threshold,
                ret_traj=ret_traj,
                emo_index=emo_index,
            )
        finally:
            self.denoising_net.clear_frame_level_condition()


__all__ = ['DiffusionSchedule', 'DenoisingNetwork', 'DitTalkingHead']

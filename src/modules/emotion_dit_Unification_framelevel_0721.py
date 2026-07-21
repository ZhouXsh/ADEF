import torch
import torch.nn as nn

from .emotion_dit import (
    DiffusionSchedule,
    DenoisingNetwork as BaseDenoisingNetwork,
)
from .emotion_dit_Unification import DitTalkingHead as BaseDitTalkingHead


def modulate(x, shift, scale):
    """Frame-wise FiLM modulation used only before motion self-attention."""
    return x * (1 + scale) + shift


class FrameLevelTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Original TransformerDecoderLayer with one minimal change:
    frame-level scale/shift modulates the motion branch only before self-attention.

    Cross-attention and feed-forward paths keep the original PyTorch
    TransformerDecoderLayer computation and do not receive FiLM modulation.
    No gate parameter is used.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )

    def forward(
        self,
        tgt,
        memory,
        scale,
        shift,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        del tgt_is_causal, memory_is_causal

        # Only this self-attention input is frame-wise modulated.
        h = modulate(tgt, shift, scale)
        self_attn_out = self.self_attn(
            h,
            h,
            h,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = self.norm1(tgt + self.dropout1(self_attn_out))

        # Original audio cross-attention: no frame-level modulation, no gate.
        cross_attn_out = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))

        # Original feed-forward network: no frame-level modulation, no gate.
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt


class FrameLevelTransformerDecoder(nn.Module):
    """Stack decoder layers while reusing one shared scale/shift pair."""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            FrameLevelTransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt,
        memory,
        scale,
        shift,
        memory_mask=None,
        tgt_mask=None,
    ):
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                scale,
                shift,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
            )
        return tgt


class DenoisingNetwork(BaseDenoisingNetwork):
    """emotion_dit denoiser extended with frame-level pre-self-attention FiLM."""

    def __init__(self, *args, emotion2vec_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion2vec_dim = emotion2vec_dim

        # emotion2vec frame feature: 1024 -> 512.
        self.framelevel_feature_map = nn.Linear(
            self.emotion2vec_dim, self.feature_dim
        )

        # One shared modulation MLP for the whole denoising-network forward:
        # 1024 -> 1024 -> 1024, then split into scale(512) and shift(512).
        self.framelevel_modulation = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 2 * self.feature_dim),
            nn.SiLU(),
            nn.Linear(2 * self.feature_dim, 2 * self.feature_dim),
        )

        # Replace only the Transformer decoder implementation. Its layer order
        # remains self-attention -> cross-attention -> feed-forward.
        self.transformer = FrameLevelTransformerDecoder(
            d_model=self.feature_dim,
            nhead=self.n_heads,
            dim_feedforward=self.mlp_ratio * self.feature_dim,
            num_layers=self.n_layers,
        )

        self._frame_level_feat = None
        self._prev_frame_level_feat = None
        self.to(self.device)

    def set_frame_level_condition(
        self, frame_level_feat, prev_frame_level_feat=None
    ):
        self._frame_level_feat = frame_level_feat
        self._prev_frame_level_feat = prev_frame_level_feat

    def clear_frame_level_condition(self):
        self._frame_level_feat = None
        self._prev_frame_level_feat = None

    def _get_frame_condition(self, batch_size, seq_len, dtype, device):
        frame = self._frame_level_feat
        prev = self._prev_frame_level_feat
        if frame is None:
            raise ValueError(
                'frame_level_feat is required for the frame-level model'
            )

        if prev is None:
            prev = torch.zeros(
                frame.shape[0],
                self.n_prev_motions,
                self.emotion2vec_dim,
                dtype=frame.dtype,
                device=frame.device,
            )

        expected_frame = (
            frame.shape[0], self.n_motions, self.emotion2vec_dim
        )
        expected_prev = (
            frame.shape[0], self.n_prev_motions, self.emotion2vec_dim
        )
        if tuple(frame.shape) != expected_frame:
            raise ValueError(
                f'frame_level_feat must be {expected_frame}, '
                f'got {tuple(frame.shape)}'
            )
        if tuple(prev.shape) != expected_prev:
            raise ValueError(
                f'prev_frame_level_feat must be {expected_prev}, '
                f'got {tuple(prev.shape)}'
            )

        # During joint CFG sampling, the Unification model concatenates
        # [unconditional, conditional] along the batch dimension.
        if batch_size == 2 * frame.shape[0]:
            frame = torch.cat([torch.zeros_like(frame), frame], dim=0)
            prev = torch.cat([torch.zeros_like(prev), prev], dim=0)
        elif batch_size != frame.shape[0]:
            raise ValueError(
                f'Condition batch {frame.shape[0]} does not match '
                f'denoiser batch {batch_size}'
            )

        frame = torch.cat([prev, frame], dim=1)
        if frame.shape[1] != seq_len:
            raise ValueError(
                f'Frame condition length {frame.shape[1]} '
                f'!= motion length {seq_len}'
            )
        return self.framelevel_feature_map(
            frame.to(device=device, dtype=dtype)
        )

    def forward(
        self,
        motion_feat,
        audio_feat,
        prev_motion_feat,
        prev_audio_feat,
        step,
        indicator=None,
    ):
        # Diffusion timestep: (B,) -> (B, 1, 512).
        diff_step_embedding = self.diff_step_map(
            self.TE.pe[0, step]
        ).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat([
                torch.zeros(
                    (indicator.shape[0], self.n_prev_motions),
                    device=indicator.device,
                ),
                indicator,
            ], dim=1).unsqueeze(-1)

        # Motion/noise sequence: previous 25 + current 100 = 125 frames.
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        feats_in = self.feature_proj(feats_in)

        # Keep the original sequence positional encoding. The diffusion
        # timestep is no longer added once at the input; it is used below to
        # construct the frame-wise self-attention modulation parameters.
        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # emotion2vec: (B, 125, 1024) -> (B, 125, 512).
        frame_feat = self._get_frame_condition(
            feats_in.shape[0],
            feats_in.shape[1],
            feats_in.dtype,
            feats_in.device,
        )

        # timestep: (B, 1, 512) -> (B, 125, 512).
        timestep_per_frame = diff_step_embedding.expand(
            -1, feats_in.shape[1], -1
        )

        # Feature concat: (B, 125, 512 + 512) -> (B, 125, 1024).
        frame_condition = torch.cat(
            [frame_feat, timestep_per_frame], dim=-1
        )

        # Compute scale and shift exactly once, then reuse them before the
        # self-attention of every Transformer layer.
        scale, shift = self.framelevel_modulation(
            frame_condition
        ).chunk(2, dim=-1)

        audio_feat_in = torch.cat(
            [prev_audio_feat, audio_feat], dim=1
        )
        feat_out = self.transformer(
            feats_in,
            audio_feat_in,
            scale,
            shift,
            memory_mask=self.alignment_mask,
        )
        return self.motion_dec(feat_out)


class DitTalkingHead(BaseDitTalkingHead):
    """Unification model with emotion2vec frame-level self-attention FiLM."""

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

    def _set_frame_condition(
        self,
        frame_level_feat,
        prev_frame_level_feat=None,
        training=False,
    ):
        if frame_level_feat is None:
            raise ValueError('frame_level_feat is required')
        expected = (
            frame_level_feat.shape[0],
            self.n_motions,
            self.emotion2vec_dim,
        )
        if tuple(frame_level_feat.shape) != expected:
            raise ValueError(
                f'frame_level_feat must be {expected}, '
                f'got {tuple(frame_level_feat.shape)}'
            )

        # Preserve the existing frame-condition dropout behavior used by the
        # previous implementation for training compatibility.
        if training:
            drop = (
                torch.rand(
                    frame_level_feat.shape[0],
                    device=frame_level_feat.device,
                ) < 0.1
            )
            frame_level_feat = torch.where(
                drop[:, None, None],
                torch.zeros_like(frame_level_feat),
                frame_level_feat,
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

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
        frame_level_feat=None,
        prev_frame_level_feat=None,
    ):
        self._set_frame_condition(
            frame_level_feat,
            prev_frame_level_feat,
            training=self.training,
        )
        try:
            return super().forward(
                motion_feat,
                audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                time_step=time_step,
                indicator=indicator,
                emo_index=emo_index,
            )
        finally:
            self.denoising_net.clear_frame_level_condition()

    @torch.no_grad()
    def sample(
        self,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
        cfg_mode=None,
        cfg_cond=None,
        cfg_scale=1.15,
        flexibility=0,
        dynamic_threshold=None,
        ret_traj=False,
        emo_index=None,
        frame_level_feat=None,
        prev_frame_level_feat=None,
    ):
        self._set_frame_condition(
            frame_level_feat,
            prev_frame_level_feat,
            training=False,
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


__all__ = [
    'DiffusionSchedule',
    'DenoisingNetwork',
    'DitTalkingHead',
]

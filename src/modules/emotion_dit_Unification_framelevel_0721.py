## 大一统版本 + emotion2vec frame-level 调制。
## 基于 emotion_dit_Unification.py 直接复制并修改，便于独立探索与管理。
## 基础扩散、音频编码和标准 Transformer 结构来自 emotion_dit.py。
## frame-level 特征与扩散时间步只在每层噪声自注意力前进行 FiLM 调制；
## 交叉注意力和前馈网络保持 emotion_dit.py 的原始实现，不使用 gate。

import torch
import torch.nn as nn

from .emotion_dit import (
    DiffusionSchedule,
    DenoisingNetwork as BaseDenoisingNetwork,
    DitTalkingHead as BaseDitTalkingHead,
)


def modulate(x, shift, scale):
    """逐帧 FiLM 调制，只用于噪声自注意力输入。"""
    return x * (1 + scale) + shift


class FrameLevelTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    保持 emotion_dit.py 中 TransformerDecoderLayer 的总体顺序：
    自注意力 -> 交叉注意力 -> 前馈网络。

    唯一结构改动：在自注意力之前，使用共享的逐帧 scale/shift 调制 tgt。
    交叉注意力和前馈网络不进行额外调制，也不引入 gate。
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

        # 仅在噪声自注意力前进行逐帧调制。
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

        # 原始音频交叉注意力：不做 frame-level 调制，不使用 gate。
        cross_attn_out = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(cross_attn_out))

        # 原始前馈网络：不做 frame-level 调制，不使用 gate。
        ff_out = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt


class FrameLevelTransformerDecoder(nn.Module):
    """多层解码器，共享一次计算得到的 scale 和 shift。"""

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
    """在 emotion_dit.py 去噪网络上增加 frame-level 自注意力前调制。"""

    def __init__(self, *args, emotion2vec_dim=1024, **kwargs):
        super().__init__(*args, **kwargs)
        self.emotion2vec_dim = emotion2vec_dim

        # emotion2vec frame-level 特征：1024 -> 512。
        self.framelevel_feature_map = nn.Linear(
            self.emotion2vec_dim, self.feature_dim
        )

        # 每次去噪网络前向只计算一次：
        # 1024 -> 1024 -> 1024，随后拆分为 scale(512) 和 shift(512)。
        self.framelevel_modulation = nn.Sequential(
            nn.Linear(2 * self.feature_dim, 2 * self.feature_dim),
            nn.SiLU(),
            nn.Linear(2 * self.feature_dim, 2 * self.feature_dim),
        )

        # 仅替换 Transformer decoder；层内顺序保持不变。
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

        # 联合 CFG 推理时，batch 顺序为 [unconditional, conditional]。
        # 无条件分支使用全零 frame-level 条件。
        if batch_size == 2 * frame.shape[0]:
            frame = torch.cat([torch.zeros_like(frame), frame], dim=0)
            prev = torch.cat([torch.zeros_like(prev), prev], dim=0)
        elif batch_size != frame.shape[0]:
            raise ValueError(
                f'Condition batch {frame.shape[0]} does not match '
                f'denoiser batch {batch_size}'
            )

        # 历史 25 帧 + 当前 100 帧 = 125 帧。
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
        # 扩散时间步：(B,) -> (B, 1, 512)。
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

        # 历史运动与当前带噪运动沿时间维拼接。
        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        feats_in = self.feature_proj(feats_in)

        # 保留原始序列位置编码；时间步不再直接加到入口特征上。
        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        # emotion2vec：(B, 125, 1024) -> (B, 125, 512)。
        frame_feat = self._get_frame_condition(
            feats_in.shape[0],
            feats_in.shape[1],
            feats_in.dtype,
            feats_in.device,
        )

        # timestep：(B, 1, 512) -> (B, 125, 512)。
        timestep_per_frame = diff_step_embedding.expand(
            -1, feats_in.shape[1], -1
        )

        # 沿特征维拼接：(B, 125, 1024)。
        frame_condition = torch.cat(
            [frame_feat, timestep_per_frame], dim=-1
        )

        # 每次去噪前向只计算一次 scale/shift，所有 Transformer 层复用。
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
    """
    Audio 与离散 emotion 作为不可分离的联合 CFG 条件；
    emotion2vec frame-level 特征额外用于噪声自注意力前的逐帧调制。
    """

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
            joint_cfg = (
                'audio' in self.guiding_conditions
                and 'emotion' in self.guiding_conditions
            )
            if not joint_cfg:
                return super().forward(
                    motion_feat,
                    audio_or_feat,
                    prev_motion_feat=prev_motion_feat,
                    prev_audio_feat=prev_audio_feat,
                    time_step=time_step,
                    indicator=indicator,
                    emo_index=emo_index,
                )

            batch_size = motion_feat.shape[0]

            if audio_or_feat.ndim == 2:
                assert audio_or_feat.shape[1] == round(
                    16000 * self.n_motions / self.fps
                ), f'Incorrect audio length {audio_or_feat.shape[1]}'
                audio_feat_saved = self.extract_audio_feature(audio_or_feat)
            elif audio_or_feat.ndim == 3:
                assert audio_or_feat.shape[1] == self.n_motions, \
                    f'Incorrect audio feature length {audio_or_feat.shape[1]}'
                audio_feat_saved = audio_or_feat
            else:
                raise ValueError(
                    f'Incorrect audio input shape {audio_or_feat.shape}'
                )
            audio_feat = audio_feat_saved.clone()

            if prev_motion_feat is None:
                prev_motion_feat = torch.index_select(
                    self.start_motion_feat, 0, emo_index
                )

            prev_audio_is_start = prev_audio_feat is None
            if prev_audio_is_start:
                prev_audio_feat = torch.index_select(
                    self.start_audio_feat, 0, emo_index
                )

            # Conditional branch: real audio + real discrete emotion.
            emo_feat = self.emo_embed(emo_index).unsqueeze(1)
            emo_shift, emo_scale = self.adaLN_modulation(
                emo_feat
            ).chunk(2, dim=2)
            audio_feat_cond = (
                self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift
            )

            if prev_audio_is_start:
                prev_audio_feat = self.audio_norm(prev_audio_feat)
            else:
                prev_audio_feat = (
                    self.audio_norm(prev_audio_feat)
                    * (1 + emo_scale)
                    + emo_shift
                )

            # Unconditional branch: null audio + null discrete emotion.
            null_audio_feat = self.null_audio_feat.expand(
                batch_size, self.n_motions, -1
            )
            null_emotion_feat = self.null_emotion_feat.expand(
                batch_size, -1, -1
            )
            null_shift, null_scale = self.adaLN_modulation(
                null_emotion_feat
            ).chunk(2, dim=2)
            audio_feat_uncond = (
                self.audio_norm(null_audio_feat)
                * (1 + null_scale)
                + null_shift
            )

            # One dropout decision controls audio and discrete emotion.
            joint_drop_prob = 0.1
            drop_joint_condition = (
                torch.rand(batch_size, device=self.device) < joint_drop_prob
            )
            audio_feat = torch.where(
                drop_joint_condition.view(-1, 1, 1),
                audio_feat_uncond,
                audio_feat_cond,
            )

            if time_step is None:
                time_step = self.diffusion_sched.uniform_sample_t(batch_size)

            alpha_bar = self.diffusion_sched.alpha_bars[time_step]
            c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
            c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

            eps = torch.randn_like(motion_feat)
            motion_feat_noisy = c0 * motion_feat + c1 * eps
            motion_feat_target = self.denoising_net(
                motion_feat_noisy,
                audio_feat,
                prev_motion_feat,
                prev_audio_feat,
                time_step,
                indicator,
            )

            return (
                eps,
                motion_feat_target,
                motion_feat.detach(),
                audio_feat_saved.detach(),
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
            batch_size = audio_or_feat.shape[0]

            if cfg_mode is None:
                cfg_mode = self.cfg_mode
            if cfg_mode not in ['incremental', 'independent']:
                raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

            if cfg_cond is None:
                cfg_cond = self.guiding_conditions
            elif isinstance(cfg_cond, str):
                cfg_cond = cfg_cond.split(',')
            cfg_cond = [
                c for c in cfg_cond if c in ['audio', 'emotion']
            ]

            use_joint_cfg = (
                len(cfg_cond) > 0
                and 'audio' in self.guiding_conditions
                and 'emotion' in self.guiding_conditions
            )
            if isinstance(cfg_scale, (list, tuple)):
                joint_cfg_scale = (
                    cfg_scale[-1] if len(cfg_scale) > 0 else 1.0
                )
            else:
                joint_cfg_scale = cfg_scale

            print(
                f"cfg_cond: {('audio+emotion',) if use_joint_cfg else ()}, "
                f"cfg_scale: {(joint_cfg_scale,) if use_joint_cfg else ()}"
            )

            if audio_or_feat.ndim == 2:
                assert (
                    audio_or_feat.shape[1]
                    == 16000 * self.n_motions / self.fps
                ), f'Incorrect audio length {audio_or_feat.shape[1]}'
                audio_feat_saved = self.extract_audio_feature(audio_or_feat)
            elif audio_or_feat.ndim == 3:
                assert audio_or_feat.shape[1] == self.n_motions, \
                    f'Incorrect audio feature length {audio_or_feat.shape[1]}'
                audio_feat_saved = audio_or_feat
            else:
                raise ValueError(
                    f'Incorrect audio input shape {audio_or_feat.shape}'
                )

            if prev_motion_feat is None:
                prev_motion_feat = torch.index_select(
                    self.start_motion_feat, 0, emo_index
                )

            prev_audio_is_start = prev_audio_feat is None
            if prev_audio_is_start:
                prev_audio_feat = torch.index_select(
                    self.start_audio_feat, 0, emo_index
                )

            if motion_at_T is None:
                motion_at_T = torch.randn(
                    batch_size,
                    self.n_motions,
                    self.motion_feat_dim,
                    device=self.device,
                )

            # Full joint condition.
            emo_feat = self.emo_embed(emo_index).unsqueeze(1)
            emo_shift, emo_scale = self.adaLN_modulation(
                emo_feat
            ).chunk(2, dim=2)
            audio_feat_cond = (
                self.audio_norm(audio_feat_saved)
                * (1 + emo_scale)
                + emo_shift
            )

            if prev_audio_is_start:
                prev_audio_feat = self.audio_norm(prev_audio_feat)
            else:
                prev_audio_feat = (
                    self.audio_norm(prev_audio_feat)
                    * (1 + emo_scale)
                    + emo_shift
                )

            # Fully dropped joint condition.
            null_audio_feat = self.null_audio_feat.expand(
                batch_size, self.n_motions, -1
            )
            null_emotion_feat = self.null_emotion_feat.expand(
                batch_size, -1, -1
            )
            null_shift, null_scale = self.adaLN_modulation(
                null_emotion_feat
            ).chunk(2, dim=2)
            audio_feat_uncond = (
                self.audio_norm(null_audio_feat)
                * (1 + null_scale)
                + null_shift
            )

            if use_joint_cfg:
                audio_feat_in = torch.cat(
                    [audio_feat_uncond, audio_feat_cond], dim=0
                )
                n_entries = 2
            else:
                audio_feat_in = audio_feat_cond
                n_entries = 1

            prev_motion_feat_in = torch.cat(
                [prev_motion_feat] * n_entries, dim=0
            )
            prev_audio_feat_in = torch.cat(
                [prev_audio_feat] * n_entries, dim=0
            )
            indicator_in = (
                torch.cat([indicator] * n_entries, dim=0)
                if indicator is not None else None
            )

            traj = {self.diffusion_sched.num_steps: motion_at_T}
            for t in range(self.diffusion_sched.num_steps, 0, -1):
                if t > 1:
                    z = torch.randn_like(motion_at_T)
                else:
                    z = torch.zeros_like(motion_at_T)

                alpha = self.diffusion_sched.alphas[t]
                alpha_bar = self.diffusion_sched.alpha_bars[t]
                alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
                sigma = self.diffusion_sched.get_sigmas(t, flexibility)

                motion_at_t = traj[t]
                motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
                step_in = torch.tensor(
                    [t] * batch_size, device=self.device
                )
                step_in = torch.cat([step_in] * n_entries, dim=0)

                results = self.denoising_net(
                    motion_in,
                    audio_feat_in,
                    prev_motion_feat_in,
                    prev_audio_feat_in,
                    step_in,
                    indicator_in,
                )

                if dynamic_threshold:
                    dt_ratio, dt_min, dt_max = dynamic_threshold
                    abs_results = results[:, -self.n_motions:].reshape(
                        batch_size * n_entries, -1
                    ).abs()
                    s = torch.quantile(abs_results, dt_ratio, dim=1)
                    s = torch.clamp(s, min=dt_min, max=dt_max)
                    s = s[..., None, None]
                    results = torch.clamp(results, min=-s, max=s)

                results = results.chunk(n_entries)
                if use_joint_cfg:
                    uncond_target = results[0][:, -self.n_motions:]
                    cond_target = results[1][:, -self.n_motions:]
                    target_theta = (
                        uncond_target
                        + joint_cfg_scale * (cond_target - uncond_target)
                    )
                else:
                    target_theta = results[0][:, -self.n_motions:]

                if self.target == 'noise':
                    c0 = 1 / torch.sqrt(alpha)
                    c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                    motion_next = (
                        c0 * (motion_at_t - c1 * target_theta)
                        + sigma * z
                    )
                elif self.target == 'sample':
                    c0 = (
                        (1 - alpha_bar_prev) * torch.sqrt(alpha)
                        / (1 - alpha_bar)
                    )
                    c1 = (
                        (1 - alpha) * torch.sqrt(alpha_bar_prev)
                        / (1 - alpha_bar)
                    )
                    motion_next = (
                        c0 * motion_at_t
                        + c1 * target_theta
                        + sigma * z
                    )
                else:
                    raise ValueError(f'Unknown target type: {self.target}')

                traj[t - 1] = motion_next.detach()
                traj[t] = traj[t].cpu()
                if not ret_traj:
                    del traj[t]

            if ret_traj:
                return traj, motion_at_T, audio_feat_cond
            return traj[0], motion_at_T, audio_feat_cond
        finally:
            self.denoising_net.clear_frame_level_condition()


__all__ = [
    'DiffusionSchedule',
    'DenoisingNetwork',
    'DitTalkingHead',
]

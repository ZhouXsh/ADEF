"""Final v1: dual-branch emotion-audio DiT.

This file is a copy-style method built on top of ``emotion_dit_clean.py`` and
``emotion_dit.py``.  It keeps the clean model's DICE-style CFG and autoregressive
context, but changes the denoiser from:

    motion query -> audio memory
    motion query -> emotion memory

to a more coherent two-audio-branch design:

    motion query -> original audio memory
    motion query -> emotion-modulated audio memory

In v1 the emotion-modulated audio feature follows the original
``emotion_dit.py`` idea: the discrete emotion embedding produces AdaLN shift and
scale, and these parameters modulate the audio feature.  The original audio
branch preserves lip-sync; the emotion-audio branch drives emotional expression.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask
from .emotion_dit import DitTalkingHead as _BaseDitTalkingHead


class DualEmotionAudioCrossDecoderLayer(nn.Module):
    """Decoder layer with original-audio and emotion-audio cross attention."""

    def __init__(
        self,
        feature_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        audio_scale: float = 0.5,
        emotion_audio_scale: float = 0.5,
    ):
        super().__init__()
        self.audio_scale = float(audio_scale)
        self.emotion_audio_scale = float(emotion_audio_scale)

        self.self_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.audio_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.emotion_audio_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(feature_dim, mlp_ratio * feature_dim)
        self.linear2 = nn.Linear(mlp_ratio * feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        audio_memory: torch.Tensor,
        emotion_audio_memory: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_update = self.self_attn(hidden, hidden, hidden, need_weights=False)[0]
        hidden = self.norm1(hidden + self.dropout1(self_update))

        audio_update = self.audio_attn(
            query=hidden,
            key=audio_memory,
            value=audio_memory,
            attn_mask=audio_mask,
            need_weights=False,
        )[0]
        cond_update = self.audio_scale * audio_update

        if emotion_audio_memory is not None:
            emotion_audio_update = self.emotion_audio_attn(
                query=hidden,
                key=emotion_audio_memory,
                value=emotion_audio_memory,
                attn_mask=audio_mask,
                need_weights=False,
            )[0]
            cond_update = cond_update + self.emotion_audio_scale * emotion_audio_update

        hidden = self.norm2(hidden + self.dropout2(cond_update))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
        hidden = self.norm3(hidden + self.dropout3(ff))
        return hidden


class DualEmotionAudioDenoisingNetwork(nn.Module):
    """DiT denoiser with two time-aligned audio memories.

    ``audio_feat`` and ``emotion_audio_feat`` are both frame-aligned to motion,
    therefore the same local alignment mask can be used for both branches.
    """

    def __init__(
        self,
        device: str = "cuda",
        motion_feat_dim: int = 70,
        use_indicator=None,
        architecture: str = "decoder",
        feature_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 1,
        no_use_learnable_pe: bool = True,
        n_prev_motions: int = 10,
        n_motions: int = 100,
        n_diff_steps: int = 500,
        decoder_dropout: float = 0.1,
        audio_scale: float = 0.5,
        emotion_audio_scale: float = 0.5,
    ):
        super().__init__()
        if architecture != "decoder":
            raise ValueError(f"Unknown architecture: {architecture}")

        self.motion_feat_dim = motion_feat_dim
        self.use_indicator = use_indicator
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        self.TE = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        if self.use_learnable_pe:
            self.PE = nn.Parameter(
                torch.randn(1, n_prev_motions + n_motions, feature_dim)
            )
        else:
            self.PE = PositionalEncoding(feature_dim)

        self.feature_proj = nn.Linear(
            motion_feat_dim + (1 if self.use_indicator else 0), feature_dim
        )
        self.layers = nn.ModuleList([
            DualEmotionAudioCrossDecoderLayer(
                feature_dim=feature_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=decoder_dropout,
                audio_scale=audio_scale,
                emotion_audio_scale=emotion_audio_scale,
            )
            for _ in range(n_layers)
        ])

        if align_mask_width > 0:
            motion_len = n_prev_motions + n_motions
            alignment_mask = enc_dec_mask(
                motion_len,
                motion_len,
                frame_width=1,
                expansion=align_mask_width - 1,
            )
            self.register_buffer("alignment_mask", alignment_mask)
        else:
            self.alignment_mask = None

        self.motion_dec = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, motion_feat_dim),
        )
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        motion_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        prev_motion_feat: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        step: torch.Tensor,
        indicator: Optional[torch.Tensor] = None,
        emotion_audio_feat: Optional[torch.Tensor] = None,
        prev_emotion_audio_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.tensor(step, device=self.device, dtype=torch.long)
        step = step.to(self.device).long()
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                indicator,
            ], dim=1).unsqueeze(-1)

        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            if indicator is None:
                indicator = torch.ones(
                    feats_in.shape[:2] + (1,),
                    device=feats_in.device,
                    dtype=feats_in.dtype,
                )
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        hidden = self.feature_proj(feats_in)
        if self.use_learnable_pe:
            hidden = hidden + self.PE[:, :hidden.shape[1], :] + diff_step_embedding
        else:
            hidden = self.PE(hidden) + diff_step_embedding

        audio_memory = torch.cat([prev_audio_feat, audio_feat], dim=1)
        emotion_audio_memory = None
        if emotion_audio_feat is not None:
            if prev_emotion_audio_feat is None:
                prev_emotion_audio_feat = prev_audio_feat
            emotion_audio_memory = torch.cat([prev_emotion_audio_feat, emotion_audio_feat], dim=1)

        for layer in self.layers:
            hidden = layer(
                hidden,
                audio_memory=audio_memory,
                emotion_audio_memory=emotion_audio_memory,
                audio_mask=self.alignment_mask,
            )
        return self.motion_dec(hidden)


class DitTalkingHead(_BaseDitTalkingHead):
    """Final v1: original audio branch + emotion-AdaLN audio branch."""

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
        align_mask_width: int = 1,
        decoder_dropout: float = 0.1,
        audio_scale: float = 0.5,
        emotion_audio_scale: float = 0.5,
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
        )
        self.denoising_net = DualEmotionAudioDenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
            n_diff_steps=n_diff_steps,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            use_indicator=None,
            decoder_dropout=decoder_dropout,
            audio_scale=audio_scale,
            emotion_audio_scale=emotion_audio_scale,
        )
        self.to(device)

    def _get_audio_feature(self, audio_or_feat: torch.Tensor) -> torch.Tensor:
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), \
                f"Incorrect audio length {audio_or_feat.shape[1]}"
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f"Incorrect audio feature length {audio_or_feat.shape[1]}"
            return audio_or_feat
        raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")

    def _init_prev_features(self, batch_size, emo_index, prev_motion_feat=None, prev_audio_feat=None):
        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
        return prev_motion_feat, prev_audio_feat

    def _make_cfg_train_masks(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Three-branch distribution: null / audio-only / audio+emotion."""
        r = torch.rand(batch_size, device=self.device)
        p_uncond = 0.10
        p_audio_only = 0.45
        mask_audio = r < p_uncond
        mask_emotion = r < (p_uncond + p_audio_only)
        return mask_audio, mask_emotion

    def _normalize_audio(self, audio_feat: torch.Tensor) -> torch.Tensor:
        return self.audio_norm(audio_feat)

    def _build_emotion_audio(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        drop_emotion: Optional[torch.Tensor] = None,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build emotion-modulated audio using original emotion_dit AdaLN."""
        batch_size = audio_feat.shape[0]
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        if drop_emotion is not None:
            null_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
            emo_feat = torch.where(drop_emotion.view(batch_size, 1, 1), null_feat, emo_feat)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        return self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

    def _prepare_train_conditions(
        self,
        audio_feat: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        mask_audio: torch.Tensor,
        mask_emotion: torch.Tensor,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        prev_emo_frame_feat: Optional[torch.Tensor] = None,
    ):
        batch_size = audio_feat.shape[0]
        audio_current = audio_feat
        if 'audio' in self.guiding_conditions:
            audio_current = torch.where(
                mask_audio.view(batch_size, 1, 1),
                self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                audio_feat,
            )
        audio_branch = self._normalize_audio(audio_current)
        prev_audio_branch = self._normalize_audio(prev_audio_feat)

        emotion_audio_branch = None
        prev_emotion_audio_branch = None
        if 'emotion' in self.guiding_conditions:
            emotion_audio_branch = self._build_emotion_audio(
                audio_current,
                emo_index,
                drop_emotion=mask_emotion,
                emo_utt_feat=emo_utt_feat,
                emo_frame_feat=emo_frame_feat,
            )
            prev_emotion_audio_branch = self._build_emotion_audio(
                prev_audio_feat,
                emo_index,
                drop_emotion=mask_emotion,
                emo_utt_feat=emo_utt_feat,
                emo_frame_feat=prev_emo_frame_feat,
            )
        return audio_branch, prev_audio_branch, emotion_audio_branch, prev_emotion_audio_branch

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
        emo_utt_feat=None,
        emo_frame_feat=None,
        prev_emo_frame_feat=None,
    ):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        prev_motion_feat, prev_audio_feat = self._init_prev_features(
            batch_size, emo_index, prev_motion_feat, prev_audio_feat
        )

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)
        if not torch.is_tensor(time_step):
            time_step = torch.tensor(time_step, device=self.device, dtype=torch.long)
        else:
            time_step = time_step.to(self.device).long()

        mask_audio = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_emotion = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if 'audio' in self.guiding_conditions and 'emotion' in self.guiding_conditions and self.cfg_mode == 'incremental':
            mask_audio, mask_emotion = self._make_cfg_train_masks(batch_size)
        else:
            if 'audio' in self.guiding_conditions:
                mask_audio = torch.rand(batch_size, device=self.device) < 0.1
            if 'emotion' in self.guiding_conditions:
                mask_emotion = torch.rand(batch_size, device=self.device) < 0.5

        audio_branch, prev_audio_branch, emotion_audio_branch, prev_emotion_audio_branch = self._prepare_train_conditions(
            audio_feat_saved,
            prev_audio_feat,
            emo_index,
            mask_audio,
            mask_emotion,
            emo_utt_feat=emo_utt_feat,
            emo_frame_feat=emo_frame_feat,
            prev_emo_frame_feat=prev_emo_frame_feat,
        )

        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_branch,
            prev_motion_feat,
            prev_audio_branch,
            time_step,
            indicator,
            emotion_audio_feat=emotion_audio_branch,
            prev_emotion_audio_feat=prev_emotion_audio_branch,
        )
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    def _prepare_sample_entries(
        self,
        audio_feat: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        cfg_cond,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        prev_emo_frame_feat: Optional[torch.Tensor] = None,
    ):
        batch_size = audio_feat.shape[0]
        audio_real_raw = audio_feat
        audio_null_raw = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        prev_audio_raw = prev_audio_feat

        if 'audio' in cfg_cond and 'emotion' in cfg_cond:
            # branch 0: null, branch 1: audio-only, branch 2: audio+emotion
            current_audio_raw = torch.cat([audio_null_raw, audio_real_raw, audio_real_raw], dim=0)
            prev_audio_raw_in = torch.cat([prev_audio_raw, prev_audio_raw, prev_audio_raw], dim=0)
            drop_emotion = torch.cat([
                torch.ones(batch_size, dtype=torch.bool, device=self.device),
                torch.ones(batch_size, dtype=torch.bool, device=self.device),
                torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            ], dim=0)
            emo_index_in = torch.cat([emo_index, emo_index, emo_index], dim=0)
            n_entries = 3
        elif 'audio' in cfg_cond:
            current_audio_raw = torch.cat([audio_null_raw, audio_real_raw], dim=0)
            prev_audio_raw_in = torch.cat([prev_audio_raw, prev_audio_raw], dim=0)
            drop_emotion = None
            emo_index_in = torch.cat([emo_index, emo_index], dim=0)
            n_entries = 2
        elif 'emotion' in cfg_cond:
            current_audio_raw = torch.cat([audio_real_raw, audio_real_raw], dim=0)
            prev_audio_raw_in = torch.cat([prev_audio_raw, prev_audio_raw], dim=0)
            drop_emotion = torch.cat([
                torch.ones(batch_size, dtype=torch.bool, device=self.device),
                torch.zeros(batch_size, dtype=torch.bool, device=self.device),
            ], dim=0)
            emo_index_in = torch.cat([emo_index, emo_index], dim=0)
            n_entries = 2
        else:
            current_audio_raw = audio_real_raw
            prev_audio_raw_in = prev_audio_raw
            drop_emotion = None
            emo_index_in = emo_index
            n_entries = 1

        audio_branch = self._normalize_audio(current_audio_raw)
        prev_audio_branch = self._normalize_audio(prev_audio_raw_in)

        emotion_audio_branch = None
        prev_emotion_audio_branch = None
        if 'emotion' in cfg_cond:
            emo_utt_in = torch.cat([emo_utt_feat] * n_entries, dim=0) if emo_utt_feat is not None else None
            emo_frame_in = torch.cat([emo_frame_feat] * n_entries, dim=0) if emo_frame_feat is not None else None
            prev_emo_frame_in = torch.cat([prev_emo_frame_feat] * n_entries, dim=0) if prev_emo_frame_feat is not None else None
            emotion_audio_branch = self._build_emotion_audio(
                current_audio_raw,
                emo_index_in,
                drop_emotion=drop_emotion,
                emo_utt_feat=emo_utt_in,
                emo_frame_feat=emo_frame_in,
            )
            prev_emotion_audio_branch = self._build_emotion_audio(
                prev_audio_raw_in,
                emo_index_in,
                drop_emotion=drop_emotion,
                emo_utt_feat=emo_utt_in,
                emo_frame_feat=prev_emo_frame_in,
            )

        return audio_branch, prev_audio_branch, emotion_audio_branch, prev_emotion_audio_branch, n_entries

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
        emo_utt_feat=None,
        emo_frame_feat=None,
        prev_emo_frame_feat=None,
    ):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = cfg_mode or self.cfg_mode
        cfg_cond = cfg_cond or self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(
                zip(cfg_cond, cfg_scale),
                key=lambda x: ['audio', 'emotion'].index(x[0]),
            ))
        else:
            cfg_cond, cfg_scale = [], []
        print(f"cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}")

        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        prev_motion_feat, prev_audio_feat = self._init_prev_features(
            batch_size, emo_index, prev_motion_feat, prev_audio_feat
        )
        if motion_at_T is None:
            motion_at_T = torch.randn(
                (batch_size, self.n_motions, self.motion_feat_dim), device=self.device
            )

        audio_branch, prev_audio_branch, emotion_audio_branch, prev_emotion_audio_branch, n_entries = self._prepare_sample_entries(
            audio_feat_saved,
            prev_audio_feat,
            emo_index,
            cfg_cond,
            emo_utt_feat=emo_utt_feat,
            emo_frame_feat=emo_frame_feat,
            prev_emo_frame_feat=prev_emo_frame_feat,
        )

        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_base = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_in = torch.cat([step_base] * n_entries, dim=0)

            results = self.denoising_net(
                motion_in,
                audio_branch,
                prev_motion_feat_in,
                prev_audio_branch,
                step_in,
                indicator_in,
                emotion_audio_feat=emotion_audio_branch,
                prev_emotion_audio_feat=prev_emotion_audio_branch,
            )

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            chunks = results.chunk(n_entries)
            if n_entries == 3:
                uncond = chunks[0][:, -self.n_motions:]
                audio_only = chunks[1][:, -self.n_motions:]
                audio_emo = chunks[2][:, -self.n_motions:]
                target_theta = uncond + cfg_scale[0] * (audio_only - uncond) + cfg_scale[1] * (audio_emo - audio_only)
            elif n_entries == 2:
                uncond = chunks[0][:, -self.n_motions:]
                cond = chunks[1][:, -self.n_motions:]
                target_theta = uncond + cfg_scale[0] * (cond - uncond)
            else:
                target_theta = chunks[0][:, -self.n_motions:]

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError(f"Unknown target type: {self.target}")

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat_saved
        return traj[0], motion_at_T, audio_feat_saved

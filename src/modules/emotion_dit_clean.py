"""Clean dual-attention sanity version for ADEF emotion DiT.

Purpose
-------
This file is a diagnostic model, not a new complicated method. It is designed to
answer one question cleanly:

    Can a DICE-Talk / IP-Adapter style decoupled audio-emotion cross-attention
    run at least as stably as the original emotion_dit.py backbone?

Compared with the original src/modules/emotion_dit.py, this version changes only
what is necessary for that question:

1. Keep the original DitTalkingHead interface and training return values.
2. Keep the original diffusion schedule, audio encoder, start features, CFG
   condition dropout style, and autoregressive prev_motion/prev_audio context.
3. Keep the original denoising depth by default: n_layers=8.
4. Do NOT use emotion to modulate audio features.
5. Instead, every decoder layer has two separated condition attentions:
      motion query -> local audio memory
      motion query -> emotion memory
   and the two updates are fused with fixed, explicit coefficients.
6. No hidden sigmoid gate is used for the fusion coefficients. This avoids the
   previous v4/v5/v6 issue where emotion was effectively scaled to almost zero.
7. Transformer dropout is disabled by default in this sanity version. CFG
   condition dropout is still kept. This isolates condition-dropout from layer
   dropout while debugging.
8. The audio alignment mask uses a small sliding window by default
   (align_mask_width=3) instead of width=1, so the first current frames can
   directly attend to the tail of prev_audio. Set align_mask_width=1 to reproduce
   the original strict per-frame alignment mask, or <=0 for no mask.
9. Sampling uses clean DICE-style three-branch CFG:
      null -> audio-only -> audio+emotion
   with previous audio context kept real for all branches.

No explicit lip/non-lip keypoint split is used because ADEF uses implicit
LivePortrait/JoyVASA-style motion features.
"""

from __future__ import annotations

from typing import Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask
from .emotion_dit import DitTalkingHead as _BaseDitTalkingHead


class CleanDualCrossDecoderLayer(nn.Module):
    """Minimal decoder layer with decoupled audio and emotion cross-attention.

    This follows the spirit of IP-Adapter's decoupled condition attention: audio
    and emotion are not mixed before attention. They are attended separately and
    then summed in hidden space.
    """

    def __init__(
        self,
        feature_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale: float = 0.5,
    ):
        super().__init__()
        self.audio_scale = float(audio_scale)
        self.emotion_scale = float(emotion_scale)

        self.self_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.audio_attn = nn.MultiheadAttention(
            feature_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.emotion_attn = nn.MultiheadAttention(
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
        emotion_memory: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_update = self.self_attn(
            hidden, hidden, hidden, need_weights=False
        )[0]
        hidden = self.norm1(hidden + self.dropout1(self_update))

        audio_update = self.audio_attn(
            query=hidden,
            key=audio_memory,
            value=audio_memory,
            attn_mask=audio_mask,
            need_weights=False,
        )[0]

        cond_update = self.audio_scale * audio_update
        if emotion_memory is not None:
            emotion_update = self.emotion_attn(
                query=hidden,
                key=emotion_memory,
                value=emotion_memory,
                attn_mask=None,
                need_weights=False,
            )[0]
            cond_update = cond_update + self.emotion_scale * emotion_update

        hidden = self.norm2(hidden + self.dropout2(cond_update))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
        hidden = self.norm3(hidden + self.dropout3(ff))
        return hidden


class CleanDualAttentionDenoisingNetwork(nn.Module):
    """Denoising network with minimal decoupled audio/emotion attention.

    Defaults intentionally match the original DenoisingNetwork unless noted:
    - n_layers=8, n_heads=8, mlp_ratio=4;
    - use_indicator=None;
    - no_use_learnable_pe=True;
    - align_mask_width defaults to 3 for sanity debugging. Set to 1 for the
      original strict alignment mask.
    """

    def __init__(
        self,
        device: str = "cuda",
        motion_feat_dim: int = 73,
        use_indicator=None,
        architecture: str = "decoder",
        feature_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 3,
        no_use_learnable_pe: bool = True,
        n_prev_motions: int = 10,
        n_motions: int = 100,
        n_diff_steps: int = 500,
        decoder_dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale: float = 0.5,
    ):
        super().__init__()
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

        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if self.use_learnable_pe:
            self.PE = nn.Parameter(
                torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim)
            )
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        if self.architecture != "decoder":
            raise ValueError(f"Unknown architecture: {self.architecture}")

        self.feature_proj = nn.Linear(
            self.motion_feat_dim + (1 if self.use_indicator else 0),
            self.feature_dim,
        )
        self.layers = nn.ModuleList([
            CleanDualCrossDecoderLayer(
                feature_dim=self.feature_dim,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=decoder_dropout,
                audio_scale=audio_scale,
                emotion_scale=emotion_scale,
            )
            for _ in range(self.n_layers)
        ])

        if self.align_mask_width > 0:
            motion_len = self.n_prev_motions + self.n_motions
            alignment_mask = enc_dec_mask(
                motion_len,
                motion_len,
                frame_width=1,
                expansion=self.align_mask_width - 1,
            )
            self.register_buffer("alignment_mask", alignment_mask)
        else:
            self.alignment_mask = None

        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),
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
        emotion_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.tensor(step, device=self.device, dtype=torch.long)
        step = step.to(self.device).long()
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                indicator,
            ], dim=1)
            indicator = indicator.unsqueeze(-1)

        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            if indicator is None:
                indicator = torch.ones(
                    feats_in.shape[:2] + (1,),
                    device=feats_in.device,
                    dtype=feats_in.dtype,
                )
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        feats_in = self.feature_proj(feats_in)
        if self.use_learnable_pe:
            pe = self.PE[:, :feats_in.shape[1], :]
            feats_in = feats_in + pe + diff_step_embedding
        else:
            feats_in = self.PE(feats_in) + diff_step_embedding

        audio_memory = torch.cat([prev_audio_feat, audio_feat], dim=1)
        hidden = feats_in
        for layer in self.layers:
            hidden = layer(
                hidden,
                audio_memory,
                emotion_memory=emotion_feat,
                audio_mask=self.alignment_mask,
            )
        return self.motion_dec(hidden)


class DitTalkingHead(_BaseDitTalkingHead):
    """Clean sanity model: original emotion embedding + dual cross-attention."""

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
        # clean sanity knobs; train.py will not pass these, but direct tests can.
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 3,
        decoder_dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale: float = 0.5,
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
        self.denoising_net = CleanDualAttentionDenoisingNetwork(
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
            emotion_scale=emotion_scale,
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

    def _encode_emotion(self, emo_index: torch.Tensor, drop_mask: Optional[torch.Tensor] = None):
        B = emo_index.shape[0]
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        if drop_mask is not None:
            null_feat = self.null_emotion_feat.expand(B, 1, -1)
            emo_feat = torch.where(drop_mask.view(B, 1, 1), null_feat, emo_feat)
        return emo_feat

    def _make_cfg_train_masks(self, batch_size: int):
        """Match three-branch DICE-style CFG training distribution.

        p_uncond=0.10: audio null, emotion null
        p_audio_only=0.45: audio real, emotion null
        p_full=0.45: audio real, emotion real
        """
        r = torch.rand(batch_size, device=self.device)
        p_uncond = 0.10
        p_audio_only = 0.45
        mask_audio = r < p_uncond
        mask_emotion = r < (p_uncond + p_audio_only)
        return mask_audio, mask_emotion

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
    ):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
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

        if 'audio' in self.guiding_conditions:
            audio_feat = torch.where(
                mask_audio.view(-1, 1, 1),
                self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                audio_feat,
            )
            audio_feat = self.audio_norm(audio_feat)
            # Keep previous audio real and normalized. This is the autoregressive context.
            prev_audio_feat = self.audio_norm(prev_audio_feat)

        emotion_feat = None
        if 'emotion' in self.guiding_conditions:
            emotion_feat = self._encode_emotion(emo_index, drop_mask=mask_emotion)

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
            emotion_feat=emotion_feat,
        )
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    @staticmethod
    def _scale_at_step(max_scale: float, min_scale: float, t: int, num_steps: int, schedule: Optional[str]):
        if schedule is None or schedule == "none" or num_steps <= 1:
            return float(max_scale)
        progress = (num_steps - float(t)) / float(num_steps - 1)
        progress = max(0.0, min(1.0, progress))
        if schedule == "linear":
            w = progress
        elif schedule == "cosine":
            w = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif schedule == "bell":
            w = math.sin(math.pi * progress)
        else:
            raise ValueError(f"Unknown cfg_schedule {schedule}")
        return float(min_scale) + w * (float(max_scale) - float(min_scale))

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
        cfg_min: Optional[Sequence[float]] = None,
        cfg_schedule: Optional[str] = None,
    ):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = cfg_mode or self.cfg_mode
        cfg_cond = cfg_cond or self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if cfg_min is None:
            cfg_min = [1.0 if c == 'audio' else 0.0 for c in cfg_cond]
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale, cfg_min = zip(*sorted(
                zip(cfg_cond, cfg_scale, cfg_min),
                key=lambda x: ['audio', 'emotion'].index(x[0]),
            ))
        else:
            cfg_cond, cfg_scale, cfg_min = [], [], []
        print(f"cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}, cfg_min: {cfg_min}, cfg_schedule: {cfg_schedule or 'none'}")

        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev_features(
            batch_size, emo_index, prev_motion_feat, prev_audio_feat
        )
        if motion_at_T is None:
            motion_at_T = torch.randn(
                (batch_size, self.n_motions, self.motion_feat_dim),
                device=self.device,
            )

        audio_real = self.audio_norm(audio_feat)
        prev_audio_real = self.audio_norm(prev_audio_feat)
        audio_null = self.audio_norm(
            self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        ) if 'audio' in cfg_cond else audio_real

        if 'audio' in cfg_cond and 'emotion' in cfg_cond:
            # null / audio-only / audio+emotion
            audio_feat_in = torch.cat([audio_null, audio_real, audio_real], dim=0)
            # Keep prev audio real in every branch; it is AR context, not the current condition.
            prev_audio_feat_in = torch.cat([prev_audio_real, prev_audio_real, prev_audio_real], dim=0)
            n_entries = 3
        elif 'audio' in cfg_cond:
            audio_feat_in = torch.cat([audio_null, audio_real], dim=0)
            prev_audio_feat_in = torch.cat([prev_audio_real, prev_audio_real], dim=0)
            n_entries = 2
        elif 'emotion' in cfg_cond:
            audio_feat_in = torch.cat([audio_real, audio_real], dim=0)
            prev_audio_feat_in = torch.cat([prev_audio_real, prev_audio_real], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_real
            prev_audio_feat_in = prev_audio_real
            n_entries = 1

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

            emotion_feat_in = None
            if 'emotion' in cfg_cond:
                drop_false = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                drop_true = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                emo_null = self._encode_emotion(emo_index, drop_mask=drop_true)
                emo_real = self._encode_emotion(emo_index, drop_mask=drop_false)
                if 'audio' in cfg_cond:
                    emotion_feat_in = torch.cat([emo_null, emo_null, emo_real], dim=0)
                else:
                    emotion_feat_in = torch.cat([emo_null, emo_real], dim=0)

            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
                emotion_feat=emotion_feat_in,
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
                audio_scale = self._scale_at_step(cfg_scale[0], cfg_min[0], t, self.diffusion_sched.num_steps, cfg_schedule)
                emo_scale = self._scale_at_step(cfg_scale[1], cfg_min[1], t, self.diffusion_sched.num_steps, cfg_schedule)
                target_theta = uncond + audio_scale * (audio_only - uncond) + emo_scale * (audio_emo - audio_only)
            elif n_entries == 2:
                uncond = chunks[0][:, -self.n_motions:]
                cond = chunks[1][:, -self.n_motions:]
                scale = self._scale_at_step(cfg_scale[0], cfg_min[0], t, self.diffusion_sched.num_steps, cfg_schedule)
                target_theta = uncond + scale * (cond - uncond)
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

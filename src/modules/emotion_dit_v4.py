"""v4: Decoupled Audio-Emotion Motion Diffusion.

This version starts from the original src/modules/emotion_dit.py training contract
but replaces emotion-as-audio-modulation with DICE-Talk / IP-Adapter style
condition cooperation:

1. The noisy implicit motion sequence is still the query stream.
2. Audio and emotion are two independent condition memories.
3. Each denoising layer performs local audio cross-attention and emotion
   cross-attention separately, then fuses the updates with conservative gates.
4. Sampling uses DICE-style three-branch CFG:
      null -> audio-only -> audio+emotion.
5. Previous audio context is kept real for all CFG branches to avoid the
   train/inference mismatch that caused unstable first-window motion before.

Emotion encoding in v4 intentionally keeps the original ADEF style:
    emotion id -> nn.Embedding -> [B, 1, C].
No explicit lip/non-lip keypoint dimensions are used.
"""

from typing import Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask
from .emotion_dit import DitTalkingHead as _BaseDitTalkingHead


class DecoupledAudioEmotionDecoderLayer(nn.Module):
    """A Transformer decoder layer with decoupled audio/emotion cross-attention."""

    def __init__(self, feature_dim: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.1, audio_local_scale: float = 1.0,
                 audio_global_scale: float = 0.0, emotion_scale: float = 0.25,
                 use_global_audio: bool = False, orthogonalize_emotion: bool = False,
                 orthogonalize_strength: float = 0.5):
        super().__init__()
        self.audio_local_scale = audio_local_scale
        self.audio_global_scale = audio_global_scale
        self.emotion_scale = emotion_scale
        self.use_global_audio = use_global_audio
        self.orthogonalize_emotion = orthogonalize_emotion
        self.orthogonalize_strength = orthogonalize_strength

        self.self_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.audio_local_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.audio_global_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.emotion_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(feature_dim, mlp_ratio * feature_dim)
        self.linear2 = nn.Linear(mlp_ratio * feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable residual scales. These start conservative and can adapt.
        self.logit_local = nn.Parameter(torch.tensor(2.0))
        self.logit_global = nn.Parameter(torch.tensor(-2.0))
        self.logit_emotion = nn.Parameter(torch.tensor(-2.0))

    @staticmethod
    def _remove_projection(x: torch.Tensor, base: torch.Tensor, strength: float, eps: float = 1e-6) -> torch.Tensor:
        """Remove the component of x that is parallel to base in hidden space."""
        denom = base.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)
        proj = (x * base).sum(dim=-1, keepdim=True) / denom * base
        return x - strength * proj

    def forward(self, hidden: torch.Tensor, audio_memory: torch.Tensor,
                emotion_memory: Optional[torch.Tensor] = None,
                audio_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_update = self.self_attn(hidden, hidden, hidden, need_weights=False)[0]
        hidden = self.norm1(hidden + self.dropout1(self_update))

        audio_local = self.audio_local_attn(
            query=hidden, key=audio_memory, value=audio_memory,
            attn_mask=audio_mask, need_weights=False,
        )[0]
        cond_update = self.audio_local_scale * torch.sigmoid(self.logit_local) * audio_local

        if self.use_global_audio:
            audio_global = self.audio_global_attn(
                query=hidden, key=audio_memory, value=audio_memory,
                attn_mask=None, need_weights=False,
            )[0]
            cond_update = cond_update + self.audio_global_scale * torch.sigmoid(self.logit_global) * audio_global

        if emotion_memory is not None:
            emotion_update = self.emotion_attn(
                query=hidden, key=emotion_memory, value=emotion_memory,
                attn_mask=None, need_weights=False,
            )[0]
            if self.orthogonalize_emotion:
                emotion_update = self._remove_projection(emotion_update, audio_local, self.orthogonalize_strength)
            cond_update = cond_update + self.emotion_scale * torch.sigmoid(self.logit_emotion) * emotion_update

        hidden = self.norm2(hidden + self.dropout2(cond_update))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
        hidden = self.norm3(hidden + self.dropout3(ff))
        return hidden


class DecoupledAudioEmotionDenoisingNetwork(nn.Module):
    """Motion denoising network using decoupled condition attention."""

    def __init__(self, device='cuda', motion_feat_dim=70, use_indicator=True,
                 architecture="decoder", feature_dim=512, n_heads=8, n_layers=6,
                 mlp_ratio=4, align_mask_width=1, no_use_learnable_pe=True,
                 n_prev_motions=25, n_motions=100, n_diff_steps=50,
                 use_global_audio: bool = False, audio_global_scale: float = 0.0,
                 emotion_scale: float = 0.25, orthogonalize_emotion: bool = False):
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
            self.PE = nn.Parameter(torch.randn(1, self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        if self.architecture != 'decoder':
            raise ValueError(f'Unknown architecture: {self.architecture}')
        self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0), self.feature_dim)
        self.layers = nn.ModuleList([
            DecoupledAudioEmotionDecoderLayer(
                feature_dim=self.feature_dim,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                audio_local_scale=1.0,
                audio_global_scale=audio_global_scale,
                emotion_scale=emotion_scale,
                use_global_audio=use_global_audio,
                orthogonalize_emotion=orthogonalize_emotion,
            ) for _ in range(self.n_layers)
        ])

        if self.align_mask_width > 0:
            seq_len = self.n_prev_motions + self.n_motions
            alignment_mask = enc_dec_mask(seq_len, seq_len, frame_width=1,
                                          expansion=self.align_mask_width - 1)
            self.register_buffer('alignment_mask', alignment_mask)
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

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None, emotion_feat=None):
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
                indicator = torch.ones(feats_in.shape[:2] + (1,), device=feats_in.device, dtype=feats_in.dtype)
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        hidden = self.feature_proj(feats_in)
        if self.use_learnable_pe:
            hidden = hidden + self.PE + diff_step_embedding
        else:
            hidden = self.PE(hidden) + diff_step_embedding

        audio_memory = torch.cat([prev_audio_feat, audio_feat], dim=1)
        for layer in self.layers:
            hidden = layer(hidden, audio_memory, emotion_feat, audio_mask=self.alignment_mask)
        return self.motion_dec(hidden)


class DitTalkingHead(_BaseDitTalkingHead):
    """v4 model: original emotion embedding + decoupled dual cross-attention."""

    min_audio_cfg = 1.0
    min_emotion_cfg = 0.20
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
        self.denoising_net = DecoupledAudioEmotionDenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
            n_diff_steps=n_diff_steps,
            use_indicator=True,
            use_global_audio=False,
            audio_global_scale=0.0,
            emotion_scale=0.25,
            orthogonalize_emotion=False,
        )
        self.to(device)

    def _step_tensor(self, time_step, batch_size):
        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)
        if torch.is_tensor(time_step):
            return time_step.to(self.device).long()
        return torch.tensor(time_step, device=self.device, dtype=torch.long)

    def _get_audio_feature(self, audio_or_feat):
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            return audio_or_feat
        raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

    def _init_prev_features(self, batch_size, emo_index, prev_motion_feat=None, prev_audio_feat=None):
        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
        return prev_motion_feat, prev_audio_feat

    def encode_emotion(self, emo_index, step=None, audio_feat=None, drop_mask=None):
        B = emo_index.shape[0]
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        if drop_mask is not None:
            null_feat = self.null_emotion_feat.expand(B, 1, -1)
            emo_feat = torch.where(drop_mask.view(B, 1, 1), null_feat, emo_feat)
        return emo_feat

    def _make_train_masks(self, batch_size):
        """Explicit uncond/audio-only/full dropout used by DICE-style CFG."""
        p_uncond = 0.10
        p_audio_only = 0.45
        r = torch.rand(batch_size, device=self.device)
        mask_audio = r < p_uncond
        mask_emotion = r < (p_uncond + p_audio_only)
        return mask_audio, mask_emotion

    def _cfg_scale_at_step(self, max_scale, min_scale, t, schedule: str = "linear"):
        if self.diffusion_sched.num_steps <= 1:
            return float(max_scale)
        progress = (self.diffusion_sched.num_steps - float(t)) / float(self.diffusion_sched.num_steps - 1)
        progress = max(0.0, min(1.0, progress))
        if schedule == "bell":
            weight = math.sin(math.pi * progress)
        elif schedule == "cosine":
            weight = 0.5 - 0.5 * math.cos(math.pi * progress)
        else:
            weight = progress
        return float(min_scale) + weight * (float(max_scale) - float(min_scale))

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
                time_step=None, indicator=None, emo_index=None):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev_features(batch_size, emo_index, prev_motion_feat, prev_audio_feat)
        step_tensor = self._step_tensor(time_step, batch_size)

        mask_audio = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_emotion = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if len(self.guiding_conditions) > 0 and self.cfg_mode == 'incremental' and \
                'audio' in self.guiding_conditions and 'emotion' in self.guiding_conditions:
            mask_audio, mask_emotion = self._make_train_masks(batch_size)
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
            prev_audio_feat = self.audio_norm(prev_audio_feat)

        emotion_feat = None
        if 'emotion' in self.guiding_conditions:
            emotion_feat = self.encode_emotion(emo_index, step_tensor, audio_feat, drop_mask=mask_emotion)

        alpha_bar = self.diffusion_sched.alpha_bars[step_tensor]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        motion_feat_target = self.denoising_net(
            motion_feat_noisy, audio_feat, prev_motion_feat, prev_audio_feat,
            step_tensor, indicator, emotion_feat=emotion_feat,
        )
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, motion_at_T=None,
               indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False, emo_index=None,
               cfg_min: Optional[Sequence[float]] = None, cfg_schedule: Optional[str] = None):
        batch_size = audio_or_feat.shape[0]
        cfg_cond = cfg_cond or self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if cfg_min is None:
            cfg_min = [self.min_audio_cfg if c == 'audio' else self.min_emotion_cfg for c in cfg_cond]
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale, cfg_min = zip(*sorted(zip(cfg_cond, cfg_scale, cfg_min),
                                                       key=lambda x: ['audio', 'emotion'].index(x[0])))
        else:
            cfg_cond, cfg_scale, cfg_min = [], [], []
        schedule = cfg_schedule or self.default_cfg_schedule
        print(f'cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}, cfg_min: {cfg_min}, cfg_schedule: {schedule}')

        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev_features(batch_size, emo_index, prev_motion_feat, prev_audio_feat)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim), device=self.device)

        audio_real = self.audio_norm(audio_feat)
        prev_audio_real = self.audio_norm(prev_audio_feat)
        audio_null = self.audio_norm(self.null_audio_feat.expand(batch_size, self.n_motions, -1)) if 'audio' in cfg_cond else audio_real

        if 'audio' in cfg_cond and 'emotion' in cfg_cond:
            audio_feat_in = torch.cat([audio_null, audio_real, audio_real], dim=0)
            prev_audio_feat_in = torch.cat([prev_audio_real, prev_audio_real, prev_audio_real], dim=0)
            n_entries = 3
        elif 'audio' in cfg_cond:
            audio_feat_in = torch.cat([audio_null, audio_real], dim=0)
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
            step_single = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_in = torch.cat([step_single] * n_entries, dim=0)

            emotion_feat_in = None
            if 'emotion' in cfg_cond:
                if 'audio' in cfg_cond:
                    drop_false = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
                    drop_true = torch.ones(batch_size, dtype=torch.bool, device=self.device)
                    emo_null = self.encode_emotion(emo_index, step_single, audio_real, drop_mask=drop_true)
                    emo_real = self.encode_emotion(emo_index, step_single, audio_real, drop_mask=drop_false)
                    emotion_feat_in = torch.cat([emo_null, emo_null, emo_real], dim=0)
                else:
                    emotion_feat_in = self.encode_emotion(emo_index, step_single, audio_real, drop_mask=None)

            results = self.denoising_net(
                motion_in, audio_feat_in, prev_motion_feat_in, prev_audio_feat_in,
                step_in, indicator_in, emotion_feat=emotion_feat_in,
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
                audio_scale = self._cfg_scale_at_step(cfg_scale[0], cfg_min[0], t, schedule='linear')
                emo_scale = self._cfg_scale_at_step(cfg_scale[1], cfg_min[1], t, schedule=schedule)
                target_theta = uncond + audio_scale * (audio_only - uncond) + emo_scale * (audio_emo - audio_only)
            elif n_entries == 2:
                uncond = chunks[0][:, -self.n_motions:]
                cond = chunks[1][:, -self.n_motions:]
                audio_scale = self._cfg_scale_at_step(cfg_scale[0], cfg_min[0], t, schedule='linear')
                target_theta = uncond + audio_scale * (cond - uncond)
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
                raise ValueError(f'Unknown target type: {self.target}')

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat_saved
        return traj[0], motion_at_T, audio_feat_saved

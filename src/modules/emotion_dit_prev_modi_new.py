# coding: utf-8
"""
A from-scratch emotion DiT variant based on ``emotion_dit_prev_modi.py``.

Why this file exists
--------------------
``emotion_dit_prev_modi.py`` injects emotion by AdaLN-modulating the audio
memory. That can make the low-density emotion label compete with the high-density
phoneme/audio condition and hurt lip-sync. ``emotion_dit_decoupled_adapter.py``
solves this with a residual adapter, but it is most useful when initialized from
an audio-only checkpoint.

This file keeps the same public ``DitTalkingHead`` API but is designed to train
from scratch:

    clean audio memory -> DenoisingNetwork -> base motion
    base motion + audio context + emotion label -> motion-space residual adapter

The start tokens are neutral/shared instead of emotion-specific, so the base path
cannot receive the emotion id through ``start_audio_feat`` or
``start_motion_feat``. Emotion only enters the residual adapter.
"""

from __future__ import annotations

import platform
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import pad_audio
from .emotion_dit_prev_modi import DiffusionSchedule, DenoisingNetwork
from ..config.base_config import make_abs_path


def _parse_int_list(value: Optional[Union[str, Iterable[int]]]) -> Tuple[int, ...]:
    if value is None:
        return tuple()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return tuple()
        return tuple(int(v.strip()) for v in value.split(',') if v.strip())
    return tuple(int(v) for v in value)


def build_emotion_dim_mask(
    motion_feat_dim: int,
    expression_dim: int = 63,
    pose_weight: float = 0.15,
    protected_dims: Optional[Union[str, Iterable[int]]] = None,
    protected_kp_indices: Optional[Union[str, Iterable[int]]] = None,
    protected_weight: float = 0.0,
) -> torch.Tensor:
    """Build a [motion_feat_dim] residual mask.

    LivePortrait motion is normally 21 * 3 + 7 = 70 dims. Dims 0:63 are
    expression keypoint deltas and 63:70 are pose/head-related coefficients.
    ``protected_kp_indices`` expands keypoint ids to xyz triplets.
    """
    mask = torch.zeros(motion_feat_dim, dtype=torch.float32)
    exp_end = min(expression_dim, motion_feat_dim)
    mask[:exp_end] = 1.0
    if motion_feat_dim > expression_dim:
        mask[expression_dim:] = pose_weight

    for dim in _parse_int_list(protected_dims):
        if 0 <= dim < motion_feat_dim:
            mask[dim] = protected_weight

    for kp in _parse_int_list(protected_kp_indices):
        start = kp * 3
        end = start + 3
        if start < exp_end:
            mask[start:min(end, exp_end)] = protected_weight
    return mask


class MotionSpaceEmotionAdapter(nn.Module):
    """Zero-init residual adapter that adds emotion in motion space only."""

    def __init__(
        self,
        motion_feat_dim: int,
        audio_feat_dim: int,
        emotion_dim: int,
        hidden_dim: int = 512,
        residual_scale: float = 0.25,
        pose_weight: float = 0.15,
        protected_dims: Optional[Union[str, Iterable[int]]] = None,
        protected_kp_indices: Optional[Union[str, Iterable[int]]] = None,
        protected_weight: float = 0.0,
    ):
        super().__init__()
        self.motion_norm = nn.LayerNorm(motion_feat_dim)
        self.audio_norm = nn.LayerNorm(audio_feat_dim)
        self.motion_proj = nn.Linear(motion_feat_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_feat_dim, hidden_dim)
        self.emo_to_film = nn.Sequential(nn.SiLU(), nn.Linear(emotion_dim, hidden_dim * 2))
        self.delta = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motion_feat_dim),
        )
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.residual_scale = residual_scale
        mask = build_emotion_dim_mask(
            motion_feat_dim=motion_feat_dim,
            pose_weight=pose_weight,
            protected_dims=protected_dims,
            protected_kp_indices=protected_kp_indices,
            protected_weight=protected_weight,
        )
        self.register_buffer('dim_mask', mask.view(1, 1, -1))

    @staticmethod
    def _align_audio(audio_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        if audio_feat.shape[1] == target_len:
            return audio_feat
        if audio_feat.shape[1] > target_len:
            return audio_feat[:, -target_len:]
        pad_len = target_len - audio_feat.shape[1]
        pad = audio_feat[:, :1].expand(-1, pad_len, -1)
        return torch.cat([pad, audio_feat], dim=1)

    def forward(
        self,
        base_motion: torch.Tensor,
        audio_context: torch.Tensor,
        emotion_feat: torch.Tensor,
        strength: Union[float, torch.Tensor] = 1.0,
    ) -> torch.Tensor:
        audio_context = self._align_audio(audio_context, base_motion.shape[1])
        h = self.motion_proj(self.motion_norm(base_motion)) + self.audio_proj(self.audio_norm(audio_context))
        if emotion_feat.ndim == 2:
            emotion_feat = emotion_feat.unsqueeze(1)
        shift, scale = self.emo_to_film(emotion_feat).chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        delta = self.delta(h)
        if not torch.is_tensor(strength):
            strength = torch.tensor(float(strength), device=base_motion.device, dtype=base_motion.dtype)
        while strength.ndim < 3:
            strength = strength.unsqueeze(-1)
        return base_motion + self.residual_scale * strength * self.dim_mask * delta


class DitTalkingHead(nn.Module):
    """Clean-audio + motion-space emotion residual DiT, trainable from scratch."""

    def __init__(self, device='cuda', target='sample', architecture='decoder',
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=10,
                 audio_model='hubert', feature_dim=512, n_diff_steps=500,
                 diff_schedule='cosine', cfg_mode='incremental', guiding_conditions='audio,emotion',
                 emo_classes=8, condition_dropout_prob=0.1, emotion_dropout_prob=0.15,
                 emotion_residual_scale=0.25, emotion_hidden_dim=512, emotion_pose_weight=0.15,
                 emotion_protected_dims=None, emotion_protected_kp_indices=None,
                 emotion_protected_weight=0.0, base_start_emotion_id=None):
        super().__init__()
        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.audio_model = audio_model
        self.cfg_mode = cfg_mode
        self.emo_classes = emo_classes
        self.condition_dropout_prob = condition_dropout_prob
        self.emotion_dropout_prob = emotion_dropout_prob

        if self.audio_model == 'wav2vec2':
            print('using wav2vec2 audio encoder ...')
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model in ['hubert_zh_ori', 'hubert_zh']:
            print('using hubert chinese ori')
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == 'Windows':
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture != 'decoder':
            raise ValueError(f'Unknown architecture {architecture}!')
        self.audio_feature_map = nn.Linear(768, feature_dim)
        # Shared neutral starts. Do not index by emotion id; this prevents an
        # emotion shortcut before the audio branch has learned lip-sync.
        self.start_audio_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, feature_dim))
        self.start_motion_feat = nn.Parameter(torch.randn(1, self.n_prev_motions, self.motion_feat_dim))

        self.denoising_net = DenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
        )
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [c for c in guiding_conditions if c in ['audio', 'emotion']]
        self.null_audio_feat = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.null_emotion_feat = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.emotion_norm = nn.LayerNorm(feature_dim)
        self.emotion_adapter = MotionSpaceEmotionAdapter(
            motion_feat_dim=motion_feat_dim,
            audio_feat_dim=feature_dim,
            emotion_dim=feature_dim,
            hidden_dim=emotion_hidden_dim,
            residual_scale=emotion_residual_scale,
            pose_weight=emotion_pose_weight,
            protected_dims=emotion_protected_dims,
            protected_kp_indices=emotion_protected_kp_indices,
            protected_weight=emotion_protected_weight,
        )
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _audio_to_feat(self, audio_or_feat):
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            return audio_or_feat
        raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

    def _init_prev(self, batch_size, prev_motion_feat=None, prev_audio_feat=None):
        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(batch_size, -1, -1)
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)
        return prev_motion_feat, prev_audio_feat

    def _emotion_feat(self, emo_index, batch_size, drop_mask=None):
        if emo_index is None:
            emo_index = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        if drop_mask is not None:
            emo_feat = torch.where(
                drop_mask.view(-1, 1, 1),
                self.null_emotion_feat.expand(batch_size, 1, -1),
                emo_feat,
            )
        return self.emotion_norm(emo_feat)

    def _apply_emotion(self, base_motion, prev_audio_feat, audio_feat, emo_index, strength):
        emotion_feat = self._emotion_feat(emo_index, base_motion.shape[0])
        audio_context = torch.cat([prev_audio_feat, audio_feat], dim=1)
        return self.emotion_adapter(base_motion, audio_context, emotion_feat, strength=strength)

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
                time_step=None, indicator=None, emo_index=None, emotion_strength=None):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._audio_to_feat(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev(batch_size, prev_motion_feat, prev_audio_feat)

        if 'audio' in self.guiding_conditions and self.training:
            mask_audio = torch.rand(batch_size, device=self.device) < self.condition_dropout_prob
            audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                     self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                     audio_feat)

        if 'emotion' in self.guiding_conditions:
            if emotion_strength is None:
                if self.training and self.emotion_dropout_prob > 0:
                    keep = torch.rand(batch_size, device=self.device) >= self.emotion_dropout_prob
                    emotion_strength = keep.float()
                else:
                    emotion_strength = 1.0
        else:
            emotion_strength = 0.0

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        base_target = self.denoising_net(
            motion_feat_noisy, audio_feat, prev_motion_feat, prev_audio_feat, time_step, indicator
        )
        target = self._apply_emotion(base_target, prev_audio_feat, audio_feat, emo_index, emotion_strength)
        return eps, target, motion_feat.detach(), audio_feat_saved.detach()

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num * 2).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None, emotion_strength=1.0):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = cfg_mode or self.cfg_mode
        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'emotion'].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []

        audio_feat = self._audio_to_feat(audio_or_feat)
        prev_motion_feat, prev_audio_feat = self._init_prev(batch_size, prev_motion_feat, prev_audio_feat)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim), device=self.device)

        null_audio = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion = self.null_emotion_feat.expand(batch_size, 1, -1)
        full_emotion = self._emotion_feat(emo_index, batch_size)

        audio_entries = []
        emotion_entries = []
        strength_entries = []
        audio_entries.append(null_audio if 'audio' in cfg_cond else audio_feat)
        emotion_entries.append(null_emotion if 'emotion' in cfg_cond else full_emotion)
        strength_entries.append(torch.zeros(batch_size, device=self.device))
        for cond in cfg_cond:
            if cond == 'audio':
                audio_entries.append(audio_feat)
                emotion_entries.append(null_emotion if 'emotion' in cfg_cond else full_emotion)
                strength_entries.append(torch.zeros(batch_size, device=self.device))
            elif cond == 'emotion':
                audio_entries.append(audio_feat)
                emotion_entries.append(full_emotion)
                strength_entries.append(torch.full((batch_size,), float(emotion_strength), device=self.device))

        n_entries = len(audio_entries)
        audio_feat_in = torch.cat(audio_entries, dim=0)
        emotion_feat_in = torch.cat(emotion_entries, dim=0)
        emotion_strength_in = torch.cat(strength_entries, dim=0)
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
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
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(
                motion_in, audio_feat_in, prev_motion_feat_in, prev_audio_feat_in, step_in, indicator_in
            )
            audio_context_in = torch.cat([prev_audio_feat_in, audio_feat_in], dim=1)
            results = self.emotion_adapter(results, audio_context_in, emotion_feat_in, emotion_strength_in)

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)
            target_theta = results[0][:, -self.n_motions:]
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta = target_theta + cfg_scale[i] * (
                        results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:]
                    )
                elif cfg_mode == 'incremental':
                    target_theta = target_theta + cfg_scale[i] * (
                        results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:]
                    )
                else:
                    raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

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
            return traj, motion_at_T, audio_feat
        return traj[0], motion_at_T, audio_feat

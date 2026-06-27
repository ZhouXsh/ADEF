import platform

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import pad_audio
from .emotion_dit import DiffusionSchedule, DenoisingNetwork
from ..config.base_config import make_abs_path


class DitTalkingHead(nn.Module):
    """Emotion-level conditioned version of emotion_dit.DitTalkingHead.

    Main differences from emotion_dit.py:
    1. add emo_level as an explicit condition: 0/1/2 corresponds to level 1/2/3;
    2. build emotion feature as class embedding + monotonic level-scaled direction + level embedding;
    3. keep audio CFG and emotion CFG separable at inference, so cfg_scale can be [audio_scale, emotion_scale];
    4. construct prev_audio branches separately during sampling to avoid leaking full emotion into the
       audio-only / unconditional CFG branches.
    """

    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=10,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental",
                 guiding_conditions="audio,emotion", emo_classes=8,
                 emo_levels=3, default_emo_index=5, default_emo_level=1):
        super().__init__()
        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.emo_classes = emo_classes
        self.emo_levels = emo_levels
        self.default_emo_index = default_emo_index
        self.default_emo_level = default_emo_level

        self.audio_model = audio_model
        if self.audio_model == 'wav2vec2':
            print("using wav2vec2 audio encoder ...")
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh':
            print("using hubert chinese ori")
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, feature_dim)
            self.start_audio_feat = nn.Parameter(torch.randn(emo_classes, self.n_prev_motions, feature_dim))
        else:
            raise ValueError(f'Unknown architecture {architecture}!')

        self.start_motion_feat = nn.Parameter(torch.randn(emo_classes, self.n_prev_motions, self.motion_feat_dim))

        self.denoising_net = DenoisingNetwork(device=device,
                                              n_motions=self.n_motions,
                                              n_prev_motions=self.n_prev_motions,
                                              motion_feat_dim=self.motion_feat_dim,
                                              feature_dim=feature_dim)
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['audio', 'emotion']]

        if 'audio' in self.guiding_conditions:
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, feature_dim))
            self.audio_norm = nn.LayerNorm(feature_dim, eps=1e-5)

        if 'emotion' in self.guiding_conditions:
            self.null_emotion_feat = nn.Parameter(torch.zeros(1, 1, feature_dim))
            self.emo_embed = nn.Embedding(emo_classes, feature_dim)
            self.emo_level_embed = nn.Embedding(emo_levels, feature_dim)
            self.emo_level_direction = nn.Embedding(emo_classes, feature_dim)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(feature_dim, 2 * feature_dim, bias=True),
            )
            # Start from the original class-only behavior. The level pathway will be learned gradually.
            nn.init.zeros_(self.emo_level_embed.weight)
            nn.init.zeros_(self.emo_level_direction.weight)

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _prepare_emo_index(self, batch_size, emo_index):
        if emo_index is None:
            emo_index = torch.full((batch_size,), self.default_emo_index, dtype=torch.long, device=self.device)
        else:
            emo_index = emo_index.to(self.device).long().view(-1)
        return emo_index.clamp(0, self.emo_classes - 1)

    def _prepare_emo_level(self, batch_size, emo_level):
        if emo_level is None:
            emo_level = torch.full((batch_size,), self.default_emo_level, dtype=torch.long, device=self.device)
        else:
            emo_level = emo_level.to(self.device).long().view(-1)
        return emo_level.clamp(0, self.emo_levels - 1)

    def build_emotion_feature(self, emo_index, emo_level):
        """Build emotion condition.

        emo_index controls the direction/category, while emo_level controls intensity.
        For dataset values 0/1/2, the scalar is 1/3, 2/3, 1. This makes level 3 a stronger
        version of the same learned emotion direction instead of merely increasing CFG scale.
        """
        base = self.emo_embed(emo_index)
        level = self.emo_level_embed(emo_level)
        level_scalar = (emo_level.float() + 1.0).unsqueeze(-1) / float(self.emo_levels)
        direction = self.emo_level_direction(emo_index)
        return (base + level + level_scalar * direction).unsqueeze(1)

    def _modulate_audio(self, audio_feat, emotion_feat):
        emo_shift, emo_scale = self.adaLN_modulation(emotion_feat).chunk(2, dim=2)
        return self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
                time_step=None, indicator=None, emo_index=None, emo_level=None):
        batch_size = motion_feat.shape[0]
        emo_index = self._prepare_emo_index(batch_size, emo_index)
        emo_level = self._prepare_emo_level(batch_size, emo_level)

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        audio_feat = audio_feat_saved.clone()
        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)

        p_AE = 0.1
        p_E = 0.55

        emotion_feat = None
        if 'emotion' in self.guiding_conditions:
            emotion_feat = self.build_emotion_feature(emo_index, emo_level)

        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            mask_flag = torch.rand(batch_size, device=self.device)

            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = torch.rand(batch_size, device=self.device) < null_cond_prob
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)
            else:
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag < p_AE
                    audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                             self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                             audio_feat)

            if len(self.guiding_conditions) == 2 and 'emotion' in self.guiding_conditions:
                mask_emotion = mask_flag < p_E
                emotion_feat = torch.where(mask_emotion.view(-1, 1, 1),
                                           self.null_emotion_feat.expand(batch_size, -1, -1),
                                           emotion_feat)

        if 'emotion' in self.guiding_conditions:
            audio_feat = self._modulate_audio(audio_feat, emotion_feat)
            # Let emotion level affect the first window as well; the original file only normalized
            # start_audio_feat when prev_audio_feat was None, which weakens initial emotion control.
            prev_audio_feat = self._modulate_audio(prev_audio_feat, emotion_feat)

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)

        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat,
                                                prev_motion_feat, prev_audio_feat,
                                                time_step, indicator)
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None, emo_level=None):
        batch_size = audio_or_feat.shape[0]
        emo_index = self._prepare_emo_index(batch_size, emo_index)
        emo_level = self._prepare_emo_level(batch_size, emo_level)

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'emotion'].index(x[0])))
            cfg_cond, cfg_scale = list(cfg_cond), list(cfg_scale)
        else:
            cfg_cond, cfg_scale = [], []
        print(f'cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}, emo_level: {emo_level.detach().cpu().tolist()}')

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim), device=self.device)

        if 'emotion' in self.guiding_conditions:
            full_emotion_feat = self.build_emotion_feature(emo_index, emo_level)
            null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        else:
            full_emotion_feat = null_emotion_feat = None

        def build_branch(use_audio, use_emotion):
            cur_audio = audio_feat if use_audio else self.null_audio_feat.expand(batch_size, self.n_motions, -1)
            cur_prev_audio = prev_audio_feat if use_audio else self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)
            if 'emotion' in self.guiding_conditions:
                cur_emotion = full_emotion_feat if use_emotion else null_emotion_feat
                cur_audio = self._modulate_audio(cur_audio, cur_emotion)
                cur_prev_audio = self._modulate_audio(cur_prev_audio, cur_emotion)
            return cur_audio, cur_prev_audio

        # Incremental CFG branch order:
        # 0: no audio / no emotion; 1: audio only; 2: audio + emotion.
        # If a condition is not listed in cfg_cond, it is treated as always enabled.
        branch_audio = 'audio' not in cfg_cond
        branch_emotion = 'emotion' not in cfg_cond
        audio_feat_entries, prev_audio_entries = [], []
        a, pa = build_branch(branch_audio, branch_emotion)
        audio_feat_entries.append(a)
        prev_audio_entries.append(pa)
        for cond in cfg_cond:
            if cond == 'audio':
                branch_audio = True
            elif cond == 'emotion':
                branch_emotion = True
            a, pa = build_branch(branch_audio, branch_emotion)
            audio_feat_entries.append(a)
            prev_audio_entries.append(pa)

        n_entries = len(audio_feat_entries)
        audio_feat_in = torch.cat(audio_feat_entries, dim=0)
        prev_audio_feat_in = torch.cat(prev_audio_entries, dim=0)
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
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in)

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
                    target_theta += cfg_scale[i] * (results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:])
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
                raise ValueError('Unknown target type: {}'.format(self.target))

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat
        return traj[0], motion_at_T, audio_feat

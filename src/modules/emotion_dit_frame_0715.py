import copy
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()
        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
            betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')

        betas = torch.cat([torch.zeros(1), betas], dim=0)
        alphas = 1 - betas
        log_alphas = torch.log(alphas)
        for i in range(1, log_alphas.shape[0]):
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.shape[0]):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)

        self.num_steps = num_steps
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        return torch.randint(1, self.num_steps + 1, (batch_size,)).tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        return self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)


class DualCrossAttentionDecoderLayer(nn.Module):
    """Transformer decoder layer with two parallel cross-attention branches."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation='gelu', batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.audio_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.frame_cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f'Unsupported activation: {activation}')

        self.cross_attn_fusion_logits = nn.Parameter(torch.zeros(2))

    def forward(self, tgt, audio_memory, frame_memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                audio_key_padding_mask=None, frame_key_padding_mask=None):
        x = tgt
        self_attn_out = self.self_attn(
            x, x, x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )[0]
        x = self.norm1(x + self.dropout1(self_attn_out))

        audio_attn_out = self.audio_cross_attn(
            x, audio_memory, audio_memory,
            attn_mask=memory_mask,
            key_padding_mask=audio_key_padding_mask,
            need_weights=False,
        )[0]
        frame_attn_out = self.frame_cross_attn(
            x, frame_memory, frame_memory,
            attn_mask=memory_mask,
            key_padding_mask=frame_key_padding_mask,
            need_weights=False,
        )[0]
        fusion_weight = torch.softmax(self.cross_attn_fusion_logits, dim=0)
        cross_attn_out = (
            fusion_weight[0] * audio_attn_out
            + fusion_weight[1] * frame_attn_out
        )
        x = self.norm2(x + self.dropout2(cross_attn_out))

        ff_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(x + self.dropout3(ff_out))
        return x


class DualCrossAttentionDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            copy.deepcopy(decoder_layer) for _ in range(num_layers)
        ])

    def forward(self, tgt, audio_memory, frame_memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                audio_key_padding_mask=None, frame_key_padding_mask=None):
        output = tgt
        for layer in self.layers:
            output = layer(
                output,
                audio_memory,
                frame_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                audio_key_padding_mask=audio_key_padding_mask,
                frame_key_padding_mask=frame_key_padding_mask,
            )
        return output


class DitTalkingHead(nn.Module):
    def __init__(self, device='cuda', target='sample', architecture='decoder',
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=10,
                 audio_model='hubert', feature_dim=512, n_diff_steps=500,
                 diff_schedule='cosine', cfg_mode='incremental',
                 guiding_conditions='audio,emotion', emo_classes=8,
                 emotion2vec_dim=1024):
        super().__init__()
        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.emotion2vec_dim = emotion2vec_dim

        self.audio_model = audio_model
        if self.audio_model == 'wav2vec2':
            print('using wav2vec2 audio encoder ...')
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                make_abs_path('../../pretrained_weights/wav2vec2-base-960h')
            )
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                make_abs_path('../../pretrained_weights/hubert-base-ls960')
            )
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
        self.frame_feature_map = nn.Linear(emotion2vec_dim, feature_dim)
        self.frame_norm = nn.LayerNorm(feature_dim, eps=1e-9)

        self.start_audio_feat = nn.Parameter(
            torch.randn(1, self.n_prev_motions, feature_dim)
        )
        self.start_frame_feat = nn.Parameter(
            torch.randn(1, self.n_prev_motions, feature_dim)
        )
        self.start_motion_feat = nn.Parameter(
            torch.randn(emo_classes, self.n_prev_motions, self.motion_feat_dim)
        )

        self.denoising_net = DenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            n_diff_steps=n_diff_steps,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
        )
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [
            cond for cond in guiding_conditions if cond in ['audio', 'emotion']
        ]

        if 'audio' in self.guiding_conditions:
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, feature_dim))
        else:
            self.register_buffer('null_audio_feat', torch.zeros(1, 1, feature_dim))

        self.null_frame_feat = nn.Parameter(torch.randn(1, 1, feature_dim))
        if 'emotion' in self.guiding_conditions:
            self.register_buffer('null_emotion_feat', torch.zeros(1, 1, feature_dim))
            self.emo_embed = nn.Embedding(emo_classes, feature_dim)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(feature_dim, 2 * feature_dim, bias=True)
            )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(
            pad_audio(audio), self.fps, frame_num=frame_num * 2
        ).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(
            hidden_states, size=frame_num, align_corners=False, mode='linear'
        )
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)

    def extract_frame_feature(self, frame_emotion_feat, frame_num=None):
        if frame_emotion_feat is None:
            raise ValueError('frame_emotion_feat is required for the frame-level model')
        if frame_emotion_feat.ndim != 3:
            raise ValueError(
                f'Incorrect frame-level emotion2vec shape {frame_emotion_feat.shape}'
            )

        frame_num = frame_num or self.n_motions
        if frame_emotion_feat.shape[1] != frame_num:
            feat = frame_emotion_feat.transpose(1, 2)
            feat = F.interpolate(feat, size=frame_num, mode='linear', align_corners=False)
            frame_emotion_feat = feat.transpose(1, 2).contiguous()

        if frame_emotion_feat.shape[-1] == self.emotion2vec_dim:
            return self.frame_feature_map(frame_emotion_feat)
        if frame_emotion_feat.shape[-1] == self.feature_dim:
            return frame_emotion_feat
        raise ValueError(
            f'Incorrect frame-level emotion2vec dim {frame_emotion_feat.shape[-1]}, '
            f'expected {self.emotion2vec_dim} or mapped dim {self.feature_dim}'
        )

    def _modulate_frame_feature(self, frame_feat, emo_feat):
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        return self.frame_norm(frame_feat) * (1 + emo_scale) + emo_shift

    def _prepare_audio_feature(self, audio_or_feat):
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(
                16000 * self.n_motions / self.fps
            ), f'Incorrect audio length {audio_or_feat.shape[1]}'
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, (
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            )
            return audio_or_feat
        raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None,
                prev_audio_feat=None, time_step=None, indicator=None,
                emo_index=None, frame_emotion_feat=None, prev_frame_feat=None):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._prepare_audio_feature(audio_or_feat)
        frame_feat_saved = self.extract_frame_feature(frame_emotion_feat)

        if prev_motion_feat is None:
            if emo_index is None:
                raise ValueError('emo_index is required to initialize previous motion')
            prev_motion_feat = torch.index_select(
                self.start_motion_feat, 0, emo_index
            )
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)
        if prev_frame_feat is None:
            prev_frame_feat = self.start_frame_feat.expand(batch_size, -1, -1)

        audio_feat = audio_feat_saved
        frame_feat = frame_feat_saved
        prev_audio_feat_in = prev_audio_feat
        prev_frame_feat_in = prev_frame_feat

        p_audio_emotion = 0.1
        p_emotion = 0.55
        mask_audio = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_emotion = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, (
                'Only support 1 or 2 CFG conditions!'
            )
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = (
                        torch.rand(batch_size, device=self.device) < null_cond_prob
                    )
                if 'emotion' in self.guiding_conditions:
                    mask_emotion = (
                        torch.rand(batch_size, device=self.device) < null_cond_prob
                    )
            else:
                mask_flag = torch.rand(batch_size, device=self.device)
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag < p_audio_emotion
                if 'emotion' in self.guiding_conditions:
                    mask_emotion = mask_flag < p_emotion

        if 'audio' in self.guiding_conditions:
            audio_feat = torch.where(
                mask_audio.view(-1, 1, 1),
                self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                audio_feat,
            )
            prev_audio_feat_in = torch.where(
                mask_audio.view(-1, 1, 1),
                self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1),
                prev_audio_feat_in,
            )

        if 'emotion' in self.guiding_conditions:
            if emo_index is None:
                raise ValueError('emo_index is required for emotion modulation')
            emo_feat = self.emo_embed(emo_index).unsqueeze(1)
            frame_feat_modulated = self._modulate_frame_feature(
                frame_feat_saved, emo_feat
            )
            prev_frame_feat_modulated = self._modulate_frame_feature(
                prev_frame_feat, emo_feat
            )
            frame_feat = torch.where(
                mask_emotion.view(-1, 1, 1),
                self.null_frame_feat.expand(batch_size, self.n_motions, -1),
                frame_feat_modulated,
            )
            prev_frame_feat_in = torch.where(
                mask_emotion.view(-1, 1, 1),
                self.null_frame_feat.expand(batch_size, self.n_prev_motions, -1),
                prev_frame_feat_modulated,
            )
        else:
            frame_feat = self.frame_norm(frame_feat_saved)
            prev_frame_feat_in = self.frame_norm(prev_frame_feat)

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
            frame_feat,
            prev_motion_feat,
            prev_audio_feat_in,
            prev_frame_feat_in,
            time_step,
            indicator,
        )
        return (
            eps,
            motion_feat_target,
            motion_feat.detach(),
            audio_feat_saved.detach(),
            frame_feat_saved.detach(),
        )

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None, frame_emotion_feat=None,
               prev_frame_feat=None):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = self.cfg_mode if cfg_mode is None else cfg_mode
        cfg_cond = self.guiding_conditions if cfg_cond is None else cfg_cond
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]

        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(
                zip(cfg_cond, cfg_scale),
                key=lambda x: ['audio', 'emotion'].index(x[0]),
            ))
        else:
            cfg_cond, cfg_scale = [], []

        audio_feat_saved = self._prepare_audio_feature(audio_or_feat)
        frame_feat_saved = self.extract_frame_feature(frame_emotion_feat)

        if prev_motion_feat is None:
            if emo_index is None:
                raise ValueError('emo_index is required to initialize previous motion')
            prev_motion_feat = torch.index_select(
                self.start_motion_feat, 0, emo_index
            )
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(batch_size, -1, -1)
        if prev_frame_feat is None:
            prev_frame_feat = self.start_frame_feat.expand(batch_size, -1, -1)
        if motion_at_T is None:
            motion_at_T = torch.randn(
                (batch_size, self.n_motions, self.motion_feat_dim),
                device=self.device,
            )

        null_audio = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )
        null_prev_audio = self.null_audio_feat.expand(
            batch_size, self.n_prev_motions, -1
        )
        null_frame = self.null_frame_feat.expand(
            batch_size, self.n_motions, -1
        )
        null_prev_frame = self.null_frame_feat.expand(
            batch_size, self.n_prev_motions, -1
        )

        if 'emotion' in cfg_cond:
            if emo_index is None:
                raise ValueError('emo_index is required for emotion modulation')
            emo_feat = self.emo_embed(emo_index).unsqueeze(1)
            full_frame = self._modulate_frame_feature(frame_feat_saved, emo_feat)
            full_prev_frame = self._modulate_frame_feature(prev_frame_feat, emo_feat)
        else:
            full_frame = self.frame_norm(frame_feat_saved)
            full_prev_frame = self.frame_norm(prev_frame_feat)

        audio_state = null_audio if 'audio' in cfg_cond else audio_feat_saved
        prev_audio_state = (
            null_prev_audio if 'audio' in cfg_cond else prev_audio_feat
        )
        frame_state = null_frame if 'emotion' in cfg_cond else full_frame
        prev_frame_state = (
            null_prev_frame if 'emotion' in cfg_cond else full_prev_frame
        )

        audio_feat_entries = [audio_state]
        prev_audio_entries = [prev_audio_state]
        frame_feat_entries = [frame_state]
        prev_frame_entries = [prev_frame_state]

        for cond in cfg_cond:
            if cond == 'audio':
                audio_state = audio_feat_saved
                prev_audio_state = prev_audio_feat
            elif cond == 'emotion':
                frame_state = full_frame
                prev_frame_state = full_prev_frame
            audio_feat_entries.append(audio_state)
            prev_audio_entries.append(prev_audio_state)
            frame_feat_entries.append(frame_state)
            prev_frame_entries.append(prev_frame_state)

        n_entries = len(audio_feat_entries)
        audio_feat_in = torch.cat(audio_feat_entries, dim=0)
        prev_audio_feat_in = torch.cat(prev_audio_entries, dim=0)
        frame_feat_in = torch.cat(frame_feat_entries, dim=0)
        prev_frame_feat_in = torch.cat(prev_frame_entries, dim=0)
        prev_motion_feat_in = torch.cat(
            [prev_motion_feat] * n_entries, dim=0
        )
        indicator_in = (
            torch.cat([indicator] * n_entries, dim=0)
            if indicator is not None else None
        )

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
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
                frame_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                prev_frame_feat_in,
                step_in,
                indicator_in,
            )

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(
                    batch_size * n_entries, -1
                ).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)
            target_theta = results[0][:, -self.n_motions:]
            for i in range(n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                        results[i + 1][:, -self.n_motions:]
                        - results[0][:, -self.n_motions:]
                    )
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                        results[i + 1][:, -self.n_motions:]
                        - results[i][:, -self.n_motions:]
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
            return traj, motion_at_T, audio_feat_saved, frame_feat_saved
        return traj[0], motion_at_T, audio_feat_saved, frame_feat_saved


class DenoisingNetwork(nn.Module):
    def __init__(self, device='cuda', motion_feat_dim=70, use_indicator=None,
                 architecture='decoder', feature_dim=512, n_heads=8,
                 n_layers=8, mlp_ratio=4, align_mask_width=1,
                 no_use_learnable_pe=True, n_prev_motions=10,
                 n_motions=100, n_diff_steps=500):
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

        self.TE = PositionalEncoding(
            self.feature_dim, max_len=n_diff_steps + 1
        )
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if self.use_learnable_pe:
            self.PE = nn.Parameter(torch.randn(
                1,
                1 + self.n_prev_motions + self.n_motions,
                self.feature_dim,
            ))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        if self.architecture != 'decoder':
            raise ValueError(f'Unknown architecture: {self.architecture}')

        self.feature_proj = nn.Linear(
            self.motion_feat_dim + (1 if self.use_indicator else 0),
            self.feature_dim,
        )
        decoder_layer = DualCrossAttentionDecoderLayer(
            d_model=self.feature_dim,
            nhead=self.n_heads,
            dim_feedforward=self.mlp_ratio * self.feature_dim,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = DualCrossAttentionDecoder(
            decoder_layer, num_layers=self.n_layers
        )

        if self.align_mask_width > 0:
            motion_len = self.n_prev_motions + self.n_motions
            alignment_mask = enc_dec_mask(
                motion_len,
                motion_len,
                frame_width=1,
                expansion=self.align_mask_width - 1,
                device=device,
            )
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

    def forward(self, motion_feat, audio_feat, frame_feat, prev_motion_feat,
                prev_audio_feat, prev_frame_feat, step, indicator=None):
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

        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        feats_in = self.feature_proj(feats_in)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE + diff_step_embedding
        else:
            feats_in = self.PE(feats_in) + diff_step_embedding

        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
        frame_feat_in = torch.cat([prev_frame_feat, frame_feat], dim=1)
        feat_out = self.transformer(
            feats_in,
            audio_feat_in,
            frame_feat_in,
            memory_mask=self.alignment_mask,
        )
        return self.motion_dec(feat_out)

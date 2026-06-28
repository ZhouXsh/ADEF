"""Step-aware emotion hidden adapter for ADEF/JoyVASA-style implicit-keypoint motion diffusion.

Version v1 is intentionally conservative:
- keep the original audio-motion TransformerDecoder and its alignment_mask;
- remove emotion-as-audio-modulation;
- add a zero-initialized hidden residual adapter after the audio-aligned transformer;
- use step-aware implicit emotion tokens [B, K, C].

The adapter never generates a standalone motion and never adds anything to x_t.
It only adapts the hidden state before the original motion decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding
from .emotion_dit import DitTalkingHead as _BaseDitTalkingHead
from .emotion_dit import DenoisingNetwork as _BaseDenoisingNetwork


class StepAwareEmotionEncoder(nn.Module):
    """Encode an emotion label into timestep-aware implicit tokens.

    Output shape: [B, K, C]. The K tokens are implicit emotion subspace tokens;
    they are not tied to explicit facial regions such as lips or eyebrows.
    """

    def __init__(self, feature_dim: int, emo_classes: int, n_diff_steps: int, n_tokens: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_diff_steps = n_diff_steps
        self.n_tokens = n_tokens

        self.emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.null_emo = nn.Parameter(torch.zeros(1, feature_dim))
        self.step_pe = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.step_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.token_base = nn.Parameter(torch.randn(n_tokens, feature_dim) * 0.02)
        self.token_proj = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, n_tokens * feature_dim),
        )
        self.token_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, n_tokens),
        )

    def forward(self, emo_index: torch.Tensor, step: torch.Tensor, drop_mask: torch.Tensor | None = None):
        B = emo_index.shape[0]
        C = self.feature_dim
        step = step.to(emo_index.device).long()

        emo = self.emo_embed(emo_index)
        if drop_mask is not None:
            null_emo = self.null_emo.expand(B, -1)
            emo = torch.where(drop_mask.view(B, 1), null_emo, emo)

        step_emb = self.step_mlp(self.step_pe.pe[0, step].to(emo.device))
        cond = torch.cat([emo, step_emb], dim=-1)
        delta = self.token_proj(cond).view(B, self.n_tokens, C)
        gate = torch.sigmoid(self.token_gate(step_emb)).view(B, self.n_tokens, 1)
        tokens = (self.token_base.unsqueeze(0) + delta) * gate
        return tokens


class EmotionHiddenAdapter(nn.Module):
    """Zero-init hidden residual adapter.

    Input:
        hidden: [B, Lp + L, C], after audio-aligned TransformerDecoder
        emo_tokens: [B, K, C], step-aware implicit emotion tokens
        step_emb: [B, C]
    Output:
        hidden_fused: [B, Lp + L, C]

    This module does not produce motion. It only modifies hidden features before
    the shared motion decoder.
    """

    def __init__(self, feature_dim: int, n_heads: int, mlp_ratio: int = 4, adapter_scale: float = 1.0):
        super().__init__()
        self.adapter_scale = adapter_scale
        self.norm_h = nn.LayerNorm(feature_dim)
        self.norm_e = nn.LayerNorm(feature_dim)
        self.cross_attn = nn.MultiheadAttention(feature_dim, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, mlp_ratio * feature_dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * feature_dim, feature_dim),
        )
        self.zero_proj = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.zero_proj.weight)
        nn.init.zeros_(self.zero_proj.bias)
        self.step_gate = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, hidden: torch.Tensor, emo_tokens: torch.Tensor, step_emb: torch.Tensor):
        query = self.norm_h(hidden)
        memory = self.norm_e(emo_tokens)
        emo_ctx, _ = self.cross_attn(query=query, key=memory, value=memory, need_weights=False)
        emo_ctx = emo_ctx + self.ffn(emo_ctx)
        gate = self.step_gate(step_emb).unsqueeze(1)
        residual = self.zero_proj(emo_ctx) * gate * self.adapter_scale
        return hidden + residual, residual


class DenoisingNetworkV1(_BaseDenoisingNetwork):
    def __init__(self, *args, emotion_adapter_scale: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
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

        feats_in = self.feature_proj(feats_in)
        if self.use_learnable_pe:
            feats_in = feats_in + self.PE + diff_step_embedding
        else:
            feats_in = self.PE(feats_in) + diff_step_embedding

        audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
        feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)

        if emo_tokens is not None:
            step_emb = diff_step_embedding.squeeze(1)
            feat_out, emo_residual = self.emotion_adapter(feat_out, emo_tokens, step_emb)
            self.last_emo_residual_norm = emo_residual.detach().pow(2).mean()
        else:
            self.last_emo_residual_norm = None

        motion_feat_target = self.motion_dec(feat_out)
        return motion_feat_target


class DitTalkingHead(_BaseDitTalkingHead):
    """v1: step-aware emotion hidden adapter, no emotion-to-audio modulation."""

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
        self.emotion_encoder = StepAwareEmotionEncoder(feature_dim, emo_classes, n_diff_steps, n_tokens=4)
        self.denoising_net = DenoisingNetworkV1(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
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

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
                time_step=None, indicator=None, emo_index=None):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev_features(batch_size, emo_index, prev_motion_feat, prev_audio_feat)
        step_tensor = self._step_tensor(time_step, batch_size)

        p_AE = 0.1
        p_E = 0.55
        mask_flag = torch.rand(batch_size, device=self.device)
        mask_audio = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_emotion = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        if len(self.guiding_conditions) > 0:
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = torch.rand(batch_size, device=self.device) < null_cond_prob
                if 'emotion' in self.guiding_conditions:
                    mask_emotion = torch.rand(batch_size, device=self.device) < null_cond_prob
            else:
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag < p_AE
                if 'emotion' in self.guiding_conditions:
                    mask_emotion = mask_flag < p_E

        if 'audio' in self.guiding_conditions:
            audio_feat = torch.where(mask_audio.view(-1, 1, 1),
                                     self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                                     audio_feat)
            audio_feat = self.audio_norm(audio_feat)
            prev_audio_feat = self.audio_norm(prev_audio_feat)

        emo_tokens = None
        if 'emotion' in self.guiding_conditions:
            emo_tokens = self.emotion_encoder(emo_index, step_tensor, drop_mask=mask_emotion)

        alpha_bar = self.diffusion_sched.alpha_bars[step_tensor]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            step_tensor,
            indicator,
            emo_tokens=emo_tokens,
        )
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, motion_at_T=None,
               indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False, emo_index=None):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = cfg_mode or self.cfg_mode
        cfg_cond = cfg_cond or self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, list):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'emotion'].index(x[0])))
        else:
            cfg_cond, cfg_scale = [], []
        print(f'cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}')

        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev_features(batch_size, emo_index, prev_motion_feat, prev_audio_feat)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        audio_real = self.audio_norm(audio_feat)
        prev_audio_real = self.audio_norm(prev_audio_feat)
        audio_null = self.audio_norm(self.null_audio_feat.expand(batch_size, self.n_motions, -1)) if 'audio' in cfg_cond else audio_real
        prev_audio_null = self.audio_norm(self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)) if 'audio' in cfg_cond else prev_audio_real

        audio_entries = [audio_null]
        prev_audio_entries = [prev_audio_null]
        emo_drop_entries = [True]
        for cond in cfg_cond:
            if cond == 'audio':
                audio_entries.append(audio_real)
                prev_audio_entries.append(prev_audio_real)
                emo_drop_entries.append(True if 'emotion' in cfg_cond else False)
            elif cond == 'emotion':
                audio_entries.append(audio_real)
                prev_audio_entries.append(prev_audio_real)
                emo_drop_entries.append(False)

        n_entries = len(audio_entries)
        audio_feat_in = torch.cat(audio_entries, dim=0)
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
            step_single = torch.tensor([t] * batch_size, device=self.device, dtype=torch.long)
            step_in = torch.cat([step_single] * n_entries, dim=0)

            emo_tokens_in = None
            if 'emotion' in cfg_cond:
                tokens = []
                for drop in emo_drop_entries:
                    drop_mask = torch.full((batch_size,), drop, dtype=torch.bool, device=self.device)
                    tokens.append(self.emotion_encoder(emo_index, step_single, drop_mask=drop_mask))
                emo_tokens_in = torch.cat(tokens, dim=0)

            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in,
                                         emo_tokens=emo_tokens_in)
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
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
                raise ValueError(f'Unknown target type: {self.target}')

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat_saved
        return traj[0], motion_at_T, audio_feat_saved

## 两阶段训练版本。直接基于 emotion_dit_Unification.py 复制并修改。
## Stage 1: audio-only condition; Stage 2: audio + emotion joint condition.
## The denoising network remains emotion_dit_timestep_0714.DenoisingNetwork.

import torch
import torch.nn as nn

from .emotion_dit_timestep_0714 import (
    DiffusionSchedule,
    DenoisingNetwork,
    DitTalkingHead as BaseDitTalkingHead,
)


class DitTalkingHead(BaseDitTalkingHead):
    """Use one model for audio pretraining and audio-emotion finetuning.

    ``condition_mode='audio'`` is used in Stage 1.  The projected audio feature
    always passes through ``audio_norm`` so that Stage 1 and Stage 2 expose the
    denoising network to the same base condition distribution.

    ``condition_mode='audio_emotion'`` is used in Stage 2.  Emotion is added as
    an adaLN residual on top of the normalized audio feature.  The modulation
    layer is zero-initialized, therefore a Stage-2 model initially behaves like
    the Stage-1 audio-only model.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not hasattr(self, 'audio_norm'):
            raise ValueError(
                "Two-stage training requires 'audio' in guiding_conditions."
            )
        if not all(hasattr(self, name) for name in (
            'emo_embed', 'adaLN_modulation', 'null_emotion_feat'
        )):
            raise ValueError(
                "Two-stage training requires 'emotion' in guiding_conditions."
            )

        # Stage 2 starts exactly from the Stage-1 audio condition path.
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def copy_neutral_start_to_all(self, neutral_index=5):
        """Initialize every emotion-specific start token from the Stage-1 token."""
        if not 0 <= neutral_index < self.start_motion_feat.shape[0]:
            raise ValueError(f'Invalid neutral emotion index: {neutral_index}')
        with torch.no_grad():
            self.start_motion_feat.copy_(
                self.start_motion_feat[neutral_index:neutral_index + 1].expand_as(
                    self.start_motion_feat
                )
            )
            self.start_audio_feat.copy_(
                self.start_audio_feat[neutral_index:neutral_index + 1].expand_as(
                    self.start_audio_feat
                )
            )

    def _resolve_emo_index(self, emo_index, batch_size, neutral_index=5):
        if emo_index is None:
            emo_index = torch.full(
                (batch_size,), neutral_index, dtype=torch.long, device=self.device
            )
        return emo_index.to(device=self.device, dtype=torch.long)

    def _extract_audio(self, audio_or_feat):
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(
                16000 * self.n_motions / self.fps
            ), f'Incorrect audio length {audio_or_feat.shape[1]}'
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            return audio_or_feat
        raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

    def _emotion_modulation(self, emo_index):
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        return self.adaLN_modulation(emo_feat).chunk(2, dim=2)

    def _null_emotion_modulation(self, batch_size):
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        return self.adaLN_modulation(null_emotion_feat).chunk(2, dim=2)

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None,
                prev_audio_feat=None, time_step=None, indicator=None,
                emo_index=None, condition_mode='audio_emotion',
                uncond_drop_prob=0.1, emotion_drop_prob=0.2,
                return_condition_info=False):
        if condition_mode not in ['audio', 'audio_emotion']:
            raise ValueError(f'Unknown condition mode: {condition_mode}')
        if not 0 <= uncond_drop_prob <= 1:
            raise ValueError('uncond_drop_prob must be in [0, 1].')
        if not 0 <= emotion_drop_prob <= 1:
            raise ValueError('emotion_drop_prob must be in [0, 1].')
        if condition_mode == 'audio_emotion' and \
                uncond_drop_prob + emotion_drop_prob > 1:
            raise ValueError(
                'uncond_drop_prob + emotion_drop_prob must not exceed 1.'
            )

        batch_size = motion_feat.shape[0]
        emo_index = self._resolve_emo_index(emo_index, batch_size)
        audio_feat_saved = self._extract_audio(audio_or_feat)

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(
                self.start_motion_feat, 0, emo_index
            )
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(
                self.start_audio_feat, 0, emo_index
            )

        # The base audio path is identical in both stages.
        audio_feat_base = self.audio_norm(audio_feat_saved)
        prev_audio_feat_base = self.audio_norm(prev_audio_feat)

        # Audio-only condition used in Stage 1 and for Stage-2 condition dropout.
        audio_feat_audio_only = audio_feat_base
        prev_audio_feat_audio_only = prev_audio_feat_base

        # Full audio + emotion condition used in Stage 2.
        emo_shift, emo_scale = self._emotion_modulation(emo_index)
        audio_feat_full = audio_feat_base * (1 + emo_scale) + emo_shift
        prev_audio_feat_full = (
            prev_audio_feat_base * (1 + emo_scale) + emo_shift
        )

        # Fully unconditional branch for CFG training.
        null_audio_feat = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )
        null_prev_audio_feat = self.null_audio_feat.expand(
            batch_size, self.n_prev_motions, -1
        )
        if condition_mode == 'audio_emotion':
            null_shift, null_scale = self._null_emotion_modulation(batch_size)
            audio_feat_uncond = (
                self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
            )
            prev_audio_feat_uncond = (
                self.audio_norm(null_prev_audio_feat)
                * (1 + null_scale) + null_shift
            )
        else:
            audio_feat_uncond = self.audio_norm(null_audio_feat)
            prev_audio_feat_uncond = self.audio_norm(null_prev_audio_feat)

        if self.training:
            random_value = torch.rand(batch_size, device=self.device)
            mask_uncond = random_value < uncond_drop_prob
            if condition_mode == 'audio_emotion':
                mask_audio_only = (
                    (random_value >= uncond_drop_prob)
                    & (random_value < uncond_drop_prob + emotion_drop_prob)
                )
            else:
                mask_audio_only = ~mask_uncond
        else:
            # Validation is deterministic and always uses the requested condition.
            mask_uncond = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )
            mask_audio_only = torch.full(
                (batch_size,), condition_mode == 'audio',
                dtype=torch.bool, device=self.device
            )

        mask_full = ~(mask_uncond | mask_audio_only)
        audio_feat = torch.where(
            mask_full.view(-1, 1, 1), audio_feat_full, audio_feat_audio_only
        )
        prev_audio_feat_in = torch.where(
            mask_full.view(-1, 1, 1),
            prev_audio_feat_full,
            prev_audio_feat_audio_only,
        )
        audio_feat = torch.where(
            mask_uncond.view(-1, 1, 1), audio_feat_uncond, audio_feat
        )
        prev_audio_feat_in = torch.where(
            mask_uncond.view(-1, 1, 1),
            prev_audio_feat_uncond,
            prev_audio_feat_in,
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
            prev_audio_feat_in,
            time_step,
            indicator,
        )

        outputs = (
            eps,
            motion_feat_target,
            motion_feat.detach(),
            audio_feat_saved.detach(),
        )
        if not return_condition_info:
            return outputs

        condition_info = {
            'audio_active': ~mask_uncond,
            'emotion_active': mask_full,
            'unconditional': mask_uncond,
            'audio_only': mask_audio_only,
            'full_condition': mask_full,
        }
        return outputs + (condition_info,)

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None,
               condition_mode='audio_emotion'):
        if condition_mode not in ['audio', 'audio_emotion']:
            raise ValueError(f'Unknown condition mode: {condition_mode}')

        batch_size = audio_or_feat.shape[0]
        emo_index = self._resolve_emo_index(emo_index, batch_size)

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_mode not in ['incremental', 'independent']:
            raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        elif isinstance(cfg_cond, str):
            cfg_cond = cfg_cond.split(',')
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        use_cfg = len(cfg_cond) > 0

        if isinstance(cfg_scale, (list, tuple)):
            joint_cfg_scale = cfg_scale[-1] if len(cfg_scale) > 0 else 1.0
        else:
            joint_cfg_scale = cfg_scale

        condition_name = (
            'audio+emotion' if condition_mode == 'audio_emotion' else 'audio'
        )
        print(
            f"cfg_cond: {(condition_name,) if use_cfg else ()}, "
            f"cfg_scale: {(joint_cfg_scale,) if use_cfg else ()}"
        )

        audio_feat_saved = self._extract_audio(audio_or_feat)

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(
                self.start_motion_feat, 0, emo_index
            )
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(
                self.start_audio_feat, 0, emo_index
            )

        if motion_at_T is None:
            motion_at_T = torch.randn(
                batch_size, self.n_motions, self.motion_feat_dim,
                device=self.device,
            )

        audio_feat_base = self.audio_norm(audio_feat_saved)
        prev_audio_feat_base = self.audio_norm(prev_audio_feat)
        if condition_mode == 'audio_emotion':
            emo_shift, emo_scale = self._emotion_modulation(emo_index)
            audio_feat_cond = (
                audio_feat_base * (1 + emo_scale) + emo_shift
            )
            prev_audio_feat_cond = (
                prev_audio_feat_base * (1 + emo_scale) + emo_shift
            )
        else:
            audio_feat_cond = audio_feat_base
            prev_audio_feat_cond = prev_audio_feat_base

        null_audio_feat = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )
        null_prev_audio_feat = self.null_audio_feat.expand(
            batch_size, self.n_prev_motions, -1
        )
        if condition_mode == 'audio_emotion':
            null_shift, null_scale = self._null_emotion_modulation(batch_size)
            audio_feat_uncond = (
                self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
            )
            prev_audio_feat_uncond = (
                self.audio_norm(null_prev_audio_feat)
                * (1 + null_scale) + null_shift
            )
        else:
            audio_feat_uncond = self.audio_norm(null_audio_feat)
            prev_audio_feat_uncond = self.audio_norm(null_prev_audio_feat)

        if use_cfg:
            audio_feat_in = torch.cat(
                [audio_feat_uncond, audio_feat_cond], dim=0
            )
            prev_audio_feat_in = torch.cat(
                [prev_audio_feat_uncond, prev_audio_feat_cond], dim=0
            )
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            prev_audio_feat_in = prev_audio_feat_cond
            n_entries = 1

        prev_motion_feat_in = torch.cat(
            [prev_motion_feat] * n_entries, dim=0
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
            if use_cfg:
                uncond_target = results[0][:, -self.n_motions:]
                cond_target = results[1][:, -self.n_motions:]
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = results[0][:, -self.n_motions:]

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (
                    motion_at_t - c1 * target_theta
                ) + sigma * z
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
                    c0 * motion_at_t + c1 * target_theta + sigma * z
                )
            else:
                raise ValueError(f'Unknown target type: {self.target}')

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        # Return the unmodulated projected audio feature.  The next window will
        # normalize/modulate it exactly once, avoiding repeated emotion adaLN.
        if ret_traj:
            return traj, motion_at_T, audio_feat_saved
        return traj[0], motion_at_T, audio_feat_saved


__all__ = ['DiffusionSchedule', 'DenoisingNetwork', 'DitTalkingHead']

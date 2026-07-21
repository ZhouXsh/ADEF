## No-CFG version based on emotion_dit_Unification.py.
## Audio and emotion are always present and are treated as one required condition.

import torch

from .emotion_dit_timestep_0714 import (
    DiffusionSchedule,
    DenoisingNetwork,
    DitTalkingHead as BaseDitTalkingHead,
)


class DitTalkingHead(BaseDitTalkingHead):
    """Audio and emotion are treated as one inseparable required condition."""

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None,
                prev_audio_feat=None, time_step=None, indicator=None,
                emo_index=None):
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
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
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

        # Required condition: real audio + real emotion.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

        # Keep the original previous-audio handling unchanged.
        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
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

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None):
        # Retain the legacy arguments so train.py and existing inference callers
        # can keep the same interface. No CFG branch is constructed or applied.
        _ = cfg_mode, cfg_cond, cfg_scale

        batch_size = audio_or_feat.shape[0]

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

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
                batch_size, self.n_motions, self.motion_feat_dim,
                device=self.device,
            )

        # Full required condition.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat_cond = (
            self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift
        )

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        # Preserve the original reverse-diffusion structure with one entry only.
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

        if ret_traj:
            return traj, motion_at_T, audio_feat_cond
        return traj[0], motion_at_T, audio_feat_cond


__all__ = ['DiffusionSchedule', 'DenoisingNetwork', 'DitTalkingHead']

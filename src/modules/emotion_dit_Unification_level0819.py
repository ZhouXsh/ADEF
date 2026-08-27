## Unification + emotion level version
## 在 emotion_dit_Unification_jianhua0803.py 的基础上，只增加一个显式的 emotion-level 条件：
##   condition = emotion embedding + level embedding
## 其余 DiT / diffusion / audio 路径保持不变，便于和 baseline 做单变量对照。

import torch

from .emotion_dit_Unification_jianhua0803 import DitTalkingHead as BaseDitTalkingHead


class DitTalkingHead(BaseDitTalkingHead):
    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=64, n_prev_motions=16,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental",
                 guiding_conditions="audio,emotion", emo_classes=8,
                 level_classes=3, align_mask_width=1):
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
            align_mask_width=align_mask_width,
        )

        if 'emotion' not in self.guiding_conditions:
            raise ValueError(
                "emotion_dit_Unification_level0819 requires 'emotion' in guiding_conditions"
            )

        self.level_classes = level_classes
        self.level_embed = torch.nn.Embedding(level_classes, feature_dim)
        self.to(device)

    def _get_emotion_level_feat(self, emo_index, emo_level=None):
        if emo_level is None:
            # 兼容旧推理调用；默认 MEAD level-1 -> index 0。
            emo_level = torch.zeros_like(emo_index)
        emo_level = emo_level.long().clamp(0, self.level_classes - 1)
        emo_feat = self.emo_embed(emo_index)
        level_feat = self.level_embed(emo_level)
        return (emo_feat + level_feat).unsqueeze(1)

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None,
                prev_audio_feat=None, time_step=None, indicator=None,
                emo_index=None, emo_level=None):
        # 本版本保持 jianhua0803 的 joint CFG：audio + (emotion, level) 一起保留/一起 drop。
        if not (
            'audio' in self.guiding_conditions
            and 'emotion' in self.guiding_conditions
        ):
            raise ValueError(
                "level0819 currently expects guiding_conditions='audio,emotion'"
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

        # Conditional branch: real audio + emotion type + emotion level.
        emo_level_feat = self._get_emotion_level_feat(emo_index, emo_level)
        emo_shift, emo_scale = self.adaLN_modulation(emo_level_feat).chunk(2, dim=2)
        audio_feat_cond = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        # Unconditional branch: null audio + null emotion/level.
        null_audio_feat = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(
            null_emotion_feat
        ).chunk(2, dim=2)
        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )

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

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None, emo_level=None):
        batch_size = audio_or_feat.shape[0]

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_mode not in ['incremental', 'independent']:
            raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        elif isinstance(cfg_cond, str):
            cfg_cond = cfg_cond.split(',')
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]

        use_joint_cfg = (
            len(cfg_cond) > 0
            and 'audio' in self.guiding_conditions
            and 'emotion' in self.guiding_conditions
        )
        if isinstance(cfg_scale, (list, tuple)):
            joint_cfg_scale = cfg_scale[-1] if len(cfg_scale) > 0 else 1.0
        else:
            joint_cfg_scale = cfg_scale

        print(
            f"cfg_cond: {('audio+emotion+level',) if use_joint_cfg else ()}, "
            f"cfg_scale: {(joint_cfg_scale,) if use_joint_cfg else ()}"
        )

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

        emo_level_feat = self._get_emotion_level_feat(emo_index, emo_level)
        emo_shift, emo_scale = self.adaLN_modulation(emo_level_feat).chunk(2, dim=2)
        audio_feat_cond = (
            self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift
        )

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        null_audio_feat = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(
            null_emotion_feat
        ).chunk(2, dim=2)
        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )

        if use_joint_cfg:
            audio_feat_in = torch.cat([audio_feat_uncond, audio_feat_cond], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            n_entries = 1

        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
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
            step_in = torch.tensor([t] * batch_size, device=self.device)
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
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = results[0][:, -self.n_motions:]

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
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


__all__ = ['DitTalkingHead']

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch


class EmotionSamplingMixin:
    def _normalise_cfg_inputs(
        self,
        cfg_cond: Optional[Iterable[str]],
        cfg_scale: float | Sequence[float],
        emotion_enabled: bool,
    ) -> tuple[list[str], list[float]]:
        if isinstance(cfg_cond, str):
            conditions = [item for item in cfg_cond.split(",") if item]
        else:
            conditions = list(cfg_cond) if cfg_cond is not None else list(self.guiding_conditions)
        conditions = [condition for condition in conditions if condition in {"audio", "emotion"}]
        if not emotion_enabled:
            conditions = [condition for condition in conditions if condition != "emotion"]
        order = {"audio": 0, "emotion": 1}
        conditions = sorted(dict.fromkeys(conditions), key=order.get)

        if isinstance(cfg_scale, (int, float)):
            scales = [float(cfg_scale)] * len(conditions)
        else:
            scales = [float(value) for value in cfg_scale]
            if len(scales) != len(conditions):
                raise ValueError("cfg_scale length must match cfg_cond length")
        return conditions, scales

    @torch.no_grad()
    def sample(
        self,
        audio_or_feat: torch.Tensor,
        prev_motion_feat: Optional[torch.Tensor] = None,
        prev_audio_feat: Optional[torch.Tensor] = None,
        motion_at_T: Optional[torch.Tensor] = None,
        indicator: Optional[torch.Tensor] = None,
        cfg_mode: Optional[str] = None,
        cfg_cond: Optional[Iterable[str]] = None,
        cfg_scale: float | Sequence[float] = 1.15,
        flexibility: float = 0.0,
        dynamic_threshold: Optional[Sequence[float]] = None,
        ret_traj: bool = False,
        emo_index: Optional[torch.Tensor] = None,
        use_emotion: Optional[bool] = None,
    ):
        batch_size = audio_or_feat.shape[0]
        audio_feat_saved = self._extract_or_validate_audio(audio_or_feat)
        emotion_enabled = self._use_emotion(emo_index, use_emotion)
        conditions, scales = self._normalise_cfg_inputs(
            cfg_cond, cfg_scale, emotion_enabled
        )
        cfg_mode = cfg_mode or self.cfg_mode

        generic_prev_motion = prev_motion_feat
        if generic_prev_motion is None:
            generic_prev_motion = self._start_feature(
                self.start_motion_feat, batch_size, emo_index, False
            )
        emotional_prev_motion = prev_motion_feat
        if emotional_prev_motion is None:
            emotional_prev_motion = self._start_feature(
                self.start_motion_feat, batch_size, emo_index, emotion_enabled
            )

        generic_prev_audio = prev_audio_feat
        if generic_prev_audio is None:
            generic_prev_audio = self._start_feature(
                self.start_audio_feat, batch_size, emo_index, False
            )
        emotional_prev_audio = prev_audio_feat
        if emotional_prev_audio is None:
            emotional_prev_audio = self._start_feature(
                self.start_audio_feat, batch_size, emo_index, emotion_enabled
            )

        null_current = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )
        branches_audio = [self._audio_content(null_current)]
        branches_prev_audio = [self._audio_content(generic_prev_audio)]
        branches_prev_motion = [generic_prev_motion]

        for condition in conditions:
            if condition == "audio":
                branches_audio.append(self._audio_content(audio_feat_saved))
                branches_prev_audio.append(self._audio_content(generic_prev_audio))
                branches_prev_motion.append(generic_prev_motion)
            elif condition == "emotion":
                branches_audio.append(
                    self._modulate_audio(
                        audio_feat_saved, emo_index, emotion_enabled
                    )
                )
                branches_prev_audio.append(
                    self._modulate_audio(
                        emotional_prev_audio, emo_index, emotion_enabled
                    )
                )
                branches_prev_motion.append(emotional_prev_motion)

        audio_in = torch.cat(branches_audio, dim=0)
        prev_audio_in = torch.cat(branches_prev_audio, dim=0)
        prev_motion_in = torch.cat(branches_prev_motion, dim=0)
        n_entries = len(branches_audio)
        indicator_in = (
            torch.cat([indicator] * n_entries, dim=0)
            if indicator is not None
            else None
        )

        if motion_at_T is None:
            motion_at_T = torch.randn(
                batch_size,
                self.n_motions,
                self.motion_feat_dim,
                device=self.device,
            )

        trajectory = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = trajectory[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step = torch.full(
                (batch_size * n_entries,), t, device=self.device, dtype=torch.long
            )
            results = self.denoising_net(
                motion_in,
                audio_in,
                prev_motion_in,
                prev_audio_in,
                step,
                indicator_in,
            )

            if dynamic_threshold is not None:
                ratio, minimum, maximum = dynamic_threshold
                current = results[:, -self.n_motions :]
                quantile_input = current.reshape(batch_size * n_entries, -1).abs()
                threshold = torch.quantile(quantile_input, ratio, dim=1)
                threshold = torch.clamp(threshold, min=minimum, max=maximum)
                threshold = threshold[:, None, None]
                results = torch.clamp(results, min=-threshold, max=threshold)

            chunks = results.chunk(n_entries)
            target_theta = chunks[0][:, -self.n_motions :].clone()
            for index in range(n_entries - 1):
                if cfg_mode == "independent":
                    delta = chunks[index + 1][:, -self.n_motions :] - chunks[0][
                        :, -self.n_motions :
                    ]
                elif cfg_mode == "incremental":
                    delta = chunks[index + 1][:, -self.n_motions :] - chunks[index][
                        :, -self.n_motions :
                    ]
                else:
                    raise ValueError(f"Unknown cfg_mode: {cfg_mode}")
                target_theta = target_theta + scales[index] * delta

            if self.target == "noise":
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            else:
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z

            trajectory[t - 1] = motion_next.detach()
            trajectory[t] = trajectory[t].cpu()
            if not ret_traj:
                del trajectory[t]

        if ret_traj:
            return trajectory, motion_at_T, audio_feat_saved
        return trajectory[0], motion_at_T, audio_feat_saved

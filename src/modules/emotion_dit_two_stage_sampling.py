"""Sampling/CFG mixin for the shared two-stage dual-audio model."""

from __future__ import annotations

from typing import Optional, Sequence

import math
import torch


class TwoStageSamplingMixin:
    @staticmethod
    def _scale_at_step(
        maximum: float,
        minimum: float,
        step: int,
        num_steps: int,
        schedule: Optional[str],
    ) -> float:
        if schedule in {None, "none"} or num_steps <= 1:
            return float(maximum)
        progress = (num_steps - float(step)) / float(num_steps - 1)
        progress = max(0.0, min(1.0, progress))
        if schedule == "linear":
            weight = progress
        elif schedule == "cosine":
            weight = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif schedule == "bell":
            weight = math.sin(math.pi * progress)
        else:
            raise ValueError(f"Unknown cfg_schedule: {schedule}")
        return float(minimum) + weight * (
            float(maximum) - float(minimum)
        )

    def _sample_conditions(
        self,
        audio_saved: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        emo_index: Optional[torch.Tensor],
        cfg_cond: Sequence[str],
        emo_utt_feat: Optional[torch.Tensor],
        emo_frame_feat: Optional[torch.Tensor],
        prev_emo_frame_feat: Optional[torch.Tensor],
    ):
        batch_size = audio_saved.shape[0]
        audio_null = self.null_audio_feat.expand(
            batch_size, self.n_motions, -1
        )

        use_audio = "audio" in cfg_cond
        use_emotion = (
            self.train_stage == 2 and "emotion" in cfg_cond
        )

        if use_audio and use_emotion:
            current_raw = torch.cat(
                [audio_null, audio_saved, audio_saved], dim=0
            )
            prev_raw = torch.cat([prev_audio_feat] * 3, dim=0)
            emotion_present = torch.cat([
                torch.zeros(batch_size, dtype=torch.bool, device=self.device),
                torch.zeros(batch_size, dtype=torch.bool, device=self.device),
                torch.ones(batch_size, dtype=torch.bool, device=self.device),
            ])
            n_entries = 3
        elif use_audio:
            current_raw = torch.cat([audio_null, audio_saved], dim=0)
            prev_raw = torch.cat([prev_audio_feat] * 2, dim=0)
            emotion_present = torch.zeros(
                2 * batch_size, dtype=torch.bool, device=self.device
            )
            n_entries = 2
        elif use_emotion:
            current_raw = torch.cat([audio_saved, audio_saved], dim=0)
            prev_raw = torch.cat([prev_audio_feat] * 2, dim=0)
            emotion_present = torch.cat([
                torch.zeros(batch_size, dtype=torch.bool, device=self.device),
                torch.ones(batch_size, dtype=torch.bool, device=self.device),
            ])
            n_entries = 2
        else:
            current_raw = audio_saved
            prev_raw = prev_audio_feat
            emotion_present = torch.zeros(
                batch_size, dtype=torch.bool, device=self.device
            )
            n_entries = 1

        audio_branch = self.audio_norm(current_raw)
        prev_audio_branch = self.audio_norm(prev_raw)
        emotion_audio = None
        prev_emotion_audio = None

        if use_emotion:
            if emo_index is None:
                raise ValueError("emo_index is required for emotion CFG")
            emo_index_in = torch.cat([emo_index] * n_entries, dim=0)
            drop_emotion = ~emotion_present
            utt_in = (
                torch.cat([emo_utt_feat] * n_entries, dim=0)
                if emo_utt_feat is not None else None
            )
            frame_in = (
                torch.cat([emo_frame_feat] * n_entries, dim=0)
                if emo_frame_feat is not None else None
            )
            prev_frame_in = (
                torch.cat([prev_emo_frame_feat] * n_entries, dim=0)
                if prev_emo_frame_feat is not None else None
            )
            emotion_audio = self._build_emotion_audio(
                current_raw,
                emo_index_in,
                utt_in,
                frame_in,
                drop_emotion,
            )
            prev_emotion_audio = self._build_emotion_audio(
                prev_raw,
                emo_index_in,
                utt_in,
                prev_frame_in,
                drop_emotion,
            )

        return (
            audio_branch,
            prev_audio_branch,
            emotion_audio,
            prev_emotion_audio,
            emotion_present,
            n_entries,
        )

    @torch.no_grad()
    def sample(
        self,
        audio_or_feat: torch.Tensor,
        prev_motion_feat: Optional[torch.Tensor] = None,
        prev_audio_feat: Optional[torch.Tensor] = None,
        motion_at_T: Optional[torch.Tensor] = None,
        indicator: Optional[torch.Tensor] = None,
        cfg_mode: Optional[str] = None,
        cfg_cond=None,
        cfg_scale=1.15,
        flexibility: float = 0.0,
        dynamic_threshold=None,
        ret_traj: bool = False,
        emo_index: Optional[torch.Tensor] = None,
        cfg_min: Optional[Sequence[float]] = None,
        cfg_schedule: Optional[str] = None,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        prev_emo_frame_feat: Optional[torch.Tensor] = None,
    ):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = cfg_mode or self.cfg_mode
        if cfg_cond is None:
            cfg_cond = (
                ["audio"] if self.train_stage == 1
                else self.guiding_conditions
            )
        if isinstance(cfg_cond, str):
            cfg_cond = [item for item in cfg_cond.split(",") if item]
        cfg_cond = [
            item for item in cfg_cond if item in {"audio", "emotion"}
        ]
        if self.train_stage == 1:
            cfg_cond = [item for item in cfg_cond if item == "audio"]

        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if cfg_min is None:
            cfg_min = [1.0 if item == "audio" else 0.0 for item in cfg_cond]
        if cfg_cond:
            ordered = sorted(
                zip(cfg_cond, cfg_scale, cfg_min),
                key=lambda item: ["audio", "emotion"].index(item[0]),
            )
            cfg_cond, cfg_scale, cfg_min = map(tuple, zip(*ordered))
        else:
            cfg_cond, cfg_scale, cfg_min = (), (), ()

        audio_saved = self._get_audio_feature(audio_or_feat)
        prev_motion_feat, prev_audio_feat = self._init_previous(
            batch_size, prev_motion_feat, prev_audio_feat
        )
        if motion_at_T is None:
            motion_at_T = torch.randn(
                batch_size,
                self.n_motions,
                self.motion_feat_dim,
                device=self.device,
            )

        (
            audio_branch,
            prev_audio_branch,
            emotion_audio,
            prev_emotion_audio,
            emotion_present,
            n_entries,
        ) = self._sample_conditions(
            audio_saved,
            prev_audio_feat,
            emo_index,
            cfg_cond,
            emo_utt_feat,
            emo_frame_feat,
            prev_emo_frame_feat,
        )

        prev_motion_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        indicator_in = (
            torch.cat([indicator] * n_entries, dim=0)
            if indicator is not None else None
        )

        trajectory = {self.diffusion_sched.num_steps: motion_at_T}
        for step in range(self.diffusion_sched.num_steps, 0, -1):
            z = (
                torch.randn_like(motion_at_T)
                if step > 1 else torch.zeros_like(motion_at_T)
            )
            alpha = self.diffusion_sched.alphas[step]
            alpha_bar = self.diffusion_sched.alpha_bars[step]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[step - 1]
            sigma = self.diffusion_sched.get_sigmas(step, flexibility)

            motion_t = trajectory[step]
            motion_in = torch.cat([motion_t] * n_entries, dim=0)
            step_in = torch.full(
                (batch_size * n_entries,),
                step,
                device=self.device,
                dtype=torch.long,
            )

            prediction = self.denoising_net(
                motion_in,
                audio_branch,
                prev_motion_in,
                prev_audio_branch,
                step_in,
                indicator_in,
                emotion_audio_feat=emotion_audio,
                prev_emotion_audio_feat=prev_emotion_audio,
                emotion_present=emotion_present,
            )

            if dynamic_threshold:
                ratio, minimum, maximum = dynamic_threshold
                values = prediction[:, -self.n_motions:].reshape(
                    batch_size * n_entries, -1
                ).abs()
                threshold = torch.quantile(values, ratio, dim=1)
                threshold = torch.clamp(
                    threshold, min=minimum, max=maximum
                )[..., None, None]
                prediction = torch.clamp(
                    prediction, min=-threshold, max=threshold
                )

            chunks = prediction.chunk(n_entries)
            if n_entries == 3:
                unconditional = chunks[0][:, -self.n_motions:]
                audio_only = chunks[1][:, -self.n_motions:]
                full = chunks[2][:, -self.n_motions:]
                audio_weight = self._scale_at_step(
                    cfg_scale[0], cfg_min[0], step,
                    self.diffusion_sched.num_steps, cfg_schedule
                )
                emotion_weight = self._scale_at_step(
                    cfg_scale[1], cfg_min[1], step,
                    self.diffusion_sched.num_steps, cfg_schedule
                )
                if cfg_mode == "independent":
                    target_theta = (
                        unconditional
                        + audio_weight * (audio_only - unconditional)
                        + emotion_weight * (full - unconditional)
                    )
                else:
                    target_theta = (
                        unconditional
                        + audio_weight * (audio_only - unconditional)
                        + emotion_weight * (full - audio_only)
                    )
            elif n_entries == 2:
                unconditional = chunks[0][:, -self.n_motions:]
                conditional = chunks[1][:, -self.n_motions:]
                weight = self._scale_at_step(
                    cfg_scale[0], cfg_min[0], step,
                    self.diffusion_sched.num_steps, cfg_schedule
                )
                target_theta = unconditional + weight * (
                    conditional - unconditional
                )
            else:
                target_theta = chunks[0][:, -self.n_motions:]

            if self.target == "noise":
                c0 = 1.0 / torch.sqrt(alpha)
                c1 = (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar)
                motion_next = c0 * (
                    motion_t - c1 * target_theta
                ) + sigma * z
            elif self.target == "sample":
                c0 = (
                    (1.0 - alpha_bar_prev) * torch.sqrt(alpha)
                    / (1.0 - alpha_bar)
                )
                c1 = (
                    (1.0 - alpha) * torch.sqrt(alpha_bar_prev)
                    / (1.0 - alpha_bar)
                )
                motion_next = c0 * motion_t + c1 * target_theta + sigma * z
            else:
                raise ValueError(f"Unknown target type: {self.target}")

            trajectory[step - 1] = motion_next.detach()
            trajectory[step] = trajectory[step].cpu()
            if not ret_traj:
                del trajectory[step]

        if ret_traj:
            return trajectory, motion_at_T, audio_saved
        return trajectory[0], motion_at_T, audio_saved

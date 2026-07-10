"""Stage-aware training forward mixin for two-stage dual-audio DiT."""

from __future__ import annotations

from typing import Optional, Tuple

import torch


class TwoStageForwardMixin:
    def _training_masks(
        self,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_audio = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        mask_emotion = torch.ones(
            batch_size, dtype=torch.bool, device=self.device
        )

        if self.train_stage == 1 or "emotion" not in self.guiding_conditions:
            if "audio" in self.guiding_conditions:
                mask_audio = (
                    torch.rand(batch_size, device=self.device) < 0.10
                )
            return mask_audio, mask_emotion

        if (
            "audio" in self.guiding_conditions
            and self.cfg_mode == "incremental"
        ):
            random_value = torch.rand(batch_size, device=self.device)
            mask_audio = random_value < 0.10
            mask_emotion = random_value < 0.55
        else:
            if "audio" in self.guiding_conditions:
                mask_audio = (
                    torch.rand(batch_size, device=self.device) < 0.10
                )
            mask_emotion = (
                torch.rand(batch_size, device=self.device) < 0.50
            )

        # With a frozen Stage-1 base, an all-audio-only mini-batch would have no
        # trainable path. Keep at least one full emotion example in every
        # Stage-2 training batch while preserving the intended distribution.
        if self.training and self.train_stage == 2 and mask_emotion.all():
            keep_index = torch.randint(
                0, batch_size, (1,), device=self.device
            )
            mask_emotion[keep_index] = False
        return mask_audio, mask_emotion

    def forward(
        self,
        motion_feat: torch.Tensor,
        audio_or_feat: torch.Tensor,
        prev_motion_feat: Optional[torch.Tensor] = None,
        prev_audio_feat: Optional[torch.Tensor] = None,
        time_step=None,
        indicator: Optional[torch.Tensor] = None,
        emo_index: Optional[torch.Tensor] = None,
        emo_utt_feat: Optional[torch.Tensor] = None,
        emo_frame_feat: Optional[torch.Tensor] = None,
        prev_emo_frame_feat: Optional[torch.Tensor] = None,
    ):
        batch_size = motion_feat.shape[0]
        audio_saved = self._get_audio_feature(audio_or_feat)
        prev_motion_feat, prev_audio_feat = self._init_previous(
            batch_size, prev_motion_feat, prev_audio_feat
        )

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)
        time_step = torch.as_tensor(
            time_step, device=self.device, dtype=torch.long
        )

        mask_audio, mask_emotion = self._training_masks(batch_size)
        current_audio = torch.where(
            mask_audio.view(batch_size, 1, 1),
            self.null_audio_feat.expand(batch_size, self.n_motions, -1),
            audio_saved,
        )
        audio_branch = self.audio_norm(current_audio)
        prev_audio_branch = self.audio_norm(prev_audio_feat)

        emotion_audio = None
        prev_emotion_audio = None
        emotion_present = None
        if self.train_stage == 2 and "emotion" in self.guiding_conditions:
            if emo_index is None:
                raise ValueError("emo_index is required in Stage 2")
            emotion_audio = self._build_emotion_audio(
                current_audio,
                emo_index,
                emo_utt_feat,
                emo_frame_feat,
                mask_emotion,
            )
            prev_emotion_audio = self._build_emotion_audio(
                prev_audio_feat,
                emo_index,
                emo_utt_feat,
                prev_emo_frame_feat,
                mask_emotion,
            )
            emotion_present = ~mask_emotion

        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1.0 - alpha_bar).view(-1, 1, 1)
        noise = torch.randn_like(motion_feat)
        noisy_motion = c0 * motion_feat + c1 * noise

        target = self.denoising_net(
            noisy_motion,
            audio_branch,
            prev_motion_feat,
            prev_audio_branch,
            time_step,
            indicator,
            emotion_audio_feat=emotion_audio,
            prev_emotion_audio_feat=prev_emotion_audio,
            emotion_present=emotion_present,
        )
        return (
            noise,
            target,
            motion_feat.detach(),
            audio_saved.detach(),
        )

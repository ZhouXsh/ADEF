import torch

from .emotion_dit import DitTalkingHead as BaseDitTalkingHead


class DitTalkingHead(BaseDitTalkingHead):
    """VASA-style DiT using explicit previous-window context.

    The original denoising architecture and sampling process are retained.
    Learned first-window tokens are replaced by fixed zeros, and training
    jointly drops the previous motion/audio conditions per sample.
    """

    def __init__(self, *args, prev_dropout_prob=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_dropout_prob = prev_dropout_prob

        # Keep checkpoint/API compatibility while ensuring the first-window
        # representation is fixed rather than learned.
        del self.start_audio_feat
        del self.start_motion_feat
        self.register_buffer(
            'start_audio_feat',
            torch.zeros(8, self.n_prev_motions, self.feature_dim),
            persistent=False,
        )
        self.register_buffer(
            'start_motion_feat',
            torch.zeros(8, self.n_prev_motions, self.motion_feat_dim),
            persistent=False,
        )
        self.last_prev_dropout_mask = None

    def _extract_current_audio(self, audio_or_feat):
        if audio_or_feat.ndim == 2:
            expected = round(16000 * self.n_motions / self.fps)
            assert audio_or_feat.shape[1] == expected, (
                f'Incorrect audio length {audio_or_feat.shape[1]}, expected {expected}'
            )
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, (
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            )
            return audio_or_feat
        raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

    def _prepare_previous(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat):
        batch_size = motion_feat.shape[0]
        if (prev_motion_feat is None) != (prev_audio_feat is None):
            raise ValueError(
                'prev_motion_feat and prev_audio_feat must be both provided or both None'
            )
        if prev_motion_feat is None:
            prev_motion_feat = torch.zeros(
                batch_size,
                self.n_prev_motions,
                self.motion_feat_dim,
                device=motion_feat.device,
                dtype=motion_feat.dtype,
            )
            prev_audio_feat = torch.zeros(
                batch_size,
                self.n_prev_motions,
                self.feature_dim,
                device=audio_feat.device,
                dtype=audio_feat.dtype,
            )
        else:
            assert prev_motion_feat.shape[1:] == (
                self.n_prev_motions,
                self.motion_feat_dim,
            )
            assert prev_audio_feat.shape[1:] == (
                self.n_prev_motions,
                self.feature_dim,
            )

        previous_valid = (
            prev_motion_feat.detach().abs().sum(dim=(1, 2))
            + prev_audio_feat.detach().abs().sum(dim=(1, 2))
        ) > 0
        return prev_motion_feat, prev_audio_feat, previous_valid

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
    ):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._extract_current_audio(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat, previous_valid = self._prepare_previous(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat
        )

        if 'emotion' in self.guiding_conditions:
            if emo_index is None:
                raise ValueError('emo_index is required when emotion guidance is enabled')
            emo_feat = self.emo_embed(emo_index).unsqueeze(1)
            emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
            modulated_prev_audio = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )
            # Explicit zero history must remain zero after emotion modulation.
            prev_audio_feat = torch.where(
                previous_valid[:, None, None],
                modulated_prev_audio,
                prev_audio_feat,
            )

        # Preserve the original classifier-free audio/emotion condition policy.
        p_audio_and_emotion = 0.1
        p_emotion = 0.55
        if self.guiding_conditions:
            assert len(self.guiding_conditions) <= 2, (
                'Only support 1 or 2 CFG conditions!'
            )
            mask_flag = torch.rand(batch_size, device=self.device)
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = (
                        torch.rand(batch_size, device=self.device) < null_cond_prob
                    )
                    audio_feat = torch.where(
                        mask_audio[:, None, None],
                        self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                        audio_feat,
                    )
            elif 'audio' in self.guiding_conditions:
                mask_audio = mask_flag < p_audio_and_emotion
                audio_feat = torch.where(
                    mask_audio[:, None, None],
                    self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                    audio_feat,
                )

            if len(self.guiding_conditions) == 2 and 'emotion' in self.guiding_conditions:
                mask_emotion = mask_flag < p_emotion
                emo_feat = torch.where(
                    mask_emotion[:, None, None],
                    self.null_emotion_feat.expand(batch_size, -1, -1),
                    emo_feat,
                )
                emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
                audio_feat = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

        # Jointly remove previous motion and audio. Applying this after all
        # normalization/modulation exactly matches zero-context inference.
        drop_previous = torch.zeros(
            batch_size, dtype=torch.bool, device=self.device
        )
        if self.training and self.prev_dropout_prob > 0:
            drop_previous = (
                torch.rand(batch_size, device=self.device) < self.prev_dropout_prob
            ) & previous_valid
            prev_motion_feat = prev_motion_feat.masked_fill(
                drop_previous[:, None, None], 0
            )
            prev_audio_feat = prev_audio_feat.masked_fill(
                drop_previous[:, None, None], 0
            )
        self.last_prev_dropout_mask = drop_previous.detach()

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
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, **kwargs):
        # The base sampler already handles None as the first-window condition.
        # Convert an explicitly supplied all-zero pair to that same path so the
        # zero audio context is not changed by emotion modulation.
        if prev_motion_feat is not None and prev_audio_feat is not None:
            is_zero_context = (
                torch.count_nonzero(prev_motion_feat).item() == 0
                and torch.count_nonzero(prev_audio_feat).item() == 0
            )
            if is_zero_context:
                prev_motion_feat = None
                prev_audio_feat = None
        return super().sample(
            audio_or_feat,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            **kwargs,
        )

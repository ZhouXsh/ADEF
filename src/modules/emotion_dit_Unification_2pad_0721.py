import torch
import torch.nn as nn

from .common import enc_dec_mask
from .emotion_dit_timestep_0714 import (
    DiffusionSchedule,
    DenoisingNetwork as BaseDenoisingNetwork,
    DitTalkingHead as BaseDitTalkingHead,
)


class DenoisingNetwork(BaseDenoisingNetwork):
    """Add two fixed reference tokens before the noisy motion tokens."""

    def __init__(self, *args, **kwargs):
        device = kwargs.get("device", "cuda")
        super().__init__(*args, **kwargs)
        self.n_reference_tokens = 2
        self.total_motion_len = self.n_prev_motions + self.n_reference_tokens + self.n_motions

        if self.use_learnable_pe:
            self.PE = nn.Parameter(
                torch.randn(1, self.total_motion_len, self.feature_dim, device=self.device)
            )

        if self.align_mask_width > 0:
            self.alignment_mask = enc_dec_mask(
                self.total_motion_len,
                self.total_motion_len,
                frame_width=1,
                expansion=self.align_mask_width - 1,
                device=device,
            )
        else:
            self.alignment_mask = None

    def _prepare_reference_token(self, token, batch_size, dtype, device, name):
        if token is None:
            return torch.zeros(
                batch_size, 1, self.motion_feat_dim, dtype=dtype, device=device
            )
        if token.ndim == 2:
            token = token.unsqueeze(1)
        if token.ndim != 3 or token.shape[1:] != (1, self.motion_feat_dim):
            raise ValueError(
                f"{name} must have shape (B, 70) or (B, 1, 70), got {tuple(token.shape)}"
            )
        if token.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch size {token.shape[0]} does not match motion batch size {batch_size}"
            )
        return token.detach().to(device=device, dtype=dtype)

    def forward(
        self,
        motion_feat,
        audio_feat,
        prev_motion_feat,
        prev_audio_feat,
        step,
        indicator=None,
        canonical_kp_feat=None,
        first_motion_feat=None,
    ):
        batch_size = motion_feat.shape[0]
        diff_step_embedding = self.diff_step_map(
            self.TE.pe[0, step]
        ).unsqueeze(1)

        canonical_kp_feat = self._prepare_reference_token(
            canonical_kp_feat,
            batch_size,
            motion_feat.dtype,
            motion_feat.device,
            "canonical_kp_feat",
        )
        first_motion_feat = self._prepare_reference_token(
            first_motion_feat,
            batch_size,
            motion_feat.dtype,
            motion_feat.device,
            "first_motion_feat",
        )

        if indicator is not None:
            prefix_len = self.n_prev_motions + self.n_reference_tokens
            indicator = torch.cat(
                [
                    torch.zeros(
                        indicator.shape[0],
                        prefix_len,
                        device=indicator.device,
                        dtype=indicator.dtype,
                    ),
                    indicator,
                ],
                dim=1,
            ).unsqueeze(-1)

        if self.architecture != "decoder":
            raise ValueError(f"Unknown architecture: {self.architecture}")

        feats_in = torch.cat(
            [prev_motion_feat, canonical_kp_feat, first_motion_feat, motion_feat],
            dim=1,
        )
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        feats_in = self.feature_proj(feats_in)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        zero_audio = torch.zeros(
            batch_size,
            self.n_reference_tokens,
            self.feature_dim,
            dtype=audio_feat.dtype,
            device=audio_feat.device,
        )
        audio_feat_in = torch.cat(
            [prev_audio_feat, zero_audio, audio_feat], dim=1
        )

        feat_out = self.transformer(
            feats_in,
            audio_feat_in,
            diff_step_embedding,
            memory_mask=self.alignment_mask,
        )
        return self.motion_dec(feat_out)


class DitTalkingHead(BaseDitTalkingHead):
    """Joint audio-emotion CFG with two fixed reference tokens."""

    def __init__(
        self,
        device="cuda",
        target="sample",
        architecture="decoder",
        motion_feat_dim=70,
        fps=25,
        n_motions=100,
        n_prev_motions=10,
        audio_model="hubert",
        feature_dim=512,
        n_diff_steps=500,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,emotion",
        emo_classes=8,
    ):
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
        self.denoising_net = DenoisingNetwork(
            device=device,
            n_motions=n_motions,
            n_prev_motions=n_prev_motions,
            n_diff_steps=n_diff_steps,
            motion_feat_dim=motion_feat_dim,
            feature_dim=feature_dim,
        )
        self._sample_canonical_kp_feat = None
        self._sample_first_motion_feat = None
        self.to(device)

    def set_reference_priors(self, canonical_kp_feat, first_motion_feat):
        self._sample_canonical_kp_feat = canonical_kp_feat.detach()
        self._sample_first_motion_feat = first_motion_feat.detach()

    def clear_reference_priors(self):
        self._sample_canonical_kp_feat = None
        self._sample_first_motion_feat = None

    def _resolve_sample_references(self, canonical_kp_feat, first_motion_feat):
        if canonical_kp_feat is None:
            canonical_kp_feat = self._sample_canonical_kp_feat
        if first_motion_feat is None:
            first_motion_feat = self._sample_first_motion_feat
        if canonical_kp_feat is None or first_motion_feat is None:
            raise ValueError(
                "Sampling requires canonical_kp_feat and first_motion_feat. "
                "Pass them to sample() or call set_reference_priors() first."
            )
        return canonical_kp_feat.detach(), first_motion_feat.detach()

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
        canonical_kp_feat=None,
        first_motion_feat=None,
    ):
        joint_cfg = (
            "audio" in self.guiding_conditions
            and "emotion" in self.guiding_conditions
        )
        if not joint_cfg:
            return super().forward(
                motion_feat,
                audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                time_step=time_step,
                indicator=indicator,
                emo_index=emo_index,
            )

        batch_size = motion_feat.shape[0]

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(
                16000 * self.n_motions / self.fps
            ), f"Incorrect audio length {audio_or_feat.shape[1]}"
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, (
                f"Incorrect audio feature length {audio_or_feat.shape[1]}"
            )
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")
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

        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat_cond = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

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

        drop_joint_condition = (
            torch.rand(batch_size, device=self.device) < 0.1
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
            canonical_kp_feat=canonical_kp_feat,
            first_motion_feat=first_motion_feat,
        )

        return (
            eps,
            motion_feat_target,
            motion_feat.detach(),
            audio_feat_saved.detach(),
        )

    @torch.no_grad()
    def sample(
        self,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
        cfg_mode=None,
        cfg_cond=None,
        cfg_scale=1.15,
        flexibility=0,
        dynamic_threshold=None,
        ret_traj=False,
        emo_index=None,
        canonical_kp_feat=None,
        first_motion_feat=None,
    ):
        batch_size = audio_or_feat.shape[0]
        canonical_kp_feat, first_motion_feat = self._resolve_sample_references(
            canonical_kp_feat, first_motion_feat
        )

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_mode not in ["incremental", "independent"]:
            raise NotImplementedError(f"Unknown cfg_mode {cfg_mode}")

        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        elif isinstance(cfg_cond, str):
            cfg_cond = cfg_cond.split(",")
        cfg_cond = [c for c in cfg_cond if c in ["audio", "emotion"]]

        use_joint_cfg = (
            len(cfg_cond) > 0
            and "audio" in self.guiding_conditions
            and "emotion" in self.guiding_conditions
        )
        if isinstance(cfg_scale, (list, tuple)):
            joint_cfg_scale = cfg_scale[-1] if len(cfg_scale) > 0 else 1.0
        else:
            joint_cfg_scale = cfg_scale

        print(
            f"cfg_cond: {('audio+emotion',) if use_joint_cfg else ()}, "
            f"cfg_scale: {(joint_cfg_scale,) if use_joint_cfg else ()}"
        )

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(
                16000 * self.n_motions / self.fps
            ), f"Incorrect audio length {audio_or_feat.shape[1]}"
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, (
                f"Incorrect audio feature length {audio_or_feat.shape[1]}"
            )
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")

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
                batch_size,
                self.n_motions,
                self.motion_feat_dim,
                device=self.device,
            )

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
            audio_feat_in = torch.cat(
                [audio_feat_uncond, audio_feat_cond], dim=0
            )
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            n_entries = 1

        prev_motion_feat_in = torch.cat(
            [prev_motion_feat] * n_entries, dim=0
        )
        prev_audio_feat_in = torch.cat(
            [prev_audio_feat] * n_entries, dim=0
        )
        canonical_kp_feat_in = torch.cat(
            [canonical_kp_feat] * n_entries, dim=0
        )
        first_motion_feat_in = torch.cat(
            [first_motion_feat] * n_entries, dim=0
        )
        indicator_in = (
            torch.cat([indicator] * n_entries, dim=0)
            if indicator is not None
            else None
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
                canonical_kp_feat=canonical_kp_feat_in,
                first_motion_feat=first_motion_feat_in,
            )

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions :].reshape(
                    batch_size * n_entries, -1
                ).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)
            if use_joint_cfg:
                uncond_target = results[0][:, -self.n_motions :]
                cond_target = results[1][:, -self.n_motions :]
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = results[0][:, -self.n_motions :]

            if self.target == "noise":
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (
                    motion_at_t - c1 * target_theta
                ) + sigma * z
            elif self.target == "sample":
                c0 = (
                    (1 - alpha_bar_prev)
                    * torch.sqrt(alpha)
                    / (1 - alpha_bar)
                )
                c1 = (
                    (1 - alpha)
                    * torch.sqrt(alpha_bar_prev)
                    / (1 - alpha_bar)
                )
                motion_next = (
                    c0 * motion_at_t + c1 * target_theta + sigma * z
                )
            else:
                raise ValueError(f"Unknown target type: {self.target}")

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat_cond
        return traj[0], motion_at_T, audio_feat_cond


__all__ = ["DiffusionSchedule", "DenoisingNetwork", "DitTalkingHead"]

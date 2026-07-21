import torch
import torch.nn as nn

from .common import enc_dec_mask
from .emotion_dit_Unification import DitTalkingHead as JointDitTalkingHead
from .emotion_dit_timestep_0714 import (
    DiffusionSchedule,
    DenoisingNetwork as BaseDenoisingNetwork,
)


class DenoisingNetwork(BaseDenoisingNetwork):
    """DiT denoiser with one fixed canonical-keypoint token.

    Motion-token order:
        [canonical keypoint, previous motions, noisy current motions]

    Audio-token order:
        [zero audio, previous audio, current audio]
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.get("device", "cuda")
        super().__init__(*args, **kwargs)
        self.n_reference_tokens = 1
        self.total_motion_len = 1 + self.n_prev_motions + self.n_motions
        self._canonical_kp_feat = None

        if self.use_learnable_pe:
            self.PE = nn.Parameter(
                torch.randn(
                    1,
                    self.total_motion_len,
                    self.feature_dim,
                    device=self.device,
                )
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

    def set_canonical_token(self, canonical_kp_feat):
        self._canonical_kp_feat = canonical_kp_feat.detach()

    def clear_canonical_token(self):
        self._canonical_kp_feat = None

    def _prepare_canonical_token(self, token, batch_size, dtype, device):
        if token is None:
            token = self._canonical_kp_feat
        if token is None:
            raise ValueError(
                "canonical_kp_feat is required for the 1pad denoising network"
            )
        if token.ndim == 2:
            token = token.unsqueeze(1)
        expected_shape = (1, self.motion_feat_dim)
        if token.ndim != 3 or token.shape[1:] != expected_shape:
            raise ValueError(
                "canonical_kp_feat must have shape (B, 70) or (B, 1, 70), "
                f"got {tuple(token.shape)}"
            )
        if token.shape[0] != batch_size:
            if batch_size % token.shape[0] != 0:
                raise ValueError(
                    "canonical_kp_feat batch size does not match the denoising "
                    f"batch: {token.shape[0]} versus {batch_size}"
                )
            token = token.repeat(batch_size // token.shape[0], 1, 1)
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
    ):
        batch_size = motion_feat.shape[0]
        canonical_kp_feat = self._prepare_canonical_token(
            canonical_kp_feat,
            batch_size,
            motion_feat.dtype,
            motion_feat.device,
        )

        diff_step_embedding = self.diff_step_map(
            self.TE.pe[0, step]
        ).unsqueeze(1)

        if indicator is not None:
            prefix_len = 1 + self.n_prev_motions
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
            [canonical_kp_feat, prev_motion_feat, motion_feat], dim=1
        )
        if self.use_indicator:
            if indicator is None:
                raise ValueError("indicator is required when use_indicator is enabled")
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        feats_in = self.feature_proj(feats_in)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        zero_audio = torch.zeros(
            batch_size,
            1,
            self.feature_dim,
            dtype=audio_feat.dtype,
            device=audio_feat.device,
        )
        audio_feat_in = torch.cat(
            [zero_audio, prev_audio_feat, audio_feat], dim=1
        )

        feat_out = self.transformer(
            feats_in,
            audio_feat_in,
            diff_step_embedding,
            memory_mask=self.alignment_mask,
        )
        return self.motion_dec(feat_out)


class DitTalkingHead(JointDitTalkingHead):
    """ADEF joint-CFG model with a canonical-keypoint prefix token only."""

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
            architecture=architecture,
            n_motions=n_motions,
            n_prev_motions=n_prev_motions,
            n_diff_steps=n_diff_steps,
            motion_feat_dim=motion_feat_dim,
            feature_dim=feature_dim,
        )
        self._sample_canonical_kp_feat = None
        self.to(device)

    def set_reference_priors(self, canonical_kp_feat):
        self._sample_canonical_kp_feat = canonical_kp_feat.detach()

    def clear_reference_priors(self):
        self._sample_canonical_kp_feat = None
        self.denoising_net.clear_canonical_token()

    def _resolve_canonical_token(self, canonical_kp_feat):
        if canonical_kp_feat is None:
            canonical_kp_feat = self._sample_canonical_kp_feat
        if canonical_kp_feat is None:
            raise ValueError(
                "canonical_kp_feat is required. Pass it explicitly or call "
                "set_reference_priors() before sampling."
            )
        return canonical_kp_feat.detach()

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
    ):
        canonical_kp_feat = self._resolve_canonical_token(canonical_kp_feat)
        self.denoising_net.set_canonical_token(canonical_kp_feat)
        try:
            return super().forward(
                motion_feat,
                audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                time_step=time_step,
                indicator=indicator,
                emo_index=emo_index,
            )
        finally:
            self.denoising_net.clear_canonical_token()

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
    ):
        canonical_kp_feat = self._resolve_canonical_token(canonical_kp_feat)
        self.denoising_net.set_canonical_token(canonical_kp_feat)
        try:
            return super().sample(
                audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                motion_at_T=motion_at_T,
                indicator=indicator,
                cfg_mode=cfg_mode,
                cfg_cond=cfg_cond,
                cfg_scale=cfg_scale,
                flexibility=flexibility,
                dynamic_threshold=dynamic_threshold,
                ret_traj=ret_traj,
                emo_index=emo_index,
            )
        finally:
            self.denoising_net.clear_canonical_token()


__all__ = ["DiffusionSchedule", "DenoisingNetwork", "DitTalkingHead"]

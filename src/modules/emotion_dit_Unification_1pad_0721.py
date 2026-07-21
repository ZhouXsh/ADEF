import torch
import torch.nn as nn

from .common import enc_dec_mask
from .emotion_dit_Unification_2pad_0721 import (
    DiffusionSchedule,
    DenoisingNetwork as TwoPadDenoisingNetwork,
    DitTalkingHead as TwoPadDitTalkingHead,
)


class DenoisingNetwork(TwoPadDenoisingNetwork):
    """Canonical-token-only variant.

    Motion-token order:
        [canonical keypoint, previous motions, noisy current motions]
        1 + n_prev_motions + n_motions

    Audio-token order:
        [zero audio, previous audio, current audio]
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.get("device", "cuda")
        super().__init__(*args, **kwargs)
        self.n_reference_tokens = 1
        self.total_motion_len = 1 + self.n_prev_motions + self.n_motions

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
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        canonical_kp_feat = self._prepare_reference_token(
            canonical_kp_feat,
            batch_size,
            motion_feat.dtype,
            motion_feat.device,
            "canonical_kp_feat",
        )

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


class DitTalkingHead(TwoPadDitTalkingHead):
    """Keep only the canonical-keypoint reference token."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.denoising_net = DenoisingNetwork(
            device=self.device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            n_diff_steps=self.diffusion_sched.num_steps,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=self.feature_dim,
        )
        self._sample_canonical_kp_feat = None
        self._sample_first_motion_feat = None
        self.to(self.device)

    def set_reference_priors(self, canonical_kp_feat, first_motion_feat=None):
        self._sample_canonical_kp_feat = canonical_kp_feat.detach()
        self._sample_first_motion_feat = None

    def clear_reference_priors(self):
        self._sample_canonical_kp_feat = None
        self._sample_first_motion_feat = None

    def _resolve_sample_references(self, canonical_kp_feat, first_motion_feat):
        if canonical_kp_feat is None:
            canonical_kp_feat = self._sample_canonical_kp_feat
        if canonical_kp_feat is None:
            raise ValueError(
                "Sampling requires canonical_kp_feat. Pass it to sample() or "
                "call set_reference_priors() first."
            )
        canonical_kp_feat = canonical_kp_feat.detach()
        dummy_first_motion = torch.zeros_like(canonical_kp_feat)
        return canonical_kp_feat, dummy_first_motion


__all__ = ["DiffusionSchedule", "DenoisingNetwork", "DitTalkingHead"]

# ICASSP27 ablation: emotion_residual. Independent compatibility wrapper.
"""Runtime-corrected 0803 unified talking-head model.

This module keeps the original 0803 implementation in
``emotion_dit_Unification_jianhua0803_legacy`` and adds a thin compatibility
layer that fixes parameter propagation, context-audio extraction, and the
``sample`` return contract without changing the public model interface.
"""

import sys

import torch

from . import emotion_dit_ablation0905_emotion_residual_legacy as _legacy

DiffusionSchedule = _legacy.DiffusionSchedule


class DenoisingNetwork(_legacy.DenoisingNetwork):
    """Legacy denoiser with corrected sequence PE and safe indicator handling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The denoiser input sequence is exactly [prev_motion, current_motion],
        # i.e. n_prev_motions + n_motions tokens. The diffusion-step embedding
        # is injected separately into DiTDecoder and is NOT concatenated as an
        # extra token. The legacy learnable PE allocated one unused extra token
        # (1 + n_prev_motions + n_motions), which causes 80-vs-81 broadcasting
        # failure as soon as learnable PE is actually enabled by the 0901 scripts.
        if self.use_learnable_pe:
            expected_seq_len = self.n_prev_motions + self.n_motions
            if self.PE.shape[1] != expected_seq_len:
                if self.PE.shape[1] < expected_seq_len:
                    raise ValueError(
                        f"Learnable PE is too short: {self.PE.shape[1]} < {expected_seq_len}"
                    )
                self.PE = torch.nn.Parameter(
                    self.PE[:, :expected_seq_len].detach().clone()
                )

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None):
        if self.use_indicator and indicator is None:
            indicator = torch.ones(
                motion_feat.shape[:2],
                device=motion_feat.device,
                dtype=motion_feat.dtype,
            )
        return super().forward(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
            step, indicator=indicator,
        )


DiTDecoderLayer = _legacy.DiTDecoderLayer
DiTDecoder = _legacy.DiTDecoder


def _main_args():
    """Return the argparse namespace of a training entrypoint when available."""
    main_module = sys.modules.get("__main__")
    return getattr(main_module, "args", None) if main_module is not None else None


def _resolve_runtime_arg(explicit_value, name, default):
    """Prefer an explicit constructor value, then CLI args, then a safe default."""
    if explicit_value is not None:
        return explicit_value
    args = _main_args()
    value = getattr(args, name, None) if args is not None else None
    return default if value is None else value


class DitTalkingHead(_legacy.DitTalkingHead):
    """0803 model with consistent train/inference parameter handling.

    ``audio`` and ``emotion`` remain the same joint condition as in the original
    model. The fixes here are intentionally orthogonal to that core method.
    """

    def __init__(
        self,
        device="cuda",
        target="sample",
        architecture="decoder",
        motion_feat_dim=70,
        fps=25,
        n_motions=64,
        n_prev_motions=16,
        audio_model="hubert",
        feature_dim=512,
        n_diff_steps=500,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,emotion",
        emo_classes=8,
        align_mask_width=1,
        n_heads=None,
        n_layers=None,
        mlp_ratio=None,
        use_indicator=None,
        no_use_learnable_pe=None,
    ):
        n_heads = _resolve_runtime_arg(n_heads, "n_heads", 8)
        n_layers = _resolve_runtime_arg(n_layers, "n_layers", 8)
        mlp_ratio = _resolve_runtime_arg(mlp_ratio, "mlp_ratio", 4)
        use_indicator = _resolve_runtime_arg(use_indicator, "use_indicator", True)
        no_use_learnable_pe = _resolve_runtime_arg(
            no_use_learnable_pe, "no_use_learnable_pe", False
        )

        # Build the legacy model first so every pre-existing public attribute and
        # checkpoint key remains stable. Replace only the denoiser with the same
        # class constructed from the parameters that the training CLI exposes.
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
        self.denoising_net = DenoisingNetwork(
            device=device,
            motion_feat_dim=motion_feat_dim,
            use_indicator=use_indicator,
            architecture=architecture,
            feature_dim=feature_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            no_use_learnable_pe=no_use_learnable_pe,
            n_prev_motions=n_prev_motions,
            n_motions=n_motions,
            n_diff_steps=n_diff_steps,
        )

        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.use_indicator = bool(use_indicator)
        self.no_use_learnable_pe = bool(no_use_learnable_pe)
        self._pending_prev_audio_raw = None

        # Training scripts save their argparse Namespace into checkpoints. Mark
        # newly trained checkpoints so inference can distinguish them from older
        # checkpoints whose CLI architecture flags existed but were not applied.
        args = _main_args()
        if args is not None:
            args.n_heads = n_heads
            args.n_layers = n_layers
            args.mlp_ratio = mlp_ratio
            args.use_indicator = bool(use_indicator)
            args.no_use_learnable_pe = bool(no_use_learnable_pe)
            args.model_params_propagated = True
            args.context_audio_encoded_once = True
            args.seed = 2026

    def extract_audio_feature(self, audio, frame_num=None):
        """Extract audio features, deferring training-history encoding when possible.

        The continuation branch in the existing training scripts first asks for
        the previous 16-frame feature and then forwards the current 64-frame raw
        audio. During training we cache that previous raw waveform and let
        ``forward`` encode the full 80-frame waveform once, then split features by
        frame index. Evaluation/inference calls retain the normal behavior.
        """
        expected_prev_samples = round(16000 * self.n_prev_motions / self.fps)
        should_defer = (
            self.training
            and audio.ndim == 2
            and frame_num == self.n_prev_motions
            and audio.shape[1] == expected_prev_samples
        )
        if should_defer:
            self._pending_prev_audio_raw = audio
            return torch.zeros(
                audio.shape[0],
                self.n_prev_motions,
                self.feature_dim,
                device=audio.device,
                dtype=audio.dtype,
            )
        return super().extract_audio_feature(audio, frame_num=frame_num)

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
        # Continuation training: encode [prev 16 + current 64] together once so
        # both slices are produced with identical Wav2Vec2/HuBERT context.
        if (
            self.training
            and audio_or_feat.ndim == 2
            and prev_audio_feat is not None
            and self._pending_prev_audio_raw is not None
        ):
            prev_audio_raw = self._pending_prev_audio_raw
            self._pending_prev_audio_raw = None
            if prev_audio_raw.shape[0] != audio_or_feat.shape[0]:
                raise ValueError("Previous/current audio batch sizes do not match.")
            full_audio = torch.cat([prev_audio_raw, audio_or_feat], dim=1)
            expected_samples = round(
                16000 * (self.n_prev_motions + self.n_motions) / self.fps
            )
            if full_audio.shape[1] != expected_samples:
                raise ValueError(
                    f"Incorrect context audio length {full_audio.shape[1]}, "
                    f"expected {expected_samples}."
                )
            full_audio_feat = super().extract_audio_feature(
                full_audio,
                frame_num=self.n_prev_motions + self.n_motions,
            )
            prev_audio_feat = full_audio_feat[:, : self.n_prev_motions].detach()
            audio_or_feat = full_audio_feat[:, self.n_prev_motions :]

        return super().forward(
            motion_feat,
            audio_or_feat,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            time_step=time_step,
            indicator=indicator,
            emo_index=emo_index,
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
    ):
        # Pre-extract raw acoustic features exactly once. Passing a 3-D tensor to
        # the legacy sampler prevents it from encoding the waveform a second time.
        if audio_or_feat.ndim == 2:
            audio_feat_saved = super().extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            if audio_or_feat.shape[1] != self.n_motions:
                raise ValueError(
                    f"Incorrect audio feature length {audio_or_feat.shape[1]}"
                )
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")

        result = super().sample(
            audio_feat_saved,
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
        if ret_traj:
            traj, noise, _ = result
            return traj, noise, audio_feat_saved
        motion_feat, noise, _ = result
        return motion_feat, noise, audio_feat_saved


__all__ = [
    "DiffusionSchedule",
    "DenoisingNetwork",
    "DiTDecoderLayer",
    "DiTDecoder",
    "DitTalkingHead",
]

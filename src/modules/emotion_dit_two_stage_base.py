"""Shared two-stage dual-audio diffusion model infrastructure.

Stage 1 trains an emotion-agnostic audio-conditioned motion base. Stage 2
freezes that base and trains only an isolated emotion-audio encoder and
zero-initialized emotion residual cross-attention adapters. Variant files supply
their own emotion-audio encoder through ``_create_emotion_audio_encoder``.

The shared base is deliberate: it guarantees identical parameter names and
freezing semantics across finalv1/finalv2/finalv3, so Stage-1 checkpoints are
comparable and can be transferred with ``strict=False``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import platform
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import pad_audio
from .emotion_dit import DiffusionSchedule
from .emotion_dit_two_stage_layers import TwoStageDualAudioDenoisingNetwork
from .emotion_dit_two_stage_sampling import TwoStageSamplingMixin
from .emotion_dit_two_stage_forward import TwoStageForwardMixin
from .emotion_dit_two_stage_training import TwoStageTrainingMixin
from ..config.base_config import make_abs_path


class TwoStageDitTalkingHead(
    TwoStageTrainingMixin,
    TwoStageForwardMixin,
    TwoStageSamplingMixin,
    nn.Module,
):
    """Two-stage dual-audio diffusion motion generator."""

    def __init__(
        self,
        device: str = "cuda",
        target: str = "sample",
        architecture: str = "decoder",
        motion_feat_dim: int = 70,
        fps: int = 25,
        n_motions: int = 100,
        n_prev_motions: int = 25,
        audio_model: str = "hubert",
        feature_dim: int = 512,
        n_diff_steps: int = 50,
        diff_schedule: str = "cosine",
        cfg_mode: str = "incremental",
        guiding_conditions: str = "audio,emotion",
        emo_classes: int = 8,
        e2v_dim: int = 1024,
        num_emotion_tokens: int = 8,
        n_layers: int = 8,
        n_heads: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 3,
        decoder_dropout: float = 0.0,
        audio_scale: float = 1.0,
        emotion_scale_init: float = 0.10,
        emotion_audio_residual_init: float = 0.05,
        use_indicator: bool = False,
        use_learnable_pe: bool = False,
    ):
        super().__init__()
        if architecture != "decoder":
            raise ValueError(f"Unknown architecture: {architecture}")

        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.audio_model = audio_model
        self.cfg_mode = cfg_mode
        conditions = guiding_conditions.split(",") if guiding_conditions else []
        self.guiding_conditions = [
            condition for condition in conditions
            if condition in {"audio", "emotion"}
        ]

        if audio_model == "wav2vec2":
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                make_abs_path(
                    "../../pretrained_weights/wav2vec2-base-960h"
                )
            )
        elif audio_model == "hubert":
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                make_abs_path(
                    "../../pretrained_weights/hubert-base-ls960"
                )
            )
        elif audio_model in {"hubert_zh", "hubert_zh_ori"}:
            model_path = (
                "../../pretrained_weights/TencentGameMate:chinese-hubert-base"
            )
            if platform.system() == "Windows":
                model_path = "../../pretrained_weights/chinese-hubert-base"
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                make_abs_path(model_path)
            )
        else:
            raise ValueError(f"Unknown audio model: {audio_model}")

        if hasattr(self.audio_encoder, "feature_extractor"):
            self.audio_encoder.feature_extractor._freeze_parameters()

        self.audio_feature_map = nn.Linear(768, feature_dim)
        # Generic start states are deliberately not indexed by an emotion label.
        self.start_audio_feat = nn.Parameter(
            torch.randn(1, n_prev_motions, feature_dim) * 0.02
        )
        self.start_motion_feat = nn.Parameter(
            torch.randn(1, n_prev_motions, motion_feat_dim) * 0.02
        )
        self.null_audio_feat = nn.Parameter(
            torch.randn(1, 1, feature_dim) * 0.02
        )
        self.audio_norm = nn.LayerNorm(feature_dim, eps=1e-9)

        self.emotion_audio_encoder = self._create_emotion_audio_encoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            e2v_dim=e2v_dim,
            num_emotion_tokens=num_emotion_tokens,
            n_heads=n_heads,
            residual_init=emotion_audio_residual_init,
        )

        self.denoising_net = TwoStageDualAudioDenoisingNetwork(
            device=device,
            motion_feat_dim=motion_feat_dim,
            feature_dim=feature_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            use_indicator=use_indicator,
            use_learnable_pe=use_learnable_pe,
            n_prev_motions=n_prev_motions,
            n_motions=n_motions,
            n_diff_steps=n_diff_steps,
            dropout=decoder_dropout,
            audio_scale=audio_scale,
            emotion_scale_init=emotion_scale_init,
        )
        self.diffusion_sched = DiffusionSchedule(
            n_diff_steps, diff_schedule
        )

        self.train_stage = 1
        self._audio_encoder_trainable = False
        self.to(device)
        self.set_train_stage(1, train_audio_encoder=False)

    def _create_emotion_audio_encoder(
        self,
        feature_dim: int,
        emo_classes: int,
        e2v_dim: int,
        num_emotion_tokens: int,
        n_heads: int,
        residual_init: float,
    ) -> nn.Module:
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def _requires_grad(module: nn.Module, enabled: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def extract_audio_feature(
        self,
        audio: torch.Tensor,
        frame_num: Optional[int] = None,
    ) -> torch.Tensor:
        frame_num = frame_num or self.n_motions
        hidden = self.audio_encoder(
            pad_audio(audio),
            self.fps,
            frame_num=frame_num * 2,
        ).last_hidden_state
        hidden = F.interpolate(
            hidden.transpose(1, 2),
            size=frame_num,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return self.audio_feature_map(hidden)

    def _get_audio_feature(self, audio_or_feat: torch.Tensor) -> torch.Tensor:
        if audio_or_feat.ndim == 2:
            expected = round(16000 * self.n_motions / self.fps)
            if audio_or_feat.shape[1] != expected:
                raise ValueError(
                    f"Incorrect audio length {audio_or_feat.shape[1]}, "
                    f"expected {expected}"
                )
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            if audio_or_feat.shape[1] != self.n_motions:
                raise ValueError(
                    f"Incorrect audio feature length "
                    f"{audio_or_feat.shape[1]}"
                )
            return audio_or_feat
        raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")

    def _init_previous(
        self,
        batch_size: int,
        prev_motion_feat: Optional[torch.Tensor],
        prev_audio_feat: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if prev_motion_feat is None:
            prev_motion_feat = self.start_motion_feat.expand(
                batch_size, -1, -1
            )
        if prev_audio_feat is None:
            prev_audio_feat = self.start_audio_feat.expand(
                batch_size, -1, -1
            )
        return prev_motion_feat, prev_audio_feat

    def _build_emotion_audio(
        self,
        audio_feat: torch.Tensor,
        emo_index: torch.Tensor,
        emo_utt_feat: Optional[torch.Tensor],
        emo_frame_feat: Optional[torch.Tensor],
        drop_emotion: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return self.emotion_audio_encoder(
            audio_feat=self.audio_norm(audio_feat),
            emo_index=emo_index,
            emo_utt_feat=emo_utt_feat,
            emo_frame_feat=emo_frame_feat,
            drop_emotion=drop_emotion,
        )

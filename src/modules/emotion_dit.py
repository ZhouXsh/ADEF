from __future__ import annotations

import math
import platform
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import pad_audio
from .emotion_dit_network import DenoisingNetwork
from .emotion_dit_sampling import EmotionSamplingMixin
from .emotion_dit_schedule import DiffusionSchedule
from ..config.base_config import make_abs_path


class DitTalkingHead(EmotionSamplingMixin, nn.Module):
    GENERAL_STAGE = "general"
    EMOTION_STAGE = "emotion"

    def __init__(
        self,
        device: str | torch.device = "cuda",
        target: str = "sample",
        architecture: str = "decoder",
        motion_feat_dim: int = 70,
        fps: int = 25,
        n_motions: int = 100,
        n_prev_motions: int = 10,
        audio_model: str = "hubert",
        feature_dim: int = 512,
        n_diff_steps: int = 500,
        diff_schedule: str = "cosine",
        cfg_mode: str = "incremental",
        guiding_conditions: str = "audio,emotion",
        emo_classes: int = 8,
        n_heads: int = 8,
        n_layers: int = 8,
        mlp_ratio: int = 4,
        align_mask_width: int = 1,
        no_use_learnable_pe: bool = True,
        use_indicator: bool = False,
        training_stage: str = EMOTION_STAGE,
        general_audio_dropout: float = 0.1,
        emotion_dropout: float = 0.5,
        emotion_gate_init: float = 0.1,
    ) -> None:
        super().__init__()
        if target not in {"sample", "noise"}:
            raise ValueError(f"Unknown target type: {target}")
        if architecture != "decoder":
            raise ValueError(f"Unknown architecture: {architecture}")
        if training_stage not in {self.GENERAL_STAGE, self.EMOTION_STAGE}:
            raise ValueError(f"Unknown training stage: {training_stage}")
        if not 0.0 <= general_audio_dropout < 1.0:
            raise ValueError("general_audio_dropout must be in [0, 1)")
        if not 0.0 <= emotion_dropout < 1.0:
            raise ValueError("emotion_dropout must be in [0, 1)")
        if not 0.0 <= emotion_gate_init < 1.0:
            raise ValueError("emotion_gate_init must be in [0, 1)")

        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = int(motion_feat_dim)
        self.fps = int(fps)
        self.n_motions = int(n_motions)
        self.n_prev_motions = int(n_prev_motions)
        self.feature_dim = int(feature_dim)
        self.emo_classes = int(emo_classes)
        self.training_stage = training_stage
        self.general_audio_dropout = float(general_audio_dropout)
        self.emotion_dropout = float(emotion_dropout)

        self.audio_model = audio_model
        self.audio_encoder = self._build_audio_encoder(audio_model)
        self.audio_feature_map = nn.Linear(768, feature_dim)

        shared_audio_start = torch.randn(1, n_prev_motions, feature_dim) * 0.02
        shared_motion_start = torch.randn(1, n_prev_motions, motion_feat_dim) * 0.02
        self.start_audio_feat = nn.Parameter(
            shared_audio_start.expand(emo_classes, -1, -1).clone()
        )
        self.start_motion_feat = nn.Parameter(
            shared_motion_start.expand(emo_classes, -1, -1).clone()
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
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        self.cfg_mode = cfg_mode
        requested_conditions = guiding_conditions.split(",") if guiding_conditions else []
        self.guiding_conditions = [
            condition
            for condition in requested_conditions
            if condition in {"audio", "emotion"}
        ]

        self.null_audio_feat = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.audio_norm = nn.LayerNorm(feature_dim, eps=1e-6)
        self.null_emotion_feat = nn.Parameter(
            torch.zeros(1, 1, feature_dim), requires_grad=False
        )
        self.emo_embed = nn.Embedding(emo_classes, feature_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(feature_dim, 2 * feature_dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        gate_value = math.atanh(max(emotion_gate_init, 1e-6))
        self.emotion_gate = nn.Parameter(torch.full((feature_dim,), gate_value))

        self.to(device)

    def _build_audio_encoder(self, audio_model: str) -> nn.Module:
        if audio_model == "wav2vec2":
            from .wav2vec2 import Wav2Vec2Model

            encoder = Wav2Vec2Model.from_pretrained(
                make_abs_path("../../pretrained_weights/wav2vec2-base-960h")
            )
        elif audio_model == "hubert":
            from .hubert import HubertModel

            encoder = HubertModel.from_pretrained(
                make_abs_path("../../pretrained_weights/hubert-base-ls960")
            )
        elif audio_model in {"hubert_zh", "hubert_zh_ori"}:
            from .hubert import HubertModel

            model_path = "../../pretrained_weights/TencentGameMate:chinese-hubert-base"
            if platform.system() == "Windows":
                model_path = "../../pretrained_weights/chinese-hubert-base"
            encoder = HubertModel.from_pretrained(make_abs_path(model_path))
        else:
            raise ValueError(f"Unknown audio model: {audio_model}")

        encoder.feature_extractor._freeze_parameters()
        return encoder

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def emotion_parameter_names(self) -> tuple[str, ...]:
        return (
            "emo_embed",
            "adaLN_modulation",
            "emotion_gate",
            "start_audio_feat",
            "start_motion_feat",
        )

    def configure_training_stage(
        self,
        stage: str,
        train_audio_encoder: bool = False,
        stage2_unfreeze_motion_decoder: bool = False,
    ) -> list[str]:
        if stage not in {self.GENERAL_STAGE, self.EMOTION_STAGE}:
            raise ValueError(f"Unknown training stage: {stage}")
        self.training_stage = stage

        for parameter in self.parameters():
            parameter.requires_grad = stage == self.GENERAL_STAGE

        if stage == self.GENERAL_STAGE:
            for module in (self.emo_embed, self.adaLN_modulation):
                for parameter in module.parameters():
                    parameter.requires_grad = False
            self.emotion_gate.requires_grad = False
            self.start_audio_feat.requires_grad = True
            self.start_motion_feat.requires_grad = True
        else:
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module in (self.emo_embed, self.adaLN_modulation):
                for parameter in module.parameters():
                    parameter.requires_grad = True
            self.emotion_gate.requires_grad = True
            self.start_audio_feat.requires_grad = True
            self.start_motion_feat.requires_grad = True
            if stage2_unfreeze_motion_decoder:
                for parameter in self.denoising_net.motion_dec.parameters():
                    parameter.requires_grad = True

        self.null_emotion_feat.requires_grad = False
        if not train_audio_encoder:
            for parameter in self.audio_encoder.parameters():
                parameter.requires_grad = False

        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def _validate_emotion_index(self, emo_index: Optional[torch.Tensor]) -> torch.Tensor:
        if emo_index is None:
            raise ValueError("emo_index is required when emotion conditioning is enabled")
        emo_index = emo_index.to(device=self.device, dtype=torch.long).view(-1)
        if torch.any(emo_index < 0) or torch.any(emo_index >= self.emo_classes):
            raise ValueError(
                f"Emotion indices must be in [0, {self.emo_classes - 1}]"
            )
        return emo_index

    def _start_feature(
        self,
        parameter: torch.Tensor,
        batch_size: int,
        emo_index: Optional[torch.Tensor],
        use_emotion: bool,
    ) -> torch.Tensor:
        if use_emotion:
            index = self._validate_emotion_index(emo_index)
            return torch.index_select(parameter, 0, index)
        return parameter.mean(dim=0, keepdim=True).expand(batch_size, -1, -1)

    def _audio_content(self, audio_feat: torch.Tensor) -> torch.Tensor:
        return self.audio_norm(audio_feat)

    def _modulate_audio(
        self,
        audio_feat: torch.Tensor,
        emo_index: Optional[torch.Tensor],
        use_emotion: bool,
        drop_emotion: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        content = self._audio_content(audio_feat)
        if not use_emotion:
            return content

        index = self._validate_emotion_index(emo_index)
        emotion = self.emo_embed(index).unsqueeze(1)
        if drop_emotion is not None:
            null_emotion = self.null_emotion_feat.expand(index.shape[0], -1, -1)
            emotion = torch.where(drop_emotion.view(-1, 1, 1), null_emotion, emotion)

        shift, scale = self.adaLN_modulation(emotion).chunk(2, dim=-1)
        gate = torch.tanh(self.emotion_gate).view(1, 1, -1)
        return content + gate * (content * scale + shift)

    def _replace_with_null_audio(
        self, audio_feat: torch.Tensor, drop_audio: torch.Tensor
    ) -> torch.Tensor:
        null_audio = self.null_audio_feat.expand(
            audio_feat.shape[0], audio_feat.shape[1], -1
        )
        return torch.where(drop_audio.view(-1, 1, 1), null_audio, audio_feat)

    def _use_emotion(
        self, emo_index: Optional[torch.Tensor], use_emotion: Optional[bool]
    ) -> bool:
        if use_emotion is not None:
            enabled = bool(use_emotion)
        else:
            enabled = self.training_stage == self.EMOTION_STAGE
        return enabled and "emotion" in self.guiding_conditions and emo_index is not None

    def extract_audio_feature(
        self, audio: torch.Tensor, frame_num: Optional[int] = None
    ) -> torch.Tensor:
        frame_num = int(frame_num or self.n_motions)
        hidden_states = self.audio_encoder(
            pad_audio(audio), self.fps, frame_num=frame_num * 2
        ).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(
            hidden_states, size=frame_num, align_corners=False, mode="linear"
        )
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)

    def _extract_or_validate_audio(
        self, audio_or_feat: torch.Tensor
    ) -> torch.Tensor:
        if audio_or_feat.ndim == 2:
            expected = round(16000 * self.n_motions / self.fps)
            if audio_or_feat.shape[1] != expected:
                raise ValueError(
                    f"Incorrect audio length {audio_or_feat.shape[1]}, expected {expected}"
                )
            return self.extract_audio_feature(audio_or_feat)
        if audio_or_feat.ndim == 3:
            if audio_or_feat.shape[1] != self.n_motions:
                raise ValueError(
                    f"Incorrect audio feature length {audio_or_feat.shape[1]}"
                )
            return audio_or_feat
        raise ValueError(f"Incorrect audio input shape: {tuple(audio_or_feat.shape)}")

    def forward(
        self,
        motion_feat: torch.Tensor,
        audio_or_feat: torch.Tensor,
        prev_motion_feat: Optional[torch.Tensor] = None,
        prev_audio_feat: Optional[torch.Tensor] = None,
        time_step: Optional[torch.Tensor | Sequence[int]] = None,
        indicator: Optional[torch.Tensor] = None,
        emo_index: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        use_emotion: Optional[bool] = None,
        apply_condition_dropout: bool = True,
    ):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._extract_or_validate_audio(audio_or_feat)
        emotion_enabled = self._use_emotion(emo_index, use_emotion)

        if prev_motion_feat is None:
            prev_motion_feat = self._start_feature(
                self.start_motion_feat,
                batch_size,
                emo_index,
                emotion_enabled,
            )
        if prev_audio_feat is None:
            prev_audio_feat = self._start_feature(
                self.start_audio_feat,
                batch_size,
                emo_index,
                emotion_enabled,
            )

        drop_audio = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        drop_emotion = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if self.training and apply_condition_dropout:
            if emotion_enabled:
                drop_emotion = (
                    torch.rand(batch_size, device=self.device) < self.emotion_dropout
                )
            elif "audio" in self.guiding_conditions:
                drop_audio = (
                    torch.rand(batch_size, device=self.device)
                    < self.general_audio_dropout
                )

        current_audio = self._replace_with_null_audio(audio_feat_saved, drop_audio)
        previous_audio = self._replace_with_null_audio(prev_audio_feat, drop_audio)
        current_audio = self._modulate_audio(
            current_audio, emo_index, emotion_enabled, drop_emotion
        )
        previous_audio = self._modulate_audio(
            previous_audio, emo_index, emotion_enabled, drop_emotion
        )

        if time_step is None:
            time_step_tensor = self.diffusion_sched.uniform_sample_t(
                batch_size, self.device
            )
        else:
            time_step_tensor = torch.as_tensor(
                time_step, device=self.device, dtype=torch.long
            ).view(-1)
            if time_step_tensor.shape[0] != batch_size:
                raise ValueError("time_step batch size does not match motion batch")

        alpha_bar = self.diffusion_sched.alpha_bars[time_step_tensor]
        clean_weight = torch.sqrt(alpha_bar).view(-1, 1, 1)
        noise_weight = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat) if noise is None else noise
        if eps.shape != motion_feat.shape:
            raise ValueError("noise shape must match motion_feat")
        noisy_motion = clean_weight * motion_feat + noise_weight * eps

        target = self.denoising_net(
            noisy_motion,
            current_audio,
            prev_motion_feat,
            previous_audio,
            time_step_tensor,
            indicator,
        )
        return eps, target, motion_feat.detach(), audio_feat_saved.detach()


__all__ = ["DiffusionSchedule", "DenoisingNetwork", "DitTalkingHead"]

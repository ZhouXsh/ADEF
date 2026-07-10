"""Freezing and optimizer mixin for two-stage dual-audio DiT."""

from __future__ import annotations

from typing import Dict, List

import torch.nn as nn


class TwoStageTrainingMixin:
    def _set_audio_encoder_trainable(self, enabled: bool) -> None:
        self._requires_grad(self.audio_encoder, enabled)
        if hasattr(self.audio_encoder, "feature_extractor"):
            self.audio_encoder.feature_extractor._freeze_parameters()
        self._audio_encoder_trainable = bool(enabled)

    def set_train_stage(
        self,
        stage: int,
        train_audio_encoder: bool = False,
        stage2_tune_tail_layers: int = 0,
        stage2_tune_motion_head: bool = False,
    ) -> Dict[str, int]:
        """Configure the exact trainable partition for Stage 1 or Stage 2."""
        if stage not in {1, 2}:
            raise ValueError(f"stage must be 1 or 2, got {stage}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        self.train_stage = stage
        if stage == 1:
            self._set_audio_encoder_trainable(train_audio_encoder)
            self._requires_grad(self.audio_feature_map, True)
            self.start_audio_feat.requires_grad_(True)
            self.start_motion_feat.requires_grad_(True)
            self.null_audio_feat.requires_grad_(True)
            self._requires_grad(self.audio_norm, True)
            self.denoising_net.enable_stage1_base()
        else:
            self._set_audio_encoder_trainable(False)
            self._requires_grad(self.emotion_audio_encoder, True)
            self.denoising_net.enable_stage2_emotion(
                tune_tail_layers=stage2_tune_tail_layers,
                tune_motion_head=stage2_tune_motion_head,
            )

        self.enforce_stage_mode()
        return self.trainable_parameter_report()

    def enforce_stage_mode(self) -> None:
        if not self._audio_encoder_trainable:
            self.audio_encoder.eval()
        if self.train_stage == 2:
            self.audio_feature_map.eval()
            self.audio_norm.eval()
            self.denoising_net.time_encoding.eval()
            if isinstance(self.denoising_net.position, nn.Module):
                self.denoising_net.position.eval()
        self.denoising_net.enforce_stage_mode(self.train_stage)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self.enforce_stage_mode()
        return self

    def trainable_parameter_report(self) -> Dict[str, int]:
        emotion_count = 0
        base_count = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "emotion" in name:
                emotion_count += parameter.numel()
            else:
                base_count += parameter.numel()
        return {
            "stage": self.train_stage,
            "base_trainable": base_count,
            "emotion_trainable": emotion_count,
            "total_trainable": base_count + emotion_count,
        }

    def optimizer_parameter_groups(
        self,
        learning_rate: float,
        tail_lr_ratio: float = 0.1,
    ) -> List[Dict]:
        emotion_parameters = []
        tail_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "emotion" in name:
                emotion_parameters.append(parameter)
            else:
                tail_parameters.append(parameter)

        groups = []
        if emotion_parameters:
            groups.append({
                "params": emotion_parameters,
                "lr": learning_rate,
                "name": "emotion_branch",
            })
        if tail_parameters:
            groups.append({
                "params": tail_parameters,
                "lr": learning_rate * tail_lr_ratio,
                "name": "shared_tail",
            })
        return groups

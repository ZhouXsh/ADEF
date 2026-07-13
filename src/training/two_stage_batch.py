from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import src.utils as utils
from src.modules.emotion_dit import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier


def temporal_high_pass(sequence: torch.Tensor, kernel_size: int) -> torch.Tensor:
    smoothed = F.avg_pool1d(
        sequence.transpose(1, 2),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).transpose(1, 2)
    return sequence - smoothed


def zeros(device: torch.device) -> torch.Tensor:
    return torch.zeros((), device=device)


def add_optional(total: torch.Tensor, value, weight: float) -> torch.Tensor:
    if value is None or weight == 0:
        return total
    return total + weight * value


def prepare_indicator(
    args: argparse.Namespace,
    batch_size: int,
    end_idx: Optional[torch.Tensor],
    device: torch.device,
) -> Optional[torch.Tensor]:
    if not args.use_indicator:
        return None
    if end_idx is None:
        return torch.ones(batch_size, args.n_motions, device=device)
    return (
        torch.arange(args.n_motions, device=device).expand(batch_size, -1)
        < end_idx.unsqueeze(1)
    ).float()


def current_prediction(target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    return target[:, -args.n_motions :, :63]


def run_batch(
    args: argparse.Namespace,
    model: DitTalkingHead,
    batch,
    classifier: Optional[Classifier],
    training: bool,
) -> dict[str, torch.Tensor]:
    audio_pair, coef_pair, emotion_index, _ = batch
    device = model.device
    audio_pair = [audio.to(device, non_blocking=True) for audio in audio_pair]
    coef_pair = [
        {key: value.to(device, non_blocking=True) for key, value in item.items()}
        for item in coef_pair
    ]
    motion_pair = [
        utils.get_motion_coef(item, args.rot_repr, not args.no_head_pose)
        for item in coef_pair
    ]
    emotion_index = emotion_index.to(device, dtype=torch.long, non_blocking=True)
    if args.stage == "general":
        emotion_index = None

    context_audio = None
    if args.use_context_audio_feat:
        context_audio = model.extract_audio_feature(
            torch.cat(audio_pair, dim=1), args.n_motions * 2
        )

    totals = {
        "noise": zeros(device),
        "exp": zeros(device),
        "exp_vel": zeros(device),
        "exp_smooth": zeros(device),
        "head_angle": zeros(device),
        "head_vel": zeros(device),
        "head_smooth": zeros(device),
        "head_trans": zeros(device),
        "emotion": zeros(device),
        "sync_highfreq": zeros(device),
        "general_anchor": zeros(device),
    }
    previous_motion = None
    previous_audio = None
    audio_unit = 16000.0 / args.fps

    for window_index in range(2):
        audio = audio_pair[window_index]
        motion = motion_pair[window_index]
        batch_size = audio.shape[0]
        should_truncate = training and (
            (window_index == 0 and np.random.rand() < args.trunc_prob1)
            or (window_index == 1 and np.random.rand() < args.trunc_prob2)
        )
        if should_truncate:
            audio_in, motion_in, end_idx = utils.truncate_motion_coef_and_audio(
                audio,
                motion,
                args.n_motions,
                audio_unit,
                args.pad_mode,
            )
            if args.use_context_audio_feat and window_index == 1:
                audio_in = model.extract_audio_feature(
                    torch.cat([audio_pair[0], audio_in], dim=1),
                    args.n_motions * 2,
                )[:, -args.n_motions :]
        else:
            motion_in = motion
            end_idx = None
            if args.use_context_audio_feat:
                audio_in = context_audio[
                    :, window_index * args.n_motions : (window_index + 1) * args.n_motions
                ]
            else:
                audio_in = audio

        indicator = prepare_indicator(args, batch_size, end_idx, device)
        time_step = model.diffusion_sched.uniform_sample_t(batch_size, device)
        noise = torch.randn_like(motion_in)

        eps, target, returned_motion, returned_audio = model(
            motion_in,
            audio_in,
            prev_motion_feat=previous_motion,
            prev_audio_feat=previous_audio,
            time_step=time_step,
            indicator=indicator,
            emo_index=emotion_index,
            noise=noise,
            use_emotion=args.stage == "emotion",
            apply_condition_dropout=training,
        )

        generic_target = None
        if args.stage == "emotion":
            with torch.no_grad():
                _, generic_target, _, _ = model(
                    motion_in,
                    audio_in,
                    prev_motion_feat=previous_motion,
                    prev_audio_feat=previous_audio,
                    time_step=time_step,
                    indicator=indicator,
                    emo_index=None,
                    noise=noise,
                    use_emotion=False,
                    apply_condition_dropout=False,
                )

        losses = utils.compute_loss_new(
            args,
            window_index == 0,
            motion_in,
            eps,
            target,
            previous_motion,
            end_idx,
        )
        names = (
            "noise",
            "exp",
            "exp_vel",
            "exp_smooth",
            "head_angle",
            "head_vel",
            "head_smooth",
            "head_trans",
        )
        for name, value in zip(names, losses):
            if value is not None:
                factor = 1.0 if name == "head_trans" else 0.5
                totals[name] = totals[name] + factor * value

        if args.target == "sample":
            predicted_expression = current_prediction(target, args)
            if args.stage == "general":
                ground_truth_expression = motion_in[:, :, :63]
                totals["sync_highfreq"] = totals["sync_highfreq"] + 0.5 * F.l1_loss(
                    temporal_high_pass(predicted_expression, args.highfreq_kernel),
                    temporal_high_pass(ground_truth_expression, args.highfreq_kernel),
                )
            else:
                generic_expression = current_prediction(generic_target, args)
                totals["general_anchor"] = totals["general_anchor"] + 0.5 * F.l1_loss(
                    temporal_high_pass(predicted_expression, args.highfreq_kernel),
                    temporal_high_pass(generic_expression, args.highfreq_kernel),
                )
                logits, _ = classifier(predicted_expression)
                totals["emotion"] = totals["emotion"] + 0.5 * F.cross_entropy(
                    logits, emotion_index
                )

        if window_index == 0:
            if end_idx is not None:
                previous_motion = motion[:, -args.n_prev_motions :].detach()
                if args.use_context_audio_feat:
                    previous_audio = context_audio[
                        :,
                        args.n_motions - args.n_prev_motions : args.n_motions,
                    ].detach()
                else:
                    with torch.no_grad():
                        previous_audio = model.extract_audio_feature(audio)[
                            :, -args.n_prev_motions :
                        ].detach()
            else:
                previous_motion = returned_motion[:, -args.n_prev_motions :]
                previous_audio = returned_audio[:, -args.n_prev_motions :]

    total_loss = totals["noise"]
    total_loss = add_optional(total_loss, totals["exp"], args.l_exp)
    total_loss = add_optional(total_loss, totals["exp_vel"], args.l_exp_vel)
    total_loss = add_optional(total_loss, totals["exp_smooth"], args.l_exp_smooth)
    total_loss = add_optional(total_loss, totals["head_angle"], args.l_head_angle)
    total_loss = add_optional(total_loss, totals["head_vel"], args.l_head_vel)
    total_loss = add_optional(total_loss, totals["head_smooth"], args.l_head_smooth)
    total_loss = add_optional(total_loss, totals["head_trans"], args.l_head_trans)
    if args.stage == "general":
        total_loss = total_loss + args.l_sync_highfreq * totals["sync_highfreq"]
    else:
        total_loss = total_loss + args.l_emotion * totals["emotion"]
        total_loss = total_loss + args.l_general_anchor * totals["general_anchor"]
    totals["loss"] = total_loss
    return totals


def scalar_dict(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {name: float(value.detach().item()) for name, value in losses.items()}


def format_losses(prefix: str, step: int, values: dict[str, float]) -> str:
    fields = [f"loss={values['loss']:.4e}", f"noise={values['noise']:.4e}"]
    for name in (
        "exp",
        "exp_vel",
        "exp_smooth",
        "head_angle",
        "head_vel",
        "head_smooth",
        "head_trans",
        "sync_highfreq",
        "emotion",
        "general_anchor",
    ):
        if abs(values.get(name, 0.0)) > 0:
            fields.append(f"{name}={values[name]:.4e}")
    return f"{prefix} step={step}: " + ", ".join(fields)

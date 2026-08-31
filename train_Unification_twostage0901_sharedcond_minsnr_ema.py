# NOTE: This file is a complete, independent copy-style training entrypoint.
# It does not modify train_Unification_twostage0819_opt.py.
#
# Performance-oriented two-stage training:
#   Phase 1: generic talking motion
#   Phase 2: short MEAD emotion/start warm-up
#   Phase 3: full MEAD fine-tuning

import argparse
import copy
from collections import Counter, defaultdict, deque
from pathlib import Path

import logging
import math
import os
import random
import sys

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torch.utils.data import WeightedRandomSampler

from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils
from src.dataset.dataset_GenericTalkingMotion_clear_start0901 import (
    GenericTalkingMotionDataset,
)
from src.dataset.dataset_EmotionLevel_clear_jianhua0803 import EmoLevelDataset
from src.modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead
from src.seed import make_generator, seed_worker, set_global_seed


# -----------------------------------------------------------------------------
# Variant defaults. Every generated file is a physical, self-contained copy;
# only this block changes between variants so experiments remain easy to read.
# -----------------------------------------------------------------------------
VARIANT_NAME = "sharedcond_minsnr_ema"
VARIANT_DESCRIPTION = (
    "Shared-condition warm-start plus stratified diffusion timesteps, Min-SNR weighting, and EMA."
)
DEFAULT_EXP_NAME = "20260901_twostage_sharedcond_minsnr_ema"
DEFAULT_SHARED_CONDITION_WARMSTART = True
DEFAULT_USE_MIN_SNR = True
DEFAULT_BALANCE_MEAD = False
DEFAULT_GENERIC_REPLAY_INTERVAL = 0
DEFAULT_MAX_ITER = 435000


cross_criterion = torch.nn.CrossEntropyLoss()

EMOTION_TABLE_PREFIXES = (
    "emo_embed",
)
EMOTION_SHARED_PREFIXES = (
    "null_emotion_feat",
    "adaLN_modulation",
)
EMOTION_CORE_PREFIXES = EMOTION_TABLE_PREFIXES + EMOTION_SHARED_PREFIXES
START_PARAMETER_PREFIXES = (
    "start_motion_feat",
    "start_audio_feat",
)
AUDIO_ENCODER_PREFIXES = (
    "audio_encoder",
)


def _matches_prefix(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def is_emotion_table_parameter(name):
    return _matches_prefix(name, EMOTION_TABLE_PREFIXES)


def is_emotion_core_parameter(name):
    return _matches_prefix(name, EMOTION_CORE_PREFIXES)


def is_start_parameter(name):
    return _matches_prefix(name, START_PARAMETER_PREFIXES)


def is_audio_encoder_parameter(name):
    return _matches_prefix(name, AUDIO_ENCODER_PREFIXES)


def initialize_generic_stage(model):
    """Start Phase 1 from an emotion-invariant and synchronized condition."""
    with torch.no_grad():
        model.start_motion_feat.zero_()
        model.start_audio_feat.zero_()

        if hasattr(model, "null_emotion_feat"):
            model.null_emotion_feat.zero_()
        if hasattr(model, "emo_embed"):
            model.emo_embed.weight.zero_()
        if hasattr(model, "adaLN_modulation"):
            last_linear = model.adaLN_modulation[-1]
            if hasattr(last_linear, "bias") and last_linear.bias is not None:
                last_linear.bias.zero_()


def sync_generic_priors(model, include_emotion_embedding):
    """Copy row 0 generic priors to every emotion-specific row."""
    with torch.no_grad():
        model.start_motion_feat[1:].copy_(
            model.start_motion_feat[0:1].expand_as(model.start_motion_feat[1:])
        )
        model.start_audio_feat[1:].copy_(
            model.start_audio_feat[0:1].expand_as(model.start_audio_feat[1:])
        )
        if include_emotion_embedding and hasattr(model, "emo_embed"):
            model.emo_embed.weight[1:].copy_(
                model.emo_embed.weight[0:1].expand_as(model.emo_embed.weight[1:])
            )


def snapshot_initial_trainability(model):
    """Preserve native frozen states, especially the SSL feature extractor."""
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def set_trainability(model, initial_trainability, mode, shared_condition_warmstart):
    """Switch trainable parameter sets without reconstructing the model."""
    for name, parameter in model.named_parameters():
        initially_trainable = initial_trainability.get(name, True)

        if mode == "stage1":
            if shared_condition_warmstart:
                trainable = initially_trainable
            else:
                trainable = initially_trainable and not is_emotion_core_parameter(name)
        elif mode == "emotion_only":
            trainable = initially_trainable and (
                is_emotion_core_parameter(name) or is_start_parameter(name)
            )
        elif mode == "full":
            trainable = initially_trainable
        else:
            raise ValueError(f"Unknown trainability mode: {mode}")

        parameter.requires_grad_(trainable)


def split_model_parameters(model):
    groups = {
        "backbone": [],
        "audio": [],
        "emotion": [],
        "start": [],
    }
    names = {key: [] for key in groups}

    for name, parameter in model.named_parameters():
        if is_start_parameter(name):
            key = "start"
        elif is_emotion_core_parameter(name):
            key = "emotion"
        elif is_audio_encoder_parameter(name):
            key = "audio"
        else:
            key = "backbone"
        groups[key].append(parameter)
        names[key].append(name)

    return groups, names


def reset_optimizer_group_state(optimizer, group_name):
    """Remove cross-domain Adam moments while preserving parameter values."""
    for group in optimizer.param_groups:
        if group.get("name") != group_name:
            continue
        for parameter in group["params"]:
            optimizer.state.pop(parameter, None)


def clear_optimizer_group_grads(optimizer, group_names):
    """Prevent unlabeled generic replay from changing emotion-specific priors."""
    group_names = set(group_names)
    for group in optimizer.param_groups:
        if group.get("name") not in group_names:
            continue
        for parameter in group["params"]:
            parameter.grad = None


def linear_lr(local_step, total_steps, start_lr, end_lr):
    total_steps = max(1, int(total_steps))
    local_step = min(max(1, int(local_step)), total_steps)
    progress = local_step / float(total_steps)
    return start_lr + (end_lr - start_lr) * progress


def warmup_cosine_tail_lr(local_step, total_steps, warmup_steps,
                          start_lr, peak_lr, min_lr, tail_steps=0):
    """Linear warm-up, cosine decay, then a useful constant-LR tail."""
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps)
    tail_steps = min(max(0, int(tail_steps)), max(0, total_steps - warmup_steps))
    local_step = min(max(1, int(local_step)), total_steps)

    if warmup_steps > 0 and local_step <= warmup_steps:
        return linear_lr(local_step, warmup_steps, start_lr, peak_lr)

    decay_steps = max(1, total_steps - warmup_steps - tail_steps)
    decay_step = local_step - warmup_steps
    if decay_step >= decay_steps:
        return min_lr

    progress = max(0.0, decay_step / float(decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def get_training_phase(args, iteration):
    if iteration <= args.stage1_iter:
        return 1
    if iteration <= args.stage1_iter + args.stage2_emotion_only_iter:
        return 2
    return 3


def update_learning_rate(args, optimizer, iteration):
    phase = get_training_phase(args, iteration)

    if phase == 1:
        backbone_lr = warmup_cosine_tail_lr(
            iteration,
            args.stage1_iter,
            args.stage1_warm_iter,
            0.0,
            args.stage1_lr,
            args.stage1_lr * args.stage1_min_lr_ratio,
            args.stage1_tail_iter,
        )
        audio_lr = warmup_cosine_tail_lr(
            iteration,
            args.stage1_iter,
            args.stage1_warm_iter,
            0.0,
            args.stage1_audio_lr,
            args.stage1_audio_lr * args.stage1_min_lr_ratio,
            args.stage1_tail_iter,
        )
        start_lr = backbone_lr
        if args.shared_condition_warmstart:
            emotion_lr = warmup_cosine_tail_lr(
                iteration,
                args.stage1_iter,
                args.stage1_warm_iter,
                0.0,
                args.stage1_condition_lr,
                args.stage1_condition_lr * args.stage1_min_lr_ratio,
                args.stage1_tail_iter,
            )
        else:
            emotion_lr = 0.0

    elif phase == 2:
        local_step = iteration - args.stage1_iter
        backbone_lr = 0.0
        audio_lr = 0.0
        condition_start_lr = (
            args.stage1_condition_lr * args.stage1_min_lr_ratio
            if args.shared_condition_warmstart else 0.0
        )
        emotion_lr = linear_lr(
            local_step,
            args.stage2_emotion_only_iter,
            condition_start_lr,
            args.stage2_emotion_lr,
        )
        start_lr = linear_lr(
            local_step,
            args.stage2_emotion_only_iter,
            args.stage1_lr * args.stage1_min_lr_ratio,
            args.stage2_start_lr,
        )

    else:
        local_step = (
            iteration - args.stage1_iter - args.stage2_emotion_only_iter
        )
        total_steps = (
            args.max_iter - args.stage1_iter - args.stage2_emotion_only_iter
        )
        backbone_lr = warmup_cosine_tail_lr(
            local_step,
            total_steps,
            args.stage2_backbone_warm_iter,
            args.stage1_lr * args.stage1_min_lr_ratio,
            args.stage2_backbone_lr,
            args.stage2_backbone_lr * args.stage2_min_lr_ratio,
            args.stage2_tail_iter,
        )
        audio_lr = warmup_cosine_tail_lr(
            local_step,
            total_steps,
            args.stage2_backbone_warm_iter,
            args.stage1_audio_lr * args.stage1_min_lr_ratio,
            args.stage2_audio_lr,
            args.stage2_audio_lr * args.stage2_min_lr_ratio,
            args.stage2_tail_iter,
        )
        emotion_lr = warmup_cosine_tail_lr(
            local_step,
            total_steps,
            0,
            args.stage2_emotion_lr,
            args.stage2_emotion_lr,
            args.stage2_emotion_lr * args.stage2_min_lr_ratio,
            args.stage2_tail_iter,
        )
        start_lr = warmup_cosine_tail_lr(
            local_step,
            total_steps,
            0,
            args.stage2_start_lr,
            args.stage2_start_lr,
            args.stage2_start_lr * args.stage2_min_lr_ratio,
            args.stage2_tail_iter,
        )

    lr_dict = {
        "backbone": backbone_lr,
        "audio": audio_lr,
        "emotion": emotion_lr,
        "start": start_lr,
    }
    for group in optimizer.param_groups:
        group["lr"] = lr_dict[group["name"]]
    return phase, lr_dict


def stratified_diffusion_timesteps(batch_size, num_steps, device):
    """Cover the diffusion-time range more uniformly inside every batch."""
    u = (
        torch.arange(batch_size, device=device)
        + torch.rand(batch_size, device=device)
    ) / batch_size
    steps = torch.clamp((u * num_steps).long() + 1, max=num_steps)
    return steps[torch.randperm(batch_size, device=device)]


def min_snr_primary_loss(args, model, is_starting_sample, motion_gt, noise,
                         prediction, prev_motion, end_idx, time_step):
    """Masked per-sample Min-SNR weighting for the primary diffusion loss."""
    if args.target == "noise":
        pred = prediction[:, args.n_prev_motions:]
        gt = noise
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    elif args.target == "sample":
        if is_starting_sample:
            pred = prediction[:, args.n_prev_motions:]
            gt = motion_gt
            mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        else:
            pred = prediction
            gt = torch.cat([prev_motion, motion_gt], dim=1)
            if args.no_constrain_prev:
                prev_mask = torch.zeros(
                    pred.shape[0], args.n_prev_motions,
                    dtype=torch.bool, device=pred.device,
                )
            else:
                prev_mask = torch.ones(
                    pred.shape[0], args.n_prev_motions,
                    dtype=torch.bool, device=pred.device,
                )
            mask = torch.cat([
                prev_mask,
                torch.ones(
                    pred.shape[0], args.n_motions,
                    dtype=torch.bool, device=pred.device,
                ),
            ], dim=1)
    else:
        raise ValueError(f"Unsupported target for Min-SNR: {args.target}")

    if end_idx is not None:
        current_mask = torch.arange(
            args.n_motions, device=pred.device
        ).expand(pred.shape[0], -1) < end_idx.unsqueeze(1)
        if args.target == "sample" and not is_starting_sample:
            mask = torch.cat([mask[:, :args.n_prev_motions], current_mask], dim=1)
        else:
            mask = current_mask

    if args.criterion == "l2":
        elementwise = (pred - gt).square()
    else:
        elementwise = (pred - gt).abs()
    valid = mask.unsqueeze(-1).to(elementwise.dtype)
    denominator = (
        valid.sum(dim=(1, 2)).clamp_min(1.0) * elementwise.shape[-1]
    )
    per_sample = (elementwise * valid).sum(dim=(1, 2)) / denominator

    alpha_bar = model.diffusion_sched.alpha_bars[time_step]
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    gamma = torch.full_like(snr, args.min_snr_gamma)
    if args.target == "sample":
        weight = torch.minimum(snr, gamma)
    else:
        weight = torch.minimum(snr, gamma) / snr.clamp_min(1e-8)
    weight = weight / weight.mean().detach().clamp_min(1e-8)
    return (per_sample * weight).mean()


@torch.no_grad()
def update_ema(ema_model, model, max_decay, num_updates):
    """EMA with a short bias-reduced warm-up at the beginning."""
    warm_decay = (1.0 + num_updates) / (10.0 + num_updates)
    decay = min(float(max_decay), float(warm_decay))

    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters()
    ):
        ema_parameter.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


def _clone_dataset_with_crop(dataset, crop_strategy):
    dataset_copy = copy.copy(dataset)
    dataset_copy.crop_strategy = crop_strategy
    return dataset_copy


def _build_train_loader(dataset, batch_size, num_workers, seed, sample_weights=None):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=seed_worker,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True

    if sample_weights is None:
        kwargs["shuffle"] = True
        kwargs["generator"] = make_generator(seed)
    else:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=make_generator(seed),
        )
        kwargs["sampler"] = sampler
        kwargs["shuffle"] = False

    return data.DataLoader(**kwargs)


def _next_restart(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


class AlternatingBatchStream:
    """Yield semantic continuation/start batches instead of relying on it parity."""

    def __init__(self, dataset, batch_size, num_workers, seed,
                 start_interval=2, sample_weights=None):
        if start_interval < 2:
            raise ValueError("start_interval must be at least 2")
        self.start_interval = int(start_interval)
        self.counter = 0

        random_dataset = _clone_dataset_with_crop(dataset, "random")
        start_dataset = _clone_dataset_with_crop(dataset, "begin64")
        self.random_loader = _build_train_loader(
            random_dataset, batch_size, num_workers, seed, sample_weights
        )
        self.start_loader = _build_train_loader(
            start_dataset, batch_size, num_workers, seed + 1, sample_weights
        )
        self.random_iterator = iter(self.random_loader)
        self.start_iterator = iter(self.start_loader)

    def __next__(self):
        is_starting_sample = (
            (self.counter + 1) % self.start_interval == 0
        )
        if is_starting_sample:
            batch, self.start_iterator = _next_restart(
                self.start_loader, self.start_iterator
            )
        else:
            batch, self.random_iterator = _next_restart(
                self.random_loader, self.random_iterator
            )
        self.counter += 1
        return batch, is_starting_sample


class ContinuationBatchStream:
    """Random 80-frame continuation samples for optional generic replay."""

    def __init__(self, dataset, batch_size, num_workers, seed):
        random_dataset = _clone_dataset_with_crop(dataset, "random")
        self.loader = _build_train_loader(
            random_dataset, batch_size, num_workers, seed
        )
        self.iterator = iter(self.loader)

    def __next__(self):
        batch, self.iterator = _next_restart(self.loader, self.iterator)
        return batch


def _mead_group_key(metadata):
    stem = Path(metadata["video_name"]).stem
    parts = stem.split("_")
    emotion = parts[2] if len(parts) > 2 else "unknown"
    level = parts[4] if len(parts) > 4 else "unknown"
    return emotion, level


def build_mead_sample_weights(dataset, power=0.5):
    """Inverse-frequency emotion-level weights with a conservative power."""
    keys = [_mead_group_key(metadata) for metadata in dataset.all_data]
    counts = Counter(keys)
    weights = np.asarray(
        [counts[key] ** (-float(power)) for key in keys],
        dtype=np.float64,
    )
    weights /= max(weights.mean(), 1e-12)
    return weights, counts


def build_validation_loaders(dataset, batch_size):
    continuation_dataset = _clone_dataset_with_crop(dataset, "begin")
    start_dataset = _clone_dataset_with_crop(dataset, "begin64")
    common = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return (
        data.DataLoader(continuation_dataset, **common),
        data.DataLoader(start_dataset, **common),
    )


def _weighted_total_loss(args, primary, emotion, expression, exp_vel,
                         exp_smooth, head_angle, head_vel, head_smooth,
                         head_trans):
    total = primary
    if emotion is not None:
        total = total + args.l_emo * emotion
    total = total + args.l_exp * expression
    total = total + args.l_exp_vel * exp_vel
    total = total + args.l_exp_smooth * exp_smooth
    total = total + args.l_head_angle * head_angle
    total = total + args.l_head_vel * head_vel
    total = total + args.l_head_smooth * head_smooth
    total = total + args.l_head_trans * head_trans
    return total


def _prepare_model_forward(args, model, audio, motion_coef_full,
                           current_audio, current_motion,
                           is_starting_sample, indicator, emo_index,
                           time_step, n_prev_audio_samples):
    """Encode audio exactly once and keep continuation context consistent."""
    if is_starting_sample:
        current_audio_feat = model.extract_audio_feature(
            current_audio, frame_num=args.n_motions
        )
        noise, target, _, _ = model(
            current_motion,
            current_audio_feat,
            time_step=time_step,
            indicator=indicator,
            emo_index=emo_index,
        )
        prev_motion_for_loss = None
    else:
        prev_motion_gt = motion_coef_full[:, :args.n_prev_motions]
        prev_audio_raw = audio[:, :n_prev_audio_samples].contiguous()
        context_audio = torch.cat([prev_audio_raw, current_audio], dim=1)
        context_audio_feat = model.extract_audio_feature(
            context_audio,
            frame_num=args.n_prev_motions + args.n_motions,
        )
        prev_audio_feat = context_audio_feat[:, :args.n_prev_motions].detach()
        current_audio_feat = context_audio_feat[:, args.n_prev_motions:]
        noise, target, _, _ = model(
            current_motion,
            current_audio_feat,
            prev_motion_gt,
            prev_audio_feat,
            time_step=time_step,
            indicator=indicator,
            emo_index=emo_index,
        )
        prev_motion_for_loss = prev_motion_gt

    return noise, target, prev_motion_for_loss


def _compute_losses(args, model, classifier, use_emotion_loss,
                    is_starting_sample, current_motion, noise, target,
                    prev_motion_for_loss, end_idx, emo_index, time_step):
    losses = utils.compute_loss_new(
        args,
        is_starting_sample,
        current_motion,
        noise,
        target,
        prev_motion_for_loss,
        end_idx,
    )
    (loss_primary, loss_expression, loss_exp_vel, loss_exp_smooth,
     loss_head_angle, loss_head_vel, loss_head_smooth,
     loss_head_trans) = losses

    if args.use_min_snr:
        if time_step is None:
            raise RuntimeError("Min-SNR requires explicit diffusion time steps")
        loss_primary = min_snr_primary_loss(
            args,
            model,
            is_starting_sample,
            current_motion,
            noise,
            target,
            prev_motion_for_loss,
            end_idx,
            time_step,
        )

    zero = torch.tensor(0.0, device=target.device)
    loss_expression = loss_expression if loss_expression is not None else zero
    loss_exp_vel = loss_exp_vel if loss_exp_vel is not None else zero
    loss_exp_smooth = loss_exp_smooth if loss_exp_smooth is not None else zero
    loss_head_angle = loss_head_angle if loss_head_angle is not None else zero
    loss_head_vel = loss_head_vel if loss_head_vel is not None else zero
    loss_head_smooth = loss_head_smooth if loss_head_smooth is not None else zero
    loss_head_trans = loss_head_trans if loss_head_trans is not None else zero

    loss_emotion = None
    if use_emotion_loss and args.target == "sample":
        exps = target[:, args.n_prev_motions:, :63].clone()
        pred_emotion, _ = classifier(exps)
        loss_emotion = cross_criterion(pred_emotion, emo_index)

    total = _weighted_total_loss(
        args,
        loss_primary,
        loss_emotion,
        loss_expression,
        loss_exp_vel,
        loss_exp_smooth,
        loss_head_angle,
        loss_head_vel,
        loss_head_smooth,
        loss_head_trans,
    )
    return {
        "total": total,
        "primary": loss_primary,
        "emotion": loss_emotion if loss_emotion is not None else zero,
        "expression": loss_expression,
        "exp_vel": loss_exp_vel,
        "exp_smooth": loss_exp_smooth,
        "head_angle": loss_head_angle,
        "head_vel": loss_head_vel,
        "head_smooth": loss_head_smooth,
        "head_trans": loss_head_trans,
    }


@torch.no_grad()
def evaluate(args, model, classifier, continuation_loader, start_loader):
    """Small deterministic MEAD validation used for checkpoint selection."""
    was_training = model.training
    model.eval()
    classifier.eval()
    device = model.device

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    random.seed(args.seed + 100003)
    np.random.seed(args.seed + 100003)
    torch.manual_seed(args.seed + 100003)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + 100003)

    sums = defaultdict(float)
    batch_count = 0
    try:
        for is_starting_sample, loader in (
            (False, continuation_loader),
            (True, start_loader),
        ):
            for batch_index, batch in enumerate(loader):
                if batch_index >= args.val_batches:
                    break

                audio, coef_dict, emo_index, _ = batch
                audio = audio.to(device, non_blocking=True)
                coef_dict = {
                    key: value.to(device, non_blocking=True)
                    for key, value in coef_dict.items()
                }
                emo_index = emo_index.to(device, non_blocking=True)
                motion_coef_full = utils.get_motion_coef(
                    coef_dict, args.rot_repr, not args.no_head_pose
                )
                audio_unit = continuation_loader.dataset.audio_unit
                n_audio_samples = round(audio_unit * args.n_motions)
                n_prev_audio_samples = round(audio_unit * args.n_prev_motions)
                current_audio = audio[:, -n_audio_samples:].contiguous()
                current_motion = motion_coef_full[:, -args.n_motions:]
                indicator = (
                    torch.ones(
                        audio.shape[0], args.n_motions,
                        device=device, dtype=torch.bool,
                    ) if args.use_indicator else None
                )
                time_step = stratified_diffusion_timesteps(
                    audio.shape[0], model.diffusion_sched.num_steps, device
                )

                noise, target, prev_motion_for_loss = _prepare_model_forward(
                    args,
                    model,
                    audio,
                    motion_coef_full,
                    current_audio,
                    current_motion,
                    is_starting_sample,
                    indicator,
                    emo_index,
                    time_step,
                    n_prev_audio_samples,
                )
                loss_dict = _compute_losses(
                    args,
                    model,
                    classifier,
                    True,
                    is_starting_sample,
                    current_motion,
                    noise,
                    target,
                    prev_motion_for_loss,
                    None,
                    emo_index,
                    time_step,
                )
                for key, value in loss_dict.items():
                    sums[key] += float(value.item())
                batch_count += 1
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        model.train(was_training)

    if batch_count == 0:
        raise RuntimeError("Validation produced no batches")
    return {key: value / batch_count for key, value in sums.items()}


def _checkpoint_payload(args, model, ema_model, iteration, phase,
                        best_val, ema_updates, optimizer=None):
    raw_state = model.state_dict()
    inference_state = (
        ema_model.state_dict() if ema_model is not None else raw_state
    )
    payload = {
        "args": args,
        "model": inference_state,
        "model_ema": inference_state,
        "model_raw": raw_state,
        "iter": iteration,
        "stage": 1 if phase == 1 else 2,
        "phase": phase,
        "variant_name": VARIANT_NAME,
        "best_val": best_val,
        "ema_updates": ema_updates,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def train(args, model, generic_dataset, mead_dataset, optimizer, save_dir,
          writer, classifier, initial_trainability, ema_model,
          continuation_val_loader, start_val_loader,
          start_iter=0, best_val=float("inf"), ema_updates=0):
    save_dir.mkdir(parents=True, exist_ok=True)
    device = model.device
    model.train()

    mead_weights = None
    if args.balance_mead:
        mead_weights, group_counts = build_mead_sample_weights(
            mead_dataset, args.balance_power
        )
        logging.info("MEAD emotion-level group counts: %s", dict(group_counts))

    generic_stream = AlternatingBatchStream(
        generic_dataset,
        args.batch_size,
        args.num_workers,
        args.seed + 11,
        start_interval=args.start_interval,
    )
    mead_stream = AlternatingBatchStream(
        mead_dataset,
        args.batch_size,
        args.num_workers,
        args.seed + 31,
        start_interval=args.start_interval,
        sample_weights=mead_weights,
    )
    generic_replay_stream = None
    if args.generic_replay_interval > 0:
        generic_replay_stream = ContinuationBatchStream(
            generic_dataset,
            args.batch_size,
            args.num_workers,
            args.seed + 71,
        )

    audio_unit = mead_dataset.audio_unit
    if abs(generic_dataset.audio_unit - audio_unit) > 1e-6:
        raise RuntimeError("Generic and MEAD audio units do not match")
    n_audio_samples = round(audio_unit * args.n_motions)
    n_prev_audio_samples = round(audio_unit * args.n_prev_motions)

    current_phase = get_training_phase(args, max(1, start_iter))
    if current_phase == 1:
        set_trainability(
            model, initial_trainability, "stage1",
            args.shared_condition_warmstart,
        )
    elif current_phase == 2:
        set_trainability(
            model, initial_trainability, "emotion_only",
            args.shared_condition_warmstart,
        )
    else:
        set_trainability(
            model, initial_trainability, "full",
            args.shared_condition_warmstart,
        )

    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    for iteration in range(start_iter + 1, args.max_iter + 1):
        phase = get_training_phase(args, iteration)
        if phase != current_phase:
            optimizer.zero_grad(set_to_none=True)
            micro_step = 0

            if phase == 2:
                sync_generic_priors(
                    model, args.shared_condition_warmstart
                )
                reset_optimizer_group_state(optimizer, "start")
                reset_optimizer_group_state(optimizer, "emotion")
                set_trainability(
                    model, initial_trainability, "emotion_only",
                    args.shared_condition_warmstart,
                )
                logging.info(
                    "========== Phase 2 at iter %d: MEAD emotion/start warm-up ==========",
                    iteration,
                )
            elif phase == 3:
                reset_optimizer_group_state(optimizer, "backbone")
                reset_optimizer_group_state(optimizer, "audio")
                set_trainability(
                    model, initial_trainability, "full",
                    args.shared_condition_warmstart,
                )
                logging.info(
                    "========== Phase 3 at iter %d: full MEAD fine-tuning ==========",
                    iteration,
                )
            current_phase = phase
            writer.add_text(
                "train/phase_transition",
                f"phase {phase} starts at iter {iteration}",
                iteration,
            )

        phase, lr_dict = update_learning_rate(args, optimizer, iteration)
        phase3_step = max(
            0,
            iteration - args.stage1_iter - args.stage2_emotion_only_iter,
        )
        is_generic_replay = (
            phase == 3
            and args.generic_replay_interval > 0
            and phase3_step % args.generic_replay_interval == 0
        )

        if phase == 1:
            (audio, coef_dict), is_starting_sample = next(generic_stream)
            batch_size = audio.shape[0]
            emo_index = torch.zeros(batch_size, dtype=torch.long)
            use_emotion_loss = False
            data_name = "generic"
        elif is_generic_replay:
            if generic_replay_stream is None:
                raise RuntimeError("generic replay stream was not initialized")
            audio, coef_dict = next(generic_replay_stream)
            batch_size = audio.shape[0]
            emo_index = torch.full(
                (batch_size,), args.generic_replay_emo_index,
                dtype=torch.long,
            )
            is_starting_sample = False
            use_emotion_loss = False
            data_name = "generic_replay"
        else:
            (audio, coef_dict, emo_index, _), is_starting_sample = next(mead_stream)
            use_emotion_loss = True
            data_name = "mead"

        audio = audio.to(device, non_blocking=True)
        coef_dict = {
            key: value.to(device, non_blocking=True)
            for key, value in coef_dict.items()
        }
        emo_index = emo_index.to(device, non_blocking=True)
        motion_coef_full = utils.get_motion_coef(
            coef_dict, args.rot_repr, not args.no_head_pose
        )
        batch_size = audio.shape[0]

        current_audio = audio[:, -n_audio_samples:].contiguous()
        current_motion = motion_coef_full[:, -args.n_motions:]
        end_idx = None
        trunc_probability = (
            args.trunc_prob1 if is_starting_sample else args.trunc_prob2
        )
        if np.random.rand() < trunc_probability:
            current_audio, current_motion, end_idx = (
                utils.truncate_motion_coef_and_audio(
                    current_audio,
                    current_motion,
                    args.n_motions,
                    audio_unit,
                    args.pad_mode,
                )
            )

        if args.use_indicator:
            if end_idx is None:
                indicator = torch.ones(
                    batch_size, args.n_motions,
                    device=device, dtype=torch.bool,
                )
            else:
                indicator = torch.arange(
                    args.n_motions, device=device
                ).expand(batch_size, -1) < end_idx.unsqueeze(1)
        else:
            indicator = None

        time_step = None
        if args.use_min_snr:
            time_step = stratified_diffusion_timesteps(
                batch_size, model.diffusion_sched.num_steps, device
            )

        noise, target, prev_motion_for_loss = _prepare_model_forward(
            args,
            model,
            audio,
            motion_coef_full,
            current_audio,
            current_motion,
            is_starting_sample,
            indicator,
            emo_index,
            time_step,
            n_prev_audio_samples,
        )
        loss_dict = _compute_losses(
            args,
            model,
            classifier,
            use_emotion_loss,
            is_starting_sample,
            current_motion,
            noise,
            target,
            prev_motion_for_loss,
            end_idx,
            emo_index,
            time_step,
        )

        (loss_dict["total"] / args.gradient_accumulation_steps).backward()
        micro_step += 1

        if is_generic_replay:
            if args.gradient_accumulation_steps != 1:
                raise RuntimeError(
                    "generic replay currently requires gradient_accumulation_steps=1"
                )
            clear_optimizer_group_grads(optimizer, {"emotion", "start"})

        should_step = (
            micro_step % args.gradient_accumulation_steps == 0
            or iteration == args.max_iter
        )
        if should_step:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters()
                     if parameter.requires_grad],
                    max_norm=args.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if phase == 1:
                sync_generic_priors(
                    model, args.shared_condition_warmstart
                )
            if ema_model is not None:
                ema_updates += 1
                update_ema(
                    ema_model, model, args.ema_decay, ema_updates
                )

        weighted_values = {
            "total": loss_dict["total"].item(),
            "primary": loss_dict["primary"].item(),
            "emotion": args.l_emo * loss_dict["emotion"].item(),
            "expression": args.l_exp * loss_dict["expression"].item(),
            "exp_vel": args.l_exp_vel * loss_dict["exp_vel"].item(),
            "exp_smooth": args.l_exp_smooth * loss_dict["exp_smooth"].item(),
            "head_angle": args.l_head_angle * loss_dict["head_angle"].item(),
            "head_vel": args.l_head_vel * loss_dict["head_vel"].item(),
            "head_smooth": args.l_head_smooth * loss_dict["head_smooth"].item(),
            "head_trans": args.l_head_trans * loss_dict["head_trans"].item(),
        }
        for key, value in weighted_values.items():
            loss_log[key].append(value)

        if iteration % args.log_iter == 0:
            logging.info(
                "Iter: %d\tPhase: %d\tData: %s\tStart: %s\t"
                "Loss[T/P/E/Emo]: %.3e/%.3e/%.3e/%.3e\t"
                "LR[B/A/E/S]: %.3e/%.3e/%.3e/%.3e",
                iteration,
                phase,
                data_name,
                is_starting_sample,
                np.mean(loss_log["total"]),
                np.mean(loss_log["primary"]),
                np.mean(loss_log["expression"]),
                np.mean(loss_log["emotion"]),
                lr_dict["backbone"],
                lr_dict["audio"],
                lr_dict["emotion"],
                lr_dict["start"],
            )
            writer.add_scalar("train/stage", 1 if phase == 1 else 2, iteration)
            writer.add_scalar("train/phase", phase, iteration)
            writer.add_scalar("train/is_start", int(is_starting_sample), iteration)
            writer.add_scalar("train/is_generic_replay", int(is_generic_replay), iteration)
            for key in (
                "total", "primary", "emotion", "expression", "exp_vel",
                "exp_smooth", "head_angle", "head_vel", "head_smooth",
                "head_trans",
            ):
                writer.add_scalar(f"train/{key}_loss", np.mean(loss_log[key]), iteration)

            # Keep the original 0819 TensorBoard tag names so old and new
            # experiments can be compared in the same dashboard.
            writer.add_scalar("train/simple_loss", np.mean(loss_log["primary"]), iteration)
            writer.add_scalar("train/exp_loss", np.mean(loss_log["expression"]), iteration)
            writer.add_scalar("train/head_angle", np.mean(loss_log["head_angle"]), iteration)
            writer.add_scalar("train/head_vel", np.mean(loss_log["head_vel"]), iteration)
            writer.add_scalar("train/head_smooth", np.mean(loss_log["head_smooth"]), iteration)
            writer.add_scalar("train/head_trans", np.mean(loss_log["head_trans"]), iteration)

            for key, value in lr_dict.items():
                writer.add_scalar(f"opt/lr_{key}", value, iteration)

        should_validate = (
            phase > 1
            and args.val_iter > 0
            and (iteration % args.val_iter == 0 or iteration == args.max_iter)
        )
        if should_validate:
            validation_model = ema_model if ema_model is not None else model
            val_metrics = evaluate(
                args,
                validation_model,
                classifier,
                continuation_val_loader,
                start_val_loader,
            )
            logging.info(
                "Validation at %d: total=%.6f primary=%.6f exp=%.6f emo=%.6f",
                iteration,
                val_metrics["total"],
                val_metrics["primary"],
                val_metrics["expression"],
                val_metrics["emotion"],
            )
            for key, value in val_metrics.items():
                writer.add_scalar(f"val/{key}_loss", value, iteration)

            if val_metrics["total"] < best_val:
                best_val = val_metrics["total"]
                torch.save(
                    _checkpoint_payload(
                        args, model, ema_model, iteration, phase,
                        best_val, ema_updates,
                    ),
                    save_dir / "best.pt",
                )

        should_save = (
            iteration % args.save_iter == 0
            or iteration == args.stage1_iter
            or iteration == args.stage1_iter + args.stage2_emotion_only_iter
            or iteration == args.max_iter
        )
        if should_save:
            torch.save(
                _checkpoint_payload(
                    args, model, ema_model, iteration, phase,
                    best_val, ema_updates,
                ),
                save_dir / f"iter_{iteration:07}.pt",
            )

        should_save_state = (
            args.resume_save_iter > 0
            and (
                iteration % args.resume_save_iter == 0
                or iteration == args.stage1_iter
                or iteration == args.stage1_iter + args.stage2_emotion_only_iter
                or iteration == args.max_iter
            )
        )
        if should_save_state:
            torch.save(
                _checkpoint_payload(
                    args, model, ema_model, iteration, phase,
                    best_val, ema_updates, optimizer=optimizer,
                ),
                save_dir / "latest_train_state.pt",
            )

    return best_val, ema_updates


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def count_all_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def add_bool_argument(parser, name, default, help_text):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(f"--{name}", dest=name, action="store_true")
    group.add_argument(f"--no_{name}", dest=name, action="store_false")
    parser.set_defaults(**{name: default})
    parser._option_string_actions[f"--{name}"].help = help_text


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train"])
    parser.add_argument("--exp_name", type=str, default=DEFAULT_EXP_NAME)
    parser.add_argument("--model_variant", type=str, default="jianhua0803")
    parser.add_argument("--device_id", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)

    # Stage-2 MEAD data.
    parser.add_argument("--data_root", type=Path, default="src/my_prepare/")
    parser.add_argument("--motion_filename", type=str, default="front_all_motions.pkl")
    parser.add_argument("--motion_template_filename", type=str, default="motion_template.pkl")

    # Stage-1 generic data.
    parser.add_argument(
        "--generic_motion_filenames",
        type=str,
        default=(
            "src/data_processing/HDTF/front_all_motions.pkl,"
            "src/my_prepare/front_all_motions.pkl"
        ),
    )
    parser.add_argument("--generic_aggregate_motion_files", type=str, default="")
    parser.add_argument("--generic_motion_template_path", type=Path, default=None)
    parser.add_argument("--generic_split_file", type=Path, default=None)
    parser.add_argument("--generic_validation_ratio", type=float, default=0.0)
    parser.add_argument("--generic_split_seed", type=int, default=2026)
    add_bool_argument(
        parser, "generic_allow_relative_paths", False,
        "Allow relative paths inside generic motion pickle files.",
    )
    parser.add_argument(
        "--generic_missing_audio_policy", type=str, default="skip",
        choices=["skip", "error"],
    )
    parser.add_argument(
        "--generic_duplicate_policy", type=str, default="keep_first",
        choices=["error", "keep_first", "keep_last"],
    )

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_strategy", type=str, default="random")
    parser.add_argument(
        "--normalize_type", type=str, default="mix",
        choices=["std", "case", "scale", "minmax", "mix"],
    )

    # Model.
    parser.add_argument("--target", type=str, default="sample", choices=["sample", "noise"])
    parser.add_argument("--guiding_conditions", type=str, default="audio,emotion")
    parser.add_argument(
        "--cfg_mode", type=str, default="incremental",
        choices=["incremental", "independent"],
    )
    parser.add_argument("--n_diff_steps", type=int, default=500)
    parser.add_argument(
        "--diff_schedule", type=str, default="cosine",
        choices=["linear", "cosine", "quadratic", "sigmoid"],
    )
    parser.add_argument(
        "--no_head_pose", action="store_true", default=False,
        help="Disable head-pose prediction.",
    )
    parser.add_argument("--rot_repr", type=str, default="aa", choices=["aa"])
    parser.add_argument(
        "--audio_model", type=str, default="wav2vec2",
        choices=["wav2vec2", "hubert", "hubert_zh", "hubert_zh_ori"],
    )
    parser.add_argument("--architecture", type=str, default="decoder", choices=["decoder"])
    parser.add_argument("--align_mask_width", type=int, default=2)
    parser.add_argument(
        "--no_use_learnable_pe", action="store_true", default=False,
        help="Use fixed positional encoding instead of learnable PE.",
    )
    add_bool_argument(parser, "use_indicator", True, "Use padded-frame indicator.")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--n_motions", type=int, default=64)
    parser.add_argument("--n_prev_motions", type=int, default=16)
    parser.add_argument("--motion_feat_dim", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "replicate"])

    # Performance-oriented two-stage budget.
    parser.add_argument("--max_iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--stage1_iter", type=int, default=190000)
    parser.add_argument("--stage2_emotion_only_iter", type=int, default=5000)
    parser.add_argument("--start_interval", type=int, default=2)

    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage1_audio_lr", type=float, default=5e-5)
    parser.add_argument("--stage1_condition_lr", type=float, default=3e-5)
    parser.add_argument("--stage1_warm_iter", type=int, default=19000)
    parser.add_argument("--stage1_min_lr_ratio", type=float, default=0.10)
    parser.add_argument("--stage1_tail_iter", type=int, default=10000)

    parser.add_argument("--stage2_emotion_lr", type=float, default=8e-5)
    parser.add_argument("--stage2_start_lr", type=float, default=4e-5)
    parser.add_argument("--stage2_backbone_lr", type=float, default=2e-5)
    parser.add_argument("--stage2_audio_lr", type=float, default=5e-6)
    parser.add_argument("--stage2_backbone_warm_iter", type=int, default=10000)
    parser.add_argument("--stage2_min_lr_ratio", type=float, default=0.10)
    parser.add_argument("--stage2_tail_iter", type=int, default=40000)

    add_bool_argument(
        parser, "shared_condition_warmstart",
        DEFAULT_SHARED_CONDITION_WARMSTART,
        "Pretrain a shared emo embedding/modulation on generic data, then specialize.",
    )
    add_bool_argument(
        parser, "use_min_snr", DEFAULT_USE_MIN_SNR,
        "Use stratified time steps and Min-SNR primary-loss weighting.",
    )
    parser.add_argument("--min_snr_gamma", type=float, default=5.0)
    add_bool_argument(
        parser, "balance_mead", DEFAULT_BALANCE_MEAD,
        "Use inverse-sqrt emotion-level sampling on MEAD.",
    )
    parser.add_argument("--balance_power", type=float, default=0.5)
    parser.add_argument(
        "--generic_replay_interval", type=int,
        default=DEFAULT_GENERIC_REPLAY_INTERVAL,
        help="0 disables replay; N uses one generic continuation batch every N Phase-3 steps.",
    )
    parser.add_argument("--generic_replay_emo_index", type=int, default=5)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    add_bool_argument(parser, "clip_grad", True, "Enable gradient clipping.")
    parser.add_argument("--max_grad_norm", type=float, default=2.0)

    # Losses.
    parser.add_argument("--criterion", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--l_emo", type=float, default=1.0)
    parser.add_argument("--l_exp", type=float, default=1.0)
    parser.add_argument("--l_exp_vel", type=float, default=1e-4)
    parser.add_argument("--l_exp_smooth", type=float, default=1e-4)
    parser.add_argument("--l_head_angle", type=float, default=1e-2)
    parser.add_argument("--l_head_vel", type=float, default=1e-2)
    parser.add_argument("--l_head_smooth", type=float, default=1e-2)
    parser.add_argument("--l_head_trans", type=float, default=1e-2)
    parser.add_argument(
        "--no_constrain_prev", action="store_true", default=False,
        help="Do not constrain predicted prev frames.",
    )
    parser.add_argument("--trunc_prob1", type=float, default=0.3)
    parser.add_argument("--trunc_prob2", type=float, default=0.4)

    # EMA, validation, logging, and recovery.
    add_bool_argument(parser, "use_ema", True, "Maintain EMA weights for validation/inference.")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--val_iter", type=int, default=10000)
    parser.add_argument("--val_batches", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=64)
    parser.add_argument("--save_iter", type=int, default=10000)
    parser.add_argument("--resume_save_iter", type=int, default=20000)
    parser.add_argument("--log_iter", type=int, default=50)
    parser.add_argument("--log_smooth_win", type=int, default=50)
    parser.add_argument(
        "--emotion_classifier_path", type=str,
        default="experiments/emo_classifier/ckpt_n64.pth",
    )
    return parser


def validate_args(args):
    if not (0 < args.stage1_iter < args.max_iter):
        raise ValueError("stage1_iter must be inside the total training budget")
    if args.stage1_iter + args.stage2_emotion_only_iter >= args.max_iter:
        raise ValueError("No room remains for full MEAD fine-tuning")
    if args.start_interval < 2:
        raise ValueError("start_interval must be at least 2")
    if args.generic_replay_interval < 0:
        raise ValueError("generic_replay_interval cannot be negative")
    if args.generic_replay_interval > 0 and args.gradient_accumulation_steps != 1:
        raise ValueError("generic replay requires gradient_accumulation_steps=1")
    if not 0 <= args.generic_replay_emo_index < 8:
        raise ValueError("generic_replay_emo_index must be in [0, 7]")


def main(args, option_text=None):
    validate_args(args)
    set_global_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device(f"cuda:{args.device_id}")
    else:
        device = torch.device("cpu")

    model = DitTalkingHead(
        device=device,
        target=args.target,
        architecture=args.architecture,
        motion_feat_dim=args.motion_feat_dim,
        fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        audio_model=args.audio_model,
        feature_dim=args.feature_dim,
        n_diff_steps=args.n_diff_steps,
        diff_schedule=args.diff_schedule,
        cfg_mode=args.cfg_mode,
        guiding_conditions=args.guiding_conditions,
        align_mask_width=args.align_mask_width,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        use_indicator=args.use_indicator,
        no_use_learnable_pe=args.no_use_learnable_pe,
    )
    initial_trainability = snapshot_initial_trainability(model)

    checkpoint = None
    start_iter = 0
    best_val = float("inf")
    ema_updates = 0
    if args.resume_checkpoint is None:
        initialize_generic_stage(model)
        sync_generic_priors(model, args.shared_condition_warmstart)
    else:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        raw_state = checkpoint.get("model_raw", checkpoint["model"])
        model.load_state_dict(raw_state, strict=True)
        start_iter = int(checkpoint.get("iter", 0))
        best_val = float(checkpoint.get("best_val", float("inf")))
        ema_updates = int(checkpoint.get("ema_updates", 0))
        # Never overwrite already-specialized rows when resuming Phase 2/3.
        if start_iter <= args.stage1_iter:
            sync_generic_priors(model, args.shared_condition_warmstart)

    generic_template_path = args.generic_motion_template_path
    if generic_template_path is None:
        generic_template_path = Path(args.data_root) / args.motion_template_filename

    generic_dataset = GenericTalkingMotionDataset(
        motion_template_path=generic_template_path,
        motion_filenames=args.generic_motion_filenames or None,
        aggregate_motion_files=args.generic_aggregate_motion_files or None,
        split="train",
        split_file=args.generic_split_file,
        validation_ratio=args.generic_validation_ratio,
        split_seed=args.generic_split_seed,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
        strict_absolute_paths=not args.generic_allow_relative_paths,
        missing_audio_policy=args.generic_missing_audio_policy,
        duplicate_policy=args.generic_duplicate_policy,
    )
    mead_dataset = EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split="train",
        coef_fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
    )
    mead_val_dataset = EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split="test",
        coef_fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        crop_strategy="begin",
        normalize_type=args.normalize_type,
    )
    continuation_val_loader, start_val_loader = build_validation_loaders(
        mead_val_dataset, args.val_batch_size
    )

    classifier = Classifier().to(device)
    classifier.load_state_dict(
        torch.load(args.emotion_classifier_path, map_location=device),
        strict=False,
    )
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    parameter_groups, parameter_names = split_model_parameters(model)
    optimizer = torch.optim.Adam([
        {"params": parameter_groups["backbone"], "lr": args.stage1_lr, "name": "backbone"},
        {"params": parameter_groups["audio"], "lr": args.stage1_audio_lr, "name": "audio"},
        {"params": parameter_groups["emotion"], "lr": 0.0, "name": "emotion"},
        {"params": parameter_groups["start"], "lr": args.stage1_lr, "name": "start"},
    ])
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    ema_model = None
    if args.use_ema:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
        if checkpoint is not None:
            ema_state = checkpoint.get("model_ema", checkpoint.get("model"))
            if ema_state is not None:
                ema_model.load_state_dict(ema_state, strict=True)

    exp_dir = Path("experiments/emo_dit") / args.exp_name
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / "options.log", "w", encoding="utf-8") as file:
            file.write(option_text)
        writer.add_text("options", option_text)

    logging.basicConfig(
        filename=str(log_dir / "log.txt"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("variant: %s", VARIANT_NAME)
    logging.info("description: %s", VARIANT_DESCRIPTION)
    logging.info("all model parameters: %d", count_all_parameters(model))
    logging.info("initial trainable parameters: %d", count_parameters(model))
    for group_name in ("backbone", "audio", "emotion", "start"):
        logging.info(
            "%s parameter tensors: %d",
            group_name,
            len(parameter_names[group_name]),
        )

    train(
        args,
        model,
        generic_dataset,
        mead_dataset,
        optimizer,
        exp_dir / "checkpoints",
        writer,
        classifier,
        initial_trainability,
        ema_model,
        continuation_val_loader,
        start_val_loader,
        start_iter=start_iter,
        best_val=best_val,
        ema_updates=ema_updates,
    )
    writer.close()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    option_text = utils.common.get_option_text(args, parser)
    main(args, option_text)

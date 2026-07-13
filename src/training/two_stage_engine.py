from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tensorboardX import SummaryWriter

from src.dataset import infinite_data_loader
from src.modules.emotion_dit import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
from .two_stage_batch import format_losses, run_batch, scalar_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_text: str) -> torch.device:
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA is unavailable; falling back to CPU")
        return torch.device("cpu")
    device = torch.device(device_text)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)
    return device


def build_model(args: argparse.Namespace, device: torch.device) -> DitTalkingHead:
    return DitTalkingHead(
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
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        align_mask_width=args.align_mask_width,
        no_use_learnable_pe=not args.use_learnable_pe,
        use_indicator=args.use_indicator,
        training_stage=args.stage,
        general_audio_dropout=args.general_audio_dropout,
        emotion_dropout=args.emotion_dropout,
        emotion_gate_init=args.emotion_gate_init,
    )


def load_model_weights(
    model: DitTalkingHead,
    checkpoint_path: Path,
    device: torch.device,
    label: str,
) -> dict:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = dict(state_dict)
    state_dict.pop("denoising_net.TE.pe", None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed_missing = {"denoising_net.TE.pe", "emotion_gate"}
    material_missing = [key for key in missing if key not in allowed_missing]
    if material_missing or unexpected:
        raise RuntimeError(
            f"Incompatible {label} checkpoint {checkpoint_path}: "
            f"missing={material_missing}, unexpected={unexpected}"
        )
    logging.info("Loaded %s checkpoint: %s", label, checkpoint_path)
    return checkpoint


def build_classifier(
    args: argparse.Namespace, device: torch.device
) -> Optional[Classifier]:
    if args.stage != "emotion":
        return None
    classifier = Classifier().to(device)
    state_dict = torch.load(args.emotion_classifier_ckpt, map_location=device)
    classifier.load_state_dict(state_dict, strict=False)
    classifier.requires_grad_(False)
    classifier.eval()
    return classifier


def lr_multiplier(step: int, args: argparse.Namespace) -> float:
    if step < args.warm_iter:
        return max(1e-8, (step + 1) / max(1, args.warm_iter))
    progress = (step - args.warm_iter) / max(1, args.max_iter - args.warm_iter)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def set_train_mode(model: DitTalkingHead, args: argparse.Namespace) -> None:
    model.train()
    if not args.train_audio_encoder:
        model.audio_encoder.eval()
    if args.stage == "emotion":
        model.audio_feature_map.eval()
        model.audio_norm.eval()
        model.denoising_net.eval()
        model.emo_embed.train()
        model.adaLN_modulation.train()
        if args.stage2_unfreeze_motion_decoder:
            model.denoising_net.motion_dec.train()


@torch.no_grad()
def validate(
    args: argparse.Namespace,
    model: DitTalkingHead,
    val_loader,
    classifier: Optional[Classifier],
    step: int,
    writer: SummaryWriter,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    accum = defaultdict(float)
    count = 0
    for batch in val_loader:
        losses = scalar_dict(run_batch(args, model, batch, classifier, training=False))
        for name, value in losses.items():
            accum[name] += value
        count += 1
    if count == 0:
        raise RuntimeError("Validation loader is empty")
    averages = {name: value / count for name, value in accum.items()}
    logging.info(format_losses("val", step, averages))
    for name, value in averages.items():
        writer.add_scalar(f"val/{name}", value, step)
    if was_training:
        set_train_mode(model, args)
    return averages


def save_checkpoint(
    path: Path,
    args: argparse.Namespace,
    model: DitTalkingHead,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "args": args,
            "stage": args.stage,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iter": step,
        },
        path,
    )


def train(
    args: argparse.Namespace,
    model: DitTalkingHead,
    train_loader,
    val_loader,
    classifier: Optional[Classifier],
    optimizer: torch.optim.Optimizer,
    scheduler,
    writer: SummaryWriter,
    checkpoint_dir: Path,
    start_iter: int,
) -> None:
    loader = infinite_data_loader(train_loader)
    smooth = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad(set_to_none=True)
    set_train_mode(model, args)

    for step in range(start_iter, args.max_iter + 1):
        losses = run_batch(args, model, next(loader), classifier, training=True)
        (losses["loss"] / args.gradient_accumulation_steps).backward()
        should_step = (step + 1) % args.gradient_accumulation_steps == 0
        if should_step:
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.clip_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        values = scalar_dict(losses)
        for name, value in values.items():
            smooth[name].append(value)

        if step % args.log_iter == 0:
            averages = {name: float(np.mean(items)) for name, items in smooth.items()}
            logging.info(format_losses("train", step, averages))
            for name, value in averages.items():
                writer.add_scalar(f"train/{name}", value, step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)

        if step % args.val_iter == 0 or step == args.max_iter:
            validate(args, model, val_loader, classifier, step, writer)

        if (step > 0 and step % args.save_iter == 0) or step == args.max_iter:
            save_checkpoint(
                checkpoint_dir / f"iter_{step:07d}.pt",
                args,
                model,
                optimizer,
                scheduler,
                step,
            )


def configure_logging(exp_dir: Path) -> SummaryWriter:
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_dir / "log.txt")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
        force=True,
    )
    return SummaryWriter(str(log_dir))

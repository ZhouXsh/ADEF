# coding: utf-8
"""Train the two-stage dual-audio ADEF motion generators.

Examples
--------
Stage 1, generic emotion-agnostic audio-motion pretraining::

    python train_two_stage.py --stage 1 --variant finalv1 \
        --generic_video_roots /data/MEAD/videos,/data/general/videos \
        --motion_template_path src/my_prepare/motion_template.pkl

Stage 2, label-AdaLN emotion branch::

    python train_two_stage.py --stage 2 --variant finalv1 \
        --stage1_checkpoint experiments/two_stage/finalv1_stage1/checkpoints/iter_0100000.pt

Stage 2, emotion-bank branch::

    python train_two_stage.py --stage 2 --variant finalv2 \
        --stage1_checkpoint experiments/two_stage/finalv2_stage1/checkpoints/iter_0100000.pt

Stage 2, hierarchical emotion2vec branch::

    python train_two_stage.py --stage 2 --variant finalv3 \
        --stage1_checkpoint experiments/two_stage/finalv3_stage1/checkpoints/iter_0100000.pt \
        --emotion2vec_root /data/MEAD/videos

Training policy
---------------
Stage 1:
    * train the generic audio feature mapping, motion/time backbone,
      original-audio cross-attention, generic start states and motion head;
    * bypass and freeze every emotion-related module;
    * use only diffusion/motion/head losses, because all data are treated as
      emotion-unlabeled.

Stage 2:
    * load the Stage-1 checkpoint with ``strict=False``;
    * freeze the audio encoder, audio feature projection, original-audio
      attention and motion/time backbone;
    * train the variant-specific emotion-audio encoder and zero-initialized
      emotion residual attention adapters;
    * optionally tune the last shared FFN layers and motion head at a smaller
      learning rate;
    * use the original motion losses plus frozen emotion-classifier losses;
      final-v3 can additionally use level and emotion2vec prosody losses.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

import src.utils as utils
from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel import EmoLevelDataset
from src.dataset.dataset_EmotionLevel_e2v import EmoLevelE2VDataset
from src.dataset.dataset_GenericTalkingMotion import GenericTalkingMotionDataset
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
from src.utils.e2v_losses import compute_prosody_curve_loss


MODEL_MODULES = {
    "finalv1": "src.modules.emotion_dit_finalv1_two_stage",
    "finalv2": "src.modules.emotion_dit_finalv2_two_stage",
    "finalv3": "src.modules.emotion_dit_finalv3_two_stage",
}


def setup_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def create_model(args, device):
    module = importlib.import_module(MODEL_MODULES[args.variant])
    return module.DitTalkingHead(
        device=str(device),
        target=args.target,
        architecture="decoder",
        motion_feat_dim=args.motion_feat_dim,
        fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        audio_model=args.audio_model,
        feature_dim=args.feature_dim,
        n_diff_steps=args.n_diff_steps,
        diff_schedule=args.diff_schedule,
        cfg_mode=args.cfg_mode,
        guiding_conditions="audio,emotion",
        emo_classes=args.emo_classes,
        e2v_dim=args.e2v_dim,
        num_emotion_tokens=args.num_emotion_tokens,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        align_mask_width=args.align_mask_width,
        decoder_dropout=args.decoder_dropout,
        audio_scale=args.audio_scale,
        emotion_scale_init=args.emotion_scale_init,
        emotion_audio_residual_init=args.emotion_audio_residual_init,
        use_indicator=args.use_indicator,
        use_learnable_pe=args.use_learnable_pe,
    )


def load_stage1_checkpoint(model, checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    logging.info(
        "Loaded Stage-1 checkpoint %s; missing=%d unexpected=%d",
        checkpoint_path,
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    if incompatible.unexpected_keys:
        logging.warning("Unexpected keys: %s", incompatible.unexpected_keys)
    return checkpoint


def make_datasets(args):
    if args.stage == 1:
        train_dataset = GenericTalkingMotionDataset(
            video_roots=args.generic_video_roots,
            motion_template_path=args.motion_template_path,
            aggregate_motion_files=args.aggregate_motion_files,
            split="train",
            split_file=args.generic_train_split,
            validation_ratio=args.generic_validation_ratio,
            split_seed=args.split_seed,
            coef_fps=args.fps,
            n_motions=args.n_motions,
            crop_strategy=args.crop_strategy,
            normalize_type=args.normalize_type,
            recursive=not args.no_recursive_scan,
            require_local_motion=args.require_local_motion,
        )
        val_dataset = GenericTalkingMotionDataset(
            video_roots=args.generic_video_roots,
            motion_template_path=args.motion_template_path,
            aggregate_motion_files=args.aggregate_motion_files,
            split="val",
            split_file=args.generic_val_split,
            validation_ratio=args.generic_validation_ratio,
            split_seed=args.split_seed,
            coef_fps=args.fps,
            n_motions=args.n_motions,
            crop_strategy="begin",
            normalize_type=args.normalize_type,
            recursive=not args.no_recursive_scan,
            require_local_motion=args.require_local_motion,
        )
        return train_dataset, val_dataset

    dataset_kwargs = dict(
        root_dir=args.emotion_prepare_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
    )
    if args.variant == "finalv3":
        train_dataset = EmoLevelE2VDataset(
            split="train",
            emotion2vec_root=args.emotion2vec_root,
            emotion2vec_dim=args.e2v_dim,
            **dataset_kwargs,
        )
        val_dataset = EmoLevelE2VDataset(
            split="test",
            emotion2vec_root=args.emotion2vec_root,
            emotion2vec_dim=args.e2v_dim,
            **dataset_kwargs,
        )
    else:
        train_dataset = EmoLevelDataset(split="train", **dataset_kwargs)
        val_dataset = EmoLevelDataset(split="test", **dataset_kwargs)
    return train_dataset, val_dataset


def make_loader(dataset, args, train: bool):
    return data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        persistent_workers=args.num_workers > 0,
    )


def make_indicator(batch_size, n_motions, end_idx, device):
    if end_idx is None:
        return torch.ones(
            batch_size, n_motions, dtype=torch.bool, device=device
        )
    return torch.arange(n_motions, device=device).expand(
        batch_size, -1
    ) < end_idx.unsqueeze(1)


def truncate_frame_feature(frame_feature, end_idx, pad_mode):
    result = frame_feature.clone()
    for batch_index in range(result.shape[0]):
        end = int(end_idx[batch_index].item())
        if pad_mode == "zero":
            result[batch_index, end:] = 0
        elif pad_mode == "replicate":
            source = max(0, end - 1)
            result[batch_index, end:] = result[batch_index, source]
        else:
            raise ValueError(f"Unknown pad_mode: {pad_mode}")
    return result


def unpack_batch(batch, args, device, predict_head_pose):
    if args.stage == 1:
        audio_pair, coefficient_pair, sample_name = batch
        emo_index = emo_level = None
        e2v_utt_pair = e2v_frame_pair = None
    elif args.variant == "finalv3":
        (
            audio_pair,
            coefficient_pair,
            emo_index,
            emo_level,
            e2v_utt_pair,
            e2v_frame_pair,
        ) = batch
        emo_index = emo_index.to(device)
        emo_level = emo_level.to(device)
        e2v_utt_pair = [item.to(device).float() for item in e2v_utt_pair]
        e2v_frame_pair = [item.to(device).float() for item in e2v_frame_pair]
        sample_name = None
    else:
        audio_pair, coefficient_pair, emo_index, emo_level = batch
        emo_index = emo_index.to(device)
        emo_level = emo_level.to(device)
        e2v_utt_pair = e2v_frame_pair = None
        sample_name = None

    audio_pair = [item.to(device).float() for item in audio_pair]
    coefficient_pair = [
        {
            key: value.to(device).float()
            for key, value in coefficient_pair[index].items()
        }
        for index in range(2)
    ]
    motion_pair = [
        utils.get_motion_coef(
            coefficient_pair[index],
            args.rot_repr,
            predict_head_pose,
        )
        for index in range(2)
    ]
    return {
        "audio_pair": audio_pair,
        "motion_pair": motion_pair,
        "emo_index": emo_index,
        "emo_level": emo_level,
        "e2v_utt_pair": e2v_utt_pair,
        "e2v_frame_pair": e2v_frame_pair,
        "sample_name": sample_name,
    }


def initialize_losses(device):
    names = (
        "noise", "emotion", "level", "prosody", "expression",
        "exp_velocity", "exp_smooth", "head_angle", "head_velocity",
        "head_smooth", "head_transition",
    )
    return {
        name: torch.tensor(0.0, device=device)
        for name in names
    }


def add_motion_losses(losses, pack, args, predict_head_pose):
    (
        loss_noise,
        loss_expression,
        loss_exp_velocity,
        loss_exp_smooth,
        loss_head_angle,
        loss_head_velocity,
        loss_head_smooth,
        loss_head_transition,
    ) = pack
    losses["noise"] += loss_noise / 2
    losses["expression"] += loss_expression / 2
    if loss_exp_velocity is not None:
        losses["exp_velocity"] += loss_exp_velocity / 2
    if loss_exp_smooth is not None:
        losses["exp_smooth"] += loss_exp_smooth / 2
    if args.target == "sample" and predict_head_pose:
        if loss_head_angle is not None:
            losses["head_angle"] += loss_head_angle / 2
        if loss_head_velocity is not None:
            losses["head_velocity"] += loss_head_velocity / 2
        if loss_head_smooth is not None:
            losses["head_smooth"] += loss_head_smooth / 2
        if loss_head_transition is not None:
            losses["head_transition"] += loss_head_transition


def total_loss(losses, args, predict_head_pose):
    result = losses["noise"]
    result = result + args.l_exp * losses["expression"]
    result = result + args.l_exp_vel * losses["exp_velocity"]
    result = result + args.l_exp_smooth * losses["exp_smooth"]
    if args.target == "sample" and predict_head_pose:
        result = result + args.l_head_angle * losses["head_angle"]
        result = result + args.l_head_vel * losses["head_velocity"]
        result = result + args.l_head_smooth * losses["head_smooth"]
        result = result + args.l_head_trans * losses["head_transition"]
    if args.stage == 2:
        result = result + args.l_emo_cls * losses["emotion"]
        result = result + args.l_emo_level * losses["level"]
        result = result + args.l_prosody_curve * losses["prosody"]
    return result


def run_batch(
    args,
    model,
    batch,
    classifier,
    training: bool,
):
    device = model.device
    predict_head_pose = not args.no_head_pose
    data_dict = unpack_batch(batch, args, device, predict_head_pose)
    audio_pair = data_dict["audio_pair"]
    motion_pair = data_dict["motion_pair"]
    emo_index = data_dict["emo_index"]
    emo_level = data_dict["emo_level"]
    e2v_utt_pair = data_dict["e2v_utt_pair"]
    e2v_frame_pair = data_dict["e2v_frame_pair"]
    audio_unit = 16000.0 / args.fps

    context_audio = None
    if args.use_context_audio_feat:
        context_audio = model.extract_audio_feature(
            torch.cat(audio_pair, dim=1), args.n_motions * 2
        )

    losses = initialize_losses(device)
    previous_motion = previous_audio = previous_frame = None
    cross_entropy = torch.nn.CrossEntropyLoss()

    for window_index in range(2):
        audio = audio_pair[window_index]
        motion = motion_pair[window_index]
        batch_size = audio.shape[0]
        frame_feature = (
            e2v_frame_pair[window_index]
            if e2v_frame_pair is not None else None
        )
        utterance_feature = (
            e2v_utt_pair[window_index]
            if e2v_utt_pair is not None else None
        )

        truncate_probability = (
            args.trunc_prob1 if window_index == 0 else args.trunc_prob2
        )
        should_truncate = training and np.random.rand() < truncate_probability
        if should_truncate:
            audio_input, motion_input, end_idx = (
                utils.truncate_motion_coef_and_audio(
                    audio,
                    motion,
                    args.n_motions,
                    audio_unit,
                    args.pad_mode,
                )
            )
            frame_input = (
                truncate_frame_feature(
                    frame_feature, end_idx, args.pad_mode
                )
                if frame_feature is not None else None
            )
            if args.use_context_audio_feat and window_index == 1:
                audio_input = model.extract_audio_feature(
                    torch.cat([audio_pair[0], audio_input], dim=1),
                    args.n_motions * 2,
                )[:, -args.n_motions:]
        else:
            audio_input = (
                context_audio[:,
                              window_index * args.n_motions:
                              (window_index + 1) * args.n_motions]
                if context_audio is not None else audio
            )
            motion_input = motion
            frame_input = frame_feature
            end_idx = None

        indicator = (
            make_indicator(
                batch_size, args.n_motions, end_idx, device
            )
            if args.use_indicator else None
        )

        model_output = model(
            motion_input,
            audio_input,
            prev_motion_feat=previous_motion,
            prev_audio_feat=previous_audio,
            indicator=indicator,
            emo_index=emo_index,
            emo_utt_feat=utterance_feature,
            emo_frame_feat=frame_input,
            prev_emo_frame_feat=previous_frame,
        )
        noise, prediction, current_motion, current_audio = model_output

        if window_index == 0:
            if end_idx is not None:
                previous_motion = motion[:, -args.n_prev_motions:].detach()
                if context_audio is not None:
                    previous_audio = context_audio[:,
                        args.n_motions - args.n_prev_motions:
                        args.n_motions].detach()
                else:
                    with torch.no_grad():
                        previous_audio = model.extract_audio_feature(audio)[
                            :, -args.n_prev_motions:
                        ].detach()
                previous_frame = (
                    frame_feature[:, -args.n_prev_motions:].detach()
                    if frame_feature is not None else None
                )
            else:
                previous_motion = current_motion[
                    :, -args.n_prev_motions:
                ].detach()
                previous_audio = current_audio[
                    :, -args.n_prev_motions:
                ].detach()
                previous_frame = (
                    frame_input[:, -args.n_prev_motions:].detach()
                    if frame_input is not None else None
                )

        pack = utils.compute_loss_new(
            args,
            window_index == 0,
            motion_input,
            noise,
            prediction,
            previous_motion,
            end_idx,
        )
        add_motion_losses(losses, pack, args, predict_head_pose)

        current_prediction = prediction[:, args.n_prev_motions:]
        if args.stage == 2 and classifier is not None:
            emotion_logits, level_logits = classifier(
                current_prediction[..., :63].clone()
            )
            losses["emotion"] += (
                cross_entropy(emotion_logits, emo_index) / 2
            )
            if args.l_emo_level > 0:
                losses["level"] += (
                    cross_entropy(level_logits, emo_level) / 2
                )

        if (
            args.stage == 2
            and args.variant == "finalv3"
            and args.l_prosody_curve > 0
        ):
            valid_mask = make_indicator(
                batch_size, args.n_motions, end_idx, device
            )
            losses["prosody"] += compute_prosody_curve_loss(
                current_prediction,
                frame_input,
                mask=valid_mask,
                rot_repr=args.rot_repr,
                use_velocity=not args.no_prosody_velocity,
                vel_weight_motion=args.prosody_motion_vel_weight,
                vel_weight_audio=args.prosody_audio_vel_weight,
            ) / 2

    return total_loss(losses, args, predict_head_pose), losses


def update_log(loss_log, loss, losses, args):
    loss_log["total"].append(loss.item())
    weights = {
        "noise": 1.0,
        "emotion": args.l_emo_cls if args.stage == 2 else 0.0,
        "level": args.l_emo_level if args.stage == 2 else 0.0,
        "prosody": args.l_prosody_curve if args.stage == 2 else 0.0,
        "expression": args.l_exp,
        "exp_velocity": args.l_exp_vel,
        "exp_smooth": args.l_exp_smooth,
        "head_angle": args.l_head_angle,
        "head_velocity": args.l_head_vel,
        "head_smooth": args.l_head_smooth,
        "head_transition": args.l_head_trans,
    }
    for name, value in losses.items():
        loss_log[name].append((weights[name] * value).item())


def describe(loss_log, prefix):
    names = [
        "total", "noise", "expression", "exp_velocity", "exp_smooth",
        "head_angle", "head_velocity", "head_smooth", "head_transition",
        "emotion", "level", "prosody",
    ]
    return prefix + " " + ", ".join(
        f"{name}={np.mean(loss_log[name]):.3e}"
        for name in names if loss_log[name]
    )


@torch.no_grad()
def validate(args, model, loader, classifier, writer, iteration):
    was_training = model.training
    model.eval()
    log = defaultdict(list)
    for batch_index, batch in enumerate(loader):
        loss, losses = run_batch(
            args, model, batch, classifier, training=False
        )
        update_log(log, loss, losses, args)
        if args.val_batches > 0 and batch_index + 1 >= args.val_batches:
            break
    logging.info(describe(log, f"Validation iter={iteration}"))
    if writer is not None:
        for name, values in log.items():
            if values:
                writer.add_scalar(
                    f"val/{name}", np.mean(values), iteration
                )
    if was_training:
        model.train()


def train(args, model, train_loader, val_loader, classifier, optimizer, scheduler, writer, output_dir):
    model.train()
    data_iterator = infinite_data_loader(train_loader)
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad(set_to_none=True)

    for iteration in range(1, args.max_iter + 1):
        batch = next(data_iterator)
        loss, losses = run_batch(
            args, model, batch, classifier, training=True
        )
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()

        if iteration % args.gradient_accumulation_steps == 0:
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters()
                     if parameter.requires_grad],
                    args.clip_grad,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        update_log(loss_log, loss, losses, args)
        if iteration % args.log_iter == 0:
            logging.info(describe(
                loss_log,
                f"Stage={args.stage} variant={args.variant} iter={iteration}",
            ))
            for name, values in loss_log.items():
                if values:
                    writer.add_scalar(
                        f"train/{name}", np.mean(values), iteration
                    )
            for group_index, group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    f"optimizer/lr_group_{group_index}",
                    group["lr"],
                    iteration,
                )

        if iteration % args.val_iter == 0 or iteration == 1:
            validate(
                args, model, val_loader, classifier, writer, iteration
            )

        if iteration % args.save_iter == 0 or iteration == args.max_iter:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "args": vars(args),
                "stage": args.stage,
                "variant": args.variant,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter": iteration,
                "trainable_report": model.trainable_parameter_report(),
            }, checkpoint_dir / f"iter_{iteration:07d}.pt")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    parser.add_argument(
        "--variant", required=True,
        choices=["finalv1", "finalv2", "finalv3"]
    )
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--exp_root", type=Path, default=Path("experiments/two_stage"))
    parser.add_argument("--exp_name", type=str, default=None)

    # Stage-1 universal dataset.
    parser.add_argument("--generic_video_roots", type=str, default=None)
    parser.add_argument("--aggregate_motion_files", type=str, default=None)
    parser.add_argument("--motion_template_path", type=str, default="src/my_prepare/motion_template.pkl")
    parser.add_argument("--generic_train_split", type=str, default=None)
    parser.add_argument("--generic_val_split", type=str, default=None)
    parser.add_argument("--generic_validation_ratio", type=float, default=0.05)
    parser.add_argument("--split_seed", type=int, default=2026)
    parser.add_argument("--no_recursive_scan", action="store_true")
    parser.add_argument("--require_local_motion", action="store_true")

    # Stage-2 MEAD dataset.
    parser.add_argument("--emotion_prepare_root", type=str, default="src/my_prepare/")
    parser.add_argument("--motion_filename", type=str, default="front_all_motions.pkl")
    parser.add_argument("--motion_template_filename", type=str, default="motion_template.pkl")
    parser.add_argument("--emotion2vec_root", type=str, default=None)
    parser.add_argument("--emotion_classifier_ckpt", type=str, default="pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth")
    parser.add_argument("--stage1_checkpoint", type=str, default=None)

    # Model.
    parser.add_argument("--target", choices=["sample", "noise"], default="sample")
    parser.add_argument("--motion_feat_dim", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--n_motions", type=int, default=100)
    parser.add_argument("--n_prev_motions", type=int, default=25)
    parser.add_argument("--audio_model", type=str, default="hubert_zh")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_diff_steps", type=int, default=50)
    parser.add_argument("--diff_schedule", type=str, default="cosine")
    parser.add_argument("--cfg_mode", type=str, default="incremental")
    parser.add_argument("--emo_classes", type=int, default=8)
    parser.add_argument("--e2v_dim", type=int, default=1024)
    parser.add_argument("--num_emotion_tokens", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--align_mask_width", type=int, default=3)
    parser.add_argument("--decoder_dropout", type=float, default=0.0)
    parser.add_argument("--audio_scale", type=float, default=1.0)
    parser.add_argument("--emotion_scale_init", type=float, default=0.10)
    parser.add_argument("--emotion_audio_residual_init", type=float, default=0.05)
    parser.add_argument("--use_indicator", action="store_true")
    parser.add_argument("--use_learnable_pe", action="store_true")

    # Stage-specific trainability.
    parser.add_argument("--train_audio_encoder_stage1", action="store_true")
    parser.add_argument("--stage2_tune_tail_layers", type=int, default=0)
    parser.add_argument("--stage2_tune_motion_head", action="store_true")
    parser.add_argument("--stage2_tail_lr_ratio", type=float, default=0.1)

    # Optimization.
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--clip_grad", type=float, default=2.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)

    # Existing motion losses.
    parser.add_argument("--criterion", choices=["l1", "l2"], default="l2")
    parser.add_argument("--l_exp", type=float, default=0.1)
    parser.add_argument("--l_exp_vel", type=float, default=1e-4)
    parser.add_argument("--l_exp_smooth", type=float, default=1e-4)
    parser.add_argument("--l_head_angle", type=float, default=1e-2)
    parser.add_argument("--l_head_vel", type=float, default=1e-2)
    parser.add_argument("--l_head_smooth", type=float, default=1e-2)
    parser.add_argument("--l_head_trans", type=float, default=1e-2)
    parser.add_argument("--no_constrain_prev", action="store_true")
    parser.add_argument("--no_head_pose", action="store_true")
    parser.add_argument("--rot_repr", type=str, default="aa", choices=["aa"])

    # Stage-2 emotion losses.
    parser.add_argument("--l_emo_cls", type=float, default=1.0)
    parser.add_argument("--l_emo_level", type=float, default=0.0)
    parser.add_argument("--l_prosody_curve", type=float, default=0.0)
    parser.add_argument("--no_prosody_velocity", action="store_true")
    parser.add_argument("--prosody_motion_vel_weight", type=float, default=0.5)
    parser.add_argument("--prosody_audio_vel_weight", type=float, default=0.5)

    # Window/data behavior.
    parser.add_argument("--crop_strategy", type=str, default="random")
    parser.add_argument("--normalize_type", type=str, default="mix")
    parser.add_argument("--pad_mode", choices=["zero", "replicate"], default="zero")
    parser.add_argument("--trunc_prob1", type=float, default=0.3)
    parser.add_argument("--trunc_prob2", type=float, default=0.4)
    parser.add_argument("--use_context_audio_feat", action="store_true")

    # Logging.
    parser.add_argument("--log_iter", type=int, default=10)
    parser.add_argument("--val_iter", type=int, default=100)
    parser.add_argument("--val_batches", type=int, default=20)
    parser.add_argument("--save_iter", type=int, default=1000)
    parser.add_argument("--log_smooth_win", type=int, default=50)
    return parser


def main(args):
    if args.stage == 1 and not args.generic_video_roots:
        raise ValueError("--generic_video_roots is required for Stage 1")
    if args.stage == 2 and not args.stage1_checkpoint:
        raise ValueError("--stage1_checkpoint is required for Stage 2")

    device = setup_device(args.device_id)
    exp_name = args.exp_name or f"{args.variant}_stage{args.stage}"
    output_dir = args.exp_root / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
        ],
    )
    writer = SummaryWriter(str(output_dir / "tensorboard"))

    model = create_model(args, device)
    if args.stage == 2:
        load_stage1_checkpoint(model, args.stage1_checkpoint, device)
        report = model.set_train_stage(
            2,
            train_audio_encoder=False,
            stage2_tune_tail_layers=args.stage2_tune_tail_layers,
            stage2_tune_motion_head=args.stage2_tune_motion_head,
        )
    else:
        report = model.set_train_stage(
            1,
            train_audio_encoder=args.train_audio_encoder_stage1,
        )
    logging.info("Trainable parameter report: %s", report)

    train_dataset, val_dataset = make_datasets(args)
    train_loader = make_loader(train_dataset, args, train=True)
    val_loader = make_loader(val_dataset, args, train=False)

    classifier = None
    if args.stage == 2:
        classifier = Classifier().to(device)
        classifier_state = torch.load(
            args.emotion_classifier_ckpt, map_location=device
        )
        classifier.load_state_dict(classifier_state, strict=False)
        classifier.eval()
        for parameter in classifier.parameters():
            parameter.requires_grad_(False)

    parameter_groups = model.optimizer_parameter_groups(
        learning_rate=args.lr,
        tail_lr_ratio=(
            args.stage2_tail_lr_ratio if args.stage == 2 else 1.0
        ),
    )
    if not parameter_groups:
        raise RuntimeError("No trainable parameters after stage configuration")
    optimizer = optim.AdamW(
        parameter_groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.max_iter // args.gradient_accumulation_steps),
            eta_min=args.lr * args.min_lr_ratio,
        )

    with open(output_dir / "options.txt", "w", encoding="utf-8") as handle:
        handle.write(utils.get_option_text(args, build_parser()))

    train(
        args,
        model,
        train_loader,
        val_loader,
        classifier,
        optimizer,
        scheduler,
        writer,
        output_dir,
    )


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())

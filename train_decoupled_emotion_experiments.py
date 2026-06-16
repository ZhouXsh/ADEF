# coding: utf-8
"""
Train entry for emotion/lip-sync ablations.

This file is intentionally additive: it does not replace ``train.py``. It is a
clean experiment runner for the question "how should emotion be injected without
hurting lip-sync?".

Main changes compared with train.py
-----------------------------------
1. Model selection is controlled by ``--experiment_variant`` so the ablation
   matrix is explicit and saved in TensorBoard/options.log.
2. The decoupled variants import
   ``src.modules.emotion_dit_decoupled_adapter.DitTalkingHead``. In that model,
   audio drives the inherited DiT, while emotion is added later by a zero-init
   motion-space residual adapter.
3. The first training stage can freeze the inherited audio DiT and train only the
   emotion adapter via ``--freeze_base_iters``. After that point, the script
   unfreezes the base and rebuilds the optimizer with a smaller base LR.
4. The old loss accumulation typo is fixed. The original code reused names like
   ``loss_exp = loss_exp + loss_exp / 2``; this file uses per-window loss names
   such as ``loss_exp_i`` and accumulates into ``loss_exp_total``.
5. Emotion classifier loss has an explicit ``--l_emo`` weight. The audio-only
   baseline keeps it at 0 by default, but still logs the emotion CE for analysis.
6. Checkpoint loading is split into ``--pretrained_ckpt`` for partial/strict=False
   initialization and ``--resume_ckpt`` for resuming an experiment.

Experiment matrix
-----------------
A_audio_only:
    Baseline. Uses ``emotion_dit_prev_modi`` with audio-only guidance.
    Recommended purpose: measure lip-sync upper bound.

B_current_modulation:
    Current emotion modulation baseline. Uses ``emotion_dit_prev_modi`` with
    ``audio,emotion`` guidance. Recommended purpose: reproduce the observed
    emotion/lip-sync conflict.

C_decoupled_no_protect:
    New decoupled adapter. Emotion is a late residual in motion space, no mouth
    protection mask. Recommended purpose: test whether decoupling alone helps.

D_decoupled_protected:
    New decoupled adapter with protected dims/keypoints. Recommended purpose:
    keep mouth/lip-sync-sensitive coefficients untouched or weakly touched.
    Set ``--emotion_protected_dims`` once you know the mouth-related coefficient
    indices. If you prefer LivePortrait keypoint granularity, set
    ``--emotion_protected_kp_indices``; each kp maps to three expression dims.

Example commands
----------------
# A. audio-only lip-sync baseline
python train_decoupled_emotion_experiments.py --experiment_variant A_audio_only --exp_name A_audio_only

# B. current modulation baseline
python train_decoupled_emotion_experiments.py --experiment_variant B_current_modulation --exp_name B_current_modulation

# C. decoupled adapter, train adapter-only for 10k iters, then joint tune
python train_decoupled_emotion_experiments.py --experiment_variant C_decoupled_no_protect --exp_name C_decoupled_no_protect --pretrained_ckpt path/to/audio_only.pt

# D. decoupled adapter with protected mouth dims/keypoints
python train_decoupled_emotion_experiments.py --experiment_variant D_decoupled_protected --exp_name D_decoupled_protected --emotion_protected_dims 0,1,2,3 --pretrained_ckpt path/to/audio_only.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
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
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier


EXPERIMENT_MATRIX: Dict[str, Dict[str, object]] = {
    "A_audio_only": {
        "model_variant": "prev_modi",
        "guiding_conditions": "audio",
        "l_emo": 0.0,
        "freeze_base_iters": 0,
        "description": "audio-only baseline; best lip-sync reference",
    },
    "B_current_modulation": {
        "model_variant": "prev_modi",
        "guiding_conditions": "audio,emotion",
        "l_emo": 1.0,
        "freeze_base_iters": 0,
        "description": "current emotion-as-audio-modulation baseline",
    },
    "C_decoupled_no_protect": {
        "model_variant": "decoupled_adapter",
        "guiding_conditions": "audio,emotion",
        "l_emo": 1.0,
        "freeze_base_iters": 10000,
        "emotion_protected_dims": "",
        "emotion_protected_kp_indices": "",
        "description": "decoupled motion residual adapter without mouth protection",
    },
    "D_decoupled_protected": {
        "model_variant": "decoupled_adapter",
        "guiding_conditions": "audio,emotion",
        "l_emo": 1.0,
        "freeze_base_iters": 10000,
        "description": "decoupled motion residual adapter with protected lip-sync dims",
    },
}


def set_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def select_model_class(model_variant: str):
    if model_variant == "prev_modi":
        from src.modules.emotion_dit_prev_modi import DitTalkingHead
    elif model_variant == "decoupled_adapter":
        from src.modules.emotion_dit_decoupled_adapter import DitTalkingHead
    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")
    return DitTalkingHead


def apply_experiment_preset(args: argparse.Namespace) -> argparse.Namespace:
    preset = EXPERIMENT_MATRIX[args.experiment_variant]
    args.model_variant = str(preset["model_variant"])

    if args.guiding_conditions is None:
        args.guiding_conditions = str(preset["guiding_conditions"])
    if args.l_emo is None:
        args.l_emo = float(preset["l_emo"])
    if args.freeze_base_iters is None:
        args.freeze_base_iters = int(preset["freeze_base_iters"])

    # For C, force no protection unless explicitly overridden after editing this file.
    # For D, keep user-provided protection values. If both are empty, D still runs
    # but is effectively the same mask as C; the warning is logged in main().
    if args.experiment_variant == "C_decoupled_no_protect":
        args.emotion_protected_dims = ""
        args.emotion_protected_kp_indices = ""

    if args.emotion_residual_scale is None:
        args.emotion_residual_scale = 0.25
    if args.emotion_dropout_prob is None:
        args.emotion_dropout_prob = 0.15
    if args.emotion_hidden_dim is None:
        args.emotion_hidden_dim = 512
    if args.emotion_pose_weight is None:
        args.emotion_pose_weight = 0.15
    if args.emotion_protected_weight is None:
        args.emotion_protected_weight = 0.0

    return args


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_norm_dict(device: torch.device):
    all_temp = pickle.load(open("pretrained_weights/ADEF/motion_template/motion_template.pkl", "rb"))
    mean_exp = torch.tensor(all_temp["mean_exp"]).to(device).unsqueeze(0).unsqueeze(0)
    std_exp = torch.tensor(all_temp["std_exp"]).to(device).unsqueeze(0).unsqueeze(0)

    alone_temp = pickle.load(open("pretrained_weights/ADEF/motion_template/emotion_template.pkl", "rb"))
    mean_exps, std_exps = [], []
    for i in range(len(alone_temp)):
        mean_exps.append(torch.tensor(alone_temp[i]["mean_exp"]))
        std_exps.append(torch.tensor(alone_temp[i]["std_exp"]))

    return {
        "mean_exp": mean_exp,
        "std_exp": std_exp,
        "mean_exps": torch.stack(mean_exps, dim=0).to(device),
        "std_exps": torch.stack(std_exps, dim=0).to(device),
    }


def make_model_kwargs(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    kwargs: Dict[str, object] = dict(
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
    )

    if args.model_variant == "decoupled_adapter":
        kwargs.update(
            emotion_dropout_prob=args.emotion_dropout_prob,
            emotion_residual_scale=args.emotion_residual_scale,
            emotion_hidden_dim=args.emotion_hidden_dim,
            emotion_pose_weight=args.emotion_pose_weight,
            emotion_protected_dims=args.emotion_protected_dims,
            emotion_protected_kp_indices=args.emotion_protected_kp_indices,
            emotion_protected_weight=args.emotion_protected_weight,
            base_start_emotion_id=args.base_start_emotion_id,
        )
    return kwargs


def load_checkpoint_if_needed(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> int:
    start_iter = 0

    if args.resume_ckpt:
        ckpt = torch.load(args.resume_ckpt, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        model.load_state_dict(state_dict, strict=False)
        start_iter = int(ckpt.get("iter", -1)) + 1
        logging.info(f"Resume from {args.resume_ckpt}; start_iter={start_iter}")
        return start_iter

    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location=device)
        state_dict = ckpt.get("model", ckpt)
        incompatible = model.load_state_dict(state_dict, strict=False)
        logging.info(f"Load pretrained checkpoint from {args.pretrained_ckpt} with strict=False")
        logging.info(f"Missing keys: {len(incompatible.missing_keys)}; unexpected keys: {len(incompatible.unexpected_keys)}")

    return start_iter


def split_param_groups(args: argparse.Namespace, model: torch.nn.Module):
    adapter_params = []
    base_params = []
    adapter_prefixes = ("decoupled_", "emotion_adapter")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if args.model_variant == "decoupled_adapter" and name.startswith(adapter_prefixes):
            adapter_params.append(param)
        else:
            base_params.append(param)

    groups = []
    if base_params:
        groups.append({"params": base_params, "lr": args.lr * args.base_lr_ratio, "name": "base"})
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.lr, "name": "emotion_adapter"})
    if not groups:
        raise RuntimeError("No trainable parameters found.")
    return groups


def build_optimizer_and_scheduler(args: argparse.Namespace, model: torch.nn.Module):
    optimizer = torch.optim.Adam(split_param_groups(args, model), lr=args.lr)

    if args.scheduler == "Warmup":
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == "WarmupThenDecay":
        from src.scheduler import GradualWarmupScheduler
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            max(1, args.cos_max_iter - args.warm_iter),
            args.lr * args.min_lr_ratio,
        )
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
    else:
        scheduler = None

    return optimizer, scheduler


def compute_emotion_ce(args, classifier, target, emo_index, norm_dict, criterion):
    if classifier is None:
        return None

    mean_exp = norm_dict["mean_exp"]
    std_exp = norm_dict["std_exp"]
    mean_exps = norm_dict["mean_exps"]
    std_exps = norm_dict["std_exps"]

    exps = target[:, args.n_prev_motions:, :63].clone()
    alone_mean = mean_exps[emo_index].unsqueeze(1)
    alone_std = std_exps[emo_index].unsqueeze(1)
    exps = (exps * alone_std + alone_mean - mean_exp) / (std_exp + 1e-9)
    pred_emo, _ = classifier(exps)
    return criterion(pred_emo, emo_index)


def maybe_unfreeze_base(args, model, optimizer, scheduler, it: int):
    if args.model_variant != "decoupled_adapter":
        return optimizer, scheduler
    if args.freeze_base_iters <= 0:
        return optimizer, scheduler
    if it != args.freeze_base_iters:
        return optimizer, scheduler
    if not hasattr(model, "set_base_trainable"):
        return optimizer, scheduler

    logging.info(f"Unfreeze base DiT at iter {it}; rebuild optimizer/scheduler.")
    model.set_base_trainable(True)
    optimizer, scheduler = build_optimizer_and_scheduler(args, model)
    optimizer.zero_grad(set_to_none=True)
    logging.info(f"Trainable params after unfreeze: {count_parameters(model)}")
    return optimizer, scheduler


def run_one_split(args, model, audio_pair, coef_pair, emo_index, audio_unit, predict_head_pose, classifier, norm_dict, criterion, is_train):
    device = model.device
    audio_pair = [audio.to(device) for audio in audio_pair]
    coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
    motion_coef_pair = [utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)]
    emo_index = emo_index.to(device)

    # A_audio_only should not receive the real emotion id even through the
    # inherited emotion-specific start tokens. The neutral class id is 5 in
    # src/config/emotion_config.py. B uses the real emotion id; C/D pass the
    # real id to the adapter while the adapter keeps its base path neutral.
    if args.experiment_variant == "A_audio_only":
        model_emo_index = torch.full_like(emo_index, int(args.base_start_emotion_id))
    else:
        model_emo_index = emo_index

    if args.use_context_audio_feat:
        audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)
    else:
        audio_feat = None

    loss_noise_total = torch.tensor(0.0, device=device)
    loss_emo_total = torch.tensor(0.0, device=device)
    loss_exp_total = torch.tensor(0.0, device=device)
    loss_exp_v_total = torch.tensor(0.0, device=device)
    loss_exp_s_total = torch.tensor(0.0, device=device)
    loss_head_angle_total = torch.tensor(0.0, device=device)
    loss_head_vel_total = torch.tensor(0.0, device=device)
    loss_head_smooth_total = torch.tensor(0.0, device=device)
    loss_head_trans_total = torch.tensor(0.0, device=device)

    prev_motion_coef = None
    prev_audio_feat = None

    for i in range(2):
        audio = audio_pair[i]
        motion_coef = motion_coef_pair[i]
        batch_size = audio.shape[0]

        do_trunc = (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2)
        if do_trunc:
            audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                audio, motion_coef, args.n_motions, audio_unit, args.pad_mode
            )
            if args.use_context_audio_feat and i != 0:
                audio_in = model.extract_audio_feature(
                    torch.cat([audio_pair[i - 1], audio_in], dim=1), args.n_motions * 2
                )[:, -args.n_motions:]
        else:
            if args.use_context_audio_feat:
                audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
            else:
                audio_in = audio
            motion_coef_in, end_idx = motion_coef, None

        if args.use_indicator:
            if end_idx is not None:
                indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(1)
            else:
                indicator = torch.ones(batch_size, args.n_motions, device=device)
        else:
            indicator = None

        if i == 0:
            noise, target, prev_motion_pred, prev_audio_pred = model(
                motion_coef_in, audio_in, indicator=indicator, emo_index=model_emo_index
            )
            if end_idx is not None:
                prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                if args.use_context_audio_feat:
                    prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                else:
                    with torch.no_grad():
                        prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
            else:
                prev_motion_coef = prev_motion_pred[:, -args.n_prev_motions:]
                prev_audio_feat = prev_audio_pred[:, -args.n_prev_motions:]
        else:
            noise, target, _, _ = model(
                motion_coef_in,
                audio_in,
                prev_motion_coef,
                prev_audio_feat,
                indicator=indicator,
                emo_index=model_emo_index,
            )

        loss_n_i, loss_exp_i, loss_exp_v_i, loss_exp_s_i, loss_ha_i, loss_hc_i, loss_hs_i, loss_ht_i = utils.compute_loss_new(
            args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx
        )

        loss_noise_total = loss_noise_total + loss_n_i / 2
        loss_exp_total = loss_exp_total + loss_exp_i / 2
        loss_exp_v_total = loss_exp_v_total + loss_exp_v_i / 2
        loss_exp_s_total = loss_exp_s_total + loss_exp_s_i / 2

        loss_e_i = compute_emotion_ce(args, classifier, target, emo_index, norm_dict, criterion)
        if loss_e_i is not None:
            loss_emo_total = loss_emo_total + loss_e_i / 2

        if args.target == "sample" and predict_head_pose and args.l_head_angle > 0:
            loss_head_angle_total = loss_head_angle_total + loss_ha_i / 2
        if args.target == "sample" and predict_head_pose and args.l_head_vel > 0 and loss_hc_i is not None:
            loss_head_vel_total = loss_head_vel_total + loss_hc_i / 2
        if args.target == "sample" and predict_head_pose and args.l_head_smooth > 0 and loss_hs_i is not None:
            loss_head_smooth_total = loss_head_smooth_total + loss_hs_i / 2
        if args.target == "sample" and predict_head_pose and args.l_head_trans > 0 and loss_ht_i is not None:
            loss_head_trans_total = loss_head_trans_total + loss_ht_i

    losses = {
        "noise": loss_noise_total,
        "emo": loss_emo_total,
        "exp": loss_exp_total,
        "exp_vel": loss_exp_v_total,
        "exp_smooth": loss_exp_s_total,
        "head_angle": loss_head_angle_total,
        "head_vel": loss_head_vel_total,
        "head_smooth": loss_head_smooth_total,
        "head_trans": loss_head_trans_total,
    }

    loss = losses["noise"]
    loss = loss + args.l_emo * losses["emo"]
    loss = loss + args.l_exp * losses["exp"]
    loss = loss + args.l_exp_vel * losses["exp_vel"]
    loss = loss + args.l_exp_smooth * losses["exp_smooth"]

    if args.target == "sample" and predict_head_pose and args.l_head_angle > 0:
        loss = loss + args.l_head_angle * losses["head_angle"]
    if args.target == "sample" and predict_head_pose and args.l_head_vel > 0:
        loss = loss + args.l_head_vel * losses["head_vel"]
    if args.target == "sample" and predict_head_pose and args.l_head_smooth > 0:
        loss = loss + args.l_head_smooth * losses["head_smooth"]
    if args.target == "sample" and predict_head_pose and args.l_head_trans > 0:
        loss = loss + args.l_head_trans * losses["head_trans"]

    losses["loss"] = loss
    return losses


def log_losses(prefix: str, loss_log, writer: Optional[SummaryWriter], step: int, args, predict_head_pose: bool):
    description = f"Iter: {step}\t{prefix} loss: [N: {np.mean(loss_log['noise']):.3e}"
    description += f", Emo: {np.mean(loss_log['emo']):.3e}"
    description += f", EX: {np.mean(loss_log['exp']):.3e}"
    description += f", EX_V: {np.mean(loss_log['exp_vel']):.3e}"
    description += f", EX_S: {np.mean(loss_log['exp_smooth']):.3e}"
    if args.target == "sample" and predict_head_pose and args.l_head_angle > 0:
        description += f", HA: {np.mean(loss_log['head_angle']):.3e}"
    if args.target == "sample" and predict_head_pose and args.l_head_vel > 0:
        description += f", HV: {np.mean(loss_log['head_vel']):.3e}"
    if args.target == "sample" and predict_head_pose and args.l_head_smooth > 0:
        description += f", HS: {np.mean(loss_log['head_smooth']):.3e}"
    if args.target == "sample" and predict_head_pose and args.l_head_trans > 0:
        description += f", HT: {np.mean(loss_log['head_trans']):.3e}"
    description += "]"
    logging.info(description)

    if writer is None:
        return
    writer.add_scalar(f"{prefix}/total_loss", np.mean(loss_log["loss"]), step)
    writer.add_scalar(f"{prefix}/simple_loss", np.mean(loss_log["noise"]), step)
    writer.add_scalar(f"{prefix}/emotion_ce_unweighted", np.mean(loss_log["emo"]), step)
    writer.add_scalar(f"{prefix}/exp_loss_weighted", np.mean(loss_log["exp"]), step)
    writer.add_scalar(f"{prefix}/exp_vel_loss_weighted", np.mean(loss_log["exp_vel"]), step)
    writer.add_scalar(f"{prefix}/exp_smooth_loss_weighted", np.mean(loss_log["exp_smooth"]), step)


def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    device = model.device
    model.train()

    norm_dict = load_norm_dict(device)
    criterion = torch.nn.CrossEntropyLoss()
    data_loader = infinite_data_loader(train_loader)
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))

    optimizer.zero_grad(set_to_none=True)
    for it in range(start_iter, args.max_iter + 1):
        optimizer, scheduler = maybe_unfreeze_base(args, model, optimizer, scheduler, it)

        audio_pair, coef_pair, emo_index, _ = next(data_loader)
        losses = run_one_split(
            args,
            model,
            audio_pair,
            coef_pair,
            emo_index,
            audio_unit,
            predict_head_pose,
            classifier,
            norm_dict,
            criterion,
            is_train=True,
        )

        losses["loss"].backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        weighted_values = {
            "loss": losses["loss"].item(),
            "noise": losses["noise"].item(),
            "emo": losses["emo"].item() * args.l_emo,
            "exp": losses["exp"].item() * args.l_exp,
            "exp_vel": losses["exp_vel"].item() * args.l_exp_vel,
            "exp_smooth": losses["exp_smooth"].item() * args.l_exp_smooth,
            "head_angle": losses["head_angle"].item() * args.l_head_angle,
            "head_vel": losses["head_vel"].item() * args.l_head_vel,
            "head_smooth": losses["head_smooth"].item() * args.l_head_smooth,
            "head_trans": losses["head_trans"].item() * args.l_head_trans,
        }
        for key, value in weighted_values.items():
            loss_log[key].append(value)

        if it % args.log_iter == 0:
            log_losses("train", loss_log, writer, it, args, predict_head_pose)
            if writer is not None:
                for idx, group in enumerate(optimizer.param_groups):
                    group_name = group.get("name", f"group_{idx}")
                    writer.add_scalar(f"opt/lr_{group_name}", group["lr"], it)

        if scheduler is not None:
            if args.scheduler != "WarmupThenDecay" or it < args.cos_max_iter:
                scheduler.step()

        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save(
                {
                    "args": args,
                    "model": model.state_dict(),
                    "iter": it,
                    "experiment_variant": args.experiment_variant,
                    "model_variant": args.model_variant,
                },
                save_dir / f"iter_{it:07}.pt",
            )

        if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:
            val(args, model, val_loader, it, 1, "val", writer, norm_dict, classifier)


@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode="val", writer=None, norm_dict=None, classifier=None):
    is_training = model.training
    model.eval()
    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    criterion = torch.nn.CrossEntropyLoss()
    loss_log = defaultdict(list)

    for _ in range(n_rounds):
        for audio_pair, coef_pair, emo_index, _ in test_loader:
            losses = run_one_split(
                args,
                model,
                audio_pair,
                coef_pair,
                emo_index,
                audio_unit,
                predict_head_pose,
                classifier,
                norm_dict,
                criterion,
                is_train=False,
            )
            loss_log["loss"].append(losses["loss"].item())
            loss_log["noise"].append(losses["noise"].item())
            loss_log["emo"].append(losses["emo"].item() * args.l_emo)
            loss_log["exp"].append(losses["exp"].item() * args.l_exp)
            loss_log["exp_vel"].append(losses["exp_vel"].item() * args.l_exp_vel)
            loss_log["exp_smooth"].append(losses["exp_smooth"].item() * args.l_exp_smooth)
            loss_log["head_angle"].append(losses["head_angle"].item() * args.l_head_angle)
            loss_log["head_vel"].append(losses["head_vel"].item() * args.l_head_vel)
            loss_log["head_smooth"].append(losses["head_smooth"].item() * args.l_head_smooth)
            loss_log["head_trans"].append(losses["head_trans"].item() * args.l_head_trans)

    log_losses(mode, loss_log, writer, current_iter, args, predict_head_pose)
    if is_training:
        model.train()


def main(args, option_text=None):
    args = apply_experiment_preset(args)
    device = set_device(args.device_id)

    exp_dir = Path(args.exp_root) / args.exp_name
    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), "log.txt"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / "options.log", "w") as f:
            f.write(option_text)
        writer.add_text("options", option_text)

    logging.info(f"experiment_variant: {args.experiment_variant}")
    logging.info(f"matrix_description: {EXPERIMENT_MATRIX[args.experiment_variant]['description']}")
    logging.info(f"model_variant: {args.model_variant}")
    logging.info(f"guiding_conditions: {args.guiding_conditions}")
    logging.info(f"l_emo: {args.l_emo}")

    if args.experiment_variant == "D_decoupled_protected" and not args.emotion_protected_dims and not args.emotion_protected_kp_indices:
        logging.warning(
            "D_decoupled_protected selected but no protected dims/keypoints were provided. "
            "It will behave like C except for the experiment name."
        )

    DitTalkingHead = select_model_class(args.model_variant)
    model = DitTalkingHead(**make_model_kwargs(args, device))

    start_iter = load_checkpoint_if_needed(args, model, device)

    if args.model_variant == "decoupled_adapter" and hasattr(model, "set_base_trainable"):
        if args.freeze_base_iters > 0 and start_iter < args.freeze_base_iters:
            logging.info(f"Freeze inherited base DiT until iter {args.freeze_base_iters}")
            model.set_base_trainable(False)
        else:
            model.set_base_trainable(True)

    logging.info(f"model parameters total: {count_parameters(model, trainable_only=False)}")
    logging.info(f"model parameters trainable: {count_parameters(model, trainable_only=True)}")

    train_dataset = EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split="train",
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
    )
    val_dataset = EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split="val",
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classifier = Classifier().to(device)
    classifier.load_state_dict(
        torch.load("pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth", map_location=device),
        strict=False,
    )
    classifier.eval()

    optimizer, scheduler = build_optimizer_and_scheduler(args, model)

    train(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        exp_dir / "checkpoints",
        scheduler,
        writer,
        start_iter=start_iter,
        classifier=classifier,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train"])
    parser.add_argument("--experiment_variant", type=str, default="C_decoupled_no_protect", choices=list(EXPERIMENT_MATRIX.keys()))
    parser.add_argument("--exp_name", type=str, default="decoupled_emotion_experiment")
    parser.add_argument("--exp_root", type=Path, default="experiments/emo_dit")
    parser.add_argument("--device_id", type=int, default=0)

    # Dataset
    parser.add_argument("--data_root", type=Path, default="src/my_prepare/")
    parser.add_argument("--motion_filename", type=str, default="front_all_motions.pkl")
    parser.add_argument("--motion_template_filename", type=str, default="motion_template.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_strategy", type=str, default="random")
    parser.add_argument("--normalize_type", type=str, default="mix", choices=["std", "case", "scale", "minmax", "mix"])

    # Model
    parser.add_argument("--target", type=str, default="sample", choices=["sample", "noise"])
    parser.add_argument("--guiding_conditions", type=str, default=None)
    parser.add_argument("--cfg_mode", type=str, default="incremental", choices=["incremental", "independent"])
    parser.add_argument("--n_diff_steps", type=int, default=50)
    parser.add_argument("--diff_schedule", type=str, default="cosine", choices=["linear", "cosine", "quadratic", "sigmoid"])
    parser.add_argument("--no_head_pose", action="store_true", default=False)
    parser.add_argument("--rot_repr", type=str, default="aa", choices=["aa"])

    # Transformer/base DiT
    parser.add_argument("--audio_model", type=str, default="wav2vec2", choices=["wav2vec2", "hubert", "hubert_zh", "hubert_zh_ori"])
    parser.add_argument("--architecture", type=str, default="decoder", choices=["decoder"])
    parser.add_argument("--use_indicator", action="store_true", default=True)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_motions", type=int, default=100)
    parser.add_argument("--n_prev_motions", type=int, default=25)
    parser.add_argument("--motion_feat_dim", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "replicate"])

    # Decoupled adapter
    parser.add_argument("--emotion_residual_scale", type=float, default=None)
    parser.add_argument("--emotion_dropout_prob", type=float, default=None)
    parser.add_argument("--emotion_hidden_dim", type=int, default=None)
    parser.add_argument("--emotion_pose_weight", type=float, default=None)
    parser.add_argument("--emotion_protected_dims", type=str, default="")
    parser.add_argument("--emotion_protected_kp_indices", type=str, default="")
    parser.add_argument("--emotion_protected_weight", type=float, default=None)
    parser.add_argument(
        "--base_start_emotion_id",
        type=int,
        default=5,
        help="Neutral emotion id used by inherited base DiT start tokens in decoupled variants; 5 matches the project neutral class.",
    )

    # Checkpoints
    parser.add_argument("--pretrained_ckpt", type=str, default="", help="partial strict=False initialization, e.g. audio-only checkpoint")
    parser.add_argument("--resume_ckpt", type=str, default="", help="resume exact experiment checkpoint")

    # Training
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_lr_ratio", type=float, default=0.1, help="base DiT LR = lr * base_lr_ratio after unfreezing")
    parser.add_argument("--freeze_base_iters", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--scheduler", type=str, default="WarmupThenDecay", choices=["None", "Warmup", "WarmupThenDecay"])

    # Losses
    parser.add_argument("--criterion", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--clip_grad", default=True, action="store_true")
    parser.add_argument("--l_emo", type=float, default=None)
    parser.add_argument("--l_exp", type=float, default=0.1)
    parser.add_argument("--l_exp_vel", type=float, default=1e-4)
    parser.add_argument("--l_exp_smooth", type=float, default=1e-4)
    parser.add_argument("--l_head_angle", type=float, default=1e-2)
    parser.add_argument("--l_head_vel", type=float, default=1e-2)
    parser.add_argument("--l_head_smooth", type=float, default=1e-2)
    parser.add_argument("--l_head_trans", type=float, default=1e-2)
    parser.add_argument("--no_constrain_prev", action="store_true")

    parser.add_argument("--use_context_audio_feat", action="store_true")
    parser.add_argument("--trunc_prob1", type=float, default=0.3)
    parser.add_argument("--trunc_prob2", type=float, default=0.4)

    parser.add_argument("--save_iter", type=int, default=1000)
    parser.add_argument("--val_iter", type=int, default=50)
    parser.add_argument("--log_iter", type=int, default=50)
    parser.add_argument("--log_smooth_win", type=int, default=50)

    # Warm-up / decay
    parser.add_argument("--warm_iter", type=int, default=10000)
    parser.add_argument("--cos_max_iter", type=int, default=100000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args = apply_experiment_preset(args)
    option_text = utils.common.get_option_text(args, parser)
    main(args, option_text)

# coding: utf-8
"""Train finalv3 emotion-audio DiT.

This file trains ``src.modules.emotion_dit_finalv3.DitTalkingHead`` with the
emotion2vec-aware dataset.  It is designed as a copy-style training entrypoint
and does not modify the original ``train.py``.

Important interface difference from ``emotion_dit_e2v.py``:
    finalv3.forward(...) returns exactly four values:
        noise, target, prev_motion_coef_or_target, prev_audio_feat

It does NOT return ``prev_emo_frame_feat``.  Therefore, this training script
derives ``prev_emo_frame_feat`` directly from the previous window's
frame-level emotion2vec features.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

import src.utils as utils
from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel_e2v import EmoLevelE2VDataset
from src.modules.emotion_dit_finalv3 import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier


cross_criterion = torch.nn.CrossEntropyLoss()


def setup_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def truncate_e2v_frame_feature(frame_feat: torch.Tensor, end_idx: torch.Tensor, pad_mode: str = "zero") -> torch.Tensor:
    """Apply the same random truncation policy to frame-level emotion2vec features.

    Args:
        frame_feat: [B, L, D_e2v].
        end_idx: [B], valid end frame for each sample.
        pad_mode: "zero" or "replicate".
    """
    out = frame_feat.clone()
    batch_size = frame_feat.shape[0]

    if pad_mode == "zero":
        for b in range(batch_size):
            out[b, end_idx[b]:] = 0
    elif pad_mode == "replicate":
        for b in range(batch_size):
            out[b, end_idx[b]:] = out[b, end_idx[b] - 1]
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")

    return out


def make_valid_mask(batch_size: int, n_motions: int, end_idx, device) -> torch.Tensor:
    """Build [B, L] valid-frame mask used by indicator and optional prosody loss."""
    if end_idx is None:
        return torch.ones(batch_size, n_motions, dtype=torch.bool, device=device)
    return torch.arange(n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(1)


def add_weighted(loss: torch.Tensor, weight: float, value) -> torch.Tensor:
    if value is None or weight <= 0:
        return loss
    return loss + weight * value


def batch_to_device(batch, device, predict_head_pose: bool, args):
    audio_pair, coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair = batch

    audio_pair = [audio.to(device) for audio in audio_pair]
    coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
    motion_coef_pair = [
        utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose)
        for i in range(2)
    ]

    emo_index = emo_index.to(device)
    emo_level = emo_level.to(device)
    e2v_utt_pair = [x.to(device).float() for x in e2v_utt_pair]
    e2v_frame_pair = [x.to(device).float() for x in e2v_frame_pair]

    return audio_pair, motion_coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair


def prepare_window_inputs(
    args,
    model,
    i: int,
    audio_pair,
    motion_coef_pair,
    e2v_frame_pair,
    audio_unit: float,
    audio_feat=None,
):
    """Prepare current audio/motion/e2v-frame window, including truncation."""
    audio = audio_pair[i]
    motion_coef = motion_coef_pair[i]
    e2v_frame = e2v_frame_pair[i]

    should_truncate = (
        (i == 0 and np.random.rand() < args.trunc_prob1)
        or (i != 0 and np.random.rand() < args.trunc_prob2)
    )

    if should_truncate:
        audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
            audio, motion_coef, args.n_motions, audio_unit, args.pad_mode
        )
        e2v_frame_in = truncate_e2v_frame_feature(e2v_frame, end_idx, args.pad_mode)

        if args.use_context_audio_feat and i != 0:
            audio_in = model.extract_audio_feature(
                torch.cat([audio_pair[i - 1], audio_in], dim=1),
                args.n_motions * 2,
            )[:, -args.n_motions:]
    else:
        if args.use_context_audio_feat:
            audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
        else:
            audio_in = audio
        motion_coef_in, end_idx = motion_coef, None
        e2v_frame_in = e2v_frame

    return audio_in, motion_coef_in, e2v_frame_in, end_idx


def initialize_loss_tensors(device):
    return {
        "noise": torch.tensor(0.0, device=device),
        "emo": torch.tensor(0.0, device=device),
        "level": torch.tensor(0.0, device=device),
        "prosody": torch.tensor(0.0, device=device),
        "exp": torch.tensor(0.0, device=device),
        "exp_vel": torch.tensor(0.0, device=device),
        "exp_smooth": torch.tensor(0.0, device=device),
        "head_angle": torch.tensor(0.0, device=device),
        "head_vel": torch.tensor(0.0, device=device),
        "head_smooth": torch.tensor(0.0, device=device),
        "head_trans": torch.tensor(0.0, device=device),
    }


def accumulate_original_losses(losses, args, predict_head_pose, loss_pack):
    loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hv, loss_hs, loss_ht = loss_pack

    losses["noise"] = losses["noise"] + loss_n / 2
    losses["exp"] = losses["exp"] + loss_exp / 2

    if loss_exp_v is not None:
        losses["exp_vel"] = losses["exp_vel"] + loss_exp_v / 2
    if loss_exp_s is not None:
        losses["exp_smooth"] = losses["exp_smooth"] + loss_exp_s / 2

    if args.target == "sample" and predict_head_pose and args.l_head_angle > 0:
        losses["head_angle"] = losses["head_angle"] + loss_ha / 2
    if args.target == "sample" and predict_head_pose and args.l_head_vel > 0 and loss_hv is not None:
        losses["head_vel"] = losses["head_vel"] + loss_hv / 2
    if args.target == "sample" and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
        losses["head_smooth"] = losses["head_smooth"] + loss_hs / 2
    if args.target == "sample" and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
        # Same as original train.py: head transition loss only applies to the
        # second window, so it is not divided by 2.
        losses["head_trans"] = losses["head_trans"] + loss_ht

    return losses


def build_original_total_loss(args, losses, predict_head_pose: bool):
    """Exactly follow the original train.py loss combination.

    Original train.py:
        loss = noise + emo
             + l_exp * exp
             + l_exp_vel * exp_vel
             + l_exp_smooth * exp_smooth
             + l_head_* * head_*
    """
    loss = losses["noise"]
    loss = loss + losses["emo"]
    loss = loss + args.l_exp * losses["exp"]
    loss = loss + args.l_exp_vel * losses["exp_vel"]
    loss = loss + args.l_exp_smooth * losses["exp_smooth"]

    if args.target == "sample" and predict_head_pose:
        loss = add_weighted(loss, args.l_head_angle, losses["head_angle"])
        loss = add_weighted(loss, args.l_head_vel, losses["head_vel"])
        loss = add_weighted(loss, args.l_head_smooth, losses["head_smooth"])
        loss = add_weighted(loss, args.l_head_trans, losses["head_trans"])

    return loss


def append_original_logs(args, loss_log, losses, predict_head_pose: bool):
    loss_log["noise"].append(losses["noise"].item())
    loss_log["emo"].append(losses["emo"].item())
    loss_log["exp"].append((args.l_exp * losses["exp"]).item())
    loss_log["exp_vel"].append((args.l_exp_vel * losses["exp_vel"]).item())
    loss_log["exp_smooth"].append((args.l_exp_smooth * losses["exp_smooth"]).item())

    if args.target == "sample" and predict_head_pose:
        loss_log["head_angle"].append((args.l_head_angle * losses["head_angle"]).item())
        loss_log["head_vel"].append((args.l_head_vel * losses["head_vel"]).item())
        loss_log["head_smooth"].append((args.l_head_smooth * losses["head_smooth"]).item())
        loss_log["head_trans"].append((args.l_head_trans * losses["head_trans"]).item())


def write_original_scalars(writer, prefix: str, loss_log, current_iter: int):
    writer.add_scalar(f"{prefix}/simple_loss", np.mean(loss_log["noise"]), current_iter)
    writer.add_scalar(f"{prefix}/emotion_loss", np.mean(loss_log["emo"]), current_iter)
    writer.add_scalar(f"{prefix}/exp_loss", np.mean(loss_log["exp"]), current_iter)
    writer.add_scalar(f"{prefix}/exp_vel_loss", np.mean(loss_log["exp_vel"]), current_iter)
    writer.add_scalar(f"{prefix}/exp_smooth_loss", np.mean(loss_log["exp_smooth"]), current_iter)
    writer.add_scalar(f"{prefix}/head_angle_loss", np.mean(loss_log["head_angle"]), current_iter)
    writer.add_scalar(f"{prefix}/head_vel_loss", np.mean(loss_log["head_vel"]), current_iter)
    writer.add_scalar(f"{prefix}/head_smooth_loss", np.mean(loss_log["head_smooth"]), current_iter)
    writer.add_scalar(f"{prefix}/head_trans_loss", np.mean(loss_log["head_trans"]), current_iter)


def describe_original_losses(loss_log, prefix: str, current_iter=None):
    if current_iter is None:
        msg = f"{prefix} loss: "
    else:
        msg = f"(Iter {current_iter:>6}) {prefix} loss: "

    msg += (
        f"[N: {np.mean(loss_log['noise']):.3e}, "
        f"Emo: {np.mean(loss_log['emo']):.3e}, "
        f"EX: {np.mean(loss_log['exp']):.3e}, "
        f"EX_V: {np.mean(loss_log['exp_vel']):.3e}, "
        f"EX_S: {np.mean(loss_log['exp_smooth']):.3e}, "
        f"HA: {np.mean(loss_log['head_angle']):.3e}, "
        f"HV: {np.mean(loss_log['head_vel']):.3e}, "
        f"HS: {np.mean(loss_log['head_smooth']):.3e}, "
        f"HT: {np.mean(loss_log['head_trans']):.3e}]"
    )
    return msg


def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None):
    save_dir.mkdir(parents=True, exist_ok=True)

    device = model.device
    model.train()
    if classifier is not None:
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad_(False)

    data_loader = infinite_data_loader(train_loader)
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))

    optimizer.zero_grad()

    for it in range(start_iter, args.max_iter + 1):
        batch = next(data_loader)
        audio_pair, motion_coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair = batch_to_device(
            batch, device, predict_head_pose, args
        )

        if args.use_context_audio_feat:
            audio_feat = model.extract_audio_feature(
                torch.cat(audio_pair, dim=1),
                args.n_motions * 2,
            )
        else:
            audio_feat = None

        losses = initialize_loss_tensors(device)
        prev_emo_frame_feat = None

        for i in range(2):
            audio_in, motion_coef_in, e2v_frame_in, end_idx = prepare_window_inputs(
                args,
                model,
                i,
                audio_pair,
                motion_coef_pair,
                e2v_frame_pair,
                audio_unit,
                audio_feat=audio_feat,
            )
            batch_size = audio_in.shape[0]
            e2v_utt = e2v_utt_pair[i]
            e2v_frame_full = e2v_frame_pair[i]

            indicator = make_valid_mask(batch_size, args.n_motions, end_idx, device) if args.use_indicator else None

            if i == 0:
                noise, target, prev_motion_coef, prev_audio_feat = model(
                    motion_coef_in,
                    audio_in,
                    indicator=indicator,
                    emo_index=emo_index,
                    emo_utt_feat=e2v_utt,
                    emo_frame_feat=e2v_frame_in,
                )

                # finalv3 does not return prev_emo_frame_feat.  Derive it from
                # the first full e2v window to match the AR context used by
                # prev_motion_coef / prev_audio_feat.
                prev_emo_frame_feat = e2v_frame_full[:, -args.n_prev_motions:].detach()

                if end_idx is not None:
                    prev_motion_coef = motion_coef_pair[i][:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio_pair[i])[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                noise, target, _, _ = model(
                    motion_coef_in,
                    audio_in,
                    prev_motion_coef,
                    prev_audio_feat,
                    indicator=indicator,
                    emo_index=emo_index,
                    emo_utt_feat=e2v_utt,
                    emo_frame_feat=e2v_frame_in,
                    prev_emo_frame_feat=prev_emo_frame_feat,
                )

            loss_pack = utils.compute_loss_new(
                args,
                i == 0,
                motion_coef_in,
                noise,
                target,
                prev_motion_coef,
                end_idx,
            )
            losses = accumulate_original_losses(losses, args, predict_head_pose, loss_pack)

            exps = target[:, args.n_prev_motions:, :63].clone()
            pred_emo, _ = classifier(exps)
            losses["emo"] = losses["emo"] + cross_criterion(pred_emo, emo_index) / 2

        loss = build_original_total_loss(args, losses, predict_head_pose)
        loss_log["loss"].append(loss.item())
        append_original_logs(args, loss_log, losses, predict_head_pose)

        loss.backward()

        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        description = describe_original_losses(loss_log, prefix=f"Iter: {it}\\tTrain")
        logging.info(description)

        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar("train/total_loss", np.mean(loss_log["loss"]), it)
            write_original_scalars(writer, "train", loss_log, it)
            writer.add_scalar("opt/lr", optimizer.param_groups[0]["lr"], it)

        if scheduler is not None:
            if args.scheduler != "WarmupThenDecay" or (
                args.scheduler == "WarmupThenDecay" and it < args.cos_max_iter
            ):
                scheduler.step()

        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save(
                {"args": args, "model": model.state_dict(), "iter": it},
                save_dir / f"iter_{it:07}.pt",
            )

        if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:
            val(args, model, val_loader, it, 1, "val", writer, classifier)


@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode="val", writer=None, classifier=None):
    is_training = model.training
    device = model.device
    model.eval()
    if classifier is not None:
        classifier.eval()

    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(list)

    for _ in range(n_rounds):
        for batch in test_loader:
            audio_pair, motion_coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair = batch_to_device(
                batch, device, predict_head_pose, args
            )

            losses = initialize_loss_tensors(device)
            prev_emo_frame_feat = None

            for i in range(2):
                audio_in = audio_pair[i]
                motion_coef_in = motion_coef_pair[i]
                e2v_utt = e2v_utt_pair[i]
                e2v_frame_in = e2v_frame_pair[i]
                batch_size = audio_in.shape[0]
                end_idx = None
                indicator = torch.ones(batch_size, args.n_motions, device=device) if args.use_indicator else None

                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat = model(
                        motion_coef_in,
                        audio_in,
                        indicator=indicator,
                        emo_index=emo_index,
                        emo_utt_feat=e2v_utt,
                        emo_frame_feat=e2v_frame_in,
                    )
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                    prev_emo_frame_feat = e2v_frame_in[:, -args.n_prev_motions:].detach()
                else:
                    noise, target, _, _ = model(
                        motion_coef_in,
                        audio_in,
                        prev_motion_coef,
                        prev_audio_feat,
                        indicator=indicator,
                        emo_index=emo_index,
                        emo_utt_feat=e2v_utt,
                        emo_frame_feat=e2v_frame_in,
                        prev_emo_frame_feat=prev_emo_frame_feat,
                    )

                loss_pack = utils.compute_loss_new(
                    args,
                    i == 0,
                    motion_coef_in,
                    noise,
                    target,
                    prev_motion_coef,
                    end_idx,
                )
                losses = accumulate_original_losses(losses, args, predict_head_pose, loss_pack)

                exps = target[:, args.n_prev_motions:, :63].clone()
                pred_emo, _ = classifier(exps)
                losses["emo"] = losses["emo"] + cross_criterion(pred_emo, emo_index) / 2

            loss = build_original_total_loss(args, losses, predict_head_pose)
            loss_log["loss"].append(loss.item())
            append_original_logs(args, loss_log, losses, predict_head_pose)

    description = describe_original_losses(loss_log, prefix=mode, current_iter=current_iter)
    print(description)

    if writer is not None:
        writer.add_scalar(f"{mode}/total_loss", np.mean(loss_log["loss"]), current_iter)
        write_original_scalars(writer, mode, loss_log, current_iter)

    if is_training:
        model.train()


def main(args, option_text=None):
    device = setup_device(args.device_id)

    model_kwargs = dict(
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
        e2v_dim=args.e2v_dim,
        num_label_tokens=args.num_label_tokens,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        align_mask_width=args.align_mask_width,
        decoder_dropout=args.decoder_dropout,
        audio_scale=args.audio_scale,
        emotion_audio_scale=args.emotion_audio_scale,
        emotion_audio_residual_init=args.emotion_audio_residual_init,
    )
    model = DitTalkingHead(**model_kwargs)

    exp_dir = Path(args.exp_root) / args.exp_name

    train_dataset = EmoLevelE2VDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split="train",
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
        emotion2vec_root=args.emotion2vec_root,
        emotion2vec_dim=args.e2v_dim,
    )
    val_dataset = EmoLevelE2VDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split="val",
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
        emotion2vec_root=args.emotion2vec_root,
        emotion2vec_dim=args.e2v_dim,
    )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(args.emotion_classifier_ckpt, map_location=device), strict=False)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / "options.log", "w", encoding="utf-8") as f:
            f.write(option_text)
        writer.add_text("options", option_text)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), "log.txt"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"exp_name: {exp_dir.name}")
    logging.info(f"model parameters: {utils.count_parameters(model)}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    if args.scheduler == "Warmup":
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == "WarmupThenDecay":
        from src.scheduler import GradualWarmupScheduler
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.cos_max_iter - args.warm_iter,
            args.lr * args.min_lr_ratio,
        )
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
    else:
        scheduler = None

    train(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        exp_dir / "checkpoints",
        scheduler=scheduler,
        writer=writer,
        classifier=classifier,
    )


def build_parser():
    parser = argparse.ArgumentParser()

    # experiment
    parser.add_argument("--exp_name", type=str, default="20260711_emotion_dit_finalv3")
    parser.add_argument("--exp_root", type=str, default="experiments/emo_dit")
    parser.add_argument("--device_id", type=int, default=3)

    # data
    parser.add_argument("--data_root", type=Path, default="src/my_prepare/")
    parser.add_argument("--motion_filename", type=str, default="front_all_motions.pkl")
    parser.add_argument("--motion_template_filename", type=str, default="motion_template.pkl")
    parser.add_argument("--emotion2vec_root", type=str, default=None)
    parser.add_argument("--emotion_classifier_ckpt", type=str, default="pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_strategy", type=str, default="random")
    parser.add_argument("--normalize_type", type=str, default="mix", choices=["std", "case", "scale", "minmax", "mix"])

    # diffusion / model
    parser.add_argument("--target", type=str, default="sample", choices=["sample", "noise"])
    parser.add_argument("--guiding_conditions", type=str, default="audio,emotion")
    parser.add_argument("--cfg_mode", type=str, default="incremental", choices=["incremental", "independent"])
    parser.add_argument("--n_diff_steps", type=int, default=50)
    parser.add_argument("--diff_schedule", type=str, default="cosine", choices=["linear", "cosine", "quadratic", "sigmoid"])
    parser.add_argument("--no_head_pose", action="store_true", default=False)
    parser.add_argument("--rot_repr", type=str, default="aa", choices=["aa"])

    parser.add_argument("--audio_model", type=str, default="wav2vec2", choices=["wav2vec2", "hubert", "hubert_zh", "hubert_zh_ori"])
    parser.add_argument("--architecture", type=str, default="decoder", choices=["decoder"])
    parser.add_argument("--align_mask_width", type=int, default=3)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--e2v_dim", type=int, default=1024)
    parser.add_argument("--num_label_tokens", type=int, default=8)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--decoder_dropout", type=float, default=0.0)
    parser.add_argument("--audio_scale", type=float, default=1.0)
    parser.add_argument("--emotion_audio_scale", type=float, default=0.5)
    parser.add_argument("--emotion_audio_residual_init", type=float, default=0.05)

    parser.add_argument("--n_motions", type=int, default=100)
    parser.add_argument("--n_prev_motions", type=int, default=25)
    parser.add_argument("--motion_feat_dim", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--pad_mode", type=str, default="zero", choices=["zero", "replicate"])

    # optimization
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--scheduler", type=str, default="WarmupThenDecay", choices=["None", "Warmup", "WarmupThenDecay"])
    parser.add_argument("--warm_iter", type=int, default=2000)
    parser.add_argument("--cos_max_iter", type=int, default=100000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    parser.add_argument("--criterion", type=str, default="l2", choices=["l1", "l2"])
    parser.add_argument("--clip_grad", default=True, action="store_true")

    # original train.py losses
    parser.add_argument("--l_exp", type=float, default=0.1)
    parser.add_argument("--l_exp_vel", type=float, default=1e-4)
    parser.add_argument("--l_exp_smooth", type=float, default=1e-4)
    parser.add_argument("--l_head_angle", type=float, default=1e-2)
    parser.add_argument("--l_head_vel", type=float, default=1e-2)
    parser.add_argument("--l_head_smooth", type=float, default=1e-2)
    parser.add_argument("--l_head_trans", type=float, default=1e-2)
    parser.add_argument("--no_constrain_prev", action="store_true")

    # train_e2v additional losses; used by train_finalv3_additional_loss.py.
    # They are harmless in train_finalv3.py because that file ignores them.
    parser.add_argument("--l_emo_cls", type=float, default=1.0)
    parser.add_argument("--l_emo_level", type=float, default=0.2)
    parser.add_argument("--l_prosody_curve", type=float, default=0.02)
    parser.add_argument("--no_prosody_velocity", action="store_true", default=False)
    parser.add_argument("--prosody_motion_vel_weight", type=float, default=0.5)
    parser.add_argument("--prosody_audio_vel_weight", type=float, default=0.5)

    # data augmentation / logging
    parser.add_argument("--use_context_audio_feat", action="store_true")
    parser.add_argument("--use_indicator", action="store_true", default=True)
    parser.add_argument("--trunc_prob1", type=float, default=0.3)
    parser.add_argument("--trunc_prob2", type=float, default=0.4)
    parser.add_argument("--save_iter", type=int, default=1000)
    parser.add_argument("--val_iter", type=int, default=50)
    parser.add_argument("--log_iter", type=int, default=10)
    parser.add_argument("--log_smooth_win", type=int, default=50)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    option_text = utils.get_option_text(args, parser)
    main(args, option_text)

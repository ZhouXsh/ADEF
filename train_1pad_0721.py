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

from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel_1pad_0721 import EmoLevelDataset
from src.modules.emotion_dit_Unification_1pad_0721 import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils


g_exp_name = "20260721_emotion_dit_Unification_1pad_0721"
device_id = 1
if torch.cuda.is_available():
    torch.cuda.set_device(device_id)
device = torch.device(
    f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
)
cross_criterion = torch.nn.CrossEntropyLoss()


def _prepare_model_inputs(batch, device):
    audio_pair, coef_pair, canonical_kp_pair, emo_index, emo_level = batch
    audio_pair = [audio.to(device) for audio in audio_pair]
    coef_pair = [
        {key: value.to(device) for key, value in coef_pair[i].items()}
        for i in range(2)
    ]
    canonical_kp_pair = [value.to(device) for value in canonical_kp_pair]
    return (
        audio_pair,
        coef_pair,
        canonical_kp_pair,
        emo_index.to(device),
        emo_level.to(device),
    )


def _prediction_for_loss(target, prev_motion_coef, args):
    current_target = target[:, -args.n_motions:]
    target_for_loss = torch.cat(
        [prev_motion_coef.detach(), current_target], dim=1
    )
    return target_for_loss, current_target


def _add_half(total, value):
    return total if value is None else total + value / 2


def _run_pair(
    args,
    model,
    classifier,
    audio_pair,
    coef_pair,
    canonical_kp_pair,
    emo_index,
    audio_unit,
    training,
):
    predict_head_pose = not args.no_head_pose
    motion_coef_pair = [
        utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose)
        for i in range(2)
    ]

    if args.use_context_audio_feat:
        audio_feat = model.extract_audio_feature(
            torch.cat(audio_pair, dim=1), args.n_motions * 2
        )
    else:
        audio_feat = None

    totals = {
        "noise": torch.tensor(0.0, device=model.device),
        "emo": torch.tensor(0.0, device=model.device),
        "exp": torch.tensor(0.0, device=model.device),
        "exp_vel": torch.tensor(0.0, device=model.device),
        "exp_smooth": torch.tensor(0.0, device=model.device),
        "head_angle": torch.tensor(0.0, device=model.device),
        "head_vel": torch.tensor(0.0, device=model.device),
        "head_smooth": torch.tensor(0.0, device=model.device),
        "head_trans": torch.tensor(0.0, device=model.device),
    }

    prev_motion_coef = None
    prev_audio_feat = None

    for i in range(2):
        audio = audio_pair[i]
        motion_coef = motion_coef_pair[i]
        batch_size = audio.shape[0]

        use_truncation = training and (
            (i == 0 and np.random.rand() < args.trunc_prob1)
            or (i != 0 and np.random.rand() < args.trunc_prob2)
        )
        if use_truncation:
            audio_in, motion_coef_in, end_idx = (
                utils.truncate_motion_coef_and_audio(
                    audio,
                    motion_coef,
                    args.n_motions,
                    audio_unit,
                    args.pad_mode,
                )
            )
            if args.use_context_audio_feat and i != 0:
                audio_in = model.extract_audio_feature(
                    torch.cat([audio_pair[i - 1], audio_in], dim=1),
                    args.n_motions * 2,
                )[:, -args.n_motions:]
        else:
            if args.use_context_audio_feat:
                audio_in = audio_feat[
                    :, i * args.n_motions:(i + 1) * args.n_motions
                ]
            else:
                audio_in = audio
            motion_coef_in = motion_coef
            end_idx = None

        if args.use_indicator:
            if end_idx is None:
                indicator = torch.ones(
                    batch_size, args.n_motions, device=model.device
                )
            else:
                indicator = (
                    torch.arange(args.n_motions, device=model.device)
                    .expand(batch_size, -1)
                    < end_idx.unsqueeze(1)
                )
        else:
            indicator = None

        model_kwargs = {
            "indicator": indicator,
            "emo_index": emo_index,
            "canonical_kp_feat": canonical_kp_pair[i],
        }

        if i == 0:
            noise, target, returned_motion, returned_audio = model(
                motion_coef_in, audio_in, **model_kwargs
            )
            if end_idx is not None:
                prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                if args.use_context_audio_feat:
                    prev_audio_feat = audio_feat[
                        :,
                        args.n_motions - args.n_prev_motions:args.n_motions,
                    ].detach()
                else:
                    with torch.no_grad():
                        prev_audio_feat = model.extract_audio_feature(audio)[
                            :, -args.n_prev_motions:
                        ]
            else:
                prev_motion_coef = returned_motion[:, -args.n_prev_motions:]
                prev_audio_feat = returned_audio[:, -args.n_prev_motions:]
        else:
            noise, target, _, _ = model(
                motion_coef_in,
                audio_in,
                prev_motion_feat=prev_motion_coef,
                prev_audio_feat=prev_audio_feat,
                **model_kwargs,
            )

        target_for_loss, current_target = _prediction_for_loss(
            target, prev_motion_coef, args
        )
        losses = utils.compute_loss_new(
            args,
            i == 0,
            motion_coef_in,
            noise,
            target_for_loss,
            prev_motion_coef,
            end_idx,
        )
        (
            loss_n,
            loss_exp,
            loss_exp_v,
            loss_exp_s,
            loss_ha,
            loss_hv,
            loss_hs,
            loss_ht,
        ) = losses

        pred_emo, _ = classifier(current_target[:, :, :63])
        loss_e = cross_criterion(pred_emo, emo_index)

        totals["noise"] += loss_n / 2
        totals["emo"] += loss_e / 2
        totals["exp"] = _add_half(totals["exp"], loss_exp)
        totals["exp_vel"] = _add_half(totals["exp_vel"], loss_exp_v)
        totals["exp_smooth"] = _add_half(totals["exp_smooth"], loss_exp_s)
        totals["head_angle"] = _add_half(totals["head_angle"], loss_ha)
        totals["head_vel"] = _add_half(totals["head_vel"], loss_hv)
        totals["head_smooth"] = _add_half(totals["head_smooth"], loss_hs)
        if loss_ht is not None:
            totals["head_trans"] += loss_ht

    loss = totals["noise"] + totals["emo"]
    loss += args.l_exp * totals["exp"]
    loss += args.l_exp_vel * totals["exp_vel"]
    loss += args.l_exp_smooth * totals["exp_smooth"]
    if args.target == "sample" and predict_head_pose:
        loss += args.l_head_angle * totals["head_angle"]
        loss += args.l_head_vel * totals["head_vel"]
        loss += args.l_head_smooth * totals["head_smooth"]
        loss += args.l_head_trans * totals["head_trans"]
    totals["loss"] = loss
    return totals


def _log_values(prefix, values, writer, step):
    if writer is not None:
        for key, value in values.items():
            writer.add_scalar(f"{prefix}/{key}", value, step)


def train(
    args,
    model,
    train_loader,
    val_loader,
    optimizer,
    save_dir,
    scheduler=None,
    writer=None,
    classifier=None,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    data_loader = infinite_data_loader(train_loader)
    audio_unit = train_loader.dataset.audio_unit
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad()

    for it in range(args.max_iter + 1):
        batch = _prepare_model_inputs(next(data_loader), model.device)
        audio_pair, coef_pair, canonical_pair, emo_index, _ = batch
        totals = _run_pair(
            args,
            model,
            classifier,
            audio_pair,
            coef_pair,
            canonical_pair,
            emo_index,
            audio_unit,
            training=True,
        )
        totals["loss"].backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        for key, value in totals.items():
            loss_log[key].append(float(value.detach().item()))
        logging.info(
            f"Iter: {it}\tTrain loss: "
            f"[N: {np.mean(loss_log['noise']):.3e}, "
            f"EX: {np.mean(loss_log['exp']):.3e}, "
            f"Emo: {np.mean(loss_log['emo']):.3e}, "
            f"Total: {np.mean(loss_log['loss']):.3e}]"
        )

        if it % args.log_iter == 0:
            _log_values(
                "train",
                {key: np.mean(values) for key, values in loss_log.items()},
                writer,
                it,
            )
        if scheduler is not None:
            if args.scheduler != "WarmupThenDecay" or it < args.cos_max_iter:
                scheduler.step()
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save(
                {"args": args, "model": model.state_dict(), "iter": it},
                save_dir / f"iter_{it:07}.pt",
            )
        if it % args.val_iter == 0 or it == args.max_iter:
            val(args, model, val_loader, it, writer, classifier)


@torch.no_grad()
def val(args, model, loader, current_iter, writer, classifier):
    was_training = model.training
    model.eval()
    loss_log = defaultdict(list)
    for raw_batch in loader:
        audio_pair, coef_pair, canonical_pair, emo_index, _ = (
            _prepare_model_inputs(raw_batch, model.device)
        )
        totals = _run_pair(
            args,
            model,
            classifier,
            audio_pair,
            coef_pair,
            canonical_pair,
            emo_index,
            loader.dataset.audio_unit,
            training=False,
        )
        for key, value in totals.items():
            loss_log[key].append(float(value.detach().item()))
    means = {key: np.mean(values) for key, values in loss_log.items()}
    _log_values("val", means, writer, current_iter)
    logging.info(
        f"(Iter {current_iter:>6}) val loss: "
        f"[N: {means['noise']:.3e}, EX: {means['exp']:.3e}, "
        f"Emo: {means['emo']:.3e}, Total: {means['loss']:.3e}]"
    )
    if was_training:
        model.train()


def main(args, option_text=None):
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
    )

    exp_dir = Path("experiments/emo_dit") / args.exp_name
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
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    classifier = Classifier().to(device)
    classifier.load_state_dict(
        torch.load(
            "pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth",
            map_location=device,
        ),
        strict=False,
    )
    classifier.eval()

    log_dir = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / "options.log", "w", encoding="utf-8") as file:
            file.write(option_text)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), "log.txt"),
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
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
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warm_iter, after_scheduler
        )
    else:
        scheduler = None

    train(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        exp_dir / "checkpoints",
        scheduler,
        writer,
        classifier,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--exp_name", type=str, default=g_exp_name)
    parser.add_argument("--data_root", type=Path, default="src/my_prepare/")
    parser.add_argument("--motion_filename", type=str, default="front_all_motions.pkl")
    parser.add_argument("--motion_template_filename", type=str, default="motion_template.pkl")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_strategy", type=str, default="random")
    parser.add_argument("--normalize_type", type=str, default="mix")
    parser.add_argument("--target", type=str, default="sample", choices=["sample", "noise"])
    parser.add_argument("--guiding_conditions", type=str, default="audio,emotion")
    parser.add_argument("--cfg_mode", type=str, default="incremental")
    parser.add_argument("--n_diff_steps", type=int, default=50)
    parser.add_argument("--diff_schedule", type=str, default="cosine")
    parser.add_argument("--no_head_pose", action="store_true", default=False)
    parser.add_argument("--rot_repr", type=str, default="aa")
    parser.add_argument("--audio_model", type=str, default="wav2vec2")
    parser.add_argument("--architecture", type=str, default="decoder")
    parser.add_argument("--use_indicator", action="store_true", default=True)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_motions", type=int, default=100)
    parser.add_argument("--n_prev_motions", type=int, default=25)
    parser.add_argument("--motion_feat_dim", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--pad_mode", type=str, default="zero")
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--scheduler", type=str, default="WarmupThenDecay")
    parser.add_argument("--criterion", type=str, default="l2")
    parser.add_argument("--clip_grad", default=True, action="store_true")
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
    parser.add_argument("--warm_iter", type=int, default=10000)
    parser.add_argument("--cos_max_iter", type=int, default=100000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    option_text = (
        utils.common.get_option_text(args, parser)
        if args.mode == "train"
        else None
    )
    main(args, option_text)

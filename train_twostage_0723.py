# 在 train.py 的基础上增加两阶段训练支持。
# Stage 1: GenericTalkingMotionDataset + audio-only condition.
# Stage 2: EmoLevelDataset + audio-emotion condition.

import argparse
from collections import deque, defaultdict
from pathlib import Path

import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
from src.modules.emotion_dit_Unification_twostage_0723 import DitTalkingHead
import src.utils as utils
from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel import EmoLevelDataset
from src.dataset.dataset_GenericTalkingMotion import GenericTalkingMotionDataset


g_exp_name = '20260723_emotion_dit_Unification_twostage'

STAGE_AUDIO = 'audio_pretrain'
STAGE_EMOTION = 'emotion_finetune'

cross_criterion = torch.nn.CrossEntropyLoss()


def get_device(device_id):
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        return torch.device(f'cuda:{device_id}')
    return torch.device('cpu')


def unpack_batch(batch, args, device):
    """Unify the two dataset return formats without changing either dataset."""
    if args.training_stage == STAGE_AUDIO:
        audio_pair, coef_pair, _ = batch
        batch_size = audio_pair[0].shape[0]
        emo_index = torch.full(
            (batch_size,),
            args.neutral_emo_index,
            dtype=torch.long,
            device=device,
        )
        emo_level = None
    else:
        audio_pair, coef_pair, emo_index, emo_level = batch
        emo_index = emo_index.to(device=device, dtype=torch.long)
        emo_level = emo_level.to(device) if torch.is_tensor(emo_level) else emo_level

    audio_pair = [audio.to(device) for audio in audio_pair]
    coef_pair = [
        {key: value.to(device) for key, value in coef_pair[i].items()}
        for i in range(2)
    ]
    return audio_pair, coef_pair, emo_index, emo_level


def get_condition_mode(training_stage):
    if training_stage == STAGE_AUDIO:
        return 'audio'
    return 'audio_emotion'


def get_emotion_loss_weight(args, current_iter):
    if args.training_stage != STAGE_EMOTION:
        return 0.0
    if args.emotion_warmup_iter <= 0:
        return args.l_emo
    warmup_ratio = min(1.0, float(current_iter + 1) / args.emotion_warmup_iter)
    return args.l_emo * warmup_ratio


def train(args, model, train_loader, val_loader, optimizer, save_dir,
          scheduler=None, writer=None, start_iter=0, classifier=None,
          generic_val_loader=None):

    save_dir.mkdir(parents=True, exist_ok=True)

    device = model.device
    model.train()

    data_loader = infinite_data_loader(train_loader)
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    condition_mode = get_condition_mode(args.training_stage)

    optimizer.zero_grad(set_to_none=True)
    for it in range(start_iter, args.max_iter + 1):
        batch = next(data_loader)
        audio_pair, coef_pair, emo_index, _ = unpack_batch(batch, args, device)
        motion_coef_pair = [
            utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose)
            for i in range(2)
        ]

        if args.use_context_audio_feat:
            audio_feat = model.extract_audio_feature(
                torch.cat(audio_pair, dim=1), args.n_motions * 2
            )

        loss_noise = torch.tensor(0.0, device=device)
        loss_emo = torch.tensor(0.0, device=device)
        loss_expression = torch.tensor(0.0, device=device)
        loss_exp_vel = torch.tensor(0.0, device=device)
        loss_exp_smooth = torch.tensor(0.0, device=device)
        loss_head_angle = torch.tensor(0.0, device=device)
        loss_head_vel = torch.tensor(0.0, device=device)
        loss_head_smooth = torch.tensor(0.0, device=device)
        loss_head_trans = torch.tensor(0.0, device=device)

        prev_motion_coef = None
        prev_audio_feat = None
        for i in range(2):
            audio = audio_pair[i]
            motion_coef = motion_coef_pair[i]
            batch_size = audio.shape[0]

            if ((i == 0 and np.random.rand() < args.trunc_prob1)
                    or (i != 0 and np.random.rand() < args.trunc_prob2)):
                audio_in, motion_coef_in, end_idx = \
                    utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions,
                        audio_unit, args.pad_mode
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
                motion_coef_in, end_idx = motion_coef, None

            if args.use_indicator:
                if end_idx is not None:
                    indicator = (
                        torch.arange(args.n_motions, device=device)
                        .expand(batch_size, -1)
                        < end_idx.unsqueeze(1)
                    )
                else:
                    indicator = torch.ones(
                        batch_size, args.n_motions, device=device
                    )
            else:
                indicator = None

            if i == 0:
                outputs = model(
                    motion_coef_in,
                    audio_in,
                    indicator=indicator,
                    emo_index=emo_index,
                    condition_mode=condition_mode,
                    uncond_drop_prob=args.uncond_drop_prob,
                    emotion_drop_prob=args.emotion_drop_prob,
                    return_condition_info=True,
                )
                noise, target, prev_motion_coef, prev_audio_feat, condition_info = outputs

                if end_idx is not None:
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[
                            :, args.n_motions - args.n_prev_motions:args.n_motions
                        ].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[
                                :, -args.n_prev_motions:
                            ]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                outputs = model(
                    motion_coef_in,
                    audio_in,
                    prev_motion_coef,
                    prev_audio_feat,
                    indicator=indicator,
                    emo_index=emo_index,
                    condition_mode=condition_mode,
                    uncond_drop_prob=args.uncond_drop_prob,
                    emotion_drop_prob=args.emotion_drop_prob,
                    return_condition_info=True,
                )
                noise, target, _, _, condition_info = outputs

            losses = utils.compute_loss_new(
                args, i == 0, motion_coef_in, noise, target,
                prev_motion_coef, end_idx
            )
            loss_n, loss_exp, loss_exp_v, loss_exp_s, \
                loss_ha, loss_hc, loss_hs, loss_ht = losses

            if args.training_stage == STAGE_EMOTION:
                if args.target != 'sample':
                    raise ValueError(
                        'Emotion finetuning currently requires --target sample.'
                    )
                emotion_active = condition_info['emotion_active']
                if emotion_active.any():
                    exps = target[:, args.n_prev_motions:, :63]
                    pred_emo, _ = classifier(exps[emotion_active])
                    loss_e = cross_criterion(
                        pred_emo, emo_index[emotion_active]
                    )
                    loss_emo = loss_emo + loss_e / 2

            loss_noise = loss_noise + loss_n / 2
            if loss_exp is not None:
                loss_expression = loss_expression + loss_exp / 2
            if loss_exp_v is not None:
                loss_exp_vel = loss_exp_vel + loss_exp_v / 2
            if loss_exp_s is not None:
                loss_exp_smooth = loss_exp_smooth + loss_exp_s / 2
            if (args.target == 'sample' and predict_head_pose
                    and args.l_head_angle > 0 and loss_ha is not None):
                loss_head_angle = loss_head_angle + loss_ha / 2
            if (args.target == 'sample' and predict_head_pose
                    and args.l_head_vel > 0 and loss_hc is not None):
                loss_head_vel = loss_head_vel + loss_hc / 2
            if (args.target == 'sample' and predict_head_pose
                    and args.l_head_smooth > 0 and loss_hs is not None):
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            if (args.target == 'sample' and predict_head_pose
                    and args.l_head_trans > 0 and loss_ht is not None):
                loss_head_trans = loss_head_trans + loss_ht

        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise

        current_l_emo = get_emotion_loss_weight(args, it)
        loss_log['emo'].append(loss_emo.item() * current_l_emo)
        loss = loss + current_l_emo * loss_emo

        loss_log['exp'].append(loss_expression.item() * args.l_exp)
        loss = loss + args.l_exp * loss_expression

        loss_log['exp_vel'].append(loss_exp_vel.item() * args.l_exp_vel)
        loss = loss + args.l_exp_vel * loss_exp_vel

        loss_log['exp_smooth'].append(
            loss_exp_smooth.item() * args.l_exp_smooth
        )
        loss = loss + args.l_exp_smooth * loss_exp_smooth

        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            loss_log['head_angle'].append(
                loss_head_angle.item() * args.l_head_angle
            )
            loss = loss + args.l_head_angle * loss_head_angle
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            loss_log['head_vel'].append(
                loss_head_vel.item() * args.l_head_vel
            )
            loss = loss + args.l_head_vel * loss_head_vel
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            loss_log['head_smooth'].append(
                loss_head_smooth.item() * args.l_head_smooth
            )
            loss = loss + args.l_head_smooth * loss_head_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            loss_log['head_trans'].append(
                loss_head_trans.item() * args.l_head_trans
            )
            loss = loss + args.l_head_trans * loss_head_trans

        loss_log['loss'].append(loss.item())
        scaled_loss = loss / args.gradient_accumulation_steps
        scaled_loss.backward()

        update_now = (
            (it + 1) % args.gradient_accumulation_steps == 0
            or it == args.max_iter
        )
        if update_now:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=args.clip_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        description = f'Iter: {it}\t  Train loss: [N: {np.mean(loss_log["noise"]):.3e}'
        description += f", EX: {np.mean(loss_log['exp']):.3e}"
        description += f", EX_V: {np.mean(loss_log['exp_vel']):.3e}"
        description += f", EX_S: {np.mean(loss_log['exp_smooth']):.3e}"
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            description += f', HA: {np.mean(loss_log["head_angle"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            description += f', HV: {np.mean(loss_log["head_vel"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            description += f', HS: {np.mean(loss_log["head_smooth"]):.3e}'
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            description += f', HT: {np.mean(loss_log["head_trans"]):.3e}'
        if args.training_stage == STAGE_EMOTION:
            description += f", Emo: {np.mean(loss_log['emo']):.3e}"
            description += f', l_emo: {current_l_emo:.3e}'
        description += ']'
        logging.info(description)

        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/simple_loss', np.mean(loss_log['noise']), it)
            writer.add_scalar('train/exp_loss', np.mean(loss_log['exp']), it)
            writer.add_scalar('train/exp_vel_loss', np.mean(loss_log['exp_vel']), it)
            writer.add_scalar('train/exp_smooth_loss', np.mean(loss_log['exp_smooth']), it)
            if args.training_stage == STAGE_EMOTION:
                writer.add_scalar('train/emotion_loss', np.mean(loss_log['emo']), it)
                writer.add_scalar('train/emotion_loss_weight', current_l_emo, it)
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                writer.add_scalar('train/head_angle', np.mean(loss_log['head_angle']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                writer.add_scalar('train/head_vel', np.mean(loss_log['head_vel']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                writer.add_scalar('train/head_smooth', np.mean(loss_log['head_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                writer.add_scalar('train/head_trans', np.mean(loss_log['head_trans']), it)
            for group_index, group in enumerate(optimizer.param_groups):
                writer.add_scalar(
                    f'opt/lr_group_{group_index}', group['lr'], it
                )

        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            checkpoint = {
                'stage': args.training_stage,
                'args': args,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
                'iter': it,
            }
            torch.save(checkpoint, save_dir / f'iter_{it:07}.pt')

        if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:
            val(
                args, model, val_loader, it, 1, 'val', writer, classifier,
                stage_override=args.training_stage
            )
            if generic_val_loader is not None:
                val(
                    args, model, generic_val_loader, it, 1,
                    'val_generic_audio', writer, None,
                    stage_override=STAGE_AUDIO
                )


@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode='val',
        writer=None, classifier=None, stage_override=None):
    is_training = model.training
    device = model.device
    model.eval()

    validation_stage = stage_override or args.training_stage
    condition_mode = get_condition_mode(validation_stage)
    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(list)
    for _ in range(n_rounds):
        for batch in test_loader:
            original_stage = args.training_stage
            args.training_stage = validation_stage
            audio_pair, coef_pair, emo_index, _ = unpack_batch(batch, args, device)
            args.training_stage = original_stage

            motion_coef_pair = [
                utils.get_motion_coef(
                    coef_pair[i], args.rot_repr, predict_head_pose
                )
                for i in range(2)
            ]

            if args.use_context_audio_feat:
                audio_feat = model.extract_audio_feature(
                    torch.cat(audio_pair, dim=1), args.n_motions * 2
                )

            loss_noise = torch.tensor(0.0, device=device)
            loss_emo = torch.tensor(0.0, device=device)
            loss_expression = torch.tensor(0.0, device=device)
            loss_exp_vel = torch.tensor(0.0, device=device)
            loss_exp_smooth = torch.tensor(0.0, device=device)
            loss_head_angle = torch.tensor(0.0, device=device)
            loss_head_vel = torch.tensor(0.0, device=device)
            loss_head_smooth = torch.tensor(0.0, device=device)
            loss_head_trans = torch.tensor(0.0, device=device)

            prev_motion_coef = None
            prev_audio_feat = None
            for i in range(2):
                audio = audio_pair[i]
                motion_coef = motion_coef_pair[i]
                batch_size = audio.shape[0]

                use_truncation = args.val_use_truncation and (
                    (i == 0 and np.random.rand() < args.trunc_prob1)
                    or (i != 0 and np.random.rand() < args.trunc_prob2)
                )
                if use_truncation:
                    audio_in, motion_coef_in, end_idx = \
                        utils.truncate_motion_coef_and_audio(
                            audio, motion_coef, args.n_motions,
                            audio_unit, args.pad_mode
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
                    motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    if end_idx is not None:
                        indicator = (
                            torch.arange(args.n_motions, device=device)
                            .expand(batch_size, -1)
                            < end_idx.unsqueeze(1)
                        )
                    else:
                        indicator = torch.ones(
                            batch_size, args.n_motions, device=device
                        )
                else:
                    indicator = None

                if i == 0:
                    outputs = model(
                        motion_coef_in,
                        audio_in,
                        indicator=indicator,
                        emo_index=emo_index,
                        condition_mode=condition_mode,
                        return_condition_info=True,
                    )
                    noise, target, prev_motion_coef, prev_audio_feat, condition_info = outputs
                    if end_idx is not None:
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:
                            prev_audio_feat = audio_feat[
                                :, args.n_motions - args.n_prev_motions:args.n_motions
                            ]
                        else:
                            prev_audio_feat = model.extract_audio_feature(audio)[
                                :, -args.n_prev_motions:
                            ]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    outputs = model(
                        motion_coef_in,
                        audio_in,
                        prev_motion_coef,
                        prev_audio_feat,
                        indicator=indicator,
                        emo_index=emo_index,
                        condition_mode=condition_mode,
                        return_condition_info=True,
                    )
                    noise, target, _, _, condition_info = outputs

                losses = utils.compute_loss_new(
                    args, i == 0, motion_coef_in, noise, target,
                    prev_motion_coef, end_idx
                )
                loss_n, loss_exp, loss_exp_v, loss_exp_s, \
                    loss_ha, loss_hc, loss_hs, loss_ht = losses

                if validation_stage == STAGE_EMOTION and classifier is not None:
                    exps = target[:, args.n_prev_motions:, :63]
                    pred_emo, _ = classifier(exps)
                    loss_e = cross_criterion(pred_emo, emo_index)
                    loss_emo = loss_emo + loss_e / 2

                loss_noise = loss_noise + loss_n / 2
                if loss_exp is not None:
                    loss_expression = loss_expression + loss_exp / 2
                if loss_exp_v is not None:
                    loss_exp_vel = loss_exp_vel + loss_exp_v / 2
                if loss_exp_s is not None:
                    loss_exp_smooth = loss_exp_smooth + loss_exp_s / 2
                if loss_ha is not None:
                    loss_head_angle = loss_head_angle + loss_ha / 2
                if loss_hc is not None:
                    loss_head_vel = loss_head_vel + loss_hc / 2
                if loss_hs is not None:
                    loss_head_smooth = loss_head_smooth + loss_hs / 2
                if loss_ht is not None:
                    loss_head_trans = loss_head_trans + loss_ht

            loss = loss_noise
            loss_log['noise'].append(loss_noise.item())

            if validation_stage == STAGE_EMOTION:
                loss = loss + args.l_emo * loss_emo
                loss_log['emo'].append((args.l_emo * loss_emo).item())

            loss = loss + args.l_exp * loss_expression
            loss_log['exp'].append((args.l_exp * loss_expression).item())
            loss = loss + args.l_exp_vel * loss_exp_vel
            loss_log['exp_vel'].append((args.l_exp_vel * loss_exp_vel).item())
            loss = loss + args.l_exp_smooth * loss_exp_smooth
            loss_log['exp_smooth'].append(
                (args.l_exp_smooth * loss_exp_smooth).item()
            )

            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss = loss + args.l_head_angle * loss_head_angle
                loss_log['head_angle'].append(
                    (args.l_head_angle * loss_head_angle).item()
                )
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                loss = loss + args.l_head_vel * loss_head_vel
                loss_log['head_vel'].append(
                    (args.l_head_vel * loss_head_vel).item()
                )
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                loss = loss + args.l_head_smooth * loss_head_smooth
                loss_log['head_smooth'].append(
                    (args.l_head_smooth * loss_head_smooth).item()
                )
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                loss = loss + args.l_head_trans * loss_head_trans
                loss_log['head_trans'].append(
                    (args.l_head_trans * loss_head_trans).item()
                )

            loss_log['loss'].append(loss.item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [N: {np.mean(loss_log["noise"]):.3e}'
    description += f", EX: {np.mean(loss_log['exp']):.3e}"
    description += f", EX_V: {np.mean(loss_log['exp_vel']):.3e}"
    description += f", EX_S: {np.mean(loss_log['exp_smooth']):.3e}"
    if validation_stage == STAGE_EMOTION:
        description += f", Emo: {np.mean(loss_log['emo']):.3e}"
    description += ']'
    logging.info(description)

    if writer is not None:
        writer.add_scalar(f'{mode}/total_loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/simple_loss', np.mean(loss_log['noise']), current_iter)
        writer.add_scalar(f'{mode}/exp_loss', np.mean(loss_log['exp']), current_iter)
        writer.add_scalar(f'{mode}/exp_vel_loss', np.mean(loss_log['exp_vel']), current_iter)
        writer.add_scalar(f'{mode}/exp_smooth_loss', np.mean(loss_log['exp_smooth']), current_iter)
        if validation_stage == STAGE_EMOTION:
            writer.add_scalar(f'{mode}/emotion_loss', np.mean(loss_log['emo']), current_iter)

    if is_training:
        model.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_generic_dataset(args, split):
    template_path = args.generic_motion_template_path
    if template_path is None:
        template_path = Path(args.data_root) / args.motion_template_filename

    split_file = (
        args.generic_train_split if split == 'train'
        else args.generic_val_split
    )
    kwargs = dict(
        motion_template_path=template_path,
        split=split,
        split_file=split_file,
        validation_ratio=args.generic_validation_ratio,
        split_seed=args.generic_split_seed,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
        missing_audio_policy=args.generic_missing_audio_policy,
        duplicate_policy=args.generic_duplicate_policy,
    )
    if args.generic_motion_files:
        kwargs['motion_filenames'] = args.generic_motion_files
    elif args.generic_aggregate_motion_files:
        kwargs['aggregate_motion_files'] = args.generic_aggregate_motion_files
    else:
        raise ValueError(
            'Stage 1 requires --generic_motion_files or '
            '--generic_aggregate_motion_files.'
        )
    return GenericTalkingMotionDataset(**kwargs)


def build_emotion_dataset(args, split):
    return EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split=split,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
    )


def freeze_module(module):
    for parameter in module.parameters():
        parameter.requires_grad = False


def configure_trainable_parameters(args, model):
    emotion_prefixes = (
        'emo_embed',
        'adaLN_modulation',
        'null_emotion_feat',
    )

    if args.training_stage == STAGE_AUDIO:
        # The model structure already contains emotion modules, but Stage 1 does
        # not train them.  Only the neutral row of start tokens is selected.
        for name, parameter in model.named_parameters():
            if name.startswith(emotion_prefixes):
                parameter.requires_grad = False
        return

    # Stage 2 starts conservatively from the Stage-1 motion prior.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        is_emotion_parameter = name.startswith(emotion_prefixes)
        is_start_token = name in ['start_motion_feat', 'start_audio_feat']
        is_cross_attention = '.cross_attn.' in name
        is_motion_decoder = name.startswith('denoising_net.motion_dec')

        if is_emotion_parameter or is_start_token:
            parameter.requires_grad = True
        elif args.stage2_train_scope == 'emotion_crossattn' and (
                is_cross_attention or is_motion_decoder):
            parameter.requires_grad = True
        elif args.stage2_train_scope == 'all' and \
                not name.startswith('audio_encoder'):
            parameter.requires_grad = True

    # The pretrained speech encoder is always frozen during emotion finetuning.
    freeze_module(model.audio_encoder)


def build_optimizer(args, model):
    if args.training_stage == STAGE_AUDIO:
        return torch.optim.Adam(
            [parameter for parameter in model.parameters()
             if parameter.requires_grad],
            lr=args.lr,
        )

    emotion_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (name.startswith(('emo_embed', 'adaLN_modulation',
                             'null_emotion_feat'))
                or name in ['start_motion_feat', 'start_audio_feat']):
            emotion_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)

    parameter_groups = []
    if emotion_parameters:
        parameter_groups.append({
            'params': emotion_parameters,
            'lr': args.stage2_emotion_lr,
        })
    if backbone_parameters:
        parameter_groups.append({
            'params': backbone_parameters,
            'lr': args.stage2_backbone_lr,
        })
    return torch.optim.Adam(parameter_groups)


def build_scheduler(args, optimizer):
    if args.scheduler == 'Warmup':
        from src.scheduler import GradualWarmupScheduler
        return GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    if args.scheduler == 'WarmupThenDecay':
        from src.scheduler import GradualWarmupScheduler
        decay_iterations = max(1, args.cos_max_iter - args.warm_iter)
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            decay_iterations,
            args.min_lr_ratio * min(group['lr'] for group in optimizer.param_groups),
        )
        return GradualWarmupScheduler(
            optimizer, 1, args.warm_iter, after_scheduler
        )
    return None


def load_checkpoint(path, model, optimizer=None, scheduler=None,
                    resume_training=False):
    checkpoint = torch.load(path, map_location=model.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    start_iter = 0
    if resume_training:
        if optimizer is not None and checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None and checkpoint.get('scheduler') is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_iter = checkpoint.get('iter', -1) + 1
    return checkpoint, start_iter


def main(args, option_text=None):
    device = get_device(args.device_id)

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
        # Keep one identical model structure for both stages.
        guiding_conditions='audio,emotion',
    )
    model = DitTalkingHead(**model_kwargs)

    exp_dir = Path('experiments/emo_dit') / args.exp_name

    if args.training_stage == STAGE_AUDIO:
        train_dataset = build_generic_dataset(args, 'train')
        val_dataset = build_generic_dataset(args, 'val')
    else:
        train_dataset = build_emotion_dataset(args, 'train')
        val_dataset = build_emotion_dataset(args, 'val')

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    generic_val_loader = None
    if args.training_stage == STAGE_EMOTION and args.stage2_validate_generic:
        generic_val_dataset = build_generic_dataset(args, 'val')
        generic_val_loader = data.DataLoader(
            generic_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    # Load Stage-1 weights before choosing the Stage-2 trainable subset.
    start_iter = 0
    if args.training_stage == STAGE_EMOTION:
        if args.stage1_checkpoint is None and args.resume_checkpoint is None:
            raise ValueError(
                'Emotion finetuning requires --stage1_checkpoint, unless '
                '--resume_checkpoint is a Stage-2 checkpoint.'
            )
        if args.resume_checkpoint is None:
            load_checkpoint(args.stage1_checkpoint, model)
            model.copy_neutral_start_to_all(args.neutral_emo_index)

    configure_trainable_parameters(args, model)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    if args.resume_checkpoint is not None:
        _, start_iter = load_checkpoint(
            args.resume_checkpoint,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_training=True,
        )
        logging.info(
            f'Loading model from {args.resume_checkpoint}, '
            f'start from iter {start_iter}.'
        )

    classifier = None
    if args.training_stage == STAGE_EMOTION:
        classifier = Classifier().to(device)
        classifier.load_state_dict(
            torch.load(args.emotion_classifier_path, map_location=device),
            strict=False,
        )
        classifier.eval()
        classifier.requires_grad_(False)

    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / 'options.log', 'w', encoding='utf-8') as file:
            file.write(option_text)
        writer.add_text('options', option_text)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'exp_name: {exp_dir.name}')
    logging.info(f'training_stage: {args.training_stage}')
    logging.info(f'trainable model parameters: {count_parameters(model)}')
    for group_index, group in enumerate(optimizer.param_groups):
        logging.info(
            f'optimizer group {group_index}: '
            f'{sum(p.numel() for p in group["params"])} params, '
            f'lr={group["lr"]}'
        )

    train(
        args,
        model,
        train_loader,
        val_loader,
        optimizer,
        exp_dir / 'checkpoints',
        scheduler,
        writer,
        start_iter=start_iter,
        classifier=classifier,
        generic_val_loader=generic_val_loader,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--exp_name', type=str, default=g_exp_name)
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument(
        '--training_stage', type=str, required=True,
        choices=[STAGE_AUDIO, STAGE_EMOTION]
    )
    parser.add_argument('--stage1_checkpoint', type=Path, default=None)
    parser.add_argument('--resume_checkpoint', type=Path, default=None)
    parser.add_argument('--neutral_emo_index', type=int, default=5)

    # Emotion dataset.  The same motion template must also be used by Stage 1.
    parser.add_argument('--data_root', type=Path, default=Path('src/my_prepare/'))
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')

    # Generic Stage-1 dataset.
    parser.add_argument('--generic_motion_files', type=str, default='')
    parser.add_argument('--generic_aggregate_motion_files', type=str, default='')
    parser.add_argument('--generic_motion_template_path', type=Path, default=None)
    parser.add_argument('--generic_train_split', type=Path, default=None)
    parser.add_argument('--generic_val_split', type=Path, default=None)
    parser.add_argument('--generic_validation_ratio', type=float, default=0.05)
    parser.add_argument('--generic_split_seed', type=int, default=2026)
    parser.add_argument(
        '--generic_missing_audio_policy', type=str, default='skip',
        choices=['skip', 'error']
    )
    parser.add_argument(
        '--generic_duplicate_policy', type=str, default='error',
        choices=['error', 'keep_first', 'keep_last']
    )
    parser.add_argument('--stage2_validate_generic', action='store_true')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_strategy', type=str, default='random')
    parser.add_argument(
        '--normalize_type', type=str, default='mix',
        choices=['std', 'case', 'scale', 'minmax', 'mix']
    )

    # Model.
    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=50)
    parser.add_argument(
        '--diff_schedule', type=str, default='cosine',
        choices=['linear', 'cosine', 'quadratic', 'sigmoid']
    )
    parser.add_argument('--no_head_pose', action='store_true', default=False)
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])
    parser.add_argument(
        '--audio_model', type=str, default='wav2vec2',
        choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori']
    )
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--use_indicator', action='store_true', default=True)
    parser.add_argument('--feature_dim', type=int, default=512)

    # Sequence.
    parser.add_argument('--n_motions', type=int, default=100)
    parser.add_argument('--n_prev_motions', type=int, default=25)
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    # Training.
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--stage2_emotion_lr', type=float, default=5e-5)
    parser.add_argument('--stage2_backbone_lr', type=float, default=1e-5)
    parser.add_argument(
        '--stage2_train_scope', type=str, default='emotion_crossattn',
        choices=['emotion_only', 'emotion_crossattn', 'all']
    )
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument(
        '--scheduler', type=str, default='WarmupThenDecay',
        choices=['None', 'Warmup', 'WarmupThenDecay']
    )

    # Loss and classifier-free condition dropout.
    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--clip_grad_norm', type=float, default=2.0)
    parser.add_argument('--l_emo', type=float, default=0.1)
    parser.add_argument('--emotion_warmup_iter', type=int, default=5000)
    parser.add_argument('--uncond_drop_prob', type=float, default=0.1)
    parser.add_argument('--emotion_drop_prob', type=float, default=0.2)
    parser.add_argument('--l_exp', type=float, default=0.1)
    parser.add_argument('--l_exp_vel', type=float, default=1e-4)
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4)
    parser.add_argument('--l_head_angle', type=float, default=1e-2)
    parser.add_argument('--l_head_vel', type=float, default=1e-2)
    parser.add_argument('--l_head_smooth', type=float, default=1e-2)
    parser.add_argument('--l_head_trans', type=float, default=1e-2)
    parser.add_argument('--no_constrain_prev', action='store_true')

    parser.add_argument('--use_context_audio_feat', action='store_true')
    parser.add_argument('--trunc_prob1', type=float, default=0.3)
    parser.add_argument('--trunc_prob2', type=float, default=0.4)
    parser.add_argument('--val_use_truncation', action='store_true')

    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--val_iter', type=int, default=50)
    parser.add_argument('--log_iter', type=int, default=50)
    parser.add_argument('--log_smooth_win', type=int, default=50)

    parser.add_argument('--warm_iter', type=int, default=10000)
    parser.add_argument('--cos_max_iter', type=int, default=100000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)
    parser.add_argument(
        '--emotion_classifier_path', type=Path,
        default=Path('pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth')
    )

    args = parser.parse_args()
    option_text = utils.common.get_option_text(args, parser)
    main(args, option_text)

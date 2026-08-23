# Unification two-stage training
# Stage 1: generic talking-motion dataset, freeze emotion-related parameters
# Stage 2: MEAD EmotionLevel dataset, unfreeze emotion-related parameters and train the whole model

import argparse
from collections import deque, defaultdict
from pathlib import Path

import os
import sys
import logging
import math
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.utils import data

from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils
from src.dataset import infinite_data_loader
from src.dataset.dataset_GenericTalkingMotion_clear import GenericTalkingMotionDataset
from src.dataset.dataset_EmotionLevel_clear_jianhua0803 import EmoLevelDataset
from src.modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead


g_exp_name = "20260819_Unification_twostage_opt160k_mead300k"
device_id = 5

torch.cuda.set_device(device_id)
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

cross_criterion = torch.nn.CrossEntropyLoss()


# 情感核心参数：Stage 1 完全冻结；Stage 2 才开始学习。
# start_* 虽然按 emo_index 存储，但它同时承担“序列起点先验”的作用，
# 所以 Stage 1 不再冻结，而是只训练第 0 行作为 shared generic start prior，
# 每次 optimizer.step 后同步到所有 emotion row，保证 Stage-1 checkpoint 对 emo_index 真正不敏感。
EMOTION_CORE_PREFIXES = (
    "null_emotion_feat",
    "emo_embed",
    "adaLN_modulation",
)

START_PARAMETER_PREFIXES = (
    "start_motion_feat",
    "start_audio_feat",
)

AUDIO_ENCODER_PREFIXES = (
    "audio_encoder",
)


def _matches_prefix(name, prefixes):
    return any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)


def is_emotion_core_parameter(name):
    return _matches_prefix(name, EMOTION_CORE_PREFIXES)


def is_start_parameter(name):
    return _matches_prefix(name, START_PARAMETER_PREFIXES)


def is_audio_encoder_parameter(name):
    return _matches_prefix(name, AUDIO_ENCODER_PREFIXES)


def initialize_generic_stage(model):
    """
    Stage 1 使用无情感条件，但学习 shared generic start prior。

    emotion embedding / null emotion 置 0；emotion adaLN 的 bias 置 0，
    因此 dummy emo_index=0 不携带情感语义。start_* 也从 0 开始，
    但允许第 0 行在 Generic 数据上学习，然后同步给其余 7 行。
    """
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


def sync_generic_start_prior(model):
    """把 Stage-1 学到的第 0 行 shared prior 同步到全部 emotion row。"""
    with torch.no_grad():
        model.start_motion_feat[1:].copy_(
            model.start_motion_feat[0:1].expand_as(model.start_motion_feat[1:])
        )
        model.start_audio_feat[1:].copy_(
            model.start_audio_feat[0:1].expand_as(model.start_audio_feat[1:])
        )


def snapshot_initial_trainability(model):
    # 保留 wav2vec2 / HuBERT feature extractor 原本的永久冻结状态。
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def set_trainability(model, initial_trainability, mode):
    """
    mode:
      stage1       : Generic，emotion core 冻结；其余按模型原始 requires_grad。
      emotion_only : MEAD 前 10k，仅 emotion core + emotion-specific start prior。
      full         : MEAD 全量训练，恢复模型原始 trainability。
    """
    for name, parameter in model.named_parameters():
        initially_trainable = initial_trainability.get(name, True)

        if mode == 'stage1':
            trainable = initially_trainable and not is_emotion_core_parameter(name)
        elif mode == 'emotion_only':
            trainable = initially_trainable and (
                is_emotion_core_parameter(name) or is_start_parameter(name)
            )
        elif mode == 'full':
            trainable = initially_trainable
        else:
            raise ValueError(f'Unknown trainability mode: {mode}')

        parameter.requires_grad_(trainable)


def split_model_parameters(model):
    groups = {
        'backbone': [],
        'audio': [],
        'emotion': [],
        'start': [],
    }
    names = {key: [] for key in groups}

    for name, parameter in model.named_parameters():
        if is_start_parameter(name):
            key = 'start'
        elif is_emotion_core_parameter(name):
            key = 'emotion'
        elif is_audio_encoder_parameter(name):
            key = 'audio'
        else:
            key = 'backbone'
        groups[key].append(parameter)
        names[key].append(name)

    return groups, names



def reset_optimizer_group_state(optimizer, group_name):
    """清空指定参数组的 Adam 动量；保留参数值，不重建 optimizer。"""
    for group in optimizer.param_groups:
        if group.get('name') != group_name:
            continue
        for parameter in group['params']:
            optimizer.state.pop(parameter, None)


def linear_lr(local_step, total_steps, start_lr, end_lr):
    total_steps = max(1, int(total_steps))
    local_step = min(max(1, int(local_step)), total_steps)
    progress = local_step / float(total_steps)
    return start_lr + (end_lr - start_lr) * progress


def warmup_cosine_lr(local_step, total_steps, warmup_steps, start_lr, peak_lr, min_lr):
    total_steps = max(1, int(total_steps))
    warmup_steps = min(max(0, int(warmup_steps)), total_steps)
    local_step = min(max(1, int(local_step)), total_steps)

    if warmup_steps > 0 and local_step <= warmup_steps:
        return linear_lr(local_step, warmup_steps, start_lr, peak_lr)

    decay_steps = max(1, total_steps - warmup_steps)
    decay_step = max(0, local_step - warmup_steps)
    progress = min(1.0, decay_step / float(decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def cosine_lr(local_step, total_steps, peak_lr, min_lr):
    total_steps = max(1, int(total_steps))
    local_step = min(max(1, int(local_step)), total_steps)
    progress = local_step / float(total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


def get_training_phase(args, it):
    if it <= args.stage1_iter:
        return 1
    if it <= args.stage1_iter + args.stage2_emotion_only_iter:
        return 2
    return 3


def update_learning_rate(args, optimizer, it):
    """
    1) Generic: backbone 1e-4 / audio 5e-5，16k warmup 后 cosine 到 10%。
    2) MEAD emotion-only 10k: backbone/audio=0，emotion 0->8e-5，start 1e-5->4e-5。
    3) MEAD full: backbone 1e-5->2e-5，audio 5e-6，emotion/start 分组 cosine。
    """
    phase = get_training_phase(args, it)

    if phase == 1:
        backbone_lr = warmup_cosine_lr(
            it, args.stage1_iter, args.stage1_warm_iter,
            0.0, args.stage1_lr, args.stage1_lr * args.stage1_min_lr_ratio,
        )
        audio_lr = warmup_cosine_lr(
            it, args.stage1_iter, args.stage1_warm_iter,
            0.0, args.stage1_audio_lr, args.stage1_audio_lr * args.stage1_min_lr_ratio,
        )
        start_lr = backbone_lr
        emotion_lr = 0.0

    elif phase == 2:
        local_step = it - args.stage1_iter
        backbone_lr = 0.0
        audio_lr = 0.0
        emotion_lr = linear_lr(
            local_step, args.stage2_emotion_only_iter,
            0.0, args.stage2_emotion_lr,
        )
        stage1_end_start_lr = args.stage1_lr * args.stage1_min_lr_ratio
        start_lr = linear_lr(
            local_step, args.stage2_emotion_only_iter,
            stage1_end_start_lr, args.stage2_start_lr,
        )

    else:
        local_step = it - args.stage1_iter - args.stage2_emotion_only_iter
        total_steps = args.max_iter - args.stage1_iter - args.stage2_emotion_only_iter

        backbone_lr = warmup_cosine_lr(
            local_step, total_steps, args.stage2_backbone_warm_iter,
            args.stage1_lr * args.stage1_min_lr_ratio,
            args.stage2_backbone_lr,
            args.stage2_backbone_lr * args.stage2_min_lr_ratio,
        )
        audio_lr = warmup_cosine_lr(
            local_step, total_steps, args.stage2_backbone_warm_iter,
            args.stage1_audio_lr * args.stage1_min_lr_ratio,
            args.stage2_audio_lr,
            args.stage2_audio_lr * args.stage2_min_lr_ratio,
        )
        emotion_lr = cosine_lr(
            local_step, total_steps,
            args.stage2_emotion_lr,
            args.stage2_emotion_lr * args.stage2_min_lr_ratio,
        )
        start_lr = cosine_lr(
            local_step, total_steps,
            args.stage2_start_lr,
            args.stage2_start_lr * args.stage2_min_lr_ratio,
        )

    lr_dict = {
        'backbone': backbone_lr,
        'audio': audio_lr,
        'emotion': emotion_lr,
        'start': start_lr,
    }
    for group in optimizer.param_groups:
        group['lr'] = lr_dict[group['name']]

    return phase, lr_dict


def train(args,
          model,
          generic_train_loader,
          mead_train_loader,
          optimizer,
          save_dir,
          writer=None,
          start_iter=0,
          classifier=None,
          initial_trainability=None):

    save_dir.mkdir(parents=True, exist_ok=True)

    # model
    device = model.device
    model.train()

    generic_data_loader = infinite_data_loader(generic_train_loader)
    mead_data_loader = infinite_data_loader(mead_train_loader)

    # 两个数据集都必须严格保持 25 fps / 16+64=80 帧的相同训练接口。
    generic_audio_unit = generic_train_loader.dataset.audio_unit
    mead_audio_unit = mead_train_loader.dataset.audio_unit
    if abs(generic_audio_unit - mead_audio_unit) > 1e-6:
        raise RuntimeError(
            f"generic/mead audio_unit mismatch: {generic_audio_unit} vs {mead_audio_unit}"
        )
    audio_unit = mead_audio_unit

    n_audio_samples = round(audio_unit * args.n_motions)
    n_prev_audio_samples = round(audio_unit * args.n_prev_motions)
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))

    if initial_trainability is None:
        initial_trainability = snapshot_initial_trainability(model)

    next_it = start_iter + 1
    current_phase = get_training_phase(args, max(1, next_it))
    if current_phase == 1:
        set_trainability(model, initial_trainability, 'stage1')
    elif current_phase == 2:
        sync_generic_start_prior(model)
        set_trainability(model, initial_trainability, 'emotion_only')
    else:
        sync_generic_start_prior(model)
        set_trainability(model, initial_trainability, 'full')

    optimizer.zero_grad(set_to_none=True)

    # 使用 1...max_iter；默认 160k Generic + 300k MEAD = 460k useful updates。
    for it in range(start_iter + 1, args.max_iter + 1):
        phase = get_training_phase(args, it)
        is_stage1 = phase == 1

        # ---------- phase transition ----------
        if phase != current_phase:
            optimizer.zero_grad(set_to_none=True)

            if phase == 2:
                sync_generic_start_prior(model)
                reset_optimizer_group_state(optimizer, 'start')
                set_trainability(model, initial_trainability, 'emotion_only')
                logging.info(
                    f"========== Stage 2 emotion warm-start at iter {it}: "
                    "MEAD + emotion/start only =========="
                )
            elif phase == 3:
                # Generic 的 Adam 动量不直接带入 MEAD backbone/audio，避免跨域 momentum shock。
                reset_optimizer_group_state(optimizer, 'backbone')
                reset_optimizer_group_state(optimizer, 'audio')
                set_trainability(model, initial_trainability, 'full')
                logging.info(
                    f"========== Stage 2 full training at iter {it}: "
                    "MEAD + full model =========="
                )

            current_phase = phase
            if writer is not None:
                writer.add_text('train/phase_transition', f'phase {phase} starts at iter {it}', it)

        phase, lr_dict = update_learning_rate(args, optimizer, it)

        # ---------- Dataset ----------
        if is_stage1:
            # GenericTalkingMotionDataset: audio, coef_dict
            audio, coef_dict = next(generic_data_loader)
            batch_size = audio.shape[0]
            # 仅用于兼容 jianhua0803 模型当前的函数签名。
            # Stage 1 的 emotion core 为零且冻结；start prior 为 shared generic prior。
            # 因此 dummy index=0 不代表 angry 等任何具体类别。
            emo_index = torch.zeros(batch_size, dtype=torch.long)
            use_emotion_loss = False
        else:
            # EmoLevelDataset: audio, coef_dict, emo_index, emo_level
            audio, coef_dict, emo_index, _ = next(mead_data_loader)
            use_emotion_loss = True

        audio = audio.to(device, non_blocking=True)
        coef_dict = {
            k: coef_dict[k].to(device, non_blocking=True)
            for k in coef_dict
        }
        motion_coef_full = utils.get_motion_coef(
            coef_dict, args.rot_repr, predict_head_pose
        )
        emo_index = emo_index.to(device, non_blocking=True)
        batch_size = audio.shape[0]

        # 单数轮次：无 GT prev；双数轮次：使用真实的前 16 帧。
        is_starting_sample = (it % 2 == 1)

        # ---------- 截断逻辑（仅作用于 current 部分；prev 不截断） ----------
        current_audio = audio[:, -n_audio_samples:].contiguous()
        current_motion = motion_coef_full[:, -args.n_motions:]
        end_idx = None
        trunc_prob = args.trunc_prob1 if is_starting_sample else args.trunc_prob2
        if np.random.rand() < trunc_prob:
            current_audio, current_motion, end_idx = utils.truncate_motion_coef_and_audio(
                current_audio,
                current_motion,
                args.n_motions,
                audio_unit,
                args.pad_mode,
            )

        # ---------- 指示器 ----------
        if args.use_indicator:
            if end_idx is not None:
                indicator = torch.arange(args.n_motions, device=device).expand(
                    batch_size, -1
                ) < end_idx.unsqueeze(1)
            else:
                indicator = torch.ones(batch_size, args.n_motions, device=device)
        else:
            indicator = None

        # ---------- 模型前向 ----------
        if is_starting_sample:
            noise, target, _, _ = model(
                current_motion,
                current_audio,
                indicator=indicator,
                emo_index=emo_index,
            )
            prev_motion_for_loss = None
        else:
            prev_motion_gt = motion_coef_full[:, :args.n_prev_motions]
            prev_audio_raw = audio[:, :n_prev_audio_samples].contiguous()
            with torch.no_grad():
                prev_audio_feat = model.extract_audio_feature(
                    prev_audio_raw,
                    frame_num=args.n_prev_motions,
                )
            noise, target, _, _ = model(
                current_motion,
                current_audio,
                prev_motion_gt,
                prev_audio_feat,
                indicator=indicator,
                emo_index=emo_index,
            )
            prev_motion_for_loss = prev_motion_gt

        # ---------- 损失 ----------
        loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(
            args,
            is_starting_sample,
            current_motion,
            noise,
            target,
            prev_motion_for_loss,
            end_idx,
        )

        loss_noise = loss_n
        loss_expression = loss_exp
        loss_exp_vel = loss_exp_v
        loss_exp_smooth = loss_exp_s
        loss_head_angle = loss_ha if (
            args.target == 'sample'
            and predict_head_pose
            and args.l_head_angle > 0
            and loss_ha is not None
        ) else torch.tensor(0.0, device=device)
        loss_head_vel = loss_hc if (
            args.target == 'sample'
            and predict_head_pose
            and args.l_head_vel > 0
            and loss_hc is not None
        ) else torch.tensor(0.0, device=device)
        loss_head_smooth = loss_hs if (
            args.target == 'sample'
            and predict_head_pose
            and args.l_head_smooth > 0
            and loss_hs is not None
        ) else torch.tensor(0.0, device=device)
        loss_head_trans = loss_ht if (
            args.target == 'sample'
            and predict_head_pose
            and args.l_head_trans > 0
            and loss_ht is not None
        ) else torch.tensor(0.0, device=device)

        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise

        # Stage 1 没有情感标签，不计算 classifier CE；Stage 2 与原 jianhua0803 完全一致。
        if use_emotion_loss:
            exps = target[:, args.n_prev_motions:, :63].clone()
            pred_emo, _ = classifier(exps)
            loss_emo = cross_criterion(pred_emo, emo_index)
            loss = loss + args.l_emo * loss_emo
            loss_log['emo'].append(loss_emo.item() * args.l_emo)
        else:
            loss_emo = torch.tensor(0.0, device=device)
            loss_log['emo'].append(0.0)

        loss_log['exp'].append(loss_expression.item() * args.l_exp)
        loss = loss + args.l_exp * loss_expression

        loss_log['exp_vel'].append(loss_exp_vel.item() * args.l_exp_vel)
        loss = loss + args.l_exp_vel * loss_exp_vel

        loss_log['exp_smooth'].append(loss_exp_smooth.item() * args.l_exp_smooth)
        loss = loss + args.l_exp_smooth * loss_exp_smooth

        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            loss_log['head_angle'].append(loss_head_angle.item() * args.l_head_angle)
            loss = loss + args.l_head_angle * loss_head_angle
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            loss_log['head_vel'].append(loss_head_vel.item() * args.l_head_vel)
            loss = loss + args.l_head_vel * loss_head_vel
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            loss_log['head_smooth'].append(loss_head_smooth.item() * args.l_head_smooth)
            loss = loss + args.l_head_smooth * loss_head_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            loss_log['head_trans'].append(loss_head_trans.item() * args.l_head_trans)
            loss = loss + args.l_head_trans * loss_head_trans

        # 梯度累积时必须除以 accumulation steps，避免有效学习率被额外放大。
        loss_for_backward = loss / args.gradient_accumulation_steps
        loss_for_backward.backward()

        if it % args.gradient_accumulation_steps == 0:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=2.0,
                )
            optimizer.step()
            if is_stage1:
                # 保证 Stage-1 checkpoint 的 8 个 start row 完全一致。
                sync_generic_start_prior(model)
            optimizer.zero_grad(set_to_none=True)

        # ---------- Logging ----------
        loss_log['loss'].append(loss.item())
        stage_name = {1: 'generic', 2: 'mead_emotion_warm', 3: 'mead_full'}[phase]
        description = f'Iter: {it}\t Stage: {stage_name}\t Train loss: [N: {np.mean(loss_log["noise"]):.3e}'
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
        description += f", Emo: {np.mean(loss_log['emo']):.3e}"
        description += (
            f", LR[B/A/E/S]: {lr_dict['backbone']:.3e}/"
            f"{lr_dict['audio']:.3e}/{lr_dict['emotion']:.3e}/"
            f"{lr_dict['start']:.3e}]"
        )
        logging.info(description)

        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/stage', 1 if is_stage1 else 2, it)
            writer.add_scalar('train/phase', phase, it)
            writer.add_scalar('train/total_loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/emotion_loss', np.mean(loss_log['emo']), it)
            writer.add_scalar('train/simple_loss', np.mean(loss_log['noise']), it)
            writer.add_scalar('train/exp_loss', np.mean(loss_log['exp']), it)
            writer.add_scalar('train/exp_vel_loss', np.mean(loss_log['exp_vel']), it)
            writer.add_scalar('train/exp_smooth_loss', np.mean(loss_log['exp_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                writer.add_scalar('train/head_angle', np.mean(loss_log['head_angle']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                writer.add_scalar('train/head_vel', np.mean(loss_log['head_vel']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                writer.add_scalar('train/head_smooth', np.mean(loss_log['head_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                writer.add_scalar('train/head_trans', np.mean(loss_log['head_trans']), it)
            writer.add_scalar('opt/lr_backbone', lr_dict['backbone'], it)
            writer.add_scalar('opt/lr_audio', lr_dict['audio'], it)
            writer.add_scalar('opt/lr_emotion', lr_dict['emotion'], it)
            writer.add_scalar('opt/lr_start', lr_dict['start'], it)

        # ---------- save model ----------
        should_save = (
            (it % args.save_iter == 0)
            or it == args.stage1_iter
            or it == args.stage1_iter + args.stage2_emotion_only_iter
            or it == args.max_iter
        )
        if should_save:
            torch.save({
                'args': args,
                'model': model.state_dict(),
                'iter': it,
                'stage': 1 if is_stage1 else 2,
                'phase': phase,
            }, save_dir / f'iter_{it:07}.pt')


# 获取训练参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main(args, option_text=None):
    if args.stage1_iter <= 0 or args.stage1_iter >= args.max_iter:
        raise ValueError(
            f"stage1_iter should be in (0, max_iter), got "
            f"{args.stage1_iter}/{args.max_iter}"
        )

    if args.stage1_iter + args.stage2_emotion_only_iter >= args.max_iter:
        raise ValueError(
            "stage2_emotion_only_iter leaves no room for full Stage-2 training: "
            f"{args.stage1_iter} + {args.stage2_emotion_only_iter} >= {args.max_iter}"
        )

    # model
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
        align_mask_width=args.align_mask_width,
    )
    model = DitTalkingHead(**model_kwargs)

    # 记录模型原生冻结状态（尤其 wav2vec2/HubERT feature extractor），
    # 再建立 Stage-1 zero-emotion + learnable shared-start 状态。
    initial_trainability = snapshot_initial_trainability(model)
    initialize_generic_stage(model)
    sync_generic_start_prior(model)
    set_trainability(model, initial_trainability, 'stage1')

    exp_dir = Path('experiments/emo_dit') / f'{args.exp_name}'
    start_iter = 0

    # ========== Dataset: Stage 1 generic ==========
    if not args.generic_motion_filenames and not args.generic_aggregate_motion_files:
        raise ValueError(
            "Stage 1 requires one generic motion source. Please set either "
            "--generic_motion_filenames or --generic_aggregate_motion_files."
        )

    generic_template_path = args.generic_motion_template_path
    if generic_template_path is None:
        # 默认强制 Stage 1 / Stage 2 共用同一份 normalization template，
        # 避免两阶段 motion coefficient 分布发生人为偏移。
        generic_template_path = Path(args.data_root) / args.motion_template_filename

    generic_train_dataset = GenericTalkingMotionDataset(
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

    generic_train_loader = data.DataLoader(
        generic_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # ========== Dataset: Stage 2 MEAD ==========
    mead_train_dataset = EmoLevelDataset(
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

    mead_train_loader = data.DataLoader(
        mead_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    # 情感分类器只在 Stage 2 参与 loss。
    classifier = Classifier().to(device)
    classifier.load_state_dict(
        torch.load(args.emotion_classifier_path, map_location=device),
        strict=False,
    )
    classifier.eval()
    # 冻结 classifier 本身，但保留从 classifier 输出到 target 的梯度路径。
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    # Logging
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / 'options.log', 'w') as f:
            f.write(option_text)
        writer.add_text('options', option_text)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), "log.txt"),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"exp_name: {exp_dir.name}")
    logging.info(f"all model parameters: {count_all_parameters(model)}")
    logging.info(f"Stage 1 trainable parameters: {count_parameters(model)}")

    parameter_groups, parameter_names = split_model_parameters(model)
    for group_name in ['backbone', 'audio', 'emotion', 'start']:
        logging.info(
            f"{group_name} parameter tensors: {len(parameter_groups[group_name])}"
        )

    logging.info("Stage 1 frozen emotion-core parameters:")
    for name in parameter_names['emotion']:
        logging.info(f"  {name}")

    # optimizer 一开始就注册 4 个 group。之后只切 requires_grad 与各组 LR，
    # 不重建 Adam，避免 phase 边界丢失已有 optimizer state。
    optimizer = torch.optim.Adam([
        {
            'params': parameter_groups['backbone'],
            'lr': args.stage1_lr,
            'name': 'backbone',
        },
        {
            'params': parameter_groups['audio'],
            'lr': args.stage1_audio_lr,
            'name': 'audio',
        },
        {
            'params': parameter_groups['emotion'],
            'lr': 0.0,
            'name': 'emotion',
        },
        {
            'params': parameter_groups['start'],
            'lr': args.stage1_lr,
            'name': 'start',
        },
    ])

    train(
        args,
        model,
        generic_train_loader,
        mead_train_loader,
        optimizer,
        exp_dir / 'checkpoints',
        writer,
        start_iter=start_iter,
        classifier=classifier,
        initial_trainability=initial_trainability,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--exp_name', type=str, default=g_exp_name, help='experiment name')
    parser.add_argument('--model_variant', type=str, default='jianhua0803', help='saved for inference loader routing')

    # Dataset - Stage 2 MEAD
    parser.add_argument('--data_root', type=Path, default="src/my_prepare/")
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')

    # Dataset - Stage 1 generic
    parser.add_argument(
        '--generic_motion_filenames',
        type=str,
        default='src/data_processing/HDTF/front_all_motions.pkl,src/my_prepare/front_all_motions.pkl',
        help='comma-separated per-dataset motion pkl files',
    )
    parser.add_argument(
        '--generic_aggregate_motion_files',
        type=str,
        default='',
        help='comma-separated merged/aggregate motion pkl files; mutually exclusive with generic_motion_filenames',
    )
    parser.add_argument(
        '--generic_motion_template_path',
        type=Path,
        default=None,
        help='default: data_root/motion_template_filename, so Stage 1 and Stage 2 share normalization',
    )
    parser.add_argument('--generic_split_file', type=Path, default=None)
    parser.add_argument('--generic_validation_ratio', type=float, default=0.0)
    parser.add_argument('--generic_split_seed', type=int, default=2026)
    parser.add_argument('--generic_allow_relative_paths', action='store_true', default=False)
    parser.add_argument(
        '--generic_missing_audio_policy',
        type=str,
        default='skip',
        choices=['skip', 'error'],
    )
    parser.add_argument(
        '--generic_duplicate_policy',
        type=str,
        default='keep_first',
        choices=['error', 'keep_first', 'keep_last'],
    )

    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--crop_strategy', type=str, default="random")
    parser.add_argument(
        '--normalize_type',
        type=str,
        default="mix",
        choices=["std", "case", "scale", "minmax", "mix"],
    )

    # Model
    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,emotion')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=500, help='number of diffusion steps')
    parser.add_argument(
        '--diff_schedule',
        type=str,
        default='cosine',
        choices=['linear', 'cosine', 'quadratic', 'sigmoid'],
    )
    parser.add_argument('--no_head_pose', action='store_true', default=False, help='do not predict head pose')
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])

    # transformer
    parser.add_argument(
        '--audio_model',
        type=str,
        default='wav2vec2',
        choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'],
    )
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=2, help='width of the alignment mask')
    parser.add_argument('--no_use_learnable_pe', action='store_true', help='do not use learnable positional encoding')
    parser.add_argument('--use_indicator', action='store_true', default=True, help='use indicator for padded frames')
    parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=4)

    # sequence
    parser.add_argument('--n_motions', type=int, default=64, help='current motion frames')
    parser.add_argument('--n_prev_motions', type=int, default=16, help='previous motion frames')
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    # Training - two stage
    # Generic 在原曲线 150k~170k 已明显进入收益递减，而 MEAD 在 200k Stage-2 updates 后仍在下降。
    # 默认改为 160k Generic + 300k MEAD = 460k useful updates。
    parser.add_argument('--max_iter', type=int, default=460000, help='total useful training updates')
    parser.add_argument('--stage1_iter', type=int, default=160000, help='last generic-training iteration')

    # Stage 1: Generic audio-motion learning。
    parser.add_argument('--stage1_lr', type=float, default=1e-4)
    parser.add_argument('--stage1_audio_lr', type=float, default=5e-5)
    parser.add_argument('--stage1_warm_iter', type=int, default=16000)
    parser.add_argument('--stage1_min_lr_ratio', type=float, default=0.10)

    # Stage 2 前 10k：只让 emotion + start prior 适应 MEAD。
    parser.add_argument('--stage2_emotion_only_iter', type=int, default=10000)
    parser.add_argument('--stage2_emotion_lr', type=float, default=8e-5)
    parser.add_argument('--stage2_start_lr', type=float, default=4e-5)

    # Stage 2 full：已预训练 backbone 使用较小 LR，audio encoder 更小。
    parser.add_argument('--stage2_backbone_lr', type=float, default=2e-5)
    parser.add_argument('--stage2_audio_lr', type=float, default=5e-6)
    parser.add_argument('--stage2_backbone_warm_iter', type=int, default=10000)
    parser.add_argument('--stage2_min_lr_ratio', type=float, default=0.10)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    # 损失函数 & 权重
    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--l_emo', type=float, default=1.0, help='Stage 2 emotion classification loss weight')
    parser.add_argument('--l_exp', type=float, default=1.0)
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

    parser.add_argument('--save_iter', type=int, default=5000)
    parser.add_argument('--log_iter', type=int, default=50)
    parser.add_argument('--log_smooth_win', type=int, default=50)

    parser.add_argument(
        '--emotion_classifier_path',
        type=str,
        default='experiments/emo_classifier/ckpt_n64.pth',
    )

    args = parser.parse_args()

    option_text = utils.common.get_option_text(args, parser)
    main(args, option_text)

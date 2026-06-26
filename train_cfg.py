# 在 train.py 的基础上改动：使用 emotion_dit_cfg_clear，并仅在 full CFG 分支计算情感损失

import argparse
from collections import deque, defaultdict
from pathlib import Path

import os
import pickle
import sys
import logging
import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils
from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel import EmoLevelDataset
from src.modules.emotion_dit_cfg_clear import DitTalkingHead

g_exp_name = '20260625_CFG_clear_full分支情感损失'
device_id = 6

torch.cuda.set_device(device_id)
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

cross_criterion = torch.nn.CrossEntropyLoss()


def compute_full_branch_emotion_loss(target, emo_index, cfg_is_full, classifier, args, device):
    exps = target[:, args.n_prev_motions:, :63].clone()
    if cfg_is_full.any():
        pred_emo, _ = classifier(exps[cfg_is_full])
        return cross_criterion(pred_emo, emo_index[cfg_is_full])
    return torch.zeros((), device=device)


def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    device = model.device
    model.train()

    data_loader = infinite_data_loader(train_loader)
    audio_unit = train_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))

    optimizer.zero_grad()
    for it in range(start_iter, args.max_iter + 1):
        audio_pair, coef_pair, emo_index, _ = next(data_loader)
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
        motion_coef_pair = [utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)]
        emo_index = emo_index.to(device)

        if args.use_context_audio_feat:
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)

        loss_noise = torch.tensor(0, device=device)
        loss_emo = torch.tensor(0, device=device)
        loss_expression = torch.tensor(0, device=device)
        loss_exp_vel = torch.tensor(0, device=device)
        loss_exp_smooth = torch.tensor(0, device=device)
        loss_head_angle = torch.tensor(0, device=device)
        loss_head_vel = torch.tensor(0, device=device)
        loss_head_smooth = torch.tensor(0, device=device)
        loss_head_trans = torch.tensor(0, device=device)

        for i in range(2):
            audio = audio_pair[i]
            motion_coef = motion_coef_pair[i]
            batch_size = audio.shape[0]

            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                if args.use_context_audio_feat and i != 0:
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                           args.n_motions * 2)[:, -args.n_motions:]
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
                noise, target, prev_motion_coef, prev_audio_feat, cfg_is_full, cfg_state = model(
                    motion_coef_in,
                    audio_in,
                    indicator=indicator,
                    emo_index=emo_index,
                    return_cfg_state=True)
                if end_idx is not None:
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
            else:
                noise, target, _, _, cfg_is_full, cfg_state = model(
                    motion_coef_in,
                    audio_in,
                    prev_motion_coef,
                    prev_audio_feat,
                    indicator=indicator,
                    emo_index=emo_index,
                    return_cfg_state=True)

            loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(
                args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)

            loss_e = compute_full_branch_emotion_loss(target, emo_index, cfg_is_full, classifier, args, device)
            loss_emo = loss_emo + loss_e / 2
            loss_noise = loss_noise + loss_n / 2
            loss_expression = loss_expression + loss_exp / 2
            loss_exp_vel = loss_exp_vel + loss_exp_v / 2
            loss_exp_smooth = loss_exp_smooth + loss_exp_s / 2

            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_head_angle = loss_head_angle + loss_ha / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                loss_head_vel = loss_head_vel + loss_hc / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                loss_head_trans = loss_head_trans + loss_ht

        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise

        loss_log['emo'].append(loss_emo.item())
        loss = loss + loss_emo

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

        loss.backward()

        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_log['loss'].append(loss.item())
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
        description += f", Emo: {np.mean(loss_log['emo']):.3e}"
        description += ']'
        logging.info(description)

        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/emotion_loss_full_branch', np.mean(loss_log['emo']), it)
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
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)

        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save({'args': args, 'model': model.state_dict(), 'iter': it}, save_dir / f'iter_{it:07}.pt')

        if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:
            val(args, model, val_loader, it, 1, 'val', writer, classifier)


@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode='val', writer=None, classifier=None):
    is_training = model.training
    device = model.device
    model.eval()

    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    loss_log = defaultdict(list)

    for test_round in range(n_rounds):
        for audio_pair, coef_pair, emo_index, _ in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            motion_coef_pair = [utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)]
            emo_index = emo_index.to(device)

            if args.use_context_audio_feat:
                audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)

            loss_noise = torch.tensor(0, device=device)
            loss_emo = torch.tensor(0, device=device)
            loss_expression = torch.tensor(0, device=device)
            loss_exp_vel = torch.tensor(0, device=device)
            loss_exp_smooth = torch.tensor(0, device=device)
            loss_head_angle = torch.tensor(0, device=device)
            loss_head_vel = torch.tensor(0, device=device)
            loss_head_smooth = torch.tensor(0, device=device)
            loss_head_trans = torch.tensor(0, device=device)

            for i in range(2):
                audio = audio_pair[i]
                motion_coef = motion_coef_pair[i]
                batch_size = audio.shape[0]

                if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                    audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                    if args.use_context_audio_feat and i != 0:
                        audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                               args.n_motions * 2)[:, -args.n_motions:]
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
                    noise, target, prev_motion_coef, prev_audio_feat, cfg_is_full, cfg_state = model(
                        motion_coef_in,
                        audio_in,
                        indicator=indicator,
                        emo_index=emo_index,
                        return_cfg_state=True)
                    if end_idx is not None:
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:
                            prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        else:
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    noise, target, _, _, cfg_is_full, cfg_state = model(
                        motion_coef_in,
                        audio_in,
                        prev_motion_coef,
                        prev_audio_feat,
                        indicator=indicator,
                        emo_index=emo_index,
                        return_cfg_state=True)

                loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(
                    args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)

                loss_e = compute_full_branch_emotion_loss(target, emo_index, cfg_is_full, classifier, args, device)
                loss_emo = loss_emo + loss_e / 2
                loss_noise = loss_noise + loss_n / 2
                loss_expression = loss_expression + loss_exp / 2
                loss_exp_vel = loss_exp_vel + loss_exp_v / 2
                loss_exp_smooth = loss_exp_smooth + loss_exp_s / 2

                if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                    loss_head_angle = loss_head_angle + loss_ha / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                    loss_head_vel = loss_head_vel + loss_hc / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                    loss_head_smooth = loss_head_smooth + loss_hs / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                    loss_head_trans = loss_head_trans + loss_ht

            loss_log['noise'].append(loss_noise.item())
            loss = loss_noise
            loss_log['emo'].append(loss_emo.item())
            loss = loss + loss_emo
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
            loss_log['loss'].append(loss.item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [N: {np.mean(loss_log["noise"]):.3e}'
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
    description += ']'
    print(description)

    if writer is not None:
        writer.add_scalar(f'{mode}/total_loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/emotion_loss_full_branch', np.mean(loss_log['emo']), current_iter)
        writer.add_scalar(f'{mode}/simple_loss', np.mean(loss_log['noise']), current_iter)
        writer.add_scalar(f'{mode}/exp_loss', np.mean(loss_log['exp']), current_iter)
        writer.add_scalar(f'{mode}/exp_vel_loss', np.mean(loss_log['exp_vel']), current_iter)
        writer.add_scalar(f'{mode}/exp_smooth_loss', np.mean(loss_log['exp_smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
            writer.add_scalar(f'{mode}/head_angle', np.mean(loss_log['head_angle']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
            writer.add_scalar(f'{mode}/head_vel', np.mean(loss_log['head_vel']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
            writer.add_scalar(f'{mode}/head_smooth', np.mean(loss_log['head_smooth']), current_iter)
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
            writer.add_scalar(f'{mode}/head_trans', np.mean(loss_log['head_trans']), current_iter)

    if is_training:
        model.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args, option_text=None):
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
    )

    model = DitTalkingHead(**model_kwargs)
    exp_dir = Path('experiments/emo_dit') / f'{args.exp_name}'
    start_iter = 0

    train_dataset = EmoLevelDataset(args.data_root,
                                    motion_filename=args.motion_filename,
                                    motion_template_filename=args.motion_template_filename,
                                    split='train',
                                    coef_fps=args.fps,
                                    n_motions=args.n_motions,
                                    crop_strategy=args.crop_strategy,
                                    normalize_type=args.normalize_type)
    val_dataset = EmoLevelDataset(args.data_root,
                                  motion_filename=args.motion_filename,
                                  motion_template_filename=args.motion_template_filename,
                                  split='val',
                                  coef_fps=args.fps,
                                  n_motions=args.n_motions,
                                  crop_strategy=args.crop_strategy,
                                  normalize_type=args.normalize_type)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load('pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth', map_location=device), strict=False)
    classifier.eval()

    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / 'options.log', 'w') as f:
            f.write(option_text)
        writer.add_text('options', option_text)

    logging.basicConfig(filename=os.path.join(str(log_dir), 'log.txt'),
                        level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'exp_name: {exp_dir.name}')
    logging.info(f'model parameters: {count_parameters(model)}')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.scheduler == 'Warmup':
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == 'WarmupThenDecay':
        from src.scheduler import GradualWarmupScheduler
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.cos_max_iter - args.warm_iter,
                                                               args.lr * args.min_lr_ratio)
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
    else:
        scheduler = None

    train(args, model, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', scheduler, writer,
          start_iter=start_iter, classifier=classifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--iter', type=int, default=1, help='iteration to test')
    parser.add_argument('--exp_name', type=str, default=g_exp_name, help='experiment name')

    parser.add_argument('--data_root', type=Path, default='src/my_prepare/')
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--crop_strategy', type=str, default='random')
    parser.add_argument('--normalize_type', type=str, default='mix', choices=['std', 'case', 'scale', 'minmax', 'mix'])

    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,emotion')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=50, help='number of diffusion steps')
    parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--no_head_pose', action='store_true', default=False, help='do not predict head pose')
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])

    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=1, help='width of the alignment mask, non-positive for no mask')
    parser.add_argument('--no_use_learnable_pe', action='store_true', help='do not use learnable positional encoding')
    parser.add_argument('--use_indicator', action='store_true', default=True, help='use indicator for padded frames')
    parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    parser.add_argument('--n_motions', type=int, default=100, help='number of motions in a sequence')
    parser.add_argument('--n_prev_motions', type=int, default=25, help='number of pre-motions in a sequence')
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25, help='frame per second')
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    parser.add_argument('--max_iter', type=int, default=100000, help='max number of iterations')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--scheduler', type=str, default='WarmupThenDecay', choices=['None', 'Warmup', 'WarmupThenDecay'])

    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--l_exp', type=float, default=0.1, help='weight of the expression loss')
    parser.add_argument('--l_exp_vel', type=float, default=1e-4, help='weight of the expression velocity loss')
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4, help='weight of the expression smooth loss')
    parser.add_argument('--l_head_angle', type=float, default=1e-2, help='weight of the head angle loss')
    parser.add_argument('--l_head_vel', type=float, default=1e-2, help='weight of the head angular velocity loss')
    parser.add_argument('--l_head_smooth', type=float, default=1e-2, help='weight of the head angular acceleration regularization')
    parser.add_argument('--l_head_trans', type=float, default=1e-2, help='weight of the head constraint during window transition')
    parser.add_argument('--no_constrain_prev', action='store_true', help='do not constrain the generated previous motions')

    parser.add_argument('--use_context_audio_feat', action='store_true')
    parser.add_argument('--trunc_prob1', type=float, default=0.3, help='truncation probability for the first sample')
    parser.add_argument('--trunc_prob2', type=float, default=0.4, help='truncation probability for the second sample')

    parser.add_argument('--save_iter', type=int, default=1000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=50, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')

    parser.add_argument('--warm_iter', type=int, default=10000)
    parser.add_argument('--cos_max_iter', type=int, default=100000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)

    args = parser.parse_args()
    if args.mode == 'train':
        option_text = utils.common.get_option_text(args, parser)
    else:
        option_text = None
    main(args, option_text)

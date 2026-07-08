# coding: utf-8
"""Train the emotion2vec-conditioned ADEF motion generator.

This is a copy-style training entrypoint.  It does not modify ``train.py``.
It expects the dataset copy ``src/dataset/dataset_EmotionLevel_e2v.py`` and the
model copy ``src/modules/emotion_dit_e2v.py``.
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
from src.modules.emotion_dit_e2v import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
from src.utils.e2v_losses import compute_prosody_curve_loss


def truncate_e2v_frame_feature(frame_feat, end_idx, pad_mode='zero'):
    """Apply the same truncation mask as audio/motion to frame-level e2v features."""
    out = frame_feat.clone()
    B = frame_feat.shape[0]
    if pad_mode == 'zero':
        for b in range(B):
            out[b, end_idx[b]:] = 0
    elif pad_mode == 'replicate':
        for b in range(B):
            out[b, end_idx[b]:] = out[b, end_idx[b] - 1]
    else:
        raise ValueError(f'Unknown pad mode: {pad_mode}')
    return out


def make_valid_mask(batch_size, n_motions, end_idx, device):
    if end_idx is None:
        return torch.ones(batch_size, n_motions, dtype=torch.bool, device=device)
    return torch.arange(n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(1)


def add_if_valid(loss, weight, value):
    if value is None or weight <= 0:
        return loss
    return loss + weight * value


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
    cross_criterion = torch.nn.CrossEntropyLoss()

    optimizer.zero_grad()
    for it in range(start_iter, args.max_iter + 1):
        audio_pair, coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair = next(data_loader)
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
        e2v_utt_pair = [x.to(device).float() for x in e2v_utt_pair]
        e2v_frame_pair = [x.to(device).float() for x in e2v_frame_pair]
        motion_coef_pair = [utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)]
        emo_index = emo_index.to(device)
        emo_level = emo_level.to(device)

        if args.use_context_audio_feat:
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)

        loss_noise = torch.tensor(0.0, device=device)
        loss_emo = torch.tensor(0.0, device=device)
        loss_level = torch.tensor(0.0, device=device)
        loss_prosody = torch.tensor(0.0, device=device)
        loss_expression = torch.tensor(0.0, device=device)
        loss_exp_vel = torch.tensor(0.0, device=device)
        loss_exp_smooth = torch.tensor(0.0, device=device)
        loss_head_angle = torch.tensor(0.0, device=device)
        loss_head_vel = torch.tensor(0.0, device=device)
        loss_head_smooth = torch.tensor(0.0, device=device)
        loss_head_trans = torch.tensor(0.0, device=device)

        prev_emo_frame_feat = None
        for i in range(2):
            audio = audio_pair[i]
            motion_coef = motion_coef_pair[i]
            e2v_utt = e2v_utt_pair[i]
            e2v_frame = e2v_frame_pair[i]
            batch_size = audio.shape[0]

            if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode
                )
                e2v_frame_in = truncate_e2v_frame_feature(e2v_frame, end_idx, args.pad_mode)
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
                e2v_frame_in = e2v_frame

            if args.use_indicator:
                indicator = make_valid_mask(batch_size, args.n_motions, end_idx, device)
            else:
                indicator = None

            if i == 0:
                noise, target, prev_motion_coef, prev_audio_feat, prev_emo_frame_feat_out = model(
                    motion_coef_in,
                    audio_in,
                    indicator=indicator,
                    emo_index=emo_index,
                    emo_utt_feat=e2v_utt,
                    emo_frame_feat=e2v_frame_in,
                )
                if end_idx is not None:
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                    if args.use_context_audio_feat:
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    prev_emo_frame_feat = e2v_frame[:, -args.n_prev_motions:]
                else:
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                    prev_emo_frame_feat = prev_emo_frame_feat_out[:, -args.n_prev_motions:] if prev_emo_frame_feat_out is not None else None
            else:
                noise, target, _, _, _ = model(
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

            loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hv, loss_hs, loss_ht = utils.compute_loss_new(
                args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx
            )

            pred_motion_current = target[:, args.n_prev_motions:]
            valid_mask = make_valid_mask(batch_size, args.n_motions, end_idx, device)
            if args.l_prosody_curve > 0:
                loss_p = compute_prosody_curve_loss(
                    pred_motion_current,
                    e2v_frame_in,
                    mask=valid_mask,
                    rot_repr=args.rot_repr,
                    use_velocity=not args.no_prosody_velocity,
                    vel_weight_motion=args.prosody_motion_vel_weight,
                    vel_weight_audio=args.prosody_audio_vel_weight,
                )
                loss_prosody = loss_prosody + loss_p / 2

            if classifier is not None:
                pred_emo, pred_level = classifier(pred_motion_current[..., :63].clone())
                loss_emo = loss_emo + cross_criterion(pred_emo, emo_index) / 2
                if args.l_emo_level > 0:
                    loss_level = loss_level + cross_criterion(pred_level, emo_level) / 2

            loss_noise = loss_noise + loss_n / 2
            loss_expression = loss_expression + loss_exp / 2
            if loss_exp_v is not None:
                loss_exp_vel = loss_exp_vel + loss_exp_v / 2
            if loss_exp_s is not None:
                loss_exp_smooth = loss_exp_smooth + loss_exp_s / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_head_angle = loss_head_angle + loss_ha / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hv is not None:
                loss_head_vel = loss_head_vel + loss_hv / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                loss_head_trans = loss_head_trans + loss_ht

        loss = loss_noise
        loss = loss + args.l_emo_cls * loss_emo
        loss = loss + args.l_emo_level * loss_level
        loss = loss + args.l_prosody_curve * loss_prosody
        loss = add_if_valid(loss, args.l_exp, loss_expression)
        loss = add_if_valid(loss, args.l_exp_vel, loss_exp_vel)
        loss = add_if_valid(loss, args.l_exp_smooth, loss_exp_smooth)
        if args.target == 'sample' and predict_head_pose:
            loss = add_if_valid(loss, args.l_head_angle, loss_head_angle)
            loss = add_if_valid(loss, args.l_head_vel, loss_head_vel)
            loss = add_if_valid(loss, args.l_head_smooth, loss_head_smooth)
            loss = add_if_valid(loss, args.l_head_trans, loss_head_trans)

        loss.backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        if it % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        loss_log['loss'].append(loss.item())
        loss_log['noise'].append(loss_noise.item())
        loss_log['emo'].append((args.l_emo_cls * loss_emo).item())
        loss_log['level'].append((args.l_emo_level * loss_level).item())
        loss_log['prosody'].append((args.l_prosody_curve * loss_prosody).item())
        loss_log['exp'].append((args.l_exp * loss_expression).item())
        loss_log['exp_vel'].append((args.l_exp_vel * loss_exp_vel).item())
        loss_log['exp_smooth'].append((args.l_exp_smooth * loss_exp_smooth).item())
        if args.target == 'sample' and predict_head_pose:
            loss_log['head_angle'].append((args.l_head_angle * loss_head_angle).item())
            loss_log['head_vel'].append((args.l_head_vel * loss_head_vel).item())
            loss_log['head_smooth'].append((args.l_head_smooth * loss_head_smooth).item())
            loss_log['head_trans'].append((args.l_head_trans * loss_head_trans).item())

        description = f'Iter: {it}\t Train loss: [N: {np.mean(loss_log["noise"]):.3e}'
        description += f", EX: {np.mean(loss_log['exp']):.3e}, Emo: {np.mean(loss_log['emo']):.3e}"
        description += f", Level: {np.mean(loss_log['level']):.3e}, Prosody: {np.mean(loss_log['prosody']):.3e}]"
        logging.info(description)

        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['loss']), it)
            writer.add_scalar('train/simple_loss', np.mean(loss_log['noise']), it)
            writer.add_scalar('train/emotion_loss', np.mean(loss_log['emo']), it)
            writer.add_scalar('train/level_loss', np.mean(loss_log['level']), it)
            writer.add_scalar('train/prosody_curve_loss', np.mean(loss_log['prosody']), it)
            writer.add_scalar('train/exp_loss', np.mean(loss_log['exp']), it)
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
    if classifier is not None:
        classifier.eval()

    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose
    cross_criterion = torch.nn.CrossEntropyLoss()
    loss_log = defaultdict(list)

    for _ in range(n_rounds):
        for audio_pair, coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            e2v_utt_pair = [x.to(device).float() for x in e2v_utt_pair]
            e2v_frame_pair = [x.to(device).float() for x in e2v_frame_pair]
            motion_coef_pair = [utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)]
            emo_index = emo_index.to(device)
            emo_level = emo_level.to(device)

            loss_noise = torch.tensor(0.0, device=device)
            loss_emo = torch.tensor(0.0, device=device)
            loss_level = torch.tensor(0.0, device=device)
            loss_prosody = torch.tensor(0.0, device=device)
            loss_expression = torch.tensor(0.0, device=device)
            prev_emo_frame_feat = None

            for i in range(2):
                audio = audio_pair[i]
                motion_coef = motion_coef_pair[i]
                e2v_utt = e2v_utt_pair[i]
                e2v_frame = e2v_frame_pair[i]
                batch_size = audio.shape[0]
                audio_in = audio
                motion_coef_in, end_idx = motion_coef, None
                e2v_frame_in = e2v_frame
                indicator = torch.ones(batch_size, args.n_motions, device=device) if args.use_indicator else None

                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat, prev_emo_frame_feat_out = model(
                        motion_coef_in, audio_in, indicator=indicator, emo_index=emo_index,
                        emo_utt_feat=e2v_utt, emo_frame_feat=e2v_frame_in,
                    )
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                    prev_emo_frame_feat = prev_emo_frame_feat_out[:, -args.n_prev_motions:] if prev_emo_frame_feat_out is not None else None
                else:
                    noise, target, _, _, _ = model(
                        motion_coef_in, audio_in, prev_motion_coef, prev_audio_feat,
                        indicator=indicator, emo_index=emo_index, emo_utt_feat=e2v_utt,
                        emo_frame_feat=e2v_frame_in, prev_emo_frame_feat=prev_emo_frame_feat,
                    )

                loss_n, loss_exp, *_ = utils.compute_loss_new(
                    args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx
                )
                pred_motion_current = target[:, args.n_prev_motions:]
                valid_mask = make_valid_mask(batch_size, args.n_motions, end_idx, device)
                loss_noise = loss_noise + loss_n / 2
                loss_expression = loss_expression + loss_exp / 2
                if classifier is not None:
                    pred_emo, pred_level = classifier(pred_motion_current[..., :63].clone())
                    loss_emo = loss_emo + cross_criterion(pred_emo, emo_index) / 2
                    if args.l_emo_level > 0:
                        loss_level = loss_level + cross_criterion(pred_level, emo_level) / 2
                if args.l_prosody_curve > 0:
                    loss_prosody = loss_prosody + compute_prosody_curve_loss(
                        pred_motion_current, e2v_frame_in, mask=valid_mask, rot_repr=args.rot_repr,
                        use_velocity=not args.no_prosody_velocity,
                        vel_weight_motion=args.prosody_motion_vel_weight,
                        vel_weight_audio=args.prosody_audio_vel_weight,
                    ) / 2

            loss = loss_noise + args.l_exp * loss_expression + args.l_emo_cls * loss_emo
            loss = loss + args.l_emo_level * loss_level + args.l_prosody_curve * loss_prosody
            loss_log['loss'].append(loss.item())
            loss_log['noise'].append(loss_noise.item())
            loss_log['emo'].append((args.l_emo_cls * loss_emo).item())
            loss_log['level'].append((args.l_emo_level * loss_level).item())
            loss_log['prosody'].append((args.l_prosody_curve * loss_prosody).item())
            loss_log['exp'].append((args.l_exp * loss_expression).item())

    description = f'(Iter {current_iter:>6}) {mode} loss: [N: {np.mean(loss_log["noise"]):.3e}'
    description += f", EX: {np.mean(loss_log['exp']):.3e}, Emo: {np.mean(loss_log['emo']):.3e}"
    description += f", Level: {np.mean(loss_log['level']):.3e}, Prosody: {np.mean(loss_log['prosody']):.3e}]"
    print(description)
    if writer is not None:
        writer.add_scalar(f'{mode}/total_loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/simple_loss', np.mean(loss_log['noise']), current_iter)
        writer.add_scalar(f'{mode}/emotion_loss', np.mean(loss_log['emo']), current_iter)
        writer.add_scalar(f'{mode}/level_loss', np.mean(loss_log['level']), current_iter)
        writer.add_scalar(f'{mode}/prosody_curve_loss', np.mean(loss_log['prosody']), current_iter)
        writer.add_scalar(f'{mode}/exp_loss', np.mean(loss_log['exp']), current_iter)
    if is_training:
        model.train()


def main(args, option_text=None):
    torch.cuda.set_device(args.device_id)
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')

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
        audio_scale=args.audio_scale,
        label_scale=args.label_scale,
        utt_scale=args.utt_scale,
        frame_scale=args.frame_scale,
    )
    model = DitTalkingHead(**model_kwargs)

    exp_dir = Path('experiments/emo_dit_e2v') / f'{args.exp_name}'
    train_dataset = EmoLevelE2VDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split='train',
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
        split='val',
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
        emotion2vec_root=args.emotion2vec_root,
        emotion2vec_dim=args.e2v_dim,
    )
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(args.emotion_classifier_ckpt, map_location=device), strict=False)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / 'options.log', 'w') as f:
            f.write(option_text)
        writer.add_text('options', option_text)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'exp_name: {exp_dir.name}')
    logging.info(f'model parameters: {utils.count_parameters(model)}')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    if args.scheduler == 'Warmup':
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == 'WarmupThenDecay':
        from src.scheduler import GradualWarmupScheduler
        after_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.cos_max_iter - args.warm_iter, args.lr * args.min_lr_ratio
        )
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after_scheduler)
    else:
        scheduler = None

    train(args, model, train_loader, val_loader, optimizer, exp_dir / 'checkpoints', scheduler, writer, classifier=classifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='e2v_decoupled_emotion_dit')
    parser.add_argument('--device_id', type=int, default=0)

    parser.add_argument('--data_root', type=Path, default='src/my_prepare/')
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')
    parser.add_argument('--emotion2vec_root', type=str, default=None)
    parser.add_argument('--emotion_classifier_ckpt', type=str, default='pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_strategy', type=str, default='random')
    parser.add_argument('--normalize_type', type=str, default='mix', choices=['std', 'case', 'scale', 'minmax', 'mix'])

    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,emotion')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=50)
    parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--no_head_pose', action='store_true', default=False)
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])

    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=3)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--e2v_dim', type=int, default=1024)
    parser.add_argument('--num_label_tokens', type=int, default=8)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--mlp_ratio', type=int, default=4)
    parser.add_argument('--audio_scale', type=float, default=1.0)
    parser.add_argument('--label_scale', type=float, default=0.6)
    parser.add_argument('--utt_scale', type=float, default=0.4)
    parser.add_argument('--frame_scale', type=float, default=0.4)

    parser.add_argument('--n_motions', type=int, default=100)
    parser.add_argument('--n_prev_motions', type=int, default=25)
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--scheduler', type=str, default='WarmupThenDecay', choices=['None', 'Warmup', 'WarmupThenDecay'])
    parser.add_argument('--warm_iter', type=int, default=2000)
    parser.add_argument('--cos_max_iter', type=int, default=100000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)

    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--l_emo_cls', type=float, default=1.0)
    parser.add_argument('--l_emo_level', type=float, default=0.2)
    parser.add_argument('--l_prosody_curve', type=float, default=0.02)
    parser.add_argument('--no_prosody_velocity', action='store_true', default=False)
    parser.add_argument('--prosody_motion_vel_weight', type=float, default=0.5)
    parser.add_argument('--prosody_audio_vel_weight', type=float, default=0.5)
    parser.add_argument('--l_exp', type=float, default=0.1)
    parser.add_argument('--l_exp_vel', type=float, default=1e-4)
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4)
    parser.add_argument('--l_head_angle', type=float, default=1e-2)
    parser.add_argument('--l_head_vel', type=float, default=1e-2)
    parser.add_argument('--l_head_smooth', type=float, default=1e-2)
    parser.add_argument('--l_head_trans', type=float, default=1e-2)
    parser.add_argument('--no_constrain_prev', action='store_true')

    parser.add_argument('--use_context_audio_feat', action='store_true')
    parser.add_argument('--use_indicator', action='store_true', default=True)
    parser.add_argument('--trunc_prob1', type=float, default=0.3)
    parser.add_argument('--trunc_prob2', type=float, default=0.4)
    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--val_iter', type=int, default=50)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--log_smooth_win', type=int, default=50)

    args = parser.parse_args()
    option_text = utils.get_option_text(args, parser)
    main(args, option_text)

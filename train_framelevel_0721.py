import argparse
from collections import defaultdict, deque
from pathlib import Path
import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel_e2v import EmoLevelE2VDataset
from src.modules.emotion_dit_Unification_framelevel_0721 import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils


g_exp_name = '20260721_emotion_dit_Unification_framelevel'
device_id = 1
if torch.cuda.is_available():
    torch.cuda.set_device(device_id)
device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
cross_criterion = torch.nn.CrossEntropyLoss()


def _truncate_frame_feature(frame_feat, end_idx):
    """Apply the same valid-frame mask used by truncated motion/audio inputs."""
    if end_idx is None:
        return frame_feat
    valid = torch.arange(frame_feat.shape[1], device=frame_feat.device)[None, :] < end_idx[:, None]
    return frame_feat * valid.unsqueeze(-1).to(frame_feat.dtype)


def _compute_batch(args, model, classifier, batch, train_mode=True):
    audio_pair, coef_pair, emo_index, _, _, e2v_frame_pair = batch
    audio_pair = [x.to(device) for x in audio_pair]
    coef_pair = [
        {k: coef_pair[i][k].to(device) for k in coef_pair[i]}
        for i in range(2)
    ]
    e2v_frame_pair = [x.to(device) for x in e2v_frame_pair]
    emo_index = emo_index.to(device)

    predict_head_pose = not args.no_head_pose
    motion_coef_pair = [
        utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose)
        for i in range(2)
    ]
    audio_unit = 16000.0 / args.fps

    if args.use_context_audio_feat:
        audio_feat = model.extract_audio_feature(
            torch.cat(audio_pair, dim=1), args.n_motions * 2
        )

    losses = {
        'noise': torch.tensor(0.0, device=device),
        'emo': torch.tensor(0.0, device=device),
        'exp': torch.tensor(0.0, device=device),
        'exp_vel': torch.tensor(0.0, device=device),
        'exp_smooth': torch.tensor(0.0, device=device),
        'head_angle': torch.tensor(0.0, device=device),
        'head_vel': torch.tensor(0.0, device=device),
        'head_smooth': torch.tensor(0.0, device=device),
        'head_trans': torch.tensor(0.0, device=device),
    }

    prev_motion_coef = None
    prev_audio_feat = None
    prev_frame_level_feat = None

    for i in range(2):
        audio = audio_pair[i]
        motion_coef = motion_coef_pair[i]
        frame_level_feat = e2v_frame_pair[i]
        batch_size = audio.shape[0]

        do_truncate = (
            (i == 0 and np.random.rand() < args.trunc_prob1)
            or (i != 0 and np.random.rand() < args.trunc_prob2)
        ) if train_mode else False

        if do_truncate:
            audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                audio, motion_coef, args.n_motions, audio_unit, args.pad_mode
            )
            frame_level_in = _truncate_frame_feature(frame_level_feat, end_idx)
            if args.use_context_audio_feat and i != 0:
                audio_in = model.extract_audio_feature(
                    torch.cat([audio_pair[i - 1], audio_in], dim=1),
                    args.n_motions * 2,
                )[:, -args.n_motions:]
        else:
            end_idx = None
            motion_coef_in = motion_coef
            frame_level_in = frame_level_feat
            if args.use_context_audio_feat:
                audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
            else:
                audio_in = audio

        if args.use_indicator:
            if end_idx is None:
                indicator = torch.ones(
                    batch_size, args.n_motions, device=device
                )
            else:
                indicator = (
                    torch.arange(args.n_motions, device=device)[None, :]
                    < end_idx[:, None]
                )
        else:
            indicator = None

        if i == 0:
            noise, target, returned_motion, returned_audio = model(
                motion_coef_in,
                audio_in,
                indicator=indicator,
                emo_index=emo_index,
                frame_level_feat=frame_level_in,
                prev_frame_level_feat=None,
            )

            # Keep the original cross-window previous-motion/audio behavior.
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
                prev_motion_coef = returned_motion[:, -args.n_prev_motions:]
                prev_audio_feat = returned_audio[:, -args.n_prev_motions:]

            # emotion2vec features are fixed conditions, so use the complete
            # first window as the previous condition for the second window.
            prev_frame_level_feat = frame_level_feat[:, -args.n_prev_motions:].detach()
        else:
            noise, target, _, _ = model(
                motion_coef_in,
                audio_in,
                prev_motion_feat=prev_motion_coef,
                prev_audio_feat=prev_audio_feat,
                indicator=indicator,
                emo_index=emo_index,
                frame_level_feat=frame_level_in,
                prev_frame_level_feat=prev_frame_level_feat,
            )

        (loss_n, loss_exp, loss_exp_v, loss_exp_s,
         loss_ha, loss_hv, loss_hs, loss_ht) = utils.compute_loss_new(
            args,
            i == 0,
            motion_coef_in,
            noise,
            target,
            prev_motion_coef,
            end_idx,
        )

        exps = target[:, args.n_prev_motions:, :63].clone()
        pred_emo, _ = classifier(exps)
        loss_emo = cross_criterion(pred_emo, emo_index)

        losses['noise'] += loss_n / 2
        losses['emo'] += loss_emo / 2
        losses['exp'] += loss_exp / 2
        losses['exp_vel'] += loss_exp_v / 2
        losses['exp_smooth'] += loss_exp_s / 2
        if loss_ha is not None:
            losses['head_angle'] += loss_ha / 2
        if loss_hv is not None:
            losses['head_vel'] += loss_hv / 2
        if loss_hs is not None:
            losses['head_smooth'] += loss_hs / 2
        if loss_ht is not None:
            losses['head_trans'] += loss_ht

    total = losses['noise'] + losses['emo']
    total = total + args.l_exp * losses['exp']
    total = total + args.l_exp_vel * losses['exp_vel']
    total = total + args.l_exp_smooth * losses['exp_smooth']
    if args.target == 'sample' and predict_head_pose:
        total = total + args.l_head_angle * losses['head_angle']
        total = total + args.l_head_vel * losses['head_vel']
        total = total + args.l_head_smooth * losses['head_smooth']
        total = total + args.l_head_trans * losses['head_trans']
    losses['loss'] = total
    return losses


def train(args, model, train_loader, val_loader, optimizer, save_dir,
          scheduler=None, writer=None, classifier=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    loader = infinite_data_loader(train_loader)
    log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad()

    for it in range(args.max_iter + 1):
        losses = _compute_batch(args, model, classifier, next(loader), train_mode=True)
        losses['loss'].backward()

        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if (it + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        for key, value in losses.items():
            log[key].append(value.detach().item())

        if scheduler is not None:
            if args.scheduler != 'WarmupThenDecay' or it < args.cos_max_iter:
                scheduler.step()

        if it % args.log_iter == 0:
            text = (
                f'Iter: {it}  loss={np.mean(log["loss"]):.4e} '
                f'noise={np.mean(log["noise"]):.4e} '
                f'emo={np.mean(log["emo"]):.4e} '
                f'exp={np.mean(log["exp"]):.4e}'
            )
            logging.info(text)
            if writer is not None:
                for key in log:
                    writer.add_scalar(f'train/{key}', np.mean(log[key]), it)
                writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)

        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save(
                {'args': args, 'model': model.state_dict(), 'iter': it},
                save_dir / f'iter_{it:07}.pt',
            )

        if it % args.val_iter == 0 or it == args.max_iter:
            validate(args, model, val_loader, it, writer, classifier)


@torch.no_grad()
def validate(args, model, val_loader, current_iter, writer, classifier):
    was_training = model.training
    model.eval()
    log = defaultdict(list)
    for batch in val_loader:
        losses = _compute_batch(args, model, classifier, batch, train_mode=False)
        for key, value in losses.items():
            log[key].append(value.item())
    text = f'Val {current_iter}: ' + ', '.join(
        f'{key}={np.mean(values):.4e}' for key, values in log.items()
    )
    logging.info(text)
    if writer is not None:
        for key, values in log.items():
            writer.add_scalar(f'val/{key}', np.mean(values), current_iter)
    if was_training:
        model.train()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
        emotion2vec_dim=args.emotion2vec_dim,
    )

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
        emotion2vec_dim=args.emotion2vec_dim,
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
        emotion2vec_dim=args.emotion2vec_dim,
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
    classifier.load_state_dict(torch.load(
        'pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth',
        map_location=device,
    ), strict=False)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    exp_dir = Path('experiments/emo_dit') / args.exp_name
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        (log_dir / 'options.log').write_text(option_text, encoding='utf-8')
        writer.add_text('options', option_text)

    logging.basicConfig(
        filename=str(log_dir / 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f'exp_name: {args.exp_name}')
    logging.info(f'model parameters: {count_parameters(model)}')

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    if args.scheduler == 'Warmup':
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == 'WarmupThenDecay':
        from src.scheduler import GradualWarmupScheduler
        after = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.cos_max_iter - args.warm_iter,
            args.lr * args.min_lr_ratio,
        )
        scheduler = GradualWarmupScheduler(
            optimizer, 1, args.warm_iter, after
        )
    else:
        scheduler = None

    train(
        args, model, train_loader, val_loader, optimizer,
        exp_dir / 'checkpoints', scheduler, writer, classifier
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default=g_exp_name)
    parser.add_argument('--data_root', type=Path, default='src/my_prepare/')
    parser.add_argument('--motion_filename', default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', default='motion_template.pkl')
    parser.add_argument('--emotion2vec_root', type=str, default=None)
    parser.add_argument('--emotion2vec_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_strategy', default='random')
    parser.add_argument('--normalize_type', default='mix')

    parser.add_argument('--target', default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', default='audio,emotion')
    parser.add_argument('--cfg_mode', default='incremental')
    parser.add_argument('--n_diff_steps', type=int, default=50)
    parser.add_argument('--diff_schedule', default='cosine')
    parser.add_argument('--no_head_pose', action='store_true', default=False)
    parser.add_argument('--rot_repr', default='aa')
    parser.add_argument('--audio_model', default='wav2vec2')
    parser.add_argument('--architecture', default='decoder')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--n_motions', type=int, default=100)
    parser.add_argument('--n_prev_motions', type=int, default=25)
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--pad_mode', default='zero')
    parser.add_argument('--use_indicator', action='store_true', default=True)
    parser.add_argument('--use_context_audio_feat', action='store_true')

    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--scheduler', default='WarmupThenDecay')
    parser.add_argument('--clip_grad', action='store_true', default=True)
    parser.add_argument('--trunc_prob1', type=float, default=0.3)
    parser.add_argument('--trunc_prob2', type=float, default=0.4)

    parser.add_argument('--l_exp', type=float, default=0.1)
    parser.add_argument('--l_exp_vel', type=float, default=1e-4)
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4)
    parser.add_argument('--l_head_angle', type=float, default=1e-2)
    parser.add_argument('--l_head_vel', type=float, default=1e-2)
    parser.add_argument('--l_head_smooth', type=float, default=1e-2)
    parser.add_argument('--l_head_trans', type=float, default=1e-2)
    parser.add_argument('--no_constrain_prev', action='store_true')
    parser.add_argument('--criterion', default='l2')

    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--val_iter', type=int, default=50)
    parser.add_argument('--log_iter', type=int, default=50)
    parser.add_argument('--log_smooth_win', type=int, default=50)
    parser.add_argument('--warm_iter', type=int, default=10000)
    parser.add_argument('--cos_max_iter', type=int, default=100000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    option_text = utils.common.get_option_text(args, parser)
    main(args, option_text)

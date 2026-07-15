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
from src.dataset.dataset_EmotionLevel_e2v import EmoLevelE2VDataset
from src.modules.emotion_dit_frame_0715 import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils

G_EXP_NAME = '20260715_emotion_dit_frame_0715'
DEVICE_ID = 1
if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE_ID)
DEVICE = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
EMO_CRITERION = torch.nn.CrossEntropyLoss()


def truncate_frame_feature(frame_feat, end_idx, pad_mode):
    """Use exactly the same frame boundary as audio/motion truncation."""
    output = frame_feat.clone()
    for batch_idx, end in enumerate(end_idx.tolist()):
        if pad_mode == 'zero':
            output[batch_idx, end:] = 0
        elif pad_mode == 'replicate':
            output[batch_idx, end:] = output[batch_idx, end - 1]
        else:
            raise ValueError(f'Unknown pad mode: {pad_mode}')
    return output


def prepare_batch(args, batch, device):
    audio_pair, coef_pair, emo_index, _, _, frame_pair = batch
    audio_pair = [x.to(device) for x in audio_pair]
    coef_pair = [{k: value.to(device) for k, value in item.items()} for item in coef_pair]
    frame_pair = [x.to(device) for x in frame_pair]
    motion_pair = [
        utils.get_motion_coef(item, args.rot_repr, not args.no_head_pose)
        for item in coef_pair
    ]
    return audio_pair, motion_pair, emo_index.to(device), frame_pair


def run_batch(args, model, batch, classifier, training):
    device = model.device
    audio_pair, motion_pair, emo_index, frame_pair = prepare_batch(args, batch, device)
    audio_unit = 16000.0 / args.fps
    context_audio = None
    if args.use_context_audio_feat:
        context_audio = model.extract_audio_feature(
            torch.cat(audio_pair, dim=1), args.n_motions * 2
        )

    totals = defaultdict(lambda: torch.tensor(0.0, device=device))
    prev_motion = prev_audio = prev_frame = None
    for clip_idx in range(2):
        audio = audio_pair[clip_idx]
        motion = motion_pair[clip_idx]
        frame = frame_pair[clip_idx]
        batch_size = audio.shape[0]
        trunc_prob = args.trunc_prob1 if clip_idx == 0 else args.trunc_prob2
        do_truncate = np.random.rand() < trunc_prob

        if do_truncate:
            audio_in, motion_in, end_idx = utils.truncate_motion_coef_and_audio(
                audio, motion, args.n_motions, audio_unit, args.pad_mode
            )
            frame_in = truncate_frame_feature(frame, end_idx, args.pad_mode)
            if args.use_context_audio_feat and clip_idx != 0:
                audio_in = model.extract_audio_feature(
                    torch.cat([audio_pair[clip_idx - 1], audio_in], dim=1),
                    args.n_motions * 2,
                )[:, -args.n_motions:]
        else:
            end_idx = None
            motion_in = motion
            frame_in = frame
            audio_in = (
                context_audio[:, clip_idx * args.n_motions:(clip_idx + 1) * args.n_motions]
                if args.use_context_audio_feat else audio
            )

        if args.use_indicator:
            indicator = torch.ones(batch_size, args.n_motions, device=device)
            if end_idx is not None:
                indicator = (
                    torch.arange(args.n_motions, device=device).expand(batch_size, -1)
                    < end_idx.unsqueeze(1)
                )
        else:
            indicator = None

        output = model(
            motion_in,
            audio_in,
            prev_motion_feat=prev_motion,
            prev_audio_feat=prev_audio,
            indicator=indicator,
            emo_index=emo_index,
            frame_emotion_feat=frame_in,
            prev_frame_feat=prev_frame,
        )
        noise, target, current_motion, current_audio, current_frame = output

        if clip_idx == 0:
            if end_idx is None:
                prev_motion = current_motion[:, -args.n_prev_motions:]
                prev_audio = current_audio[:, -args.n_prev_motions:]
                prev_frame = current_frame[:, -args.n_prev_motions:]
            else:
                prev_motion = motion[:, -args.n_prev_motions:]
                with torch.no_grad():
                    prev_audio = (
                        context_audio[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        if args.use_context_audio_feat
                        else model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    )
                    prev_frame = model.extract_frame_feature(frame)[:, -args.n_prev_motions:]

        losses = utils.compute_loss_new(
            args, clip_idx == 0, motion_in, noise, target, prev_motion, end_idx
        )
        names = ('noise', 'exp', 'exp_vel', 'exp_smooth',
                 'head_angle', 'head_vel', 'head_smooth', 'head_trans')
        for name, value in zip(names, losses):
            if value is not None:
                divisor = 1.0 if name == 'head_trans' and clip_idx == 1 else 2.0
                totals[name] = totals[name] + value / divisor

        pred_emo, _ = classifier(target[:, args.n_prev_motions:, :63].clone())
        totals['emo'] = totals['emo'] + EMO_CRITERION(pred_emo, emo_index) / 2

    total = totals['noise'] + totals['emo']
    for name, weight in (
        ('exp', args.l_exp), ('exp_vel', args.l_exp_vel),
        ('exp_smooth', args.l_exp_smooth), ('head_angle', args.l_head_angle),
        ('head_vel', args.l_head_vel), ('head_smooth', args.l_head_smooth),
        ('head_trans', args.l_head_trans),
    ):
        total = total + weight * totals[name]
    totals['loss'] = total
    return totals


def train(args, model, train_loader, val_loader, optimizer, save_dir,
          scheduler=None, writer=None, classifier=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    loader = infinite_data_loader(train_loader)
    history = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad()
    for iteration in range(args.max_iter + 1):
        totals = run_batch(args, model, next(loader), classifier, training=True)
        totals['loss'].backward()
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        if iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        for name, value in totals.items():
            history[name].append(value.item())
        logging.info(
            'Iter: %d Train loss: %s', iteration,
            ', '.join(f'{k}={np.mean(v):.3e}' for k, v in history.items())
        )
        if writer is not None and iteration % args.log_iter == 0:
            for name, values in history.items():
                writer.add_scalar(f'train/{name}', np.mean(values), iteration)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], iteration)
        if scheduler is not None and (
            args.scheduler != 'WarmupThenDecay' or iteration < args.cos_max_iter
        ):
            scheduler.step()
        if (iteration % args.save_iter == 0 and iteration != 0) or iteration == args.max_iter:
            torch.save({'args': args, 'model': model.state_dict(), 'iter': iteration},
                       save_dir / f'iter_{iteration:07}.pt')
        if iteration % args.val_iter == 0 or iteration == args.max_iter:
            validate(args, model, val_loader, iteration, writer, classifier)


@torch.no_grad()
def validate(args, model, val_loader, iteration, writer, classifier):
    was_training = model.training
    model.eval()
    history = defaultdict(list)
    for batch in val_loader:
        totals = run_batch(args, model, batch, classifier, training=False)
        for name, value in totals.items():
            history[name].append(value.item())
    description = ', '.join(f'{k}={np.mean(v):.3e}' for k, v in history.items())
    logging.info('(Iter %d) val loss: %s', iteration, description)
    if writer is not None:
        for name, values in history.items():
            writer.add_scalar(f'val/{name}', np.mean(values), iteration)
    if was_training:
        model.train()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default=G_EXP_NAME)
    parser.add_argument('--data_root', type=Path, default='src/my_prepare/')
    parser.add_argument('--motion_filename', default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', default='motion_template.pkl')
    parser.add_argument('--emotion2vec_root', default=None)
    parser.add_argument('--emotion2vec_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--crop_strategy', default='random')
    parser.add_argument('--normalize_type', default='mix')
    parser.add_argument('--target', default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', default='audio,emotion')
    parser.add_argument('--cfg_mode', default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=50)
    parser.add_argument('--diff_schedule', default='cosine')
    parser.add_argument('--no_head_pose', action='store_true')
    parser.add_argument('--rot_repr', default='aa')
    parser.add_argument('--audio_model', default='wav2vec2')
    parser.add_argument('--architecture', default='decoder')
    parser.add_argument('--use_indicator', action='store_true', default=True)
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--n_motions', type=int, default=100)
    parser.add_argument('--n_prev_motions', type=int, default=25)
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--pad_mode', default='zero', choices=['zero', 'replicate'])
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--scheduler', default='WarmupThenDecay')
    parser.add_argument('--criterion', default='l2')
    parser.add_argument('--clip_grad', action='store_true', default=True)
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
    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--val_iter', type=int, default=50)
    parser.add_argument('--log_iter', type=int, default=50)
    parser.add_argument('--log_smooth_win', type=int, default=50)
    parser.add_argument('--warm_iter', type=int, default=10000)
    parser.add_argument('--cos_max_iter', type=int, default=100000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)
    return parser


def main(args):
    model = DitTalkingHead(
        device=DEVICE, target=args.target, architecture=args.architecture,
        motion_feat_dim=args.motion_feat_dim, fps=args.fps,
        n_motions=args.n_motions, n_prev_motions=args.n_prev_motions,
        audio_model=args.audio_model, feature_dim=args.feature_dim,
        n_diff_steps=args.n_diff_steps, diff_schedule=args.diff_schedule,
        cfg_mode=args.cfg_mode, guiding_conditions=args.guiding_conditions,
        emotion2vec_dim=args.emotion2vec_dim,
    )
    dataset_kwargs = dict(
        root_dir=args.data_root, motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        coef_fps=args.fps, n_motions=args.n_motions,
        crop_strategy=args.crop_strategy, normalize_type=args.normalize_type,
        emotion2vec_root=args.emotion2vec_root,
        emotion2vec_dim=args.emotion2vec_dim,
    )
    train_dataset = EmoLevelE2VDataset(split='train', **dataset_kwargs)
    val_dataset = EmoLevelE2VDataset(split='val', **dataset_kwargs)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers,
                                   pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.num_workers)
    classifier = Classifier().to(DEVICE)
    classifier.load_state_dict(torch.load(
        'pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth',
        map_location=DEVICE), strict=False)
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)

    exp_dir = Path('experiments/emo_dit') / args.exp_name
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    logging.basicConfig(filename=os.path.join(log_dir, 'log.txt'),
                        level=logging.INFO, format='%(asctime)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    if args.scheduler == 'Warmup':
        from src.scheduler import GradualWarmupScheduler
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter)
    elif args.scheduler == 'WarmupThenDecay':
        from src.scheduler import GradualWarmupScheduler
        after = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.cos_max_iter - args.warm_iter,
            args.lr * args.min_lr_ratio
        )
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after)
    else:
        scheduler = None
    train(args, model, train_loader, val_loader, optimizer,
          exp_dir / 'checkpoints', scheduler, writer, classifier)


if __name__ == '__main__':
    parser = build_parser()
    parsed_args = parser.parse_args()
    main(parsed_args)

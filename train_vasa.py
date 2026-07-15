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
from src.dataset.dataset_EmotionLevel_vasa import EmoLevelDataset
from src.modules.emotion_dit_vasa import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
from src.utils.vasa_loss import compute_loss_vasa


cross_criterion = torch.nn.CrossEntropyLoss()


def prepare_batch(args, model, batch, device, truncate_current):
    audio, coef_dict, emo_index, _ = batch
    audio = audio.to(device)
    coef_dict = {key: value.to(device) for key, value in coef_dict.items()}
    emo_index = emo_index.to(device)

    motion_all = utils.get_motion_coef(
        coef_dict, args.rot_repr, not args.no_head_pose
    )
    prev_motion = motion_all[:, :args.n_prev_motions]
    current_motion = motion_all[:, args.n_prev_motions:]

    audio_unit = 16000.0 / args.fps
    prev_audio_samples = round(audio_unit * args.n_prev_motions)
    prev_audio = audio[:, :prev_audio_samples]
    current_audio = audio[:, prev_audio_samples:]

    end_idx = None
    if truncate_current and np.random.rand() < args.current_trunc_prob:
        current_audio, current_motion, end_idx = (
            utils.truncate_motion_coef_and_audio(
                current_audio,
                current_motion,
                args.n_motions,
                audio_unit,
                args.pad_mode,
            )
        )

    # Extract one contextualized feature sequence for all 25+100 frames and
    # split it once. This replaces the original two model passes per iteration.
    audio_all = torch.cat([prev_audio, current_audio], dim=1)
    audio_feat_all = model.extract_audio_feature(
        audio_all, args.n_prev_motions + args.n_motions
    )
    prev_audio_feat = audio_feat_all[:, :args.n_prev_motions].detach()
    current_audio_feat = audio_feat_all[:, args.n_prev_motions:]

    if args.use_indicator:
        if end_idx is None:
            indicator = torch.ones(
                audio.shape[0], args.n_motions, dtype=torch.bool, device=device
            )
        else:
            indicator = (
                torch.arange(args.n_motions, device=device)[None, :]
                < end_idx[:, None]
            )
    else:
        indicator = None

    return (
        current_motion,
        current_audio_feat,
        prev_motion,
        prev_audio_feat,
        indicator,
        emo_index,
        end_idx,
    )


def forward_losses(args, model, classifier, prepared):
    (
        current_motion,
        current_audio_feat,
        prev_motion,
        prev_audio_feat,
        indicator,
        emo_index,
        end_idx,
    ) = prepared

    noise, target, _, _ = model(
        current_motion,
        current_audio_feat,
        prev_motion,
        prev_audio_feat,
        indicator=indicator,
        emo_index=emo_index,
    )
    previous_valid = ~model.last_prev_dropout_mask
    losses = compute_loss_vasa(
        args,
        current_motion,
        noise,
        target,
        prev_motion,
        end_idx=end_idx,
        previous_valid=previous_valid,
    )
    (
        loss_noise,
        loss_exp,
        loss_exp_vel,
        loss_exp_smooth,
        loss_head_angle,
        loss_head_vel,
        loss_head_smooth,
        loss_head_trans,
    ) = losses

    loss_emo = torch.tensor(0.0, device=model.device)
    if args.target == 'sample':
        current_target = target[:, args.n_prev_motions:]
        pred_emo, _ = classifier(current_target[..., :63])
        loss_emo = cross_criterion(pred_emo, emo_index)

    total = loss_noise + loss_emo
    weighted = {'noise': loss_noise, 'emo': loss_emo}
    optional = [
        ('exp', loss_exp, args.l_exp),
        ('exp_vel', loss_exp_vel, args.l_exp_vel),
        ('exp_smooth', loss_exp_smooth, args.l_exp_smooth),
        ('head_angle', loss_head_angle, args.l_head_angle),
        ('head_vel', loss_head_vel, args.l_head_vel),
        ('head_smooth', loss_head_smooth, args.l_head_smooth),
        ('head_trans', loss_head_trans, args.l_head_trans),
    ]
    for name, value, weight in optional:
        if value is not None:
            weighted[name] = weight * value
            total = total + weighted[name]
    weighted['loss'] = total
    return total, weighted


def append_log(loss_log, values):
    for key, value in values.items():
        loss_log[key].append(value.detach().item())


def describe(prefix, loss_log):
    labels = [
        ('noise', 'N'), ('exp', 'EX'), ('exp_vel', 'EX_V'),
        ('exp_smooth', 'EX_S'), ('head_angle', 'HA'),
        ('head_vel', 'HV'), ('head_smooth', 'HS'),
        ('head_trans', 'HT'), ('emo', 'Emo'),
    ]
    parts = [
        f'{label}: {np.mean(loss_log[key]):.3e}'
        for key, label in labels if loss_log.get(key)
    ]
    return f'{prefix} loss: [' + ', '.join(parts) + ']'


def write_log(writer, mode, loss_log, step):
    if writer is None:
        return
    for key, values in loss_log.items():
        if values:
            writer.add_scalar(f'{mode}/{key}', np.mean(values), step)


def train(args, model, train_loader, val_loader, optimizer, save_dir,
          scheduler=None, writer=None, start_iter=0, classifier=None):
    save_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    classifier.eval()
    loader = infinite_data_loader(train_loader)
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))
    optimizer.zero_grad(set_to_none=True)

    for it in range(start_iter, args.max_iter + 1):
        prepared = prepare_batch(
            args, model, next(loader), model.device, truncate_current=True
        )
        loss, values = forward_losses(args, model, classifier, prepared)
        (loss / args.gradient_accumulation_steps).backward()
        append_log(loss_log, values)

        micro_step = it - start_iter + 1
        should_step = (
            micro_step % args.gradient_accumulation_steps == 0
            or it == args.max_iter
        )
        if should_step:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                if args.scheduler != 'WarmupThenDecay' or it < args.cos_max_iter:
                    scheduler.step()

        logging.info(describe(f'Iter {it}', loss_log))
        if it % args.log_iter == 0:
            write_log(writer, 'train', loss_log, it)
            if writer is not None:
                writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)

        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter:
            torch.save(
                {'args': args, 'model': model.state_dict(), 'iter': it},
                save_dir / f'iter_{it:07}.pt',
            )
        if it % args.val_iter == 0 or it == args.max_iter:
            validate(args, model, val_loader, it, writer, classifier)


@torch.no_grad()
def validate(args, model, loader, current_iter, writer, classifier):
    was_training = model.training
    model.eval()
    classifier.eval()
    loss_log = defaultdict(list)
    for batch in loader:
        prepared = prepare_batch(
            args, model, batch, model.device, truncate_current=False
        )
        _, values = forward_losses(args, model, classifier, prepared)
        append_log(loss_log, values)
    logging.info(describe(f'Iter {current_iter} val', loss_log))
    write_log(writer, 'val', loss_log, current_iter)
    if was_training:
        model.train()


def build_dataset(args, split):
    return EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split=split,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        n_prev_motions=args.n_prev_motions,
        crop_strategy=args.crop_strategy,
        normalize_type=args.normalize_type,
    )


def main(args, option_text=None):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device_id)
        device = torch.device(f'cuda:{args.device_id}')
    else:
        device = torch.device('cpu')

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
        prev_dropout_prob=args.prev_dropout_prob,
    )

    train_dataset = build_dataset(args, 'train')
    val_dataset = build_dataset(args, 'val')
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    classifier = Classifier().to(device)
    classifier.load_state_dict(
        torch.load(args.classifier_path, map_location=device), strict=False
    )
    classifier.eval()
    # for parameter in classifier.parameters():
    #     parameter.requires_grad_(False)

    exp_dir = Path('experiments/emo_dit') / args.exp_name
    log_dir = exp_dir / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        (log_dir / 'options.log').write_text(option_text, encoding='utf-8')
        writer.add_text('options', option_text)

    logging.basicConfig(
        filename=os.path.join(str(log_dir), 'log.txt'),
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
    )
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
            optimizer,
            args.cos_max_iter - args.warm_iter,
            args.lr * args.min_lr_ratio,
        )
        scheduler = GradualWarmupScheduler(optimizer, 1, args.warm_iter, after)
    else:
        scheduler = None

    train(
        args, model, train_loader, val_loader, optimizer,
        exp_dir / 'checkpoints', scheduler, writer, classifier=classifier,
    )


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='20260714_vasa_prev25_current100')
    parser.add_argument('--device_id', type=int, default=1)
    parser.add_argument('--data_root', type=Path, default='src/my_prepare/')
    parser.add_argument('--motion_filename', default='front_all_motions.pkl')
    parser.add_argument('--motion_template_filename', default='motion_template.pkl')
    parser.add_argument('--classifier_path', default='pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth')
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
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--n_motions', type=int, default=100)
    parser.add_argument('--n_prev_motions', type=int, default=25)
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--pad_mode', default='zero', choices=['zero', 'replicate'])
    parser.add_argument('--use_indicator', action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--scheduler', default='WarmupThenDecay', choices=['None', 'Warmup', 'WarmupThenDecay'])
    parser.add_argument('--prev_dropout_prob', type=float, default=0.1)
    parser.add_argument('--current_trunc_prob', type=float, default=0.4)
    parser.add_argument('--criterion', default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--l_exp', type=float, default=0.1)
    parser.add_argument('--l_exp_vel', type=float, default=1e-4)
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4)
    parser.add_argument('--l_head_angle', type=float, default=1e-2)
    parser.add_argument('--l_head_vel', type=float, default=1e-2)
    parser.add_argument('--l_head_smooth', type=float, default=1e-2)
    parser.add_argument('--l_head_trans', type=float, default=1e-2)
    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--val_iter', type=int, default=50)
    parser.add_argument('--log_iter', type=int, default=50)
    parser.add_argument('--log_smooth_win', type=int, default=50)
    parser.add_argument('--warm_iter', type=int, default=10000)
    parser.add_argument('--cos_max_iter', type=int, default=100000)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    option_text = utils.common.get_option_text(args, parser)
    main(args, option_text)

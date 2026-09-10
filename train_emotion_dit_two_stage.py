import argparse
import pickle
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils import data

from src.dataset import infinite_data_loader
from src.dataset.dataset_EmotionLevel import EmoLevelDataset
from src.dataset.dataset_GeneralTalkingMotion import GeneralTalkingMotionDataset
from src.modules.emotion_dit_two_stage import DitTalkingHead
from src.modules.emotion_level_classifier import EmotionTransformer as Classifier
import src.utils as utils

cross_criterion = torch.nn.CrossEntropyLoss()


def build_dataset(args, split):
    crop_strategy = args.crop_strategy if split == 'train' else 'begin'
    if args.stage == 'general':
        return GeneralTalkingMotionDataset(
            root_dir=args.general_data_root,
            motion_filenames=args.general_motion_filenames,
            motion_template_filename=args.motion_template_filename,
            split=split,
            split_filename=args.general_train_split if split == 'train' else args.general_test_split,
            coef_fps=args.fps,
            n_motions=args.n_motions,
            crop_strategy=crop_strategy,
            normalize_type=args.normalize_type,
        )
    return EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split=split,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=crop_strategy,
        normalize_type=args.normalize_type,
    )


def run_batch(args, model, batch, classifier=None, norm_dict=None, training=True):
    device = model.device
    audio_pair, coef_pair, emo_index, _ = batch
    audio_pair = [audio.to(device) for audio in audio_pair]
    coef_pair = [{key: value.to(device) for key, value in item.items()} for item in coef_pair]
    motion_pair = [utils.get_motion_coef(item, args.rot_repr, not args.no_head_pose) for item in coef_pair]
    emo_index = emo_index.to(device)

    audio_unit = 16000. / args.fps
    previous_motion = previous_audio = None
    losses = defaultdict(lambda: torch.tensor(0., device=device))

    for window_index in range(2):
        audio = audio_pair[window_index]
        motion = motion_pair[window_index]
        batch_size = audio.shape[0]
        trunc_prob = args.trunc_prob1 if window_index == 0 else args.trunc_prob2
        if training and np.random.rand() < trunc_prob:
            audio_in, motion_in, end_idx = utils.truncate_motion_coef_and_audio(
                audio, motion, args.n_motions, audio_unit, args.pad_mode)
        else:
            audio_in, motion_in, end_idx = audio, motion, None

        if args.use_indicator:
            if end_idx is None:
                indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(1)
        else:
            indicator = None

        if window_index == 0:
            noise, target, current_motion, current_audio = model(
                motion_in, audio_in, indicator=indicator, emo_index=emo_index)
            if end_idx is not None:
                previous_motion = motion[:, -args.n_prev_motions:].detach()
                with torch.no_grad():
                    previous_audio = model.extract_audio_feature(audio)[:, -args.n_prev_motions:].detach()
            else:
                previous_motion = current_motion[:, -args.n_prev_motions:].detach()
                previous_audio = current_audio[:, -args.n_prev_motions:].detach()
        else:
            noise, target, _, _ = model(
                motion_in, audio_in, previous_motion, previous_audio,
                indicator=indicator, emo_index=emo_index)

        loss_pack = utils.compute_loss_new(
            args, window_index == 0, motion_in, noise, target,
            previous_motion, end_idx)
        loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hv, loss_hs, loss_ht = loss_pack
        losses['noise'] += loss_n / 2
        losses['exp'] += loss_exp / 2
        if loss_exp_v is not None:
            losses['exp_vel'] += loss_exp_v / 2
        if loss_exp_s is not None:
            losses['exp_smooth'] += loss_exp_s / 2
        if loss_ha is not None:
            losses['head_angle'] += loss_ha / 2
        if loss_hv is not None:
            losses['head_vel'] += loss_hv / 2
        if loss_hs is not None:
            losses['head_smooth'] += loss_hs / 2
        if loss_ht is not None:
            losses['head_trans'] += loss_ht

        if args.stage == 'emotion' and classifier is not None:
            expressions = target[:, args.n_prev_motions:, :63].clone()
            emotion_mean = norm_dict['mean_exps'][emo_index].unsqueeze(1)
            emotion_std = norm_dict['std_exps'][emo_index].unsqueeze(1)
            expressions = (expressions * emotion_std + emotion_mean - norm_dict['mean_exp']) / (norm_dict['std_exp'] + 1e-9)
            pred_emo, _ = classifier(expressions)
            losses['emo'] += cross_criterion(pred_emo, emo_index) / 2

    total = losses['noise']
    total = total + args.l_exp * losses['exp']
    total = total + args.l_exp_vel * losses['exp_vel']
    total = total + args.l_exp_smooth * losses['exp_smooth']
    if not args.no_head_pose:
        total = total + args.l_head_angle * losses['head_angle']
        total = total + args.l_head_vel * losses['head_vel']
        total = total + args.l_head_smooth * losses['head_smooth']
        total = total + args.l_head_trans * losses['head_trans']
    if args.stage == 'emotion':
        total = total + args.l_emo * losses['emo']
    return total, losses


@torch.no_grad()
def validate(args, model, loader, classifier, norm_dict, max_batches=20):
    model.eval()
    values = []
    for batch_index, batch in enumerate(loader):
        loss, _ = run_batch(args, model, batch, classifier, norm_dict, training=False)
        values.append(loss.item())
        if max_batches > 0 and batch_index + 1 >= max_batches:
            break
    model.train()
    return float(np.mean(values))


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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
        guiding_conditions='audio,emotion',
        training_stage=args.stage,
    )
    if args.stage == 'emotion':
        if args.general_checkpoint is None:
            raise ValueError('--general_checkpoint is required for Stage 2')
        model.load_general_checkpoint(args.general_checkpoint)
    trainable = model.set_train_stage(args.stage, args.train_motion_decoder)
    print(f'trainable parameters ({len(trainable)} tensors): {trainable}')

    train_dataset = build_dataset(args, 'train')
    val_dataset = build_dataset(args, 'test')
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers)

    classifier = None
    norm_dict = None
    if args.stage == 'emotion':
        classifier = Classifier().to(device)
        classifier.load_state_dict(torch.load(args.classifier_checkpoint, map_location=device), strict=False)
        classifier.eval()
        for parameter in classifier.parameters():
            parameter.requires_grad = False
        motion_template = pickle.load(open(args.motion_template_for_loss, 'rb'))
        emotion_templates = pickle.load(open(args.emotion_template_for_loss, 'rb'))
        norm_dict = {
            'mean_exp': torch.tensor(motion_template['mean_exp'], device=device)[None, None],
            'std_exp': torch.tensor(motion_template['std_exp'], device=device)[None, None],
            'mean_exps': torch.stack([torch.tensor(item['mean_exp']) for item in emotion_templates]).to(device),
            'std_exps': torch.stack([torch.tensor(item['std_exp']) for item in emotion_templates]).to(device),
        }

    optimizer = optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter, eta_min=args.lr * args.min_lr_ratio)

    output_dir = Path('experiments/emo_dit') / args.exp_name
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output_dir / 'logs'))
    iterator = infinite_data_loader(train_loader)
    loss_log = deque(maxlen=args.log_smooth_win)
    optimizer.zero_grad()

    for iteration in range(1, args.max_iter + 1):
        loss, losses = run_batch(args, model, next(iterator), classifier, norm_dict, training=True)
        (loss / args.gradient_accumulation_steps).backward()
        if iteration % args.gradient_accumulation_steps == 0:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda parameter: parameter.requires_grad, model.parameters()), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        loss_log.append(loss.item())
        if iteration % args.log_iter == 0:
            mean_loss = float(np.mean(loss_log))
            print(f'[{args.stage}] iter={iteration}, loss={mean_loss:.6f}')
            writer.add_scalar('train/loss', mean_loss, iteration)
        if iteration % args.val_iter == 0:
            val_loss = validate(args, model, val_loader, classifier, norm_dict, args.val_batches)
            print(f'[{args.stage}] iter={iteration}, val_loss={val_loss:.6f}')
            writer.add_scalar('val/loss', val_loss, iteration)
        if iteration % args.save_iter == 0 or iteration == args.max_iter:
            torch.save({
                'stage': args.stage,
                'iter': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }, checkpoint_dir / f'iter_{iteration:07d}.pt')


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['general', 'emotion'], required=True)
    parser.add_argument('--exp_name', type=str, default='two_stage_emotion_dit')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--data_root', type=Path, default=Path('src/my_prepare/'))
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')
    parser.add_argument('--general_data_root', type=str, default='src/my_prepare/')
    parser.add_argument('--general_motion_filenames', type=str, default='front_all_motions.pkl')
    parser.add_argument('--general_train_split', type=str, default=None)
    parser.add_argument('--general_test_split', type=str, default=None)
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')
    parser.add_argument('--crop_strategy', type=str, default='random')
    parser.add_argument('--normalize_type', type=str, default='mix')

    parser.add_argument('--general_checkpoint', type=str, default=None)
    parser.add_argument('--classifier_checkpoint', type=str,
                        default='pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth')
    parser.add_argument('--motion_template_for_loss', type=str,
                        default='pretrained_weights/ADEF/motion_template/motion_template.pkl')
    parser.add_argument('--emotion_template_for_loss', type=str,
                        default='pretrained_weights/ADEF/motion_template/emotion_template.pkl')
    parser.add_argument('--train_motion_decoder', action='store_true')

    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,emotion')
    parser.add_argument('--cfg_mode', type=str, default='incremental')
    parser.add_argument('--n_diff_steps', type=int, default=50)
    parser.add_argument('--diff_schedule', type=str, default='cosine')
    parser.add_argument('--audio_model', type=str, default='wav2vec2')
    parser.add_argument('--architecture', type=str, default='decoder')
    parser.add_argument('--feature_dim', type=int, default=512)
    parser.add_argument('--n_motions', type=int, default=100)
    parser.add_argument('--n_prev_motions', type=int, default=25)
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--use_indicator', action='store_true', default=True)
    parser.add_argument('--no_head_pose', action='store_true')
    parser.add_argument('--rot_repr', type=str, default='aa')
    parser.add_argument('--pad_mode', type=str, default='zero')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_iter', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--clip_grad', action='store_true', default=True)
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)
    parser.add_argument('--trunc_prob1', type=float, default=0.3)
    parser.add_argument('--trunc_prob2', type=float, default=0.4)

    parser.add_argument('--criterion', type=str, default='l2')
    parser.add_argument('--l_emo', type=float, default=1.0)
    parser.add_argument('--l_exp', type=float, default=0.1)
    parser.add_argument('--l_exp_vel', type=float, default=1e-4)
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4)
    parser.add_argument('--l_head_angle', type=float, default=1e-2)
    parser.add_argument('--l_head_vel', type=float, default=1e-2)
    parser.add_argument('--l_head_smooth', type=float, default=1e-2)
    parser.add_argument('--l_head_trans', type=float, default=1e-2)
    parser.add_argument('--no_constrain_prev', action='store_true')

    parser.add_argument('--save_iter', type=int, default=1000)
    parser.add_argument('--val_iter', type=int, default=100)
    parser.add_argument('--val_batches', type=int, default=20)
    parser.add_argument('--log_iter', type=int, default=50)
    parser.add_argument('--log_smooth_win', type=int, default=50)
    return parser


if __name__ == '__main__':
    main(build_parser().parse_args())

# 全力冲刺Unification
# n_diff_steps 50 to 500
# align_mask_width 1 to 2
# iter至少是200000，尽管增大bs
# warm 采用 10%是对的
# 可以：l_exp  0.1 to 1.0

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
from src.dataset.dataset_EmotionLevel_clear_jianhua0803 import EmoLevelDataset


# 这个效果不错，复刻一下
# batch_size:  128
# iter:  300000
# warm:  20000   decay:200000
# l_exp  0.1 to 1.0
# mask： 2
from src.modules.emotion_dit_Unification_jianhua0803_deep_adam import DitTalkingHead
g_exp_name = "20260823_matrix_A1B0_deep512_L12_adam"
device_id = 1

# 512 维、12 层、8 头 原 Adam 配方

torch.cuda.set_device(device_id)  # 设置默认 GPU
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")  # 显式指定设备

cross_criterion = torch.nn.CrossEntropyLoss()

def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None):

    save_dir.mkdir(parents=True, exist_ok=True) 

    # model
    device = model.device
    model.train()

    data_loader = infinite_data_loader(train_loader)   # 将数据加载器（train_loader）转换为一个无限循环的迭代器
    audio_unit = train_loader.dataset.audio_unit       # 每一帧的样本数  self.audio_unit = 16000. / self.coef_fps (float)
    n_audio_samples = round(audio_unit * args.n_motions)         # current 段对应的音频采样数（int）
    n_prev_audio_samples = round(audio_unit * args.n_prev_motions)  # prev 段对应的音频采样数（int）
    predict_head_pose = not args.no_head_pose          # False -> True  预测头部姿势
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))  # maxlen = 50

    optimizer.zero_grad()
    for it in range(start_iter, args.max_iter + 1):   # 迭代次数  0 ~ max_iter
        # 数据集一次采样 (n_prev_motions + n_motions) = 16 + 64 = 80 帧：
        #   前 n_prev_motions=16 帧作为 prev，后 n_motions=64 帧作为 current。
        audio, coef_dict, emo_index, _ = next(data_loader)
        audio = audio.to(device)                                                 # (N, (n_prev + n_motions) * audio_unit)
        coef_dict = {k: coef_dict[k].to(device) for k in coef_dict}              # {exp:(N,80,63), pose:(N,80,7)}
        motion_coef_full = utils.get_motion_coef(
            coef_dict, args.rot_repr, predict_head_pose
        )                                                                        # (N, 80, 70) — 前 16 是 prev，后 64 是 current
        emo_index = emo_index.to(device)
        batch_size = audio.shape[0]

        # 单数轮次 → 只用 current 64 帧训练"无先前参考"形式（模型用 start_*_feat 内部初始化 prev）
        # 双数轮次 → 用 80 帧（16 prev + 64 current）训练"有先前参考"形式（传入 GT prev）
        is_starting_sample = (it % 2 == 1)

        # ---------- 截断逻辑（仅作用于 current 部分；prev 不截断） ----------
        current_audio  = audio[:, -n_audio_samples:].contiguous()                 # (N, n_audio_samples)
        current_motion = motion_coef_full[:, -args.n_motions:]                    # (N, n_motions, 70)
        end_idx = None
        trunc_prob = args.trunc_prob1 if is_starting_sample else args.trunc_prob2
        if np.random.rand() < trunc_prob:
            current_audio, current_motion, end_idx = utils.truncate_motion_coef_and_audio(
                current_audio, current_motion, args.n_motions, audio_unit, args.pad_mode)

        # ---------- 指示器（指示 current 中被填充的帧） ----------
        if args.use_indicator:
            if end_idx is not None:                                                # 被截断
                indicator = torch.arange(args.n_motions, device=device).expand(
                    batch_size, -1) < end_idx.unsqueeze(1)                          # 超过 end_idx 的位置是 False
            else:                                                                  # 没被截断
                indicator = torch.ones(batch_size, args.n_motions, device=device)
        else:
            indicator = None

        # ---------- 模型前向：单数轮次不传 prev，双数轮次传 GT prev ----------
        if is_starting_sample:
            # 单数：prev_motion_feat=None、prev_audio_feat=None → 模型内部用 start_motion_feat / start_audio_feat
            noise, target, _, _ = model(
                current_motion, current_audio,
                indicator=indicator, emo_index=emo_index,
            )
            prev_motion_for_loss = None
        else:
            # 双数：前 16 帧的真实 GT prev。
            prev_motion_gt = motion_coef_full[:, :args.n_prev_motions]              # (N, n_prev_motions, 70)
            prev_audio_raw = audio[:, :n_prev_audio_samples].contiguous()             # (N, n_prev_motions * audio_unit)
            with torch.no_grad():
                prev_audio_feat = model.extract_audio_feature(
                    prev_audio_raw, frame_num=args.n_prev_motions
                )                                                                   # (N, n_prev_motions, 512)
            noise, target, _, _ = model(
                current_motion, current_audio,
                prev_motion_gt, prev_audio_feat,
                indicator=indicator, emo_index=emo_index,
            )
            prev_motion_for_loss = prev_motion_gt

        # ---------- 损失 ----------
        loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(
            args, is_starting_sample, current_motion, noise, target, prev_motion_for_loss, end_idx,
        )

        # 情感分类损失：取 target 的后 64 帧（即去噪后的 current 段）的 exp 系数
        exps = target[:, args.n_prev_motions:, :63].clone()                        # (N, n_motions, 63)
        pred_emo, _ = classifier(exps)                                              # (N, 8)
        loss_emo = cross_criterion(pred_emo, emo_index)                             # 不再除以 2（每 iter 仅一段）

        # 累加各项损失（每轮只用一 clip，所以无需 /2）
        loss_noise     = loss_n
        loss_expression = loss_exp
        loss_exp_vel    = loss_exp_v
        loss_exp_smooth = loss_exp_s
        loss_head_angle = loss_ha if (args.target == 'sample' and predict_head_pose and args.l_head_angle > 0 and loss_ha is not None) else torch.tensor(0, device=device)
        loss_head_vel   = loss_hc if (args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None) else torch.tensor(0, device=device)
        loss_head_smooth = loss_hs if (args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None) else torch.tensor(0, device=device)
        loss_head_trans = loss_ht if (args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None) else torch.tensor(0, device=device)

        # 扩散（采样）损失
        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise

        # 情感损失
        loss_log['emo'].append(loss_emo.item())
        loss = loss + loss_emo

        # 表情相关损失(计算级联损失时，需要乘以相应的权重)
        loss_log['exp'].append(loss_expression.item() * args.l_exp)             # l_exp： 0.1  权重
        loss = loss + args.l_exp * loss_expression

        loss_log['exp_vel'].append(loss_exp_vel.item() * args.l_exp_vel)   # l_exp_vel： 1e-4  权重
        loss = loss + args.l_exp_vel * loss_exp_vel

        loss_log['exp_smooth'].append(loss_exp_smooth.item() * args.l_exp_smooth)  # l_exp_smooth： 1e-4  权重
        loss = loss + args.l_exp_smooth * loss_exp_smooth

        # 头部姿势相关损失(计算级联损失时，需要乘以相应的权重)
        if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:  # 采样；预测头部姿势；权重大于0
            loss_log['head_angle'].append(loss_head_angle.item() * args.l_head_angle)   # 1e-2
            loss = loss + args.l_head_angle * loss_head_angle
        if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:       # 1e-2
            loss_log['head_vel'].append(loss_head_vel.item() * args.l_head_vel)
            loss = loss + args.l_head_vel * loss_head_vel
        if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:    # 1e-2
            loss_log['head_smooth'].append(loss_head_smooth.item() * args.l_head_smooth)
            loss = loss + args.l_head_smooth * loss_head_smooth
        if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:     # 1e-2
            loss_log['head_trans'].append(loss_head_trans.item() * args.l_head_trans)
            loss = loss + args.l_head_trans * loss_head_trans

        (loss / args.gradient_accumulation_steps).backward()
        micro_step = it - start_iter + 1
        should_step = (micro_step % args.gradient_accumulation_steps == 0) or it == args.max_iter
        optimizer_update = (micro_step + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        if should_step:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        # Logging  日志写入
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
        description += f", Emo: {np.mean(loss_log['emo']):.3e}"   # 情感损失-交叉熵
        description += ']'
        logging.info(description)

        # write to tensorboard  写入tensorboard，记录曲线
        if it % args.log_iter == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['loss']), it)     # 总损失
            writer.add_scalar('train/emotion_loss', np.mean(loss_log['emo']), it)   # 情感损失-交叉熵
            writer.add_scalar('train/simple_loss', np.mean(loss_log['noise']), it)   # 扩散采样损失
            writer.add_scalar('train/exp_loss', np.mean(loss_log['exp']), it)        # 表情损失
            writer.add_scalar('train/exp_vel_loss', np.mean(loss_log['exp_vel']), it)
            writer.add_scalar('train/exp_smooth_loss', np.mean(loss_log['exp_smooth']), it)
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                writer.add_scalar('train/head_angle', np.mean(loss_log['head_angle']), it)    #
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0:
                writer.add_scalar('train/head_vel', np.mean(loss_log['head_vel']), it)        #
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0:
                writer.add_scalar('train/head_smooth', np.mean(loss_log['head_smooth']), it)  #
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0:
                writer.add_scalar('train/head_trans', np.mean(loss_log['head_trans']), it)    #  
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], it)          # 学习率曲线

        # update learning rate  更新学习率
        if scheduler is not None and should_step:
            if args.scheduler != 'WarmupThenDecay' or optimizer_update <= args.cos_max_iter:
                scheduler.step()

        # save model   保存模型中间结果
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter: # 每1000次迭代 保存一次。 第50000次 保存一次
            torch.save({
                'args': args,                 # args：模型参数
                'model': model.state_dict(),   # 模型
                'iter': it,                    # 训练轮次
            }, save_dir / f'iter_{it:07}.pt')   

        # # validation  验证模型
        # if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:  # 每50次迭代 验证一次。 第0次和第50000次 验证一次
        #     val(args, model, val_loader, it, 1, 'val', writer, classifier)

# 测试部分
@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode='val', writer=None,  classifier=None):
    # print("test ... ")
    is_training = model.training
    device = model.device
    model.eval()   # 设置为eval模式

    audio_unit = test_loader.dataset.audio_unit
    n_audio_samples = round(audio_unit * args.n_motions)            # current 段对应的音频采样数（int）
    n_prev_audio_samples = round(audio_unit * args.n_prev_motions)   # prev 段对应的音频采样数（int）
    predict_head_pose = not args.no_head_pose

    loss_log = defaultdict(list)
    val_sample_counter = 0     # 局部计数器，使 val 也按"单数轮次无 prev / 双数轮次有 prev"交替
    for test_round in range(n_rounds):     # 1  只测试一次
        # 与训练部分逻辑保持一致：每个样本按 16+64=80 帧采样，单数轮次无 prev、双数轮次有 prev。
        for audio, coef_dict, emo_index, _ in test_loader:
            audio = audio.to(device)
            coef_dict = {k: coef_dict[k].to(device) for k in coef_dict}
            motion_coef_full = utils.get_motion_coef(
                coef_dict, args.rot_repr, predict_head_pose
            )                                                                # (N, 80, 70)
            emo_index = emo_index.to(device)
            batch_size = audio.shape[0]

            is_starting_sample = (val_sample_counter % 2 == 1)
            val_sample_counter += 1

            # current 段（最后 64 帧）
            current_audio  = audio[:, -n_audio_samples:].contiguous()
            current_motion = motion_coef_full[:, -args.n_motions:]
            end_idx = None
            trunc_prob = args.trunc_prob1 if is_starting_sample else args.trunc_prob2
            if np.random.rand() < trunc_prob:
                current_audio, current_motion, end_idx = utils.truncate_motion_coef_and_audio(
                    current_audio, current_motion, args.n_motions, audio_unit, args.pad_mode)

            if args.use_indicator:
                if end_idx is not None:
                    indicator = torch.arange(args.n_motions, device=device).expand(
                        batch_size, -1) < end_idx.unsqueeze(1)
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)
            else:
                indicator = None

            # 模型前向
            if is_starting_sample:
                noise, target, _, _ = model(
                    current_motion, current_audio,
                    indicator=indicator, emo_index=emo_index,
                )
                prev_motion_for_loss = None
            else:
                prev_motion_gt = motion_coef_full[:, :args.n_prev_motions]
                prev_audio_raw = audio[:, :n_prev_audio_samples].contiguous()
                with torch.no_grad():
                    prev_audio_feat = model.extract_audio_feature(
                        prev_audio_raw, frame_num=args.n_prev_motions
                    )
                noise, target, _, _ = model(
                    current_motion, current_audio,
                    prev_motion_gt, prev_audio_feat,
                    indicator=indicator, emo_index=emo_index,
                )
                prev_motion_for_loss = prev_motion_gt

            loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(
                args, is_starting_sample, current_motion, noise, target, prev_motion_for_loss, end_idx,
            )

            # 情感分类（情感损失）
            exps = target[:, args.n_prev_motions:, :63].clone()      # (N, n_motions, 63)
            pred_emo, _ = classifier(exps)                            # (N, 8)
            loss_emo = cross_criterion(pred_emo, emo_index)

            # simple loss
            loss_noise     = loss_n
            loss_expression = loss_exp
            loss_exp_vel    = loss_exp_v
            loss_exp_smooth = loss_exp_s
            loss_head_angle = loss_ha if (args.target == 'sample' and predict_head_pose and args.l_head_angle > 0 and loss_ha is not None) else torch.tensor(0, device=device)
            loss_head_vel   = loss_hc if (args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None) else torch.tensor(0, device=device)
            loss_head_smooth = loss_hs if (args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None) else torch.tensor(0, device=device)
            loss_head_trans = loss_ht if (args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None) else torch.tensor(0, device=device)

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
    description += f", Emo: {np.mean(loss_log['emo']):.3e}"   # 情感损失-交叉熵
    description += ']'
    print(description)

    # write to tensorboard
    if writer is not None:
        writer.add_scalar(f'{mode}/total_loss', np.mean(loss_log['loss']), current_iter)
        writer.add_scalar(f'{mode}/emotion_loss', np.mean(loss_log['emo']), current_iter)
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
        model.train()   # 设置为训练模式

# 获取训练参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args, option_text=None):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # model 模型
    model_kwargs = dict(
        device              = device,
        target              = args.target,              # ('--target', type=str, default='sample', choices=['sample', 'noise'])
        architecture        = args.architecture,        # ('--architecture', type=str, default='decoder', choices=['decoder'])
        motion_feat_dim     = args.motion_feat_dim,     # ('--motion_feat_dim', type=int, default=70)
        fps                 = args.fps,                 # ('--fps', type=int, default=25, help='frame per second')
        n_motions           = args.n_motions,           # ('--n_motions', type=int, default=64, help='number of motions in a sequence')
        n_prev_motions      = args.n_prev_motions,      # ('--n_prev_motions', type=int, default=16, help='number of pre-motions in a sequence')
        audio_model         = args.audio_model,         # ('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
        feature_dim         = args.feature_dim,         # ('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
        n_diff_steps        = args.n_diff_steps,        # ('--n_diff_steps', type=int, default=500, help='number of diffusion steps')
        diff_schedule       = args.diff_schedule,       # ('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
        cfg_mode            = args.cfg_mode,            # ('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
        guiding_conditions  = args.guiding_conditions,  # ('--guiding_conditions', type=str, default='audio,emotion')
        align_mask_width    = args.align_mask_width,    # ('--align_mask_width', type=int, default=2, help='width of the alignment mask, non-positive for no mask')
    )

    model = DitTalkingHead(**model_kwargs)            

    exp_dir = Path('experiments/emo_dit') / f'{args.exp_name}'     
    start_iter = 0

    # Dataset
    train_dataset = EmoLevelDataset(args.data_root,                          # 'prepare_data/'  数据集根目录
                                            motion_filename=args.motion_filename,    # 运动文件  'motions.pkl'
                                            motion_template_filename=args.motion_template_filename,   # motion_template.pkl
                                            split="train",
                                            coef_fps=args.fps,                           # 25
                                            n_motions=args.n_motions,                    # 64
                                            n_prev_motions=args.n_prev_motions,          # 16
                                            crop_strategy=args.crop_strategy,            # random
                                            normalize_type=args.normalize_type)          # mix
    # val_dataset = EmoLevelDataset(args.data_root, motion_filename=args.motion_filename,
    #                                        motion_template_filename=args.motion_template_filename, split="val", coef_fps=args.fps, n_motions=args.n_motions,
    #                                        n_prev_motions=args.n_prev_motions,
    #                                        crop_strategy=args.crop_strategy, normalize_type=args.normalize_type)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,     
                                    shuffle=True,
                                    num_workers=args.num_workers,    
                                    pin_memory=True)
    # val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = None

    # 情感分类器
    classifier = Classifier().to(device)
    # classifier.load_state_dict(torch.load(f'pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth', map_location=device), strict=False)
    classifier.load_state_dict(torch.load(f'experiments/emo_classifier/ckpt_n64.pth', map_location=device), strict=False)
    classifier.requires_grad_(False)
    classifier.eval()

    # Logging    TensorBoard的日志
    log_dir = exp_dir / 'logs'                 
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(log_dir))
    if option_text is not None:
        with open(log_dir / 'options.log', 'w') as f:
            f.write(option_text)
        writer.add_text('options', option_text)

    # logger   日志，保存到log_dir/log.txt
    logging.basicConfig(filename=os.path.join(str(log_dir), "log.txt"), 
                    level=logging.INFO,
                    format='%(asctime)s %(message)s', 
                    datefmt='%Y/%m/%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"exp_name: {exp_dir.name}")
    logging.info(f'model parameters: {count_parameters(model)}')

    # optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)  # 选取需要训练的部分  lr=1e-4
    # Scheduler（学习率调度器）用于 动态调整学习率（Learning Rate, LR）
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

    # train
    train(args,
          model,
          train_loader,      # 训练集
          val_loader,        # 测试集
          optimizer,
          exp_dir / 'checkpoints',  
          scheduler,         
          writer,
          start_iter=start_iter,
          classifier=classifier)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--iter', type=int, default=1, help='iteration to test')
    parser.add_argument('--exp_name', type=str, default=g_exp_name, help='experiment name')

    # Dataset
    parser.add_argument('--data_root', type=Path, default="src/my_prepare/",)
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')             # templates
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')     # motion_template
    parser.add_argument('--batch_size', type=int, default=80, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--crop_strategy', type=str, default="random")
    parser.add_argument('--normalize_type', type=str, default="mix", choices=["std", "case", "scale", "minmax", "mix"])

    # Model
    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,emotion')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=500, help='number of diffusion steps')
    parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--no_head_pose', action='store_true', default=False, help='do not predict head pose')
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])

    # transformer
    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=2, help='width of the alignment mask, non-positive for no mask')
    parser.add_argument('--no_use_learnable_pe', action='store_true', help='do not use learnable positional encoding')
    parser.add_argument('--use_indicator', action='store_true', default=True, help='use indicator for padded frames')
    parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=12, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    # sequence
    parser.add_argument('--n_motions', type=int, default=64, help='number of motions in a sequence')
    parser.add_argument('--n_prev_motions', type=int, default=16, help='number of pre-motions in a sequence')
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25, help='frame per second')      
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    # Training
    parser.add_argument('--max_iter', type=int, default=300000, help='max number of iterations')   
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--scheduler', type=str, default='WarmupThenDecay', choices=['None', 'Warmup', 'WarmupThenDecay'])

    # 损失函数 & 权重
    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--l_exp', type=float, default=1.0, help='weight of the head angle loss')
    parser.add_argument('--l_exp_vel', type=float, default=1e-4, help='weight of the head angle loss')     
    parser.add_argument('--l_exp_smooth', type=float, default=1e-4, help='weight of the head angle loss')  
    parser.add_argument('--l_head_angle', type=float, default=1e-2, help='weight of the head angle loss')
    parser.add_argument('--l_head_vel', type=float, default=1e-2, help='weight of the head angular velocity loss')
    parser.add_argument('--l_head_smooth', type=float, default=1e-2, help='weight of the head angular acceleration regularization')
    parser.add_argument('--l_head_trans', type=float, default=1e-2, help='weight of the head constraint during window transition')
    parser.add_argument('--no_constrain_prev', action='store_true', help='do not constrain the generated previous motions')

    parser.add_argument('--use_context_audio_feat', action='store_true')
    parser.add_argument('--trunc_prob1', type=float, default=0.3, help='truncation probability for the first sample')
    parser.add_argument('--trunc_prob2', type=float, default=0.4, help='truncation probability for the second sample')

    parser.add_argument('--save_iter', type=int, default=5000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=50, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')

    # warm_up
    parser.add_argument('--warm_iter', type=int, default=20000)         
    parser.add_argument('--cos_max_iter', type=int, default=200000)    
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)

    args = parser.parse_args()

    if args.mode == 'train':
        option_text = utils.common.get_option_text(args, parser)
    else:
        option_text = None

    main(args, option_text)

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
from src.modules.emotion_dit import DitTalkingHead

device_id = 2  # 选择 GPU
torch.cuda.set_device(device_id)  # 设置默认 GPU
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")  # 显式指定设备

cross_criterion = torch.nn.CrossEntropyLoss()

def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None):

    save_dir.mkdir(parents=True, exist_ok=True) 

    # model
    device = model.device
    model.train()

############
    all_temp = pickle.load(open('pretrained_weights/ADEF/motion_template/motion_template.pkl','rb'))
    mean_exp, std_exp = torch.tensor(all_temp['mean_exp']).to(device).unsqueeze(0).unsqueeze(0), torch.tensor(all_temp['std_exp']).to(device).unsqueeze(0).unsqueeze(0)
    alone_temp = pickle.load(open('pretrained_weights/ADEF/motion_template/emotion_template.pkl','rb'))
    mean_exps, std_exps = [], []
    for i in range(len(alone_temp)):
        mean_exps.append(torch.tensor(alone_temp[i]['mean_exp']))
        std_exps.append(torch.tensor(alone_temp[i]['std_exp']))
    mean_exps = torch.stack(mean_exps,dim=0).to(device)   # [8, 63]
    std_exps = torch.stack(std_exps,dim=0).to(device)     # [8, 63]
    norm_dict = {
        'mean_exp': mean_exp,
        'std_exp': std_exp,
        'mean_exps': mean_exps,
        'std_exps': std_exps
    }
#############

    data_loader = infinite_data_loader(train_loader)   # 将数据加载器（train_loader）转换为一个无限循环的迭代器
    audio_unit = train_loader.dataset.audio_unit       # 每一帧的样本数  self.audio_unit = 16000. / self.coef_fps
    predict_head_pose = not args.no_head_pose          # False -> True  预测头部姿势
    loss_log = defaultdict(lambda: deque(maxlen=args.log_smooth_win))  # maxlen = 50

    optimizer.zero_grad()
    for it in range(start_iter, args.max_iter + 1):   # 迭代次数  0 ~ 50000
        audio_pair, coef_pair, emo_index, _ = next(data_loader)  # 音频(N=8, 100帧的采样数)*2 及其运动系数{'exp':[(8, 100, 63)],'pose':[(8, 100, 7)]} *2  
        audio_pair = [audio.to(device) for audio in audio_pair]
        coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]  # 每一种系数（旋转、表情等）都放入device
        motion_coef_pair = [  # 按照关键点的位置进行拼接。   (8, 100, 70) * 2     即 torch.cat([coef_dict['exp'], coef_dict['pose']], dim=-1) 
            utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)  # rot_repr= aa
        ] 
        emo_index = emo_index.to(device)

        # Extract audio features  提取音频特征
        if args.use_context_audio_feat:   # False
            # (N, L_audio) -> (N, L_audio = audio_unit * n_units + pad_threshold) -> (N, 2L=200, 768) -> (N, 768, L) ->  (N, L=100, feature_dim=512)
            audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

        loss_noise = 0                                    # 去噪损失
        loss_emo = torch.tensor(0, device=device)         # 情感损失
        loss_exp = torch.tensor(0, device=device)         # 表情变形损失
        loss_exp_v = torch.tensor(0, device=device)
        loss_exp_s = torch.tensor(0, device=device)
        loss_head_angle = torch.tensor(0, device=device)
        loss_head_vel = torch.tensor(0, device=device)
        loss_head_smooth = torch.tensor(0, device=device)
        loss_head_trans = 0
        for i in range(2):   # 前n_motions 和 后n_motions
            audio = audio_pair[i]  # (N=8, L_a)           N：批次大小  L_a：100帧对应的音频长度（样本）
            motion_coef = motion_coef_pair[i]  # (N=8, L=100, 50+x=70)   L：帧数    50+x：运动参数的总数
            batch_size = audio.shape[0]  # N = 16 

            # truncate input audio and motion according to trunc_prob 根据trunc_prob截断输入音频和运动
            # 随机截断 被截断部分填0            截断是为了适应不同长度的audio
            if (i == 0 and np.random.rand() < args.trunc_prob1                # 0.3  第一个样本的截断概率
                ) or (i != 0 and np.random.rand() < args.trunc_prob2):         # 0.4 第二个样本的截断概率
                # 随机裁断并填充。返回裁断且填充0后的audio，motion_coef，裁断位置索引end_idx
                audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(   
                    audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                if args.use_context_audio_feat and i != 0:   # False
                    # use contextualized audio feature for the second clip  为第二个剪辑使用情境化音频功能
                    audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                           args.n_motions * 2)[:, -args.n_motions:]
            else:  # 不截断
                if args.use_context_audio_feat:   # False
                    audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                else:
                    audio_in = audio      # (N=8, L_a)   
                motion_coef_in, end_idx = motion_coef, None       # motion_coef_in：(N=8, L=100, 系数个数=70)

            if args.use_indicator:  # True  使用指示器：用于指示哪些部分被截断了。
                if end_idx is not None:    # 被截断
                    indicator = torch.arange(args.n_motions, device=device).expand(batch_size, -1) < end_idx.unsqueeze(
                        1)    # 生成一个 0, 1, 2, ..., args.n_motions-1 的一维 Tensor ； expand扩展为(batch_size, args.n_motions)
                    # 与end_idx进行比较，索引超出end_idx的部分都是 False，说明这些部分被截断了.
                else:
                    indicator = torch.ones(batch_size, args.n_motions, device=device)   # (batch_size, args.n_motions)全1张量,没被截断
            else:
                indicator = None

            # Inference   使用DiT推理得到noise, target, prev_motion_coef, prev_audio_feat
            if i == 0:    # 前n_motions，没有先前的特征
                noise, target, prev_motion_coef, prev_audio_feat = model(motion_coef_in, audio_in, indicator=indicator, emo_index = emo_index)  #  (8, 100, 70) , (8, 125, 70) , (8, 100, 70) , (8, 100, 256)
                if end_idx is not None:  # was truncated, needs to use the complete feature  被截断，需要使用完整功能
                    prev_motion_coef = motion_coef[:, -args.n_prev_motions:]   #  (8, 25, 70)  选取前 n_prev_motions=25 帧作为 先前运动特征 
                    if args.use_context_audio_feat:   # False
                        # 前面已经提取过了audio_feat，直接截取部分即可
                        prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions].detach()
                    else:
                        with torch.no_grad():   # 音频->音频特征   并选取结果的 前 n_prev_motions=25 帧作为 先前音频特征
                            prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]  # (8 25 256)
                else:   # 没被截断，直接使用预测结果的  前n_prev_motions=25 帧 作为参考
                    prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]   #  (8, 25, 70)
                    prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]     #  (8  25  256)
            else:  #  后n_motions部分：使用前n_motions的特征作为先前特征
                noise, target, _, _ = model(motion_coef_in, audio_in, prev_motion_coef, prev_audio_feat, indicator=indicator, emo_index = emo_index)
                #  (8, 100, 70) , (8, 125, 70)
            loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)

            exps = target[:, args.n_prev_motions:, :63].clone()   # (N,100,63)    各自归一化的

            # 增强归一化
            alone_mean = mean_exps[emo_index].unsqueeze(1)        # (N,1,63)
            alone_std = std_exps[emo_index].unsqueeze(1)          # (N,1,63)

            exps = (exps * alone_std + alone_mean - mean_exp) / (std_exp + 1e-9)             # 反归一化 再 归一化

            # 情感分类（情感损失）
            pred_emo, _ = classifier(exps)   # (N,100,63)  -> (N,8)
            loss_e = cross_criterion(pred_emo, emo_index)
            loss_emo = loss_emo + loss_e / 2

            loss_noise = loss_noise + loss_n / 2        # 前n_motions和后n_motions各占一半，因此除以2。下同
            loss_exp = loss_exp + loss_exp / 2
            loss_exp_v = loss_exp_v + loss_exp_v / 2.
            loss_exp_s = loss_exp_s + loss_exp_s / 2.
            # 预测头部姿势，且有相应的权重和loss值
            if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                loss_head_angle = loss_head_angle + loss_ha / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                loss_head_vel = loss_head_vel + loss_hc / 2
            if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                loss_head_smooth = loss_head_smooth + loss_hs / 2
            # 无需除以2，因为它只适用于第二个剪辑
            if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                # no need to divide by 2 because it only applies to the second clip
                loss_head_trans = loss_head_trans + loss_ht

        # 扩散（采样）损失
        loss_log['noise'].append(loss_noise.item())
        loss = loss_noise

        # 情感损失
        loss_log['emo'].append(loss_emo.item())
        loss = loss + loss_emo

        # 表情相关损失(计算级联损失时，需要乘以相应的权重)
        loss_log['exp'].append(loss_exp.item() * args.l_exp)             # l_exp： 0.1  权重
        loss = loss + args.l_exp * loss_exp

        loss_log['exp_vel'].append(loss_exp_v.item() * args.l_exp_vel)   # l_exp_vel： 1e-4  权重
        loss = loss + args.l_exp_vel * loss_exp_v

        loss_log['exp_smooth'].append(loss_exp_s.item() * args.l_exp_smooth)  # l_exp_smooth： 1e-4  权重
        loss = loss + args.l_exp_smooth * loss_exp_s

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

        loss.backward()   # 级联损失

        if args.clip_grad:       # 梯度裁剪（gradient clipping）   防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 如果梯度的 L2 范数大于 2.0，则进行裁剪，使其不超过 2.0

        if it % args.gradient_accumulation_steps == 0:     # 梯度累积（gradient accumulation）  gradient_accumulation_steps=1
            # 累积 N 次梯度后再更新一次参数 N= gradient_accumulation_steps=1
            optimizer.step()
            optimizer.zero_grad()

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
        if scheduler is not None:   # 调度器用于更新学习率  区分：优化器optimizor
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()

        # save model   保存模型中间结果
        if (it % args.save_iter == 0 and it != 0) or it == args.max_iter: # 每1000次迭代 保存一次。 第50000次 保存一次
            torch.save({
                'args': args,                 # args：模型参数
                'model': model.state_dict(),   # 模型
                'iter': it,                    # 训练轮次
            }, save_dir / f'iter_{it:07}.pt')   

        # validation  验证模型
        if (it % args.val_iter == 0 or it == 0) or it == args.max_iter:  # 每50次迭代 验证一次。 第0次和第50000次 验证一次
            val(args, model, val_loader, it, 1, 'val', writer, norm_dict,classifier)

# 测试部分
@torch.no_grad()
def val(args, model, test_loader, current_iter, n_rounds=1, mode='val', writer=None, norm_dict=None, classifier=None):
    # print("test ... ")
    is_training = model.training
    device = model.device
    model.eval()   # 设置为eval模式

    audio_unit = test_loader.dataset.audio_unit
    predict_head_pose = not args.no_head_pose

############
    mean_exp, std_exp = norm_dict['mean_exp'], norm_dict['std_exp']
    mean_exps, std_exps = norm_dict['mean_exps'], norm_dict['std_exps']
#############

    loss_log = defaultdict(list)
    for test_round in range(n_rounds):     # 1  只测试一次
        # 后面的代码与训练部分基本一致。。
        for audio_pair, coef_pair, emo_index, _ in test_loader:
            audio_pair = [audio.to(device) for audio in audio_pair]
            coef_pair = [{x: coef_pair[i][x].to(device) for x in coef_pair[i]} for i in range(2)]
            motion_coef_pair = [
                utils.get_motion_coef(coef_pair[i], args.rot_repr, predict_head_pose) for i in range(2)
            ]  # (N, L, 50+x)
            emo_index = emo_index.to(device)

            # Extract audio features
            if args.use_context_audio_feat:   # False
                audio_feat = model.extract_audio_feature(torch.cat(audio_pair, dim=1), args.n_motions * 2)  # (N, 2L, :)

            loss_noise = 0
            loss_emo = torch.tensor(0, device=device)
            loss_exp = 0
            loss_exp_v = 0
            loss_exp_s = 0
            loss_head_angle = 0
            loss_head_vel = torch.tensor(0, device=device)
            loss_head_smooth = torch.tensor(0, device=device)
            loss_head_trans = 0
            for i in range(2):   # 前n_motions 和 后n_motions
                audio = audio_pair[i]  # (N, L_a)
                motion_coef = motion_coef_pair[i]  # (N, L, 50+x)
                batch_size = audio.shape[0]

                # truncate input audio and motion according to trunc_prob
                if (i == 0 and np.random.rand() < args.trunc_prob1) or (i != 0 and np.random.rand() < args.trunc_prob2):
                    audio_in, motion_coef_in, end_idx = utils.truncate_motion_coef_and_audio(
                        audio, motion_coef, args.n_motions, audio_unit, args.pad_mode)
                    if args.use_context_audio_feat and i != 0:   # False
                        # use contextualized audio feature for the second clip
                        audio_in = model.extract_audio_feature(torch.cat([audio_pair[i - 1], audio_in], dim=1),
                                                               args.n_motions * 2)[:, -args.n_motions:]
                else:
                    if args.use_context_audio_feat:   # False
                        audio_in = audio_feat[:, i * args.n_motions:(i + 1) * args.n_motions]
                    else:
                        audio_in = audio
                    motion_coef_in, end_idx = motion_coef, None

                if args.use_indicator:
                    if end_idx is not None:
                        indicator = torch.arange(args.n_motions, device=device).expand(batch_size,
                                                                                       -1) < end_idx.unsqueeze(1)
                    else:
                        indicator = torch.ones(batch_size, args.n_motions, device=device)
                else:
                    indicator = None

                # Inference
                if i == 0:
                    noise, target, prev_motion_coef, prev_audio_feat = model(motion_coef_in, audio_in, indicator=indicator, emo_index=emo_index)
                    if end_idx is not None:  # was truncated, needs to use the complete feature
                        prev_motion_coef = motion_coef[:, -args.n_prev_motions:]
                        if args.use_context_audio_feat:   # False
                            prev_audio_feat = audio_feat[:, args.n_motions - args.n_prev_motions:args.n_motions]
                        else:
                            with torch.no_grad():
                                prev_audio_feat = model.extract_audio_feature(audio)[:, -args.n_prev_motions:]
                    else:
                        prev_motion_coef = prev_motion_coef[:, -args.n_prev_motions:]
                        prev_audio_feat = prev_audio_feat[:, -args.n_prev_motions:]
                else:
                    noise, target, _, _ = model(motion_coef_in, audio_in, prev_motion_coef, prev_audio_feat, indicator=indicator, emo_index=emo_index)

                loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)

                exps = target[:, args.n_prev_motions:, :63].clone()   # (N,100,63)    各自归一化的

                # 增强归一化
                alone_mean = mean_exps[emo_index].unsqueeze(1)        # (N,1,63)
                alone_std = std_exps[emo_index].unsqueeze(1)          # (N,1,63)

                exps = (exps * alone_std + alone_mean - mean_exp) / (std_exp + 1e-9)             # 反归一化 再 归一化

                # 情感分类（情感损失）
                pred_emo, _ = classifier(exps)   # (N,100,63)  -> (N,8)
                loss_e = cross_criterion(pred_emo, emo_index)
                loss_emo = loss_emo + loss_e / 2

                # simple loss   简单损失：真实运动序列 和 生成的干净运动序列 之间的L2距离。
                loss_noise = loss_noise + loss_n / 2    # 前n_motions和后n_motions各占一半，因此除以2。下同

                # exp-related loss 表情相关损失
                loss_exp = loss_exp + loss_exp / 2
                loss_exp_v = loss_exp_v + loss_exp_v / 2
                loss_exp_s = loss_exp_s + loss_exp_s / 2
                
                # head pose loss   头部姿势损失
                if args.target == 'sample' and predict_head_pose and args.l_head_angle > 0:
                    loss_head_angle = loss_head_angle + loss_ha / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_vel > 0 and loss_hc is not None:
                    loss_head_vel = loss_head_vel + loss_hc / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_smooth > 0 and loss_hs is not None:
                    loss_head_smooth = loss_head_smooth + loss_hs / 2
                if args.target == 'sample' and predict_head_pose and args.l_head_trans > 0 and loss_ht is not None:
                    # no need to divide by 2 because it only applies to the second clip
                    loss_head_trans = loss_head_trans + loss_ht

            loss_log['noise'].append(loss_noise.item())
            loss = loss_noise
            
            loss_log['emo'].append(loss_emo.item())
            loss = loss + loss_emo

            loss_log['exp'].append(loss_exp.item() * args.l_exp)
            loss = loss + args.l_exp * loss_exp

            loss_log['exp_vel'].append(loss_exp_v.item() * args.l_exp_vel)
            loss = loss + args.l_exp_vel * loss_exp_v

            loss_log['exp_smooth'].append(loss_exp_s.item() * args.l_exp_smooth)
            loss = loss + args.l_exp_smooth * loss_exp_s

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
    # model 模型
    model_kwargs = dict(
        device              = device,  
        target              = args.target,              # ('--target', type=str, default='sample', choices=['sample', 'noise'])
        architecture        = args.architecture,        # ('--architecture', type=str, default='decoder', choices=['decoder'])
        motion_feat_dim     = args.motion_feat_dim,     # ('--motion_feat_dim', type=int, default=70)
        fps                 = args.fps,                 # ('--fps', type=int, default=25, help='frame per second')
        n_motions           = args.n_motions,           # ('--n_motions', type=int, default=100, help='number of motions in a sequence')
        n_prev_motions      = args.n_prev_motions,      # ('--n_prev_motions', type=int, default=25, help='number of pre-motions in a sequence')
        audio_model         = args.audio_model,         # ('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
        feature_dim         = args.feature_dim,         # ('--feature_dim', type=int, default=256, help='dimension of the hidden feature')
        n_diff_steps        = args.n_diff_steps,        # ('--n_diff_steps', type=int, default=50, help='number of diffusion steps')
        diff_schedule       = args.diff_schedule,       # ('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
        cfg_mode            = args.cfg_mode,            # ('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
        guiding_conditions  = args.guiding_conditions,  # ('--guiding_conditions', type=str, default='audio,')
    )

    model = DitTalkingHead(**model_kwargs)            

    exp_dir = Path('experiments/emo_dit') / f'{args.exp_name}'     
    start_iter = 0
    # ckpt_dir = exp_dir / 'checkpoints'       
    # pt_files = list(ckpt_dir.glob('*.pt'))
    # if pt_files:   # 模型加载（lr的加载没完成，建议还是从头开始。。。）
    #     ckpt_path = sorted(pt_files, reverse=True)[0]    # 获取最新的模型文件路径
    #     model_data = torch.load(ckpt_path, map_location=device)
    #     model.load_state_dict(model_data['model'], strict=False)
    #     start_iter = model_data['iter'] + 1
    #     print(f"Loading model from {ckpt_path}, start from iter {start_iter}.")

    # Dataset
    train_dataset = EmoLevelDataset(args.data_root,                          # 'prepare_data/'  数据集根目录
                                            motion_filename=args.motion_filename,    # 运动文件  'motions.pkl'
                                            motion_template_filename=args.motion_template_filename,   # motion_template.pkl
                                            split="train",
                                            coef_fps=args.fps,                           # 25    为什么是30呢？？？
                                            n_motions=args.n_motions,                    # 100
                                            crop_strategy=args.crop_strategy,            # random
                                            normalize_type=args.normalize_type)          # mix
    val_dataset = EmoLevelDataset(args.data_root, motion_filename=args.motion_filename, 
                                           motion_template_filename=args.motion_template_filename, split="val", coef_fps=args.fps, n_motions=args.n_motions, 
                                           crop_strategy=args.crop_strategy, normalize_type=args.normalize_type)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=args.batch_size,       # 16
                                    shuffle=True,
                                    num_workers=args.num_workers,     # 4
                                    pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 情感分类器
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(f'pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth', map_location=device), strict=False)
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
    parser.add_argument('--iter', type=int, default=1, help='iteration to test')
    parser.add_argument('--exp_name', type=str, default='emo_dit', help='experiment name')

    # Dataset
    parser.add_argument('--data_root', type=Path, default="src/my_prepare/",)
    parser.add_argument('--motion_filename', type=str, default='front_all_motions.pkl')             # templates
    parser.add_argument('--motion_template_filename', type=str, default='motion_template.pkl')     # motion_template
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--crop_strategy', type=str, default="random")
    parser.add_argument('--normalize_type', type=str, default="mix", choices=["std", "case", "scale", "minmax", "mix"])

    # Model
    parser.add_argument('--target', type=str, default='sample', choices=['sample', 'noise'])
    parser.add_argument('--guiding_conditions', type=str, default='audio,emotion')
    parser.add_argument('--cfg_mode', type=str, default='incremental', choices=['incremental', 'independent'])
    parser.add_argument('--n_diff_steps', type=int, default=50, help='number of diffusion steps')
    parser.add_argument('--diff_schedule', type=str, default='cosine', choices=['linear', 'cosine', 'quadratic', 'sigmoid'])
    parser.add_argument('--no_head_pose', action='store_true', default=False, help='do not predict head pose')
    parser.add_argument('--rot_repr', type=str, default='aa', choices=['aa'])

    # transformer
    parser.add_argument('--audio_model', type=str, default='wav2vec2', choices=['wav2vec2', 'hubert', 'hubert_zh', 'hubert_zh_ori'])
    parser.add_argument('--architecture', type=str, default='decoder', choices=['decoder'])
    parser.add_argument('--align_mask_width', type=int, default=1, help='width of the alignment mask, non-positive for no mask')
    parser.add_argument('--no_use_learnable_pe', action='store_true', help='do not use learnable positional encoding')
    parser.add_argument('--use_indicator', action='store_true', default=True, help='use indicator for padded frames')
    parser.add_argument('--feature_dim', type=int, default=512, help='dimension of the hidden feature')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--n_layers', type=int, default=6, help='number of encoder/decoder layers')
    parser.add_argument('--mlp_ratio', type=int, default=4, help='ratio of the hidden dimension of the MLP')

    # sequence
    parser.add_argument('--n_motions', type=int, default=100, help='number of motions in a sequence')
    parser.add_argument('--n_prev_motions', type=int, default=25, help='number of pre-motions in a sequence')
    parser.add_argument('--motion_feat_dim', type=int, default=70)
    parser.add_argument('--fps', type=int, default=25, help='frame per second')      
    parser.add_argument('--pad_mode', type=str, default='zero', choices=['zero', 'replicate'])

    # Training
    parser.add_argument('--max_iter', type=int, default=100000, help='max number of iterations')   
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation')
    parser.add_argument('--scheduler', type=str, default='WarmupThenDecay', choices=['None', 'Warmup', 'WarmupThenDecay'])

    # 损失函数 & 权重
    parser.add_argument('--criterion', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--clip_grad', default=True, action='store_true')
    parser.add_argument('--l_exp', type=float, default=0.1, help='weight of the head angle loss')
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

    parser.add_argument('--save_iter', type=int, default=1000, help='save model every x iterations')
    parser.add_argument('--val_iter', type=int, default=50, help='validate every x iterations')
    parser.add_argument('--log_iter', type=int, default=50, help='log to tensorboard every x iterations')
    parser.add_argument('--log_smooth_win', type=int, default=50, help='smooth window for logging')

    # warm_up
    parser.add_argument('--warm_iter', type=int, default=10000)         
    parser.add_argument('--cos_max_iter', type=int, default=100000)    
    parser.add_argument('--min_lr_ratio', type=float, default=0.02)

    args = parser.parse_args()

    if args.mode == 'train':
        option_text = utils.common.get_option_text(args, parser)
    else:
        option_text = None

    main(args, option_text)
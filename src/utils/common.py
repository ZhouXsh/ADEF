from functools import reduce
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Normal

class NullableArgs:
    def __init__(self, namespace):
        for key, value in namespace.__dict__.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        # when an attribute lookup has not found the attribute
        if key == 'align_mask_width':
            if 'use_alignment_mask' in self.__dict__:
                return 1 if self.use_alignment_mask else 0
            else:
                return 0
        if key == 'no_head_pose':
            return not self.predict_head_pose
        if key == 'no_use_learnable_pe':
            return not self.use_learnable_pe

        return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_option_text(args, parser):
    '''优先选取传入值，否则选取默认值'''
    message = ''
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f'\t[default: {str(default)}]'
        message += f'{str(k):>30}: {str(v):<30}{comment}\n'
    return message


def get_model_path(exp_name, iteration, model_type='DPT'):
    exp_root_dir = Path(__file__).parent.parent / 'experiments' / model_type
    exp_dir = exp_root_dir / exp_name
    if not exp_dir.exists():
        exp_dir = next(exp_root_dir.glob(f'{exp_name}*'))
    model_path = exp_dir / f'checkpoints/iter_{iteration:07}.pt'
    return model_path, exp_dir.relative_to(exp_root_dir)


def get_pose_input(coef_dict, rot_repr, with_global_pose):
    if rot_repr == 'aa':     # this
        pose_input = coef_dict['pose'] if with_global_pose else coef_dict['pose'][..., 63:70]
        # Remove mouth rotation round y, z axis
        # pose_input = pose_input[..., :-2]
    else:
        raise ValueError(f'Unknown rotation representation: {rot_repr}')
    return pose_input


def get_motion_coef(coef_dict, rot_repr, with_global_pose=False, norm_stats=None):  # norm_stats: 归一化参数
    if norm_stats is not None:     # 需要归一化
        if rot_repr == 'aa':       # 旋转表示rotation representation 
            keys = ['exp', 'pose']
            coef_dict = {k: (coef_dict[k] - norm_stats[f'{k}_mean']) / norm_stats[f'{k}_std'] for k in keys}  # 标准化
            pose_coef = get_pose_input(coef_dict, rot_repr, with_global_pose)
            return torch.cat([coef_dict['exp'], pose_coef], dim=-1)
        elif rot_repr == 'emo':
            print(f"coef_dict.keys(): {coef_dict.keys()}")
            keys = ['exp', 'pose', 'emotion']
            return torch.cat([coef_dict['exp'], coef_dict["pose"], coef_dict["emotion"]], dim=-1)

            # return torch.cat([[coef_dict[key]] for key in keys], dim=-1)
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')
    else:      # this
        if rot_repr == 'aa':      # this   'aa'（轴角表示，Axis-Angle Representation）
            keys = ['exp', 'pose']
            pose_coef = get_pose_input(coef_dict, rot_repr, with_global_pose)   # coef_dict['pose'] if with_global_pose else coef_dict['pose'][..., 63:70]
            return torch.cat([coef_dict['exp'], pose_coef], dim=-1)
        elif rot_repr == 'emo':
            # print(f"coef_dict.keys(): {coef_dict.keys()}")
            keys = ['exp', 'pose', 'emotion']
            return torch.cat([coef_dict['exp'], coef_dict["pose"], coef_dict["emotion"]], dim=-1)
            # return torch.cat([[coef_dict[key]] for key in keys], dim=-1)
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')




def get_coef_dict(motion_coef, shape_coef=None, denorm_stats=None, with_global_pose=False, rot_repr='aa'):
    coef_dict = {
        'exp': motion_coef[..., :63]
    }
    if rot_repr == 'aa':
        if with_global_pose:
            coef_dict['pose'] = motion_coef[..., 63:]
        else:
            placeholder = torch.zeros_like(motion_coef[..., :3])
            coef_dict['pose'] = torch.cat([placeholder, motion_coef[..., -1:]], dim=-1)
        # Add back rotation around y, z axis
        # coef_dict['pose'] = torch.cat([coef_dict['pose'], torch.zeros_like(motion_coef[..., :2])], dim=-1)
    else:
        raise ValueError(f'Unknown rotation representation {rot_repr}!')

    if denorm_stats is not None:
        coef_dict = {k: coef_dict[k] * denorm_stats[f'{k}_std'] + denorm_stats[f'{k}_mean'] for k in coef_dict}

    if not with_global_pose:
        if rot_repr == 'aa':
            coef_dict['pose'][..., :3] = 0
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')

    return coef_dict


def coef_dict_to_vertices(coef_dict, flame, rot_repr='aa', ignore_global_rot=False, flame_batch_size=512):
    shape = coef_dict['exp'].shape[:-1]
    coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
    n_samples = reduce(lambda x, y: x * y, shape, 1)

    # Convert to vertices
    vert_list = []
    for i in range(0, n_samples, flame_batch_size):
        batch_coef_dict = {k: v[i:i + flame_batch_size] for k, v in coef_dict.items()}
        if rot_repr == 'aa':
            vert, _, _ = flame(
                batch_coef_dict['shape'], batch_coef_dict['exp'], batch_coef_dict['pose'],
                pose2rot=True, ignore_global_rot=ignore_global_rot, return_lm2d=False, return_lm3d=False)
        else:
            raise ValueError(f'Unknown rot_repr: {rot_repr}')
        vert_list.append(vert)

    vert_list = torch.cat(vert_list, dim=0)  # (n_samples, 5023, 3)
    vert_list = vert_list.view(*shape, -1, 3)  # (..., 5023, 3)

    return vert_list

# 裁断 音频
# 被裁断的部分进行填充
def _truncate_audio(audio, end_idx, pad_mode='zero'):  # 音频； 结束的索引；  填充方式
    batch_size = audio.shape[0]
    audio_trunc = audio.clone()
    if pad_mode == 'replicate':
        for i in range(batch_size):
            audio_trunc[i, end_idx[i]:] = audio_trunc[i, end_idx[i] - 1]  # 重复最后一个采样
    elif pad_mode == 'zero':
        for i in range(batch_size):
            audio_trunc[i, end_idx[i]:] = 0   # 被裁断的部分填充0
    else:
        raise ValueError(f'Unknown pad mode {pad_mode}!')

    return audio_trunc

# 裁断 系数字典
# 被裁断的部分进行填充
def _truncate_coef_dict(coef_dict, end_idx, pad_mode='zero'):
    # coef_dict = {'exp': motion_coef[..., :63], 'pose_any': motion_coef[..., 63:]}   # 表情：0~62   姿势：63~69
    batch_size = coef_dict['exp'].shape[0]
    coef_dict_trunc = {k: v.clone() for k, v in coef_dict.items()}
    if pad_mode == 'replicate': 
        for i in range(batch_size):      # 每一批次
            for k in coef_dict_trunc:    #  'exp' and 'pose_any'
                coef_dict_trunc[k][i, end_idx[i]:] = coef_dict_trunc[k][i, end_idx[i] - 1]    # 重复最后一个系数
    elif pad_mode == 'zero':
        for i in range(batch_size):
            for k in coef_dict:
                coef_dict_trunc[k][i, end_idx[i]:] = 0   # 填0
    else:
        raise ValueError(f'Unknown pad mode: {pad_mode}!')

    return coef_dict_trunc

# 裁断 系数字典 和 音频  （没用）
def truncate_coef_dict_and_audio(audio, coef_dict, n_motions, audio_unit=640, pad_mode='zero'):
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
    audio_end_idx = (end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio
    audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # truncate coef dict
    coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)

    return audio_trunc, coef_dict_trunc, end_idx

# 截断 运动系数 和 音频
# 被裁断的部分填充0
def truncate_motion_coef_and_audio(audio, motion_coef,    # 音频 ； 运动系数
                                   n_motions,             # 100
                                   audio_unit=640, pad_mode='zero'):
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)     # 随机裁断位置索引
    audio_end_idx = (end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio 截断音频   被裁断的部分填充0
    audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # prepare coef dict and stats  准备系数字典 和 
    coef_dict = {'exp': motion_coef[..., :63], 'pose_any': motion_coef[..., 63:]}   # 表情：0~62   姿势：63~69

    # truncate coef dict   裁断系数字典    被裁断的部分填充0
    coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)
    motion_coef_trunc = torch.cat([coef_dict_trunc['exp'], coef_dict_trunc['pose_any']], dim=-1)

    return audio_trunc, motion_coef_trunc, end_idx


def nt_xent_loss(feature_a, feature_b, temperature):
    """
    Normalized temperature-scaled cross entropy loss.

    (Adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py)

    Args:
        feature_a (torch.Tensor): shape (batch_size, feature_dim)
        feature_b (torch.Tensor): shape (batch_size, feature_dim)
        temperature (float): temperature scaling factor

    Returns:
        torch.Tensor: scalar
    """
    batch_size = feature_a.shape[0]
    device = feature_a.device

    features = torch.cat([feature_a, feature_b], dim=0)

    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)

    # select the positives and negatives
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    labels = torch.zeros(labels.shape[0], dtype=torch.long).to(device)

    loss = F.cross_entropy(logits, labels)
    return loss

# utils.compute_loss_new(args, i == 0, motion_coef_in, noise, target, prev_motion_coef, end_idx)
# 是否开始采样； livePortrait的运动系数； 去噪添加的噪声； 去噪的结果；   先前的运动系数（两阶段是一样的）；   截断位置的索引
def compute_loss_new(args, is_starting_sample, motion_coef_gt, noise, target, prev_motion_coef, end_idx=None):
    '''计算损失
    表情类损失：loss_exp，loss_exp_vel，loss_exp_smooth 。
    头部运动类：loss_head_angle，loss_head_vel，loss_head_smooth 。
    Trans类：loss_head_trans_vel，loss_head_trans_accel，loss_head_trans 。
    '''
    # 定义损失函数
    if args.criterion.lower() == 'l2':   # 均方差
        criterion_func = F.mse_loss
    elif args.criterion.lower() == 'l1':  # L1损失
        criterion_func = F.l1_loss
    else:
        raise NotImplementedError(f'Criterion {args.criterion} not implemented.')

    # 表情类损失
    loss_exp = None          # 表情本身的损失 √
    loss_exp_vel = None      # 表情的速度损失 √
    loss_exp_smooth = None   # 表情的平滑损失 √
    # 头部运动类
    loss_head_angle = None    # 头部pose本身的损失 √
    loss_head_vel = None      # 头部pose的速度损失 √
    loss_head_smooth = None   # 头部pose的平滑损失 √
    # Trans类          ？？？？
    loss_head_trans_vel = None    # ？？？？的速度损失
    loss_head_trans_accel = None  # ？？？？的平滑损失
    loss_head_trans = None        # ？？？？的总损失  = loss_head_trans_vel + loss_head_trans_accel 
    
    if args.target == 'noise':
        # 简单损失：计算真实运动序列motion_coef_gt 与 生成的干净运动序列target 之间的L2距离。
        loss_noise = criterion_func(noise, target[:, args.n_prev_motions:], reduction='none')
    elif args.target == 'sample':
        if is_starting_sample:    # 前n_motions部分
            target = target[:, args.n_prev_motions:]
        else:   # 后n_motions部分
            # motion_coef_gt和target都需要拼接 先前的运动特征
            motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_gt], dim=1)
            if args.no_constrain_prev:  # False   不约束生成的先前运动
                target = torch.cat([prev_motion_coef, target[:, args.n_prev_motions:]], dim=1)

        # print(f"loss, motion_coef_gt: {motion_coef_gt.shape}, target: {target.shape}")
        loss_noise = criterion_func(motion_coef_gt, target, reduction='none')   # 提取真实的 与 去噪采样生成的
        # print("loss_noise: ", loss_noise)

        # 表情相关损失
        if args.rot_repr == "aa":         # （轴角表示，"aa"）
            exp_gt = motion_coef_gt[:, :, :63]   # 真实表情
            exp_pred = target[:, :, :63]         # 预测的表情
        elif args.rot_repr == "emo":   # 带情感的表情。  其中情感是motion系数的最后三维
            exp_gt = torch.cat([motion_coef_gt[:, :, :63], motion_coef_gt[:, :, -3:]], -1)
            exp_pred = torch.cat([target[:, :, :63], target[:, :, -3:]], -1)
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')
        loss_exp = criterion_func(exp_gt, exp_pred, reduction='none')  # 真实与预测表情的损失

        if args.l_exp_vel > 0:       # 表情速度损失    1e-4
            vel_exp_gt = exp_gt[:, 1:] - exp_gt[:, :-1]         # 真实表情的后一帧减前一帧  变化量  delta△
            vel_exp_pred = exp_pred[:, 1:] - exp_pred[:, :-1]   # 预测表情的 变化量
            loss_exp_vel = criterion_func(vel_exp_gt, vel_exp_pred, reduction='none')
        if args.l_exp_smooth > 0:   # 表情平滑损失   1e-4
            vel_exp_pred = exp_pred[:, 1:] - exp_pred[:, :-1]    # 预测表情的 变化量
            loss_exp_smooth = criterion_func(vel_exp_pred[:, 1:], vel_exp_pred[:, :-1], reduction='none')

        # 头部pose相关损失
        if not args.no_head_pose:    # 需要头部姿势
            if args.rot_repr == 'aa': # 旋转表征，aa，单独计算exp和（R, t, delta）
                head_pose_gt = motion_coef_gt[:, :, 63:]    # 真实值pose
                head_pose_pred = target[:, :, 63:]           # 预测pose
            elif args.rot_repr == 'emo': # 单独计算exp和（R, t, delta）
                head_pose_gt = motion_coef_gt[:, :, 63:70]
                head_pose_pred = target[:, :, 63:70]
            else:
                raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

            # angle, gt_pose和pred_pose之间的损失
            if args.l_head_angle > 0:   # 预测值与真实值的loss         1e-2
                loss_head_angle = criterion_func(head_pose_gt, head_pose_pred, reduction='none')
            if args.l_head_vel > 0:               # 1e-2
                # print("head_pose_gt: ", head_pose_gt.shape, head_pose_pred.shape)
                head_vel_gt = head_pose_gt[:, 1:] - head_pose_gt[:, :-1]         # 真实变化量
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]   # 预测变化量
                loss_head_vel = criterion_func(head_vel_gt, head_vel_pred, reduction='none')
            if args.l_head_smooth > 0:           # 1e-2
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                loss_head_smooth = criterion_func(head_vel_pred[:, 1:], head_vel_pred[:, :-1], reduction='none')

            # 窗口过渡期间头部约束的权重
            if not is_starting_sample and args.l_head_trans > 0:      # 后n_motions部分  1e-2
                # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})  约束预测的先前和当前运动
                # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})  仅约束预测的当前运动
                head_pose_trans = torch.cat([head_pose_gt[:, args.n_prev_motions - 3:args.n_prev_motions],            # [:, 22:25]
                                             head_pose_pred[:, args.n_prev_motions:args.n_prev_motions + 3]], dim=1)  # [:, 25:28]
                # 按dim=0即帧的维度进行拼接。最后得到[B, 6, 特征维度]   22~27
                head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]  # 23-22  ;  24-23 ; 25-24  ;  26-25  ;  27-26  
                head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]    # (24-23)-(23-22) ； (25-24)-(24-23) ； (26-25)-(25-24) ；(27-26)-(26-25)
                # will constrain x_{-2|0} ~ x_{1}  将会约束
                loss_head_trans_vel = criterion_func(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3], reduction='none')
                # will constrain x_{-3|0} ~ x_{2}
                loss_head_trans_accel = criterion_func(head_accel_pred[:, 1:], head_accel_pred[:, :-1], reduction='none')
    else:  # 报错
        raise ValueError(f'Unknown diffusion target: {args.target}')

    # 计算掩码，用于区分是否被截断
    if end_idx is None:  # 没被裁断
        mask = torch.ones((target.shape[0], args.n_motions), dtype=torch.bool, device=target.device)  
        # （B，n_motions=100）   全1张量
    else:   # 被裁断
        mask = torch.arange(args.n_motions, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)
        # [0,1,2,3,...,n_motions-1=99]  扩展到(B,n_motions)  其中大于end_idx即被截断的部分为False，其他为True

    if args.target == 'sample' and not is_starting_sample:      # 采样模式  且是 后n_motions部分
        if args.no_constrain_prev:   # False
            # Warning: this option will be deprecated in the future  警告：此选项将来将被弃用
            mask = torch.cat([torch.zeros_like(mask[:, :args.n_prev_motions]), mask], dim=1)
            # shape：(B,n_prev_motions + n_motions)  其中前半全0张量，后半掩码
        else:  # this
            mask = torch.cat([torch.ones_like(mask[:, :args.n_prev_motions]), mask], dim=1)
            # shape：(B,n_prev_motions + n_motions)  其中前半全1张量，后半掩码

    # mask 用于区分 被裁断的部分。其中被裁断的部分不参与计算，其他帧 计算均值。
    # mask：(B,n_prev_motions + n_motions)
    # n_prev_motions全为1（或0），表示先前帧的部分都保留（用于计算均值）
    loss_noise = loss_noise[mask].mean()
    if loss_exp is not None:
        loss_exp = loss_exp[mask].mean()
    if loss_exp_vel is not None:
        loss_exp_vel = loss_exp_vel[mask[:, 1:]].mean()   # 速度损失少一帧
    if loss_exp_smooth is not None:
        loss_exp_smooth = loss_exp_smooth[mask[:, 2:]].mean()  # 平滑损失少两帧
    if loss_head_angle is not None:
        loss_head_angle = loss_head_angle[mask].mean()
    if loss_head_vel is not None:
        loss_head_vel = loss_head_vel[mask[:, 1:]]
        loss_head_vel = loss_head_vel.mean() if torch.numel(loss_head_vel) > 0 else None  # 如果张量loss_head_vel中的元素个数大于0
    if loss_head_smooth is not None:
        loss_head_smooth = loss_head_smooth[mask[:, 2:]]
        loss_head_smooth = loss_head_smooth.mean() if torch.numel(loss_head_smooth) > 0 else None
    if loss_head_trans_vel is not None:     # mask:  (B,n_prev_motions + n_motions)
        vel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 2]     # (B, 25:27)
        accel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 3]   # (B, 25:28)
        loss_head_trans_vel = loss_head_trans_vel[vel_mask].mean()
        loss_head_trans_accel = loss_head_trans_accel[accel_mask].mean()
        loss_head_trans = loss_head_trans_vel + loss_head_trans_accel    # 速度损失 + 平滑损失

    # 当target='noise'时，只有去噪损失。   “sample”时都有
    return loss_noise, loss_exp, loss_exp_vel, loss_exp_smooth, loss_head_angle, loss_head_vel, loss_head_smooth, loss_head_trans
    # loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht =
    # 去噪损失，表情损失，表情速度损失，表情平滑损失，头部姿势损失，~~速度损失，~~平滑损失，？？？总损失


def compute_loss_combine(args, is_starting_sample, motion_coef_gt, noise, target, prev_motion_coef, end_idx=None, target_transf=None):
    '''计算损失
    表情类损失：loss_exp，loss_exp_vel，loss_exp_smooth 。
    头部运动类：loss_head_angle，loss_head_vel，loss_head_smooth 。
    Trans类：loss_head_trans_vel，loss_head_trans_accel，loss_head_trans 。
    '''
    # 定义损失函数
    if args.criterion.lower() == 'l2':   # 均方差
        criterion_func = F.mse_loss
    elif args.criterion.lower() == 'l1':  # L1损失
        criterion_func = F.l1_loss
    else:
        raise NotImplementedError(f'Criterion {args.criterion} not implemented.')

    # 表情类损失
    loss_exp = None          # 表情本身的损失 √
    loss_exp_vel = None      # 表情的速度损失 √
    loss_exp_smooth = None   # 表情的平滑损失 √
    # 头部运动类
    loss_head_angle = None    # 头部pose本身的损失 √
    loss_head_vel = None      # 头部pose的速度损失 √
    loss_head_smooth = None   # 头部pose的平滑损失 √
    # Trans类          ？？？？
    loss_head_trans_vel = None    # ？？？？的速度损失
    loss_head_trans_accel = None  # ？？？？的平滑损失
    loss_head_trans = None        # ？？？？的总损失  = loss_head_trans_vel + loss_head_trans_accel 
    
    if args.target == 'noise':
        # 简单损失：计算真实运动序列motion_coef_gt 与 生成的干净运动序列target 之间的L2距离。
        loss_noise = criterion_func(noise, target[:, args.n_prev_motions:], reduction='none')
    elif args.target == 'sample':
        if is_starting_sample:    # 后n_motions部分
            target = target[:, args.n_prev_motions:]
            target_transf = target_transf[:, args.n_prev_motions:]
        else:   # 后n_motions部分
            # motion_coef_gt和target都需要拼接 先前的运动特征
            motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_gt], dim=1)
            if args.no_constrain_prev:  # False   不约束生成的先前运动
                target = torch.cat([prev_motion_coef, target[:, args.n_prev_motions:]], dim=1)
                target_transf = torch.cat([prev_motion_coef, target_transf[:, args.n_prev_motions:]], dim=1)

        # print(f"loss, motion_coef_gt: {motion_coef_gt.shape}, target: {target.shape}")
        loss_noise = criterion_func(motion_coef_gt, target, reduction='none')   # 提取真实的 与 去噪采样生成的
        # print("loss_noise: ", loss_noise)

        # 表情相关损失
        if args.rot_repr == "aa":         # （轴角表示，"aa"）
            exp_gt = motion_coef_gt[:, :, :63]   # 真实表情
            exp_pred = target_transf[:, :, :63]         # 预测的表情
        elif args.rot_repr == "emo":   # 带情感的表情。  其中情感是motion系数的最后三维
            exp_gt = torch.cat([motion_coef_gt[:, :, :63], motion_coef_gt[:, :, -3:]], -1)
            exp_pred = torch.cat([target_transf[:, :, :63], target_transf[:, :, -3:]], -1)
        else:
            raise ValueError(f'Unknown rotation representation {args.rot_repr}!')
        loss_exp = criterion_func(exp_gt, exp_pred, reduction='none')  # 真实与预测表情的损失

        if args.l_exp_vel > 0:       # 表情速度损失    1e-4
            vel_exp_gt = exp_gt[:, 1:] - exp_gt[:, :-1]         # 真实表情的后一帧减前一帧  变化量  delta△
            vel_exp_pred = exp_pred[:, 1:] - exp_pred[:, :-1]   # 预测表情的 变化量
            loss_exp_vel = criterion_func(vel_exp_gt, vel_exp_pred, reduction='none')
        if args.l_exp_smooth > 0:   # 表情平滑损失   1e-4
            vel_exp_pred = exp_pred[:, 1:] - exp_pred[:, :-1]    # 预测表情的 变化量
            loss_exp_smooth = criterion_func(vel_exp_pred[:, 1:], vel_exp_pred[:, :-1], reduction='none')

        # 头部pose相关损失
        if not args.no_head_pose:    # 需要头部姿势
            if args.rot_repr == 'aa': # 旋转表征，aa，单独计算exp和（R, t, delta）
                head_pose_gt = motion_coef_gt[:, :, 63:]    # 真实值pose
                head_pose_pred = target[:, :, 63:]           # 预测pose
            elif args.rot_repr == 'emo': # 单独计算exp和（R, t, delta）
                head_pose_gt = motion_coef_gt[:, :, 63:70]
                head_pose_pred = target[:, :, 63:70]
            else:
                raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

            # angle, gt_pose和pred_pose之间的损失
            if args.l_head_angle > 0:   # 预测值与真实值的loss         1e-2
                loss_head_angle = criterion_func(head_pose_gt, head_pose_pred, reduction='none')
            if args.l_head_vel > 0:               # 1e-2
                # print("head_pose_gt: ", head_pose_gt.shape, head_pose_pred.shape)
                head_vel_gt = head_pose_gt[:, 1:] - head_pose_gt[:, :-1]         # 真实变化量
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]   # 预测变化量
                loss_head_vel = criterion_func(head_vel_gt, head_vel_pred, reduction='none')
            if args.l_head_smooth > 0:           # 1e-2
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                loss_head_smooth = criterion_func(head_vel_pred[:, 1:], head_vel_pred[:, :-1], reduction='none')

            # 窗口过渡期间头部约束的权重
            if not is_starting_sample and args.l_head_trans > 0:      # 后n_motions部分  1e-2
                # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})  约束预测的先前和当前运动
                # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})  仅约束预测的当前运动
                head_pose_trans = torch.cat([head_pose_gt[:, args.n_prev_motions - 3:args.n_prev_motions],            # [:, 22:25]
                                             head_pose_pred[:, args.n_prev_motions:args.n_prev_motions + 3]], dim=1)  # [:, 25:28]
                # 按dim=0即帧的维度进行拼接。最后得到[B, 6, 特征维度]   22~27
                head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]  # 23-22  ;  24-23 ; 25-24  ;  26-25  ;  27-26  
                head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]    # (24-23)-(23-22) ； (25-24)-(24-23) ； (26-25)-(25-24) ；(27-26)-(26-25)
                # will constrain x_{-2|0} ~ x_{1}  将会约束
                loss_head_trans_vel = criterion_func(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3], reduction='none')
                # will constrain x_{-3|0} ~ x_{2}
                loss_head_trans_accel = criterion_func(head_accel_pred[:, 1:], head_accel_pred[:, :-1], reduction='none')
    else:  # 报错
        raise ValueError(f'Unknown diffusion target: {args.target}')

    # 计算掩码，用于区分是否被截断
    if end_idx is None:  # 没被裁断
        mask = torch.ones((target.shape[0], args.n_motions), dtype=torch.bool, device=target.device)  
        # （B，n_motions=100）   全1张量
    else:   # 被裁断
        mask = torch.arange(args.n_motions, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)
        # [0,1,2,3,...,n_motions-1=99]  扩展到(B,n_motions)  其中大于end_idx即被截断的部分为False，其他为True

    if args.target == 'sample' and not is_starting_sample:      # 采样模式  且是 后n_motions部分
        if args.no_constrain_prev:   # False
            # Warning: this option will be deprecated in the future  警告：此选项将来将被弃用
            mask = torch.cat([torch.zeros_like(mask[:, :args.n_prev_motions]), mask], dim=1)
            # shape：(B,n_prev_motions + n_motions)  其中前半全0张量，后半掩码
        else:  # this
            mask = torch.cat([torch.ones_like(mask[:, :args.n_prev_motions]), mask], dim=1)
            # shape：(B,n_prev_motions + n_motions)  其中前半全1张量，后半掩码

    # mask 用于区分 被裁断的部分。其中被裁断的部分不参与计算，其他帧 计算均值。
    # mask：(B,n_prev_motions + n_motions)
    # n_prev_motions全为1（或0），表示先前帧的部分都保留（用于计算均值）
    loss_noise = loss_noise[mask].mean()
    if loss_exp is not None:
        loss_exp = loss_exp[mask].mean()
    if loss_exp_vel is not None:
        loss_exp_vel = loss_exp_vel[mask[:, 1:]].mean()   # 速度损失少一帧
    if loss_exp_smooth is not None:
        loss_exp_smooth = loss_exp_smooth[mask[:, 2:]].mean()  # 平滑损失少两帧
    if loss_head_angle is not None:
        loss_head_angle = loss_head_angle[mask].mean()
    if loss_head_vel is not None:
        loss_head_vel = loss_head_vel[mask[:, 1:]]
        loss_head_vel = loss_head_vel.mean() if torch.numel(loss_head_vel) > 0 else None  # 如果张量loss_head_vel中的元素个数大于0
    if loss_head_smooth is not None:
        loss_head_smooth = loss_head_smooth[mask[:, 2:]]
        loss_head_smooth = loss_head_smooth.mean() if torch.numel(loss_head_smooth) > 0 else None
    if loss_head_trans_vel is not None:     # mask:  (B,n_prev_motions + n_motions)
        vel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 2]     # (B, 25:27)
        accel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 3]   # (B, 25:28)
        loss_head_trans_vel = loss_head_trans_vel[vel_mask].mean()
        loss_head_trans_accel = loss_head_trans_accel[accel_mask].mean()
        loss_head_trans = loss_head_trans_vel + loss_head_trans_accel    # 速度损失 + 平滑损失

    # 当target='noise'时，只有去噪损失。   “sample”时都有
    return loss_noise, loss_exp, loss_exp_vel, loss_exp_smooth, loss_head_angle, loss_head_vel, loss_head_smooth, loss_head_trans
    # loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht =
    # 去噪损失，表情损失，表情速度损失，表情平滑损失，头部姿势损失，~~速度损失，~~平滑损失，？？？总损失


class Template:
    import pickle
    motions = pickle.load(open('/mnt/disk2/zhouxishi/JoyVASA/src/my_prepare/front_level3_motions_template.pkl', 'rb'))  # 读取运动系数

    # 类方法
    @classmethod
    def get_template(cls):
        return cls.motions

def compute_emotion_loss(args, is_starting_sample, motion_coef_gt, target, prev_motion_coef,emo_index):
    '''情感损失    似然函数
    情感损失：loss_emotion

    计算情感损失：包括

    输入：情感index，预测的exp，真实的exp。。。读取or传入分布值
    '''
    if is_starting_sample:    # 前n_motions部分
        target = target[:, args.n_prev_motions:]        # 100
    else:   # 后n_motions部分
        motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_gt], dim=1)   # 125
        if args.no_constrain_prev:  # False   不约束生成的先前运动
            target = torch.cat([prev_motion_coef, target[:, args.n_prev_motions:]], dim=1)  # (N,125,70)

    exp_gt = motion_coef_gt[:, :, :63]   # 真实表情    （B, n_motions, 63）
    exp_pred = target[:, :, :63]         # 预测的表情  （B, n_motions, 63）

    motions = Template.get_template()  # 调用类方法

    mean_exp_list = [torch.tensor(motions[i.item()]['mean_exp']) for i in emo_index]  # (Batch_size, 63)
    std_exp_list = [torch.tensor(motions[i.item()]['std_exp']) for i in emo_index]    # (Batch_size, 63)

    target_mu = torch.stack(mean_exp_list,dim=0).unsqueeze(1).to(exp_pred.device)            # (Batch_size, 1, 63) 
    target_sigma = torch.stack(std_exp_list,dim=0).unsqueeze(1).to(exp_pred.device)          # (Batch_size, 1, 63)

    gt_dist = Normal(target_mu, target_sigma)    # 正态分布 # (Batch_size, 63)

    likelihood_loss = -gt_dist.log_prob(exp_pred)  # 真实表情的似然函数  (Batch_size, n, 63)

    likelihood_loss_gt = -gt_dist.log_prob(exp_gt)  # gt值的分布。  (Batch_size, n, 63)

    distance_loss = F.mse_loss(likelihood_loss, likelihood_loss_gt)

    # # 创建七种其他情感的均值和方差
    # other_mus = [motion[i]['mean_exp'] for i in range(8) if i != emo_index]  # 七种其他情感的均值
    # other_sigmas = [motion[i]['std_exp'] for i in range(8) if i != emo_index]  # 七种其他情感的标准差

    # # 计算每种其他情感的 KL 散度损失
    # for mu_other, sigma_other in zip(other_mus, other_sigmas):
    #     other_dist = Normal(mu_other, sigma_other)
    #     likelihood_loss += other_dist.log_prob(exp_pred).mean()  # 真实表情的似然函数

    return likelihood_loss.mean(),distance_loss.mean()

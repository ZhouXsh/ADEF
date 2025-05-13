# coding: utf-8

"""
functions for processing and transforming 3D facial keypoints
用于处理和转换3D面部关键点的功能
"""

import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi

# 将预测结果 转换为 角度数  ？？？？？？？
def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5       请注意，平均值已修改为97.5？
        device = pred.device
        idx_tensor = [idx for idx in range(0, 66)]              # 包含 [0, 1, 2, ..., 65] 的列表，表示 66 个类别的索引。
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred, dim=1)         # 对每张图像的66个预测结果进行softmax（即按行进行）
        degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 97.5            # ？？？

        return degree

    return pred   # 其他情况直接返回预测结果

# 计算旋转矩阵
def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree     输入以角度为单位
    输出(batch_size, 3, 3)旋转矩阵
    """
    # transform to radian   转换为弧度
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device

    if pitch.ndim == 1:           
        pitch = pitch.unsqueeze(1)      # (batch_size,)  ->  [bs, 1]       ？？
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix     基于欧拉角的三个旋转矩阵
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)          # [bs, 1] 
    zeros = torch.zeros([bs, 1]).to(device)        # [bs, 1] 
    x, y, z = pitch, yaw, roll                     # [bs, 1] 

    rot_x = torch.cat([          # 绕 x 轴的旋转矩阵（rot_x）
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([           # 绕 y 轴的旋转矩阵（rot_y）
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([          # 绕 z 轴的旋转矩阵（rot_z）
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x           # 依次绕x,y,z旋转      (batch_size, 3, 3)
    return rot.permute(0, 2, 1)  # transpose        ？？？？？

# coding: utf-8

"""
Motion extractor(M), which directly predicts the canonical keypoints, head pose and expression deformation of the input image
运动提取器（M），直接预测输入图像的规范关键点、头部姿势和表情变形

与LivePortrait完全一致
"""

from torch import nn
import torch

from .convnextv2 import convnextv2_tiny
from .util import filter_state_dict

model_dict = {   # 用于存储不同的网络结构。这使得可以灵活地选择不同的主干网络（backbone）。
    'convnextv2_tiny': convnextv2_tiny,     # # 提供不同类型的 backbone，这里是 'convnextv2_tiny'   纯卷积实现，无Transformer
}

# M 运动提取器
class MotionExtractor(nn.Module):
    def __init__(self, **kwargs):
        super(MotionExtractor, self).__init__()

        # default is convnextv2_base
        backbone = kwargs.get('backbone', 'convnextv2_tiny')  # 主干网络    # convnextv2_tiny
        self.detector = model_dict.get(backbone)(**kwargs)   # 初始化

    def load_pretrained(self, init_path: str):   # 加载预训练模型（无用）
        if init_path not in (None, ''):
            state_dict = torch.load(init_path, map_location=lambda storage, loc: storage)['model']
            state_dict = filter_state_dict(state_dict, remove_name='head')  # 过滤掉与模型头部（head）相关的参数。通常模型的头部包含与分类、回归等任务相关的参数，在迁移学习中常常需要去掉这些。
            ret = self.detector.load_state_dict(state_dict, strict=False)   # 将过滤后的参数加载到 detector（即 backbone 网络）中，strict=False 表示可以加载部分匹配的参数。
            print(f'Load pretrained model from {init_path}, ret: {ret}')

    def forward(self, x):       # x: (B,3,H=256,W=256)
        out = self.detector(x)
        return out
        # out = ret_dct = {
        #         'pitch': pitch,        # (B, C=66)
        #         'yaw': yaw,            # (B, C=66)
        #         'roll': roll,          # (B, C=66)
        #         't': t,                # (B, C=3)
        #         'exp': exp,            # (B, C=3 * 21)
        #         'scale': scale,        # (B, C=1)
        #         'kp': kp,   # canonical keypoint   (B, C=3 * 21)
        #         }

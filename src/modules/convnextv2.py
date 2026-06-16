# coding: utf-8

"""
This moudle is adapted to the ConvNeXtV2 version for the extraction of implicit keypoints, poses, and expression deformation.

此模型适用于ConvNeXtV2版本，用于提取隐式关键点、姿势和表情变形。    

与LivePortrait完全一致
"""

import torch
import torch.nn as nn
from .util import LayerNorm, DropPath, trunc_normal_, GRN

__all__ = ['convnextv2_tiny']

# ConvNeXtV2的块
class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.                     输入通道数
        drop_path (float): Stochastic depth rate. Default: 0.0   随机深度速率（浮点数）
    """

    def __init__(self, dim,          #  96 * 3块, 192 * 3块, 384 * 9块, 768 * 3块
                 drop_path=0.):      #  0 * 18
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv  深度卷积：每个输入 通道 上独立进行卷积。H,W不变
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers   逐点卷积：使用全连接层实现
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)          # GRN（全局响应归一化）层
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()    # 随机丢弃部分路径; nn.Identity()，即不做任何处理

    def forward(self, x):     #  (B, C = 96 -> 192 -> 384 -> 768, H=256, W=256)
        input = x
        x = self.dwconv(x)         # 深度可分卷积：每个通道内 独立进行卷积
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)           # （通道）层归一化
        x = self.pwconv1(x)        # (N, H, W, C) -> (N, H, W, 4 * C)
        x = self.act(x)            # GELU 激活函数
        x = self.grn(x)            # GRN（全局响应归一化）层
        x = self.pwconv2(x)        # (N, H, W, 4 * C) -> (N, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x      # shape不变   #  (B, C = 96 -> 192 -> 384 -> 768, H=256, W=256)


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000                   分类头的类数。
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.    分类器权重和偏差的初始缩放值。默认值：1
    """

    def __init__(
        self,
        in_chans=3,                  # 输入图像通道的数量。默认值：3
        depths=[3, 3, 9, 3],         # 每个阶段的块数
        dims=[96, 192, 384, 768],    # 每个阶段的特征维度
        drop_path_rate=0.,           # 随机深度速率
        **kwargs
    ):
        super().__init__()
        self.depths = depths

        # 茎 + 3个中间下采样conv层
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers  
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),          # 3 -> 96    size/4
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")      # 层归一化
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(        # 层归一化 + 下采样
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),       # 96 -> 192 -> 384 -> 768  size/2
            )
            self.downsample_layers.append(downsample_layer)
        
        # 4个特征分辨率阶段，每个阶段由多个残差块组成
        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks   
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]   # 路径丢弃率（dp_rates），为0说明网络中的所有路径都会被保留下来。[0,0,0,...,0]共18个
        cur = 0
        for i in range(4):  # 0~3
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]  # cur = 0,3,6,15  dp_rates[0~2,3~5,6~14,15~17]
            )
            self.stages.append(stage)
            # stage0 Block(dim=96, drop_path=dp_rates[0~2])       共3个block
            # stage1 Block(dim=192, drop_path=dp_rates[3~5])      共3个block
            # stage2 Block(dim=384, drop_path=dp_rates[6~14])     共9个block
            # stage3 Block(dim=768, drop_path=dp_rates[15~17])    共3个block
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer  归一化层

        # NOTE: the output semantic items 输出语义项
        num_bins = kwargs.get('num_bins', 66)
        num_kp = kwargs.get('num_kp', 24)      # the number of implicit keypoints              21
        self.fc_kp = nn.Linear(dims[-1], 3 * num_kp)  # 规范关键点 的坐标xyz       768 -> 3 * 21

        # print('dims[-1]: ', dims[-1])
        self.fc_scale = nn.Linear(dims[-1], 1)          # scale                     768 -> 1
        self.fc_pitch = nn.Linear(dims[-1], num_bins)   # pitch bins                768 -> 66
        self.fc_yaw = nn.Linear(dims[-1], num_bins)     # yaw bins                  768 -> 66
        self.fc_roll = nn.Linear(dims[-1], num_bins)    # roll bins                 768 -> 66
        self.fc_t = nn.Linear(dims[-1], 3)              # translation               768 -> 3
        self.fc_exp = nn.Linear(dims[-1], 3 * num_kp)   # expression / delta        768 -> 3 * 21

    def _init_weights(self, m):          # 无用
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):             # x: (B,3,H=256,W=256)
        for i in range(4):
            x = self.downsample_layers[i](x)        # 茎和3个中间下采样conv层
            # (B,3,256,256) -> (B,96,64,64) -> (B,192,32,32) -> (B,384,16,16) -> (B,768,8,8)
            x = self.stages[i](x)                   # 4个特征分辨率阶段           shape不变
        return self.norm(x.mean([-2, -1]))  # global average pooling全局平均池, (N, C, H, W) -> (N, C) :  (B, C=768, H=8, W=8) -> (B, C=768)

    def forward(self, x):               # x: (B,3,H=256,W=256)
        x = self.forward_features(x)   # (B, 3 -> 96 -> 192 -> 384 -> 768, H=256, W=256) -> (B=1, C=768)

        # implicit keypoints 规范关键点canonical keypoint坐标xyz三维
        kp = self.fc_kp(x)         # (B, C=768) -> (B, C=3 * 21=63)

        # pose and expression deformation
        pitch = self.fc_pitch(x)    # (B, C=768) -> (B, C=66)
        yaw = self.fc_yaw(x)        # (B, C=768) -> (B, C=66)
        roll = self.fc_roll(x)      # (B, C=768) -> (B, C=66)
        # 尽管 fc_pitch、fc_yaw 和 fc_roll 这三个全连接层的结构完全相同（即它们的输入和输出维度相同），它们的 权重矩阵是不同的。具体来说，它们会有各自独立的 权重矩阵 和 偏置向量，因为它们是独立训练的，并且对应不同的旋转分量。
        t = self.fc_t(x)            # (B, C=768) -> (B, C=3)
        exp = self.fc_exp(x)        # (B, C=768) -> (B, C=3 * 21)
        scale = self.fc_scale(x)    # (B, C=768) -> (B, C=1)

        ret_dct = {
            'pitch': pitch,        # (B, C=66)
            'yaw': yaw,            # (B, C=66)
            'roll': roll,          # (B, C=66)
            't': t,                # (B, C=3)
            'exp': exp,            # (B, C=3 * 21)
            'scale': scale,        # (B, C=1)
            'kp': kp,   # canonical keypoint   (B, C=3 * 21)
        }

        return ret_dct


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

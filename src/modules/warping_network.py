# coding: utf-8

"""
Warping field estimator(W) defined in the paper, which generates a warping field using the implicit
keypoint representations x_s and x_d, and employs this flow field to warp the source feature volume f_s.

本文定义了扭曲场估计器（W），它使用隐式关键点表示x_s和x-d生成扭曲场，并利用该流场来扭曲源特征体积f_s。

与LivePortrait完全一致
"""

from torch import nn
import torch.nn.functional as F
from .util import SameBlock2d
from .dense_motion import DenseMotionNetwork


class WarpingNetwork(nn.Module):
    def __init__(
        self,
        num_kp,                # 隐式关键点的数量      21
        block_expansion,       # 卷积层扩展因子，用于控制每个卷积块的输出通道数     64
        max_features,          # 最大特征数，用于确定网络中最大通道数的限制。        512
        num_down_blocks,       # 下采样块的数量                                   2
        reshape_channel,       # 调整特征通道数的参数                              32
        estimate_occlusion_map=False,    # 是否生成遮挡图，默认是 False    配置yaml内是True
        dense_motion_params=None,
        **kwargs
    ):
        super(WarpingNetwork, self).__init__()

        self.upscale = kwargs.get('upscale', 1)
        self.flag_use_occlusion_map = kwargs.get('flag_use_occlusion_map', True)

        # 根据稀疏的表示，得到密集的运动
        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp,         # 21
                feature_channel=reshape_channel,            # 32
                estimate_occlusion_map=estimate_occlusion_map,
                **dense_motion_params
                # dense_motion_params:        # 密集运动参数
                #     block_expansion: 32
                #     max_features: 1024
                #     num_blocks: 5
                #     reshape_depth: 16
                #     compress: 4
            )
        else:
            self.dense_motion_network = None

        self.third = SameBlock2d(max_features, block_expansion * (2 ** num_down_blocks), kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=block_expansion * (2 ** num_down_blocks), out_channels=block_expansion * (2 ** num_down_blocks), kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map

    # 根据给定的扭曲场（deformation）对输入特征进行空间变换
    def deform_input(self, inp, deformation):
        return F.grid_sample(inp, deformation, align_corners=False)  # grid_sample 根据提供的流场对输入图像进行采样，从而实现图像的空间扭曲。

    def forward(self, feature_3d, kp_driving, kp_source):
        # feature_3d: Bx32x16x64x64    外观特征
        # kp_driving: B*N*3            驱动隐式关键点
        # kp_source:  B*N*3            源隐式关键点
        # 计算密集的运动系数
        if self.dense_motion_network is not None:
            # Feature warper, Transforming feature representation according to deformation and occlusion
            # 特征扭曲器，根据变形和遮挡 转换 特征表示
            # 1~7 计算相关参数
            dense_motion = self.dense_motion_network(
                feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source
            )
            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']  # Bx1x64x64
            else:
                occlusion_map = None

            deformation = dense_motion['deformation']        # Bx16x64x64x3       组合流场w
            # 6. 使用组合流场 扭曲 3D源特征fs，得到 扭曲的特征 Warped feature w(fs)
            out = self.deform_input(feature_3d, deformation)  # Bx32x16x64x64  -> Bx32x16x64x64

            bs, c, d, h, w = out.shape  # Bx32x16x64x64
            out = out.view(bs, c * d, h, w)  # Bx32x16x64x64 -> Bx512x64x64
            out = self.third(out)   # Bx512x64x64 -> Bx256x64x64
            out = self.fourth(out)  # Bx256x64x64 -> Bx256x64x64     特征重组？？

            if self.flag_use_occlusion_map and (occlusion_map is not None):   # 使用遮挡贴图
                out = out * occlusion_map      # Bx256x64x64 * Bx1x64x64 = Bx256x64x64

        ret_dct = {
            'occlusion_map': occlusion_map,      # Bx1x64x64        遮挡图
            'deformation': deformation,          # Bx16x64x64x3     组合流场
            'out': out,                          # Bx256x64x64      被扭曲的源外观特征
        }

        return ret_dct

# coding: utf-8

"""
The module that predicting a dense motion from sparse motion representation given by kp_source and kp_driving
从kp_source和kp_driving给出的 稀疏运动表示 预测密集运动 的模块

与LivePortrait完全一致
"""

from torch import nn
import torch.nn.functional as F
import torch
from .util import Hourglass, make_coordinate_grid, kp2gaussian


# 密集运动 网络
class DenseMotionNetwork(nn.Module):
    def __init__(self, block_expansion,     # 32
                 num_blocks,                # 5
                 max_features,              # 1024
                 num_kp,                    # 21
                 feature_channel,           # 32
                 reshape_depth,             # 16
                 compress,                  # 4   压缩外观特征的通道数量
                 estimate_occlusion_map=True):
        super(DenseMotionNetwork, self).__init__()
        self.hourglass = Hourglass(block_expansion=block_expansion, in_features=(num_kp+1)*(compress+1), max_features=max_features, num_blocks=num_blocks)  # ~60+G

        self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)  # 65G! NOTE: computation cost is large
        self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)  # 0.8G
        self.norm = nn.BatchNorm3d(compress, affine=True)
        self.num_kp = num_kp        # 隐式关键点个数  21
        self.flag_estimate_occlusion_map = estimate_occlusion_map

        if self.flag_estimate_occlusion_map:
            # 142 * 16 -> 1
            self.occlusion = nn.Conv2d(self.hourglass.out_filters*reshape_depth, 1, kernel_size=7, padding=3)
        else:
            self.occlusion = None

    # √ 创建稀疏运动（根据 隐式关键点 估计的K个流: w1~wk）
    # 外观特征主要提供空间形状；具体计算只与隐式关键点有关
    def create_sparse_motions(self, feature, kp_driving, kp_source):        # (bs, 21, 3)
        bs, _, d, h, w = feature.shape                                      # (bs, 4, 16, 64, 64)
        identity_grid = make_coordinate_grid((d, h, w), ref=kp_source)      # (16, 64, 64, 3)
        identity_grid = identity_grid.view(1, 1, d, h, w, 3)                # (1, 1, d=16, h=64, w=64, 3)
        # 通过广播机制对齐后，(1, 1, 16, 64, 64, 3) - (bs, 21, 1, 1, 1, 3) = (1,21,16,64,64,3)     需要bs==1才能广播？？？
        coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)  # (bs, num_kp, d, h, w, 3)

        k = coordinate_grid.shape[1]     # k=self.num_kp=21

        # NOTE: there lacks an one-order flow
        driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 1, 3)    # (bs, num_kp, d, h, w, 3)

        # adding background feature  添加背景特征
        identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)                # (bs, 1, d=16, h=64, w=64, 3)
        sparse_motions = torch.cat([identity_grid, driving_to_source], dim=1)  # (bs, 1+num_kp, d, h, w, 3)
        return sparse_motions       # (bs, 1+num_kp=22, d=16, h=64, w=64, 3)   稀疏运动

    # √ 创建变形特征： 用k个流分别扭曲外观特征，得到 变形的特征：  w1(fs) ~ wk(fs)
    # 该函数的作用是基于给定的稀疏运动场（sparse_motions），对输入的特征图（feature）进行空间变换，并生成变形后的特征图。
    # 通过 grid_sample 函数实现变换，其中 sparse_motions 表示每个位置的三维运动（dx, dy, dz）。
    def create_deformed_feature(self, feature, sparse_motions):
        # feature：        (bs, c=4, d=16, h=64, w=64)
        # sparse_motions： (bs, 1+num_kp=22, d=16, h=64, w=64, 3)
        bs, _, d, h, w = feature.shape
        feature_repeat = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp+1, 1, 1, 1, 1, 1)      # (bs, num_kp+1, 1, c, d, h, w)
        feature_repeat = feature_repeat.view(bs * (self.num_kp+1), -1, d, h, w)                         # (bs*(num_kp+1), c, d, h, w)
        sparse_motions = sparse_motions.view((bs * (self.num_kp+1), d, h, w, -1))                       # (bs*(num_kp+1), d, h, w, 3)
        # F.grid_sample 是 PyTorch 中的一个函数，用于根据给定的网格坐标（在这里是 sparse_motions）对输入张量（在这里是 feature_repeat）进行空间变换。
        # 变换只会影响到每个位置的像素值，在这个操作中，并不会改变空间维度或通道数量。 因此sparse_deformed的形状与feature_repeat一样
        sparse_deformed = F.grid_sample(feature_repeat, sparse_motions, align_corners=False)            # (bs*(num_kp+1), c, d, h, w)
        sparse_deformed = sparse_deformed.view((bs, self.num_kp+1, -1, d, h, w))                        # (bs, num_kp+1, c, d, h, w)

        return sparse_deformed   # (bs, 1+num_kp, c=4, d=16, h=64, w=64)   扭曲后的图像特征

    # 创建热图表示
    def create_heatmap_representations(self, feature, kp_driving, kp_source):
        # feature：deformed_feature：(bs, 1+num_kp, c=4, d=16, h=64, w=64)
        # kp_driving：kp_source：(bs, num_kp=21, 3)
        spatial_size = feature.shape[3:]  # (d=16, h=64, w=64)
        # 计算标准网格空间坐标 相对于 关键点（均值） 的 （未归一化）高斯概率密度。 
        # 高斯热图（高斯分布）。表示关键点在空间中的影响范围。
        gaussian_driving = kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)  # (bs, num_kp, d, h, w)
        gaussian_source = kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)    # (bs, num_kp, d, h, w)

        # 驱动和源之间的热图差异。。通过计算两者的差异，网络能够捕捉到源和驱动关键点之间的空间变换或差异。
        heatmap = gaussian_driving - gaussian_source      # (bs, num_kp, d, h, w)

        # adding background feature  添加背景特征
        # 创建与 heatmap 相同深度、宽度和高度的全零张量
        zeros = torch.zeros(heatmap.shape[0], 1, spatial_size[0], spatial_size[1], spatial_size[2]).type(heatmap.dtype).to(heatmap.device) # (bs, 1, d, h, w)
        # 看zeros, heatmap  它被添加到热图的最前面，表示没有关键点影响的区域。
        heatmap = torch.cat([zeros, heatmap], dim=1)           # (bs, 1+ num_kp, d, h, w)
        heatmap = heatmap.unsqueeze(2)         # (bs, 1+num_kp, 1, d, h, w)
        return heatmap                         # (bs, 1+num_kp, 1, d, h, w)

    def forward(self, feature, kp_driving, kp_source):         # （B, 21, 3）
        bs, _, d, h, w = feature.shape    # (bs, 32, 16, 64, 64)

        feature = self.compress(feature)  # (bs, 4, 16, 64, 64)
        feature = self.norm(feature)      # (bs, 4, 16, 64, 64)
        feature = F.relu(feature)         # (bs, 4, 16, 64, 64)

        out_dict = dict()

        # 1. sparse_motion： 根据 隐式关键点 估计的K个流: w1~wk
        sparse_motion = self.create_sparse_motions(feature, kp_driving, kp_source)  # (bs, 1+num_kp=22, d=16, h=64, w=64, 3)
        # 2. 用k个流分别扭曲外观特征，得到 变形的特征：  w1(fs) ~ wk(fs)
        deformed_feature = self.create_deformed_feature(feature, sparse_motion)     # (bs, 1+num_kp=22, c=4, d=16, h=64, w=64)

        # 3. 创建热图 ？？？
        heatmap = self.create_heatmap_representations(deformed_feature, kp_driving, kp_source)  # (bs, 1+num_kp=22, 1, d=16, h=64, w=64)

        # 3.5 构建输入 ？？？   input为Motion field estimator的输入
        input = torch.cat([heatmap, deformed_feature], dim=2)  # (bs, 1+num_kp=22, c=4+1=5, d=16, h=64, w=64)
        input = input.view(bs, -1, d, h, w)  # (bs, (1+num_kp)*c=110, d=16, h=64, w=64)  ？？？？ maybe：(bs, (1+num_kp)*c=110, d=16, h=64, w=64)

        # 4. prediction为Motion field estimator的Unet部分输出结果
        prediction = self.hourglass(input)  # 预测密集运动    # prediction: (bs, 32+110=142 , d=16, h=64, w=64)

        # 4.5a. 计算流组合掩码：prediction通过卷积和softmax得到 流组合掩码 m
        # self.mask = nn.Conv3d(142, 22, kernel_size=7, padding=3)
        mask = self.mask(prediction)   # (bs, 32+110=142, d=16, h=64, w=64) -> (bs, 1+num_kp=22, d=16, h=64, w=64)
        mask = F.softmax(mask, dim=1)  # (bs, 1+num_kp=22, d=16, h=64, w=64)     每个关键点的概率？？
        out_dict['mask'] = mask                   # Paper中的 Flow composition mask 流组合掩码 - m

        mask = mask.unsqueeze(2)                                   # (bs=1, num_kp+1=22, 1, d=16, h=64, w=64)
        sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)    # (bs=1, 1+num_kp=22, d=16, h=64, w=64, 3) -> (bs, num_kp+1, 3, d, h, w)
        
        # 5. 流组合掩码m-mask 与 估计的k个流w1~k 线性组合产生 组合流场w
        deformation = (sparse_motion * mask).sum(dim=1)            # (bs, 3, d, h, w)  mask take effect in this place 对应相乘相加（即线性组合）
        deformation = deformation.permute(0, 2, 3, 4, 1)           # (bs=1, d=16, h=64, w=64, 3)

        out_dict['deformation'] = deformation     # Paper中的 Composited flow field 组合流场 - w

        # 4.5b. 估计遮挡图  ： prediction通过reshape和 卷积 得到 遮挡图o
        if self.flag_estimate_occlusion_map:
            bs, _, d, h, w = prediction.shape                         # (bs, 32+110=142 , d=16, h=64, w=64)
            prediction_reshape = prediction.view(bs, -1, h, w)        # (bs, 32+110=142 * d=16, h=64, w=64)
            # 生成遮挡图，表示哪些区域被遮挡。 self.occlusion = nn.Conv2d( 142 * 16, 1, kernel_size=7, padding=3)
            occlusion_map = torch.sigmoid(self.occlusion(prediction_reshape))  # Bx1x64x64
            out_dict['occlusion_map'] = occlusion_map    # 遮挡图？？

        return out_dict   # 包含 流组合掩码m，组合流场w，遮挡图

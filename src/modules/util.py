# coding: utf-8

"""
This file defines various neural network modules and utility functions, including convolutional and residual blocks,
normalizations, and functions for spatial transformation and tensor manipulation.

该文件定义了各种神经网络模块和实用函数，包括卷积和残差块、归一化以及空间变换和张量操作函数。

与LivePortrait完全一致
"""

from torch import nn
import torch.nn.functional as F
import torch
import torch.nn.utils.spectral_norm as spectral_norm
import math
import warnings
import collections.abc
from itertools import repeat

# 对于每个关键点，它生成一个类似高斯的二维（或三维）分布，并将其映射到一个空间中，
# 最终输出的张量表示每个关键点的高斯分布在三维空间中（深度、高度、宽度）的值。
def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation 
    将关键点转换为类高斯表示
    kp:           (bs, num_kp=21, 3)
    spatial_size: (d=16, h=64, w=64)
    kp_variance = 0.01     高斯分布的方差
    输出：未归一化的概率密度
    """
    mean = kp         #  (bs, num_kp=21, 3)

    # 创建标准坐标网格并改变其形状
    coordinate_grid = make_coordinate_grid(spatial_size, mean)            # 标准坐标网格 (16, 64, 64, 3)
    number_of_leading_dimensions = len(mean.shape) - 1                    # 2
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape   # (1, 1, 16, 64, 64, 3)
    coordinate_grid = coordinate_grid.view(*shape)                        # (1, 1, 16, 64, 64, 3)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 1)    # (bs, num_kp, 1, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats)  # (1, 1, 16, 64, 64, 3) -> (bs, num_kp, 16, 64, 64, 3)

    # Preprocess kp shape 改变kp的形状（不改变其值）
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 1, 3)   # (bs, num_kp, 1, 1, 1, 3)
    mean = mean.view(*shape)      # (bs, num_kp=21, 3) -> (bs, num_kp, 1, 1, 1, 3)
 
    mean_sub = (coordinate_grid - mean)      # 广播，最后得到(bs, num_kp, 16, 64, 64, 3)

    # 三维高斯分布公式（未归一化）。其中关键点作为均值。
    # 这个公式描述了每个空间位置（在图像或体素网格上的每个点）相对于 关键点 的高斯权重
    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)    # (bs, num_kp, d, h, w)

    return out   # (bs, num_kp=21, d=16, h=64, w=64)

# 制造坐标网格
# 只与spatial_size有关，ref负责提供设备和类型。
# 相当于给定spatial_size，该网格是固定的
# meshed的每个位置 代表其所在空间的坐标值   如左前上角(-1,-1,-1)  右后下角(1,1,1)  共(16, 64, 64)个网格位置
def make_coordinate_grid(spatial_size, ref, **kwargs):     # ref：(bs, 21, 3)
    d, h, w = spatial_size                  # （16,64,64）
    x = torch.arange(w).type(ref.dtype).to(ref.device)  # [0,1,2,...,63]
    y = torch.arange(h).type(ref.dtype).to(ref.device)  # [0,1,2,...,63]
    z = torch.arange(d).type(ref.dtype).to(ref.device)  # [0,1,2,...,15]

    # NOTE: must be right-down-in   右-下-内
    # 坐标被标准化到：[-1, 1]
    x = (2 * (x / (w - 1)) - 1)  # the x axis faces to the right    # [-1,....,1]
    y = (2 * (y / (h - 1)) - 1)  # the y axis faces to the bottom   # [-1,....,1]  64
    z = (2 * (z / (d - 1)) - 1)  # the z axis faces to the inner    # [-1,....,1]  16

    yy = y.view(1, -1, 1).repeat(d, 1, w)   # (64,) -> (1, 64, 1) -> (16, 64, 64)
    xx = x.view(1, 1, -1).repeat(d, h, 1)   # (64,) -> (1, 1, 64) -> (16, 64, 64)
    zz = z.view(-1, 1, 1).repeat(1, h, w)   # (16,) -> (16, 1, 1) -> (16, 64, 64)

    # (16, 64, 64) --unsqueeze--> (16, 64, 64, 1) --cat--> (16, 64, 64, 3)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)

    return meshed       # (16, 64, 64, 3)


class ConvT2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(ConvT2d, self).__init__()

        self.convT = nn.ConvTranspose2d(in_features, out_features, kernel_size=kernel_size, stride=stride,
                                        padding=padding, output_padding=output_padding)
        self.norm = nn.InstanceNorm2d(out_features)

    def forward(self, x):
        out = self.convT(x)
        out = self.norm(out)
        out = F.leaky_relu(out)
        return out

# 残差块，保持空间分辨率。
class ResBlock3d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size, padding=padding)
        self.norm1 = nn.BatchNorm3d(in_features, affine=True)
        self.norm2 = nn.BatchNorm3d(in_features, affine=True)

    def forward(self, x):  # (归一化，激活，卷积）*2 , 残差连接
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out


class UpBlock3d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock3d, self).__init__()

        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=(1, 2, 2))
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

# # 编码器中使用的下采样块。
class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out


# 编码器中使用的下采样块。
class DownBlock3d(nn.Module):
    """
    Downsampling block for use in encoder.     编码器中使用的下采样块。
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock3d, self).__init__()
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                                padding=padding, groups=groups, stride=(1, 2, 2))
        '''
        self.conv = nn.Conv3d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = nn.BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)      # 卷积
        out = self.norm(out)    # 归一化
        out = F.relu(out)       # 激活
        out = self.pool(out)    # 池化
        return out

# 简单的块，保持空间分辨率。
class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.   简单的块，保持空间分辨率。
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1, lrelu=False):
                           # 3,        64,          1,       (3, 3),         (1, 1),     False
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = nn.BatchNorm2d(out_features, affine=True)        # 组归一化
        if lrelu:
            self.ac = nn.LeakyReLU()
        else:
            self.ac = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)       # 卷积
        out = self.norm(out)     # 归一化
        out = self.ac(out)       # 激活
        return out

# 沙漏的编码器（下采样）
# 110 -> 64 -> 128 -> 256 -> 512 -> 1024 
class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):          # 32 110 5 1024
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks): # 0~4
            down_blocks.append(DownBlock3d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)), 
                                           min(max_features, block_expansion * (2 ** (i + 1))), 
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):                            # x：(bs, (1+num_kp)*c=110, d=16, h=64, w=64)
        outs = [x]      # 保存每一次下采样后的结果
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))  # (1,110,16,64,64) -> (1,64,16,32,32) -> (1,128,16,16,16) -> (1,256,16,8,8) -> (1,512,16,4,4) -> (1,1024,16,2,2)    
        return outs  # 原x 一下 二下 三下 四下 五下

# 沙漏的解码器（上采样）
# 1024 -> 512 (+ 512) -> 256 (+ 256) -> 128 (+ 128) -> 64 (+ 64) -> 32(+ 110) == 142
class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):          # 32 110 5 1024
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:      # 逆序遍历 [4, 3, 2, 1, 0]
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock3d(in_filters, out_filters, kernel_size=3, padding=1))

            # 1024-->512
            # 1024-->256
            # 512-->128
            # 256-->64
            # 128-->32

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features     # 32 + 110 = 142  最后一次上采样后，拼接后的层数

        self.conv = nn.Conv3d(in_channels=self.out_filters, out_channels=self.out_filters, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

    def forward(self, x):# x: (1,110,16,64,64) -> (1,64,16,32,32) -> (1,128,16,16,16) -> (1,256,16,8,8) -> (1,512,16,4,4) -> (1,1024,16,2,2) 
        out = x.pop()        # (1,1024,16,2,2)
        for up_block in self.up_blocks:
            out = up_block(out)           # (1,1024,16,2,2) - (1,512,16,4,4) - (1,256,16,8,8) - (1,128,16,16,16) - (1,64,16,32,32) - (1,32,16,64,64)
            skip = x.pop()                #                +  (1,512,16,4,4)   (1,256,16,8,8)   (1,128,16,16,16)   (1,64,16,32,32)   (1,110,16,64,64)
            out = torch.cat([out, skip], dim=1)   #          (1,1024,16,4,4)   (1,512,16,8,8)   (1,256,16,16,16)   (1,128,16,32,32)  (1,142,16,64,64)
        # 总的上采样流程：
        # 1024 -> 512 (+ 512) -> 256 (+ 256) -> 128 (+ 128) -> 64 (+ 64) -> 32(+ 110) == 142
        out = self.conv(out)       # 无损卷积   142 -> 142
        out = self.norm(out)       # 批次归一化
        out = F.relu(out)          # 激活
        return out   # (bs=1, 32+110=142 , d=16, h=64, w=64)

# 沙漏架构（Encoder+Decoder）
class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, 
                 block_expansion,         # 32
                 in_features,             # (21+1)*(4+1) = 110
                 num_blocks=3,            # 5
                 max_features=256):       # 1024
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters     # 32 + 110 = 142  最后一次上采样后，拼接后的层数

    def forward(self, x):  # x：(bs, (1+num_kp)*c=110, d=16, h=64, w=64)
        return self.decoder(self.encoder(x))    # (bs, 32+110=142 , d=16, h=64, w=64)

# 本质：归一化函数
# Spatially Adaptive Denormalization 空间自适应非规范化
# SPADE 是一种条件归一化方法，通过输入的条件图（通常是标签图或分割图）来调整图像的归一化参数，从而实现更细粒度的控制。
# 通过卷积网络生成的 gamma 和 beta 调整了标准化后的图像特征，使得最终的输出图像能够根据条件图的不同产生变化。
class SPADE(nn.Module):
    def __init__(self, 
                 norm_nc,         # norm_nc 是输入的特征图通道数     # 512 
                 label_nc):       #  标签图的通道数                   256
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)    # 将输入的特征图进行实例归一化; affine=False 表示没有学习的尺度（scale）和偏移（shift）参数
        nhidden = 128     # 中间层的通道数，在这个网络中使用了128作为隐藏层的维度。

        self.mlp_shared = nn.Sequential(   # 多层感知机（MLP）共享网络,由一个卷积层和 ReLU 激活函数组成
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU())
        # 两个卷积层分别用于计算归一化后的缩放因子（gamma）和偏移因子（beta）
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):    # Bx512x64x64 , Bx256x64x64
        normalized = self.param_free_norm(x)           # 得到实例归一化后的特征图  (B, label_nc=256, H=64, W=64)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')   # 最近邻插值  (B, label_nc=256, H=64, W=64)
        actv = self.mlp_shared(segmap)         # (B, label_nc=256, H=64, W=64) -> (B, nhidden=128, H=64, W=64)
        gamma = self.mlp_gamma(actv)      # (B, nhidden=128, H=64, W=64) -> (B, norm_nc=512, H=64, W=64)  缩放因子
        beta = self.mlp_beta(actv)        # (B, nhidden=128, H=64, W=64) -> (B, norm_nc=512, H=64, W=64)  偏移因子
        out = normalized * (1 + gamma) + beta
        return out        # (B, norm_nc=512, H=64, W=64)

# SPADEResnetBlock 是一个典型的残差模块，结合了 SPADE 归一化层和卷积层。
# 它的设计考虑了空间自适应的条件归一化，通过将条件输入（如标签图或分割图）与图像特征结合，从而生成条件图像。
# 通过残差连接和可选择的学习 shortcut，网络能够更好地进行训练并保持稳定。
class SPADEResnetBlock(nn.Module):
    def __init__(self, 
                 fin,            # 输入通道数   2*256 = 512
                 fout,           # 输出通道数   2*256 = 512
                 norm_G,         # 'spadespectralinstance'
                 label_nc,       # 即label_num_channels 条件输入（标签图或分割图）的通道数      256
                 use_se=False,   # 是否使用 Squeeze-and-Excitation（SE）模块，用于通道注意力（默认为 False）
                 dilation=1):    # 卷积的膨胀率（默认为1）
        super().__init__()
        # Attributes             属性
        self.learned_shortcut = (fin != fout)   # False
        fmiddle = min(fin, fout)           # 512
        self.use_se = use_se
        # create conv layers     创建卷积层
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:  # 调整输入特征图与输出特征图的通道数，使得它们相同
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)
        # apply spectral norm if specified       如果指定，则应用光谱范数
        if 'spectral' in norm_G:        # √
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        # define normalization layers           定义归一化层
        self.norm_0 = SPADE(fin, label_nc)           # 512, 256
        self.norm_1 = SPADE(fmiddle, label_nc)       # 512, 256
        if self.learned_shortcut:
            self.norm_s = SPADE(fin, label_nc)       # 512, 256

    def forward(self, x, seg1):   # Bx512x64x64 , Bx256x64x64
        x_s = self.shortcut(x, seg1)    # (B,fin,H,W) -> (B,fout,H,W)
        # norm_0即SPADE过后，通道数量不变
        dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))    # (B,fin,H,W) -> (B,fmiddle,H,W)
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))   # (B,fmiddle,H,W) -> (B,fout,H,W)
        out = x_s + dx   # 残差连接
        return out       # (B,fout,H,W)   即Bx512x64x64

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))   # (B,fin,H,W) -> (B,fout,H,W)
        else:           # fin=fout
            x_s = x     # (B,fin=fout,H,W)
        return x_s

    def actvn(self, x):  # 使用 Leaky ReLU 激活函数，其中负半轴的斜率为 0.2，即 F.leaky_relu(x, 2e-1)
        return F.leaky_relu(x, 2e-1)


def filter_state_dict(state_dict, remove_name='fc'):
    new_state_dict = {}
    for key in state_dict:
        if remove_name in key:
            continue
        new_state_dict[key] = state_dict[key]
    return new_state_dict

# GRN（全局响应归一化）层
class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))     # 可学习参数γ    (1,1,1,dim)，用于控制归一化强度
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))      # 可学习参数β    (1,1,1,dim)，用于调整归一化偏移量

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)    # 计算全局响应 G(x)       计算了特征图在 (H, W) 维度上的 L2 范数
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)     # 计算归一化因子 N(x)      1e-6 是一个小的数值，用于防止除零错误
        return self.gamma * (x * Nx) + self.beta + x         # 计算最终归一化输出

# LayerNorm（层归一化）
# 每个批次内 每个通道内，对于H和W  进行的  层  归一化
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    LayerNorm支持两种数据格式：channels_last（默认）或channels_first。
    输入中尺寸的排序。channels_last对应于具有形状（batch_size，height，width，channels）的输入，而channels_first对应于具有形式（batch_size，channels，height，width）的输入。
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):  # normalized_shape：通道数
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))         # 可学习缩放参数 γ
        self.bias = nn.Parameter(torch.zeros(normalized_shape))          # 可学习偏移参数 β
        self.eps = eps                                                   # 避免除零的稳定项
        self.data_format = data_format                                   # 指定数据格式（channels_last / channels_first）
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )                    # 归一化的维度

    def forward(self, x):
        if self.data_format == "channels_last":         # (B,H,W,C)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":      # (B,C,H,W)
            # 计算均值 u 和方差 s
            u = x.mean(1, keepdim=True)               # 计算 x 在通道维度 (dim=1) 上的均值   数据量为H*W
            s = (x - u).pow(2).mean(1, keepdim=True)  # 计算 x 在通道维度 (dim=1) 上的方差   数据量为H*W
            x = (x - u) / torch.sqrt(s + self.eps)    # 归一化
            # 进行缩放和偏移
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x          # (B,H,W,C) 或 (B,C,H,W)

# 无用
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def drop_path(x, drop_prob=0., training=False, scale_by_keep=True):
    """ Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
        在残差块体的主路径中应用时，每个样本的下降随机深度的路径
        “Drop paths”（随机深度）每个样本（当应用于残差块的主路径时）。
        这与我为 EfficientNet 等网络创建的 DropConnect 实现相同。
        然而，原始名称具有误导性，因为“Drop Connect”是另一种形式的 dropout，来源于不同的论文…
        请参见讨论：https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956。
        因此，我选择将层和参数名称更改为“drop path”，而不是混用“DropConnect”作为层名，并使用“survival rate”作为参数。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """ Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
        每个样本的下降路径（随机深度）（当应用于残差块的主路径时）。
    """

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

# 无用
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

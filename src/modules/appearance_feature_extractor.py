# coding: utf-8

"""
Appearance extractor(F) defined in paper, which maps the source image s to a 3D appearance feature volume.
"""

import torch
from torch import nn
from .util import SameBlock2d, DownBlock2d, ResBlock3d  # 尺寸不变的卷积；下采样块；残差块。


class AppearanceFeatureExtractor(nn.Module):

    def __init__(self, image_channel, block_expansion, num_down_blocks, max_features, reshape_channel, reshape_depth, num_resblocks):
        super(AppearanceFeatureExtractor, self).__init__()
        self.image_channel = image_channel         # 输入图像的通道数（例如RGB图像是3）。                3
        self.block_expansion = block_expansion     # 每个块的扩展因子，用于定义卷积层输出通道数          64
        self.num_down_blocks = num_down_blocks     # 下采样块的数量。                                   2 
        self.max_features = max_features           # 网络中最大特征数的限制。                          512
        self.reshape_channel = reshape_channel     # 用于3D重塑时通道数的数量。                         32
        self.reshape_depth = reshape_depth         # 3D重塑时的深度                                    16

        # 第一个卷积块，它将输入图像从 image_channel 个通道转换为 block_expansion 个通道。
        # 卷积核大小是 (3, 3)，padding为 (1, 1)，即在图像边缘填充1个像素，保持输出尺寸不变。
        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1)) # image_channel = 3; block_expansion=64

        down_blocks = []
        for i in range(num_down_blocks):     # 创建了 num_down_blocks 个下采样块              (64->128)  ->   (128->256)
            in_features = min(max_features, block_expansion * (2 ** i))          # 输入特征图的通道数
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))   # 输出特征图的通道数
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))   # 通道变化。
        self.down_blocks = nn.ModuleList(down_blocks)

        # 1x1卷积层，将下采样后的特征图转换为具有 max_features 个通道的输出。
        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.resblocks_3d = torch.nn.Sequential()     # PyTorch 中的一个容器，用于按顺序将多个神经网络层（如卷积层、激活函数、池化层等）串联在一起。它使得模型的构建变得更加简洁、清晰，尤其是当模型的各层按顺序排列时。
        for i in range(num_resblocks):           # 3D残差块的数量       6
            # 每个块对输入的3D特征进行处理，维度保持为(reshape_channel, reshape_depth, h, w)
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

    def forward(self, source_image):         # 输入的源图像，形状为 [B, C, H, W]
        out = self.first(source_image)  # Bx3x256x256 -> Bx64x256x256

        for i in range(len(self.down_blocks)):      # Bx64x64x64  -> Bx128x64x64 -> Bx256x64x64
            out = self.down_blocks[i](out)
        out = self.second(out)        # Bx256x64x64  -> Bx512x64x64
        bs, c, h, w = out.shape       # ->Bx512x64x64

        f_s = out.view(bs, self.reshape_channel, self.reshape_depth, h, w)  #   Bx512x64x64 reshape->  Bx32x16x64x64
        f_s = self.resblocks_3d(f_s)  # ->Bx32x16x64x64
        return f_s   # [1, 32, 16, 64, 64]

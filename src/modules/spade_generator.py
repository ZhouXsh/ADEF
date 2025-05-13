# coding: utf-8

"""
Spade decoder(G) defined in the paper, which input the warped feature to generate the animated image.
本文定义了Spade解码器（G），它输入扭曲特征来生成动画图像。

与LivePortrait完全一致
"""

import torch
from torch import nn
import torch.nn.functional as F
from .util import SPADEResnetBlock

# SPADE 解码器的任务是从条件输入（如分割图、标签图或其他类型的特征图）中生成目标图像。
# 它通过一系列残差块（SPADEResnetBlock）和上采样操作来逐步恢复图像的高分辨率。
class SPADEDecoder(nn.Module):
    def __init__(self, 
                 upscale=1,             # 2 # represents upsample factor 256x256 -> 512x512
                 max_features=256,      # 512
                 block_expansion=64, out_channels=64, num_down_blocks=2):
        for i in range(num_down_blocks):  # 0~1
            input_channels = min(max_features, block_expansion * (2 ** (i + 1)))
        self.upscale = upscale
        super().__init__()
        norm_G = 'spadespectralinstance'
        label_num_channels = input_channels  # 256

        self.fc = nn.Conv2d(input_channels, 2 * input_channels, 3, padding=1)
        # 多个 SPADEResnetBlock，用于从低维特征映射逐步构建和提升图像的细节
        self.G_middle_0 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_1 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_2 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_3 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_4 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        self.G_middle_5 = SPADEResnetBlock(2 * input_channels, 2 * input_channels, norm_G, label_num_channels)
        # 这些残差块用于图像上采样，从低分辨率逐步将图像的尺寸扩大
        self.up_0 = SPADEResnetBlock(2 * input_channels, input_channels, norm_G, label_num_channels)   # 512 -> 256
        self.up_1 = SPADEResnetBlock(input_channels, out_channels, norm_G, label_num_channels)         # 256 -> 64
        self.up = nn.Upsample(scale_factor=2)  # 通过指定的 scale_factor=2 将图像尺寸在每次上采样时扩大一倍

        # self.conv_img: 最终的输出层。根据 upscale 值决定如何构建图像输出层
        if self.upscale is None or self.upscale <= 1:
            self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)   # 64 -> 3  RGB
        else:   # 2        this      represents upsample factor 256x256 -> 512x512
            self.conv_img = nn.Sequential(
                nn.Conv2d(out_channels, 3 * (2 * 2), kernel_size=3, padding=1),  # 64 -> 12
                # represents upsample factor 256x256 -> 512x512
                nn.PixelShuffle(upscale_factor=2)  # 如果有上采样，则通过 PixelShuffle 层（像素重排列层）来实现超分辨率上采样。这通常用于将低分辨率图像放大到高分辨率
            )

    def forward(self, feature):      # Bx256x64x64
        seg = feature                # Bx256x64x64  条件
        x = self.fc(feature)         # Bx512x64x64   通道翻倍
        x = self.G_middle_0(x, seg)  # Bx512x64x64 -> Bx512x64x64
        x = self.G_middle_1(x, seg)  # Bx512x64x64 -> Bx512x64x64
        x = self.G_middle_2(x, seg)  # Bx512x64x64 -> Bx512x64x64
        x = self.G_middle_3(x, seg)  # Bx512x64x64 -> Bx512x64x64
        x = self.G_middle_4(x, seg)  # Bx512x64x64 -> Bx512x64x64
        x = self.G_middle_5(x, seg)  # Bx512x64x64 -> Bx512x64x64

        x = self.up(x)  # Bx512x64x64 -> Bx512x128x128   上采样使图像分辨率增大
        x = self.up_0(x, seg)  # Bx512x128x128 -> Bx256x128x128 减小通道数
        x = self.up(x)  # Bx256x128x128 -> Bx256x256x256
        x = self.up_1(x, seg)  # Bx256x256x256 -> Bx64x256x256

        # 通过卷积和激活函数生成最终的图像。使用 leaky_relu 激活函数
        x = self.conv_img(F.leaky_relu(x, 2e-1))  # Bx64x256x256 -> Bx3x512x512   超分辨率
        x = torch.sigmoid(x)  # Bx3x512x512   通过 sigmoid 确保输出范围在 [0, 1] 之间，适用于图像输出

        return x  # Bx3x512x512
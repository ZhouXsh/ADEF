# coding: utf-8

"""
Stitching module(S) and two retargeting modules(R) defined in the paper.
本文定义了缝合模块（S）和两个重定向模块（R）。

- The stitching module pastes the animated portrait back into the original image space without pixel misalignment, such as in
the stitching region.
-拼接模块将动画肖像粘贴回原始图像空间，而不会出现像素错位，例如在拼接区域。

- The eyes retargeting module is designed to address the issue of incomplete eye closure during cross-id reenactment, especially
when a person with small eyes drives a person with larger eyes.
-眼睛重定向模块旨在解决在交叉id重现过程中眼睛闭合不完整的问题，特别是当一个小眼睛的人驱使一个大眼睛的人时。

- The lip retargeting module is designed similarly to the eye retargeting module, and can also normalize the input by ensuring that
the lips are in a closed state, which facilitates better animation driving.
-嘴唇重定向模块的设计类似于眼睛重定向模块，并且还可以通过确保嘴唇处于闭合状态来规范输入，这有助于更好的动画驱动。

与LivePortrait完全一致
"""
from torch import nn

# stitching:
#   input_size: 126 # (21*3)*2
#   hidden_sizes: [128, 128, 64]
#   output_size: 65 # (21*3)+2(tx,ty)
# lip:
#   input_size: 65 # (21*3)+2
#   hidden_sizes: [128, 128, 64]
#   output_size: 63 # (21*3)
# eye:
#   input_size: 66 # (21*3)+3
#   hidden_sizes: [256, 256, 128, 128, 64]
#   output_size: 63 # (21*3)

# 简单的多层感知机
class StitchingRetargetingNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(StitchingRetargetingNetwork, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))      # 输入层到隐藏层
            else:
                layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))  # 三个隐藏层之间
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_sizes[-1], output_size))      # 隐藏层到输出层
        self.mlp = nn.Sequential(*layers)

    # 将模型中所有的 Linear 层（全连接层）的权重和偏置初始化为零
    def initialize_weights_to_zero(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

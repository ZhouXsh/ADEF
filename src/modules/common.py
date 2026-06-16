import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# 位置编码器（时间嵌入 + 位置嵌入）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=600): # d_model=256：输入特征的维度 dropout：用于防止过拟合的丢弃率  max_len：输入序列的最大长度
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # vanilla sinusoidal encoding
        pe = torch.zeros(max_len, d_model)          # (max_len, d_model=256)  全0  位置编码矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [0,1,2,...,599]       shape = (max_len,1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))   # 缩放因子    shape: (d_model // 2,)
        pe[:, 0::2] = torch.sin(position * div_term)     # (特征维度）偶数位置，使用正弦函数 sin
        pe[:, 1::2] = torch.cos(position * div_term)     # 奇数位置，使用余弦函数 cos
        pe = pe.unsqueeze(0)    # (1, max_len=600, d_model)
        self.register_buffer('pe', pe)  # (1, max_len=600, d_model=256)  注册缓冲区，内容不变

    def forward(self, x):    # x: (N, L_p + L, feature_dim=256)
        x = x + self.pe[:, x.shape[1], :]  #  (N, L_p + L, feature_dim=256) +  (1, L_p + L=110<600, d_model=256)  广播后相加 = (N, L_p + L, feature_dim=256) 
        return self.dropout(x)     # 丢弃，减少过拟合风险  (N, L_p + L, feature_dim=256) 

# Encoder-Decoder掩码
def enc_dec_mask(T,       # Lp + L = 10 + 100 = 110
                 S,       # Lp + L = 10 + 100 = 110
                 frame_width=2,   # 1
                 expansion=0, device='cuda'):
    mask = torch.ones(T, S)      # (110,110)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)

# 对音频进行填充   (N, L_audio) -> (N, L_audio = audio_unit * n_units + pad_threshold)
# 通过反射填充或复制填充的方式，使音频的总长度从  audio_len  变成   audio_unit * n_units + pad_threshold  ！！！！
def pad_audio(audio, audio_unit=320, pad_threshold=80):     # (N, L_audio)
    batch_size, audio_len = audio.shape       # N, L_audio        1， 64000
    n_units = audio_len // audio_unit          # 音频长度所需的单位块数（n_units）   n_units =  L_audio // audio_unit  整除 =200
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)     # 音频的 每一侧 需要增加的长度。两侧都加。
    if side_len >= 0:   # eg:40
        reflect_len = side_len // 2         # 反射填充    eg:20
        replicate_len = side_len % 2        # 复制填充
        # side_len = reflect_len * 2 + replicate_len
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')   # 对音频的两侧进行反射填充，即音频数据的边缘部分会被反射过来填充
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')   # (1, 1)表示音频左右各填1

    return audio  # (N, L_audio = audio_unit * n_units + pad_threshold)   eg:[1, 64080]  （加上了pad_thereshold）

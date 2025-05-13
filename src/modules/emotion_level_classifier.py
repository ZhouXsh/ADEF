import torch
import torch.nn as nn
from src.modules.common import PositionalEncoding

torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=63, embed_dim=512, num_heads=8, ff_dim=4*512, num_layers=8, num_classes=8, level_classes=3):
        super(EmotionTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 运动参数嵌入层 (B, L, 63) -> (B, L, D)
        self.motion_embedding = nn.Linear(input_dim, embed_dim)
        
        # 位置编码
        self.PE = PositionalEncoding(embed_dim)

        # Transformer Encoder
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # 输出层 (B, D) -> (B, 8)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier_level = nn.Linear(embed_dim, level_classes)

    def forward(self, motion_seq):
        """
        motion_seq: (B, L=100, 63)  待预测情感的运动序列
        """
        # 运动参数嵌入 + 位置编码
        motion_embed = self.motion_embedding(motion_seq)  # (B, L, D)
        motion_embed = self.PE(motion_embed)

        # 经过 Transformer Encoder
        encoded_features = self.encoder(motion_embed)  # (B, L, D=128)

        # 平均池化操作
        encoded_features = encoded_features.mean(dim=1)  # (B, L, D) -> (B, D)

        out_emo = self.classifier(encoded_features)   # (B, D) -> (B, 8)
        out_level = self.classifier_level(encoded_features)         # 情感等级输出 (B, 3)

        return out_emo, out_level   # (B, 8)   # (B, 3)

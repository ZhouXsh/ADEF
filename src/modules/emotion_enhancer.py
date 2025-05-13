import torch
import torch.nn as nn
from src.modules.common import PositionalEncoding
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=63, emotion_dim=8, embed_dim=512, num_heads=8, ff_dim=512*4, num_layers=6):
        super(EmotionTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 运动参数嵌入层 (B, L, 63) -> (B, L, D)
        self.motion_embedding = nn.Linear(input_dim, embed_dim)
        
        # 位置编码
        # self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_dim))  # 假设最大 L=100
        self.PE = PositionalEncoding(self.embed_dim)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)

        self.emotion_embedding = nn.Embedding(emotion_dim, embed_dim)  # (B, 1, 8) -> (B, 1, D)
        self.emotion_level_embedding = nn.Embedding(3, embed_dim) 

        # 融合情感类别+等级（拼接后映射）
        self.emotion_fusion = nn.Linear(embed_dim * 2, embed_dim)

        # 输出层 (B, 1, D) -> (B, 1, 63)
        self.output_layer = nn.Linear(embed_dim, input_dim)

    def forward(self, motion_seq, emotion, level):
        """
        motion_seq: (B, L, 63)  无表情运动序列
        emotion: (B,)  One-hot 情感信息
        """

        emotion = self.emotion_embedding(emotion)    # (B, )  ->  (B, D)
        level = self.emotion_level_embedding(level)  # (B, )  ->  (B, D)

        emotion_embed = torch.cat([emotion, level], dim=-1)        # (B, D*2)
        emotion_embed = self.emotion_fusion(emotion_embed).unsqueeze(1)  # (B, 1, D)

        # 运动参数嵌入 + 位置编码
        motion_embed = self.motion_embedding(motion_seq)  # (B, L, D)
        motion_embed = self.PE(motion_embed)

        # 经过 Transformer Encoder
        encoded_features = self.encoder(motion_embed)  # (B, L, D)

        # 经过 Transformer Decoder
        decoded_output = self.decoder(encoded_features, emotion_embed)  # (B, 1, D)

        # 输出运动参数 (B, 1, D) -> (B, 1, 63)
        output = self.output_layer(decoded_output)

        return output


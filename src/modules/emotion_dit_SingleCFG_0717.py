import torch
import torch.nn as nn
import torch.nn.functional as F
import platform

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path

class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()
        if mode == 'linear':   
            betas = torch.linspace(beta_1, beta_T, num_steps)    # betas 从 beta_1 到
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
            betas = torch.linspace(beta_1, beta_T, num_steps)    # betas 从 beta_1 到 beta_T 线性变化
        elif mode == 'quadratic': 
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2   # 先从 beta_1 和 beta_T 的平方根开始线性变化，然后平方结果。此模式使得噪声的增加速度更加平缓
        elif mode == 'sigmoid':  
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1  # 通过 sigmoid 函数映射到 [beta_1, beta_T] 范围内的
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)   # x = [0,1,2,...,num_steps=500]
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])  # 1 减 后一项除以前一项
            betas = torch.clip(betas, 0.0001, 0.999)
        else:
            raise ValueError(f'Unknown diffusion schedule {mode}!')
        betas = torch.cat([torch.zeros(1), betas], dim=0)  
        alphas = 1 - betas                                  
        log_alphas = torch.log(alphas)   
        for i in range(1, log_alphas.shape[0]): 
            log_alphas[i] += log_alphas[i - 1]               
        alpha_bars = log_alphas.exp() 
        sigmas_flex = torch.sqrt(betas)                
        sigmas_inflex = torch.zeros_like(sigmas_flex)   
        for i in range(1, sigmas_flex.shape[0]):     
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)    

        self.num_steps = num_steps
        self.register_buffer('betas', betas)                 # 扩散系数（噪声强度） # exp：1-A，1-B，1-C
        self.register_buffer('alphas', alphas)               # 原图强度 1-betas     # exp：A，B，C
        self.register_buffer('alpha_bars', alpha_bars)        # 简化计算的α累乘     # exp：A，A·B，A·B·C
        self.register_buffer('sigmas_flex', sigmas_flex)     # 扩散系数betas 的平方根
        self.register_buffer('sigmas_inflex', sigmas_inflex) # sigmas_inflex[t]：xt-1时刻的概率分布P(Xt-1|xt,x0)的标准差（也是固定值）

    def uniform_sample_t(self, batch_size):  # 采样时间步
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))   # 范围[1, self.num_steps + 1)，大小batch_size的随机整数张量
        return ts.tolist()  # (B,)


    # 根据给定的时间步 t 返回相应的噪声 标准差
    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        # sigmas_flex[t] 和 sigmas_inflex[t] 是噪声的两个版本，flexibility 控制这两个版本之间的平衡
        return sigmas

class DitTalkingHead(nn.Module):
    def __init__(self, device='cuda', target="sample", architecture="decoder",  
                 motion_feat_dim=70, fps=25, n_motions=100, n_prev_motions=10,                               
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,                                             
                 diff_schedule="cosine", cfg_mode="incremental", guiding_conditions="audio,emotion", emo_classes = 8):
        super().__init__()
        # Model parameters
        self.target = target # 预测原始图像还是预测噪声
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim   # motion 特征维度 
        self.fps = fps
        self.n_motions = n_motions # 当前motion100个, window_length, T_w   窗口长度
        self.n_prev_motions = n_prev_motions # 前续motion
        self.feature_dim = feature_dim

        # Audio encoder  音频编码器（音频->向量）
        self.audio_model = audio_model         # hubert
        # 加载预训练模型 并 冻结参数
        if self.audio_model == 'wav2vec2':
            print("using wav2vec2 audio encoder ...")
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()       # 冻结参数
        elif self.audio_model == 'hubert': # 根据经验，hubert特征提取器效果更好
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh': # 根据经验，hubert特征提取器效果更好   # 这个 √
            print("using hubert chinese ori")
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')
        # 音频编码器的输出通常是一个形状为 [batch_size, seq_len, 768] 的张量，其中 768 是编码器输出的特征维度。   seq_len这里为帧数
        if architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, feature_dim)              # 768 -> 256
            self.start_audio_feat = nn.Parameter(torch.randn(emo_classes, self.n_prev_motions, feature_dim))        # shape：（1, 25, feature_dim=256） 初始随机的音频特征向量
        else:
            raise ValueError(f'Unknown architecture {architecture}!')

        self.start_motion_feat = nn.Parameter(torch.randn(emo_classes, self.n_prev_motions, self.motion_feat_dim))  # shape：（1,  25, motion_feat_dim=70） 初始随机的运动特征向量

        # Diffusion model       扩散模型
        self.denoising_net = DenoisingNetwork(device=device, n_motions=self.n_motions, n_prev_motions=self.n_prev_motions, 
                                              motion_feat_dim=self.motion_feat_dim, feature_dim=feature_dim)
        # diffusion schedule    扩散调度器
        # 这个模块定义了扩散过程中的噪声调度，它决定了噪声在不同扩散步骤中的变化方式。例如，使用余弦调度时，噪声会逐渐减小。
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)  # 50  cosine

        # Classifier-free settings  无分类器设置
        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []        # ['audio', 'emotion', '']
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['audio', 'emotion']]    # ['audio', 'emotion']
        if 'audio' in self.guiding_conditions:   # True
            audio_feat_dim = feature_dim         # 256
            # self.null_audio_feat 就会作为一个学习的参数，其形状为 [1, 1, feature_dim=256]。这是一个占位符，用来在条件音频的引导下生成对应的运动特征。
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, audio_feat_dim)) # 1, 1, 256
            self.audio_norm = nn.LayerNorm(audio_feat_dim, eps=1e-5)
        if 'emotion' in self.guiding_conditions:   # True
            emotion_feat_dim = feature_dim         # 512
            self.null_emotion_feat = nn.Parameter(torch.zeros(1, 1, emotion_feat_dim)) # 1, 1, 512
            self.emo_embed = nn.Embedding(emo_classes, emotion_feat_dim)  # 8个情感类别的嵌入层
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(emotion_feat_dim, 2 * emotion_feat_dim, bias=True))

        self.to(device)

    @property
    def device(self):  # 返回设备信息（不必理会）
        return next(self.parameters()).device

    # 提取音频特征  (N, L_audio) -> (N, L_audio = audio_unit * n_units + pad_threshold) -> (N, 2L=200, 768) -> (N, 768, L) ->  (N, L=100, feature_dim=256)
    def extract_audio_feature(self, audio, frame_num=None):      # audio: (N, L_audio)  L_audio是通过采样率计算的音频长度
        frame_num = frame_num or self.n_motions         # 当前序列内的帧数  L = 100

        # # Strategy 1: resample during audio feature extraction               音频特征提取过程中 进行 重采样
        # hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num).last_hidden_state  # (N, L, 768)

        # Strategy 2: resample after audio feature extraction (BackResample)   音频特征提取后 进行 重采样（反向重采样）
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,     # pad_audio = (N, L_audio = audio_unit * n_units + pad_threshold)     eg:[1, 64080] 
                                           frame_num=frame_num * 2).last_hidden_state     # (N=8, 2L=200, 768)  编码器得到的音频特征向量   768为编码器输出的特征维度   eg:(1, 200, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L=200)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')  # (N, 768, L)   线性插值（重采样）
        hidden_states = hidden_states.transpose(1, 2)  # (N, L=100, 768)

        audio_feat = self.audio_feature_map(hidden_states)     # (N, L, 768)  ->  (N, L=100, feature_dim=256)   特征维度映射
        return audio_feat        # (N=8, L=100, feature_dim=256)        L：帧数   feature_dim：最终的音频特征维度

    def _cfg_lip_mask(self, x):  # 优化CFG，2026年6月25日
        lip_mask = torch.zeros_like(x, dtype=torch.bool)  # 优化CFG，2026年6月25日
        if x.shape[-1] >= 63:  # 优化CFG，2026年6月25日
            for lip_idx in [6, 12, 14, 17, 19, 20]:  # 优化CFG，2026年6月25日
                lip_mask[..., lip_idx * 3:lip_idx * 3 + 3] = True  # 优化CFG，2026年6月25日
        return lip_mask  # 优化CFG，2026年6月25日

    def _cfg_modulate_audio(self, audio_feat, emo_feat):  # 优化CFG，2026年6月25日
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)  # 优化CFG，2026年6月25日
        return self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift  # 优化CFG，2026年6月25日

    def _cfg_make_scale(self, cfg_cond, cfg_scale):  # 优化CFG，2026年6月25日
        raw_cfg_scale = cfg_scale  # 优化CFG，2026年6月25日
        if not isinstance(cfg_scale, list):  # 优化CFG，2026年6月25日
            cfg_scale = [cfg_scale] * len(cfg_cond)  # 优化CFG，2026年6月25日
        elif len(cfg_scale) < len(cfg_cond):  # 优化CFG，2026年6月25日
            cfg_scale = cfg_scale + [cfg_scale[-1]] * (len(cfg_cond) - len(cfg_scale))  # 优化CFG，2026年6月25日
        if len(cfg_cond) > 0:  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = zip(*sorted(zip(cfg_cond, cfg_scale), key=lambda x: ['audio', 'emotion'].index(x[0])))  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = list(cfg_cond), list(cfg_scale)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            cfg_cond, cfg_scale = [], []  # 优化CFG，2026年6月25日
        emo_lip_scale = raw_cfg_scale[len(cfg_scale)] if isinstance(raw_cfg_scale, list) and len(raw_cfg_scale) > len(cfg_scale) else 0.2  # 优化CFG，2026年6月25日
        return cfg_cond, cfg_scale, emo_lip_scale  # 优化CFG，2026年6月25日

    def _cfg_target_theta(self, results, cfg_cond, cfg_scale, emo_lip_scale):  # 优化CFG，2026年6月25日
        result_uncond = results[0][:, -self.n_motions:]  # 优化CFG，2026年6月25日
        if 'audio' in cfg_cond and 'emotion' in cfg_cond and len(results) == 3:  # 优化CFG，2026年6月25日
            result_audio = results[cfg_cond.index('audio') + 1][:, -self.n_motions:]  # 优化CFG，2026年6月25日
            result_full = results[cfg_cond.index('emotion') + 1][:, -self.n_motions:]  # 优化CFG，2026年6月25日
            audio_scale = cfg_scale[cfg_cond.index('audio')]  # 优化CFG，2026年6月25日
            emotion_scale = cfg_scale[cfg_cond.index('emotion')]  # 优化CFG，2026年6月25日
            delta_audio = result_audio - result_uncond  # 优化CFG，2026年6月25日
            delta_emo = result_full - result_audio  # 优化CFG，2026年6月25日
            lip_mask = self._cfg_lip_mask(result_uncond)  # 优化CFG，2026年6月25日
            target_theta = result_uncond + audio_scale * delta_audio  # 优化CFG，2026年6月25日
            target_theta = target_theta + emotion_scale * delta_emo.masked_fill(lip_mask, 0)  # 优化CFG，2026年6月25日
            target_theta = target_theta + emo_lip_scale * delta_emo.masked_fill(~lip_mask, 0)  # 优化CFG，2026年6月25日
            return target_theta  # 优化CFG，2026年6月25日
        target_theta = result_uncond  # 优化CFG，2026年6月25日
        for i in range(0, len(results) - 1):  # 优化CFG，2026年6月25日
            target_theta += cfg_scale[i] * (results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:])  # 优化CFG，2026年6月25日
        return target_theta  # 优化CFG，2026年6月25日

    def _cfg_build_inputs(self, audio_feat, prev_audio_feat, cfg_cond, emo_index):  # 优化CFG，2026年6月25日
        batch_size = audio_feat.shape[0]  # 优化CFG，2026年6月25日
        audio_feat_in, prev_audio_feat_in = [], []  # 优化CFG，2026年6月25日
        if 'emotion' in cfg_cond:  # 优化CFG，2026年6月25日
            emo_null = self.null_emotion_feat.expand(batch_size, -1, -1)  # 优化CFG，2026年6月25日
            emo_real = self.emo_embed(emo_index).unsqueeze(1)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            emo_null, emo_real = None, None  # 优化CFG，2026年6月25日
        if 'audio' in cfg_cond:  # 优化CFG，2026年6月25日
            audio_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)  # 优化CFG，2026年6月25日
            prev_audio_null = self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            audio_null, prev_audio_null = audio_feat, prev_audio_feat  # 优化CFG，2026年6月25日
        if 'emotion' in cfg_cond:  # 优化CFG，2026年6月25日
            audio_feat_in.append(self._cfg_modulate_audio(audio_null, emo_null))  # 优化CFG，2026年6月25日
            prev_audio_feat_in.append(self._cfg_modulate_audio(prev_audio_null, emo_null))  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            audio_feat_in.append(audio_null)  # 优化CFG，2026年6月25日
            prev_audio_feat_in.append(prev_audio_null)  # 优化CFG，2026年6月25日
        for cond in cfg_cond:  # 优化CFG，2026年6月25日
            if cond == 'audio':  # 优化CFG，2026年6月25日
                audio_item = self._cfg_modulate_audio(audio_feat, emo_null) if 'emotion' in cfg_cond else audio_feat  # 优化CFG，2026年6月25日
                prev_audio_item = self._cfg_modulate_audio(prev_audio_feat, emo_null) if 'emotion' in cfg_cond else prev_audio_feat  # 优化CFG，2026年6月25日
            else:  # 优化CFG，2026年6月25日
                audio_item = self._cfg_modulate_audio(audio_feat, emo_real)  # 优化CFG，2026年6月25日
                prev_audio_item = self._cfg_modulate_audio(prev_audio_feat, emo_real)  # 优化CFG，2026年6月25日
            audio_feat_in.append(audio_item)  # 优化CFG，2026年6月25日
            prev_audio_feat_in.append(prev_audio_item)  # 优化CFG，2026年6月25日
        return torch.cat(audio_feat_in, dim=0), torch.cat(prev_audio_feat_in, dim=0), len(audio_feat_in)  # 优化CFG，2026年6月25日

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, time_step=None, indicator=None, emo_index=None, return_cfg_state=False):  # 优化CFG，2026年6月25日
        batch_size = motion_feat.shape[0]  # 优化CFG，2026年6月25日
        if audio_or_feat.ndim == 2:  # 优化CFG，2026年6月25日
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), f'Incorrect audio length {audio_or_feat.shape[1]}'  # 优化CFG，2026年6月25日
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)  # 优化CFG，2026年6月25日
        elif audio_or_feat.ndim == 3:  # 优化CFG，2026年6月25日
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'  # 优化CFG，2026年6月25日
            audio_feat_saved = audio_or_feat  # 优化CFG，2026年6月25日
        else:  # 优化CFG，2026年6月25日
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')  # 优化CFG，2026年6月25日
        audio_feat = audio_feat_saved.clone()  # 优化CFG，2026年6月25日
        if prev_motion_feat is None:  # 优化CFG，2026年6月25日
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)  # 优化CFG，2026年6月25日
        if prev_audio_feat is None:  # 优化CFG，2026年6月25日
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)  # 优化CFG，2026年6月25日
        prev_audio_feat_saved = prev_audio_feat.clone()  # 优化CFG，2026年6月25日
        cfg_state = torch.full((batch_size,), 2, dtype=torch.long, device=self.device)  # 优化CFG，2026年6月25日
        cfg_is_full = torch.ones(batch_size, dtype=torch.bool, device=self.device)  # 优化CFG，2026年6月25日
        p_AE = 0.1  # 优化CFG，2026年6月25日
        p_E = 0.55  # 优化CFG，2026年6月25日
        if len(self.guiding_conditions) > 0:  # 优化CFG，2026年6月25日
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'  # 优化CFG，2026年6月25日
            if 'audio' in self.guiding_conditions and 'emotion' in self.guiding_conditions:  # 优化CFG，2026年6月25日
                mask_flag = torch.rand(batch_size, device=self.device)  # 优化CFG，2026年6月25日
                mask_uncond = mask_flag < p_AE  # 优化CFG，2026年6月25日
                mask_audio_only = (mask_flag >= p_AE) & (mask_flag < p_E)  # 优化CFG，2026年6月25日
                cfg_is_full = ~(mask_uncond | mask_audio_only)  # 优化CFG，2026年6月25日
                cfg_state = torch.where(mask_uncond, torch.zeros_like(cfg_state), torch.where(mask_audio_only, torch.ones_like(cfg_state), cfg_state))  # 优化CFG，2026年6月25日
                audio_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)  # 优化CFG，2026年6月25日
                prev_audio_null = self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)  # 优化CFG，2026年6月25日
                emo_null = self.null_emotion_feat.expand(batch_size, -1, -1)  # 优化CFG，2026年6月25日
                emo_real = self.emo_embed(emo_index).unsqueeze(1)  # 优化CFG，2026年6月25日
                audio_cond = torch.where(mask_uncond.view(-1, 1, 1), audio_null, audio_feat_saved)  # 优化CFG，2026年6月25日
                prev_audio_cond = torch.where(mask_uncond.view(-1, 1, 1), prev_audio_null, prev_audio_feat_saved)  # 优化CFG，2026年6月25日
                emo_cond = torch.where((mask_uncond | mask_audio_only).view(-1, 1, 1), emo_null, emo_real)  # 优化CFG，2026年6月25日
                audio_feat = self._cfg_modulate_audio(audio_cond, emo_cond)  # 优化CFG，2026年6月25日
                prev_audio_feat = self._cfg_modulate_audio(prev_audio_cond, emo_cond)  # 优化CFG，2026年6月25日
            elif 'audio' in self.guiding_conditions:  # 优化CFG，2026年6月25日
                mask_audio = torch.rand(batch_size, device=self.device) < p_AE  # 优化CFG，2026年6月25日
                cfg_state = torch.where(mask_audio, torch.zeros_like(cfg_state), torch.ones_like(cfg_state))  # 优化CFG，2026年6月25日
                cfg_is_full = ~mask_audio  # 优化CFG，2026年6月25日
                audio_null = self.null_audio_feat.expand(batch_size, self.n_motions, -1)  # 优化CFG，2026年6月25日
                prev_audio_null = self.null_audio_feat.expand(batch_size, self.n_prev_motions, -1)  # 优化CFG，2026年6月25日
                audio_feat = torch.where(mask_audio.view(-1, 1, 1), audio_null, audio_feat_saved)  # 优化CFG，2026年6月25日
                prev_audio_feat = torch.where(mask_audio.view(-1, 1, 1), prev_audio_null, prev_audio_feat_saved)  # 优化CFG，2026年6月25日
            elif 'emotion' in self.guiding_conditions:  # 优化CFG，2026年6月25日
                mask_emotion = torch.rand(batch_size, device=self.device) < p_AE  # 优化CFG，2026年6月25日
                cfg_state = torch.where(mask_emotion, torch.zeros_like(cfg_state), cfg_state)  # 优化CFG，2026年6月25日
                cfg_is_full = ~mask_emotion  # 优化CFG，2026年6月25日
                emo_null = self.null_emotion_feat.expand(batch_size, -1, -1)  # 优化CFG，2026年6月25日
                emo_real = self.emo_embed(emo_index).unsqueeze(1)  # 优化CFG，2026年6月25日
                emo_cond = torch.where(mask_emotion.view(-1, 1, 1), emo_null, emo_real)  # 优化CFG，2026年6月25日
                audio_feat = self._cfg_modulate_audio(audio_feat_saved, emo_cond)  # 优化CFG，2026年6月25日
                prev_audio_feat = self._cfg_modulate_audio(prev_audio_feat_saved, emo_cond)  # 优化CFG，2026年6月25日
        if time_step is None:  # 优化CFG，2026年6月25日
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)  # 优化CFG，2026年6月25日
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]  # 优化CFG，2026年6月25日
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)  # 优化CFG，2026年6月25日
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)  # 优化CFG，2026年6月25日
        eps = torch.randn_like(motion_feat)  # 优化CFG，2026年6月25日
        motion_feat_noisy = c0 * motion_feat + c1 * eps  # 优化CFG，2026年6月25日
        motion_feat_target = self.denoising_net(motion_feat_noisy, audio_feat, prev_motion_feat, prev_audio_feat, time_step, indicator)  # 优化CFG，2026年6月25日
        if return_cfg_state:  # 优化CFG，2026年6月25日
            return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach(), cfg_is_full.detach(), cfg_state.detach()  # 优化CFG，2026年6月25日
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()  # 优化CFG，2026年6月25日

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None, motion_at_T=None,
               indicator=None, cfg_mode=None, cfg_cond=None, cfg_scale=1.15, flexibility=0,
               dynamic_threshold=None, ret_traj=False, emo_index=None):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = self.cfg_mode if cfg_mode is None else cfg_mode
        cfg_cond = self.guiding_conditions if cfg_cond is None else cfg_cond
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        cfg_cond, cfg_scale, emo_lip_scale = self._cfg_make_scale(cfg_cond, cfg_scale)  # 优化CFG，2026年6月25日
        print(f'cfg_cond: {cfg_cond}, cfg_scale: {cfg_scale}, emo_lip_scale: {emo_lip_scale}')  # 优化CFG，2026年6月25日

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim)).to(self.device)

        audio_feat_in, prev_audio_feat_in, n_entries = self._cfg_build_inputs(audio_feat.clone(), prev_audio_feat.clone(), cfg_cond, emo_index)  # 优化CFG，2026年6月25日
        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)
            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)
            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in, prev_audio_feat_in, step_in, indicator_in)
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)
            results = results.chunk(n_entries)
            if cfg_mode in ['independent', 'incremental']:  # 优化CFG，2026年6月25日
                target_theta = self._cfg_target_theta(results, cfg_cond, cfg_scale, emo_lip_scale)  # 优化CFG，2026年6月25日
            else:
                raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')
            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError('Unknown target type: {}'.format(self.target))
            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]
        if ret_traj:
            return traj, motion_at_T, audio_feat  # 优化CFG，2026年6月25日
        return traj[0], motion_at_T, audio_feat  # 优化CFG，2026年6月25日

# 去噪网络 DiT
class DenoisingNetwork(nn.Module):
    def __init__(self, device='cuda', motion_feat_dim=73, 
                 use_indicator=None, architecture="decoder", feature_dim=256, n_heads=8, 
                 n_layers=8, mlp_ratio=4, align_mask_width=1, no_use_learnable_pe=True, n_prev_motions=10,
                 n_motions=100, n_diff_steps=500, ):
        super().__init__()
        # Model parameters
        self.motion_feat_dim = motion_feat_dim     # 推理73
        self.use_indicator = use_indicator

        # Transformer
        self.architecture = architecture          # "decoder"
        self.feature_dim = feature_dim            # 256
        self.n_heads = n_heads                    # 多头注意力的头数   8
        self.n_layers = n_layers                  # Transformer块的层数  8。
        self.mlp_ratio = mlp_ratio                # MLP部分的扩展比率。用于计算feedforward层的维度，默认为 4。
        self.align_mask_width = align_mask_width  # 对齐掩码宽度，控制自注意力的局部性。默认为 1。
        self.use_learnable_pe = not no_use_learnable_pe  # 是否使用可学习的位置编码 False

        # sequence length
        self.n_prev_motions = n_prev_motions   # 先前运动特征数（帧         
        self.n_motions = n_motions             # 当前运动特征数（帧

        # Temporal embedding for the diffusion time step   扩散时间步长的时间嵌入
        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)  # 时间嵌入   256 , 501
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),   # 256 -> 256
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim)    # 256 -> 256
        )

        if self.use_learnable_pe:
            # Learnable positional encoding  可学习的位置编码
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))   # (1, 1 + L_p + L, feature_dim=256)
        else:       # this
            self.PE = PositionalEncoding(self.feature_dim)   #  256         # self.PE.pe : (1, 600, 256)

        # Transformer decoder
        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),   # （73or74) or (70or71) -> 256
                                          self.feature_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.feature_dim,         # 输入和输出的特征维度  256
                nhead=self.n_heads,               # 注意力头数   8
                dim_feedforward=self.mlp_ratio * self.feature_dim,  # 前馈层的维度   4 * 256               
                activation='gelu', batch_first=True
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)   # num_layers=8个块（层）
            if self.align_mask_width > 0:     # 1
                motion_len = self.n_prev_motions + self.n_motions   # Lp + L =  125
                alignment_mask = enc_dec_mask(motion_len, motion_len, frame_width=1, expansion=self.align_mask_width - 1)     # (Lp + L, Lp + L)
                # print(f"alignment_mask: ", alignment_mask.shape)
                # alignment_mask = F.pad(alignment_mask, (0, 0, 1, 0), value=False)
                self.register_buffer('alignment_mask', alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Motion decoder  运动解码器
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),     # 256 -> 128
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),  # 128 -> 70
            # nn.Tanh() # 增加了一个tanh
            # nn.Softmax()
        )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        """
        Args:
            motion_feat: (N, L, d_motion). Noisy motion feature    forward(N=8, L=100, d_motion=70) 加噪后的最终噪声         sample: (2, 100, 73)  当前step的噪声（用于预测前一step的）
            audio_feat: (N, L, feature_dim)   forward：“屏蔽”后的音频特征(N=8, L=100, feature_dim=256)     sample时：随机+真实(2, L=100, feature_dim=256)
            prev_motion_feat: (N, L_p, d_motion). Padded previous motion coefficients or feature  填充的先前运动特征 forward(8, n_prev_motions=25, motion_feat_dim=70)   sample(N, n_prev_motions=10, motion_feat_dim=73)
            prev_audio_feat: (N, L_p, d_audio). Padded previous motion coefficients or feature    填充的先前音频特征 forward(8, n_prev_motions=25, feature_dim=256)      sample(N, n_prev_motions=10, motion_feat_dim=256)
            step: (N,)                                         时间步，1~500的随机值  forward(8,)       sample(2,)
            indicator: (N, L). 0/1 indicator for the real (unpadded) motion feature  # (N, L) None      forward(8,100)       sample(2,100)
        Returns:
            motion_feat_target: (N, L_p + L, d_motion)    forward(8, 125, 70)   sample(2, 110, 73)
        """
        # Diffusion time step embedding  扩散时间步长嵌入  
        # TE.pe                shape: [1, n_diff_steps + 1=501, d_model=256]
        # TE.pe[0, step]       shape: [d_model=256]        维度1的位置0，维度2的位置step  批次0的第step步
        # self.TE.pe[0, step]                               (N=8, diff_step_dim=256)   
        # self.diff_step_map(self.TE.pe[0, step])           (N=8, diff_step_dim=256)   

        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)    # 时间步嵌入 forward(N=8 or 2, 1, diff_step_dim=256)    (N=2, 1, diff_step_dim=256)

        # 指示器用于指示 最后一个音频片段 中 填充的部分。
        if indicator is not None:   # 包含指示器
            indicator = torch.cat([torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                                   indicator], dim=1)         # (N, L_p=25 or 10) cat (N, L) = (N, L_p + L=125 or 110)
            indicator = indicator.unsqueeze(-1)               # forward (8, 125, 1)      sample (2, 110, 1)

        # Concat features and embeddings  拼接（先前运动）特征和（指示器）嵌入
        if self.architecture == 'decoder':
            # print("prev_motion_feat: ", prev_motion_feat.shape, "motion_feat: ", motion_feat.shape)
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)  # (N, L_p + L, d_motion) 
        else:       
            raise ValueError(f'Unknown architecture: {self.architecture}')
        if self.use_indicator:   # 拼接指示器   
            feats_in = torch.cat([feats_in, indicator], dim=-1)  # (N, L_p + L, d_motion)+(N, L_p + L, 1) = (N, L_p + L, d_motion + 1 )

        feats_in = self.feature_proj(feats_in)  # (N, L_p + L=125 or 110, 70 or 73) -> (N, L_p + L=125 or 110, feature_dim=256)
        # feats_in = torch.cat([person_feat, feats_in], dim=1)  # (N, 1 + L_p + L, feature_dim)

        if self.use_learnable_pe:      # 可学习的位置嵌入
            # feats_in = feats_in + self.PE
            # self.PE : (1, 1 + L_p + L, feature_dim=256)
            feats_in = feats_in + self.PE + diff_step_embedding # (N, L_p + L, feature_dim=256) + (1, 1 + L_p + L, feature_dim=256) + (N=2, 1, diff_step_dim=256)
        else:
            # feats_in = self.PE(feats_in)         forward(8 125 256)+(8 1 256) = (8 125 256)
            feats_in = self.PE(feats_in) + diff_step_embedding  # (N, L_p + L, feature_dim=256) + (N, 1, diff_step_dim=256) = (N, L_p + L, feature_dim=256)

        # Transformer
        if self.architecture == 'decoder':   # forard(N, n_prev_motions=25, feature_dim=256) cat (N=8, L=100, feature_dim=256) = (8 125 256)
            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)        # (N, L_p + L, d_audio= feature_dim=256)
            feat_out = self.transformer(feats_in, audio_feat_in, memory_mask=self.alignment_mask)     # (N, L_p + L, d_audio= feature_dim=256)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Decode predicted motion feature noise / sample
        # motion_feat_target = self.motion_dec(feat_out[:, 1:])  # (N, L_p + L, d_motion)
        motion_feat_target = self.motion_dec(feat_out)          # (N, L_p + L=110, 512 -> 256 -> 73 or 70)

        return motion_feat_target

if __name__ == "__main__":
    device = "cuda"
    motion_feat_dim = 76
    n_motions = 100 # L
    n_prev_motions = 10 # L_p

    L_audio = int(16000 * n_motions / 25) # 64000
    d_audio = 768

    N = 5
    feature_dim = 512

    motion_feat = torch.ones((N, n_motions, motion_feat_dim)).to(device)
    prev_motion_feat = torch.ones((N, n_prev_motions, motion_feat_dim)).to(device)

    audio_or_feat = torch.ones((N, L_audio)).to(device)
    prev_audio_feat = torch.ones((N, n_prev_motions, d_audio)).to(device)

    time_step = torch.ones(N, dtype=torch.long).to(device)

    model = DitTalkingHead().to(device)

    z = model(motion_feat, audio_or_feat, prev_motion_feat=None, 
              prev_audio_feat=None, time_step=None, indicator=None)
    traj, motion_at_T, audio_feat = z[0], z[1], z[2]
    print(motion_at_T.shape, audio_feat.shape)

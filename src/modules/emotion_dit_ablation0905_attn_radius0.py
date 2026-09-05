"""Self-contained ICASSP27 controlled ablation: attn_radius0.

The complete model implementation and the runtime-correction layer live in this
single file.  ``_CoreDenoisingNetwork`` and ``_CoreDitTalkingHead`` are private
implementation classes; ``DenoisingNetwork`` and ``DitTalkingHead`` are the
public checkpoint-compatible classes.  This module never imports another
ablation model and has no companion ``*_legacy.py`` file.
"""

## 大一统版本。
## 本文件是自包含的：内联了 emotion_dit_timestep_0714 中的
##   - DiffusionSchedule
##   - DenoisingNetwork（使用 adaLN-Zero DiTDecoder，逐层 FiLM 注入扩散时间步）
##   - DiTDecoderLayer / DiTDecoder
## 以及 emotion_dit 的音频编码器与 start_motion_feat / start_audio_feat 逻辑。
## DitTalkingHead 覆写 ``forward`` 和 ``sample``，把 audio 与 emotion 视为不可分的
## 联合 CFG 条件。``align_mask_width`` 通过 ``__init__`` 直接传入并下放到
## ``DenoisingNetwork.alignment_mask``，不再是基类的默认硬编码值。

import platform

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path


class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps, mode='linear', beta_1=1e-4, beta_T=0.02, s=0.008):
        super().__init__()
        if mode == 'linear':
            betas = torch.linspace(beta_1, beta_T, num_steps)
        elif mode == 'quadratic':
            betas = torch.linspace(beta_1 ** 0.5, beta_T ** 0.5, num_steps) ** 2
        elif mode == 'sigmoid':
            betas = torch.sigmoid(torch.linspace(-5, 5, num_steps)) * (beta_T - beta_1) + beta_1
        elif mode == 'cosine':
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alpha_bars = alpha_bars / alpha_bars[0]
            betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
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
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility=0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


def modulate(x, shift, scale):
    # adaLN 调制： x * (1 + scale) + shift
    return x * (1 + scale) + shift


class DiTDecoderLayer(nn.Module):
    """
    adaLN-Zero 风格的 Transformer decoder block。
    扩散时间步嵌入 t_emb 通过 FiLM(shift/scale/gate) 分别注入到
    自注意力、交叉注意力、前馈三条路径之间。
    参考 FaceTalk / DiT：每个 block 独立地接收时间步调制，避免时间步信号随网络加深被稀释。
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # 前馈
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu

        # 调制前使用无仿射的 LayerNorm（仿射由 shift/scale 提供）
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)

        # 由时间步嵌入生成 3 条路径 x (shift, scale, gate) = 9 组调制向量
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 9 * d_model, bias=True),
        )
        # adaLN-Zero: 最后一层零初始化，保证训练初期每个 block 近似恒等（gate=0）
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, tgt, memory, t_emb, memory_mask=None, tgt_mask=None):
        # t_emb: (N, 1, d_model)  时间步嵌入，在各路径上广播
        (shift_sa, scale_sa, gate_sa,
         shift_ca, scale_ca, gate_ca,
         shift_ff, scale_ff, gate_ff) = self.adaLN_modulation(t_emb).chunk(9, dim=-1)

        # 自注意力
        h = modulate(self.norm1(tgt), shift_sa, scale_sa)
        sa = self.self_attn(h, h, h, attn_mask=tgt_mask, need_weights=False)[0]
        tgt = tgt + gate_sa * sa

        # 交叉注意力（对音频特征 memory）
        h = modulate(self.norm2(tgt), shift_ca, scale_ca)
        ca = self.cross_attn(h, memory, memory, attn_mask=memory_mask, need_weights=False)[0]
        tgt = tgt + gate_ca * ca

        # 前馈
        h = modulate(self.norm3(tgt), shift_ff, scale_ff)
        ff = self.linear2(self.dropout(self.activation(self.linear1(h))))
        tgt = tgt + gate_ff * ff

        return tgt


class DiTDecoder(nn.Module):
    """DiTDecoderLayer 的堆叠，逐层向每个 block 传入扩散时间步嵌入。"""

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            DiTDecoderLayer(d_model, nhead, dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, t_emb, memory_mask=None, tgt_mask=None):
        for layer in self.layers:
            tgt = layer(tgt, memory, t_emb, memory_mask=memory_mask, tgt_mask=tgt_mask)
        return tgt


# 去噪网络 DiT
class _CoreDenoisingNetwork(nn.Module):
    def __init__(self, device='cuda', motion_feat_dim=70,
                 use_indicator=None, architecture="decoder", feature_dim=512, n_heads=8,
                 n_layers=8, mlp_ratio=4, align_mask_width=1, no_use_learnable_pe=True, n_prev_motions=16,
                 n_motions=64, n_diff_steps=500):
        super().__init__()
        # Model parameters
        self.motion_feat_dim = motion_feat_dim
        self.use_indicator = use_indicator

        # Transformer
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe

        # sequence length
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        # Temporal embedding for the diffusion time step
        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if self.use_learnable_pe:
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        # Transformer decoder
        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(self.motion_feat_dim + (1 if self.use_indicator else 0),
                                          self.feature_dim)
            # adaLN-Zero DiT decoder：扩散时间步逐层注入到每个 block 的自注意力/交叉注意力/前馈之间
            self.transformer = DiTDecoder(
                d_model=self.feature_dim,
                nhead=self.n_heads,
                dim_feedforward=self.mlp_ratio * self.feature_dim,
                num_layers=self.n_layers,
            )
            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len,
                                              frame_width=1, expansion=self.align_mask_width - 1)
                self.register_buffer('alignment_mask', alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        # Motion decoder
        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),
        )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):
        # Diffusion time step embedding
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        # 指示器用于指示 最后一个音频片段 中 填充的部分。
        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                indicator,
            ], dim=1)
            indicator = indicator.unsqueeze(-1)

        if self.architecture == 'decoder':
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        feats_in = self.feature_proj(feats_in)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE
        else:
            feats_in = self.PE(feats_in)

        if self.architecture == 'decoder':
            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
            feat_out = self.transformer(feats_in, audio_feat_in, diff_step_embedding,
                                        memory_mask=self.alignment_mask)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        motion_feat_target = self.motion_dec(feat_out)
        return motion_feat_target


class _CoreDitTalkingHead(nn.Module):
    """Audio and emotion are treated as one inseparable CFG condition.

    与 emotion_dit.DitTalkingHead 同构：构造相同的音频编码器、start 特征、CFG
    占位符与 adaLN 调制层；``align_mask_width`` 通过 ``__init__`` 直接传入到
    ``DenoisingNetwork``，避免依赖基类的默认行为。
    """

    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=64, n_prev_motions=16,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental",
                 guiding_conditions="audio,emotion", emo_classes=8,
                 align_mask_width=1):
        super().__init__()
        # Model parameters
        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim

        # Audio encoder
        self.audio_model = audio_model
        if self.audio_model == 'wav2vec2':
            print("using wav2vec2 audio encoder ...")
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh':
            print("using hubert chinese ori")
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, feature_dim)
            self.start_audio_feat = nn.Parameter(
                torch.randn(emo_classes, self.n_prev_motions, feature_dim))
        else:
            raise ValueError(f'Unknown architecture {architecture}!')

        self.start_motion_feat = nn.Parameter(
            torch.randn(emo_classes, self.n_prev_motions, self.motion_feat_dim))

        # Diffusion model — align_mask_width 直接下放到 DenoisingNetwork
        self.denoising_net = _CoreDenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            n_diff_steps=n_diff_steps,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
            align_mask_width=align_mask_width,
        )
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        # Classifier-free settings
        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [
            cond for cond in guiding_conditions if cond in ['audio', 'emotion']
        ]
        if 'audio' in self.guiding_conditions:
            audio_feat_dim = feature_dim
            self.null_audio_feat = nn.Parameter(torch.randn(1, 1, audio_feat_dim))
            self.audio_norm = nn.LayerNorm(audio_feat_dim, eps=1e-9)
        if 'emotion' in self.guiding_conditions:
            emotion_feat_dim = feature_dim
            self.null_emotion_feat = nn.Parameter(torch.zeros(1, 1, emotion_feat_dim))
            self.emo_embed = nn.Embedding(emo_classes, emotion_feat_dim)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emotion_feat_dim, 2 * emotion_feat_dim, bias=True),
            )

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps,
                                           frame_num=frame_num * 2).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=frame_num,
                                      align_corners=False, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)
        audio_feat = self.audio_feature_map(hidden_states)
        return audio_feat

    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None,
                prev_audio_feat=None, time_step=None, indicator=None,
                emo_index=None):
        """Joint-CFG forward.

        Audio 与 emotion 作为不可分的联合 CFG 条件：要么都保留，要么一起 drop。
        若 guiding_conditions 不同时包含 audio 和 emotion，则回退到 emotion_dit
        风格的标准 CFG 路径（``audio`` / ``emotion`` 各自独立丢弃）。
        """
        joint_cfg = (
            'audio' in self.guiding_conditions
            and 'emotion' in self.guiding_conditions
        )
        if not joint_cfg:
            return self._forward_independent_cfg(
                motion_feat, audio_or_feat,
                prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat,
                time_step=time_step,
                indicator=indicator,
                emo_index=emo_index,
            )

        batch_size = motion_feat.shape[0]

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(
                16000 * self.n_motions / self.fps
            ), f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(
                self.start_motion_feat, 0, emo_index)

        prev_audio_is_start = prev_audio_feat is None
        if prev_audio_is_start:
            prev_audio_feat = torch.index_select(
                self.start_audio_feat, 0, emo_index)

        # Conditional branch: real audio + real emotion.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat_cond = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        # Unconditional branch: null audio + null emotion.
        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(
            null_emotion_feat
        ).chunk(2, dim=2)
        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )

        # One dropout decision controls both audio and emotion.
        joint_drop_prob = 0.1
        drop_joint_condition = (
            torch.rand(batch_size, device=self.device) < joint_drop_prob
        )
        audio_feat = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_cond,
        )

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)

        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps
        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
        )

        return (
            eps,
            motion_feat_target,
            motion_feat.detach(),
            audio_feat_saved.detach(),
        )

    def _forward_independent_cfg(self, motion_feat, audio_or_feat,
                                 prev_motion_feat=None, prev_audio_feat=None,
                                 time_step=None, indicator=None, emo_index=None):
        """emotion_dit 风格的标准 CFG 路径：audio / emotion 各自独立 drop。"""
        batch_size = motion_feat.shape[0]

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(
                16000 * self.n_motions / self.fps
            ), f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')
        audio_feat = audio_feat_saved.clone()

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        pre_None = False
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
            pre_None = True

        p_AE = 0.1
        p_E = 0.55

        if 'emotion' in self.guiding_conditions:
            emo_feat = self.emo_embed(emo_index)
            emo_feat = emo_feat.unsqueeze(1)
            emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
            if pre_None:
                prev_audio_feat = self.audio_norm(prev_audio_feat)
            else:
                prev_audio_feat = self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift

        if len(self.guiding_conditions) > 0:
            assert len(self.guiding_conditions) <= 2, 'Only support 1 or 2 CFG conditions!'
            mask_flag = torch.rand(batch_size, device=self.device)
            if len(self.guiding_conditions) == 1 or self.cfg_mode == 'independent':
                null_cond_prob = 0.5 if len(self.guiding_conditions) >= 2 else 0.1
                if 'audio' in self.guiding_conditions:
                    mask_audio = torch.rand(batch_size, device=self.device) < null_cond_prob
                    audio_feat = torch.where(
                        mask_audio.view(-1, 1, 1),
                        self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                        audio_feat,
                    )
            else:
                if 'audio' in self.guiding_conditions:
                    mask_audio = mask_flag < p_AE
                    audio_feat = torch.where(
                        mask_audio.view(-1, 1, 1),
                        self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                        audio_feat,
                    )
            if len(self.guiding_conditions) == 2 and 'emotion' in self.guiding_conditions:
                mask_emotion = mask_flag < p_E
                emo_feat = torch.where(
                    mask_emotion.view(-1, 1, 1),
                    self.null_emotion_feat.expand(batch_size, -1, -1),
                    emo_feat,
                )
                emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
                audio_feat = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)

        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)

        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps
        motion_feat_target = self.denoising_net(
            motion_feat_noisy, audio_feat,
            prev_motion_feat, prev_audio_feat, time_step, indicator,
        )

        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility=0, dynamic_threshold=None,
               ret_traj=False, emo_index=None):
        batch_size = audio_or_feat.shape[0]

        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_mode not in ['incremental', 'independent']:
            raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        elif isinstance(cfg_cond, str):
            cfg_cond = cfg_cond.split(',')
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]

        # Audio and emotion form one condition. Any legacy request for either
        # name enables the same joint CFG branch while preserving the interface.
        use_joint_cfg = (
            len(cfg_cond) > 0
            and 'audio' in self.guiding_conditions
            and 'emotion' in self.guiding_conditions
        )
        if isinstance(cfg_scale, (list, tuple)):
            joint_cfg_scale = cfg_scale[-1] if len(cfg_scale) > 0 else 1.0
        else:
            joint_cfg_scale = cfg_scale

        print(
            f"cfg_cond: {('audio+emotion',) if use_joint_cfg else ()}, "
            f"cfg_scale: {(joint_cfg_scale,) if use_joint_cfg else ()}"
        )

        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == 16000 * self.n_motions / self.fps, \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)

        prev_audio_is_start = prev_audio_feat is None
        if prev_audio_is_start:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)

        if motion_at_T is None:
            motion_at_T = torch.randn(
                batch_size, self.n_motions, self.motion_feat_dim,
                device=self.device,
            )

        # Full joint condition.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat_cond = (
            self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift
        )

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        # Fully dropped joint condition.
        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(
            null_emotion_feat
        ).chunk(2, dim=2)
        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )

        if use_joint_cfg:
            audio_feat_in = torch.cat([audio_feat_uncond, audio_feat_cond], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            n_entries = 1

        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
        indicator_in = (
            torch.cat([indicator] * n_entries, dim=0)
            if indicator is not None else None
        )

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            if t > 1:
                z = torch.randn_like(motion_at_T)
            else:
                z = torch.zeros_like(motion_at_T)

            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = torch.cat([step_in] * n_entries, dim=0)

            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
            )

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(
                    batch_size * n_entries, -1
                ).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)
            if use_joint_cfg:
                uncond_target = results[0][:, -self.n_motions:]
                cond_target = results[1][:, -self.n_motions:]
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = results[0][:, -self.n_motions:]

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (
                    (1 - alpha_bar_prev) * torch.sqrt(alpha)
                    / (1 - alpha_bar)
                )
                c1 = (
                    (1 - alpha) * torch.sqrt(alpha_bar_prev)
                    / (1 - alpha_bar)
                )
                motion_next = (
                    c0 * motion_at_t + c1 * target_theta + sigma * z
                )
            else:
                raise ValueError(f'Unknown target type: {self.target}')

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat_cond
        return traj[0], motion_at_T, audio_feat_cond


__all__ = ['DiffusionSchedule', 'DenoisingNetwork', 'DiTDecoderLayer', 'DiTDecoder', 'DitTalkingHead']


if __name__ == "__main__":
    device = "cuda"
    motion_feat_dim = 70
    n_motions = 64
    n_prev_motions = 16

    L_audio = int(16000 * n_motions / 25)
    d_audio = 768

    N = 5
    feature_dim = 512

    motion_feat = torch.ones((N, n_motions, motion_feat_dim)).to(device)
    prev_motion_feat = torch.ones((N, n_prev_motions, motion_feat_dim)).to(device)

    audio_or_feat = torch.ones((N, L_audio)).to(device)
    prev_audio_feat = torch.ones((N, n_prev_motions, d_audio)).to(device)

    time_step = torch.ones(N, dtype=torch.long).to(device)
    emo_index = torch.zeros(N, dtype=torch.long).to(device)

    model = DitTalkingHead(
        device=device,
        motion_feat_dim=motion_feat_dim,
        n_motions=n_motions,
        n_prev_motions=n_prev_motions,
        feature_dim=feature_dim,
        align_mask_width=2,
    ).to(device)

    out = model(motion_feat, audio_or_feat, prev_motion_feat=prev_motion_feat,
                prev_audio_feat=prev_audio_feat, time_step=time_step,
                indicator=torch.ones(N, n_motions, device=device),
                emo_index=emo_index)
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)
    print('alignment_mask width =', model.denoising_net.align_mask_width)

# ---- Runtime-correction/public compatibility layer (same file) ----

import sys

import torch



class DenoisingNetwork(_CoreDenoisingNetwork):
    """Legacy denoiser with corrected sequence PE and safe indicator handling."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # The denoiser input sequence is exactly [prev_motion, current_motion],
        # i.e. n_prev_motions + n_motions tokens. The diffusion-step embedding
        # is injected separately into DiTDecoder and is NOT concatenated as an
        # extra token. The legacy learnable PE allocated one unused extra token
        # (1 + n_prev_motions + n_motions), which causes 80-vs-81 broadcasting
        # failure as soon as learnable PE is actually enabled by the 0901 scripts.
        if self.use_learnable_pe:
            expected_seq_len = self.n_prev_motions + self.n_motions
            if self.PE.shape[1] != expected_seq_len:
                if self.PE.shape[1] < expected_seq_len:
                    raise ValueError(
                        f"Learnable PE is too short: {self.PE.shape[1]} < {expected_seq_len}"
                    )
                self.PE = torch.nn.Parameter(
                    self.PE[:, :expected_seq_len].detach().clone()
                )

    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None):
        if self.use_indicator and indicator is None:
            indicator = torch.ones(
                motion_feat.shape[:2],
                device=motion_feat.device,
                dtype=motion_feat.dtype,
            )
        return super().forward(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
            step, indicator=indicator,
        )



def _main_args():
    """Return the argparse namespace of a training entrypoint when available."""
    main_module = sys.modules.get("__main__")
    return getattr(main_module, "args", None) if main_module is not None else None


def _resolve_runtime_arg(explicit_value, name, default):
    """Prefer an explicit constructor value, then CLI args, then a safe default."""
    if explicit_value is not None:
        return explicit_value
    args = _main_args()
    value = getattr(args, name, None) if args is not None else None
    return default if value is None else value


class DitTalkingHead(_CoreDitTalkingHead):
    """0803 model with consistent train/inference parameter handling.

    ``audio`` and ``emotion`` remain the same joint condition as in the original
    model. The fixes here are intentionally orthogonal to that core method.
    """

    def __init__(
        self,
        device="cuda",
        target="sample",
        architecture="decoder",
        motion_feat_dim=70,
        fps=25,
        n_motions=64,
        n_prev_motions=16,
        audio_model="hubert",
        feature_dim=512,
        n_diff_steps=500,
        diff_schedule="cosine",
        cfg_mode="incremental",
        guiding_conditions="audio,emotion",
        emo_classes=8,
        align_mask_width=1,
        n_heads=None,
        n_layers=None,
        mlp_ratio=None,
        use_indicator=None,
        no_use_learnable_pe=None,
    ):
        n_heads = _resolve_runtime_arg(n_heads, "n_heads", 8)
        n_layers = _resolve_runtime_arg(n_layers, "n_layers", 8)
        mlp_ratio = _resolve_runtime_arg(mlp_ratio, "mlp_ratio", 4)
        use_indicator = _resolve_runtime_arg(use_indicator, "use_indicator", True)
        no_use_learnable_pe = _resolve_runtime_arg(
            no_use_learnable_pe, "no_use_learnable_pe", False
        )

        # Build the legacy model first so every pre-existing public attribute and
        # checkpoint key remains stable. Replace only the denoiser with the same
        # class constructed from the parameters that the training CLI exposes.
        super().__init__(
            device=device,
            target=target,
            architecture=architecture,
            motion_feat_dim=motion_feat_dim,
            fps=fps,
            n_motions=n_motions,
            n_prev_motions=n_prev_motions,
            audio_model=audio_model,
            feature_dim=feature_dim,
            n_diff_steps=n_diff_steps,
            diff_schedule=diff_schedule,
            cfg_mode=cfg_mode,
            guiding_conditions=guiding_conditions,
            emo_classes=emo_classes,
            align_mask_width=align_mask_width,
        )
        self.denoising_net = DenoisingNetwork(
            device=device,
            motion_feat_dim=motion_feat_dim,
            use_indicator=use_indicator,
            architecture=architecture,
            feature_dim=feature_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            no_use_learnable_pe=no_use_learnable_pe,
            n_prev_motions=n_prev_motions,
            n_motions=n_motions,
            n_diff_steps=n_diff_steps,
        )

        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.use_indicator = bool(use_indicator)
        self.no_use_learnable_pe = bool(no_use_learnable_pe)
        self._pending_prev_audio_raw = None

        # Training scripts save their argparse Namespace into checkpoints. Mark
        # newly trained checkpoints so inference can distinguish them from older
        # checkpoints whose CLI architecture flags existed but were not applied.
        args = _main_args()
        if args is not None:
            args.n_heads = n_heads
            args.n_layers = n_layers
            args.mlp_ratio = mlp_ratio
            args.use_indicator = bool(use_indicator)
            args.no_use_learnable_pe = bool(no_use_learnable_pe)
            args.model_params_propagated = True
            args.context_audio_encoded_once = True
            args.seed = 2026

    def extract_audio_feature(self, audio, frame_num=None):
        """Extract audio features, deferring training-history encoding when possible.

        The continuation branch in the existing training scripts first asks for
        the previous 16-frame feature and then forwards the current 64-frame raw
        audio. During training we cache that previous raw waveform and let
        ``forward`` encode the full 80-frame waveform once, then split features by
        frame index. Evaluation/inference calls retain the normal behavior.
        """
        expected_prev_samples = round(16000 * self.n_prev_motions / self.fps)
        should_defer = (
            self.training
            and audio.ndim == 2
            and frame_num == self.n_prev_motions
            and audio.shape[1] == expected_prev_samples
        )
        if should_defer:
            self._pending_prev_audio_raw = audio
            return torch.zeros(
                audio.shape[0],
                self.n_prev_motions,
                self.feature_dim,
                device=audio.device,
                dtype=audio.dtype,
            )
        return super().extract_audio_feature(audio, frame_num=frame_num)

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
    ):
        # Continuation training: encode [prev 16 + current 64] together once so
        # both slices are produced with identical Wav2Vec2/HuBERT context.
        if (
            self.training
            and audio_or_feat.ndim == 2
            and prev_audio_feat is not None
            and self._pending_prev_audio_raw is not None
        ):
            prev_audio_raw = self._pending_prev_audio_raw
            self._pending_prev_audio_raw = None
            if prev_audio_raw.shape[0] != audio_or_feat.shape[0]:
                raise ValueError("Previous/current audio batch sizes do not match.")
            full_audio = torch.cat([prev_audio_raw, audio_or_feat], dim=1)
            expected_samples = round(
                16000 * (self.n_prev_motions + self.n_motions) / self.fps
            )
            if full_audio.shape[1] != expected_samples:
                raise ValueError(
                    f"Incorrect context audio length {full_audio.shape[1]}, "
                    f"expected {expected_samples}."
                )
            full_audio_feat = super().extract_audio_feature(
                full_audio,
                frame_num=self.n_prev_motions + self.n_motions,
            )
            prev_audio_feat = full_audio_feat[:, : self.n_prev_motions].detach()
            audio_or_feat = full_audio_feat[:, self.n_prev_motions :]

        return super().forward(
            motion_feat,
            audio_or_feat,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            time_step=time_step,
            indicator=indicator,
            emo_index=emo_index,
        )

    @torch.no_grad()
    def sample(
        self,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        motion_at_T=None,
        indicator=None,
        cfg_mode=None,
        cfg_cond=None,
        cfg_scale=1.15,
        flexibility=0,
        dynamic_threshold=None,
        ret_traj=False,
        emo_index=None,
    ):
        # Pre-extract raw acoustic features exactly once. Passing a 3-D tensor to
        # the legacy sampler prevents it from encoding the waveform a second time.
        if audio_or_feat.ndim == 2:
            audio_feat_saved = super().extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            if audio_or_feat.shape[1] != self.n_motions:
                raise ValueError(
                    f"Incorrect audio feature length {audio_or_feat.shape[1]}"
                )
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f"Incorrect audio input shape {audio_or_feat.shape}")

        result = super().sample(
            audio_feat_saved,
            prev_motion_feat=prev_motion_feat,
            prev_audio_feat=prev_audio_feat,
            motion_at_T=motion_at_T,
            indicator=indicator,
            cfg_mode=cfg_mode,
            cfg_cond=cfg_cond,
            cfg_scale=cfg_scale,
            flexibility=flexibility,
            dynamic_threshold=dynamic_threshold,
            ret_traj=ret_traj,
            emo_index=emo_index,
        )
        if ret_traj:
            traj, noise, _ = result
            return traj, noise, audio_feat_saved
        motion_feat, noise, _ = result
        return motion_feat, noise, audio_feat_saved


__all__ = [
    "DiffusionSchedule",
    "DenoisingNetwork",
    "DiTDecoderLayer",
    "DiTDecoder",
    "DitTalkingHead",
]

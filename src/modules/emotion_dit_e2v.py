"""emotion2vec-conditioned DiT for ADEF.

This is a new copy-style implementation.  It does not modify
``src/modules/emotion_dit.py``.  Compared with the original model, this file
factorizes conditions into four paths:

1. HuBERT/Wav2Vec audio content memory for lip-sync and basic motion.
2. Emotion-label basis tokens as a discrete affect anchor.
3. emotion2vec utterance token as global affect/prosody tone.
4. emotion2vec frame memory as local temporal affect dynamics.

The denoiser uses separated cross-attention paths rather than mixing emotion
features into audio features, which is safer for lip synchronization.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import math
import platform

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import PositionalEncoding, enc_dec_mask, pad_audio
from ..config.base_config import make_abs_path
from .emotion_dit import DiffusionSchedule


class Emotion2VecConditionEncoder(nn.Module):
    """Encode label, utterance-level, and frame-level emotion conditions."""

    def __init__(
        self,
        feature_dim: int,
        emo_classes: int = 8,
        e2v_dim: int = 1024,
        num_label_tokens: int = 8,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.emo_classes = emo_classes
        self.e2v_dim = e2v_dim
        self.num_label_tokens = num_label_tokens

        self.label_basis = nn.Parameter(torch.randn(emo_classes, num_label_tokens, feature_dim) * init_std)
        self.null_label_basis = nn.Parameter(torch.zeros(1, num_label_tokens, feature_dim))
        self.label_pos = nn.Parameter(torch.randn(1, num_label_tokens, feature_dim) * init_std)
        self.label_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )

        self.utt_proj = nn.Sequential(
            nn.LayerNorm(e2v_dim),
            nn.Linear(e2v_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.frame_proj = nn.Sequential(
            nn.LayerNorm(e2v_dim),
            nn.Linear(e2v_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.null_utt = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.null_frame = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.global_mod = nn.Sequential(nn.SiLU(), nn.Linear(feature_dim, 2 * feature_dim))

    @staticmethod
    def _as_bool_mask(mask: Optional[torch.Tensor], batch_size: int, device) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        return mask.to(device=device, dtype=torch.bool).view(batch_size)

    def _resize_frame_feat(self, frame_feat: torch.Tensor, frame_len: int) -> torch.Tensor:
        if frame_feat.shape[1] == frame_len:
            return frame_feat
        feat = frame_feat.transpose(1, 2)
        feat = F.interpolate(feat, size=frame_len, mode="linear", align_corners=False)
        return feat.transpose(1, 2).contiguous()

    def encode_label(self, emo_index: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = emo_index.shape[0]
        tokens = self.label_basis[emo_index] + self.label_pos
        tokens = self.label_proj(tokens)
        drop_mask = self._as_bool_mask(drop_mask, B, emo_index.device)
        if drop_mask is not None:
            null_tokens = self.null_label_basis.expand(B, -1, -1)
            tokens = torch.where(drop_mask.view(B, 1, 1), null_tokens, tokens)
        return tokens

    def encode_utterance(
        self,
        emo_utt_feat: Optional[torch.Tensor],
        batch_size: int,
        device,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emo_utt_feat is None:
            token = self.null_utt.expand(batch_size, -1, -1)
        else:
            if emo_utt_feat.ndim == 3:
                emo_utt_feat = emo_utt_feat.squeeze(1)
            token = self.utt_proj(emo_utt_feat.to(device)).unsqueeze(1)
        drop_mask = self._as_bool_mask(drop_mask, batch_size, device)
        if drop_mask is not None:
            token = torch.where(drop_mask.view(batch_size, 1, 1), self.null_utt.expand(batch_size, -1, -1), token)
        return token

    def encode_frame(
        self,
        emo_frame_feat: Optional[torch.Tensor],
        batch_size: int,
        frame_len: int,
        device,
        drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if emo_frame_feat is None:
            tokens = self.null_frame.expand(batch_size, frame_len, -1)
        else:
            emo_frame_feat = emo_frame_feat.to(device)
            if emo_frame_feat.shape[-1] == self.feature_dim:
                tokens = self._resize_frame_feat(emo_frame_feat, frame_len)
            else:
                tokens = self._resize_frame_feat(emo_frame_feat, frame_len)
                tokens = self.frame_proj(tokens)
        drop_mask = self._as_bool_mask(drop_mask, batch_size, device)
        if drop_mask is not None:
            null_tokens = self.null_frame.expand(batch_size, frame_len, -1)
            tokens = torch.where(drop_mask.view(batch_size, 1, 1), null_tokens, tokens)
        return tokens

    def forward(
        self,
        emo_index: torch.Tensor,
        emo_utt_feat: Optional[torch.Tensor],
        emo_frame_feat: Optional[torch.Tensor],
        frame_len: int,
        drop_label: Optional[torch.Tensor] = None,
        drop_utt: Optional[torch.Tensor] = None,
        drop_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = emo_index.shape[0]
        label_tokens = self.encode_label(emo_index, drop_mask=drop_label)
        utt_token = self.encode_utterance(emo_utt_feat, B, emo_index.device, drop_mask=drop_utt)
        gamma, beta = self.global_mod(utt_token).chunk(2, dim=-1)
        label_tokens = label_tokens * (1.0 + gamma) + beta
        frame_tokens = self.encode_frame(emo_frame_feat, B, frame_len, emo_index.device, drop_mask=drop_frame)
        return label_tokens, utt_token, frame_tokens


class E2VCrossDecoderLayer(nn.Module):
    """Decoder layer with four separated condition paths."""

    def __init__(
        self,
        feature_dim: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        audio_scale: float = 1.0,
        label_scale: float = 0.6,
        utt_scale: float = 0.4,
        frame_scale: float = 0.4,
    ):
        super().__init__()
        self.audio_scale = float(audio_scale)
        self.label_scale = float(label_scale)
        self.utt_scale = float(utt_scale)
        self.frame_scale = float(frame_scale)

        self.self_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.audio_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.label_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.utt_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)
        self.frame_attn = nn.MultiheadAttention(feature_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(feature_dim, mlp_ratio * feature_dim)
        self.linear2 = nn.Linear(mlp_ratio * feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        audio_memory: torch.Tensor,
        label_memory: Optional[torch.Tensor] = None,
        utt_memory: Optional[torch.Tensor] = None,
        frame_memory: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        frame_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_update = self.self_attn(hidden, hidden, hidden, need_weights=False)[0]
        hidden = self.norm1(hidden + self.dropout1(self_update))

        audio_update = self.audio_attn(
            query=hidden, key=audio_memory, value=audio_memory, attn_mask=audio_mask, need_weights=False
        )[0]
        cond_update = self.audio_scale * audio_update

        if label_memory is not None:
            label_update = self.label_attn(hidden, label_memory, label_memory, need_weights=False)[0]
            cond_update = cond_update + self.label_scale * label_update
        if utt_memory is not None:
            utt_update = self.utt_attn(hidden, utt_memory, utt_memory, need_weights=False)[0]
            cond_update = cond_update + self.utt_scale * utt_update
        if frame_memory is not None:
            frame_update = self.frame_attn(
                query=hidden, key=frame_memory, value=frame_memory, attn_mask=frame_mask, need_weights=False
            )[0]
            cond_update = cond_update + self.frame_scale * frame_update

        hidden = self.norm2(hidden + self.dropout2(cond_update))
        ff = self.linear2(self.dropout(F.gelu(self.linear1(hidden))))
        hidden = self.norm3(hidden + self.dropout3(ff))
        return hidden


class E2VDenoisingNetwork(nn.Module):
    def __init__(
        self,
        device='cuda',
        motion_feat_dim=70,
        use_indicator=None,
        architecture='decoder',
        feature_dim=512,
        n_heads=8,
        n_layers=8,
        mlp_ratio=4,
        align_mask_width=3,
        no_use_learnable_pe=True,
        n_prev_motions=25,
        n_motions=100,
        n_diff_steps=500,
        decoder_dropout=0.0,
        audio_scale=1.0,
        label_scale=0.6,
        utt_scale=0.4,
        frame_scale=0.4,
    ):
        super().__init__()
        if architecture != 'decoder':
            raise ValueError(f'Unknown architecture: {architecture}')
        self.motion_feat_dim = motion_feat_dim
        self.use_indicator = use_indicator
        self.architecture = architecture
        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.align_mask_width = align_mask_width
        self.use_learnable_pe = not no_use_learnable_pe
        self.n_prev_motions = n_prev_motions
        self.n_motions = n_motions

        self.TE = PositionalEncoding(feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, feature_dim)
        )
        if self.use_learnable_pe:
            self.PE = nn.Parameter(torch.randn(1, n_prev_motions + n_motions, feature_dim))
        else:
            self.PE = PositionalEncoding(feature_dim)

        self.feature_proj = nn.Linear(motion_feat_dim + (1 if self.use_indicator else 0), feature_dim)
        self.layers = nn.ModuleList([
            E2VCrossDecoderLayer(
                feature_dim=feature_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                dropout=decoder_dropout,
                audio_scale=audio_scale,
                label_scale=label_scale,
                utt_scale=utt_scale,
                frame_scale=frame_scale,
            )
            for _ in range(n_layers)
        ])
        if align_mask_width > 0:
            motion_len = n_prev_motions + n_motions
            alignment_mask = enc_dec_mask(
                motion_len, motion_len, frame_width=1, expansion=align_mask_width - 1, device=device
            )
            self.register_buffer('alignment_mask', alignment_mask)
        else:
            self.alignment_mask = None

        self.motion_dec = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2), nn.GELU(), nn.Linear(feature_dim // 2, motion_feat_dim)
        )
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        motion_feat: torch.Tensor,
        audio_feat: torch.Tensor,
        prev_motion_feat: torch.Tensor,
        prev_audio_feat: torch.Tensor,
        step: torch.Tensor,
        indicator: Optional[torch.Tensor] = None,
        label_memory: Optional[torch.Tensor] = None,
        utt_memory: Optional[torch.Tensor] = None,
        frame_memory: Optional[torch.Tensor] = None,
        prev_frame_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not torch.is_tensor(step):
            step = torch.tensor(step, device=self.device, dtype=torch.long)
        step = step.to(self.device).long()
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat([
                torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                indicator,
            ], dim=1).unsqueeze(-1)

        feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        if self.use_indicator:
            if indicator is None:
                indicator = torch.ones(feats_in.shape[:2] + (1,), device=feats_in.device, dtype=feats_in.dtype)
            feats_in = torch.cat([feats_in, indicator], dim=-1)
        hidden = self.feature_proj(feats_in)
        if self.use_learnable_pe:
            hidden = hidden + self.PE[:, :hidden.shape[1], :] + diff_step_embedding
        else:
            hidden = self.PE(hidden) + diff_step_embedding

        audio_memory = torch.cat([prev_audio_feat, audio_feat], dim=1)
        frame_memory_full = None
        if frame_memory is not None:
            if prev_frame_memory is None:
                prev_frame_memory = frame_memory[:, :1].expand(-1, self.n_prev_motions, -1)
            frame_memory_full = torch.cat([prev_frame_memory, frame_memory], dim=1)

        for layer in self.layers:
            hidden = layer(
                hidden,
                audio_memory,
                label_memory=label_memory,
                utt_memory=utt_memory,
                frame_memory=frame_memory_full,
                audio_mask=self.alignment_mask,
                frame_mask=self.alignment_mask,
            )
        return self.motion_dec(hidden)


class DitTalkingHead(nn.Module):
    def __init__(
        self,
        device='cuda',
        target='sample',
        architecture='decoder',
        motion_feat_dim=70,
        fps=25,
        n_motions=100,
        n_prev_motions=25,
        audio_model='hubert',
        feature_dim=512,
        n_diff_steps=500,
        diff_schedule='cosine',
        cfg_mode='incremental',
        guiding_conditions='audio,emotion',
        emo_classes=8,
        e2v_dim=1024,
        num_label_tokens=8,
        n_layers=8,
        n_heads=8,
        mlp_ratio=4,
        align_mask_width=3,
        decoder_dropout=0.0,
        audio_scale=1.0,
        label_scale=0.6,
        utt_scale=0.4,
        frame_scale=0.4,
    ):
        super().__init__()
        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.audio_model = audio_model
        self.cfg_mode = cfg_mode
        guiding_conditions = guiding_conditions.split(',') if guiding_conditions else []
        self.guiding_conditions = [cond for cond in guiding_conditions if cond in ['audio', 'emotion']]

        if self.audio_model == 'wav2vec2':
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh':
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == 'Windows':
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture != 'decoder':
            raise ValueError(f'Unknown architecture {architecture}!')
        self.audio_feature_map = nn.Linear(768, feature_dim)
        self.start_audio_feat = nn.Parameter(torch.randn(emo_classes, n_prev_motions, feature_dim))
        self.start_motion_feat = nn.Parameter(torch.randn(emo_classes, n_prev_motions, motion_feat_dim))
        self.null_audio_feat = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.audio_norm = nn.LayerNorm(feature_dim, eps=1e-9)

        self.condition_encoder = Emotion2VecConditionEncoder(
            feature_dim=feature_dim,
            emo_classes=emo_classes,
            e2v_dim=e2v_dim,
            num_label_tokens=num_label_tokens,
        )
        self.denoising_net = E2VDenoisingNetwork(
            device=device,
            n_motions=n_motions,
            n_prev_motions=n_prev_motions,
            motion_feat_dim=motion_feat_dim,
            feature_dim=feature_dim,
            n_diff_steps=n_diff_steps,
            n_layers=n_layers,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            use_indicator=None,
            decoder_dropout=decoder_dropout,
            audio_scale=audio_scale,
            label_scale=label_scale,
            utt_scale=utt_scale,
            frame_scale=frame_scale,
        )
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)
        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def _get_audio_feature(self, audio_or_feat: torch.Tensor, frame_num: Optional[int] = None) -> torch.Tensor:
        frame_num = frame_num or self.n_motions
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(16000 * frame_num / self.fps), f'Incorrect audio length {audio_or_feat.shape[1]}'
            return self.extract_audio_feature(audio_or_feat, frame_num=frame_num)
        if audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == frame_num, f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            return audio_or_feat
        raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

    def _init_prev_features(self, batch_size, emo_index, prev_motion_feat=None, prev_audio_feat=None):
        if prev_motion_feat is None:
            prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)
        if prev_audio_feat is None:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
        return prev_motion_feat, prev_audio_feat

    def _make_cfg_train_masks(self, batch_size: int):
        """Four-stage dropout for null/audio/audio+label/full conditions."""
        r = torch.rand(batch_size, device=self.device)
        p_uncond = 0.10
        p_audio_only = 0.35
        p_audio_label = 0.20
        mask_audio = r < p_uncond
        mask_label = r < (p_uncond + p_audio_only)
        mask_utt = r < (p_uncond + p_audio_only + p_audio_label)
        mask_frame = mask_utt.clone()
        return mask_audio, mask_label, mask_utt, mask_frame

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(pad_audio(audio), self.fps, frame_num=frame_num * 2).last_hidden_state
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = F.interpolate(hidden_states, size=frame_num, align_corners=False, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)
        return self.audio_feature_map(hidden_states)

    def _encode_conditions(
        self,
        emo_index,
        emo_utt_feat,
        emo_frame_feat,
        drop_label=None,
        drop_utt=None,
        drop_frame=None,
    ):
        return self.condition_encoder(
            emo_index,
            emo_utt_feat=emo_utt_feat,
            emo_frame_feat=emo_frame_feat,
            frame_len=self.n_motions,
            drop_label=drop_label,
            drop_utt=drop_utt,
            drop_frame=drop_frame,
        )

    def _prepare_prev_frame_tokens(self, prev_emo_frame_feat, batch_size):
        if prev_emo_frame_feat is None:
            return None
        if prev_emo_frame_feat.shape[-1] == self.feature_dim:
            return self.condition_encoder._resize_frame_feat(prev_emo_frame_feat.to(self.device), self.n_prev_motions)
        return self.condition_encoder.encode_frame(
            prev_emo_frame_feat,
            batch_size=batch_size,
            frame_len=self.n_prev_motions,
            device=self.device,
            drop_mask=None,
        )

    def forward(
        self,
        motion_feat,
        audio_or_feat,
        prev_motion_feat=None,
        prev_audio_feat=None,
        time_step=None,
        indicator=None,
        emo_index=None,
        emo_utt_feat=None,
        emo_frame_feat=None,
        prev_emo_frame_feat=None,
    ):
        batch_size = motion_feat.shape[0]
        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_feat = audio_feat_saved.clone()
        prev_motion_feat, prev_audio_feat = self._init_prev_features(batch_size, emo_index, prev_motion_feat, prev_audio_feat)

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)
        if not torch.is_tensor(time_step):
            time_step = torch.tensor(time_step, device=self.device, dtype=torch.long)
        else:
            time_step = time_step.to(self.device).long()

        mask_audio = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_label = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_utt = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        mask_frame = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        if 'audio' in self.guiding_conditions and 'emotion' in self.guiding_conditions and self.cfg_mode == 'incremental':
            mask_audio, mask_label, mask_utt, mask_frame = self._make_cfg_train_masks(batch_size)
        else:
            if 'audio' in self.guiding_conditions:
                mask_audio = torch.rand(batch_size, device=self.device) < 0.1
            if 'emotion' in self.guiding_conditions:
                mask_label = torch.rand(batch_size, device=self.device) < 0.5
                mask_utt = mask_label.clone()
                mask_frame = mask_label.clone()

        if 'audio' in self.guiding_conditions:
            audio_feat = torch.where(
                mask_audio.view(-1, 1, 1),
                self.null_audio_feat.expand(batch_size, self.n_motions, -1),
                audio_feat,
            )
        audio_feat = self.audio_norm(audio_feat)
        prev_audio_feat = self.audio_norm(prev_audio_feat)

        label_memory = utt_memory = frame_memory = None
        if 'emotion' in self.guiding_conditions:
            label_memory, utt_memory, frame_memory = self._encode_conditions(
                emo_index, emo_utt_feat, emo_frame_feat,
                drop_label=mask_label, drop_utt=mask_utt, drop_frame=mask_frame,
            )
        prev_frame_memory = self._prepare_prev_frame_tokens(prev_emo_frame_feat, batch_size)

        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        motion_feat_target = self.denoising_net(
            motion_feat_noisy, audio_feat, prev_motion_feat, prev_audio_feat, time_step,
            indicator, label_memory=label_memory, utt_memory=utt_memory,
            frame_memory=frame_memory, prev_frame_memory=prev_frame_memory,
        )
        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach(), frame_memory.detach() if frame_memory is not None else None

    @staticmethod
    def _scale_at_step(max_scale: float, min_scale: float, t: int, num_steps: int, schedule: Optional[str]):
        if schedule is None or schedule == 'none' or num_steps <= 1:
            return float(max_scale)
        progress = (num_steps - float(t)) / float(num_steps - 1)
        progress = max(0.0, min(1.0, progress))
        if schedule == 'linear':
            w = progress
        elif schedule == 'cosine':
            w = 0.5 - 0.5 * math.cos(math.pi * progress)
        elif schedule == 'bell':
            w = math.sin(math.pi * progress)
        else:
            raise ValueError(f'Unknown cfg_schedule {schedule}')
        return float(min_scale) + w * (float(max_scale) - float(min_scale))

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
        emo_utt_feat=None,
        emo_frame_feat=None,
        prev_emo_frame_feat=None,
        cfg_min: Optional[Sequence[float]] = None,
        cfg_schedule: Optional[str] = None,
    ):
        batch_size = audio_or_feat.shape[0]
        cfg_mode = cfg_mode or self.cfg_mode
        cfg_cond = cfg_cond or self.guiding_conditions
        cfg_cond = [c for c in cfg_cond if c in ['audio', 'emotion']]
        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        if cfg_min is None:
            cfg_min = [1.0 if c == 'audio' else 0.0 for c in cfg_cond]
        if len(cfg_cond) > 0:
            cfg_cond, cfg_scale, cfg_min = zip(*sorted(
                zip(cfg_cond, cfg_scale, cfg_min), key=lambda x: ['audio', 'emotion'].index(x[0])
            ))
        else:
            cfg_cond, cfg_scale, cfg_min = [], [], []

        audio_feat_saved = self._get_audio_feature(audio_or_feat)
        audio_real = self.audio_norm(audio_feat_saved.clone())
        prev_motion_feat, prev_audio_feat = self._init_prev_features(batch_size, emo_index, prev_motion_feat, prev_audio_feat)
        prev_audio_real = self.audio_norm(prev_audio_feat)
        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim), device=self.device)

        audio_null = self.audio_norm(self.null_audio_feat.expand(batch_size, self.n_motions, -1)) if 'audio' in cfg_cond else audio_real
        if 'audio' in cfg_cond and 'emotion' in cfg_cond:
            audio_feat_in = torch.cat([audio_null, audio_real, audio_real], dim=0)
            prev_audio_feat_in = torch.cat([prev_audio_real, prev_audio_real, prev_audio_real], dim=0)
            n_entries = 3
        elif 'audio' in cfg_cond:
            audio_feat_in = torch.cat([audio_null, audio_real], dim=0)
            prev_audio_feat_in = torch.cat([prev_audio_real, prev_audio_real], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_real
            prev_audio_feat_in = prev_audio_real
            n_entries = 1

        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        indicator_in = torch.cat([indicator] * n_entries, dim=0) if indicator is not None else None
        prev_frame_real = self._prepare_prev_frame_tokens(prev_emo_frame_feat, batch_size)
        prev_frame_in = torch.cat([prev_frame_real] * n_entries, dim=0) if prev_frame_real is not None else None

        label_in = utt_in = frame_in = None
        if 'emotion' in cfg_cond:
            drop_true = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            drop_false = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            label_null, utt_null, frame_null = self._encode_conditions(
                emo_index, emo_utt_feat, emo_frame_feat,
                drop_label=drop_true, drop_utt=drop_true, drop_frame=drop_true,
            )
            label_real, utt_real, frame_real = self._encode_conditions(
                emo_index, emo_utt_feat, emo_frame_feat,
                drop_label=drop_false, drop_utt=drop_false, drop_frame=drop_false,
            )
            if 'audio' in cfg_cond:
                label_in = torch.cat([label_null, label_null, label_real], dim=0)
                utt_in = torch.cat([utt_null, utt_null, utt_real], dim=0)
                frame_in = torch.cat([frame_null, frame_null, frame_real], dim=0)
            else:
                label_in = label_real
                utt_in = utt_real
                frame_in = frame_real

        traj = {self.diffusion_sched.num_steps: motion_at_T}
        for t in range(self.diffusion_sched.num_steps, 0, -1):
            z = torch.randn_like(motion_at_T) if t > 1 else torch.zeros_like(motion_at_T)
            alpha = self.diffusion_sched.alphas[t]
            alpha_bar = self.diffusion_sched.alpha_bars[t]
            alpha_bar_prev = self.diffusion_sched.alpha_bars[t - 1]
            sigma = self.diffusion_sched.get_sigmas(t, flexibility)

            motion_at_t = traj[t]
            motion_in = torch.cat([motion_at_t] * n_entries, dim=0)
            step_base = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            step_in = torch.cat([step_base] * n_entries, dim=0)

            results = self.denoising_net(
                motion_in, audio_feat_in, prev_motion_feat_in, prev_audio_feat_in, step_in,
                indicator_in, label_memory=label_in, utt_memory=utt_in,
                frame_memory=frame_in, prev_frame_memory=prev_frame_in,
            )
            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            chunks = results.chunk(n_entries)
            if n_entries == 3:
                uncond = chunks[0][:, -self.n_motions:]
                audio_only = chunks[1][:, -self.n_motions:]
                audio_emo = chunks[2][:, -self.n_motions:]
                audio_scale = self._scale_at_step(cfg_scale[0], cfg_min[0], t, self.diffusion_sched.num_steps, cfg_schedule)
                emo_scale = self._scale_at_step(cfg_scale[1], cfg_min[1], t, self.diffusion_sched.num_steps, cfg_schedule)
                target_theta = uncond + audio_scale * (audio_only - uncond) + emo_scale * (audio_emo - audio_only)
            elif n_entries == 2:
                uncond = chunks[0][:, -self.n_motions:]
                cond = chunks[1][:, -self.n_motions:]
                scale = self._scale_at_step(cfg_scale[0], cfg_min[0], t, self.diffusion_sched.num_steps, cfg_schedule)
                target_theta = uncond + scale * (cond - uncond)
            else:
                target_theta = chunks[0][:, -self.n_motions:]

            if self.target == 'noise':
                c0 = 1 / torch.sqrt(alpha)
                c1 = (1 - alpha) / torch.sqrt(1 - alpha_bar)
                motion_next = c0 * (motion_at_t - c1 * target_theta) + sigma * z
            elif self.target == 'sample':
                c0 = (1 - alpha_bar_prev) * torch.sqrt(alpha) / (1 - alpha_bar)
                c1 = (1 - alpha) * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
                motion_next = c0 * motion_at_t + c1 * target_theta + sigma * z
            else:
                raise ValueError(f'Unknown target type: {self.target}')

            traj[t - 1] = motion_next.detach()
            traj[t] = traj[t].cpu()
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj, motion_at_T, audio_feat_saved
        return traj[0], motion_at_T, audio_feat_saved

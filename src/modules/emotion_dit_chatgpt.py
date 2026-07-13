# coding: utf-8

"""
Merged & ADEF_remake-adapted port of JoyVASA-emotion-audio's
``src/modules/dit_talking_head.py``.

This file was produced by merging the JoyVASA DiT module together with all of
its direct functional dependencies into a single self-contained source file so
that it can replace :mod:`src.modules.emotion_dit` inside the ADEF_remake
project without requiring any new sibling modules.

Inlined dependencies (originally imported from sibling modules):
    - ``PositionalEncoding``, ``enc_dec_mask``, ``pad_audio`` from
      ``src/modules/common.py`` (the ADEF_remake copy of common.py had a
      ``self.pe[:, x.shape[1], :]`` indexing bug -- the fixed variant from
      JoyVASA is used here).
    - ``make_abs_path`` from ``src/config/base_config.py``.
    - ``parse_emotion_labels`` / ``emotion_to_id`` / ``build_emotion_to_id`` /
      ``normalize_emotion_name`` from ``src/utils/emotion.py``.
    - ``DEFAULT_EMOTION_LABELS`` from ``src/config/emotion_config.py``. The
      ADEF_remake project exposes an ``8``-element list at
      ``src.config.emotion_config.global_emo_list`` -- that list is reused
      here as the default emotion vocabulary, which keeps the index space
      compatible with the existing ADEF_remake callers (train.py /
      train_new.py / helper.py / ADEF_wrapper.py all pass ``emo_index``
      values that index into this list).

Interface adaptations for ADEF_remake compatibility:
    * ``__init__`` keeps every keyword argument that the original
      ``emotion_dit.DittoTalkingHead.__init__`` accepted (``device``,
      ``target``, ``architecture``, ``motion_feat_dim``, ``fps``,
      ``n_motions``, ``n_prev_motions``, ``audio_model``, ``feature_dim``,
      ``n_diff_steps``, ``diff_schedule``, ``cfg_mode``, ``guiding_conditions``)
      and additionally accepts ``emo_classes`` as a deprecated alias for
      ``num_emotions`` and ``emotion_labels`` / ``emotion_intensity`` /
      ``condition_dropout_prob`` / ``emotion_condition_mode`` from JoyVASA.
    * ``forward`` and ``sample`` both accept the legacy ``emo_index`` kwarg
      (a LongTensor of emotion ids into ``global_emo_list``) in addition to
      the JoyVASA-style ``emotion`` / ``emotion_id`` / ``emotion_intensity``
      arguments.
    * ``self.start_audio_feat`` and ``self.start_motion_feat`` keep the
      ``(num_emotions, n_prev_motions, dim)`` shape used by the ADEF_remake
      checkpoint so existing checkpoints load with matching parameter
      shapes. ``prev_audio_feat`` / ``prev_motion_feat`` are looked up via
      ``torch.index_select(self.start_*, 0, emo_index)`` when ``None``.
    * ``DenoisingNetwork.PE`` keeps the ``(1, 1 + n_prev_motions +
      n_motions, feature_dim)`` shape used by the ADEF_remake checkpoint
      (the leading +1 slot was reserved for an unused person/global token
      in the original architecture and is preserved for backward
      compatibility).
    * ``prev_audio_feat`` is *not* nulled-out per CFG branch in
      ``forward``/``sample`` -- it is modulated once with the emotion
      condition and then replicated across CFG branches, matching the
      behavior of the original ``emotion_dit.DittoTalkingHead``. Only the
      current-window ``audio_feat`` participates in the per-branch
      classifier-free-guidence masking.
"""

from __future__ import annotations

import math
import platform
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Inlined from src/config/base_config.py
# ---------------------------------------------------------------------------

def _make_abs_path(fn: str) -> str:
    """Resolve ``fn`` relative to this file (replacement for the sibling
    ``make_abs_path`` that lives in ``src/config/base_config.py``)."""
    import os.path as osp
    return osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), fn))


# ---------------------------------------------------------------------------
# Inlined from src/modules/common.py (with the indexing bug fixed)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 600):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NOTE: ADEF_remake's original PositionalEncoding had
        # ``self.pe[:, x.shape[1], :]`` which silently zero-positions the
        # last frame. The JoyVASA fixed variant uses slicing.
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


def enc_dec_mask(T: int, S: int, frame_width: int = 2, expansion: int = 0,
                 device: Union[str, torch.device] = 'cuda') -> torch.Tensor:
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, max(0, (i - expansion) * frame_width):(i + expansion + 1) * frame_width] = 0
    return (mask == 1).to(device=device)


def pad_audio(audio: torch.Tensor, audio_unit: int = 320,
              pad_threshold: int = 80) -> torch.Tensor:
    batch_size, audio_len = audio.shape
    n_units = audio_len // audio_unit
    side_len = math.ceil((audio_unit * n_units + pad_threshold - audio_len) / 2)
    if side_len >= 0:
        reflect_len = side_len // 2
        replicate_len = side_len % 2
        if reflect_len > 0:
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
            audio = F.pad(audio, (reflect_len, reflect_len), mode='reflect')
        if replicate_len > 0:
            audio = F.pad(audio, (1, 1), mode='replicate')
    return audio


# ---------------------------------------------------------------------------
# Inlined from src/config/emotion_config.py (ADEF_remake vocabulary)
# ---------------------------------------------------------------------------

# The canonical 8-emotion vocabulary used by ADEF_remake. Index = emo_index
# value passed by the training / inference scripts. Keep this list stable
# between training and inference -- changing the order silently remaps
# every existing checkpoint.
DEFAULT_EMOTION_LABELS: Tuple[str, ...] = (
    'angry',       # 0
    'contempt',    # 1
    'disgusted',   # 2
    'fear',        # 3
    'happy',       # 4
    'neutral',     # 5
    'sad',         # 6
    'surprised',   # 7
)
DEFAULT_EMOTION_LABELS_CSV: str = ",".join(DEFAULT_EMOTION_LABELS)


# Try to fall back to the project-wide emotion config when available so that
# callers editing ``src/config/emotion_config.py`` see consistent results.
try:
    from ..config.emotion_config import global_emo_list as _PROJECT_EMO_LIST  # type: ignore
    if list(_PROJECT_EMO_LIST) != list(DEFAULT_EMOTION_LABELS):
        DEFAULT_EMOTION_LABELS = tuple(_PROJECT_EMO_LIST)
        DEFAULT_EMOTION_LABELS_CSV = ",".join(DEFAULT_EMOTION_LABELS)
except Exception:  # pragma: no cover - tolerate missing import path
    pass


# ---------------------------------------------------------------------------
# Inlined from src/utils/emotion.py (adapted for ADEF_remake vocabulary)
# ---------------------------------------------------------------------------

_EMOTION_ALIASES: dict = {
    "none": "neutral",
    "calm": "neutral",
    "normal": "neutral",
    "neutrality": "neutral",
    "中性": "neutral",
    "平静": "neutral",
    "自然": "neutral",
    "开心": "happy",
    "高兴": "happy",
    "快乐": "happy",
    "喜悦": "happy",
    "joy": "happy",
    "joyful": "happy",
    "smile": "happy",
    "smiling": "happy",
    "sadness": "sad",
    "悲伤": "sad",
    "难过": "sad",
    "伤心": "sad",
    "angry": "angry",
    "anger": "angry",
    "生气": "angry",
    "愤怒": "angry",
    "surprise": "surprised",
    "surprised": "surprised",
    "惊讶": "surprised",
    "惊喜": "surprised",
    "fearful": "fear",
    "fear": "fear",
    "害怕": "fear",
    "恐惧": "fear",
    "disgusted": "disgusted",
    "disgust": "disgusted",
    "厌恶": "disgusted",
    "嫌弃": "disgusted",
    "contemptuous": "contempt",
    "contempt": "contempt",
    "轻蔑": "contempt",
}


def parse_emotion_labels(labels: Optional[Union[str, Sequence[str]]] = None) -> List[str]:
    """Return a clean, ordered list of emotion label names."""
    if labels is None:
        parsed = list(DEFAULT_EMOTION_LABELS)
    elif isinstance(labels, str):
        parsed = [item.strip() for item in labels.split(",") if item.strip()]
    else:
        parsed = [str(item).strip() for item in labels if str(item).strip()]
    if not parsed:
        raise ValueError("emotion label list is empty")
    return parsed


def build_emotion_to_id(labels: Optional[Union[str, Sequence[str]]] = None) -> dict:
    """Build a lookup table containing canonical labels and known aliases."""
    labels_list = parse_emotion_labels(labels)
    canonical_to_id = {label.lower(): idx for idx, label in enumerate(labels_list)}
    mapping = dict(canonical_to_id)
    for alias, canonical in _EMOTION_ALIASES.items():
        if canonical.lower() in canonical_to_id:
            mapping[alias.lower()] = canonical_to_id[canonical.lower()]
    return mapping


def normalize_emotion_name(emotion: Optional[Union[str, int]],
                           labels: Optional[Union[str, Sequence[str]]] = None) -> str:
    labels_list = parse_emotion_labels(labels)
    if emotion is None:
        return labels_list[0]
    if isinstance(emotion, int):
        if emotion < 0 or emotion >= len(labels_list):
            raise ValueError(
                f"emotion id {emotion} is outside [0, {len(labels_list) - 1}]"
            )
        return labels_list[emotion]
    emotion_text = str(emotion).strip()
    if emotion_text.isdigit():
        return normalize_emotion_name(int(emotion_text), labels_list)
    emotion_key = emotion_text.lower()
    emotion_key = _EMOTION_ALIASES.get(emotion_key, emotion_key)
    mapping = {label.lower(): label for label in labels_list}
    if emotion_key not in mapping:
        raise ValueError(
            f"Unknown emotion '{emotion}'. Available emotions: {', '.join(labels_list)}"
        )
    return mapping[emotion_key]


def emotion_to_id(
    emotion: Optional[Union[str, int, Sequence[Union[str, int]], torch.Tensor]],
    labels: Optional[Union[str, Sequence[str]]] = None,
    *,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Convert emotion names/ids to a long tensor."""
    labels_list = parse_emotion_labels(labels)
    mapping = build_emotion_to_id(labels_list)

    if emotion is None:
        ids = [0]
    elif isinstance(emotion, torch.Tensor):
        return emotion.to(device=device, dtype=torch.long)
    elif isinstance(emotion, (list, tuple)):
        ids = []
        for item in emotion:
            if isinstance(item, int):
                idx = item
            else:
                text = str(item).strip()
                if text.isdigit():
                    idx = int(text)
                else:
                    key = _EMOTION_ALIASES.get(text.lower(), text.lower())
                    if key not in mapping:
                        raise ValueError(
                            f"Unknown emotion '{item}'. Available emotions: {', '.join(labels_list)}"
                        )
                    idx = mapping[key]
            if idx < 0 or idx >= len(labels_list):
                raise ValueError(f"emotion id {idx} is outside [0, {len(labels_list) - 1}]")
            ids.append(idx)
    else:
        text = str(emotion).strip()
        if text.isdigit():
            idx = int(text)
        else:
            key = _EMOTION_ALIASES.get(text.lower(), text.lower())
            if key not in mapping:
                raise ValueError(
                    f"Unknown emotion '{emotion}'. Available emotions: {', '.join(labels_list)}"
                )
            idx = mapping[key]
        if idx < 0 or idx >= len(labels_list):
            raise ValueError(f"emotion id {idx} is outside [0, {len(labels_list) - 1}]")
        ids = [idx]

    return torch.tensor(ids, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Core DiT modules
# ---------------------------------------------------------------------------

class DiffusionSchedule(nn.Module):
    def __init__(self, num_steps: int, mode: str = 'linear', beta_1: float = 1e-4,
                 beta_T: float = 0.02, s: float = 0.008):
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

    def uniform_sample_t(self, batch_size: int) -> list:
        ts = torch.randint(1, self.num_steps + 1, (batch_size,))
        return ts.tolist()

    def get_sigmas(self, t, flexibility: float = 0):
        assert 0 <= flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class DenoisingNetwork(nn.Module):
    def __init__(self, device='cuda', motion_feat_dim: int = 76,
                 use_indicator=None, architecture: str = "decoder",
                 feature_dim: int = 512, n_heads: int = 8,
                 n_layers: int = 8, mlp_ratio: int = 4,
                 align_mask_width: int = 1, no_use_learnable_pe: bool = True,
                 n_prev_motions: int = 10,
                 n_motions: int = 100, n_diff_steps: int = 500):
        super().__init__()

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

        self.TE = PositionalEncoding(self.feature_dim, max_len=n_diff_steps + 1)
        self.diff_step_map = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if self.use_learnable_pe:
            # NOTE: original ADEF_remake layout was
            # ``(1, 1 + n_prev_motions + n_motions, feature_dim)`` -- the
            # leading +1 slot was reserved for an unused person/global
            # feature. Preserved here so the existing checkpoint loads.
            self.PE = nn.Parameter(torch.randn(1, 1 + self.n_prev_motions + self.n_motions,
                                               self.feature_dim))
        else:
            self.PE = PositionalEncoding(self.feature_dim)

        if self.architecture == 'decoder':
            self.feature_proj = nn.Linear(
                self.motion_feat_dim + (1 if self.use_indicator else 0),
                self.feature_dim,
            )
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.feature_dim,
                nhead=self.n_heads,
                dim_feedforward=self.mlp_ratio * self.feature_dim,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=self.n_layers)
            if self.align_mask_width > 0:
                motion_len = self.n_prev_motions + self.n_motions
                alignment_mask = enc_dec_mask(motion_len, motion_len,
                                              frame_width=1,
                                              expansion=self.align_mask_width - 1)
                self.register_buffer('alignment_mask', alignment_mask)
            else:
                self.alignment_mask = None
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

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
        """
        Args:
            motion_feat: (N, L, d_motion). Noisy motion feature
            audio_feat: (N, L, feature_dim). Audio memory (already fused with emotion condition).
            prev_motion_feat: (N, L_p, d_motion). Padded previous motion feature.
            prev_audio_feat: (N, L_p, feature_dim). Padded previous audio memory.
            step: (N,) diffusion time step.
            indicator: (N, L) 0/1 indicator for the real (unpadded) motion feature.

        Returns:
            motion_feat_target: (N, L_p + L, d_motion)
        """
        diff_step_embedding = self.diff_step_map(self.TE.pe[0, step]).unsqueeze(1)

        if indicator is not None:
            indicator = torch.cat(
                [torch.zeros((indicator.shape[0], self.n_prev_motions), device=indicator.device),
                 indicator], dim=1)
            indicator = indicator.unsqueeze(-1)

        if self.architecture == 'decoder':
            feats_in = torch.cat([prev_motion_feat, motion_feat], dim=1)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')
        if self.use_indicator:
            feats_in = torch.cat([feats_in, indicator], dim=-1)

        feats_in = self.feature_proj(feats_in)

        if self.use_learnable_pe:
            feats_in = feats_in + self.PE + diff_step_embedding
        else:
            feats_in = self.PE(feats_in) + diff_step_embedding

        if self.architecture == 'decoder':
            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
            feat_out = self.transformer(feats_in, audio_feat_in,
                                        memory_mask=self.alignment_mask)
        else:
            raise ValueError(f'Unknown architecture: {self.architecture}')

        motion_feat_target = self.motion_dec(feat_out)
        return motion_feat_target


class DitTalkingHead(nn.Module):
    """ADEF_remake-compatible port of JoyVASA's ``DitTalkingHead``.

    The signature is a strict superset of the original ADEF_remake
    ``DitTalkingHead``: every keyword that ``train.py`` /
    ``train_new.py`` / ``helper.py`` pass to the constructor is honored,
    and every keyword that ``train*.py`` / ``ADEF_wrapper.py`` pass to
    ``forward`` / ``sample`` (notably the legacy ``emo_index``) is also
    accepted.
    """

    def __init__(self, device: str = 'cuda', target: str = "sample",
                 architecture: str = "decoder",
                 motion_feat_dim: int = 76, fps: int = 25, n_motions: int = 100,
                 n_prev_motions: int = 10,
                 audio_model: str = "hubert", feature_dim: int = 512,
                 n_diff_steps: int = 500, diff_schedule: str = "cosine",
                 cfg_mode: str = "incremental",
                 guiding_conditions: str = "audio,emotion",
                 # --- ADEF_remake legacy / JoyVASA extensions ---
                 emotion_labels: Optional[Union[str, Sequence[str]]] = None,
                 num_emotions: Optional[int] = None,
                 emo_classes: Optional[int] = None,           # ADEF_remake alias
                 emotion_condition_mode: str = "add",
                 condition_dropout_prob: float = 0.1,
                 audio_dropout_prob: Optional[float] = None,  # ADEF_remake legacy override
                 emotion_dropout_prob: Optional[float] = None,  # ADEF_remake legacy override
                 use_indicator=None, n_heads: int = 8,
                 n_layers: int = 8, mlp_ratio: int = 4,
                 align_mask_width: int = 1, no_use_learnable_pe: bool = True):
        super().__init__()

        # Model parameters
        self.target = target
        self.architecture = architecture
        self.motion_feat_dim = motion_feat_dim
        self.fps = fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.feature_dim = feature_dim
        self.use_indicator = use_indicator
        self.condition_dropout_prob = condition_dropout_prob
        # Per-condition dropout probs (preserved for backward compatibility
        # with the ADEF_remake training script which used p_AE=0.1 audio /
        # p_E=0.55 emotion). When ``audio_dropout_prob`` / ``emotion_dropout_prob``
        # are explicitly passed, they override ``condition_dropout_prob``
        # for the respective mask.
        self.audio_dropout_prob = audio_dropout_prob if audio_dropout_prob is not None else condition_dropout_prob
        self.emotion_dropout_prob = emotion_dropout_prob if emotion_dropout_prob is not None else condition_dropout_prob
        self.emotion_condition_mode = emotion_condition_mode

        # Audio encoder
        self.audio_model = audio_model
        if self.audio_model == 'wav2vec2':
            print("using wav2vec2 audio encoder ...")
            # NOTE: ADEF_remake ships a local ``Wav2Vec2Model`` subclass in
            # ``src/modules/wav2vec2.py`` that overrides ``forward`` to
            # accept a ``frame_num`` argument (the HuggingFace version
            # does not). Importing the HF version here would crash with
            # ``TypeError: Wav2Vec2Model.forward() got an unexpected
            # keyword argument 'frame_num'`` at the first sample.
            from .wav2vec2 import Wav2Vec2Model
            self.audio_encoder = Wav2Vec2Model.from_pretrained(
                _make_abs_path('../../pretrained_weights/wav2vec2-base-960h'))
            self.audio_encoder.feature_extractor._freeze_parameters()
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
            self.audio_encoder.eval()
        elif self.audio_model == 'hubert':
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(
                _make_abs_path('../../pretrained_weights/hubert-base-ls960'))
            self.audio_encoder.feature_extractor._freeze_parameters()
        elif self.audio_model == 'hubert_zh_ori' or self.audio_model == 'hubert_zh':
            print("using hubert chinese ori")
            model_path = '../../pretrained_weights/TencentGameMate:chinese-hubert-base'
            if platform.system() == "Windows":
                model_path = '../../pretrained_weights/chinese-hubert-base'
            from .hubert import HubertModel
            self.audio_encoder = HubertModel.from_pretrained(_make_abs_path(model_path))
            self.audio_encoder.feature_extractor._freeze_parameters()
        else:
            raise ValueError(f'Unknown audio model {self.audio_model}!')

        if architecture == 'decoder':
            self.audio_feature_map = nn.Linear(768, feature_dim)
        else:
            raise ValueError(f'Unknown architecture {architecture}!')

        # Emotion vocabulary -------------------------------------------------
        self.emotion_labels = parse_emotion_labels(emotion_labels)
        if num_emotions is None and emo_classes is not None:
            num_emotions = emo_classes
        self.num_emotions = int(num_emotions if num_emotions is not None else len(self.emotion_labels))
        if self.num_emotions < len(self.emotion_labels):
            raise ValueError(
                f"num_emotions={self.num_emotions} is smaller than "
                f"emotion label count={len(self.emotion_labels)}"
            )
        if self.emotion_condition_mode != "add":
            raise ValueError(
                "Only emotion_condition_mode='add' is supported by the current alignment mask."
            )
        self.emotion_embedding = nn.Embedding(self.num_emotions, feature_dim)
        self.emotion_feature_map = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

        # ------------------------------------------------------------------
        # ADEF_remake checkpoint compatibility: the original ADEF_remake
        # ``DitTalkingHead`` stored ``start_audio_feat`` and
        # ``start_motion_feat`` with shape
        # ``(num_emotions, n_prev_motions, dim)`` and indexed them via
        # ``torch.index_select(self.start_audio_feat, 0, emo_index)`` when
        # the caller did not supply a previous feature. We keep the same
        # layout so an existing ADEF_remake checkpoint loads with matching
        # tensor shapes for ``start_audio_feat`` and ``start_motion_feat``.
        # ------------------------------------------------------------------
        self.start_audio_feat = nn.Parameter(
            torch.randn(self.num_emotions, self.n_prev_motions, feature_dim)
        )
        self.start_motion_feat = nn.Parameter(
            torch.randn(self.num_emotions, self.n_prev_motions, self.motion_feat_dim)
        )

        # Diffusion model
        self.denoising_net = DenoisingNetwork(
            device=device,
            n_motions=self.n_motions,
            n_prev_motions=self.n_prev_motions,
            motion_feat_dim=self.motion_feat_dim,
            feature_dim=feature_dim,
            use_indicator=use_indicator,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            align_mask_width=align_mask_width,
            no_use_learnable_pe=no_use_learnable_pe,
            n_diff_steps=n_diff_steps,
        )
        self.diffusion_sched = DiffusionSchedule(n_diff_steps, diff_schedule)

        # Classifier-free guidance settings
        self.cfg_mode = cfg_mode
        self.guiding_conditions = self._normalize_cfg_conditions(guiding_conditions)
        # null_audio_feat / null_emotion_feat shapes: keep the ADEF_remake
        # convention of ``(1, 1, feature_dim)`` and zero-initialize the
        # emotion null so that ad-hoc audio-only checkpoints do not
        # accidentally consume random noise as an emotion bias.
        self.null_audio_feat = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.null_emotion_feat = nn.Parameter(torch.zeros(1, 1, feature_dim))

        self.to(device)

    # ------------------------------------------------------------------
    # Properties / helpers
    # ------------------------------------------------------------------
    @property
    def device(self):
        return next(self.parameters()).device

    def _normalize_cfg_conditions(self, conditions) -> List[str]:
        if conditions is None:
            conditions = self.guiding_conditions if hasattr(self, 'guiding_conditions') else []
        if isinstance(conditions, str):
            conditions = [cond.strip() for cond in conditions.split(',')]
        conditions = [cond for cond in conditions if cond in ['audio', 'emotion']]
        order = {'audio': 0, 'emotion': 1}
        return sorted(dict.fromkeys(conditions), key=lambda item: order[item])

    def _cfg_branches(self, cfg_cond, cfg_mode):
        """Return ordered condition branches for CFG.

        incremental with [audio, emotion]: {}, {audio}, {audio, emotion}
        independent with [audio, emotion]: {}, {audio}, {emotion}
        """
        cfg_cond = self._normalize_cfg_conditions(cfg_cond)
        if not cfg_cond:
            return [set()]
        if cfg_mode == 'incremental':
            branches = [set()]
            running = set()
            for cond in cfg_cond:
                running = running | {cond}
                branches.append(set(running))
            return branches
        if cfg_mode == 'independent':
            return [set()] + [{cond} for cond in cfg_cond]
        raise NotImplementedError(f'Unknown cfg_mode {cfg_mode}')

    def _expand_null_audio(self, batch_size: int, frame_num: int) -> torch.Tensor:
        return self.null_audio_feat.expand(batch_size, frame_num, -1)

    def _expand_null_emotion(self, batch_size: int, frame_num: int) -> torch.Tensor:
        return self.null_emotion_feat.expand(batch_size, frame_num, -1)

    def _as_batch_mask(self, mask, batch_size: int) -> Optional[torch.Tensor]:
        if mask is None or mask is False:
            return None
        if mask is True:
            return torch.ones(batch_size, dtype=torch.bool, device=self.device)
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device=self.device, dtype=torch.bool)
        else:
            mask = torch.as_tensor(mask, device=self.device, dtype=torch.bool)
        if mask.ndim == 0:
            mask = mask.expand(batch_size)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size)
        if mask.shape[0] != batch_size:
            raise ValueError(f"mask batch size {mask.shape[0]} does not match {batch_size}")
        return mask

    def _select_audio_feature(self, audio_feat, null_mask=None):
        batch_size, frame_num = audio_feat.shape[:2]
        null_feat = self._expand_null_audio(batch_size, frame_num)
        mask = self._as_batch_mask(null_mask, batch_size)
        if mask is None:
            return audio_feat
        return torch.where(mask.view(batch_size, 1, 1), null_feat, audio_feat)

    # ------------------------------------------------------------------
    # Emotion helpers (id handling)
    # ------------------------------------------------------------------
    def _resolve_emotion_argument(self, emotion, emotion_id, emo_index) -> Optional[torch.Tensor]:
        """Normalize the three accepted emotion kwargs into a single tensor
        of ids. Returns ``None`` when no emotion argument was supplied."""
        if emo_index is not None:
            return emo_index if isinstance(emo_index, torch.Tensor) else torch.as_tensor(emo_index)
        if emotion_id is not None:
            return emotion_id if isinstance(emotion_id, torch.Tensor) else torch.as_tensor(emotion_id)
        if emotion is not None:
            return emotion  # defer conversion to emotion_to_id later
        return None

    def _materialize_emotion_ids(self, emotion_value: Optional[Union[str, int, list, torch.Tensor]],
                                 batch_size: int) -> torch.Tensor:
        """Convert the (possibly string) emotion argument into a LongTensor
        of ids of shape ``(batch_size,)`` on the model device."""
        if emotion_value is None:
            ids = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        elif isinstance(emotion_value, torch.Tensor):
            ids = emotion_value.to(device=self.device, dtype=torch.long)
            if ids.ndim == 0:
                ids = ids.view(1)
        else:
            ids = emotion_to_id(emotion_value, self.emotion_labels, device=self.device)
        if ids.ndim == 0:
            ids = ids.view(1)
        if ids.numel() == 1 and batch_size > 1:
            ids = ids.expand(batch_size)
        if ids.numel() != batch_size:
            raise ValueError(
                f"emotion batch size {ids.numel()} does not match {batch_size}"
            )
        if torch.any(ids < 0) or torch.any(ids >= self.num_emotions):
            raise ValueError(f"emotion ids must be in [0, {self.num_emotions - 1}]")
        return ids.long()

    def _prepare_emotion_ids(self, emotion_value, batch_size: int) -> torch.Tensor:
        return self._materialize_emotion_ids(emotion_value, batch_size)

    def _prepare_emotion_intensity(self, intensity, batch_size: int) -> torch.Tensor:
        if intensity is None:
            intensity = 1.0
        if isinstance(intensity, torch.Tensor):
            intensity_t = intensity.to(device=self.device, dtype=torch.float32)
        else:
            intensity_t = torch.as_tensor(intensity, device=self.device, dtype=torch.float32)
        if intensity_t.ndim == 0:
            intensity_t = intensity_t.view(1)
        if intensity_t.numel() == 1 and batch_size > 1:
            intensity_t = intensity_t.expand(batch_size)
        if intensity_t.numel() != batch_size:
            raise ValueError(
                f"emotion_intensity batch size {intensity_t.numel()} does not match {batch_size}"
            )
        return intensity_t.view(batch_size, 1, 1)

    def encode_emotion(self, emotion, batch_size: int, frame_num: int,
                       intensity: float = 1.0, null_mask=None) -> torch.Tensor:
        ids = self._prepare_emotion_ids(emotion, batch_size)
        intensity_t = self._prepare_emotion_intensity(intensity, batch_size)
        emotion_feat = self.emotion_feature_map(self.emotion_embedding(ids)).unsqueeze(1)
        null_feat = self._expand_null_emotion(batch_size, 1)
        emotion_feat = null_feat + intensity_t * (emotion_feat - null_feat)
        emotion_feat = emotion_feat.expand(batch_size, frame_num, -1)
        mask = self._as_batch_mask(null_mask, batch_size)
        if mask is not None:
            null_frame_feat = self._expand_null_emotion(batch_size, frame_num)
            emotion_feat = torch.where(mask.view(batch_size, 1, 1), null_frame_feat, emotion_feat)
        return emotion_feat

    def condition_audio_feature(self, audio_feat, emotion=None, emotion_intensity: float = 1.0,
                                audio_null_mask=None, emotion_null_mask=None,
                                enable_emotion: bool = False) -> torch.Tensor:
        """Apply CFG ``audio_null`` mask and add the optional emotion bias.

        ``prev_audio_feat`` should NOT be passed through this method --
        it is processed separately so that all CFG branches share the same
        emotion-modulated history (matches the original ADEF_remake
        ``DitTalkingHead`` behavior)."""
        conditioned_audio = self._select_audio_feature(audio_feat, audio_null_mask)
        if enable_emotion:
            emotion_feat = self.encode_emotion(
                emotion,
                batch_size=audio_feat.shape[0],
                frame_num=audio_feat.shape[1],
                intensity=emotion_intensity,
                null_mask=emotion_null_mask,
            )
            conditioned_audio = conditioned_audio + emotion_feat
        return conditioned_audio

    # ------------------------------------------------------------------
    # Training-time dropout masks
    # ------------------------------------------------------------------
    def _training_null_masks(self, batch_size: int):
        """Sample condition-dropout masks used for classifier-free training."""
        conds = set(self.guiding_conditions)
        audio_null_mask = None
        emotion_null_mask = None
        if not conds:
            return audio_null_mask, emotion_null_mask

        if self.cfg_mode == 'incremental' and conds == {'audio', 'emotion'}:
            # Match the original ADEF_remake training distribution: each
            # sample has independent audio / emotion null masks drawn from
            # ``audio_dropout_prob`` / ``emotion_dropout_prob`` so that the
            # incremental CFG branches ({} / {audio} / {audio, emotion})
            # all appear with comparable probability.
            audio_null_mask = (
                torch.rand(batch_size, device=self.device) < self.audio_dropout_prob
            )
            emotion_null_mask = (
                torch.rand(batch_size, device=self.device) < self.emotion_dropout_prob
            )
        else:
            if 'audio' in conds:
                audio_null_mask = (
                    torch.rand(batch_size, device=self.device) < self.audio_dropout_prob
                )
            if 'emotion' in conds:
                emotion_null_mask = (
                    torch.rand(batch_size, device=self.device) < self.emotion_dropout_prob
                )
        return audio_null_mask, emotion_null_mask

    # ------------------------------------------------------------------
    # Start-feature indexing (ADEF_remake checkpoint compatibility)
    # ------------------------------------------------------------------
    def _resolve_prev_motion(self, prev_motion_feat, emo_ids: Optional[torch.Tensor]):
        if prev_motion_feat is not None:
            return prev_motion_feat
        if emo_ids is None:
            return self.start_motion_feat[0:1].expand(self.start_motion_feat.shape[1:].numel() // 0,
                                                       -1, -1)  # pragma: no cover - defensive
        # ADEF_remake convention: index by emo_id -> (N, n_prev_motions, motion_feat_dim)
        return torch.index_select(self.start_motion_feat, 0, emo_ids)

    def _resolve_prev_audio(self, prev_audio_feat, emo_ids: Optional[torch.Tensor]):
        if prev_audio_feat is not None:
            return prev_audio_feat
        if emo_ids is None:
            emo_ids = torch.zeros(1, dtype=torch.long, device=self.device)
        return torch.index_select(self.start_audio_feat, 0, emo_ids)

    # ------------------------------------------------------------------
    # Forward (training)
    # ------------------------------------------------------------------
    def forward(self, motion_feat, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
                time_step=None, indicator=None,
                emotion=None, emotion_intensity: float = 1.0,
                emotion_id: Optional[torch.Tensor] = None,
                emo_index: Optional[torch.Tensor] = None):
        """
        Args:
            motion_feat: (N, L, d_coef) motion coefficients or features
            audio_or_feat: (N, L_audio) raw audio or (N, L, feature_dim) audio feature
            prev_motion_feat: (N, n_prev_motions, d_motion) previous motion feature
            prev_audio_feat: (N, n_prev_motions, feature_dim) previous audio feature
            time_step: (N,)
            indicator: (N, L) 0/1 indicator of real (unpadded) motion coefficients
            emotion: scalar/list/tensor emotion id or label
            emotion_id: backward-compatible alias for ``emotion``
            emo_index: ADEF_remake legacy kwarg (LongTensor of emotion ids)
            emotion_intensity: scalar/list/tensor strength of the emotion condition

        Returns:
            noise, denoised target, clean motion feature, raw audio feature
        """
        # Resolve ADEF_remake ``emo_index`` / JoyVASA ``emotion_id`` / ``emotion``
        emotion_value = self._resolve_emotion_argument(emotion, emotion_id, emo_index)

        batch_size = motion_feat.shape[0]

        # Materialize emotion ids (None -> zeros so we still have a valid tensor)
        emo_ids = (self._materialize_emotion_ids(emotion_value, batch_size)
                   if emotion_value is not None
                   else torch.zeros(batch_size, dtype=torch.long, device=self.device))

        # Load speech features
        if audio_or_feat.ndim == 2:  # raw waveform
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat_saved = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat_saved = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        # Previous features (ADEF_remake convention: index by emo_id when None)
        prev_motion_feat = self._resolve_prev_motion(prev_motion_feat, emo_ids)
        prev_audio_feat = self._resolve_prev_audio(prev_audio_feat, emo_ids)

        audio_null_mask, emotion_null_mask = self._training_null_masks(batch_size)
        enable_emotion = 'emotion' in self.guiding_conditions

        # Current-window audio memory: full per-sample CFG masking.
        audio_feat = self.condition_audio_feature(
            audio_feat_saved,
            emotion=emo_ids,
            emotion_intensity=emotion_intensity,
            audio_null_mask=audio_null_mask,
            emotion_null_mask=emotion_null_mask,
            enable_emotion=enable_emotion,
        )
        # prev_audio_feat: only emotion modulation, NOT null_audio masking
        # (matches the original ADEF_remake ``DitTalkingHead`` -- the
        # history window is treated as fixed context that is fused with the
        # current emotion bias once and then shared across branches).
        if enable_emotion:
            prev_audio_feat_in = self.condition_audio_feature(
                prev_audio_feat,
                emotion=emo_ids,
                emotion_intensity=emotion_intensity,
                audio_null_mask=None,
                emotion_null_mask=None,
                enable_emotion=True,
            )
        else:
            prev_audio_feat_in = prev_audio_feat

        if time_step is None:
            time_step = self.diffusion_sched.uniform_sample_t(batch_size)

        # Forward diffusion
        alpha_bar = self.diffusion_sched.alpha_bars[time_step]
        c0 = torch.sqrt(alpha_bar).view(-1, 1, 1)
        c1 = torch.sqrt(1 - alpha_bar).view(-1, 1, 1)
        eps = torch.randn_like(motion_feat)
        motion_feat_noisy = c0 * motion_feat + c1 * eps

        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat_in,
            time_step,
            indicator,
        )

        return eps, motion_feat_target, motion_feat.detach(), audio_feat_saved.detach()

    def extract_audio_feature(self, audio, frame_num=None):
        frame_num = frame_num or self.n_motions
        hidden_states = self.audio_encoder(
            pad_audio(audio), self.fps, frame_num=frame_num * 2
        ).last_hidden_state  # (N, 2L, 768)
        hidden_states = hidden_states.transpose(1, 2)  # (N, 768, 2L)
        hidden_states = F.interpolate(hidden_states, size=frame_num,
                                     align_corners=False, mode='linear')
        hidden_states = hidden_states.transpose(1, 2)  # (N, L, 768)
        return self.audio_feature_map(hidden_states)  # (N, L, feature_dim)

    # ------------------------------------------------------------------
    # Sample (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, audio_or_feat, prev_motion_feat=None, prev_audio_feat=None,
               motion_at_T=None, indicator=None, cfg_mode=None, cfg_cond=None,
               cfg_scale=1.15, flexibility: float = 0,
               dynamic_threshold=None, ret_traj: bool = False,
               emotion=None, emotion_intensity: float = 1.0,
               emotion_id: Optional[torch.Tensor] = None,
               emo_index: Optional[torch.Tensor] = None):
        # Resolve ADEF_remake ``emo_index`` / JoyVASA ``emotion_id`` / ``emotion``
        emotion_value = self._resolve_emotion_argument(emotion, emotion_id, emo_index)

        batch_size = audio_or_feat.shape[0]
        emo_ids = (self._materialize_emotion_ids(emotion_value, batch_size)
                   if emotion_value is not None
                   else torch.zeros(batch_size, dtype=torch.long, device=self.device))

        # CFG config
        if cfg_mode is None:
            cfg_mode = self.cfg_mode
        if cfg_cond is None:
            cfg_cond = self.guiding_conditions
        cfg_cond = self._normalize_cfg_conditions(cfg_cond)

        if not isinstance(cfg_scale, (list, tuple)):
            cfg_scale = [cfg_scale] * len(cfg_cond)
        else:
            cfg_scale = list(cfg_scale)
        if len(cfg_scale) != len(cfg_cond):
            raise ValueError(
                f"cfg_scale length {len(cfg_scale)} does not match cfg_cond length {len(cfg_cond)}"
            )

        # Audio feature
        if audio_or_feat.ndim == 2:
            assert audio_or_feat.shape[1] == round(16000 * self.n_motions / self.fps), \
                f'Incorrect audio length {audio_or_feat.shape[1]}'
            audio_feat = self.extract_audio_feature(audio_or_feat)
        elif audio_or_feat.ndim == 3:
            assert audio_or_feat.shape[1] == self.n_motions, \
                f'Incorrect audio feature length {audio_or_feat.shape[1]}'
            audio_feat = audio_or_feat
        else:
            raise ValueError(f'Incorrect audio input shape {audio_or_feat.shape}')

        # Previous features (ADEF_remake convention: index by emo_id when None)
        prev_motion_feat = self._resolve_prev_motion(prev_motion_feat, emo_ids)
        prev_audio_feat = self._resolve_prev_audio(prev_audio_feat, emo_ids)

        if motion_at_T is None:
            motion_at_T = torch.randn((batch_size, self.n_motions, self.motion_feat_dim),
                                       device=self.device)

        branches = self._cfg_branches(cfg_cond, cfg_mode)
        # Only enable the learned emotion path when the checkpoint was
        # trained with emotion in ``guiding_conditions``. This protects
        # old audio-only checkpoints from random emotion bias.
        enable_emotion = 'emotion' in self.guiding_conditions

        audio_feat_entries = []
        for branch in branches:
            audio_null = ('audio' in cfg_cond) and ('audio' not in branch)
            emotion_null = ('emotion' in cfg_cond) and ('emotion' not in branch)
            audio_feat_entries.append(
                self.condition_audio_feature(
                    audio_feat,
                    emotion=emo_ids,
                    emotion_intensity=emotion_intensity,
                    audio_null_mask=audio_null,
                    emotion_null_mask=emotion_null,
                    enable_emotion=enable_emotion,
                )
            )

        # ``prev_audio_feat`` is processed exactly once with the current
        # emotion bias and replicated across CFG branches (matches the
        # original ADEF_remake ``DitTalkingHead.sample`` behavior).
        if enable_emotion:
            prev_audio_feat_modulated = self.condition_audio_feature(
                prev_audio_feat,
                emotion=emo_ids,
                emotion_intensity=emotion_intensity,
                audio_null_mask=None,
                emotion_null_mask=None,
                enable_emotion=True,
            )
        else:
            prev_audio_feat_modulated = prev_audio_feat

        n_entries = len(audio_feat_entries)
        audio_feat_in = torch.cat(audio_feat_entries, dim=0)
        prev_audio_feat_in = prev_audio_feat_modulated.repeat(n_entries, 1, 1)
        prev_motion_feat_in = prev_motion_feat.repeat(n_entries, 1, 1)
        indicator_in = (torch.cat([indicator] * n_entries, dim=0)
                        if indicator is not None else None)

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
            motion_in = motion_at_t.repeat(n_entries, 1, 1)
            step_in = torch.tensor([t] * batch_size, device=self.device)
            step_in = step_in.repeat(n_entries)

            results = self.denoising_net(motion_in, audio_feat_in, prev_motion_feat_in,
                                         prev_audio_feat_in, step_in, indicator_in)

            if dynamic_threshold:
                dt_ratio, dt_min, dt_max = dynamic_threshold
                abs_results = results[:, -self.n_motions:].reshape(batch_size * n_entries, -1).abs()
                s = torch.quantile(abs_results, dt_ratio, dim=1)
                s = torch.clamp(s, min=dt_min, max=dt_max)
                s = s[..., None, None]
                results = torch.clamp(results, min=-s, max=s)

            results = results.chunk(n_entries)
            target_theta = results[0][:, -self.n_motions:]
            for i in range(0, n_entries - 1):
                if cfg_mode == 'independent':
                    target_theta += cfg_scale[i] * (
                        results[i + 1][:, -self.n_motions:] - results[0][:, -self.n_motions:]
                    )
                elif cfg_mode == 'incremental':
                    target_theta += cfg_scale[i] * (
                        results[i + 1][:, -self.n_motions:] - results[i][:, -self.n_motions:]
                    )
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
            return traj, motion_at_T, audio_feat
        return traj[0], motion_at_T, audio_feat


# ---------------------------------------------------------------------------
# Smoke test (mirrors the original dit_talking_head.py __main__ but defaults
# to the ADEF_remake motion_feat_dim of 70 instead of 76).
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    motion_feat_dim = 70     # ADEF_remake default
    n_motions = 100
    n_prev_motions = 25      # ADEF_remake default
    feature_dim = 512        # ADEF_remake default

    L_audio = int(16000 * n_motions / 25)
    N = 5

    motion_feat = torch.ones((N, n_motions, motion_feat_dim)).to(device)
    audio_or_feat = torch.ones((N, L_audio)).to(device)

    model = DitTalkingHead(
        device=device,
        motion_feat_dim=motion_feat_dim,
        n_motions=n_motions,
        n_prev_motions=n_prev_motions,
        feature_dim=feature_dim,
    ).to(device)

    z = model(motion_feat, audio_or_feat, prev_motion_feat=None,
              prev_audio_feat=None, time_step=None, indicator=None, emo_index=torch.zeros(N, dtype=torch.long))
    traj, motion_at_T, audio_feat = z[0], z[1], z[2]
    print(motion_at_T.shape, audio_feat.shape)

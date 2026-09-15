# coding: utf-8
"""Configuration classes for emotion2vec-conditioned ADEF inference."""

from dataclasses import dataclass
from typing import Literal, Optional, Union

from .argument_config import ArgumentConfig
from .base_config import make_abs_path
from .inference_config import InferenceConfig


E2V_VARIANTS = Literal[
    "auto",
    "emotion_dit_e2v",
    "emotion_dit_finalv3",
    "emotion_dit_finalv3_two_stage",
]


@dataclass(repr=False)
class E2VArgumentConfig(ArgumentConfig):
    """User-facing arguments for ``inference_e2v.py``."""

    checkpoint_MotionGenerator: str = ""
    motion_template_path: str = make_abs_path(
        "../../pretrained_weights/ADEF/motion_template/motion_template.pkl"
    )
    motion_generator_variant: E2V_VARIANTS = "auto"

    emotion2vec_model_id: str = "iic/emotion2vec_plus_large"
    emotion2vec_hub: Literal["ms", "modelscope", "hf", "huggingface"] = "ms"
    emotion2vec_device: Optional[str] = None
    emotion2vec_cache_dir: Optional[str] = None
    emotion2vec_utterance_path: Optional[str] = None
    emotion2vec_frame_path: Optional[str] = None
    emotion2vec_force_extract: bool = False

    # Old motion caches only depend on the audio filename and therefore cannot
    # distinguish changed target labels/e2v features. Keep them disabled unless
    # the user explicitly accepts that legacy behavior.
    allow_legacy_motion_cache: bool = False

    cfg_cond: Optional[str] = None
    cfg_scale: Union[float, list[float]] = 2.8
    cfg_min: Optional[list[float]] = None
    cfg_schedule: Optional[Literal["none", "linear", "cosine", "bell"]] = None


@dataclass(repr=False)
class E2VInferenceConfig(InferenceConfig):
    """Internal configuration consumed by ``ADEFE2VWrapper``."""

    motion_generator_variant: E2V_VARIANTS = "auto"
    emotion2vec_model_id: str = "iic/emotion2vec_plus_large"
    emotion2vec_hub: Literal["ms", "modelscope", "hf", "huggingface"] = "ms"
    emotion2vec_device: Optional[str] = None

"""Wide (640-dim, 10-head) 0803 model variant."""

from .emotion_dit_Unification_jianhua0803 import (
    DiffusionSchedule,
    DenoisingNetwork,
    DiTDecoderLayer,
    DiTDecoder,
    DitTalkingHead as _BaseDitTalkingHead,
    _resolve_runtime_arg,
)


class DitTalkingHead(_BaseDitTalkingHead):
    def __init__(self, *args, n_heads=None, n_layers=None, mlp_ratio=None, **kwargs):
        n_heads = _resolve_runtime_arg(n_heads, "n_heads", 10)
        n_layers = _resolve_runtime_arg(n_layers, "n_layers", 8)
        mlp_ratio = _resolve_runtime_arg(mlp_ratio, "mlp_ratio", 4)
        kwargs.setdefault("feature_dim", 640)
        super().__init__(
            *args,
            n_heads=n_heads,
            n_layers=n_layers,
            mlp_ratio=mlp_ratio,
            **kwargs,
        )


__all__ = ["DiffusionSchedule", "DenoisingNetwork", "DiTDecoderLayer", "DiTDecoder", "DitTalkingHead"]

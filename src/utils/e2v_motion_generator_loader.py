"""Load emotion2vec-capable ADEF motion generators from checkpoints.

Supported variants:

* ``emotion_dit_e2v``
* ``emotion_dit_finalv3``
* ``emotion_dit_finalv3_two_stage``

The loader accepts checkpoints saved by both the original training scripts and
``train_two_stage.py``. Variant selection can be explicit or inferred from the
checkpoint arguments/state-dict keys.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import torch


VARIANT_MODULES = {
    "emotion_dit_e2v": "src.modules.emotion_dit_e2v",
    "emotion_dit_finalv3": "src.modules.emotion_dit_finalv3",
    "emotion_dit_finalv3_two_stage": "src.modules.emotion_dit_finalv3_two_stage",
}

VARIANT_ALIASES = {
    "e2v": "emotion_dit_e2v",
    "finalv3": "emotion_dit_finalv3",
    "finalv3_two_stage": "emotion_dit_finalv3_two_stage",
    "two_stage_finalv3": "emotion_dit_finalv3_two_stage",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported checkpoint args type: {type(value)}")


def _normalize_variant(name: str | None) -> str | None:
    if not name or name == "auto":
        return None
    name = VARIANT_ALIASES.get(name, name)
    if name not in VARIANT_MODULES:
        raise ValueError(
            f"Unknown motion generator variant '{name}'. "
            f"Choose from {sorted(VARIANT_MODULES)} or 'auto'."
        )
    return name


def infer_variant(
    checkpoint_args: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    explicit_variant: str | None = None,
) -> str:
    explicit = _normalize_variant(explicit_variant)
    if explicit is not None:
        return explicit

    for key in (
        "motion_generator_variant",
        "model_variant",
        "variant",
        "model_name",
    ):
        candidate = _normalize_variant(checkpoint_args.get(key))
        if candidate is not None:
            return candidate

    keys = tuple(state_dict.keys())
    if any("emotion_audio_encoder.label_basis" in key for key in keys):
        return "emotion_dit_finalv3_two_stage"
    if any("hierarchical_emotion_audio_encoder" in key for key in keys):
        return "emotion_dit_finalv3"
    if any("condition_encoder" in key for key in keys):
        return "emotion_dit_e2v"

    raise ValueError(
        "Could not infer an emotion2vec motion-generator variant from the "
        "checkpoint. Pass --motion-generator-variant explicitly."
    )


def _constructor_kwargs(model_class, checkpoint_args, device):
    signature = inspect.signature(model_class.__init__)
    aliases = {
        "num_label_tokens": (
            "num_label_tokens",
            "num_emotion_tokens",
        ),
        "num_emotion_tokens": (
            "num_emotion_tokens",
            "num_label_tokens",
        ),
        "e2v_dim": ("e2v_dim", "emotion2vec_dim"),
        "use_learnable_pe": (
            "use_learnable_pe",
            "no_use_learnable_pe",
        ),
    }

    kwargs = {"device": str(device)}
    for name, parameter in signature.parameters.items():
        if name in {"self", "device"}:
            continue
        if name in checkpoint_args and checkpoint_args[name] is not None:
            kwargs[name] = checkpoint_args[name]
            continue
        for source in aliases.get(name, ()):  # compatibility names
            if source not in checkpoint_args:
                continue
            value = checkpoint_args[source]
            if name == "use_learnable_pe" and source == "no_use_learnable_pe":
                value = not bool(value)
            kwargs[name] = value
            break
        if name not in kwargs and parameter.default is inspect.Parameter.empty:
            raise ValueError(
                f"Checkpoint is missing required constructor argument '{name}'"
            )
    return kwargs


def _clean_state_dict(state_dict):
    result = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        result[key] = value
    return result


def _load_compatible_state(model, state_dict):
    model_state = model.state_dict()
    compatible = {}
    skipped_shape = []
    unexpected = []
    for key, value in _clean_state_dict(state_dict).items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append(
                (key, tuple(value.shape), tuple(model_state[key].shape))
            )
            continue
        compatible[key] = value

    result = model.load_state_dict(compatible, strict=False)
    return {
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": unexpected + list(result.unexpected_keys),
        "shape_mismatches": skipped_shape,
        "loaded_keys": len(compatible),
        "model_keys": len(model_state),
    }


def load_e2v_motion_generator(
    checkpoint_path: str,
    device: str,
    variant: str = "auto",
) -> Tuple[torch.nn.Module, SimpleNamespace, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    checkpoint_args = _as_dict(checkpoint.get("args", {}))
    selected_variant = infer_variant(
        checkpoint_args,
        state_dict,
        explicit_variant=variant,
    )

    module = __import__(VARIANT_MODULES[selected_variant], fromlist=["DitTalkingHead"])
    model_class = module.DitTalkingHead
    kwargs = _constructor_kwargs(model_class, checkpoint_args, device)
    model = model_class(**kwargs)
    report = _load_compatible_state(model, state_dict)

    if selected_variant == "emotion_dit_finalv3_two_stage":
        model.set_train_stage(2)
    model.to(device)
    model.eval()

    checkpoint_args.setdefault("motion_generator_variant", selected_variant)
    for key, value in kwargs.items():
        checkpoint_args.setdefault(key, value)
    args_namespace = SimpleNamespace(**checkpoint_args)
    report["variant"] = selected_variant
    report["checkpoint_path"] = checkpoint_path
    return model, args_namespace, report

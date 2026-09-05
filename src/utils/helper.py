"""Compatibility wrapper for ADEF model loading.

The original helper is kept in ``helper_legacy``. Motion-generator construction
is overridden so checkpoints trained after the 0803 parameter-propagation fix
are reconstructed with the exact architecture recorded in ``args``. ICASSP27
controlled-ablation checkpoints additionally record a dedicated ``model_variant``;
those self-contained variants are resolved through a strict allow-list prefix so
inference never silently falls back to the final ADEF class. Older checkpoints retain their
existing fallback behavior.
"""

import importlib
import inspect

import torch

from . import helper_legacy as _legacy
from .helper_legacy import *  # noqa: F401,F403


_ABLATION_PREFIX = "emotion_dit_ablation0905_"
_DEFAULT_MOTION_MODULE = "emotion_dit_Unification_jianhua0803_minsnr_ema"


def _arg(model_args, name, default=None):
    value = getattr(model_args, name, None)
    return default if value is None else value


def _resolve_motion_architecture(model_args):
    """Resolve architecture parameters with backward compatibility."""
    feature_dim = int(_arg(model_args, "feature_dim", 512))
    n_layers = int(_arg(model_args, "n_layers", 8))
    n_heads = int(_arg(model_args, "n_heads", 10 if feature_dim == 640 else 8))
    mlp_ratio = int(_arg(model_args, "mlp_ratio", 4))
    is_new = bool(_arg(model_args, "model_params_propagated", False))

    if is_new:
        use_indicator = bool(_arg(model_args, "use_indicator", True))
        no_use_learnable_pe = bool(_arg(model_args, "no_use_learnable_pe", False))
    else:
        # Effective architecture of pre-fix 0803 checkpoints. Keep the values
        # already stored by the compatibility layer rather than guessing from
        # the filename or from a sibling training script.
        use_indicator = bool(_arg(model_args, "use_indicator", True))
        no_use_learnable_pe = bool(_arg(model_args, "no_use_learnable_pe", False))

    return {
        "feature_dim": feature_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "mlp_ratio": mlp_ratio,
        "use_indicator": use_indicator,
        "no_use_learnable_pe": no_use_learnable_pe,
    }


def _resolve_motion_model_class(model_args):
    """Return the exact motion-model class declared by a checkpoint.

    Only the controlled ``emotion_dit_ablation0905_*`` namespace is imported
    dynamically. This prevents arbitrary module imports from untrusted checkpoint
    metadata while guaranteeing that ablation checkpoints use their own physical
    model copies during generation/evaluation.
    """
    model_variant = str(_arg(model_args, "model_variant", "")).strip()
    if model_variant.startswith(_ABLATION_PREFIX):
        if model_variant.endswith("_legacy"):
            raise ValueError(
                "legacy ablation module names are no longer valid after the one-file "
                f"refactor: {model_variant}"
            )
        if not all(ch.isalnum() or ch == "_" for ch in model_variant):
            raise ValueError(f"unsafe model_variant in checkpoint: {model_variant!r}")
        module = importlib.import_module(f"src.modules.{model_variant}")
        return module.DitTalkingHead

    module = importlib.import_module(f"src.modules.{_DEFAULT_MOTION_MODULE}")
    return module.DitTalkingHead


def load_model(ckpt_path, model_config, device, model_type):
    if model_type != "motion_generator":
        return _legacy.load_model(ckpt_path, model_config, device, model_type)

    model_data = torch.load(ckpt_path, map_location=device)
    model_args = _legacy.NullableArgs(model_data["args"])
    arch = _resolve_motion_architecture(model_args)
    DitTalkingHead = _resolve_motion_model_class(model_args)

    model_kwargs = {
        "device": device,
        "target": _arg(model_args, "target", "sample"),
        "architecture": _arg(model_args, "architecture", "decoder"),
        "motion_feat_dim": int(_arg(model_args, "motion_feat_dim", 70)),
        "fps": int(_arg(model_args, "fps", 25)),
        "n_motions": int(_arg(model_args, "n_motions", 64)),
        "n_prev_motions": int(_arg(model_args, "n_prev_motions", 16)),
        "audio_model": _arg(model_args, "audio_model", "wav2vec2"),
        "feature_dim": arch["feature_dim"],
        "n_diff_steps": int(_arg(model_args, "n_diff_steps", 500)),
        "diff_schedule": _arg(model_args, "diff_schedule", "cosine"),
        "cfg_mode": _arg(model_args, "cfg_mode", "incremental"),
        "guiding_conditions": _arg(model_args, "guiding_conditions", "audio,emotion"),
        "align_mask_width": int(_arg(model_args, "align_mask_width", 1)),
        "n_heads": arch["n_heads"],
        "n_layers": arch["n_layers"],
        "mlp_ratio": arch["mlp_ratio"],
        "use_indicator": arch["use_indicator"],
        "no_use_learnable_pe": arch["no_use_learnable_pe"],
    }

    # The spatial-partition ablation exposes one additional constructor option.
    # Pass it only to model classes that explicitly declare the parameter so old
    # and non-partition checkpoints remain byte-for-byte compatible at loading.
    signature = inspect.signature(DitTalkingHead.__init__)
    if "partition_keypoint_indices" in signature.parameters:
        model_kwargs["partition_keypoint_indices"] = _arg(
            model_args,
            "partition_keypoint_indices",
            "0,1,2,3,4,5,6,7,8,9,10",
        )

    model = DitTalkingHead(**model_kwargs)

    state_dict = model_data["model"]
    state_dict.pop("denoising_net.TE.pe", None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, model_args

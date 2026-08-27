"""Compatibility-aware ADEF model loading.

The original helper is kept in ``helper_legacy``. Motion-generator loading is
handled here so that both historical checkpoints and the full-copy 0803
optimization variants are reconstructed with the model class and architecture
that were actually used during training.
"""

import importlib

import torch

from . import helper_legacy as _legacy
from .helper_legacy import *  # noqa: F401,F403


_BASE_MOTION_MODEL_MODULE = "src.modules.emotion_dit_Unification_jianhua0803"
_SUPPORTED_MOTION_MODEL_MODULES = {
    _BASE_MOTION_MODEL_MODULE,
    "src.modules.emotion_dit_Unification_jianhua0803_lipaware",
    "src.modules.emotion_dit_Unification_jianhua0803_audio_pyramid",
    "src.modules.emotion_dit_Unification_jianhua0803_channelgate",
    "src.modules.emotion_dit_Unification_jianhua0803_minsnr_ema",
}


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
        # Effective architecture of pre-fix 0803 checkpoints. The base model did
        # not receive indicator; deep/wide matrix variants hard-coded it. None of
        # the pre-fix variants propagated the learnable-PE flag.
        use_indicator = bool(
            feature_dim != 512 or n_layers != 8 or n_heads != 8
        )
        no_use_learnable_pe = True

    return {
        "feature_dim": feature_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "mlp_ratio": mlp_ratio,
        "use_indicator": use_indicator,
        "no_use_learnable_pe": no_use_learnable_pe,
    }


def _resolve_motion_model_class(model_args):
    """Return the exact DitTalkingHead class recorded by the training script.

    New optimization checkpoints store ``args.model_module``. Historical
    checkpoints do not, so they intentionally fall back to the corrected 0803
    base implementation.
    """
    model_module = _arg(model_args, "model_module", _BASE_MOTION_MODEL_MODULE)
    if model_module not in _SUPPORTED_MOTION_MODEL_MODULES:
        supported = "\n  - ".join(sorted(_SUPPORTED_MOTION_MODEL_MODULES))
        raise ValueError(
            f"Unsupported motion-generator module in checkpoint: {model_module!r}.\n"
            f"Supported modules:\n  - {supported}"
        )

    module = importlib.import_module(model_module)
    model_class = getattr(module, "DitTalkingHead", None)
    if model_class is None:
        raise AttributeError(
            f"Checkpoint module {model_module!r} does not expose DitTalkingHead."
        )
    return model_class, model_module


def load_model(ckpt_path, model_config, device, model_type):
    if model_type != "motion_generator":
        return _legacy.load_model(ckpt_path, model_config, device, model_type)

    model_data = torch.load(ckpt_path, map_location=device)
    model_args = _legacy.NullableArgs(model_data["args"])
    arch = _resolve_motion_architecture(model_args)
    model_class, model_module = _resolve_motion_model_class(model_args)

    model = model_class(
        device=device,
        target=_arg(model_args, "target", "sample"),
        architecture=_arg(model_args, "architecture", "decoder"),
        motion_feat_dim=int(_arg(model_args, "motion_feat_dim", 70)),
        fps=int(_arg(model_args, "fps", 25)),
        n_motions=int(_arg(model_args, "n_motions", 64)),
        n_prev_motions=int(_arg(model_args, "n_prev_motions", 16)),
        audio_model=_arg(model_args, "audio_model", "wav2vec2"),
        feature_dim=arch["feature_dim"],
        n_diff_steps=int(_arg(model_args, "n_diff_steps", 500)),
        diff_schedule=_arg(model_args, "diff_schedule", "cosine"),
        cfg_mode=_arg(model_args, "cfg_mode", "incremental"),
        guiding_conditions=_arg(model_args, "guiding_conditions", "audio,emotion"),
        align_mask_width=int(_arg(model_args, "align_mask_width", 1)),
        n_heads=arch["n_heads"],
        n_layers=arch["n_layers"],
        mlp_ratio=arch["mlp_ratio"],
        use_indicator=arch["use_indicator"],
        no_use_learnable_pe=arch["no_use_learnable_pe"],
    )

    # Min-SNR+EMA checkpoints intentionally store EMA parameters in ``model``
    # and raw online parameters in ``model_raw``. Using ``model`` therefore
    # automatically selects EMA for normal inference.
    state_dict = dict(model_data["model"])
    state_dict.pop("denoising_net.TE.pe", None)
    incompatible = model.load_state_dict(state_dict, strict=False)

    # A missing TE.pe is expected because it is regenerated from n_diff_steps.
    missing = [
        key for key in incompatible.missing_keys
        if key != "denoising_net.TE.pe"
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Motion-generator checkpoint does not match its recorded model "
            f"module {model_module!r}. Missing keys: {missing}; unexpected keys: "
            f"{list(incompatible.unexpected_keys)}"
        )

    model.to(device)
    model.eval()
    return model, model_args

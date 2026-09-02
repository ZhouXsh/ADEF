"""Compatibility wrapper for ADEF model loading.

The original helper is kept in ``helper_legacy``. Only motion-generator
construction is overridden so checkpoints trained after the 0803 parameter
propagation fix are reconstructed with the exact architecture recorded in
``args``. Older checkpoints retain their effective legacy architecture.
"""

import torch

from . import helper_legacy as _legacy
from .helper_legacy import *  # noqa: F401,F403


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
        use_indicator = bool(_arg(model_args, "use_indicator", True))
        no_use_learnable_pe = bool(_arg(model_args, "no_use_learnable_pe", False))
        # use_indicator = bool(
        #     feature_dim != 512 or n_layers != 8 or n_heads != 8
        # )
        # no_use_learnable_pe = True

    return {
        "feature_dim": feature_dim,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "mlp_ratio": mlp_ratio,
        "use_indicator": use_indicator,
        "no_use_learnable_pe": no_use_learnable_pe,
    }


def load_model(ckpt_path, model_config, device, model_type):
    if model_type != "motion_generator":
        return _legacy.load_model(ckpt_path, model_config, device, model_type)

    from ..modules.emotion_dit_Unification_jianhua0803_minsnr_ema import DitTalkingHead

    model_data = torch.load(ckpt_path, map_location=device)
    model_args = _legacy.NullableArgs(model_data["args"])
    arch = _resolve_motion_architecture(model_args)

    model = DitTalkingHead(
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

    state_dict = model_data["model"]
    state_dict.pop("denoising_net.TE.pe", None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, model_args

import torch

from .helper import *
from .helper import load_model as _base_load_model
from ..modules.emotion_dit_Unification_2pad_0721 import DitTalkingHead


def _get_arg(args, name, default):
    value = getattr(args, name)
    return default if value is None else value


def load_model(ckpt_path, model_config, device, model_type):
    if model_type != "motion_generator":
        return _base_load_model(ckpt_path, model_config, device, model_type)

    model_data = torch.load(ckpt_path, map_location=device)
    model_args = NullableArgs(model_data["args"])
    model = DitTalkingHead(
        device=device,
        target=_get_arg(model_args, "target", "sample"),
        architecture=_get_arg(model_args, "architecture", "decoder"),
        motion_feat_dim=_get_arg(model_args, "motion_feat_dim", 70),
        fps=_get_arg(model_args, "fps", 25),
        n_motions=_get_arg(model_args, "n_motions", 100),
        n_prev_motions=_get_arg(model_args, "n_prev_motions", 25),
        feature_dim=_get_arg(model_args, "feature_dim", 512),
        audio_model=_get_arg(model_args, "audio_model", "hubert"),
        n_diff_steps=_get_arg(model_args, "n_diff_steps", 50),
        diff_schedule=_get_arg(model_args, "diff_schedule", "cosine"),
        cfg_mode=_get_arg(model_args, "cfg_mode", "incremental"),
        guiding_conditions=_get_arg(
            model_args, "guiding_conditions", "audio,emotion"
        ),
    )

    state_dict = dict(model_data["model"])
    model_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key not in model_state or state_dict[key].shape != model_state[key].shape:
            state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, model_args

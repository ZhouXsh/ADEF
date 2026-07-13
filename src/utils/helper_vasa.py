"""Model-loading adapter for the VASA-style motion generator."""

import torch

from .helper import *  # noqa: F401,F403
from .helper import NullableArgs, load_model as load_base_model
from ..modules.emotion_dit_vasa import DitTalkingHead


def _value(args, name, default):
    value = getattr(args, name, None)
    return default if value is None else value


def load_model(ckpt_path, model_config, device, model_type):
    """Load all original modules, replacing only the motion generator."""
    if model_type != 'motion_generator':
        return load_base_model(ckpt_path, model_config, device, model_type)

    model_data = torch.load(ckpt_path, map_location=device)
    model_args = NullableArgs(model_data['args'])
    model = DitTalkingHead(
        device=device,
        target=_value(model_args, 'target', 'sample'),
        architecture=_value(model_args, 'architecture', 'decoder'),
        motion_feat_dim=_value(model_args, 'motion_feat_dim', 70),
        fps=_value(model_args, 'fps', 25),
        n_motions=_value(model_args, 'n_motions', 100),
        n_prev_motions=_value(model_args, 'n_prev_motions', 25),
        audio_model=_value(model_args, 'audio_model', 'wav2vec2'),
        feature_dim=_value(model_args, 'feature_dim', 512),
        n_diff_steps=_value(model_args, 'n_diff_steps', 50),
        diff_schedule=_value(model_args, 'diff_schedule', 'cosine'),
        cfg_mode=_value(model_args, 'cfg_mode', 'incremental'),
        guiding_conditions=_value(model_args, 'guiding_conditions', 'audio,emotion'),
        prev_dropout_prob=_value(model_args, 'prev_dropout_prob', 0.1),
    )

    state_dict = dict(model_data['model'])
    state_dict.pop('denoising_net.TE.pe', None)
    # Allow loading an older checkpoint into the copied model without reviving
    # the removed learned first-window tokens.
    state_dict.pop('start_audio_feat', None)
    state_dict.pop('start_motion_feat', None)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, model_args

# coding: utf-8

"""Frame-level 推理专用模型加载器。

保留 helper.py 的其他工具函数，只替换 motion_generator 的构造类型，
使推理端加载 emotion_dit_Unification_framelevel_0721.DitTalkingHead。
"""

import torch

from .helper import *  # noqa: F401,F403
from .helper import load_model as _base_load_model
from ..modules.emotion_dit_Unification_framelevel_0721 import DitTalkingHead


def load_model(ckpt_path, model_config, device, model_type):
    if model_type != 'motion_generator':
        return _base_load_model(ckpt_path, model_config, device, model_type)

    model_data = torch.load(ckpt_path, map_location=device)
    model_args = NullableArgs(model_data['args'])

    emotion2vec_dim = getattr(model_args, 'emotion2vec_dim', None) or 1024
    model = DitTalkingHead(
        device=device,
        target=getattr(model_args, 'target', None) or 'sample',
        architecture=getattr(model_args, 'architecture', None) or 'decoder',
        motion_feat_dim=model_args.motion_feat_dim,
        fps=model_args.fps,
        n_motions=model_args.n_motions,
        n_prev_motions=model_args.n_prev_motions,
        audio_model=model_args.audio_model,
        feature_dim=model_args.feature_dim,
        n_diff_steps=model_args.n_diff_steps,
        diff_schedule=getattr(model_args, 'diff_schedule', None) or 'cosine',
        cfg_mode=getattr(model_args, 'cfg_mode', None) or 'incremental',
        guiding_conditions=(
            getattr(model_args, 'guiding_conditions', None)
            or 'audio,emotion'
        ),
        emotion2vec_dim=emotion2vec_dim,
    )

    # PositionalEncoding 的 pe 是 buffer，由当前模型按 n_diff_steps 重建。
    state_dict = model_data['model'].copy()
    state_dict.pop('denoising_net.TE.pe', None)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'[framelevel loader] missing keys: {missing}')
    if unexpected:
        print(f'[framelevel loader] unexpected keys: {unexpected}')

    model.to(device)
    model.eval()
    return model, model_args

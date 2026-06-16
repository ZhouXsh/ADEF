# coding: utf-8
"""
Training wrapper for ``src.modules.emotion_dit_prev_modi_new``.

This wrapper reuses the dataloading, loss, logging, validation and checkpointing
logic from ``train_decoupled_emotion_experiments.py`` but registers a new
from-scratch experiment variant:

    E_prev_modi_new_from_scratch

Why a wrapper instead of editing the big trainer again?
------------------------------------------------------
C/D in ``train_decoupled_emotion_experiments.py`` are still useful when you have
an A_audio_only checkpoint.  This file is for the case where you do not want to
wait for A.  It trains the new architecture from random initialization, so it
sets ``freeze_base_iters=0`` and does not require ``--pretrained_ckpt``.

Example
-------
python train_prev_modi_new_from_scratch.py \
  --device_id 2 \
  --exp_name E_prev_modi_new_from_scratch

Optional protected lip/keypoint experiment, still from scratch:

python train_prev_modi_new_from_scratch.py \
  --device_id 3 \
  --exp_name E_prev_modi_new_protected \
  --emotion_protected_kp_indices 你确认的嘴唇kp编号 \
  --emotion_protected_weight 0.0
"""

from __future__ import annotations

import src.utils as utils
import train_decoupled_emotion_experiments as base


BASE_SELECT_MODEL_CLASS = base.select_model_class
BASE_MAKE_MODEL_KWARGS = base.make_model_kwargs


base.EXPERIMENT_MATRIX["E_prev_modi_new_from_scratch"] = {
    "model_variant": "prev_modi_new",
    "guiding_conditions": "audio,emotion",
    "l_emo": 1.0,
    "freeze_base_iters": 0,
    "description": "from-scratch clean-audio DiT + motion-space emotion residual; no A checkpoint required",
}


def select_model_class(model_variant: str):
    if model_variant == "prev_modi_new":
        from src.modules.emotion_dit_prev_modi_new import DitTalkingHead
        return DitTalkingHead
    return BASE_SELECT_MODEL_CLASS(model_variant)


def make_model_kwargs(args, device):
    kwargs = BASE_MAKE_MODEL_KWARGS(args, device)
    if args.model_variant == "prev_modi_new":
        kwargs.update(
            emotion_dropout_prob=args.emotion_dropout_prob,
            emotion_residual_scale=args.emotion_residual_scale,
            emotion_hidden_dim=args.emotion_hidden_dim,
            emotion_pose_weight=args.emotion_pose_weight,
            emotion_protected_dims=args.emotion_protected_dims,
            emotion_protected_kp_indices=args.emotion_protected_kp_indices,
            emotion_protected_weight=args.emotion_protected_weight,
            base_start_emotion_id=args.base_start_emotion_id,
        )
    return kwargs


base.select_model_class = select_model_class
base.make_model_kwargs = make_model_kwargs


if __name__ == "__main__":
    parser = base.build_parser()
    parser.set_defaults(
        experiment_variant="E_prev_modi_new_from_scratch",
        exp_name="E_prev_modi_new_from_scratch",
        freeze_base_iters=0,
        pretrained_ckpt="",
    )
    args = parser.parse_args()
    args = base.apply_experiment_preset(args)
    option_text = utils.common.get_option_text(args, parser)
    base.main(args, option_text)

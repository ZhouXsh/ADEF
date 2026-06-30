"""Training entrypoint for emotion_dit_v6.

v6 trains the stochastic affective motion-prior model:
- v5 emotion basis bank;
- small Gaussian affective latent tokens for one-to-many nonverbal dynamics;
- orthogonalized emotion attention to protect the local audio/lipsync path.

The default loop reuses train.py. The model exposes get_aux_loss() for future
custom KL regularization, but this entrypoint intentionally keeps training
identical to the baseline for a clean first ablation.
"""

import torch

from train_v1 import build_parser, _base_train
from src.modules.emotion_dit_v6 import DitTalkingHead


g_exp_name = "20260701_v6_stochastic_affective_prior"


def patch_base_train(device_id: int):
    _base_train.DitTalkingHead = DitTalkingHead
    _base_train.g_exp_name = g_exp_name
    _base_train.device_id = device_id
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    _base_train.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = build_parser()
    parser.set_defaults(exp_name=g_exp_name, device_id=2)
    args = parser.parse_args()
    patch_base_train(args.device_id)
    option_text = _base_train.utils.common.get_option_text(args, parser) if args.mode == 'train' else None
    _base_train.main(args, option_text)

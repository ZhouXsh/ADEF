"""Training entrypoint for emotion_dit_v4.

v4 trains the conservative decoupled audio/emotion attention model:
- original emotion-id embedding;
- local audio cross-attention + emotion cross-attention;
- DICE-style CFG at sampling time.

The loop reuses train.py through train_v1.py's safe-import wrapper, so the
original loss functions and checkpointing behavior stay unchanged.
"""

import torch

from train_v1 import build_parser, _base_train
from src.modules.emotion_dit_v4 import DitTalkingHead


g_exp_name = "20260701_v4_decoupled_audio_emotion_attn"


def patch_base_train(device_id: int):
    _base_train.DitTalkingHead = DitTalkingHead
    _base_train.g_exp_name = g_exp_name
    _base_train.device_id = device_id
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    _base_train.device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = build_parser()
    parser.set_defaults(exp_name=g_exp_name, device_id=0)
    args = parser.parse_args()
    patch_base_train(args.device_id)
    option_text = _base_train.utils.common.get_option_text(args, parser) if args.mode == 'train' else None
    _base_train.main(args, option_text)

import torch

from .dataset_EmotionLevel_2pad_0721 import EmoLevelDataset as TwoPadEmoLevelDataset


class EmoLevelDataset(TwoPadEmoLevelDataset):
    """Return the canonical token while preserving the trainer tuple contract.

    The fourth tuple entry is a zero placeholder kept only for compatibility with
    the copied training loop. The 1-pad model ignores it completely.
    """

    def __getitem__(self, index):
        (
            audio_pair,
            coef_pair,
            canonical_kp_pair,
            _first_motion_pair,
            emo_index,
            emo_level,
        ) = super().__getitem__(index)

        dummy_first_motion_pair = [
            torch.zeros_like(canonical_kp_pair[0]),
            torch.zeros_like(canonical_kp_pair[1]),
        ]
        return (
            audio_pair,
            coef_pair,
            canonical_kp_pair,
            dummy_first_motion_pair,
            emo_index,
            emo_level,
        )


__all__ = ["EmoLevelDataset"]

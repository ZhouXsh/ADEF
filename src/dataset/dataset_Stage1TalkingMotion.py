"""Generic talking-motion dataset used by the first training stage.

This file intentionally keeps ``dataset_GenericTalkingMotion.py`` unchanged.
The data reading, alignment, cropping and normalization behavior are inherited
from ``GenericTalkingMotionDataset``; only the returned sample contract is
adapted to the existing ADEF training loop by adding a neutral label.
"""

import torch

from .dataset_GenericTalkingMotion import GenericTalkingMotionDataset


class Stage1TalkingMotionDataset(GenericTalkingMotionDataset):
    """Return generic samples with the four-item ADEF training interface.

    Returns:
        audio_pair: Two consecutive audio windows.
        coefficient_pair: Two consecutive normalized motion windows.
        neutral_index: Always ``0``. It is an interface placeholder and is not
            used as an emotion supervision signal in stage 1.
        sample_path: Source video path.
    """

    def __getitem__(self, index):
        audio_pair, coefficient_pair, sample_path = super().__getitem__(index)
        neutral_index = torch.tensor(0, dtype=torch.long)
        return audio_pair, coefficient_pair, neutral_index, sample_path

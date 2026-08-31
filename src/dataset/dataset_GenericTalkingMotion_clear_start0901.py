"""Generic talking-motion dataset with true video-start batches.

This is a non-destructive extension of ``dataset_GenericTalkingMotion_clear``.
It keeps the original random 80-frame continuation samples and adds a
``begin64`` mode that packs the real first 64 frames into the current section.
The training variants alternate continuation and start samples explicitly.
"""

import torch

from .dataset_GenericTalkingMotion_clear import (
    GenericTalkingMotionDataset as _GenericTalkingMotionDataset,
)


class GenericTalkingMotionDataset(_GenericTalkingMotionDataset):
    """Expose true first-64-frame samples without changing the base dataset."""

    alternate_start_random = True

    def __getitem__(self, index):
        if self.crop_strategy != "begin64":
            return super().__getitem__(index)

        original_strategy = self.crop_strategy
        self.crop_strategy = "begin"
        try:
            audio, coef_dict = super().__getitem__(index)
        finally:
            self.crop_strategy = original_strategy

        n_prev_audio_samples = round(self.audio_unit * self.n_prev_motions)
        current_audio = audio[: self.n_audio_samples]
        prefix_audio = torch.zeros(
            n_prev_audio_samples,
            dtype=audio.dtype,
            device=audio.device,
        )
        audio = torch.cat([prefix_audio, current_audio], dim=0)

        packed_coef = {}
        for key, value in coef_dict.items():
            current = value[: self.n_motions]
            if self.n_prev_motions > 0:
                prefix = value[:1].expand(
                    self.n_prev_motions, *value.shape[1:]
                ).clone()
                packed_coef[key] = torch.cat([prefix, current], dim=0)
            else:
                packed_coef[key] = current

        return audio, packed_coef


__all__ = ["GenericTalkingMotionDataset"]

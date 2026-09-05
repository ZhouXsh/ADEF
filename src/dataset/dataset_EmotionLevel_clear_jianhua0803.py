"""0803 dataset compatibility layer with true video-start samples."""

import torch

from .dataset_EmotionLevel_clear_jianhua0803_legacy import EmoLevelDataset as _LegacyEmoLevelDataset


class EmoLevelDataset(_LegacyEmoLevelDataset):
    """Keep random continuation crops and expose a true first-64-frame mode.

    ``infinite_data_loader`` detects ``alternate_start_random`` and alternates:
    random 80-frame continuation batch -> true video-start 64-frame batch.
    The legacy training loop still slices ``[-n_motions:]`` for a starting batch,
    so ``begin64`` packs the real frames [0, n_motions) into that tail section.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_start_random = True

    def __getitem__(self, index):
        if self.crop_strategy != "begin64":
            return super().__getitem__(index)

        original_strategy = self.crop_strategy
        self.crop_strategy = "begin"
        try:
            audio, coef_dict, emo_index, emo_level = super().__getitem__(index)
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
                prefix = value[:1].expand(self.n_prev_motions, *value.shape[1:]).clone()
                packed_coef[key] = torch.cat([prefix, current], dim=0)
            else:
                packed_coef[key] = current

        return audio, packed_coef, emo_index, emo_level


__all__ = ["EmoLevelDataset"]

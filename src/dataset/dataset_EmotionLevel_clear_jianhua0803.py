"""0803 dataset compatibility layer with true video-start samples.

Besides the ``begin64`` compatibility mode, this wrapper repairs a practical
indexing problem of the legacy MEAD dataset: ``train.txt`` / ``test.txt`` and
``front_all_motions.pkl`` can occasionally disagree on an absolute repository
prefix, or a small number of list entries can be absent from the merged motion
pickle.  The legacy implementation indexes the dictionary directly and would
therefore crash inside a DataLoader worker with a ``KeyError`` only after the
expensive Generic stage has completed.

The wrapper now validates the list against the motion dictionary immediately
when MEAD is constructed.  A uniquely matching MEAD-relative key is aliased
when only the absolute root changed; truly missing motion entries are removed
from ``all_data`` with an explicit warning.  This changes neither valid samples
nor the motion values used for training.
"""

import os
import warnings
from pathlib import Path

import torch

from .dataset_EmotionLevel_clear_jianhua0803_legacy import EmoLevelDataset as _LegacyEmoLevelDataset


class EmoLevelDataset(_LegacyEmoLevelDataset):
    """Keep random continuation crops and expose a true first-64-frame mode.

    ``infinite_data_loader`` detects ``alternate_start_random`` and alternates:
    random 80-frame continuation batch -> true video-start 64-frame batch.
    The legacy training loop still slices ``[-n_motions:]`` for a starting batch,
    so ``begin64`` packs the real frames [0, n_motions) into that tail section.

    The constructor also makes the metadata-to-motion lookup robust to a moved
    project root and to isolated stale entries in the train/test manifest.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alternate_start_random = True
        self._validate_motion_index()

    @staticmethod
    def _mead_relative_key(path):
        """Return a root-independent MEAD key, or ``None`` if unavailable."""
        normalized = str(path).replace('\\', '/')
        marker = '/MEAD11/'
        if marker in normalized:
            return normalized.split(marker, 1)[1]

        # Keep this fallback conservative: the usual layout below is
        # speaker/front/emotion/level_x/file.wav.  It is only used when the
        # explicit MEAD11 marker is absent.
        parts = [part for part in normalized.split('/') if part]
        if len(parts) >= 5:
            return '/'.join(parts[-5:])
        return None

    def _validate_motion_index(self):
        """Repair relocated keys and drop only genuinely missing motion rows.

        ``03_merge_gt_motions.py`` stores the audio path string as the dictionary
        key.  Those paths are often absolute, so moving ADEFv4 -> ADEF_remake can
        make two otherwise identical manifests disagree.  We first prefer exact
        keys.  For misses, a MEAD-relative suffix is used only when it uniquely
        identifies one motion entry.  If no motion exists at all, the metadata
        row is excluded before any DataLoader worker is spawned.
        """
        if not isinstance(self.motion_data, dict):
            raise TypeError(
                f"Expected motion_data to be a dict, got {type(self.motion_data).__name__}"
            )

        exact_keys = set(self.motion_data.keys())
        relative_to_keys = {}
        for motion_key in exact_keys:
            relative = self._mead_relative_key(motion_key)
            if relative is not None:
                relative_to_keys.setdefault(relative, []).append(motion_key)

        valid_data = []
        relocated = []
        missing = []

        for metadata in self.all_data:
            audio_name = metadata["audio_name"]
            if audio_name in exact_keys:
                valid_data.append(metadata)
                continue

            relative = self._mead_relative_key(audio_name)
            candidates = relative_to_keys.get(relative, []) if relative is not None else []
            if len(candidates) == 1:
                motion_key = candidates[0]

                # Keep the audio path from the manifest when it exists.  When the
                # repository itself was moved and that old path no longer exists,
                # use the relocated motion-key path if it points to the audio file.
                repaired_audio_name = audio_name
                if not os.path.isfile(repaired_audio_name) and os.path.isfile(motion_key):
                    repaired_audio_name = motion_key

                repaired = dict(metadata)
                repaired["audio_name"] = repaired_audio_name

                # The legacy __getitem__ indexes motion_data by audio_name.  Add
                # a lightweight alias when the audio remains at the manifest path.
                if repaired_audio_name not in self.motion_data:
                    self.motion_data[repaired_audio_name] = self.motion_data[motion_key]
                    exact_keys.add(repaired_audio_name)

                valid_data.append(repaired)
                relocated.append((audio_name, repaired_audio_name, motion_key))
                continue

            missing.append(audio_name)

        if not valid_data:
            example = missing[0] if missing else '<none>'
            raise RuntimeError(
                "No MEAD manifest entry can be matched to front_all_motions.pkl. "
                "This usually means train/test.txt and front_all_motions.pkl were "
                f"generated from different datasets or roots. Example: {example}"
            )

        self.all_data = valid_data

        if relocated:
            print(
                "MEAD motion index: recovered "
                f"{len(relocated)} relocated path(s) by MEAD-relative matching."
            )

        if missing:
            preview = '\n  '.join(missing[:5])
            warnings.warn(
                "MEAD motion index: skipped "
                f"{len(missing)} manifest entr{'y' if len(missing) == 1 else 'ies'} "
                "that are absent from front_all_motions.pkl. "
                f"First missing path(s):\n  {preview}",
                RuntimeWarning,
            )
            print(
                "MEAD motion index: valid="
                f"{len(valid_data)}, missing_motion={len(missing)}, "
                f"relocated={len(relocated)}"
            )

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

"""Universal emotion-agnostic talking-motion dataset for Stage-1 training.

The loader intentionally ignores all emotion/level information.  It discovers
``.wav`` files recursively, so the following layouts can be mixed without
special cases:

MEAD-style hierarchy::

    videos/<speaker>/front/<emotion>/level_<n>/<clip>.wav/.pkl

Flat generic hierarchy::

    videos/RD_Radio*.wav/.pkl
    videos/WDA_*.wav/.pkl
    videos/WRA_*.wav/.pkl

Multiple roots may be supplied.  Motion can be read either from the per-clip
``.pkl`` next to each wav, or from one/more aggregate dictionaries such as
``front_all_motions.pkl``.  The returned interface matches the two-window
training logic used by ``dataset_EmotionLevel.py`` but contains no labels:

    audio_pair, coef_pair, sample_name

Both MEAD and generic videos are therefore treated uniformly as unlabeled
speech-motion examples in Stage 1.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils import data

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)


class GenericTalkingMotionDataset(data.Dataset):
    def __init__(
        self,
        video_roots: Sequence[str] | str,
        motion_template_path: str,
        aggregate_motion_files: Optional[Sequence[str] | str] = None,
        split: str = "train",
        split_file: Optional[str] = None,
        validation_ratio: float = 0.05,
        split_seed: int = 2026,
        coef_fps: int = 25,
        n_motions: int = 100,
        crop_strategy: str = "random",
        normalize_type: str = "mix",
        recursive: bool = True,
        require_local_motion: bool = False,
    ):
        super().__init__()
        self.video_roots = self._as_paths(video_roots)
        self.template_dict = pickle.load(
            open(motion_template_path, "rb")
        )
        self.aggregate_motion = self._load_aggregate_motion(
            aggregate_motion_files
        )
        self.require_local_motion = require_local_motion
        self.eps = 1e-9
        self.normalize_type = normalize_type
        self.coef_fps = coef_fps
        self.audio_unit = 16000.0 / coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * n_motions)
        self.coef_total_len = 2 * n_motions
        self.audio_total_len = round(
            self.audio_unit * self.coef_total_len
        )
        self.crop_strategy = crop_strategy

        discovered = self._discover(recursive=recursive)
        if split_file:
            allowed = self._read_split_file(split_file)
            discovered = [
                item for item in discovered
                if self._matches_split(item[0], allowed)
            ]
        else:
            discovered = self._deterministic_split(
                discovered,
                split=split,
                validation_ratio=validation_ratio,
                seed=split_seed,
            )

        if not discovered:
            raise RuntimeError(
                "No valid wav/motion pairs were found. Check video_roots, "
                "aggregate_motion_files and split settings."
            )
        self.samples = discovered
        print(
            f"GenericTalkingMotionDataset[{split}]: "
            f"{len(self.samples)} unlabeled clips from "
            f"{len(self.video_roots)} root(s)"
        )

    @staticmethod
    def _as_paths(value) -> List[Path]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            text = str(value)
            values = [part for part in text.split(",") if part]
        else:
            values = list(value)
        return [Path(item).expanduser().resolve() for item in values]

    @staticmethod
    def _read_split_file(path: str) -> set[str]:
        with open(path, "r", encoding="utf-8") as handle:
            return {
                line.strip().replace("\\", "/")
                for line in handle
                if line.strip()
            }

    @staticmethod
    def _matches_split(audio_path: Path, allowed: set[str]) -> bool:
        candidates = {
            audio_path.name,
            audio_path.stem,
            str(audio_path).replace("\\", "/"),
        }
        return any(
            candidate in allowed
            or any(candidate.endswith(item) for item in allowed)
            for candidate in candidates
        )

    @staticmethod
    def _load_aggregate_motion(files) -> Dict[str, dict]:
        result: Dict[str, dict] = {}
        if files is None:
            return result
        paths = GenericTalkingMotionDataset._as_paths(files)
        for path in paths:
            with open(path, "rb") as handle:
                content = pickle.load(handle)
            if not isinstance(content, dict):
                raise TypeError(
                    f"Aggregate motion file must contain a dict: {path}"
                )
            for key, value in content.items():
                key_text = str(key).replace("\\", "/")
                result[key_text] = value
                result[Path(key_text).name] = value
                result[Path(key_text).stem] = value
        return result

    def _resolve_motion(self, audio_path: Path) -> Optional[Tuple[str, object]]:
        local_path = audio_path.with_suffix(".pkl")
        if local_path.exists():
            return "local", local_path
        if self.require_local_motion:
            return None

        candidates = [
            str(audio_path).replace("\\", "/"),
            audio_path.name,
            audio_path.stem,
        ]
        for root in self.video_roots:
            try:
                relative = audio_path.relative_to(root)
                candidates.extend([
                    str(relative).replace("\\", "/"),
                    str(relative.with_suffix(".wav")).replace("\\", "/"),
                    str(relative.with_suffix(".pkl")).replace("\\", "/"),
                ])
            except ValueError:
                pass
        for key in candidates:
            if key in self.aggregate_motion:
                return "aggregate", key
        return None

    def _discover(self, recursive: bool) -> List[Tuple[Path, Tuple[str, object]]]:
        samples = []
        for root in self.video_roots:
            iterator: Iterable[Path]
            iterator = root.rglob("*.wav") if recursive else root.glob("*.wav")
            for audio_path in sorted(iterator):
                motion_ref = self._resolve_motion(audio_path)
                if motion_ref is not None:
                    samples.append((audio_path, motion_ref))
        return samples

    @staticmethod
    def _deterministic_split(
        samples,
        split: str,
        validation_ratio: float,
        seed: int,
    ):
        if split in {"all", "full"}:
            return samples
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")
        selected = []
        threshold = int(validation_ratio * 10000)
        for sample in samples:
            key = f"{seed}:{sample[0]}".encode("utf-8")
            bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 10000
            is_validation = bucket < threshold
            if (split == "train" and not is_validation) or (
                split in {"val", "test"} and is_validation
            ):
                selected.append(sample)
        return selected

    def __len__(self):
        return len(self.samples)

    def _load_motion(self, motion_ref: Tuple[str, object]) -> dict:
        source, value = motion_ref
        if source == "local":
            with open(value, "rb") as handle:
                motion = pickle.load(handle)
        else:
            motion = self.aggregate_motion[value]
        if "motion" not in motion:
            raise KeyError("Motion data must contain a 'motion' list")
        if "n_frames" not in motion:
            motion = dict(motion)
            motion["n_frames"] = len(motion["motion"])
        return motion

    def _repeat_to_minimum(
        self,
        motion_data: dict,
        audio_clip: torch.Tensor,
        min_frames: int,
    ) -> Tuple[dict, torch.Tensor]:
        frames = list(motion_data["motion"][:min_frames])
        audio_clip = audio_clip[:round(min_frames * self.audio_unit)]
        if not frames:
            raise ValueError("Empty motion sequence")
        while len(frames) < self.coef_total_len + 4:
            frames = frames + frames
            audio_clip = torch.cat([audio_clip, audio_clip], dim=0)
        target_frames = min(
            len(frames), int(audio_clip.shape[0] / self.audio_unit)
        )
        frames = frames[:target_frames]
        audio_clip = audio_clip[:round(target_frames * self.audio_unit)]
        return {
            "motion": frames,
            "n_frames": target_frames,
        }, audio_clip

    def _choose_start(self, sequence_length: int) -> int:
        maximum = sequence_length - self.coef_total_len
        if maximum < 0:
            raise ValueError("Sequence shorter than two training windows")
        if self.crop_strategy == "random":
            return np.random.randint(0, maximum + 1) if maximum else 0
        if self.crop_strategy == "begin":
            return 0
        if self.crop_strategy == "end":
            return maximum
        raise ValueError(f"Unknown crop strategy: {self.crop_strategy}")

    def _normalize_motion(self, motion_data: dict, start: int) -> Dict[str, torch.Tensor]:
        expressions = []
        poses = []
        template = self.template_dict
        for frame_index in range(start, start + self.coef_total_len):
            frame = motion_data["motion"][frame_index]
            if self.normalize_type != "mix":
                raise ValueError(
                    "GenericTalkingMotionDataset currently supports "
                    "normalize_type='mix' only"
                )
            expression = (
                np.asarray(frame["exp"]).reshape(-1)
                - template["mean_exp"]
            ) / (template["std_exp"] + self.eps)
            pose = np.concatenate([
                (np.asarray(frame["scale"]).reshape(-1) - template["min_scale"])
                / (template["max_scale"] - template["min_scale"] + self.eps),
                (np.asarray(frame["t"]).reshape(-1) - template["min_t"])
                / (template["max_t"] - template["min_t"] + self.eps),
                (np.asarray(frame["pitch"]).reshape(-1) - template["min_pitch"])
                / (template["max_pitch"] - template["min_pitch"] + self.eps),
                (np.asarray(frame["yaw"]).reshape(-1) - template["min_yaw"])
                / (template["max_yaw"] - template["min_yaw"] + self.eps),
                (np.asarray(frame["roll"]).reshape(-1) - template["min_roll"])
                / (template["max_roll"] - template["min_roll"] + self.eps),
            ])
            expressions.append(expression)
            poses.append(pose)
        return {
            "exp": torch.tensor(np.stack(expressions), dtype=torch.float32),
            "pose": torch.tensor(np.stack(poses), dtype=torch.float32),
        }

    def __getitem__(self, index):
        attempts = 0
        while attempts < len(self.samples):
            audio_path, motion_ref = self.samples[index]
            try:
                motion_data = self._load_motion(motion_ref)
                audio_clip, sample_rate = torchaudio.load(str(audio_path))
                if sample_rate != 16000:
                    audio_clip = torchaudio.functional.resample(
                        audio_clip, sample_rate, 16000
                    )
                audio_clip = audio_clip.mean(dim=0).float()
                audio_frames = int(audio_clip.shape[0] / self.audio_unit)
                motion_frames = int(motion_data["n_frames"])
                minimum = min(audio_frames, motion_frames)
                motion_data, audio_clip = self._repeat_to_minimum(
                    motion_data, audio_clip, minimum
                )
                start = self._choose_start(motion_data["n_frames"])
                end = start + self.coef_total_len
                coefficients = self._normalize_motion(motion_data, start)
                audio = audio_clip[
                    round(start * self.audio_unit):round(end * self.audio_unit)
                ]
                if audio.shape[0] != self.audio_total_len:
                    raise ValueError("Invalid cropped audio length")

                keys = ("exp", "pose")
                audio_pair = [
                    audio[:self.n_audio_samples].clone(),
                    audio[-self.n_audio_samples:].clone(),
                ]
                coefficient_pair = [
                    {
                        key: coefficients[key][:self.n_motions].clone()
                        for key in keys
                    },
                    {
                        key: coefficients[key][-self.n_motions:].clone()
                        for key in keys
                    },
                ]
                return audio_pair, coefficient_pair, audio_path.stem
            except (OSError, KeyError, ValueError, RuntimeError) as error:
                warnings.warn(f"Skip invalid sample {audio_path}: {error}")
                index = (index + 1) % len(self.samples)
                attempts += 1
        raise RuntimeError("Unable to load a valid sample from the dataset")

from __future__ import annotations

import hashlib
import pickle
import warnings
from pathlib import Path
from typing import Mapping, Optional, Sequence

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
        motion_template_path,
        motion_filenames=None,
        aggregate_motion_files=None,
        split="train",
        split_file=None,
        validation_ratio=0.05,
        split_seed=2026,
        coef_fps=25,
        n_motions=100,
        crop_strategy="random",
        normalize_type="mix",
        strict_absolute_paths=True,
        missing_audio_policy="skip",
        duplicate_policy="error",
        max_retries=20,
    ):
        super().__init__()
        if normalize_type != "mix":
            raise ValueError("Only normalize_type='mix' is supported")
        if crop_strategy not in {"random", "begin", "end"}:
            raise ValueError(f"Unknown crop strategy: {crop_strategy}")
        if missing_audio_policy not in {"skip", "error"}:
            raise ValueError("missing_audio_policy must be 'skip' or 'error'")
        if duplicate_policy not in {"error", "keep_first", "keep_last"}:
            raise ValueError(
                "duplicate_policy must be 'error', 'keep_first', or 'keep_last'"
            )
        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in [0, 1)")

        separate_files = self._parse_paths(motion_filenames)
        aggregate_files = self._parse_paths(aggregate_motion_files)
        if bool(separate_files) == bool(aggregate_files):
            raise ValueError(
                "Provide exactly one of motion_filenames or aggregate_motion_files"
            )
        source_files = separate_files or aggregate_files

        template_path = Path(motion_template_path).expanduser().resolve()
        with open(template_path, "rb") as file:
            self.template_dict = pickle.load(file)
        self._validate_template(self.template_dict, template_path)

        self.all_data = self._load_metadata(
            source_files,
            strict_absolute_paths=strict_absolute_paths,
            missing_audio_policy=missing_audio_policy,
            duplicate_policy=duplicate_policy,
        )
        if split_file is not None:
            allowed = self._read_split_file(split_file)
            self.all_data = [
                item
                for item in self.all_data
                if self._canonical_id(item["video_path"]) in allowed
                or self._canonical_id(item["audio_path"]) in allowed
            ]
        else:
            self.all_data = self._split_metadata(
                self.all_data,
                split=split,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
            )

        if not self.all_data:
            raise RuntimeError(f"No valid samples found for split={split!r}")

        self.coef_fps = int(coef_fps)
        self.audio_unit = 16000.0 / self.coef_fps
        self.n_motions = int(n_motions)
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy
        self.normalize_type = normalize_type
        self.max_retries = max(1, int(max_retries))
        self.eps = 1e-9

        print(
            f"GenericTalkingMotionDataset[{split}]: "
            f"{len(self.all_data)} samples from {len(source_files)} motion file(s)"
        )

    @staticmethod
    def _parse_paths(value) -> list[Path]:
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            raw_items = str(value).split(",")
        else:
            raw_items = []
            for item in value:
                raw_items.extend(str(item).split(","))

        paths = []
        for item in raw_items:
            item = item.strip()
            if not item:
                continue
            path = Path(item).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Motion file does not exist: {path}")
            paths.append(path)
        return paths

    @staticmethod
    def _validate_template(template: Mapping, template_path: Path) -> None:
        required = {
            "mean_exp",
            "std_exp",
            "min_scale",
            "max_scale",
            "min_t",
            "max_t",
            "min_pitch",
            "max_pitch",
            "min_yaw",
            "max_yaw",
            "min_roll",
            "max_roll",
        }
        missing = required.difference(template.keys())
        if missing:
            raise KeyError(
                f"Motion template {template_path} is missing keys: {sorted(missing)}"
            )

    @staticmethod
    def _normalise_motion_record(value, sample_path: Path) -> Mapping:
        if isinstance(value, list):
            value = {"motion": value, "n_frames": len(value)}
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Motion entry for {sample_path} must be a mapping, "
                f"got {type(value).__name__}"
            )
        if "motion" not in value and "motions" in value:
            value = dict(value)
            value["motion"] = value.pop("motions")
        if "motion" not in value:
            raise KeyError(f"Motion entry for {sample_path} has no 'motion' field")
        if "n_frames" not in value:
            value = dict(value)
            value["n_frames"] = len(value["motion"])
        return value

    @classmethod
    def _load_metadata(
        cls,
        source_files: Sequence[Path],
        strict_absolute_paths: bool,
        missing_audio_policy: str,
        duplicate_policy: str,
    ) -> list[dict]:
        metadata_by_path = {}
        skipped = 0

        for source_file in source_files:
            with open(source_file, "rb") as file:
                source_data = pickle.load(file)
            if not isinstance(source_data, Mapping):
                raise TypeError(
                    f"Motion file must contain a path-to-motion mapping: {source_file}"
                )

            for raw_path, raw_motion in source_data.items():
                video_path = Path(str(raw_path)).expanduser()
                if strict_absolute_paths and not video_path.is_absolute():
                    raise ValueError(
                        f"Motion key must be an absolute path: {raw_path!r}"
                    )
                if not video_path.is_absolute():
                    video_path = source_file.parent / video_path
                video_path = video_path.resolve(strict=False)
                audio_path = video_path.with_suffix(".wav")

                if not audio_path.is_file():
                    if missing_audio_policy == "error":
                        raise FileNotFoundError(
                            f"Derived audio file does not exist: {audio_path}"
                        )
                    skipped += 1
                    continue

                canonical = str(video_path)
                if canonical in metadata_by_path:
                    if duplicate_policy == "error":
                        raise ValueError(f"Duplicate motion sample: {video_path}")
                    if duplicate_policy == "keep_first":
                        continue

                metadata_by_path[canonical] = {
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "motion_data": cls._normalise_motion_record(
                        raw_motion, video_path
                    ),
                    "source_file": source_file,
                }

        if skipped:
            warnings.warn(f"Skipped {skipped} samples with missing audio files")
        return [metadata_by_path[key] for key in sorted(metadata_by_path)]

    @staticmethod
    def _canonical_id(path) -> str:
        text = str(path).strip().strip('"').strip("'").replace("\\", "/")
        if not text:
            return ""
        return Path(text).expanduser().resolve(strict=False).with_suffix("").as_posix()

    @classmethod
    def _read_split_file(cls, split_file) -> set[str]:
        split_path = Path(split_file).expanduser().resolve()
        allowed = set()
        with open(split_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                sample_id = cls._canonical_id(line)
                if sample_id:
                    allowed.add(sample_id)
        if not allowed:
            raise RuntimeError(f"Split file has no valid entries: {split_path}")
        return allowed

    @staticmethod
    def _split_metadata(
        metadata: Sequence[dict],
        split: str,
        validation_ratio: float,
        split_seed: int,
    ) -> list[dict]:
        split = split.lower()
        if split in {"all", "full"}:
            return list(metadata)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        threshold = int(validation_ratio * 10000)
        selected = []
        for item in metadata:
            key = f"{split_seed}:{item['video_path']}".encode("utf-8")
            bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 10000
            is_validation = bucket < threshold
            if split == "train" and not is_validation:
                selected.append(item)
            elif split in {"val", "test"} and is_validation:
                selected.append(item)
        return selected

    def __len__(self):
        return len(self.all_data)

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(str(audio_path))
        if audio.ndim != 2:
            raise ValueError(f"Unexpected audio shape: {tuple(audio.shape)}")
        audio = audio.mean(dim=0).float()
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        return audio

    def _align_and_extend(self, motion_data: Mapping, audio: torch.Tensor):
        motions = motion_data["motion"]
        motion_frames = min(int(motion_data["n_frames"]), len(motions))
        audio_frames = int(audio.shape[0] / self.audio_unit)
        valid_frames = min(motion_frames, audio_frames)
        if valid_frames <= 0:
            raise ValueError("Audio and motion have no aligned frames")

        motions = list(motions[:valid_frames])
        audio = audio[: round(valid_frames * self.audio_unit)]
        minimum_length = self.coef_total_len + 4
        while len(motions) < minimum_length:
            motions = motions + motions
            audio = torch.cat([audio, audio], dim=0)

        aligned_frames = min(len(motions), int(audio.shape[0] / self.audio_unit))
        motions = motions[:aligned_frames]
        audio = audio[: round(aligned_frames * self.audio_unit)]
        if aligned_frames < self.coef_total_len:
            raise ValueError("Aligned sequence is shorter than two windows")
        return motions, audio

    def _choose_start(self, sequence_length: int) -> int:
        maximum = sequence_length - self.coef_total_len
        if maximum < 0:
            raise ValueError("Sequence is shorter than two windows")
        if self.crop_strategy == "random":
            return int(np.random.randint(0, maximum + 1)) if maximum else 0
        if self.crop_strategy == "begin":
            return 0
        return maximum

    def _normalise_motion(self, frames: Sequence[Mapping]) -> dict[str, torch.Tensor]:
        template = self.template_dict
        expressions = []
        poses = []
        required_keys = {"exp", "scale", "t", "pitch", "yaw", "roll"}

        for frame in frames:
            missing = required_keys.difference(frame.keys())
            if missing:
                raise KeyError(f"Motion frame is missing keys: {sorted(missing)}")

            expression = (
                np.asarray(frame["exp"]).reshape(-1) - template["mean_exp"]
            ) / (template["std_exp"] + self.eps)
            pose = np.concatenate(
                [
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
                ]
            )
            expressions.append(expression.astype(np.float32))
            poses.append(pose.astype(np.float32))

        return {
            "exp": torch.from_numpy(np.stack(expressions)),
            "pose": torch.from_numpy(np.stack(poses)),
        }

    def _get_item(self, metadata: Mapping):
        audio = self._load_audio(metadata["audio_path"])
        motions, audio = self._align_and_extend(metadata["motion_data"], audio)
        start_frame = self._choose_start(len(motions))
        end_frame = start_frame + self.coef_total_len

        coefficients = self._normalise_motion(motions[start_frame:end_frame])
        cropped_audio = audio[
            round(start_frame * self.audio_unit) : round(end_frame * self.audio_unit)
        ]
        if cropped_audio.shape[0] != self.audio_total_len:
            raise ValueError(
                f"Invalid cropped audio length {cropped_audio.shape[0]}, "
                f"expected {self.audio_total_len}"
            )

        audio_pair = [
            cropped_audio[: self.n_audio_samples].clone(),
            cropped_audio[-self.n_audio_samples :].clone(),
        ]
        coefficient_pair = [
            {
                "exp": coefficients["exp"][: self.n_motions].clone(),
                "pose": coefficients["pose"][: self.n_motions].clone(),
            },
            {
                "exp": coefficients["exp"][-self.n_motions :].clone(),
                "pose": coefficients["pose"][-self.n_motions :].clone(),
            },
        ]
        return (
            audio_pair,
            coefficient_pair,
            torch.tensor(-1, dtype=torch.long),
            str(metadata["video_path"]),
        )

    def __getitem__(self, index):
        last_error: Optional[Exception] = None
        attempt_limit = min(self.max_retries, len(self.all_data))
        for offset in range(attempt_limit):
            metadata = self.all_data[(index + offset) % len(self.all_data)]
            try:
                return self._get_item(metadata)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                last_error = error
                warnings.warn(
                    f"Skip invalid sample {metadata['video_path']}: "
                    f"{type(error).__name__}: {error}"
                )
        raise RuntimeError(
            f"Unable to load a valid sample after {attempt_limit} attempts"
        ) from last_error

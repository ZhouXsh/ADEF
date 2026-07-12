"""Emotion-agnostic talking-motion dataset for Stage-1 training.

The sample index is built exclusively from one or more motion dictionaries.
No media root is scanned.

Each motion dictionary must have the following form::

    {
        "/absolute/path/to/video_001.mp4": {
            "motion": [...],
            "n_frames": 250,
        },
        "/absolute/path/to/video_002.mp4": {...},
    }

The corresponding audio path is derived by replacing the video suffix with
``.wav``.  Therefore, when the key is ``/data/a/video_001.mp4``, the loader
reads ``/data/a/video_001.wav``.

Two Stage-1 input modes are supported:

1. ``motion_filenames``: one motion dictionary per source dataset, e.g. MEAD,
   WDA/WRA and RD_Radio each have their own pickle file.
2. ``aggregate_motion_files``: one already-merged dictionary containing all
   videos.  A sequence is also accepted for convenience, although normally a
   single aggregate file is sufficient.

Exactly one of these modes must be supplied.  All dictionary keys are expected
to be complete absolute video paths.  This design intentionally removes the
old ``video_roots`` argument and all recursive file-system discovery.

The returned interface is compatible with the two-window training logic used
by ADEF::

    audio_pair, coefficient_pair, sample_path

where ``audio_pair`` and ``coefficient_pair`` each contain two consecutive
windows of ``n_motions`` frames.  Emotion labels are not returned.
"""

from __future__ import annotations

import hashlib
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchaudio
from torch.utils import data

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)

PathLike = Union[str, Path]
PathInput = Optional[Union[PathLike, Sequence[PathLike]]]


@dataclass(frozen=True)
class MotionSample:
    """One indexed video/audio/motion sample."""

    video_path: Path
    audio_path: Path
    motion_data: Mapping
    source_file: Path


class GenericTalkingMotionDataset(data.Dataset):
    """Read multiple motion dictionaries as one emotion-unlabelled dataset.

    Args:
        motion_template_path:
            A single template shared by Stage 1, Stage 2 and inference.
        motion_filenames:
            Separate per-dataset pickle files.  Every file maps absolute video
            paths to detailed motion dictionaries.
        aggregate_motion_files:
            One already-merged pickle file, or a sequence of merged files.
            Mutually exclusive with ``motion_filenames``.
        split:
            ``train``, ``val``, ``test``, ``all`` or ``full``.
        split_file:
            Optional text file that explicitly selects samples.  Each line may
            be an absolute video path, absolute audio path, basename or stem.
        validation_ratio:
            Deterministic validation fraction when ``split_file`` is absent.
        strict_absolute_paths:
            Require every motion-dictionary key to be an absolute path.
        missing_audio_policy:
            ``skip`` removes entries whose derived wav file is missing;
            ``error`` raises immediately.
        duplicate_policy:
            Action when two source dictionaries contain the same canonical
            video path: ``error``, ``keep_first`` or ``keep_last``.
        max_retries:
            Maximum number of alternative samples attempted after a corrupt
            item is encountered in ``__getitem__``.
    """

    def __init__(
        self,
        motion_template_path: PathLike,
        motion_filenames: PathInput = None,
        aggregate_motion_files: PathInput = None,
        split: str = "train",
        split_file: Optional[PathLike] = None,
        validation_ratio: float = 0.05,
        split_seed: int = 2026,
        coef_fps: int = 25,
        n_motions: int = 100,
        crop_strategy: str = "random",
        normalize_type: str = "mix",
        strict_absolute_paths: bool = True,
        missing_audio_policy: str = "skip",
        duplicate_policy: str = "error",
        max_retries: int = 20,
    ):
        super().__init__()

        separate_files = self._as_paths(motion_filenames)
        aggregate_files = self._as_paths(aggregate_motion_files)
        if bool(separate_files) == bool(aggregate_files):
            raise ValueError(
                "Exactly one of motion_filenames or aggregate_motion_files "
                "must be provided."
            )
        if missing_audio_policy not in {"skip", "error"}:
            raise ValueError(
                "missing_audio_policy must be 'skip' or 'error', got "
                f"{missing_audio_policy!r}"
            )
        if duplicate_policy not in {"error", "keep_first", "keep_last"}:
            raise ValueError(
                "duplicate_policy must be 'error', 'keep_first' or "
                f"'keep_last', got {duplicate_policy!r}"
            )
        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in [0, 1)")
        if normalize_type != "mix":
            raise ValueError(
                "GenericTalkingMotionDataset currently supports only "
                "normalize_type='mix'."
            )

        template_path = Path(motion_template_path).expanduser().resolve()
        with open(template_path, "rb") as handle:
            self.template_dict = pickle.load(handle)
        self._validate_template(self.template_dict, template_path)

        self.motion_source_mode = (
            "motion_filenames" if separate_files else "aggregate_motion_files"
        )
        source_files = separate_files or aggregate_files
        all_samples = self._load_motion_sources(
            source_files=source_files,
            strict_absolute_paths=strict_absolute_paths,
            missing_audio_policy=missing_audio_policy,
            duplicate_policy=duplicate_policy,
        )

        if split_file is not None:
            allowed = self._read_split_file(split_file)
            selected = [
                sample
                for sample in all_samples
                if self._matches_split(sample, allowed)
            ]
        else:
            selected = self._deterministic_split(
                all_samples,
                split=split,
                validation_ratio=validation_ratio,
                seed=split_seed,
            )

        if not selected:
            split_preview = []
            sample_preview = []

            if split_file is not None:
                split_preview = list(allowed)[:5]

            for sample in all_samples[:5]:
                sample_preview.append(
                    self._canonical_sample_id(sample.video_path)
                )

            raise RuntimeError(
                f"No samples remain for split={split!r}.\n"
                f"Loaded motion samples: {len(all_samples)}\n"
                f"Loaded split entries: "
                f"{len(allowed) if split_file is not None else 0}\n"
                f"First split IDs: {split_preview}\n"
                f"First motion IDs: {sample_preview}\n"
                "Check whether the directory portions of the paths agree."
            )

        self.samples = selected
        self.eps = 1e-9
        self.normalize_type = normalize_type
        self.coef_fps = int(coef_fps)
        self.audio_unit = 16000.0 / self.coef_fps
        self.n_motions = int(n_motions)
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = 2 * self.n_motions
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy
        self.max_retries = max(1, int(max_retries))

        print(
            f"GenericTalkingMotionDataset[{split}]: {len(self.samples)} clips "
            f"from {len(source_files)} {self.motion_source_mode} file(s)"
        )

    @staticmethod
    def _as_paths(value: PathInput) -> List[Path]:
        """Accept a path, a sequence, or comma-separated CLI text."""
        if value is None:
            return []
        if isinstance(value, (str, Path)):
            raw_values: Iterable[PathLike] = [
                item.strip() for item in str(value).split(",") if item.strip()
            ]
        else:
            expanded_values: List[PathLike] = []
            for item in value:
                expanded_values.extend(
                    part.strip()
                    for part in str(item).split(",")
                    if part.strip()
                )
            raw_values = expanded_values

        result: List[Path] = []
        for item in raw_values:
            path = Path(item).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Motion file does not exist: {path}")
            result.append(path)
        return result

    @staticmethod
    def _validate_template(template: Mapping, path: Path) -> None:
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
                f"Motion template {path} is missing keys: {sorted(missing)}"
            )

    @staticmethod
    def _normalise_motion_value(value, video_path: Path) -> Mapping:
        """Validate and lightly normalise one dictionary value."""
        if isinstance(value, list):
            value = {"motion": value, "n_frames": len(value)}
        if not isinstance(value, Mapping):
            raise TypeError(
                f"Motion entry for {video_path} must be a mapping, got "
                f"{type(value).__name__}"
            )
        if "motion" not in value and "motions" in value:
            value = dict(value)
            value["motion"] = value.pop("motions")
        if "motion" not in value:
            raise KeyError(
                f"Motion entry for {video_path} does not contain 'motion'."
            )
        if "n_frames" not in value:
            value = dict(value)
            value["n_frames"] = len(value["motion"])
        return value

    @classmethod
    def _load_motion_sources(
        cls,
        source_files: Sequence[Path],
        strict_absolute_paths: bool,
        missing_audio_policy: str,
        duplicate_policy: str,
    ) -> List[MotionSample]:
        by_video: Dict[Path, MotionSample] = {}
        missing_audio_count = 0

        for source_file in source_files:
            with open(source_file, "rb") as handle:
                content = pickle.load(handle)
            if not isinstance(content, Mapping):
                raise TypeError(
                    f"Motion file must contain a path-to-motion mapping: "
                    f"{source_file}"
                )

            for raw_path, raw_motion in content.items():
                video_path = Path(str(raw_path)).expanduser()
                if strict_absolute_paths and not video_path.is_absolute():
                    raise ValueError(
                        f"Motion key must be an absolute video path: "
                        f"{raw_path!r} in {source_file}"
                    )
                if not video_path.is_absolute():
                    # Optional compatibility mode.  Relative keys are resolved
                    # against the motion pickle's parent, never via video_roots.
                    video_path = source_file.parent / video_path
                video_path = video_path.resolve(strict=False)
                audio_path = video_path.with_suffix(".wav")

                if not audio_path.is_file():
                    message = (
                        f"Derived audio file is missing: {audio_path} "
                        f"(motion key: {video_path}, source: {source_file})"
                    )
                    if missing_audio_policy == "error":
                        raise FileNotFoundError(message)
                    missing_audio_count += 1
                    continue

                motion_data = cls._normalise_motion_value(
                    raw_motion, video_path
                )
                sample = MotionSample(
                    video_path=video_path,
                    audio_path=audio_path,
                    motion_data=motion_data,
                    source_file=source_file,
                )

                if video_path in by_video:
                    previous = by_video[video_path]
                    if duplicate_policy == "error":
                        raise ValueError(
                            "Duplicate video path across motion files: "
                            f"{video_path}\nfirst: {previous.source_file}\n"
                            f"second: {source_file}"
                        )
                    if duplicate_policy == "keep_first":
                        continue
                by_video[video_path] = sample

        if missing_audio_count:
            warnings.warn(
                f"Skipped {missing_audio_count} motion entries because the "
                "derived .wav file does not exist."
            )
        return sorted(by_video.values(), key=lambda item: str(item.video_path))

    @staticmethod
    def _canonical_sample_id(raw_path: Union[str, Path]) -> str:
        """Convert mp4/wav/pkl paths of the same sample to one canonical ID.

        Examples:
            /data/a/001.mp4
            /data/a/001.wav
            /data/a/001.pkl

        are all converted to:
            /data/a/001
        """
        text = str(raw_path).strip()

        # Support split lines accidentally surrounded by quotes.
        text = text.strip('"').strip("'")
        text = text.replace("\\", "/")

        if not text:
            return ""

        path = Path(text).expanduser()

        # strict=False means that the mp4 itself does not need to exist.
        # This is important when only wav exists on disk.
        path = path.resolve(strict=False)

        return path.with_suffix("").as_posix()

    @classmethod
    def _read_split_file(cls, path: PathLike) -> set[str]:
        """Read split entries and normalize all media suffixes."""
        split_path = Path(path).expanduser().resolve()

        sample_ids: set[str] = set()

        with open(split_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                sample_id = cls._canonical_sample_id(line)
                if sample_id:
                    sample_ids.add(sample_id)

        if not sample_ids:
            raise RuntimeError(
                f"Split file contains no valid sample paths: {split_path}"
            )

        return sample_ids

    @classmethod
    def _matches_split(
        cls,
        sample: MotionSample,
        allowed_sample_ids: set[str],
    ) -> bool:
        """Match video/audio paths independent of their suffix."""
        video_id = cls._canonical_sample_id(sample.video_path)
        audio_id = cls._canonical_sample_id(sample.audio_path)

        return (
            video_id in allowed_sample_ids
            or audio_id in allowed_sample_ids
        )

    @staticmethod
    def _deterministic_split(
        samples: Sequence[MotionSample],
        split: str,
        validation_ratio: float,
        seed: int,
    ) -> List[MotionSample]:
        split = split.lower()
        if split in {"all", "full"}:
            return list(samples)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split}")

        threshold = int(validation_ratio * 10000)
        result = []
        for sample in samples:
            key = f"{seed}:{sample.video_path}".encode("utf-8")
            bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 10000
            belongs_to_validation = bucket < threshold
            if split == "train" and not belongs_to_validation:
                result.append(sample)
            elif split in {"val", "test"} and belongs_to_validation:
                result.append(sample)
        return result

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, sample: MotionSample) -> torch.Tensor:
        audio, sample_rate = torchaudio.load(str(sample.audio_path))
        if audio.ndim != 2:
            raise ValueError(
                f"Unexpected audio shape {tuple(audio.shape)}: "
                f"{sample.audio_path}"
            )
        audio = audio.mean(dim=0).float()
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(
                audio, sample_rate, 16000
            )
        return audio

    def _align_and_repeat(
        self, sample: MotionSample, audio: torch.Tensor
    ) -> Tuple[List[Mapping], torch.Tensor]:
        motion_list = sample.motion_data["motion"]
        motion_frames = min(
            int(sample.motion_data["n_frames"]), len(motion_list)
        )
        audio_frames = int(audio.shape[0] / self.audio_unit)
        valid_frames = min(motion_frames, audio_frames)
        if valid_frames <= 0:
            raise ValueError(f"Empty aligned sequence: {sample.video_path}")

        frames = list(motion_list[:valid_frames])
        audio = audio[: round(valid_frames * self.audio_unit)]

        minimum_frames = self.coef_total_len + 4
        while len(frames) < minimum_frames:
            frames = frames + frames
            audio = torch.cat([audio, audio], dim=0)

        aligned_frames = min(
            len(frames), int(audio.shape[0] / self.audio_unit)
        )
        frames = frames[:aligned_frames]
        audio = audio[: round(aligned_frames * self.audio_unit)]
        if aligned_frames < self.coef_total_len:
            raise ValueError(
                f"Sequence remains shorter than {self.coef_total_len}: "
                f"{sample.video_path}"
            )
        return frames, audio

    def _choose_start(self, sequence_length: int) -> int:
        maximum = sequence_length - self.coef_total_len
        if maximum < 0:
            raise ValueError("Sequence shorter than two training windows")
        if self.crop_strategy == "random":
            return int(np.random.randint(0, maximum + 1)) if maximum else 0
        if self.crop_strategy == "begin":
            return 0
        if self.crop_strategy == "end":
            return maximum
        raise ValueError(f"Unknown crop strategy: {self.crop_strategy}")

    def _normalise_motion(
        self, frames: Sequence[Mapping]
    ) -> Dict[str, torch.Tensor]:
        template = self.template_dict
        expressions: List[np.ndarray] = []
        poses: List[np.ndarray] = []
        required_frame_keys = {
            "exp", "scale", "t", "pitch", "yaw", "roll"
        }

        for frame in frames:
            missing = required_frame_keys.difference(frame.keys())
            if missing:
                raise KeyError(f"Motion frame is missing keys: {sorted(missing)}")

            expression = (
                np.asarray(frame["exp"]).reshape(-1)
                - template["mean_exp"]
            ) / (template["std_exp"] + self.eps)
            pose = np.concatenate([
                (np.asarray(frame["scale"]).reshape(-1)
                 - template["min_scale"])
                / (template["max_scale"] - template["min_scale"] + self.eps),
                (np.asarray(frame["t"]).reshape(-1)
                 - template["min_t"])
                / (template["max_t"] - template["min_t"] + self.eps),
                (np.asarray(frame["pitch"]).reshape(-1)
                 - template["min_pitch"])
                / (template["max_pitch"] - template["min_pitch"] + self.eps),
                (np.asarray(frame["yaw"]).reshape(-1)
                 - template["min_yaw"])
                / (template["max_yaw"] - template["min_yaw"] + self.eps),
                (np.asarray(frame["roll"]).reshape(-1)
                 - template["min_roll"])
                / (template["max_roll"] - template["min_roll"] + self.eps),
            ])
            expressions.append(expression.astype(np.float32))
            poses.append(pose.astype(np.float32))

        return {
            "exp": torch.from_numpy(np.stack(expressions)),
            "pose": torch.from_numpy(np.stack(poses)),
        }

    def _get_item(self, sample: MotionSample):
        audio = self._load_audio(sample)
        frames, audio = self._align_and_repeat(sample, audio)
        start = self._choose_start(len(frames))
        end = start + self.coef_total_len

        coefficients = self._normalise_motion(frames[start:end])
        cropped_audio = audio[
            round(start * self.audio_unit): round(end * self.audio_unit)
        ]
        if cropped_audio.shape[0] != self.audio_total_len:
            raise ValueError(
                f"Invalid cropped audio length {cropped_audio.shape[0]} "
                f"for {sample.audio_path}; expected {self.audio_total_len}."
            )

        keys = ("exp", "pose")
        audio_pair = [
            cropped_audio[: self.n_audio_samples].clone(),
            cropped_audio[-self.n_audio_samples:].clone(),
        ]
        coefficient_pair = [
            {
                key: coefficients[key][: self.n_motions].clone()
                for key in keys
            },
            {
                key: coefficients[key][-self.n_motions:].clone()
                for key in keys
            },
        ]
        return audio_pair, coefficient_pair, str(sample.video_path)

    def __getitem__(self, index: int):
        last_error: Optional[Exception] = None
        attempt_limit = min(self.max_retries, len(self.samples))
        for offset in range(attempt_limit):
            sample = self.samples[(index + offset) % len(self.samples)]
            try:
                return self._get_item(sample)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                last_error = error
                warnings.warn(
                    f"Skip invalid sample {sample.video_path}: "
                    f"{type(error).__name__}: {error}"
                )
        raise RuntimeError(
            f"Unable to load a valid sample after {attempt_limit} attempts"
        ) from last_error

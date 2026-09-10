"""Runtime emotion2vec extraction and temporal alignment for ADEF inference.

The implementation follows ``src/my_prepare/06_extract_emotion2vec.py`` but is
safe for online inference:

* FunASR is imported lazily only when extraction is required;
* precomputed utterance/frame ``.npy`` files can be supplied directly;
* generated features are cached under separate utterance/frame directories;
* both FunASR return dictionaries and files written by ``output_dir`` are
  supported;
* frame features are converted to ``[1, T, D]`` and can be aligned to ADEF's
  padded video-frame timeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Emotion2VecPaths:
    utterance: Path
    frame: Path


class Emotion2VecExtractor:
    """Lazy wrapper around FunASR ``AutoModel`` for emotion2vec features."""

    def __init__(
        self,
        model_id: str = "iic/emotion2vec_plus_large",
        hub: str = "ms",
        device: Optional[str] = None,
        expected_dim: Optional[int] = 1024,
    ):
        self.model_id = model_id
        self.hub = hub
        self.device = device
        self.expected_dim = expected_dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from funasr import AutoModel
            except ImportError as error:
                raise ImportError(
                    "emotion2vec extraction requires FunASR. Install funasr, "
                    "or provide --emotion2vec-utterance-path and "
                    "--emotion2vec-frame-path."
                ) from error

            kwargs = {"model": self.model_id, "hub": self.hub}
            if self.device:
                kwargs["device"] = self.device
            self._model = AutoModel(**kwargs)
        return self._model

    @staticmethod
    def _normalize_utterance(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32)
        if value.ndim == 0:
            raise ValueError("Invalid scalar utterance emotion2vec feature")
        if value.ndim > 1:
            value = value.reshape(-1, value.shape[-1]).mean(axis=0)
        return value.reshape(-1)

    @staticmethod
    def _normalize_frame(array: np.ndarray) -> np.ndarray:
        value = np.asarray(array, dtype=np.float32)
        value = np.squeeze(value)
        if value.ndim == 1:
            value = value[None, :]
        if value.ndim != 2:
            raise ValueError(
                f"Frame emotion2vec feature must be [T, D], got {value.shape}"
            )
        return value

    def _validate(self, utterance: np.ndarray, frame: np.ndarray) -> None:
        if utterance.shape[-1] != frame.shape[-1]:
            raise ValueError(
                "Utterance/frame emotion2vec dimension mismatch: "
                f"{utterance.shape} vs {frame.shape}"
            )
        if self.expected_dim and utterance.shape[-1] != self.expected_dim:
            raise ValueError(
                f"Expected emotion2vec dim {self.expected_dim}, "
                f"got {utterance.shape[-1]}"
            )

    @staticmethod
    def _feature_from_result(result: Any) -> Optional[np.ndarray]:
        """Extract ``feats`` from common FunASR return structures."""
        if result is None:
            return None
        if isinstance(result, dict):
            if "feats" in result:
                return np.asarray(result["feats"])
            return None
        if isinstance(result, (list, tuple)):
            for item in result:
                feature = Emotion2VecExtractor._feature_from_result(item)
                if feature is not None:
                    return feature
        return None

    @staticmethod
    def _find_generated_feature(
        output_dir: Path,
        stem: str,
        files_before: set[Path],
    ) -> Optional[Path]:
        preferred = output_dir / f"{stem}.npy"
        if preferred.exists():
            return preferred
        candidates = sorted(output_dir.rglob("*.npy"))
        new_files = [path for path in candidates if path not in files_before]
        if len(new_files) == 1:
            return new_files[0]
        stem_matches = [path for path in candidates if path.stem == stem]
        if len(stem_matches) == 1:
            return stem_matches[0]
        return None

    def _generate_one(
        self,
        audio_path: Path,
        output_path: Path,
        granularity: str,
    ) -> np.ndarray:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        files_before = set(output_path.parent.rglob("*.npy"))
        result = self._get_model().generate(
            str(audio_path),
            output_dir=str(output_path.parent),
            granularity=granularity,
            extract_embedding=True,
        )
        feature = self._feature_from_result(result)
        if feature is None:
            generated = self._find_generated_feature(
                output_path.parent, audio_path.stem, files_before
            )
            if generated is None:
                raise FileNotFoundError(
                    "FunASR returned no 'feats' and no generated .npy could "
                    f"be resolved in {output_path.parent}"
                )
            feature = np.load(generated)
        np.save(output_path, np.asarray(feature, dtype=np.float32))
        return np.asarray(feature, dtype=np.float32)

    @staticmethod
    def resolve_cache_paths(
        audio_path: str | Path,
        cache_dir: Optional[str | Path] = None,
    ) -> Emotion2VecPaths:
        audio_path = Path(audio_path).expanduser().resolve()
        root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else audio_path.parent / ".adef_emotion2vec"
        )
        return Emotion2VecPaths(
            utterance=root / "utterance" / f"{audio_path.stem}.npy",
            frame=root / "frame" / f"{audio_path.stem}.npy",
        )

    def extract(
        self,
        audio_path: str | Path,
        utterance_path: Optional[str | Path] = None,
        frame_path: Optional[str | Path] = None,
        cache_dir: Optional[str | Path] = None,
        force: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Emotion2VecPaths]:
        audio_path = Path(audio_path).expanduser().resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        cache_paths = self.resolve_cache_paths(audio_path, cache_dir)
        resolved_paths = Emotion2VecPaths(
            utterance=(
                Path(utterance_path).expanduser().resolve()
                if utterance_path else cache_paths.utterance
            ),
            frame=(
                Path(frame_path).expanduser().resolve()
                if frame_path else cache_paths.frame
            ),
        )

        if resolved_paths.utterance.exists() and not force:
            utterance = np.load(resolved_paths.utterance)
        else:
            if utterance_path and not resolved_paths.utterance.exists():
                raise FileNotFoundError(
                    f"Explicit utterance feature not found: {resolved_paths.utterance}"
                )
            utterance = self._generate_one(
                audio_path, resolved_paths.utterance, "utterance"
            )

        if resolved_paths.frame.exists() and not force:
            frame = np.load(resolved_paths.frame)
        else:
            if frame_path and not resolved_paths.frame.exists():
                raise FileNotFoundError(
                    f"Explicit frame feature not found: {resolved_paths.frame}"
                )
            frame = self._generate_one(
                audio_path, resolved_paths.frame, "frame"
            )

        utterance = self._normalize_utterance(utterance)
        frame = self._normalize_frame(frame)
        self._validate(utterance, frame)
        return utterance, frame, resolved_paths

    @staticmethod
    def align_frame_timeline(
        frame_feature: np.ndarray | torch.Tensor,
        raw_audio_samples: int,
        fps: int,
        sample_rate: int,
        left_padding_samples: int,
        right_padding_samples: int,
        padded_clip_frames: int,
        total_window_frames: int,
        final_pad_mode: str = "zero",
    ) -> torch.Tensor:
        """Align raw frame-level features to the exact ADEF window timeline.

        The source feature is first interpolated to the unpadded audio duration.
        The legacy wrapper's fixed left/right audio padding is represented by
        zero emotion frames. The remaining tail required to fill the last DiT
        window follows the wrapper's ``zero`` or ``replicate`` pad mode.
        """
        feature = torch.as_tensor(frame_feature, dtype=torch.float32)
        if feature.ndim == 2:
            feature = feature.unsqueeze(0)
        if feature.ndim != 3:
            raise ValueError(
                f"Expected frame feature [B, T, D], got {feature.shape}"
            )

        left_frames = round(left_padding_samples / sample_rate * fps)
        right_frames = round(right_padding_samples / sample_rate * fps)
        left_frames = min(left_frames, padded_clip_frames)
        right_frames = min(
            right_frames, max(0, padded_clip_frames - left_frames)
        )
        middle_frames = max(
            0, padded_clip_frames - left_frames - right_frames
        )

        if middle_frames > 0:
            middle = F.interpolate(
                feature.transpose(1, 2),
                size=middle_frames,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        else:
            middle = feature[:, :0]

        zero_left = torch.zeros(
            feature.shape[0], left_frames, feature.shape[-1],
            dtype=feature.dtype,
        )
        zero_right = torch.zeros(
            feature.shape[0], right_frames, feature.shape[-1],
            dtype=feature.dtype,
        )
        timeline = torch.cat([zero_left, middle, zero_right], dim=1)

        if timeline.shape[1] < total_window_frames:
            count = total_window_frames - timeline.shape[1]
            if final_pad_mode == "zero" or timeline.shape[1] == 0:
                tail = torch.zeros(
                    timeline.shape[0], count, timeline.shape[-1],
                    dtype=timeline.dtype,
                )
            elif final_pad_mode == "replicate":
                tail = timeline[:, -1:].expand(-1, count, -1).clone()
            else:
                raise ValueError(f"Unknown pad mode: {final_pad_mode}")
            timeline = torch.cat([timeline, tail], dim=1)
        elif timeline.shape[1] > total_window_frames:
            timeline = timeline[:, :total_window_frames]

        if timeline.shape[1] != total_window_frames:
            raise RuntimeError(
                f"Aligned emotion2vec timeline length {timeline.shape[1]} "
                f"!= expected {total_window_frames}"
            )
        return timeline.contiguous()

import os
import hashlib
import pickle
import warnings
from pathlib import Path

import torchaudio
import numpy as np
import torch
from torch.utils import data

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class GenericTalkingMotionDataset(data.Dataset):
    def __init__(self,
                 motion_template_path,
                 motion_filenames=None,
                 aggregate_motion_files=None,
                 split="train",
                 split_file=None,
                 validation_ratio=0.0,
                 split_seed=2026,
                 coef_fps=25,
                 n_motions=64,
                 n_prev_motions=16,
                 crop_strategy="random",
                 normalize_type="mix",
                 strict_absolute_paths=True,
                 missing_audio_policy="skip",
                 duplicate_policy="keep_first"):
        self.template_dir = Path(motion_template_path).expanduser().resolve()
        self.template_dict = pickle.load(open(self.template_dir, 'rb'))
        self.eps = 1e-9
        self.normalize_type = normalize_type
        self.split = split

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_prev_motions + self.n_motions
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy

        if self.normalize_type != "mix":
            raise RuntimeError("GenericTalkingMotionDataset only supports normalize_type='mix'")
        if missing_audio_policy not in ["skip", "error"]:
            raise ValueError("missing_audio_policy should be 'skip' or 'error'")
        if duplicate_policy not in ["error", "keep_first", "keep_last"]:
            raise ValueError("duplicate_policy should be 'error', 'keep_first' or 'keep_last'")
        if not 0.0 <= validation_ratio < 1.0:
            raise ValueError("validation_ratio should be in [0, 1)")

        motion_files = self._parse_motion_files(motion_filenames)
        aggregate_files = self._parse_motion_files(aggregate_motion_files)
        if bool(motion_files) == bool(aggregate_files):
            raise ValueError(
                "Exactly one of motion_filenames or aggregate_motion_files should be provided."
            )
        source_files = motion_files if motion_files else aggregate_files

        # 直接在初始化阶段筛除长度不足的数据，不做任何复制/拼接延长。
        all_data, motion_data = self._load_and_filter_data(
            source_files,
            strict_absolute_paths=strict_absolute_paths,
            missing_audio_policy=missing_audio_policy,
            duplicate_policy=duplicate_policy,
        )

        if split_file is not None:
            all_data = self._filter_by_split_file(all_data, split_file)
        elif validation_ratio > 0:
            all_data = self._deterministic_split(
                all_data,
                split=split,
                validation_ratio=validation_ratio,
                split_seed=split_seed,
            )

        if len(all_data) == 0:
            raise RuntimeError(f"No valid generic talking-motion samples remain for split={split}")

        self.all_data = all_data
        self.motion_data = motion_data
        print(f"load generic motion data done... split={split}, valid videos: {len(self.all_data)}")

    def __len__(self, ):
        return len(self.all_data)

    @staticmethod
    def _parse_motion_files(value):
        if value is None:
            return []

        if isinstance(value, (str, Path)):
            raw_list = [item.strip() for item in str(value).split(',') if item.strip()]
        else:
            raw_list = []
            for item in value:
                raw_list.extend([part.strip() for part in str(item).split(',') if part.strip()])

        result = []
        for raw_path in raw_list:
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(f"motion file not found: {path}")
            result.append(path)
        return result

    @staticmethod
    def _normalize_motion_data(motion_data):
        if isinstance(motion_data, list):
            motion_data = {
                "motion": motion_data,
                "n_frames": len(motion_data),
            }
        elif isinstance(motion_data, dict):
            if "motion" not in motion_data and "motions" in motion_data:
                motion_data = dict(motion_data)
                motion_data["motion"] = motion_data.pop("motions")
            if "motion" not in motion_data:
                raise KeyError("motion data does not contain 'motion'")
            if "n_frames" not in motion_data:
                motion_data = dict(motion_data)
                motion_data["n_frames"] = len(motion_data["motion"])
        else:
            raise TypeError(f"invalid motion data type: {type(motion_data)}")

        return motion_data

    def _load_and_filter_data(self,
                              source_files,
                              strict_absolute_paths=True,
                              missing_audio_policy="skip",
                              duplicate_policy="keep_first"):
        motion_data_all = {}
        json_data = []
        known_video_paths = {}

        skip_short = 0
        skip_missing_audio = 0
        skip_invalid = 0

        for source_file in source_files:
            source_motion_data = pickle.load(open(source_file, "rb"))
            if not isinstance(source_motion_data, dict):
                raise TypeError(f"motion file should contain a dict: {source_file}")

            for raw_video_path, raw_motion_data in source_motion_data.items():
                try:
                    video_path = Path(str(raw_video_path)).expanduser()
                    if strict_absolute_paths and not video_path.is_absolute():
                        raise ValueError(
                            f"motion key should be an absolute path: {raw_video_path}"
                        )
                    if not video_path.is_absolute():
                        video_path = source_file.parent / video_path
                    video_path = video_path.resolve(strict=False)
                    audio_path = video_path.with_suffix('.wav')

                    if not audio_path.is_file():
                        if missing_audio_policy == "error":
                            raise FileNotFoundError(f"audio file not found: {audio_path}")
                        skip_missing_audio += 1
                        continue

                    motion_data = self._normalize_motion_data(raw_motion_data)
                    motion_frames = min(
                        int(motion_data["n_frames"]),
                        len(motion_data["motion"]),
                    )

                    audio_info = torchaudio.info(str(audio_path))
                    if audio_info.sample_rate <= 0 or audio_info.num_frames <= 0:
                        skip_invalid += 1
                        continue
                    audio_seconds = audio_info.num_frames / float(audio_info.sample_rate)
                    audio_frames = int(audio_seconds * self.coef_fps)
                    min_frames = min(audio_frames, motion_frames)

                    # 与 MEAD clear 数据集一致：至少保证 80 帧训练窗口之外还有少量裁剪余量。
                    if min_frames < self.coef_total_len + 4:
                        skip_short += 1
                        continue

                    video_key = str(video_path)
                    if video_key in known_video_paths:
                        if duplicate_policy == "error":
                            raise ValueError(
                                f"duplicate video path: {video_key}\n"
                                f"first: {known_video_paths[video_key]}\n"
                                f"second: {source_file}"
                            )
                        if duplicate_policy == "keep_first":
                            continue

                        # keep_last：移除之前的 metadata，后面的样本覆盖 motion。
                        json_data = [item for item in json_data if item["video_name"] != video_key]

                    known_video_paths[video_key] = str(source_file)
                    audio_key = str(audio_path)
                    motion_data_all[audio_key] = motion_data
                    json_data.append({
                        "video_name": video_key,
                        "audio_name": audio_key,
                        "source_file": str(source_file),
                    })
                except (OSError, KeyError, TypeError, ValueError) as error:
                    if missing_audio_policy == "error" and isinstance(error, FileNotFoundError):
                        raise
                    skip_invalid += 1
                    warnings.warn(
                        f"skip invalid generic sample {raw_video_path}: "
                        f"{type(error).__name__}: {error}"
                    )

        print(
            f"generic dataset filter: valid={len(json_data)}, "
            f"short={skip_short}, missing_audio={skip_missing_audio}, invalid={skip_invalid}"
        )
        return json_data, motion_data_all

    @staticmethod
    def _canonical_sample_id(raw_path):
        text = str(raw_path).strip().strip('"').strip("'").replace('\\', '/')
        if not text:
            return ""
        path = Path(text).expanduser().resolve(strict=False)
        return path.with_suffix('').as_posix()

    def _filter_by_split_file(self, all_data, split_file):
        split_file = Path(split_file).expanduser().resolve()
        allowed = set()
        with open(split_file, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                sample_id = self._canonical_sample_id(line)
                if sample_id:
                    allowed.add(sample_id)

        result = []
        for metadata in all_data:
            video_id = self._canonical_sample_id(metadata["video_name"])
            audio_id = self._canonical_sample_id(metadata["audio_name"])
            if video_id in allowed or audio_id in allowed:
                result.append(metadata)
        return result

    def _deterministic_split(self, all_data, split="train", validation_ratio=0.05, split_seed=2026):
        split = split.lower()
        if split in ["all", "full"]:
            return all_data
        if split not in ["train", "val", "test"]:
            raise ValueError(f"unknown split: {split}")

        threshold = int(validation_ratio * 10000)
        result = []
        for metadata in all_data:
            key = f"{split_seed}:{metadata['video_name']}".encode("utf-8")
            bucket = int(hashlib.sha1(key).hexdigest()[:8], 16) % 10000
            is_validation = bucket < threshold
            if split == "train" and not is_validation:
                result.append(metadata)
            elif split in ["val", "test"] and is_validation:
                result.append(metadata)
        return result

    def check_motion_length(self, motion_data, min_frames):
        """检查运动数据长度是否足够，不足则返回None（初始化时正常已被筛掉）。"""
        if min_frames < self.coef_total_len + 4:
            return None

        exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
        for frame_index in range(min_frames):
            exp_list.append(motion_data["motion"][frame_index]["exp"])
            t_list.append(motion_data["motion"][frame_index]["t"])
            scale_list.append(motion_data["motion"][frame_index]["scale"])
            pitch_list.append(motion_data["motion"][frame_index]["pitch"])
            yaw_list.append(motion_data["motion"][frame_index]["yaw"])
            roll_list.append(motion_data["motion"][frame_index]["roll"])

        motion_new = {"motion": []}
        for i in range(len(exp_list)):
            motion = {
                "exp": exp_list[i],
                "t": t_list[i],
                "scale": scale_list[i],
                "pitch": pitch_list[i],
                "yaw": yaw_list[i],
                "roll": roll_list[i],
            }
            motion_new["motion"].append(motion)
        motion_new["n_frames"] = len(exp_list)
        return motion_new

    # 获取一个 80 帧连续片段：前 16 帧作为 prev，后 64 帧作为 current。
    # 通用数据集没有情感标签，因此这里只返回 audio 和 coef_single。
    def __getitem__(self, index):
        has_valid_audio = False
        retry_count = 0
        max_retry = min(20, len(self.all_data))

        while not has_valid_audio:
            metadata = self.all_data[index]

            try:
                motion_data = self.motion_data[metadata["audio_name"]]

                audio_path = metadata["audio_name"]
                audio_clip, sr = torchaudio.load(audio_path)
                if audio_clip.ndim == 2:
                    audio_clip = audio_clip.mean(dim=0)
                else:
                    audio_clip = audio_clip.squeeze()
                audio_clip = audio_clip.float()
                if sr != 16000:
                    audio_clip = torchaudio.functional.resample(audio_clip, sr, 16000)

                audio_frames = int(audio_clip.shape[0] / self.audio_unit)
                motion_frames = min(motion_data["n_frames"], len(motion_data["motion"]))
                min_frames = min(audio_frames, motion_frames)

                motion_data = self.check_motion_length(motion_data, min_frames)
                if motion_data is None:
                    raise ValueError(
                        f"short sample after loading: min_frames={min_frames}, "
                        f"required={self.coef_total_len + 4}"
                    )

                audio_clip = audio_clip[:int(min_frames * self.audio_unit)]

                seq_len = motion_data["n_frames"]
                assert int(seq_len * self.audio_unit) == audio_clip.shape[0], \
                    f'frame mismatch: {seq_len * self.audio_unit} != {audio_clip.shape[0]}'

                if self.crop_strategy == 'random':
                    start_frame = np.random.randint(0, seq_len - self.coef_total_len - 2)
                elif self.crop_strategy == 'begin':
                    start_frame = 0
                elif self.crop_strategy == 'end':
                    start_frame = seq_len - self.coef_total_len - 2
                else:
                    raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
                end_frame = start_frame + self.coef_total_len

                Generic_template_dict = self.template_dict

                coef_keys = ["exp", "pose"]
                coef_dict = {k: [] for k in coef_keys}
                for frame_idx in range(start_frame, end_frame):
                    for coef_key in coef_keys:
                        if coef_key == "exp":
                            if self.normalize_type == "mix":
                                normalized_exp = (
                                    motion_data['motion'][frame_idx]["exp"].flatten()
                                    - Generic_template_dict["mean_exp"]
                                ) / (Generic_template_dict["std_exp"] + self.eps)
                            else:
                                raise RuntimeError("error")
                            coef_dict[coef_key].append([normalized_exp, ])
                        elif coef_key == "pose":
                            if self.normalize_type == "mix":
                                pose_data = np.concatenate((
                                    (motion_data['motion'][frame_idx]["scale"].flatten() - Generic_template_dict["mean_scale"]) / (Generic_template_dict["std_scale"] + self.eps),
                                    (motion_data['motion'][frame_idx]["t"].flatten() - Generic_template_dict["mean_t"]) / (Generic_template_dict["std_t"] + self.eps),
                                    (motion_data['motion'][frame_idx]["pitch"].flatten() - Generic_template_dict["mean_pitch"]) / (Generic_template_dict["std_pitch"] + self.eps),
                                    (motion_data['motion'][frame_idx]["yaw"].flatten() - Generic_template_dict["mean_yaw"]) / (Generic_template_dict["std_yaw"] + self.eps),
                                    (motion_data['motion'][frame_idx]["roll"].flatten() - Generic_template_dict["mean_roll"]) / (Generic_template_dict["std_roll"] + self.eps),
                                ))
                            else:
                                raise RuntimeError("pose data error")
                            coef_dict[coef_key].append([pose_data, ])
                        else:
                            raise RuntimeError("coef_key error: ", coef_key)

                coef_dict = {
                    k: torch.tensor(np.concatenate(coef_dict[k], axis=0), dtype=torch.float32)
                    for k in coef_keys
                }
                assert coef_dict['exp'].shape[0] == self.coef_total_len, \
                    f'Invalid coef length: {coef_dict["exp"].shape[0]}'

                audio = []
                audio.append(
                    audio_clip[
                        round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)
                    ]
                )
                audio = torch.cat(audio, dim=0)
                if not (audio.shape[0] == self.audio_total_len):
                    raise ValueError(
                        f"audio length invalid! audio: {audio.shape[0]}, "
                        f"expected: {self.audio_total_len}"
                    )

                keys = ['exp', 'pose']
                coef_single = {k: coef_dict[k].clone() for k in keys}
                has_valid_audio = True
                return audio, coef_single

            except (OSError, KeyError, TypeError, ValueError, RuntimeError, AssertionError) as error:
                retry_count += 1
                if retry_count >= max_retry:
                    raise RuntimeError(
                        f"Unable to load a valid generic sample after {retry_count} retries; "
                        f"last sample: {metadata['video_name']}"
                    ) from error
                warnings.warn(
                    f"skip invalid generic sample {metadata['video_name']}: "
                    f"{type(error).__name__}: {error}"
                )
                index = np.random.randint(0, len(self.all_data))

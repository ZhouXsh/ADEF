import os
import pickle
import warnings

import numpy as np
import torch
import torchaudio
from torch.utils import data


emo_list = [
    "angry", "contempt", "disgusted", "fear",
    "happy", "neutral", "sad", "surprised",
]

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead."
)


class EmoLevelDataset(data.Dataset):
    """MEAD dataset returning one canonical-keypoint token per motion window."""

    def __init__(
        self,
        root_dir="src/my_prepare/",
        motion_filename="front_all_motions.pkl",
        motion_template_filename="motion_template.pkl",
        split="train",
        coef_fps=25,
        n_motions=100,
        crop_strategy="random",
        normalize_type="mix",
    ):
        self.template_dict = pickle.load(
            open(os.path.join(root_dir, motion_template_filename), "rb")
        )
        self.motion_data = pickle.load(
            open(os.path.join(root_dir, motion_filename), "rb")
        )
        self.eps = 1e-9
        self.normalize_type = normalize_type

        split_file = "train.txt" if split == "train" else "test.txt"
        with open(os.path.join(root_dir, split_file), "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
        self.all_data = [
            {
                "video_name": line,
                "audio_name": line[:-4] + ".wav",
                "motion_name": line[:-4] + ".pkl",
            }
            for line in lines
        ]

        self.coef_fps = coef_fps
        self.audio_unit = 16000.0 / coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * n_motions)
        self.coef_total_len = n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy

    def __len__(self):
        return len(self.all_data)

    def check_motion_length(self, motion_data, min_frames):
        keys = ["exp", "t", "scale", "pitch", "yaw", "roll", "kp"]
        values = {key: [] for key in keys}
        for frame_index in range(min_frames):
            frame = motion_data["motion"][frame_index]
            for key in keys:
                values[key].append(frame[key])

        repeat = 0
        if min_frames <= self.coef_total_len + 4:
            while len(values["exp"]) < self.coef_total_len + 4:
                for key in keys:
                    values[key] = values[key] * 2
                repeat += 1

        motion_new = {
            "motion": [
                {key: values[key][i] for key in keys}
                for i in range(len(values["exp"]))
            ],
            "n_frames": len(values["exp"]),
        }
        if repeat > 0:
            motion_new["repeat"] = repeat
        return motion_new

    def _normalize_motion_frame(self, frame):
        if self.normalize_type != "mix":
            raise RuntimeError("Only normalize_type='mix' is supported")
        template = self.template_dict
        exp = (frame["exp"].flatten() - template["mean_exp"]) / (
            template["std_exp"] + self.eps
        )
        pose = np.concatenate((
            (frame["scale"].flatten() - template["min_scale"]) /
            (template["max_scale"] - template["min_scale"] + self.eps),
            (frame["t"].flatten() - template["min_t"]) /
            (template["max_t"] - template["min_t"] + self.eps),
            (frame["pitch"].flatten() - template["min_pitch"]) /
            (template["max_pitch"] - template["min_pitch"] + self.eps),
            (frame["yaw"].flatten() - template["min_yaw"]) /
            (template["max_yaw"] - template["min_yaw"] + self.eps),
            (frame["roll"].flatten() - template["min_roll"]) /
            (template["max_roll"] - template["min_roll"] + self.eps),
        ))
        return exp, pose

    @staticmethod
    def _build_canonical_token(frame):
        canonical_kp = torch.as_tensor(
            frame["kp"].reshape(-1), dtype=torch.float32
        )
        if canonical_kp.numel() != 63:
            raise ValueError(
                f"Expected 63 canonical keypoint values, got {canonical_kp.numel()}"
            )
        return torch.cat(
            [torch.zeros(7, dtype=canonical_kp.dtype), canonical_kp], dim=0
        ).unsqueeze(0)

    def __getitem__(self, index):
        while True:
            metadata = self.all_data[index]
            emotype = metadata["video_name"].split("/")[-1].split("_")[2]
            emo_index = torch.tensor(emo_list.index(emotype))
            emo_level = torch.tensor(
                int(metadata["video_name"].split("/")[-1].split("_")[4]) - 1
            )

            motion_data = self.motion_data[metadata["audio_name"]]
            audio_clip, sr = torchaudio.load(metadata["audio_name"])
            audio_clip = audio_clip.squeeze()
            assert sr == 16000, f"Invalid sampling rate: {sr}"

            min_frames = min(
                int(audio_clip.shape[0] / self.audio_unit),
                motion_data["n_frames"],
            )
            motion_data = self.check_motion_length(motion_data, min_frames)
            audio_clip = audio_clip[:int(min_frames * self.audio_unit)]
            if "repeat" in motion_data:
                for _ in range(motion_data["repeat"]):
                    audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

            seq_len = motion_data["n_frames"]
            if self.crop_strategy == "random":
                if seq_len - self.coef_total_len - 2 <= 0:
                    index = np.random.randint(0, len(self.all_data))
                    continue
                start_frame = np.random.randint(
                    0, seq_len - self.coef_total_len - 2
                )
            elif self.crop_strategy == "begin":
                start_frame = 0
            elif self.crop_strategy == "end":
                start_frame = seq_len - self.coef_total_len - 2
            else:
                raise ValueError(f"Unknown crop strategy: {self.crop_strategy}")

            end_frame = start_frame + self.coef_total_len
            coef_dict = {"exp": [], "pose": []}
            for frame_idx in range(start_frame, end_frame):
                exp, pose = self._normalize_motion_frame(
                    motion_data["motion"][frame_idx]
                )
                coef_dict["exp"].append([exp])
                coef_dict["pose"].append([pose])
            coef_dict = {
                key: torch.tensor(
                    np.concatenate(value, axis=0), dtype=torch.float32
                )
                for key, value in coef_dict.items()
            }

            audio = audio_clip[
                round(start_frame * self.audio_unit):
                round(end_frame * self.audio_unit)
            ]
            if audio.shape[0] != self.audio_total_len:
                index = np.random.randint(0, len(self.all_data))
                continue

            audio_pair = [
                audio[:self.n_audio_samples].clone(),
                audio[-self.n_audio_samples:].clone(),
            ]
            coef_pair = [
                {key: coef_dict[key][:self.n_motions].clone() for key in ("exp", "pose")},
                {key: coef_dict[key][-self.n_motions:].clone() for key in ("exp", "pose")},
            ]
            canonical_kp_pair = [
                self._build_canonical_token(motion_data["motion"][start_frame]),
                self._build_canonical_token(
                    motion_data["motion"][start_frame + self.n_motions]
                ),
            ]
            return audio_pair, coef_pair, canonical_kp_pair, emo_index, emo_level


__all__ = ["EmoLevelDataset"]

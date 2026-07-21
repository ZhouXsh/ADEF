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
    def __init__(self, root_dir="src/my_prepare/",
                 motion_filename="front_all_motions.pkl",
                 motion_template_filename="motion_template.pkl",
                 split="train", coef_fps=25, n_motions=100,
                 crop_strategy="random", normalize_type="mix"):
        self.template_dir = os.path.join(root_dir, motion_template_filename)
        self.template_dict = pickle.load(open(self.template_dir, "rb"))
        self.motion_dir = os.path.join(root_dir, motion_filename)
        self.eps = 1e-9
        self.normalize_type = normalize_type

        split_file = "train.txt" if split == "train" else "test.txt"
        self.root_dir = os.path.join(root_dir, split_file)
        with open(self.root_dir, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
        self.all_data = [
            {
                "video_name": line,
                "audio_name": line[:-4] + ".wav",
                "motion_name": line[:-4] + ".pkl",
            }
            for line in lines
        ]
        self.motion_data = pickle.load(open(self.motion_dir, "rb"))
        print("load all motion data done...")

        self.coef_fps = coef_fps
        self.audio_unit = 16000.0 / self.coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
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

        motion_new = {"motion": []}
        for i in range(len(values["exp"])):
            motion_new["motion"].append(
                {key: values[key][i] for key in keys}
            )
        motion_new["n_frames"] = len(values["exp"])
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

    def _build_reference_tokens(self, frame):
        exp, pose = self._normalize_motion_frame(frame)
        first_motion = torch.as_tensor(
            np.concatenate([exp, pose]), dtype=torch.float32
        ).unsqueeze(0)

        canonical_kp = torch.as_tensor(
            frame["kp"].reshape(-1), dtype=torch.float32
        )
        canonical_token = torch.cat(
            [torch.zeros(7, dtype=canonical_kp.dtype), canonical_kp], dim=0
        ).unsqueeze(0)
        return canonical_token, first_motion

    def __getitem__(self, index):
        has_valid_audio = False
        while not has_valid_audio:
            metadata = self.all_data[index]
            emotype = metadata["video_name"].split("/")[-1].split("_")[2]
            emo_index = torch.tensor(emo_list.index(emotype))
            emolevel = int(
                metadata["video_name"].split("/")[-1].split("_")[4]
            ) - 1
            emo_level = torch.tensor(emolevel)

            motion_data = self.motion_data[metadata["audio_name"]]
            audio_clip, sr = torchaudio.load(metadata["audio_name"])
            audio_clip = audio_clip.squeeze()
            assert sr == 16000, f"Invalid sampling rate: {sr}"

            audio_frames = int(audio_clip.shape[0] / self.audio_unit)
            motion_frames = motion_data["n_frames"]
            min_frames = min(audio_frames, motion_frames)
            motion_data = self.check_motion_length(motion_data, min_frames)

            audio_clip = audio_clip[:int(min_frames * self.audio_unit)]
            if "repeat" in motion_data:
                for _ in range(motion_data["repeat"]):
                    audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

            seq_len = motion_data["n_frames"]
            assert int(seq_len * self.audio_unit) == audio_clip.shape[0], (
                f"Frame mismatch: {seq_len * self.audio_unit} != "
                f"{audio_clip.shape[0]}"
            )

            if self.crop_strategy == "random":
                end = seq_len - self.coef_total_len
                if end < 0:
                    print(
                        f"current data invalid: "
                        f"{os.path.basename(metadata['audio_name'])}, "
                        f"n_frames: {seq_len}"
                    )
                    continue
                start_frame = np.random.randint(
                    0, seq_len - self.coef_total_len - 2
                )
            elif self.crop_strategy == "begin":
                start_frame = 0
            elif self.crop_strategy == "end":
                start_frame = seq_len - self.coef_total_len - 2
            else:
                raise ValueError(
                    f"Unknown crop strategy: {self.crop_strategy}"
                )
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
                    np.concatenate(coef_dict[key], axis=0),
                    dtype=torch.float32,
                )
                for key in coef_dict
            }
            assert coef_dict["exp"].shape[0] == self.coef_total_len

            audio = audio_clip[
                round(start_frame * self.audio_unit):
                round(end_frame * self.audio_unit)
            ]
            if audio.shape[0] != self.audio_total_len:
                print(
                    f"audio length invalid! audio: {audio.shape[0]}, "
                    f"coef: {self.audio_total_len}"
                )
                continue

            keys = ["exp", "pose"]
            audio_pair = [
                audio[:self.n_audio_samples].clone(),
                audio[-self.n_audio_samples:].clone(),
            ]
            coef_pair = [
                {
                    key: coef_dict[key][:self.n_motions].clone()
                    for key in keys
                },
                {
                    key: coef_dict[key][-self.n_motions:].clone()
                    for key in keys
                },
            ]

            first_indices = [start_frame, start_frame + self.n_motions]
            canonical_kp_pair = []
            first_motion_pair = []
            for frame_idx in first_indices:
                canonical_token, first_motion = self._build_reference_tokens(
                    motion_data["motion"][frame_idx]
                )
                canonical_kp_pair.append(canonical_token)
                first_motion_pair.append(first_motion)

            has_valid_audio = True
            return (
                audio_pair,
                coef_pair,
                canonical_kp_pair,
                first_motion_pair,
                emo_index,
                emo_level,
            )

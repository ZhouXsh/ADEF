import os

import numpy as np
import torch
import torchaudio

from .dataset_EmotionLevel import EmoLevelDataset as BaseEmoLevelDataset, emo_list


class EmoLevelDataset(BaseEmoLevelDataset):
    """Return one 25-frame previous context plus one 100-frame target window."""

    def __init__(
        self,
        root_dir='src/my_prepare/',
        motion_filename='front_all_motions.pkl',
        motion_template_filename='motion_template.pkl',
        split='train',
        coef_fps=25,
        n_motions=100,
        n_prev_motions=25,
        crop_strategy='random',
        normalize_type='mix',
    ):
        super().__init__(
            root_dir=root_dir,
            motion_filename=motion_filename,
            motion_template_filename=motion_template_filename,
            split=split,
            coef_fps=coef_fps,
            n_motions=n_motions,
            crop_strategy=crop_strategy,
            normalize_type=normalize_type,
        )
        self.n_prev_motions = n_prev_motions
        self.coef_total_len = self.n_prev_motions + self.n_motions
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)

    def __getitem__(self, index):
        while True:
            metadata = self.all_data[index]
            filename_parts = metadata['video_name'].split('/')[-1].split('_')
            emo_index = torch.tensor(emo_list.index(filename_parts[2]))
            emo_level = torch.tensor(int(filename_parts[4]) - 1)

            motion_data = self.motion_data[metadata['audio_name']]
            audio_clip, sample_rate = torchaudio.load(metadata['audio_name'])
            if audio_clip.ndim == 2:
                audio_clip = audio_clip.mean(dim=0)
            else:
                audio_clip = audio_clip.squeeze()
            assert sample_rate == 16000, f'Invalid sampling rate: {sample_rate}'

            audio_frames = int(audio_clip.shape[0] / self.audio_unit)
            min_frames = min(audio_frames, motion_data['n_frames'])
            if min_frames <= 0:
                index = np.random.randint(0, len(self.all_data))
                continue
            motion_data = self.check_motion_length(motion_data, min_frames)

            audio_clip = audio_clip[:round(min_frames * self.audio_unit)]
            if 'repeat' in motion_data:
                for _ in range(motion_data['repeat']):
                    audio_clip = torch.cat([audio_clip, audio_clip], dim=0)

            seq_len = motion_data['n_frames']
            assert round(seq_len * self.audio_unit) == audio_clip.shape[0], (
                f'Frame mismatch: {seq_len * self.audio_unit} != {audio_clip.shape[0]}'
            )

            max_start = seq_len - self.coef_total_len
            if max_start < 0:
                index = np.random.randint(0, len(self.all_data))
                continue
            if self.crop_strategy == 'random':
                start_frame = np.random.randint(0, max_start + 1)
            elif self.crop_strategy == 'begin':
                start_frame = 0
            elif self.crop_strategy == 'end':
                start_frame = max_start
            else:
                raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
            end_frame = start_frame + self.coef_total_len

            coef_dict = {'exp': [], 'pose': []}
            for frame_idx in range(start_frame, end_frame):
                frame = motion_data['motion'][frame_idx]
                if self.normalize_type != 'mix':
                    raise RuntimeError(f'Unsupported normalize_type: {self.normalize_type}')

                normalized_exp = (
                    frame['exp'].flatten() - self.template_dict['mean_exp']
                ) / (self.template_dict['std_exp'] + self.eps)
                pose_data = np.concatenate(
                    (
                        (frame['scale'].flatten() - self.template_dict['min_scale'])
                        / (
                            self.template_dict['max_scale']
                            - self.template_dict['min_scale']
                            + self.eps
                        ),
                        (frame['t'].flatten() - self.template_dict['min_t'])
                        / (self.template_dict['max_t'] - self.template_dict['min_t'] + self.eps),
                        (frame['pitch'].flatten() - self.template_dict['min_pitch'])
                        / (
                            self.template_dict['max_pitch']
                            - self.template_dict['min_pitch']
                            + self.eps
                        ),
                        (frame['yaw'].flatten() - self.template_dict['min_yaw'])
                        / (
                            self.template_dict['max_yaw']
                            - self.template_dict['min_yaw']
                            + self.eps
                        ),
                        (frame['roll'].flatten() - self.template_dict['min_roll'])
                        / (
                            self.template_dict['max_roll']
                            - self.template_dict['min_roll']
                            + self.eps
                        ),
                    )
                )
                coef_dict['exp'].append(normalized_exp)
                coef_dict['pose'].append(pose_data)

            coef_dict = {
                key: torch.as_tensor(np.stack(value, axis=0), dtype=torch.float32)
                for key, value in coef_dict.items()
            }
            audio = audio_clip[
                round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)
            ].clone()
            if audio.shape[0] != self.audio_total_len:
                index = np.random.randint(0, len(self.all_data))
                continue

            return audio, coef_dict, emo_index, emo_level

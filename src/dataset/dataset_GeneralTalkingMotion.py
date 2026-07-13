import os
import pickle
import warnings

import numpy as np
import torch
import torchaudio
from torch.utils import data

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class GeneralTalkingMotionDataset(data.Dataset):
    """Generic audio-motion dataset for Stage-1 training.

    The organization and returned values follow ``EmoLevelDataset``. Motion
    pickle files are dictionaries whose keys are audio paths and whose values
    contain ``motion`` and ``n_frames``. Multiple pickle files can be supplied
    so Stage 1 can combine several talking-video datasets.
    """

    def __init__(self,
                 root_dir='src/my_prepare/',
                 motion_filenames='front_all_motions.pkl',
                 motion_template_filename='motion_template.pkl',
                 split='train',
                 split_filename=None,
                 coef_fps=25,
                 n_motions=100,
                 crop_strategy='random',
                 normalize_type='mix'):
        self.template_dir = os.path.join(root_dir, motion_template_filename)
        self.template_dict = pickle.load(open(self.template_dir, 'rb'))
        self.eps = 1e-9
        self.normalize_type = normalize_type

        if isinstance(motion_filenames, str):
            motion_filenames = [name.strip() for name in motion_filenames.split(',') if name.strip()]
        if not motion_filenames:
            raise ValueError('motion_filenames must contain at least one pickle file')

        self.motion_data = {}
        for motion_filename in motion_filenames:
            motion_path = motion_filename
            if not os.path.isabs(motion_path):
                motion_path = os.path.join(root_dir, motion_path)
            current_motion = pickle.load(open(motion_path, 'rb'))
            duplicated = set(self.motion_data).intersection(current_motion)
            if duplicated:
                raise ValueError(f'duplicated motion keys in {motion_path}: {len(duplicated)}')
            self.motion_data.update(current_motion)
        print(f'load general motion data done: {len(self.motion_data)} clips')

        if split_filename is None:
            split_filename = 'train.txt' if split == 'train' else 'test.txt'
        split_path = split_filename
        if not os.path.isabs(split_path):
            split_path = os.path.join(root_dir, split_path)

        if os.path.isfile(split_path):
            with open(split_path, 'r', encoding='utf-8') as file:
                audio_names = [line.strip() for line in file if line.strip()]
            audio_names = [name[:-4] + '.wav' if not name.endswith('.wav') else name for name in audio_names]
            self.all_data = [name for name in audio_names if name in self.motion_data]
        else:
            all_names = sorted(self.motion_data.keys())
            self.all_data = [name for index, name in enumerate(all_names)
                             if (index % 20 != 0) == (split == 'train')]

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy

    def __len__(self):
        return len(self.all_data)

    def check_motion_length(self, motion_data, min_frames):
        motion = motion_data['motion'][:min_frames]
        if not motion:
            raise ValueError('empty motion sequence')
        repeat = 0
        while len(motion) < self.coef_total_len + 4:
            motion = motion + motion
            repeat += 1
        return {'motion': motion, 'n_frames': len(motion), 'repeat': repeat}

    def __getitem__(self, index):
        while True:
            audio_name = self.all_data[index]
            motion_data = self.motion_data[audio_name]

            audio_clip, sample_rate = torchaudio.load(audio_name)
            audio_clip = audio_clip.mean(dim=0)
            if sample_rate != 16000:
                audio_clip = torchaudio.functional.resample(audio_clip, sample_rate, 16000)

            audio_frames = int(audio_clip.shape[0] / self.audio_unit)
            motion_frames = motion_data['n_frames']
            min_frames = min(audio_frames, motion_frames)
            motion_data = self.check_motion_length(motion_data, min_frames)

            audio_clip = audio_clip[:round(min_frames * self.audio_unit)]
            for _ in range(motion_data['repeat']):
                audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

            seq_len = min(motion_data['n_frames'], int(audio_clip.shape[0] / self.audio_unit))
            if seq_len < self.coef_total_len + 2:
                index = (index + 1) % len(self.all_data)
                continue

            if self.crop_strategy == 'random':
                start_frame = np.random.randint(0, seq_len - self.coef_total_len - 1)
            elif self.crop_strategy == 'begin':
                start_frame = 0
            elif self.crop_strategy == 'end':
                start_frame = seq_len - self.coef_total_len - 2
            else:
                raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
            end_frame = start_frame + self.coef_total_len

            coef_dict = {'exp': [], 'pose': []}
            template = self.template_dict
            for frame_idx in range(start_frame, end_frame):
                frame = motion_data['motion'][frame_idx]
                if self.normalize_type != 'mix':
                    raise RuntimeError('GeneralTalkingMotionDataset only supports normalize_type=mix')
                normalized_exp = (frame['exp'].flatten() - template['mean_exp']) / (
                    template['std_exp'] + self.eps)
                pose_data = np.concatenate((
                    (frame['scale'].flatten() - template['min_scale']) / (template['max_scale'] - template['min_scale'] + self.eps),
                    (frame['t'].flatten() - template['min_t']) / (template['max_t'] - template['min_t'] + self.eps),
                    (frame['pitch'].flatten() - template['min_pitch']) / (template['max_pitch'] - template['min_pitch'] + self.eps),
                    (frame['yaw'].flatten() - template['min_yaw']) / (template['max_yaw'] - template['min_yaw'] + self.eps),
                    (frame['roll'].flatten() - template['min_roll']) / (template['max_roll'] - template['min_roll'] + self.eps),
                ))
                coef_dict['exp'].append(normalized_exp[None])
                coef_dict['pose'].append(pose_data[None])

            coef_dict = {key: torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32)
                         for key, value in coef_dict.items()}
            audio = audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)]
            if audio.shape[0] != self.audio_total_len:
                index = (index + 1) % len(self.all_data)
                continue

            keys = ['exp', 'pose']
            audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
            coef_pair = [
                {key: coef_dict[key][:self.n_motions].clone() for key in keys},
                {key: coef_dict[key][-self.n_motions:].clone() for key in keys},
            ]
            return audio_pair, coef_pair, torch.tensor(0), torch.tensor(0)

"""MEAD motion dataset copy with emotion2vec feature loading.

This file is a rewritten copy of ``dataset_EmotionLevel.py``.  The original
file is intentionally left untouched.  It returns the original audio/motion
pairs plus:

    e2v_utt_pair:   [Tensor(D_e2v), Tensor(D_e2v)]
    e2v_frame_pair: [Tensor(n_motions, D_e2v), Tensor(n_motions, D_e2v)]

Expected MEAD layout, matching the user's current directory:

videos/<speaker>/front/<emotion>/<level>/
    <name>.wav
    frame/<name>.npy
    utterance/<name>.npy

It also supports the legacy fallback where ``<name>.npy`` sits next to the wav.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils import data

from src.config.emotion_config import global_emo_list

emo_list = global_emo_list
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class EmoLevelE2VDataset(data.Dataset):
    def __init__(
        self,
        root_dir='src/my_prepare/',
        motion_filename='front_all_motions.pkl',
        motion_template_filename='motion_template.pkl',
        split='train',
        coef_fps=25,
        n_motions=100,
        crop_strategy='random',
        normalize_type='mix',
        emotion2vec_root: Optional[str] = None,
        emotion2vec_dim: int = 1024,
    ):
        self.template_dir = os.path.join(root_dir, motion_template_filename)
        self.template_dict = pickle.load(open(self.template_dir, 'rb'))
        self.motion_dir = os.path.join(root_dir, motion_filename)
        self.eps = 1e-9
        self.normalize_type = normalize_type
        self.emotion2vec_root = emotion2vec_root
        self.emotion2vec_dim = emotion2vec_dim

        if split == 'train':
            split_path = os.path.join(root_dir, 'train.txt')
        else:
            split_path = os.path.join(root_dir, 'test.txt')

        with open(split_path, 'r', encoding='utf-8') as file:
            lines = [line.strip() for line in file.readlines()]
            self.all_data = [
                {
                    'video_name': line,
                    'audio_name': line[:-4] + '.wav',
                    'motion_name': line[:-4] + '.pkl',
                }
                for line in lines
            ]

        self.motion_data = pickle.load(open(self.motion_dir, 'rb'))
        print('load all motion data done...')

        self.coef_fps = coef_fps
        self.audio_unit = 16000. / self.coef_fps
        self.n_motions = n_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.coef_total_len = self.n_motions * 2
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)
        self.crop_strategy = crop_strategy

    def __len__(self):
        return len(self.all_data)

    def _resolve_e2v_paths(self, audio_name: str) -> Tuple[Path, Path]:
        audio_path = Path(audio_name)
        npy_name = audio_path.with_suffix('.npy').name

        candidates = []
        if self.emotion2vec_root is not None:
            root = Path(self.emotion2vec_root)
            # Preserve the MEAD suffix if possible.
            parts = audio_path.parts
            if 'videos' in parts:
                rel = Path(*parts[parts.index('videos') + 1:]).with_suffix('.npy')
                base = root / rel.parent
                candidates.append((base / 'utterance' / npy_name, base / 'frame' / npy_name))
            candidates.append((root / 'utterance' / npy_name, root / 'frame' / npy_name))

        # User-provided current layout: level_X/utterance/name.npy and level_X/frame/name.npy.
        level_dir = audio_path.parent
        candidates.append((level_dir / 'utterance' / npy_name, level_dir / 'frame' / npy_name))
        # Legacy fallback: a single utterance feature beside the wav.
        candidates.append((audio_path.with_suffix('.npy'), audio_path.with_suffix('.npy')))

        for utt_path, frame_path in candidates:
            if utt_path.exists() and frame_path.exists():
                return utt_path, frame_path

        raise FileNotFoundError(
            f'Cannot find emotion2vec features for {audio_name}. Tried: '
            + '; '.join([f'utt={u}, frame={f}' for u, f in candidates])
        )

    def _load_e2v(self, audio_name: str):
        utt_path, frame_path = self._resolve_e2v_paths(audio_name)
        utt = np.load(utt_path)
        frame = np.load(frame_path)
        utt = np.asarray(utt, dtype=np.float32).reshape(-1)
        frame = np.asarray(frame, dtype=np.float32)
        if frame.ndim == 1:
            frame = frame.reshape(1, -1)
        if utt.shape[-1] != frame.shape[-1]:
            raise ValueError(f'emotion2vec dim mismatch: {utt_path} {utt.shape}, {frame_path} {frame.shape}')
        return torch.from_numpy(utt), torch.from_numpy(frame)

    @staticmethod
    def _resample_frame_feature(frame_feat: torch.Tensor, target_len: int) -> torch.Tensor:
        if frame_feat.shape[0] == target_len:
            return frame_feat
        feat = frame_feat.transpose(0, 1).unsqueeze(0)  # [1, D, T]
        feat = F.interpolate(feat, size=target_len, mode='linear', align_corners=False)
        return feat.squeeze(0).transpose(0, 1).contiguous()

    def check_motion_length(self, motion_data, min_frames):
        exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
        for frame_index in range(min_frames):
            exp_list.append(motion_data['motion'][frame_index]['exp'])
            t_list.append(motion_data['motion'][frame_index]['t'])
            scale_list.append(motion_data['motion'][frame_index]['scale'])
            pitch_list.append(motion_data['motion'][frame_index]['pitch'])
            yaw_list.append(motion_data['motion'][frame_index]['yaw'])
            roll_list.append(motion_data['motion'][frame_index]['roll'])

        if min_frames > self.coef_total_len + 4:
            motion_new = {'motion': []}
            for i in range(len(exp_list)):
                motion_new['motion'].append({
                    'exp': exp_list[i], 't': t_list[i], 'scale': scale_list[i],
                    'pitch': pitch_list[i], 'yaw': yaw_list[i], 'roll': roll_list[i],
                })
            motion_new['n_frames'] = len(exp_list)
            return motion_new

        repeat = 0
        while len(exp_list) < self.coef_total_len + 4:
            exp_list = exp_list * 2
            t_list = t_list * 2
            scale_list = scale_list * 2
            pitch_list = pitch_list * 2
            yaw_list = yaw_list * 2
            roll_list = roll_list * 2
            repeat += 1

        motion_new = {'motion': []}
        for i in range(len(exp_list)):
            motion_new['motion'].append({
                'exp': exp_list[i], 't': t_list[i], 'scale': scale_list[i],
                'pitch': pitch_list[i], 'yaw': yaw_list[i], 'roll': roll_list[i],
            })
        motion_new['n_frames'] = len(exp_list)
        motion_new['repeat'] = repeat
        return motion_new

    def __getitem__(self, index):
        has_valid_audio = False
        while not has_valid_audio:
            metadata = self.all_data[index]
            emotype = metadata['video_name'].split('/')[-1].split('_')[2]
            emo_index = torch.tensor(emo_list.index(emotype), dtype=torch.long)
            emolevel = int(metadata['video_name'].split('/')[-1].split('_')[4]) - 1
            emo_level = torch.tensor(emolevel, dtype=torch.long)

            motion_data = self.motion_data[metadata['audio_name']]
            audio_path = metadata['audio_name']
            audio_clip, sr = torchaudio.load(audio_path)
            audio_clip = audio_clip.squeeze()
            assert sr == 16000, f'Invalid sampling rate: {sr}'

            e2v_utt, e2v_frame = self._load_e2v(audio_path)

            audio_frames = int(audio_clip.shape[0] / self.audio_unit)
            motion_frames = motion_data['n_frames']
            min_frames = min(audio_frames, motion_frames)

            motion_data = self.check_motion_length(motion_data, min_frames)
            audio_clip = audio_clip[:int(min_frames * self.audio_unit)]
            if 'repeat' in motion_data:
                for _ in range(motion_data['repeat']):
                    audio_clip = torch.cat((audio_clip, audio_clip), dim=0)

            seq_len = motion_data['n_frames']
            assert int(seq_len * self.audio_unit) == audio_clip.shape[0], (
                f'帧数不匹配: {seq_len * self.audio_unit} != {audio_clip.shape[0]}'
            )

            # Align emotion2vec frame features to the final motion/audio frame length.
            e2v_frame = self._resample_frame_feature(e2v_frame, seq_len)

            if self.crop_strategy == 'random':
                end = seq_len - self.coef_total_len
                if end < 0:
                    print(f"current data invalid: {os.path.basename(metadata['audio_name'])}, n_frames: {seq_len}")
                    has_valid_audio = False
                    continue
                start_frame = np.random.randint(0, seq_len - self.coef_total_len - 2)
            elif self.crop_strategy == 'begin':
                start_frame = 0
            elif self.crop_strategy == 'end':
                start_frame = seq_len - self.coef_total_len - 2
            else:
                raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
            end_frame = start_frame + self.coef_total_len

            template_dict = self.template_dict
            coef_keys = ['exp', 'pose']
            coef_dict = {k: [] for k in coef_keys}
            for frame_idx in range(start_frame, end_frame):
                for coef_key in coef_keys:
                    if coef_key == 'exp':
                        if self.normalize_type == 'mix':
                            normalized_exp = (
                                motion_data['motion'][frame_idx]['exp'].flatten() - template_dict['mean_exp']
                            ) / (template_dict['std_exp'] + self.eps)
                        else:
                            raise RuntimeError('error')
                        coef_dict[coef_key].append([normalized_exp])
                    elif coef_key == 'pose':
                        if self.normalize_type == 'mix':
                            pose_data = np.concatenate((
                                (motion_data['motion'][frame_idx]['scale'].flatten() - template_dict['mean_scale']) /
                                (template_dict['std_scale'] + self.eps),
                                (motion_data['motion'][frame_idx]['t'].flatten() - template_dict['mean_t']) /
                                (template_dict['std_t'] + self.eps),
                                (motion_data['motion'][frame_idx]['pitch'].flatten() - template_dict['mean_pitch']) /
                                (template_dict['std_pitch'] + self.eps),
                                (motion_data['motion'][frame_idx]['yaw'].flatten() - template_dict['mean_yaw']) /
                                (template_dict['std_yaw'] + self.eps),
                                (motion_data['motion'][frame_idx]['roll'].flatten() - template_dict['mean_roll']) /
                                (template_dict['std_roll'] + self.eps),
                            ))
                        else:
                            raise RuntimeError('pose data error')
                        coef_dict[coef_key].append([pose_data])
                    else:
                        raise RuntimeError('coef_key error: ', coef_key)
            coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0), dtype=torch.float32) for k in coef_keys}
            assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'

            audio = audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)]
            if not audio.shape[0] == self.audio_total_len:
                print(f'audio length invalid! audio: {audio.shape[0]}, coef: {self.audio_total_len}')
                has_valid_audio = False
                continue

            e2v_frame = e2v_frame[start_frame:end_frame]
            if e2v_frame.shape[0] != self.coef_total_len:
                raise ValueError(f'Invalid emotion2vec frame length: {e2v_frame.shape[0]}')

            keys = ['exp', 'pose']
            audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
            coef_pair = [
                {k: coef_dict[k][:self.n_motions].clone() for k in keys},
                {k: coef_dict[k][-self.n_motions:].clone() for k in keys},
            ]
            e2v_utt_pair = [e2v_utt.clone(), e2v_utt.clone()]
            e2v_frame_pair = [e2v_frame[:self.n_motions].clone(), e2v_frame[-self.n_motions:].clone()]
            has_valid_audio = True
            return audio_pair, coef_pair, emo_index, emo_level, e2v_utt_pair, e2v_frame_pair

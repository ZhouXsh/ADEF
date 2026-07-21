# coding: utf-8

import math
import os
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track

from src.config.emotion_config import global_emo_list
from . import ADEF_wrapper as base_wrapper_module
from .ADEF_wrapper import ADEFWrapper as BaseADEFWrapper
from .utils.camera import get_rotation_matrix
from .utils.filter import smooth_
from .utils.helper_framelevel_0721 import load_model as framelevel_load_model
from .utils.rprint import rlog as log

emo_list = global_emo_list


class ADEFWrapperFrameLevel0721(BaseADEFWrapper):
    """ADEFWrapper 副本扩展：只改 motion generator 加载与运动生成输入。"""

    def __init__(self, inference_cfg):
        # 原包装器在模块级引用 load_model。初始化期间临时替换为新模型加载器，
        # 其余 F/M/W/G/S 初始化代码完全沿用原实现。
        original_loader = base_wrapper_module.load_model
        base_wrapper_module.load_model = framelevel_load_model
        try:
            super().__init__(inference_cfg)
        finally:
            base_wrapper_module.load_model = original_loader
        self.emotion2vec_dim = (
            getattr(self.motion_generator_args, 'emotion2vec_dim', None) or 1024
        )

    @staticmethod
    def _resample(feat, length):
        if feat.shape[0] == length:
            return feat
        feat = feat.T.unsqueeze(0)
        feat = F.interpolate(feat, size=length, mode='linear', align_corners=False)
        return feat.squeeze(0).T.contiguous()

    def _find_frame_path(self, args):
        explicit = getattr(args, 'emotion2vec_frame_path', None)
        if explicit:
            path = Path(explicit)
            if not path.exists():
                raise FileNotFoundError(path)
            return path
        audio = Path(args.audio)
        for path in (
            audio.parent / 'frame' / f'{audio.stem}.npy',
            audio.with_suffix('.frame.npy'),
            audio.with_suffix('.npy'),
        ):
            if path.exists():
                return path
        return None

    def _extract_frame_feature(self, args):
        from funasr import AutoModel

        output_dir = getattr(args, 'emotion2vec_output_dir', None)
        output_dir = output_dir or str(Path(args.audio).parent)
        os.makedirs(output_dir, exist_ok=True)
        model = AutoModel(
            model=getattr(args, 'emotion2vec_model', 'iic/emotion2vec_plus_large'),
            hub=getattr(args, 'emotion2vec_hub', 'ms'),
        )
        result = model.generate(
            args.audio,
            output_dir=output_dir,
            granularity='frame',
            extract_embedding=True,
        )
        if isinstance(result, list) and result and isinstance(result[0], dict):
            for key in ('feats', 'feature', 'embedding', 'embeddings'):
                value = result[0].get(key)
                if value is not None:
                    value = np.asarray(value, dtype=np.float32).squeeze()
                    if value.ndim == 2:
                        return torch.from_numpy(value)
        audio = Path(args.audio)
        for path in (
            Path(output_dir) / 'frame' / f'{audio.stem}.npy',
            Path(output_dir) / f'{audio.stem}.npy',
            audio.parent / 'frame' / f'{audio.stem}.npy',
        ):
            if path.exists():
                return torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))
        raise FileNotFoundError('emotion2vec frame-level feature was not produced')

    def _load_frame_feature(self, args, raw_samples):
        path = self._find_frame_path(args)
        feat = (
            torch.from_numpy(np.asarray(np.load(path), dtype=np.float32))
            if path is not None else self._extract_frame_feature(args)
        ).float().squeeze()
        if feat.ndim == 1:
            feat = feat[None]
        if feat.ndim != 2 or feat.shape[-1] != self.emotion2vec_dim:
            raise ValueError(
                f'frame feature must be [T,{self.emotion2vec_dim}], got {tuple(feat.shape)}'
            )
        raw_frames = max(1, int(raw_samples / 16000.0 * self.fps))
        feat = self._resample(feat, raw_frames)
        # 与原音频左 1280、右 640 点填充对应：25fps 下为左2帧、右1帧。
        return F.pad(feat, (0, 0, 2, 1), value=0.0)

    def gen_motion_sequence(self, args):
        audio_np, _ = librosa.load(args.audio, sr=16000, mono=True)
        frame_feat = self._load_frame_feature(args, len(audio_np))
        audio = torch.from_numpy(audio_np).to(self.device)
        audio = F.pad(audio, (1280, 640), 'constant', 0)

        clip_len = int(len(audio) / 16000.0 * self.fps)
        n_subdivision = max(1, math.ceil(clip_len / self.n_motions))
        total_frames = n_subdivision * self.n_motions
        if frame_feat.shape[0] < total_frames:
            frame_feat = F.pad(
                frame_feat, (0, 0, 0, total_frames - frame_feat.shape[0]), value=0.0
            )
        frame_feat = frame_feat[:total_frames].to(self.device)

        pad_samples = self.n_audio_samples * n_subdivision - len(audio)
        pad_frames = math.ceil(pad_samples / self.audio_unit)
        if pad_samples > 0:
            value = 0 if self.pad_mode == 'zero' else audio[-1]
            audio = F.pad(audio, (0, pad_samples), value=value)

        coef_list = []
        prev_motion = prev_audio = prev_frame = noise = None
        emo_index = torch.tensor(
            [emo_list.index(args.emotype)], dtype=torch.long, device=self.device
        )

        for i in range(n_subdivision):
            start = i * self.n_motions
            end = start + self.n_motions
            indicator = (
                torch.ones(1, self.n_motions, device=self.device)
                if self.use_indicator else None
            )
            if indicator is not None and i == n_subdivision - 1 and pad_frames > 0:
                indicator[:, -pad_frames:] = 0
            audio_in = audio[
                round(start * self.audio_unit):round(end * self.audio_unit)
            ][None]
            frame_in = frame_feat[start:end][None]

            motion, noise, returned_audio = self.motion_generator.sample(
                audio_in,
                prev_motion_feat=prev_motion,
                prev_audio_feat=prev_audio,
                motion_at_T=noise,
                indicator=indicator,
                cfg_mode=args.cfg_mode,
                cfg_cond=args.cfg_cond,
                cfg_scale=args.cfg_scale,
                dynamic_threshold=0,
                emo_index=emo_index,
                frame_level_feat=frame_in,
                prev_frame_level_feat=prev_frame,
            )
            prev_motion = motion[:, -self.n_prev_motions:].clone()
            prev_audio = returned_audio[:, -self.n_prev_motions:]
            prev_frame = frame_in[:, -self.n_prev_motions:].clone()

            if self.emo_ehance:
                level = torch.tensor(
                    [args.enhance_level - 1], dtype=torch.long, device=self.device
                )
                delta = self.emo_enhancer(
                    motion[:, self.n_prev_motions:, :63], emo_index, level
                )
                motion[:, self.n_prev_motions:, :63] += delta.detach()

            if i == n_subdivision - 1 and pad_frames > 0:
                motion = motion[:, :-pad_frames]
            coef_list.append(motion)

        motion_coef = torch.cat(coef_list, dim=1).squeeze(0)
        template = self.template_dict
        motion_list = []
        for idx in track(range(len(motion_coef)), description='Generating Motion Sequence'):
            item = motion_coef[idx].cpu()
            exp = item[:63] * template['std_exp'] + template['mean_exp']
            scale = item[63:64] * (template['max_scale']-template['min_scale']) + template['min_scale']
            trans = item[64:67] * (template['max_t']-template['min_t']) + template['min_t']
            pitch = item[67:68] * (template['max_pitch']-template['min_pitch']) + template['min_pitch']
            yaw = item[68:69] * (template['max_yaw']-template['min_yaw']) + template['min_yaw']
            roll = item[69:70] * (template['max_roll']-template['min_roll']) + template['min_roll']
            R = get_rotation_matrix(pitch, yaw, roll)
            motion_list.append({
                'exp': exp.reshape(1,21,3).numpy().astype(np.float32),
                'scale': scale.reshape(1,1).numpy().astype(np.float32),
                'R': R.reshape(1,3,3).cpu().numpy().astype(np.float32),
                't': trans.reshape(1,3).numpy().astype(np.float32),
                'pitch': pitch.reshape(1,1).numpy().astype(np.float32),
                'yaw': yaw.reshape(1,1).numpy().astype(np.float32),
                'roll': roll.reshape(1,1).numpy().astype(np.float32),
            })
        result = {'n_frames': len(motion_coef), 'output_fps': self.fps, 'motion': motion_list}
        return smooth_(result, method='ema') if args.is_smooth_motion else result


ADEFWrapper = ADEFWrapperFrameLevel0721

# coding: utf-8
"""Emotion2vec-aware inference wrapper for ADEF motion generators."""

from __future__ import annotations

import math
import pickle

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import track

from .ADEF_wrapper import ADEFWrapper
from .config.emotion_config import global_emo_list
from .utils.camera import get_rotation_matrix
from .utils.e2v_motion_generator_loader import load_e2v_motion_generator
from .utils.emotion2vec_inference import Emotion2VecExtractor
from .utils.filter import smooth_
from .utils.rprint import rlog as log


class ADEFE2VWrapper(ADEFWrapper):
    """ADEF wrapper that supplies utterance/frame emotion2vec conditions."""

    def __init__(self, inference_cfg):
        super().__init__(inference_cfg)
        variant = getattr(inference_cfg, "motion_generator_variant", "auto")
        self.motion_generator, self.motion_generator_args, report = load_e2v_motion_generator(
            inference_cfg.checkpoint_MotionGenerator,
            self.device,
            variant=variant,
        )
        log(
            f"Load emotion2vec motion generator {report['variant']} "
            f"({report['loaded_keys']}/{report['model_keys']} compatible keys)."
        )
        if report["shape_mismatches"]:
            log(f"Skipped {len(report['shape_mismatches'])} shape-mismatched keys.")

        self.n_motions = self.motion_generator.n_motions
        self.n_prev_motions = self.motion_generator.n_prev_motions
        self.fps = self.motion_generator.fps
        self.audio_unit = 16000.0 / self.fps
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = getattr(self.motion_generator_args, "pad_mode", "zero") or "zero"
        self.use_indicator = bool(getattr(self.motion_generator_args, "use_indicator", False))
        self.template_dict = pickle.load(open(inference_cfg.motion_template_path, "rb"))

        expected_dim = int(getattr(self.motion_generator_args, "e2v_dim", 1024) or 1024)
        self.e2v_extractor = Emotion2VecExtractor(
            model_id=getattr(inference_cfg, "emotion2vec_model_id", "iic/emotion2vec_plus_large"),
            hub=getattr(inference_cfg, "emotion2vec_hub", "ms"),
            device=getattr(inference_cfg, "emotion2vec_device", None),
            expected_dim=expected_dim,
        )

    def _extract_e2v(self, args, raw_audio_samples, padded_clip_frames, total_window_frames):
        utt, frame, paths = self.e2v_extractor.extract(
            args.audio,
            utterance_path=getattr(args, "emotion2vec_utterance_path", None),
            frame_path=getattr(args, "emotion2vec_frame_path", None),
            cache_dir=getattr(args, "emotion2vec_cache_dir", None),
            force=bool(getattr(args, "emotion2vec_force_extract", False)),
        )
        log(f"emotion2vec utterance: {paths.utterance}")
        log(f"emotion2vec frame: {paths.frame}")
        utt_tensor = torch.from_numpy(utt).float().unsqueeze(0).to(self.device)
        frame_timeline = self.e2v_extractor.align_frame_timeline(
            frame,
            raw_audio_samples=raw_audio_samples,
            fps=self.fps,
            sample_rate=16000,
            left_padding_samples=1280,
            right_padding_samples=640,
            padded_clip_frames=padded_clip_frames,
            total_window_frames=total_window_frames,
            final_pad_mode=self.pad_mode,
        ).to(self.device)
        return utt_tensor, frame_timeline

    def gen_motion_sequence(self, args):
        log(f"start loading audio from {args.audio}")
        audio_np, _ = librosa.load(args.audio, sr=16000, mono=True)
        raw_audio_samples = len(audio_np)
        audio = torch.from_numpy(audio_np).float().to(self.device)
        audio = F.pad(audio, (1280, 640), "constant", 0)

        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        n_subdivision = 1 if clip_len <= stride else math.ceil(clip_len / stride)
        total_window_frames = n_subdivision * self.n_motions
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(max(0, n_padding_audio_samples) / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == "zero":
                audio = F.pad(audio, (0, n_padding_audio_samples), value=0)
            elif self.pad_mode == "replicate":
                audio = F.pad(audio, (0, n_padding_audio_samples), value=float(audio[-1]))
            else:
                raise ValueError(f"Unknown pad mode: {self.pad_mode}")

        emo_utt_feat, frame_timeline = self._extract_e2v(
            args,
            raw_audio_samples,
            clip_len,
            total_window_frames,
        )
        try:
            emo_index_value = global_emo_list.index(args.emotype)
        except ValueError as error:
            raise ValueError(
                f"Unknown emotion '{args.emotype}', choose from {global_emo_list}"
            ) from error
        emo_index = torch.tensor([emo_index_value], dtype=torch.long, device=self.device)

        coef_list = []
        prev_motion_feat = None
        prev_audio_feat = None
        noise = None
        prev_emo_frame_feat = None

        for i in range(n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = torch.ones((1, self.n_motions), device=self.device) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0

            audio_in = audio[
                round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)
            ].unsqueeze(0)
            emo_frame_feat = frame_timeline[:, start_idx:end_idx]

            kwargs = dict(
                indicator=indicator,
                cfg_mode=args.cfg_mode,
                cfg_cond=args.cfg_cond,
                cfg_scale=args.cfg_scale,
                dynamic_threshold=0,
                emo_index=emo_index,
                emo_utt_feat=emo_utt_feat,
                emo_frame_feat=emo_frame_feat,
                prev_emo_frame_feat=prev_emo_frame_feat,
                cfg_min=getattr(args, "cfg_min", None),
                cfg_schedule=getattr(args, "cfg_schedule", None),
            )
            if i == 0:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(
                    audio_in,
                    **kwargs,
                )
            else:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(
                    audio_in,
                    prev_motion_feat,
                    prev_audio_feat,
                    noise,
                    **kwargs,
                )

            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:].clone()
            prev_emo_frame_feat = emo_frame_feat[:, -self.n_prev_motions:].clone()

            motion_coef = motion_feat
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]
            coef_list.append(motion_coef)

        motion_coef = torch.cat(coef_list, dim=1).squeeze(0)
        motion_list = []
        template = self.template_dict
        for idx in track(
            range(motion_coef.shape[0]),
            description="Generating Motion Sequence...",
            total=motion_coef.shape[0],
        ):
            coef = motion_coef[idx].detach().cpu()
            exp = coef[:63] * template["std_exp"] + template["mean_exp"]
            scale = coef[63:64] * (template["max_scale"] - template["min_scale"]) + template["min_scale"]
            t = coef[64:67] * (template["max_t"] - template["min_t"]) + template["min_t"]
            pitch = coef[67:68] * (template["max_pitch"] - template["min_pitch"]) + template["min_pitch"]
            yaw = coef[68:69] * (template["max_yaw"] - template["min_yaw"]) + template["min_yaw"]
            roll = coef[69:70] * (template["max_roll"] - template["min_roll"]) + template["min_roll"]
            R = get_rotation_matrix(pitch, yaw, roll).reshape(1, 3, 3).numpy().astype(np.float32)
            motion_list.append({
                "exp": exp.reshape(1, 21, 3).numpy().astype(np.float32),
                "scale": scale.reshape(1, 1).numpy().astype(np.float32),
                "R": R,
                "t": t.reshape(1, 3).numpy().astype(np.float32),
                "pitch": pitch.reshape(1, 1).numpy().astype(np.float32),
                "yaw": yaw.reshape(1, 1).numpy().astype(np.float32),
                "roll": roll.reshape(1, 1).numpy().astype(np.float32),
            })

        result = {
            "n_frames": motion_coef.shape[0],
            "output_fps": self.fps,
            "motion": motion_list,
        }
        if args.is_smooth_motion:
            result = smooth_(result, method="ema")
        return result

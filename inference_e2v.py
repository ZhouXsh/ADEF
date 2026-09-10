# coding: utf-8
"""Run ADEF inference with an emotion2vec-conditioned motion generator."""

import os
import os.path as osp
import platform
import subprocess

import tyro

from src.config.crop_config import CropConfig
from src.config.e2v_inference_config import (
    E2VArgumentConfig,
    E2VInferenceConfig,
)


if platform.system() == "Windows":
    import pathlib

    pathlib.PosixPath = pathlib.WindowsPath


def partial_fields(target_class, kwargs):
    return target_class(**{
        key: value
        for key, value in kwargs.items()
        if hasattr(target_class, key)
    })


def fast_check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True
        )
        return True
    except Exception:
        return False


def fast_check_args(args: E2VArgumentConfig):
    for label, path in (
        ("reference", args.reference),
        ("audio", args.audio),
        ("motion-generator checkpoint", args.checkpoint_MotionGenerator),
        ("motion template", args.motion_template_path),
    ):
        if not path or not osp.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")
    if bool(args.emotion2vec_utterance_path) != bool(
        args.emotion2vec_frame_path
    ):
        raise ValueError(
            "Provide both --emotion2vec-utterance-path and "
            "--emotion2vec-frame-path, or neither."
        )


def main():
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(E2VArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += os.pathsep + ffmpeg_dir
    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Install ffmpeg and ffprobe before "
            "running inference."
        )
    fast_check_args(args)

    inference_cfg = partial_fields(E2VInferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    from src.ADEF_pipeline_e2v import ADEFE2VPipeline

    pipeline = ADEFE2VPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg,
    )
    pipeline.execute(args)


if __name__ == "__main__":
    main()

# coding: utf-8

"""emotion2vec frame-level 条件模型的独立推理入口。"""

import os
import os.path as osp
import platform
import subprocess

import tyro

from src.config.argument_config_framelevel_0721 import (
    ArgumentConfigFrameLevel0721,
)
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig


if platform.system() == 'Windows':
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
            ['ffmpeg', '-version'], capture_output=True, check=True
        )
        return True
    except Exception:
        return False


def fast_check_args(args):
    if not osp.exists(args.reference):
        raise FileNotFoundError(
            f'reference info not found: {args.reference}'
        )
    if not osp.exists(args.audio):
        raise FileNotFoundError(f'audio info not found: {args.audio}')
    if args.emotion2vec_frame_path and not osp.exists(
        args.emotion2vec_frame_path
    ):
        raise FileNotFoundError(
            'emotion2vec frame feature not found: '
            f'{args.emotion2vec_frame_path}'
        )
    if not osp.exists(args.checkpoint_MotionGenerator):
        raise FileNotFoundError(
            'frame-level motion generator checkpoint not found: '
            f'{args.checkpoint_MotionGenerator}'
        )


def main():
    tyro.extras.set_accent_color('bright_cyan')
    args = tyro.cli(ArgumentConfigFrameLevel0721)

    ffmpeg_dir = os.path.join(os.getcwd(), 'ffmpeg')
    if osp.exists(ffmpeg_dir):
        os.environ['PATH'] += os.pathsep + ffmpeg_dir
    if not fast_check_ffmpeg():
        raise ImportError('FFmpeg is not installed.')
    fast_check_args(args)

    # checkpoint_MotionGenerator 会从新的 ArgumentConfig 传入原 InferenceConfig。
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    from src.ADEF_pipeline_framelevel_0721 import (
        ADEFPipelineFrameLevel0721,
    )
    pipeline = ADEFPipelineFrameLevel0721(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg,
    )
    pipeline.execute(args)


if __name__ == '__main__':
    main()

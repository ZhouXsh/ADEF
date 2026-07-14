# coding: utf-8
import os
import os.path as osp
import platform
import subprocess

import tyro

from src.config.argument_config import ArgumentConfig
from src.config.crop_config import CropConfig
from src.config.inference_config import InferenceConfig


if platform.system() == 'Windows':
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath


def partial_fields(target_class, kwargs):
    return target_class(
        **{key: value for key, value in kwargs.items() if hasattr(target_class, key)}
    )


def fast_check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except Exception:
        return False


def fast_check_args(args):
    if not osp.exists(args.reference):
        raise FileNotFoundError(f'reference info not found: {args.reference}')
    if not osp.exists(args.audio):
        raise FileNotFoundError(f'audio info not found: {args.audio}')


def main():
    tyro.extras.set_accent_color('bright_cyan')
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), 'ffmpeg')
    if osp.exists(ffmpeg_dir):
        os.environ['PATH'] += os.pathsep + ffmpeg_dir
    if not fast_check_ffmpeg():
        raise ImportError('FFmpeg is not installed. Please install ffmpeg and ffprobe.')

    fast_check_args(args)
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    from src.ADEF_pipeline_vasa import ADEFPipeline

    pipeline = ADEFPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    pipeline.execute(args)


if __name__ == '__main__':
    main()

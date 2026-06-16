# coding: utf-8
import os
import os.path as osp
import tyro
import subprocess
import platform
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig

if platform.system() == "Windows":
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

def fast_check_ffmpeg():           # 检查ffmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

def fast_check_args(args: ArgumentConfig):       # 检查参数
    if not osp.exists(args.reference):
        raise FileNotFoundError(f"reference info not found: {args.reference}")
    if not osp.exists(args.audio):
        raise FileNotFoundError(f"audio info not found: {args.audio}")

def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # specify configs for inference   指定推理的配置
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)    # 从args.__dict__选取InferenceConfig对应的字段 组成的字典
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    # init pipeline
    from src.ADEF_pipeline import ADEFPipeline
    pipeline = ADEFPipeline(
        inference_cfg=inference_cfg,          # 推理配置
        crop_cfg=crop_cfg                     # 裁剪（视频or图片）的配置
    )

    # run  执行推理
    pipeline.execute(args)

if __name__ == "__main__":
    main()


# -r /mnt/disk2/zhouxishi/JoyVASA/single_image_test/images/开心.jpg -a /mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD/raw_videos/M003/front/angry/level_3/M003_front_angry_level_3_001.wav --animation_mode human --cfg_scale 2.0 --output_dir "new"
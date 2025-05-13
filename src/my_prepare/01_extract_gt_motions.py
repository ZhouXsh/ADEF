import os
from tqdm import tqdm
import tyro  # 用于解析命令行参数
import multiprocessing
import sys  # 用于操作 Python 解释器的环境

sys.path.append(os.path.dirname(os.path.abspath("../")))

from src.config.argument_config import ArgumentConfig  # 用于解析命令行参数
from src.motion_extractor import make_motion_templete  # 用于处理视频并生成运动模板


# 定义一个函数，用于并行处理多个视频
def process_videos(args, video_list, suffix, cuda):
    # 创建参数列表，每个参数包含解析出的参数 args、视频路径 driving_video 和后缀 suffix

    # 单核进行
    # params = [(args, driving_video, suffix) for driving_video in video_list]

    # 四核进行
    videos = [video_list[i] for i in range(len(video_list)) if i%4==cuda]
    params = [
        (args, video, suffix, cuda) for video in videos 
    ]

    # 使用 multiprocessing.Pool 创建进程池，并设定同时运行 4 个进程
    with multiprocessing.Pool(processes=4) as pool:
        # 使用 starmap 方法并行执行 make_motion_templete 函数
        pool.starmap(make_motion_templete, params)


args = tyro.cli(ArgumentConfig)     # 解析命令行参数，获取 ArgumentConfig 类的实例
args.flag_do_crop = False           # 设置参数，flag_do_crop 设为 False，表示不进行裁剪
args.scale = 2.3                    # 设置缩放比例，将 scale 设为 2.3

# 定义根目录，该目录包含待处理的视频文件
root_dir = "/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos"

video_names = sorted(
    [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files if file.lower().endswith(".mp4")]
)

# 调用 process_videos 函数，并行处理视频，生成 ".pkl" 格式的运动模板文件
process_videos(args, video_names, suffix=".pkl", cuda=0)
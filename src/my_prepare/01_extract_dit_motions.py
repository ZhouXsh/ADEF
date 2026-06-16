import os
from tqdm import tqdm
import tyro  # 用于解析命令行参数
import multiprocessing
import sys  # 用于操作 Python 解释器的环境

# 将上级目录的绝对路径添加到 sys.path 中，以便导入上级目录中的模块
sys.path.append(os.path.dirname(os.path.abspath("../")))

from src.config.argument_config import ArgumentConfig  # 用于解析命令行参数
from src.dit_motion_extractor import make_motion_templete  # 用于处理视频并生成运动模板


# 定义一个函数，用于并行处理多个视频
def process_audios(args, audio_list, suffix, cuda):
    params = [(args, driving_audio, suffix, cuda) for driving_audio in audio_list]

    # 四核进行
    # audios = [audio_list[i] for i in range(len(audio_list)) if i%2==cuda]
    # params = [
    #     (args, audio, suffix, cuda) for audio in audios 
    # ]

    # 使用 multiprocessing.Pool 创建进程池，并设定同时运行 4 个进程
    with multiprocessing.Pool(processes=4) as pool:
        # 使用 starmap 方法并行执行 make_motion_templete 函数
        pool.starmap(make_motion_templete, params)

args = tyro.cli(ArgumentConfig)     # 解析命令行参数，获取 ArgumentConfig 类的实例
args.scale = 2.3                    # 设置缩放比例，将 scale 设为 2.3

audio_list = []
root_dir = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos'
for id in os.listdir(root_dir):    # 演员id
    front = os.path.join(root_dir, id, 'front')
    audio_list.extend([os.path.join(root, file) for root, _, files in os.walk(front) for file in files if file.lower().endswith(".wav")])
audio_list = sorted(audio_list)

process_audios(args, audio_list, suffix=".pkl", cuda=0)

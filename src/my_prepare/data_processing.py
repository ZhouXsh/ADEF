"""
数据处理模块
将视频处理、数据集划分、运动数据合并和模板生成整合为一个文件
"""
import os
import pickle
import random
import numpy as np
from tqdm import tqdm
import multiprocessing
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath("../")))   # src目录

from src.config.argument_config import ArgumentConfig
from src.motion_extractor import make_motion_template

video_dataset_root = "/home/zhouxishi/VirtualMan_proj/dataset/MEAD11/videos"

# ============================================
# 01_extract_motions - 从视频中提取运动模板
# ============================================
def _process_videos_single_gpu(args, video_list, suffix, cuda_id, cuda_count):
    """单GPU处理函数"""
    videos = [video_list[i] for i in range(len(video_list)) if i % cuda_count == cuda_id]
    args.device_id = cuda_id
    params = [(args, video, suffix) for video in videos]
    if videos:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            pool.starmap(make_motion_template, params)


def process_videos(args, video_list, suffix, cuda):
    """
    并行处理多个视频，生成运动模板（支持多GPU）

    Args:
        args: 命令行解析参数
        video_list: 视频文件路径列表
        suffix: 输出文件后缀
        cuda: GPU编号列表，如 [0,1,2,3]，或单个整数
    """
    import torch
    import multiprocessing

    if not torch.cuda.is_available():
        # CPU 模式
        print("No GPU available, using CPU")
        _process_videos_single_gpu(args, video_list, suffix, 0, 1)
        return

    cuda_count = 8

    cuda_list = [0,1,2,3,4,5,6,7]  # 默认使用所有GPU

    # 并行在多张GPU上执行
    processes = []
    for cuda_id in cuda_list:
        p = multiprocessing.Process(
            target=_process_videos_single_gpu,
            args=(args, video_list, suffix, cuda_id, cuda_count)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def extract_motions(args=None, video_list=None, suffix=".pkl", cuda=[0,1,2,3], root_dir=video_dataset_root):
    """
    从视频中提取运动模板

    Args:
        args: 命令行解析参数（若为None则使用默认配置）
        video_list: 视频文件路径列表（若为None则从root_dir扫描）
        suffix: 输出文件后缀
        cuda: GPU编号
        root_dir: 视频根目录

    Returns:
        处理后的视频列表
    """
    if args is None:
        args = ArgumentConfig()
        args.flag_do_crop = False
        args.scale = 2.3

    if video_list is None:
        video_names = sorted([
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files
            if file.lower().endswith(".mp4")
        ])
    else:
        video_names = video_list
    process_videos(args, video_names, suffix=suffix, cuda=cuda)
    return video_names


# ============================================
# 02_divide_dataset - 划分训练集和测试集
# ============================================
def prefix(filename):
    """获取文件名的前缀（去掉扩展名）"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """获取文件的基本名（去掉路径和扩展名）"""
    return prefix(os.path.basename(filename))


def remove_suffix(filepath):
    """去掉文件的后缀，保留路径和文件名"""
    return os.path.join(os.path.dirname(filepath), basename(filepath))


def divide_dataset(root_dir=video_dataset_root, train_ratio=0.9, output_dir=None):
    """
    将视频数据按比例划分为训练集和测试集

    Args:
        root_dir: 视频根目录
        train_ratio: 训练集比例
        output_dir: 输出目录（默认为当前目录）

    Returns:
        (train_list, test_list): 训练集和测试集视频路径列表
    """
    video_list = []

    for id in os.listdir(root_dir):
        front = os.path.join(root_dir, id, 'front')
        for emo in os.listdir(front):
            emo_dir = os.path.join(front, emo)
            for level in os.listdir(emo_dir):
                level_dir = os.path.join(emo_dir, level)
                for video_file in os.listdir(level_dir):
                    if video_file.endswith('.mp4'):
                        video_path = os.path.join(level_dir, video_file)
                        video_list.append(video_path)
                        if emo == 'neutral':
                            video_list.append(video_path)
                            video_list.append(video_path)

    random.shuffle(video_list)
    num_train_labels = int(train_ratio * len(video_list))

    train_list = video_list[:num_train_labels]
    test_list = video_list[num_train_labels:]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        train_path = os.path.join(output_dir, "train.txt")
        test_path = os.path.join(output_dir, "test.txt")
    else:
        train_path = "train.txt"
        test_path = "test.txt"

    with open(train_path, "w") as f:
        for item in train_list:
            f.write(f"{item}\n")

    with open(test_path, "w") as f:
        for item in test_list:
            f.write(f"{item}\n")

    return train_list, test_list


# ============================================
# 03_merge_motions - 合并运动数据
# ============================================
def merge_motions(train_data=None, test_data=None, output_name='front_all_motions.pkl'):
    """
    合并训练集和测试集的运动数据

    Args:
        train_data: 训练集数据列表（若为None则从train.txt读取）
        test_data: 测试集数据列表（若为None则从test.txt读取）
        output_name: 输出文件名

    Returns:
        合并后的运动数据字典
    """
    if train_data is None:
        with open("train.txt", "r") as f:
            train_data = [line.strip() for line in f.readlines()]

    if test_data is None:
        with open("test.txt", "r") as f:
            test_data = [line.strip() for line in f.readlines()]

    data = train_data + test_data
    all_items = {}

    for item in tqdm(data, desc="加载运动数据"):
        key = item[:-4] + '.wav'
        motion_name = item[:-4] + '.pkl'
        motions = pickle.load(open(motion_name, 'rb'))
        all_items[key] = motions

    save_name = output_name
    pickle.dump(all_items, open(save_name, 'wb'))
    print("运动数据处理完成")

    return all_items


# ============================================
# 04_generate_template - 生成运动模板统计信息
# ============================================
def generate_template(data_root='front_all_motions.pkl', save_name='motion_template.pkl'):
    """
    计算运动数据的统计信息并生成模板

    Args:
        data_root: 运动数据文件路径
        save_name: 输出模板文件名

    Returns:
        包含所有统计信息的运动模板字典
    """
    scale_list = []
    R_list = []
    pitch_list = []
    yaw_list = []
    roll_list = []
    t_list = []
    exp_list = []

    motions = pickle.load(open(data_root, 'rb'))

    lip_lst_array = None
    eyes_lst_array = None
    audio_names = motions.keys()
    for audio_name in audio_names:
        motion_data = motions[audio_name]
        seq_len = motion_data["n_frames"]
        for frame_idx in range(seq_len):
            scale_list.append(motion_data['motion'][frame_idx]["scale"].flatten())
            R_list.append(motion_data['motion'][frame_idx]["R"].flatten())
            t_list.append(motion_data['motion'][frame_idx]["t"].flatten())
            exp_list.append(motion_data['motion'][frame_idx]["exp"].flatten())
            pitch_list.append(motion_data['motion'][frame_idx]["pitch"].flatten())
            yaw_list.append(motion_data['motion'][frame_idx]["yaw"].flatten())
            roll_list.append(motion_data['motion'][frame_idx]["roll"].flatten())

    lip_lst_array = np.array([data.flatten() for data in motion_data['c_lip_lst']]).astype(np.float32)
    eyes_lst_array = np.array([data.flatten() for data in motion_data['c_eyes_lst']]).astype(np.float32)
    
    R_array = np.array(R_list)
    t_array = np.array(t_list)
    exp_array = np.array(exp_list)
    scale_array = np.array(scale_list)
    pitch_array = np.array(pitch_list)
    yaw_array = np.array(yaw_list)
    roll_array = np.array(roll_list)

    abs_max_scale = np.max(abs(scale_array), axis=0)
    abs_max_R = np.max(abs(R_array), axis=0)
    abs_max_t = np.max(abs(t_array), axis=0)
    abs_max_exp = np.max(abs(exp_array), axis=0)
    abs_max_pitch = np.max(abs(pitch_array), axis=0)
    abs_max_yaw = np.max(abs(yaw_array), axis=0)
    abs_max_roll = np.max(abs(roll_array), axis=0)
    abs_max_lip = np.max(abs(lip_lst_array), axis=0)
    abs_max_eyes = np.max(abs(eyes_lst_array), axis=0)

    max_scale = np.max(scale_array, axis=0)
    max_R = np.max(R_array, axis=0)
    max_t = np.max(t_array, axis=0)
    max_exp = np.max(exp_array, axis=0)
    max_pitch = np.max(pitch_array, axis=0)
    max_yaw = np.max(yaw_array, axis=0)
    max_roll = np.max(roll_array, axis=0)
    max_lip = np.max(lip_lst_array, axis=0)
    max_eyes = np.max(eyes_lst_array, axis=0)

    min_scale = np.min(scale_array, axis=0)
    min_R = np.min(R_array, axis=0)
    min_t = np.min(t_array, axis=0)
    min_exp = np.min(exp_array, axis=0)
    min_pitch = np.min(pitch_array, axis=0)
    min_yaw = np.min(yaw_array, axis=0)
    min_roll = np.min(roll_array, axis=0)
    min_lip = np.min(lip_lst_array, axis=0)
    min_eyes = np.min(eyes_lst_array, axis=0)

    mean_scale = np.mean(scale_array, axis=0)
    mean_R = np.mean(R_array, axis=0)
    mean_t = np.mean(t_array, axis=0)
    mean_exp = np.mean(exp_array, axis=0)
    mean_pitch = np.mean(pitch_array, axis=0)
    mean_yaw = np.mean(yaw_array, axis=0)
    mean_roll = np.mean(roll_array, axis=0)
    mean_lip = np.mean(lip_lst_array, axis=0)
    mean_eyes = np.mean(eyes_lst_array, axis=0)

    std_scale = np.std(scale_array, axis=0)
    std_R = np.std(R_array, axis=0)
    std_t = np.std(t_array, axis=0)
    std_exp = np.std(exp_array, axis=0)
    std_pitch = np.std(pitch_array, axis=0)
    std_yaw = np.std(yaw_array, axis=0)
    std_roll = np.std(roll_array, axis=0)
    std_lip = np.std(lip_lst_array, axis=0)
    std_eyes = np.std(eyes_lst_array, axis=0)

    motion_template = {
        "mean_scale": mean_scale,
        "mean_R": mean_R,
        "mean_t": mean_t,
        "mean_exp": mean_exp,
        "mean_pitch": mean_pitch,
        "mean_yaw": mean_yaw,
        "mean_roll": mean_roll,
        "mean_lip": mean_lip,
        "mean_eyes": mean_eyes,
        "std_scale": std_scale,
        "std_R": std_R,
        "std_t": std_t,
        "std_exp": std_exp,
        "std_pitch": std_pitch,
        "std_yaw": std_yaw,
        "std_roll": std_roll,
        "std_lip": std_lip,
        "std_eyes": std_eyes,
        "max_scale": max_scale,
        "max_R": max_R,
        "max_t": max_t,
        "max_exp": max_exp,
        "max_pitch": max_pitch,
        "max_yaw": max_yaw,
        "max_roll": max_roll,
        "max_lip": max_lip,
        "max_eyes": max_eyes,
        "min_scale": min_scale,
        "min_R": min_R,
        "min_t": min_t,
        "min_exp": min_exp,
        "min_pitch": min_pitch,
        "min_yaw": min_yaw,
        "min_roll": min_roll,
        "min_lip": min_lip,
        "min_eyes": min_eyes,
        "abs_max_scale": abs_max_scale,
        "abs_max_R": abs_max_R,
        "abs_max_t": abs_max_t,
        "abs_max_exp": abs_max_exp,
        "abs_max_pitch": abs_max_pitch,
        "abs_max_yaw": abs_max_yaw,
        "abs_max_roll": abs_max_roll,
        "abs_max_lip": abs_max_lip,
        "abs_max_eyes": abs_max_eyes,
    }

    pickle.dump(motion_template, open(save_name, 'wb'))
    return motion_template


if __name__ == "__main__":
    import tyro
    args = tyro.cli(ArgumentConfig)
    args.flag_do_crop = False
    args.scale = 2.3
    extract_motions(args=args)
    divide_dataset()
    merge_motions()
    generate_template()

    # 单个测试
    # args.device_id = 1
    # make_motion_template(args, '/mnt/disk3/zhouxishi/ADEFv4/src/dataset/RAVDESS_Processed/videos/Actor_01/front/angry/level_1/01-01-05-01-01-01-01.mp4', suffix=".pkl")
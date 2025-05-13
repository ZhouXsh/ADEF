import os
import subprocess
import multiprocessing

from tqdm import tqdm

# 功能：检查给定路径是否为视频文件或目录。
# 实现：如果文件的扩展名是 .mp4, .mov, .avi, 或 .webm，则认为是视频文件，或者文件是一个目录。
def is_video(file_path):
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or os.path.isdir(file_path):
        return True
    return False

# 功能：获取文件名的前缀（去除扩展名）。
# 实现：通过 rfind() 查找文件名中最后一个点的位置，从而提取文件扩展名前的部分。
def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]

# 功能：获取文件名（不包含路径）。
# 实现：通过 os.path.basename() 获取文件的基本名称，并去除扩展名。
def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(os.path.basename(filename))

# 功能：获取去除扩展名后的文件路径。
# 实现：通过 basename() 提取文件名并去除扩展名，返回不带后缀的文件路径。
def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return os.path.join(os.path.dirname(filepath), basename(filepath))

def extract_audio(filename):
    suffix=".wav"
    audio_output_name = remove_suffix(filename) + suffix
    if os.path.exists(audio_output_name):
        # print("audio already generated~")
        return
    if os.path.exists(filename) and is_video(filename):
        subprocess.run(['ffmpeg', '-loglevel', 'quiet', '-i', filename, '-vn', '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', audio_output_name, '-y'])
    else:
        raise Exception(f"{filename} is not a supported type!")
    if not os.path.exists(audio_output_name):
        print(f"无法提取音频{audio_output_name}")
        return

# 功能：并行处理视频列表中的所有视频文件，提取音频。
# 实现：使用 multiprocessing.Pool 创建一个进程池，将 extract_audio 函数应用到 video_list 中的每一个视频文件。这里使用了 starmap 来传递参数。
def process_videos(video_list):
    with multiprocessing.Pool(processes=12) as pool:
        with tqdm(total=len(video_list), desc="视频处理进度") as pbar:
            for _ in pool.imap_unordered(extract_audio, video_list):  
                pbar.update(1)  # 实时更新进度

# 设置视频根目录 root_dir，并获取该目录下所有以 .mp4 为后缀的文件（排序后）。
# 调用 process_videos() 函数并传递视频文件列表，开始并行提取音频。
if __name__ == "__main__":
    root_dir = "/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos"
    # 多层目录
    video_names = sorted(
        [os.path.join(root, file) for root, _, files in os.walk(root_dir) for file in files if file.lower().endswith(".mp4")]
    )
    process_videos(video_names)
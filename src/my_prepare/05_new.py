import os
import subprocess
import multiprocessing
from tqdm import tqdm


def is_video(file_path):
    return (
        os.path.isfile(file_path)
        and file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm"))
    )


def remove_suffix(filepath):
    return os.path.splitext(filepath)[0]


def get_video_duration(filename):
    """
    返回视频流时长，而不是容器总时长或音频流时长。
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filename,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    duration_str = result.stdout.strip()

    if not duration_str or duration_str == "N/A":
        raise RuntimeError(f"无法读取视频时长: {filename}")

    return float(duration_str)


def extract_audio(filename, align_to_video=True):
    suffix = ".wav"
    audio_output_name = remove_suffix(filename) + suffix

    # if os.path.exists(audio_output_name):
    #     return

    if not os.path.exists(filename) or not is_video(filename):
        raise Exception(f"{filename} is not a supported video file!")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", filename,
        "-map", "0:a:0",
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
    ]

    if align_to_video:
        video_duration = get_video_duration(filename)
        cmd += [
            "-af", "apad",
            "-t", str(video_duration),
        ]

    cmd += [audio_output_name]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"音频提取失败: {filename}")
        print(e)
        return

    if not os.path.exists(audio_output_name):
        print(f"无法提取音频: {audio_output_name}")


def process_videos(video_list):
    with multiprocessing.Pool(processes=12) as pool:
        with tqdm(total=len(video_list), desc="视频处理进度") as pbar:
            for _ in pool.imap_unordered(extract_audio, video_list):
                pbar.update(1)


if __name__ == "__main__":
    root_dir = "/home/Zhouxishi/VirtualMan_proj/dataset/MEAD11/videos"

    video_names = sorted([
        os.path.join(root, file)
        for root, _, files in os.walk(root_dir)
        for file in files
        if file.lower().endswith(".mp4")
    ])

    process_videos(video_names)
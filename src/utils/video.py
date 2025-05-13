# coding: utf-8

"""
Functions for processing video

ATTENTION: you need to install ffmpeg and ffprobe in your env!

视频处理功能
注意：您需要在环境中安装ffmpeg和ffprobe！

"""
import os
import os.path as osp
import numpy as np
import subprocess
import imageio
import cv2
from rich.progress import track

from .rprint import rlog as log
from .rprint import rprint as print
from .helper import prefix

# 执行指令
def exec_cmd(cmd):
    return subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# 将图像转换为视频（无声音）
def images2video(images, wfp, **kwargs):
    fps = kwargs.get('fps', 25)
    video_format = kwargs.get('format', 'mp4')  # default is mp4 format
    codec = kwargs.get('codec', 'libx264')  # default is libx264 encoding
    quality = kwargs.get('quality')  # video quality
    pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
    image_mode = kwargs.get('image_mode', 'rgb')
    macro_block_size = kwargs.get('macro_block_size', 2)
    ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

    writer = imageio.get_writer(
        wfp, fps=fps, format=video_format,
        codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat, macro_block_size=macro_block_size
    )

    n = len(images)
    print(f'应该的视频长度 {n}')
    for i in track(range(n), description='Writing', transient=True):
        if image_mode.lower() == 'bgr':
            writer.append_data(images[i][..., ::-1])
        else:
            writer.append_data(images[i])

    writer.close()

# 没用
def video2gif(video_fp, fps=30, size=256):
    if osp.exists(video_fp):
        d = osp.split(video_fp)[0]
        fn = prefix(osp.basename(video_fp))
        palette_wfp = osp.join(d, 'palette.png')
        gif_wfp = osp.join(d, f'{fn}.gif')
        # generate the palette
        cmd = f'ffmpeg -i "{video_fp}" -vf "fps={fps},scale={size}:-1:flags=lanczos,palettegen" "{palette_wfp}" -y'
        exec_cmd(cmd)
        # use the palette to generate the gif
        cmd = f'ffmpeg -i "{video_fp}" -i "{palette_wfp}" -filter_complex "fps={fps},scale={size}:-1:flags=lanczos[x];[x][1:v]paletteuse" "{gif_wfp}" -y'
        exec_cmd(cmd)
        return gif_wfp
    else:
        raise FileNotFoundError(f"video_fp: {video_fp} not exists!")

# 没用
def merge_audio_video(video_fp, audio_fp, wfp):
    if osp.exists(video_fp) and osp.exists(audio_fp):
        cmd = f'ffmpeg -i "{video_fp}" -i "{audio_fp}" -c:v copy -c:a aac "{wfp}" -y'
        exec_cmd(cmd)
        print(f'merge {video_fp} and {audio_fp} to {wfp}')
    else:
        print(f'video_fp: {video_fp} or audio_fp: {audio_fp} not exists!')

# 没用
def blend(img: np.ndarray, mask: np.ndarray, background_color=(255, 255, 255)):
    mask_float = mask.astype(np.float32) / 255.
    background_color = np.array(background_color).reshape([1, 1, 3])
    bg = np.ones_like(img) * background_color
    img = np.clip(mask_float * img + (1 - mask_float) * bg, 0, 255).astype(np.uint8)
    return img

# 没用
def concat_frames(driving_image_lst, source_image_lst, I_p_lst):
    # TODO: add more concat style, e.g., left-down corner driving
    out_lst = []
    h, w, _ = I_p_lst[0].shape
    source_image_resized_lst = [cv2.resize(img, (w, h)) for img in source_image_lst]

    for idx, _ in track(enumerate(I_p_lst), total=len(I_p_lst), description='Concatenating result...'):
        I_p = I_p_lst[idx]
        source_image_resized = source_image_resized_lst[idx] if len(source_image_lst) > 1 else source_image_resized_lst[0]

        if driving_image_lst is None:
            out = np.hstack((source_image_resized, I_p))
        else:
            driving_image = driving_image_lst[idx]
            driving_image_resized = cv2.resize(driving_image, (w, h))
            out = np.hstack((driving_image_resized, source_image_resized, I_p))

        out_lst.append(out)
    return out_lst

# 没用
class VideoWriter:
    def __init__(self, **kwargs):
        self.fps = kwargs.get('fps', 30)
        self.wfp = kwargs.get('wfp', 'video.mp4')
        self.video_format = kwargs.get('format', 'mp4')
        self.codec = kwargs.get('codec', 'libx264')
        self.quality = kwargs.get('quality')
        self.pixelformat = kwargs.get('pixelformat', 'yuv420p')
        self.image_mode = kwargs.get('image_mode', 'rgb')
        self.ffmpeg_params = kwargs.get('ffmpeg_params')

        self.writer = imageio.get_writer(
            self.wfp, fps=self.fps, format=self.video_format,
            codec=self.codec, quality=self.quality,
            ffmpeg_params=self.ffmpeg_params, pixelformat=self.pixelformat
        )

    def write(self, image):
        if self.image_mode.lower() == 'bgr':
            self.writer.append_data(image[..., ::-1])
        else:
            self.writer.append_data(image)

    def close(self):
        if self.writer is not None:
            self.writer.close()

# 没用
def change_video_fps(input_file, output_file, fps=20, codec='libx264', crf=12):
    cmd = f'ffmpeg -i "{input_file}" -c:v {codec} -crf {crf} -r {fps} "{output_file}" -y'
    exec_cmd(cmd)

# 获取帧率
def get_fps(filepath, default_fps=25):
    try:
        fps = cv2.VideoCapture(filepath).get(cv2.CAP_PROP_FPS)

        if fps in (0, None):
            fps = default_fps
    except Exception as e:
        log(e)
        fps = default_fps

    return fps

# 没用
def has_audio_stream(video_path: str) -> bool:
    """
    Check if the video file contains an audio stream.

    :param video_path: Path to the video file
    :return: True if the video contains an audio stream, False otherwise
    """
    if osp.isdir(video_path):
        return False

    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        f'"{video_path}"'
    ]

    try:
        # result = subprocess.run(cmd, capture_output=True, text=True)
        result = exec_cmd(' '.join(cmd))
        if result.returncode != 0:
            log(f"Error occurred while probing video: {result.stderr}")
            return False

        # Check if there is any output from ffprobe command
        return bool(result.stdout.strip())
    except Exception as e:
        log(
            f"Error occurred while probing video: {video_path}, "
            "you may need to install ffprobe! (https://ffmpeg.org/download.html) "
            "Now set audio to false!",
            style="bold red"
        )
    return False

# 为视频添加音频
def add_audio_to_video(silent_video_path: str, audio_video_path: str, output_video_path: str, remove_temp=True):
    print("audio: ", audio_video_path)
    # '-c:a', 'mp3',
    if audio_video_path.endswith("wav"):
        # cmd = [
        #     'ffmpeg',
        #     '-y',
        #     '-loglevel', 'quiet',
        #     '-i', f'{silent_video_path}',
        #     '-i', f'{audio_video_path}',
        #     '-c:v', 'copy',                     # 'libx264' --> 'copy'      √
        #     '-b:v', '2M', 
        #     '-r', '25', 
        #     f'{output_video_path}'
        # ]  
        cmd = f'ffmpeg -y -loglevel quiet -i {silent_video_path} -i {audio_video_path} -c:v copy -b:v 2M -r 25 {output_video_path}'
    else:
        # cmd = [
        #     'ffmpeg',
        #     '-y',
        #     '-i', f'"{silent_video_path}"',
        #     '-i', f'"{audio_video_path}"',
        #     '-map', '0:v',
        #     '-map', '1:a',
        #     '-c:v', 'copy',
        #     '-shortest',
        #     f'"{output_video_path}"'
        # ]
        cmd = f'ffmpeg -y -i {silent_video_path} -i {audio_video_path} -map 0:v -map 1:a -c:v copy -shortest {output_video_path}'
    try:
        print(f'cmd:{cmd}')
        # subprocess.run(cmd)                   # 会出现无法检测文件的情况
        subprocess.run(cmd, shell=True)
        log(f"Video with audio generated successfully: {output_video_path}")
        if remove_temp:
            os.remove(silent_video_path)
    except subprocess.CalledProcessError as e:
        log(f"Error occurred: {e}")

# 我写的
def stack_videos_opencv(video_path1, video_path2, output_path, remove_temp=True):
    # 打开视频流
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)
    
    # 获取原始视频参数
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    target_size=(width//2, height//2)

    # 打开视频流
    cap1 = cv2.VideoCapture(video_path1)
    cap2 = cv2.VideoCapture(video_path2)

    # 获取视频参数
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)

    # 验证尺寸一致性
    if (width, height) != (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))):
        raise ValueError("视频尺寸必须相同")

    # 创建输出视频写入器 (目标宽度, 2倍目标高度)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 或 mp4v
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height//2))

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # 任一视频结束则停止
        if not (ret1 and ret2):
            break

        ## 直接缩放（保持宽高比一致时无需黑边）
        resized1 = cv2.resize(frame1, target_size, interpolation=cv2.INTER_AREA)
        resized2 = cv2.resize(frame2, target_size, interpolation=cv2.INTER_AREA)

        # 垂直拼接
        stacked = cv2.hconcat([resized1, resized2])   # hconcat：左右拼接      vconcat：上下拼接
        
        # 写入帧
        out.write(stacked)

    # 释放资源
    cap1.release()
    cap2.release()
    out.release()

    if remove_temp:
        os.remove(video_path1)
        os.remove(video_path2)

# 没用
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

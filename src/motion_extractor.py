# coding: utf-8
import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import numpy as np
import os
import os.path as osp
import subprocess
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import  get_fps
from .utils.io import load_video, dump
from .utils.helper import is_video, is_template, remove_suffix, is_square_video
from .ADEF_wrapper import ADEFWrapper

'''
用于训练时提取motion的Motion Extractor
'''

# 检查ffmpeg是否存在
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

# 检查参数是否存在
def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.reference):
        raise FileNotFoundError(f"reference info not found: {args.reference}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

# 返回总参数kwargs中，target_class部分的参数字段
def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

# 生成绝对路径
def make_abs_path(fn):
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)

# LivePortrait的运动提取器
class LivePortraitMotionExtractor(object):

    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.adef_wrapper: ADEFWrapper = ADEFWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def make_motion_template(self, I_lst, c_eyes_lst, c_lip_lst, **kwargs):
        n_frames = I_lst.shape[0]       # T x 1 x 3 x H x W
        template_dct = {
            'n_frames': n_frames,         # 总帧数
            'output_fps': kwargs.get('output_fps', 25),
            'motion': [],
            'c_eyes_lst': [],
            'c_lip_lst': [],
        }

        for i in track(range(n_frames), description='Making motion templates...', total=n_frames):  # 逐帧处理
            # collect s, R, δ and t for inference
            I_i = I_lst[i]          # 1 x 3 x H x W 该帧的图像
            x_i_info = self.adef_wrapper.get_kp_info(I_i)          # 获取图像的运动参数
            x_s = self.adef_wrapper.transform_keypoint(x_i_info)   # 计算图像的隐式关键点
            R_i = get_rotation_matrix(x_i_info['pitch'], x_i_info['yaw'], x_i_info['roll'])   # 旋转矩阵

            item_dct = {
                'scale': x_i_info['scale'].cpu().numpy().astype(np.float32),
                'R': R_i.cpu().numpy().astype(np.float32),
                'exp': x_i_info['exp'].cpu().numpy().astype(np.float32),
                't': x_i_info['t'].cpu().numpy().astype(np.float32),
                'kp': x_i_info['kp'].cpu().numpy().astype(np.float32),
                'x_s': x_s.cpu().numpy().astype(np.float32),
                'pitch': x_i_info['pitch'].cpu().numpy().astype(np.float32),
                'yaw': x_i_info['yaw'].cpu().numpy().astype(np.float32),
                'roll': x_i_info['roll'].cpu().numpy().astype(np.float32)
            }

            template_dct['motion'].append(item_dct)

            c_eyes = c_eyes_lst[i].astype(np.float32)
            template_dct['c_eyes_lst'].append(c_eyes)

            c_lip = c_lip_lst[i].astype(np.float32)
            template_dct['c_lip_lst'].append(c_lip)

        return template_dct

    def execute(self, args, suffix=".pkl"):  # copy from LivePortrait pipeline的process driving info部分
        # for convenience
        inf_cfg = self.adef_wrapper.inference_cfg

# ------######## process driving info ########   生成motion字典----------------------------------------------------------
        flag_load_from_template = is_template(args.driving)
        driving_rgb_crop_256x256_lst = None
        wfp_template = None
        wfp_template = remove_suffix(args.driving) + suffix    # xxx.pkl

        if osp.exists(args.driving):  # args.driving：视频的绝对路径
            if is_video(args.driving):
                flag_is_driving_video = True
                # load from video file, AND make motion template
                output_fps = int(get_fps(args.driving))
                # log(f"Load driving video from: {args.driving}, FPS is {output_fps}")
                driving_rgb_lst = load_video(args.driving)   # 视频 ——》 RGB图像列表
            else:
                raise Exception(f"{args.driving} is not a supported type!")

            ######## make motion template ########
            # log("Start making driving motion template...")
            driving_n_frames = len(driving_rgb_lst)
            n_frames = driving_n_frames    # 总帧数

            # 裁剪，resize，landmark    基于跟踪的landmark/对齐
            if inf_cfg.flag_crop_driving_video or (not is_square_video(args.driving)):  # 需要裁剪 或 不是1:1的
                # print("croping: ", inf_cfg.flag_crop_driving_video)
                ret_d = self.cropper.crop_driving_video(driving_rgb_lst)
                # log(f'Driving video is cropped, {len(ret_d["frame_crop_lst"])} frames are processed.')
                if len(ret_d["frame_crop_lst"]) is not n_frames and flag_is_driving_video:   # 有些帧没有检测到人脸
                    n_frames = min(n_frames, len(ret_d["frame_crop_lst"]))
                driving_rgb_crop_lst, driving_lmk_crop_lst = ret_d['frame_crop_lst'], ret_d['lmk_crop_lst']
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_crop_lst]
            else:  # 不需要裁剪 且 是1:1的 （但不一定是256x256的）  需要resize
                # print("without crop ...")
                driving_lmk_crop_lst = self.cropper.calc_lmks_from_cropped_video(driving_rgb_lst)
                driving_rgb_crop_256x256_lst = [cv2.resize(_, (256, 256)) for _ in driving_rgb_lst]  # force to resize to 256x256
            # 获得裁剪&resize成256x256的RGB图像列表：driving_rgb_crop_256x256_lst
            # 和 裁剪后的landmark列表：driving_lmk_crop_lst

            c_d_eyes_lst, c_d_lip_lst = self.adef_wrapper.calc_ratio(driving_lmk_crop_lst)  # 计算眼睛嘴巴close的比例
            # save the motion template
            I_d_lst = self.adef_wrapper.prepare_videos(driving_rgb_crop_256x256_lst)    # T x 1 x 3 x H x W
            driving_template_dct = self.make_motion_template(I_d_lst, c_d_eyes_lst, c_d_lip_lst, output_fps=output_fps)
                # driving_template_dct = template_dct = {
                #     'n_frames': n_frames,         # 总帧数
                #     'output_fps': kwargs.get('output_fps', 25),
                #     'motion': [],
                #     'c_eyes_lst': [],
                #     'c_lip_lst': [],
                # }
            wfp_template = remove_suffix(args.driving) + suffix     # xxx.pkl
            dump(wfp_template, driving_template_dct)          # 保存文件
            # log(f"Dump motion template to {wfp_template}")
        else:
            raise Exception(f"{args.driving} does not exist!")

# （处理训练视频）生成输入视频的 运动模版
def make_motion_templete(args, driving_video, suffix=".pkl", gpu_id=0): 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)       # 让该进程只看到指定 GPU
    # print(f"Process {os.getpid()} using GPU {gpu_id} for video {driving_video}")

    wfp_template = remove_suffix(driving_video) + suffix    # xxx.pkl
    if os.path.exists(wfp_template):  # 已处理
        return

    # configs
    args.driving = driving_video    # 视频的绝对路径
    fast_check_args(args)
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)      # 推理参数
    crop_cfg = partial_fields(CropConfig, args.__dict__)                # 裁剪参数

    # ffmpeg
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
    if not fast_check_ffmpeg():
        raise ImportError( "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html")

    try:
        # feature_extract
        motion_extractor = LivePortraitMotionExtractor(
            inference_cfg=inference_cfg,
            crop_cfg=crop_cfg
        )
        motion_extractor.execute(args, suffix=suffix)
    except Exception as e:
        print(f"Exception in motion extractor: {e}")
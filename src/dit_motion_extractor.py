# coding: utf-8
import torch
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import numpy as np
import os
import os.path as osp
import subprocess
from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .utils.camera import get_rotation_matrix
from .utils.io import dump
from .utils.helper import remove_suffix
from .utils.rprint import rlog as log

import pickle
import yaml
import math
import librosa
import torch.nn.functional as F

from .utils.helper import load_model
from .utils.filter import smooth_

# 检查ffmpeg是否存在
def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False

# 返回总参数kwargs中，target_class部分的参数字段
def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

# coding: utf-8

class DiTMotionExtractor(object):
    def __init__(self, inference_cfg: InferenceConfig):
        self.inference_cfg = inference_cfg
        self.device_id = inference_cfg.device_id
        if inference_cfg.flag_force_cpu:
            self.device = 'cpu'
        else:
            try:
                if torch.backends.mps.is_available():
                    self.device = 'mps'
                else:
                    self.device = 'cuda:' + str(self.device_id)
            except:
                self.device = 'cuda:' + str(self.device_id)
        
        model_config = yaml.load(open(inference_cfg.models_config, 'r'), Loader=yaml.SafeLoader)

        # Motion Genertor     运动生成器
        self.motion_generator, self.motion_generator_args = load_model(inference_cfg.checkpoint_MotionGenerator, model_config, self.device, 'motion_generator')
        log(f'Load motion_generator from {osp.realpath(inference_cfg.checkpoint_MotionGenerator)} done.')
        self.n_motions = self.motion_generator_args.n_motions                  # 单个音频子序列（片段）的帧数
        self.n_prev_motions = self.motion_generator_args.n_prev_motions
        self.fps = self.motion_generator_args.fps
        self.audio_unit = 16000. / self.fps     # num of samples per frame     每帧样本数
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.motion_generator_args.pad_mode
        self.use_indicator = self.motion_generator_args.use_indicator
        self.templete_dict = pickle.load(open(inference_cfg.motion_template_path, 'rb'))
    
    # （根据音频）生成运动序列！！！
    def gen_motion_sequence(self, args, suffix=".pkl"): 
       # preprocess audio    处理音频
        log(f"start loading audio from {args.audio}")
        audio, _ = librosa.load(args.audio, sr=16000, mono=True)  # 从给定路径加载音频，设置采样率为 16,000 Hz，mono=True表示将音频转换为单声道
        # audio shape: (总采样时长, )
        log(f"audio loaded! {audio.shape}")
        if isinstance(audio, np.ndarray):       # (n_samples,)   
            audio = torch.from_numpy(audio).to(self.device)         # 转成torch张量类型  [n_samples]    eg:[170528]
        assert audio.ndim == 1, 'Audio must be 1D tensor.'          # 确保音频是一个一维张量（即单通道音频）
        log(f"loading audio from: {args.audio}")
        audio = F.pad(audio, (1280, 640), "constant", 0)     # F.pad: 对音频进行填充，左侧填充 1280 个样本，右侧填充 640 个样本。填充值为 0。  eg: [170528] -> [172448]


        # crop audio into n_subdivision according to n_motions   
        # 根据n_motions（帧/片段数）将音频裁剪为n_subdivision（片段数）份
        clip_len = int(len(audio) / 16000 * self.fps)       # len(audio) / 16000为音频时长；clip_len为总帧数     eg:269
        stride = self.n_motions                   # 步长（每个音频片段的帧数）
        if clip_len <= self.n_motions:
            n_subdivision = 1                               # 总帧数小于步长，只分成一份
        else:
            n_subdivision = math.ceil(clip_len / stride)    # 按步长分成n_subdivision份      eg:3

        # padding  填充
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)  # 计算音频需要填充的样本数量，以保证音频长度与 n_subdivision 对应的运动生成样本数量匹配   eg:  n_padding_audio_samples = 19552
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)           # 需要填充的帧数
        if n_padding_audio_samples > 0:          # 需要填充的样本数量   eg: 19552
            if self.pad_mode == 'zero':               # 使用 0 填充
                padding_value = 0
            elif self.pad_mode == 'replicate':        # 复制最后一个音频样本填充
                padding_value = audio[-1]
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)     #  172448 + 19552 = 192000 = self.n_audio_samples * n_subdivision
            # 其中 n_audio_samples：n_motions=100帧对应的采样数   ；  n_subdivision：分成了多少个n_motions=100帧
            # 左侧填充0个样本，右侧填充 n_padding_audio_samples 个样本。填充值为 padding_value。  shape：(n_samples,)

        # generate motions
        coef_list = []
        for i in range(0, n_subdivision):              # 音频的份数（子序列），每一份包含 stride = n_motions=100帧。
            start_idx = i * stride                     # 这份 起始 帧的索引
            end_idx = start_idx + self.n_motions       # 这份 结束 帧的索引
            indicator = torch.ones((1, self.n_motions)).to(self.device) if self.use_indicator else None  # 指示器（全 1 的张量），用于在最后一个子序列上标记填充部分。(1, self.n_motions=100)
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0     # 指示器（真实音频为True，填充部分为0），用于在最后一个子序列上标记 填充的部分。
            audio_in = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0)  # audio_in: 根据音频的子序列索引提取音频片段，并增加一个维度以形成 (1, n_samples_in_subdivision) 的形状。
            # shape: [1, self.n_audio_samples = audio_len / n_subdivision]   eg:[1, 64000]
  
            if i == 0:    # 第一个片段没有先前帧的指导
                # 调用 motion_generator.sample 生成运动特征。返回的 motion_feat 是生成的运动特征，noise 是噪声，prev_audio_feat 是先前的音频特征。
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,        # (1, n_motion=100（即frame per seq), n_features=73) - 每个子序列生成的运动特征
                                                                        indicator=indicator, cfg_mode=args.cfg_mode,
                                                                        cfg_cond=args.cfg_cond, cfg_scale=args.cfg_scale,
                                                                        dynamic_threshold=0)
                # 其中motion_feat：(N=1, L=100, motion_feat_dim=73)   生成的运动特征（去噪的结果）
                # noise:           (N=1, L=100, motion_feat_dim=73)  随机噪声（去噪的开始）
                # prev_audio_feat: (N=1, L=100, feature_dim=256)     当前片段提取的音频特征。
            else:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                        prev_motion_feat, prev_audio_feat, noise,     # 需要添加先前的 运动特征，音频特征，噪声
                                                                        indicator=indicator, cfg_mode=args.cfg_mode,
                                                                        cfg_cond=args.cfg_cond, cfg_scale=args.cfg_scale,
                                                                        dynamic_threshold=0)
            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()        # copy临近的n_prev_motions=0帧，先前的运动特征  (N=1, L=10, motion_feat_dim=73)
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]             # copy，作为先前的音频特征                      (N=1, L=10, feature_dim=256)

            motion_coef = motion_feat         #  运动系数 "coefficient"（系数）  (N=1, L=100, motion_feat_dim=73)
            if i == n_subdivision - 1 and n_padding_frames > 0:         # 最后一段音频序列
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames   删除右侧新增的n_padding_frames帧  (N=1, L - n_padding_frames, motion_feat_dim=73)
            coef_list.append(motion_coef)   # len = 3
            motion_coef = torch.cat(coef_list, dim=1)                # (1, n_frames视频总帧数, n_features=73) - 整个运动序列的运动系数      n_frames = n_motions * n_subdivision
            # motion_coef = self.reformat_motion(args, motion_coef)

        motion_coef = motion_coef.squeeze() #.cpu().numpy().astype(np.float32)     # 去除张量中所有尺寸为1的维度。(n_frames, n_features=70)
        motion_list = []              # 运动列表
        for idx in track(range(motion_coef.shape[0]), description='🚀Generating Motion Sequence...', total=motion_coef.shape[0]):    # 总帧数
            # 按照模板字典中的标准差和均值进行反归一化（与 exp 一致：正态分布归一化的逆变换）
            exp = motion_coef[idx][:63].cpu() * self.templete_dict["std_exp"] + self.templete_dict["mean_exp"]    # [63]
            scale = motion_coef[idx][63:64].cpu() * self.templete_dict["std_scale"] + self.templete_dict["mean_scale"]   # [1]
            t = motion_coef[idx][64:67].cpu() * self.templete_dict["std_t"] + self.templete_dict["mean_t"]    # [3]
            pitch = motion_coef[idx][67:68].cpu() * self.templete_dict["std_pitch"] + self.templete_dict["mean_pitch"]   # [1]
            yaw = motion_coef[idx][68:69].cpu() * self.templete_dict["std_yaw"] + self.templete_dict["mean_yaw"]   # [1]
            roll = motion_coef[idx][69:70].cpu() * self.templete_dict["std_roll"] + self.templete_dict["mean_roll"]   # [1]

            R = get_rotation_matrix(pitch, yaw, roll)                    # 旋转矩阵
            R = R.reshape(1, 3, 3).cpu().numpy().astype(np.float32)           # (1, 3, 3)     1代表只有一张图片    3*3的旋转矩阵
            
            exp = exp.reshape(1, 21, 3).cpu().numpy().astype(np.float32)      # (1, 21, 3)    21个表情关键点     
            scale = scale.reshape(1, 1).cpu().numpy().astype(np.float32)      # (1, 1)
            t = t.reshape(1, 3).cpu().numpy().astype(np.float32)              # (1, 3)         x,y,z三个坐标
            pitch = pitch.reshape(1, 1).cpu().numpy().astype(np.float32)      # (1, 1)         旋转对全局（整个头部）有效，因此只有一个值
            yaw = yaw.reshape(1, 1).cpu().numpy().astype(np.float32)          # (1, 1)
            roll = roll.reshape(1, 1).cpu().numpy().astype(np.float32)        # (1, 1)
            
            motion_list.append({"exp": exp, "scale": scale, "R": R, "t": t, "pitch": pitch, "yaw": yaw, "roll": roll})
        tgt_motion = {'n_frames': motion_coef.shape[0], 'output_fps': 25, 'motion': motion_list}    # 帧数。帧率。运动参数列表。

        if args.is_smooth_motion:      # 平滑运动
            tgt_motion = smooth_(tgt_motion, method="ema")      # 使用指数移动平均（EMA）方法对运动进行平滑
        wfp_template = remove_suffix(args.audio) + '_DiT' + suffix     # xxx.pkl
        dump(wfp_template, tgt_motion)          # 保存文件
    
# （处理训练视频）生成输入视频的 运动模版
def make_motion_templete(args, driving_audio, suffix=".pkl", gpu_id=0): 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)       # 让该进程只看到指定 GPU

    wfp_template = remove_suffix(driving_audio) + '_DiT' + suffix    # xxx.pkl
    if os.path.exists(wfp_template):  # 已处理
        return

    # configs
    args.audio = driving_audio    # 视频的绝对路径
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)      # 推理参数

    # ffmpeg
    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)
    if not fast_check_ffmpeg():
        raise ImportError( "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html")

    try:
        # feature_extract
        motion_extractor = DiTMotionExtractor(
            inference_cfg=inference_cfg
        )
        motion_extractor.gen_motion_sequence(args, suffix=suffix)
    except Exception as e:
        print(f"Exception in motion extractor: {e}")
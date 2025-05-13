# coding: utf-8

"""
config dataclass used for inference  用于推理的配置数据类

推理时使用该文件的参数。如果与argument_config中的相同参数有冲突，以该文件的为主。
"""

import cv2
from numpy import ndarray
import pickle as pkl
from dataclasses import dataclass, field
from typing import Literal, Tuple
from .base_config import PrintableConfig, make_abs_path
import tyro
from typing_extensions import Annotated

def load_lip_array():
    with open(make_abs_path('../utils/resources/lip_array.pkl'), 'rb') as f:
        return pkl.load(f)

@dataclass(repr=False)  # use repr from PrintableConfig
class InferenceConfig(PrintableConfig):
    # MOTION SEQUENCE GENERATOR           运动序列生成器（音频toMotion） 相关权重

    checkpoint_MotionGenerator: str = make_abs_path("../../pretrained_weights/ADEF/motion_generator/iter_0100000.pt")
    checkpoint_AudioEncoder: str = make_abs_path("../../pretrained_weights/hubert-base-ls960/")
    motion_template_path: str = make_abs_path("../../pretrained_weights/ADEF/motion_template/front_all_motion_template.pkl")

    # HUMAN MODEL CONFIG, NOT EXPORTED PARAMS   人体模型配置，未导出参数      相关权重
    models_config: str = make_abs_path('./models.yaml')  # portrait animation config  肖像动画配置
    checkpoint_F: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth')  # path to checkpoint of F
    checkpoint_M: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/motion_extractor.pth')  # path to checkpoint pf M
    checkpoint_G: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/spade_generator.pth')  # path to checkpoint of G
    checkpoint_W: str = make_abs_path('../../pretrained_weights/liveportrait/base_models/warping_module.pth')  # path to checkpoint of W
    checkpoint_S: str = make_abs_path('../../pretrained_weights/liveportrait/retargeting_models/stitching_retargeting_module.pth')  # path to checkpoint to S and R_eyes, R_lip

    emotype: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "angry"    # emotion type, only for human animation, choose from ["happy", "sad", "angry", "fear", "disgusted", "surprised", "neutral", "contempt"]

    # EXPORTED PARAMS  导出参数
    use_emo_enhancer: bool = False
    enhance_level: int = 1  # emotion enhancer level, 0,1,2

    use_emo_analyzer: bool = True  # whether to use emotion analyzer

    flag_use_half_precision: bool = True      # 半精度（不必理会）
    flag_crop_driving_video: bool = False     # 裁剪驱动视频（无用）
    device_id: int = 1                                                  # GPU设备id
    flag_normalize_lip: bool = True                                     # 标准化嘴唇
    flag_source_video_eye_retargeting: bool = False  # 源视频眼睛重定向（无用）
    flag_eye_retargeting: bool = False                                  # 眼睛重定向
    flag_lip_retargeting: bool = False                                  # 嘴唇重定向
    flag_stitching: bool = True                                         # 缝合
    flag_relative_motion: bool = True                                   # 相对运动（相对鼻子转动）
    flag_pasteback: bool = True                                         # 粘贴回原空间
    flag_do_crop: bool = True                                           # 执行裁剪
    flag_do_rot: bool = True                                            # 执行旋转
    flag_force_cpu: bool = False                                        # 强制使用CPU
    flag_do_torch_compile: bool = False     # torch编译？不必理会
    driving_option: str = "pose-friendly"                               # "expression-friendly" or "pose-friendly"
    driving_multiplier: float = 1.0
    driving_smooth_observation_variance: float = 3e-7 # smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy
    source_max_dim: int = 1280 # the max dim of height and width of source image or video
    source_division: int = 2 # make sure the height and width of source image or video can be divided by this number
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all" # the region where the animation was performed, "exp" means the expression, "pose" means the head pose

    # NOT EXPORTED PARAMS
    lip_normalize_threshold: float = 0.03 # threshold for flag_normalize_lip
    source_video_eye_retargeting_threshold: float = 0.18 # threshold for eyes retargeting if the input is a source video
    anchor_frame: int = 0 # TO IMPLEMENT

    input_shape: Tuple[int, int] = (256, 256)  # input shape
    output_format: Literal['mp4', 'gif'] = 'mp4'  # output video format
    crf: int = 15  # crf for output video
    output_fps: int = 25 # default output fps

    mask_crop: ndarray = field(default_factory=lambda: cv2.imread(make_abs_path('../utils/resources/mask_template.png'), cv2.IMREAD_COLOR))
    lip_array: ndarray = field(default_factory=load_lip_array)
    size_gif: int = 256 # default gif size, TO IMPLEMENT

# coding: utf-8

"""
All configs for user
用户的所有配置
"""
from dataclasses import dataclass
import tyro
from typing_extensions import Annotated
from typing import Optional, Literal, Union
from .base_config import PrintableConfig, make_abs_path

@dataclass(repr=False)  # use repr from PrintableConfig
class ArgumentConfig(PrintableConfig):
    animation_mode: str = "human" # human

    ########## input arguments 输入参数 ##########     参考图像路径。驱动音频路径。文件生成目录。
    reference: Annotated[str, tyro.conf.arg(aliases=["-r"])] = make_abs_path('../../assets/examples/imgs/joyvasa_001.png')  # path to the source portrait (human/animal) or video (human)
    audio:  Annotated[str, tyro.conf.arg(aliases=["-a"])] = make_abs_path('../../assets/examples/audios/joyvasa_001.wav')  # path to driving video or template (.pkl format)
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = make_abs_path('../../new_animations/')  # directory to save output video

    emotype: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "angry"    # emotion type, only for human animation, choose from ["happy", "sad", "angry", "fear", "disgusted", "surprised", "neutral", "contempt"]

    use_emo_enhancer: Literal[True, False] = False
    enhance_level: int = 1  # emotion enhancer level, 1,2,3

    use_emo_analyzer: Literal[True, False] = True  # whether to use emotion analyzer

    save_results: Literal[True, False] = False  # 是否保存DiT的生成结果

    ########## inference arguments 推理参数 ##########
    flag_use_half_precision: bool = False  # whether to use half precision (FP16). If black boxes appear, it might be due to GPU incompatibility; set to False.
    device_id: int = 1  # gpu device id
    flag_force_cpu: bool = False  # force cpu inference, WIP!
    flag_normalize_lip: bool = False  # False,  whether to let the lip to close state before animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False

    flag_source_video_eye_retargeting: bool = False  # when the input is a source video, whether to let the eye-open scalar of each frame to be the same as the first source frame before the animation, only take effect when flag_eye_retargeting and flag_lip_retargeting is False, may cause the inter-frame jittering
    flag_eye_retargeting: bool = False  # not recommend to be True, WIP; whether to transfer the eyes-open ratio of each driving frame to the source image or the corresponding source frame
    flag_lip_retargeting: bool = False  # not recommend to be True, WIP; whether to transfer the lip-open ratio of each driving frame to the source image or the corresponding source frame
    flag_stitching: bool = False  # recommend to True if head movement is small, False if head movement is large or the source image is an animal

    flag_relative_motion: bool = False # True, whether to use relative motion
    flag_pasteback: bool = False  # whether to paste-back/stitch the animated face cropping from the face-cropping space to the original image space

    driving_option: Literal["expression-friendly", "pose-friendly"] = "expression-friendly" # "expression-friendly" or "pose-friendly"; "expression-friendly" would adapt the driving motion with the global multiplier, and could be used when the source is a human image
    driving_multiplier: float = 1.10 #default 1.0, be used only when driving_option is "expression-friendly"

    driving_smooth_observation_variance: float = 3e-7  # 3e-7, smooth strength scalar for the animated video when the input is a source video, the larger the number, the smoother the animated video; too much smoothness would result in loss of motion accuracy
    audio_priority: Literal['source', 'driving'] = 'driving'  # whether to use the audio from source or driving video
    animation_region: Literal["exp", "pose", "lip", "eyes", "all"] = "all" # the region where the animation was performed, "exp" means the expression, "pose" means the head pose, "all" means all regions

    ########## source crop arguments 源裁剪参数 ##########
    flag_do_crop: bool = False  # whether to crop the source portrait or video to the face-cropping space
    det_thresh: float = 0.15 # detection threshold
    scale: float = 3.0  # the ratio of face area is smaller if scale is larger
    vx_ratio: float = 0  # the ratio to move the face to left or right in cropping space
    vy_ratio: float = -0.125  # the ratio to move the face to up or down in cropping space
    flag_do_rot: bool = True  # whether to conduct the rotation when flag_do_crop is True
    source_max_dim: int = 1280 # the max dim of height and width of source image or video, you can change it to a larger number, e.g., 1920
    source_division: int = 2 # make sure the height and width of source image or video can be divided by this number

    ########## driving crop arguments 驱动裁剪参数 ##########
    flag_crop_driving_video: bool = False  # whether to crop the driving video, if the given driving info is a video
    scale_crop_driving_video: float = 2.0  # scale factor for cropping driving video
    vx_ratio_crop_driving_video: float = 0.  # adjust y offset
    vy_ratio_crop_driving_video: float = -0.1  # adjust x offset

    ########## moiton generator 运动生成器 #############
    cfg_mode: str = "incremental" # "incremental", "independent"
    cfg_cond = None
    cfg_scale: Union[float, list[float]] = 2.8
    is_smooth_motion: bool = True

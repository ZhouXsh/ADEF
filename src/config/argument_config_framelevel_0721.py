# coding: utf-8

"""独立的 frame-level 推理参数配置。

由 argument_config.py 复制并增加 emotion2vec frame-level 推理参数，
原始配置文件保持不变。
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

import tyro
from typing_extensions import Annotated

from .base_config import PrintableConfig, make_abs_path


@dataclass(repr=False)
class ArgumentConfigFrameLevel0721(PrintableConfig):
    animation_mode: str = "human"

    # input
    reference: Annotated[str, tyro.conf.arg(aliases=["-r"])] = make_abs_path(
        '../../assets/examples/imgs/joyvasa_001.png'
    )
    audio: Annotated[str, tyro.conf.arg(aliases=["-a"])] = make_abs_path(
        '../../assets/examples/audios/joyvasa_001.wav'
    )
    output_dir: Annotated[str, tyro.conf.arg(aliases=["-o"])] = make_abs_path(
        '../../new_animations/'
    )
    emotype: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "angry"

    use_emo_enhancer: Literal[True, False] = False
    enhance_level: int = 1
    use_emo_analyzer: Literal[True, False] = False
    save_results: Literal[True, False] = False

    # frame-level model checkpoint
    checkpoint_MotionGenerator: str = make_abs_path(
        '../../experiments/emo_dit/'
        '20260721_emotion_dit_Unification_framelevel/'
        'checkpoints/iter_0100000.pt'
    )

    # emotion2vec frame-level condition
    # 优先读取显式 npy；为 None 时先查找 audio 同目录下的 frame/<name>.npy，
    # 若仍不存在，则通过 FunASR 在线提取。
    emotion2vec_frame_path: Optional[str] = None
    emotion2vec_model: str = 'iic/emotion2vec_plus_large'
    emotion2vec_hub: Literal['ms', 'modelscope', 'hf', 'huggingface'] = 'ms'
    emotion2vec_output_dir: Optional[str] = None
    emotion2vec_dim: int = 1024

    # inference
    flag_use_half_precision: bool = False
    device_id: int = 1
    flag_force_cpu: bool = False
    flag_normalize_lip: bool = False
    flag_source_video_eye_retargeting: bool = False
    flag_eye_retargeting: bool = False
    flag_lip_retargeting: bool = False
    flag_stitching: bool = False
    flag_relative_motion: bool = False
    flag_pasteback: bool = False
    driving_option: Literal['expression-friendly', 'pose-friendly'] = 'expression-friendly'
    driving_multiplier: float = 1.10
    driving_smooth_observation_variance: float = 3e-7
    audio_priority: Literal['source', 'driving'] = 'driving'
    animation_region: Literal['exp', 'pose', 'lip', 'eyes', 'all'] = 'all'

    # source crop
    flag_do_crop: bool = False
    det_thresh: float = 0.15
    scale: float = 3.0
    vx_ratio: float = 0
    vy_ratio: float = -0.125
    flag_do_rot: bool = True
    source_max_dim: int = 1280
    source_division: int = 2

    # driving crop
    flag_crop_driving_video: bool = False
    scale_crop_driving_video: float = 2.0
    vx_ratio_crop_driving_video: float = 0.0
    vy_ratio_crop_driving_video: float = -0.1

    # motion generator
    cfg_mode: str = 'incremental'
    cfg_cond = None
    cfg_scale: Union[float, list[float]] = 2.8
    is_smooth_motion: bool = True


# 保留与原入口相似的名称，便于阅读。
ArgumentConfig = ArgumentConfigFrameLevel0721

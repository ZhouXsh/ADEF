# coding: utf-8

"""
Independent utility helpers for the 1pad-canonical-token pipeline.

This file is a verbatim copy of `src/utils/helper.py` with one change:
the `motion_generator` branch in `load_model` builds the 1pad
`DitTalkingHead` (from `emotion_dit_Unification_1pad_0721`) instead of the
no-CFG one. Everything else (utility functions, F / M / W / G / S-R
loading) is intentionally identical so the 1pad pipeline does not depend
on the original `helper.py` at all.
"""
import os
import os.path as osp
import torch
from collections import OrderedDict
import numpy as np
from scipy.spatial import ConvexHull  # pylint: disable=E0401,E0611
from typing import Union
import cv2

from ..modules.spade_generator import SPADEDecoder
from ..modules.warping_network import WarpingNetwork
from ..modules.motion_extractor import MotionExtractor
from ..modules.appearance_feature_extractor import AppearanceFeatureExtractor
from ..modules.stitching_retargeting_network import StitchingRetargetingNetwork

from ..modules.emotion_dit_Unification_1pad_0721 import DitTalkingHead


class NullableArgs:
    def __init__(self, namespace):
        for key, value in namespace.__dict__.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        # when an attribute lookup has not found the attribute
        if key == 'align_mask_width':
            if 'use_alignment_mask' in self.__dict__:
                return 1 if self.use_alignment_mask else 0
            else:
                return 0
        if key == 'no_head_pose':
            return not self.predict_head_pose
        if key == 'no_use_learnable_pe':
            return not self.use_learnable_pe

        return None


def _get_arg(args, name, default):
    value = getattr(args, name)
    return default if value is None else value


def tensor_to_numpy(data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """transform torch.Tensor into numpy.ndarray"""
    if isinstance(data, torch.Tensor):
        return data.data.cpu().numpy()
    return data


def calc_motion_multiplier(
    kp_source: Union[np.ndarray, torch.Tensor],
    kp_driving_initial: Union[np.ndarray, torch.Tensor]
) -> float:
    """calculate motion_multiplier based on the source image and the first driving frame  基于源图像和第一驱动帧计算运动倍增器"""
    """这里存在一个问题，如果驱动图像张嘴了，效果比较差。"""
    kp_source_np = tensor_to_numpy(kp_source)
    kp_driving_initial_np = tensor_to_numpy(kp_driving_initial)

    source_area = ConvexHull(kp_source_np.squeeze(0)).volume
    driving_area = ConvexHull(kp_driving_initial_np.squeeze(0)).volume
    motion_multiplier = np.sqrt(source_area) / np.sqrt(driving_area)
    # motion_multiplier = np.cbrt(source_area) / np.cbrt(driving_area)

    return motion_multiplier


def suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind(".")
    if pos == -1:
        return ""
    return filename[pos + 1:]


def prefix(filename):
    """a.jpg -> a"""
    pos = filename.rfind(".")
    if pos == -1:
        return filename
    return filename[:pos]


def basename(filename):
    """a/b/c.jpg -> c"""
    return prefix(osp.basename(filename))


def remove_suffix(filepath):
    """a/b/c.jpg -> a/b/c"""
    return osp.join(osp.dirname(filepath), basename(filepath))


def is_image(file_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    return file_path.lower().endswith(image_extensions)


def is_video(file_path):
    '''
    检查是否.mp4", ".mov", ".avi", ".webm视频；或者图像的目录
    '''
    if file_path.lower().endswith((".mp4", ".mov", ".avi", ".webm")) or osp.isdir(file_path):
        return True
    return False


def is_template(file_path):
    '''
    返回文件是否以.pkl结尾
    '''
    if file_path.endswith(".pkl"):
        return True
    return False


def mkdir(d, log=False):
    # return self-assined `d`, for one line code
    if not osp.exists(d):
        os.makedirs(d, exist_ok=True)
        if log:
            print(f"Make dir: {d}")
    return d


def squeeze_tensor_to_numpy(tensor):
    out = tensor.data.squeeze(0).cpu().numpy()
    return out


def dct2device(dct: dict, device):
    for key in dct:
        if isinstance(dct[key], torch.Tensor):
            dct[key] = dct[key].to(device)
        else:
            dct[key] = torch.tensor(dct[key]).to(device)
    return dct


def concat_feat(kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
    """
    kp_source: (bs, k, 3)
    kp_driving: (bs, k, 3)
    Return: (bs, 2k*3)
    """
    bs_src = kp_source.shape[0]
    bs_dri = kp_driving.shape[0]
    assert bs_src == bs_dri, 'batch size must be equal'

    feat = torch.cat([kp_source.view(bs_src, -1), kp_driving.view(bs_dri, -1)], dim=1)
    return feat


def remove_ddp_dumplicate_key(state_dict):
    state_dict_new = OrderedDict()
    for key in state_dict.keys():
        state_dict_new[key.replace('module.', '')] = state_dict[key]
    return state_dict_new


def load_model(ckpt_path, model_config, device, model_type):
    model_params = model_config['model_params'][f'{model_type}_params']

    if model_type == 'appearance_feature_extractor':         # 外观特征提取器 F
        model = AppearanceFeatureExtractor(**model_params).to(device)
    elif model_type == 'motion_extractor':                    # 运动提取器 M
        model = MotionExtractor(**model_params).to(device)
    elif model_type == 'motion_generator':                    # 运动生成器 (1pad)
        model_data = torch.load(ckpt_path, map_location=device)
        model_args = NullableArgs(model_data['args'])
        # 1pad 版本的 DitTalkingHead：序列最前面多 1 个 canonical keypoint token
        model = DitTalkingHead(
            device=device,
            target=_get_arg(model_args, "target", "sample"),
            architecture=_get_arg(model_args, "architecture", "decoder"),
            motion_feat_dim=_get_arg(model_args, "motion_feat_dim", 70),
            fps=_get_arg(model_args, "fps", 25),
            n_motions=_get_arg(model_args, "n_motions", 100),
            n_prev_motions=_get_arg(model_args, "n_prev_motions", 25),
            feature_dim=_get_arg(model_args, "feature_dim", 512),
            audio_model=_get_arg(model_args, "audio_model", "hubert"),
            n_diff_steps=_get_arg(model_args, "n_diff_steps", 50),
            diff_schedule=_get_arg(model_args, "diff_schedule", "cosine"),
            cfg_mode=_get_arg(model_args, "cfg_mode", "incremental"),
            guiding_conditions=_get_arg(
                model_args, "guiding_conditions", "audio,emotion"
            ),
            align_mask_width=_get_arg(model_args, "align_mask_width", 1),
            no_use_learnable_pe=_get_arg(
                model_args, "no_use_learnable_pe", False
            ),
            use_indicator=_get_arg(model_args, "use_indicator", True),
            n_heads=_get_arg(model_args, "n_heads", 8),
            n_layers=_get_arg(model_args, "n_layers", 6),
            mlp_ratio=_get_arg(model_args, "mlp_ratio", 4),
        )

        # 过滤掉 shape 不匹配的参数（与训练时可能多出 / 少出的 buffer 对齐）
        state_dict = dict(model_data["model"])
        model_state = model.state_dict()
        for key in list(state_dict.keys()):
            if key not in model_state or state_dict[key].shape != model_state[key].shape:
                state_dict.pop(key)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, model_args
    elif model_type == 'warping_module':                       # 扭曲模块 W
        model = WarpingNetwork(**model_params).to(device)
    elif model_type == 'spade_generator':                      # 图像生成器 G
        model = SPADEDecoder(**model_params).to(device)
    elif model_type == 'stitching_retargeting_module':         # 缝合模块 和 重定向模块 S and R
        # Special handling for stitching and retargeting module
        config = model_config['model_params']['stitching_retargeting_module_params']
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)

        stitcher = StitchingRetargetingNetwork(**config.get('stitching'))
        stitcher.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_shoulder']))
        stitcher = stitcher.to(device)
        stitcher.eval()

        retargetor_lip = StitchingRetargetingNetwork(**config.get('lip'))
        retargetor_lip.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_mouth']))
        retargetor_lip = retargetor_lip.to(device)
        retargetor_lip.eval()

        retargetor_eye = StitchingRetargetingNetwork(**config.get('eye'))
        retargetor_eye.load_state_dict(remove_ddp_dumplicate_key(checkpoint['retarget_eye']))
        retargetor_eye = retargetor_eye.to(device)
        retargetor_eye.eval()

        return {
            'stitching': stitcher,
            'lip': retargetor_lip,
            'eye': retargetor_eye
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    model.eval()
    return model


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def is_square_video(video_path):
    '''
    返回视频是否正方形尺寸的
    '''
    video = cv2.VideoCapture(video_path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()
    # if width != height:
        # log.info("Uploaded video is not square, force do crop (driving) to be True")

    return width == height


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


__all__ = [
    "NullableArgs",
    "tensor_to_numpy",
    "calc_motion_multiplier",
    "suffix",
    "prefix",
    "basename",
    "remove_suffix",
    "is_image",
    "is_video",
    "is_template",
    "mkdir",
    "squeeze_tensor_to_numpy",
    "dct2device",
    "concat_feat",
    "remove_ddp_dumplicate_key",
    "load_model",
    "load_description",
    "is_square_video",
    "clean_state_dict",
]

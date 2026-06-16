# coding: utf-8

import os.path as osp
import imageio
import numpy as np
import pickle
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

from .helper import mkdir, suffix

# 加载RGB图像
def load_image_rgb(image_path: str):
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_video(video_info, n_frames=-1):
    '''
    输入视频转换为逐帧的 RGB 图像列表
    '''
    reader = imageio.get_reader(video_info, "ffmpeg")

    ret = []
    for idx, frame_rgb in enumerate(reader):
        if n_frames > 0 and idx >= n_frames:
            break
        ret.append(frame_rgb)

    reader.close()
    return ret


def contiguous(obj):
    if not obj.flags.c_contiguous:
        obj = obj.copy(order="C")
    return obj

# 调整图像的大小，使最大尺寸不超过max_dim，图像的宽度和高度是n的倍数。
def resize_to_limit(img: np.ndarray, max_dim=1920, division=2):
    """
    调整图像的大小，使最大尺寸不超过max_dim，图像的宽度和高度是n的倍数。
    ajust the size of the image so that the maximum dimension does not exceed max_dim, and the width and the height of the image are multiples of n.
    :param img: the image to be processed.
    :param max_dim: the maximum dimension constraint.
    :param n: the number that needs to be multiples of.
    :return: the adjusted image.
    """
    h, w = img.shape[:2]

    # ajust the size of the image according to the maximum dimension
    if max_dim > 0 and max(h, w) > max_dim:
        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))
        img = cv2.resize(img, (new_w, new_h))

    # ensure that the image dimensions are multiples of n
    division = max(division, 1)
    new_h = img.shape[0] - (img.shape[0] % division)
    new_w = img.shape[1] - (img.shape[1] % division)

    if new_h == 0 or new_w == 0:
        # when the width or height is less than n, no need to process
        return img

    if new_h != img.shape[0] or new_w != img.shape[1]:
        img = img[:new_h, :new_w]

    return img

# 无用
def load(fp):
    suffix_ = suffix(fp)

    if suffix_ == "npy":
        return np.load(fp)
    elif suffix_ == "pkl":
        return pickle.load(open(fp, "rb"))
    else:
        raise Exception(f"Unknown type: {suffix}")


def dump(wfp, obj):
    '''
    保存文件
    '''
    wd = osp.split(wfp)[0]
    if wd != "" and not osp.exists(wd):
        mkdir(wd)

    _suffix = suffix(wfp)
    if _suffix == "npy":
        np.save(wfp, obj)
    elif _suffix == "pkl":
        pickle.dump(obj, open(wfp, "wb"))
    else:
        raise Exception("Unknown type: {}".format(_suffix))

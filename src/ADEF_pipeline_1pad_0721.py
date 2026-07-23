# coding: utf-8

"""
Independent ADEF pipeline for the 1pad-canonical-token inference.

This is a verbatim copy of `src/ADEF_pipeline.py` with the `ADEFWrapper`
import redirected to `ADEF_wrapper_1pad_0721.ADEFWrapper`. No inheritance
from `ADEFPipeline`, so the 1pad pipeline does not depend on the original
`ADEF_pipeline.py` at all.
"""

import numpy as np
from src.modules.audio2emotion import Audio2EmotionModel
import torch

torch.backends.cudnn.benchmark = True

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)
import os
import os.path as osp

from rich.progress import track

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix
from .utils.video import images2video, add_audio_to_video
from .utils.crop import prepare_paste_back, paste_back
from .utils.io import load_image_rgb, resize_to_limit, load, dump
from .utils.helper_1pad_0721 import (
    mkdir,
    basename,
    dct2device,
    is_image,
    calc_motion_multiplier,
    remove_suffix,
)
from .utils.rprint import rlog as log
from .ADEF_wrapper_1pad_0721 import ADEFWrapper

############       这个写在config里面应该更好
from src.config.emotion_config import global_emo_list
emo_list = global_emo_list


def make_abs_path(fn):      # 生成绝对路径
    return osp.join(osp.dirname(osp.realpath(__file__)), fn)


from scipy.optimize import curve_fit


def smooth_driving_template_cosine(left_frame, right_frame, driving_template_dct):
    """
    对指定帧区间内的 driving_template_dct 进行余弦插值和平滑拟合。

    参数:
        left_frame (int): 区间起始帧索引
        right_frame (int): 区间结束帧索引
        driving_template_dct (dict): 存储关键帧参数的字典，每个键为帧编号

    修改:
        该函数将直接修改 driving_template_dct 中 left_frame 到 right_frame 区间内的数据。
    """

    # ------------------ scale, t, R 进行余弦插值 ------------------
    for i in range(left_frame, right_frame):
        t = (i - left_frame) / (right_frame - left_frame)
        cos_val = np.cos(np.pi * t)  # t ∈ [0,1] 上的余弦值

        for key in ['scale', 't', 'R']:
            left_val = driving_template_dct[left_frame][key]
            right_val = driving_template_dct[right_frame][key]
            A = 0.5 * (left_val - right_val)
            C = 0.5 * (left_val + right_val)
            driving_template_dct[i][key] = A * cos_val + C

    # ------------------ 表达参数 exp 的余弦拟合 ------------------

    def cosine_func(x, A, B, C, D):
        return A * np.cos(B * x + C) + D

    def estimate_initial_params(x, y):
        D = np.mean(y)
        A = (np.max(y) - np.min(y)) / 2
        T = x[-1] - x[0]
        B = 2 * np.pi / T if T != 0 else 1.0
        return [A, B, 0, D]

    x_data = np.arange(left_frame, right_frame + 1)
    y_all = np.array([
        driving_template_dct[i]['exp'].reshape(-1)
        for i in x_data
    ])  # shape: (N, 63)

    y_fit = np.zeros_like(y_all)

    for kp in range(63):
        y_kp = y_all[:, kp]
        try:
            p0 = estimate_initial_params(x_data, y_kp)
            params, _ = curve_fit(cosine_func, x_data, y_kp, p0=p0, maxfev=10000)
            y_fit[:, kp] = cosine_func(x_data, *params)
        except RuntimeError:
            y_fit[:, kp] = y_kp  # 拟合失败则保留原始值

    for i, frame_id in enumerate(x_data):
        driving_template_dct[frame_id]['exp'] = y_fit[i].reshape(1, 21, 3)


class ADEFPipeline(object):
    def __init__(self, inference_cfg: InferenceConfig, crop_cfg: CropConfig):
        self.adef_wrapper: ADEFWrapper = ADEFWrapper(inference_cfg=inference_cfg)       # 核心功能包装器
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)                        # 裁剪器
        # 1pad: 1pad wrapper 在 gen_motion_sequence 里需要 cropper 来抽 canonical kp
        self.adef_wrapper.cropper = self.cropper

    def execute(self, args: ArgumentConfig, ):
        inf_cfg = self.adef_wrapper.inference_cfg          # 推理配置
        device = self.adef_wrapper.device          # cuda:0
        crop_cfg = self.cropper.crop_cfg                           # 裁剪器参数

######## load reference image  加载参考图像########
        if is_image(args.reference):    # 参考图像
            img_rgb = load_image_rgb(args.reference)       # (h,w,3)   (336,336,3)
            # 调整图像的大小，使最大尺寸不超过max_dim，图像的宽度和高度是n（division）的倍数。
            img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)       # (h,w,3)   (336,336,3)
            log(f"Load reference image from {args.reference}")
            source_rgb_lst = [img_rgb]    # 源rgb图像 列表，source_rgb_lst[0]就是img_rgb    len==1
        else:
            raise Exception(f"Unknown reference image format: {args.reference}")

        if inf_cfg.use_emo_analyzer:     # 情感分析
            np_path = args.audio[:-4] + '.npy'
            if not os.path.exists(np_path):      # 从npy模版加载
                from funasr import AutoModel
                model = AutoModel(
                    model="iic/emotion2vec_plus_large",
                    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
                )
                model.generate(args.audio, output_dir=os.path.dirname(args.audio), granularity="utterance", extract_embedding=True)
            emo_vec = torch.tensor(np.load(np_path)).unsqueeze(0).to(device)  # [1, 1024]

            a2e_model = Audio2EmotionModel().to(device)
            dict = torch.load('/mnt/disk2/zhouxishi/ADEF/pretrained_weights/ADEF/audio2emo/audio2emo.pth', map_location=device)
            a2e_model.load_state_dict(dict)
            a2e_model.eval()

            outputs = a2e_model(emo_vec)
            argmax = outputs.argmax(dim=1).item()
            args.emotype = emo_list[argmax]
            print(f"Emotion: {args.emotype}")

        if args.save_results:
        #  从模版中读取motion  （用于验证，加速驱动）
            wfp_template = remove_suffix(args.audio) + '.pkl'     # 保存到指定目录
            if os.path.exists(wfp_template):      # 从pkl模版加载
                log(f"Load from template: {wfp_template}.", style='bold green')
                driving_template_dct = load(wfp_template)
            else:
                # generate motion sequence 根据音频生成运动序列
                driving_template_dct = self.adef_wrapper.gen_motion_sequence(args)   # {'n_frames':XX, 'output_fps':XX, 'motions':[{exp,t,...},{},{},...,{}共n_frames个字典]}
                dump(wfp_template, driving_template_dct)
                log(f"Dump motion template to {wfp_template}")
        else:
            driving_template_dct = self.adef_wrapper.gen_motion_sequence(args)
        n_frames = driving_template_dct['n_frames']                        # （音频对应的）总帧数
        print(n_frames)

        if args.use_emo_enhancer:     # 情感增强
            for window in range(99, n_frames, 100):   # 平滑窗口
                left_frame = window - 10
                right_frame = window + 10
                if left_frame < 0 or right_frame >= n_frames:
                    continue
                smooth_driving_template_cosine(left_frame, right_frame, driving_template_dct['motion'])

######## prepare for pasteback 将被裁剪的图像粘贴回原图像位置 ########
        I_p_pstbk_lst = None                              # 生成的视频的图像列表，并且粘贴回完整的原图空间
        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:   # 需要粘贴回；需要被裁剪；需要被缝合
            I_p_pstbk_lst = []                            # 用于存储经过 pasteback 处理后的图像帧
            log("Prepared pasteback mask done.")
        I_p_lst = []                                      # 生成的视频的图像列表，还没有粘贴到原图空间
        R_d_0, x_d_0_info = None, None             # 第一帧的旋转矩阵（R_d_0）和相应的运动信息（x_d_0_info） ？？？？？？？？
        flag_normalize_lip = inf_cfg.flag_normalize_lip  # not overwrite          False                      # 进行嘴唇的标准化处理（False）
        flag_source_video_eye_retargeting = inf_cfg.flag_source_video_eye_retargeting  # not overwrite       # 是否启用原视频眼睛重定向（False）
        lip_delta_before_animation, eye_delta_before_animation = None, None      # 动画前的嘴唇和眼睛位移差异

######## process source info 处理源参考图像相关的信息 ########
        if inf_cfg.flag_do_crop:     # 参考图像需要被裁剪
            crop_info = self.cropper.crop_source_image(source_rgb_lst[0], crop_cfg)     # 返回landmark等信息
            if crop_info is None:
                raise Exception("No face detected in the source image!")
            source_lmk = crop_info['lmk_crop']                             # 图像的landmark
            img_crop_256x256 = crop_info['img_crop_256x256']               # 裁剪成256x256后的图像
        else:           # 参考图像本就是裁剪过的
            source_lmk = self.cropper.calc_lmk_from_cropped_image(source_rgb_lst[0])    # 图像的landmark   # (203, 2)
            img_crop_256x256 = cv2.resize(source_rgb_lst[0], (256, 256))  # force to resize to 256x256  强制调整大小为256x256   （256,256,3）
        I_s = self.adef_wrapper.prepare_source(img_crop_256x256)         # 处理后的图像  H x W x 3  ->  B x 3 x H x W
        x_s_info = self.adef_wrapper.get_kp_info(I_s)                    # 参考图像的计算隐式关键点时相关的信息
        x_c_s = x_s_info['kp']                                                    # 源图像的规范关键点Xc,s      B x 21 x 3
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])    # 旋转矩阵R         (B, 3, 3)
        f_s = self.adef_wrapper.extract_feature_3d(I_s)                  # 参考图像的外观特征（预训练的F）   [1, 32, 16, 64, 64]

        x_s = self.adef_wrapper.transform_keypoint(x_s_info)             # 計算源隐式关键点Xs,k       (bs, k, 3)  B x 21 x 3

        # let lip-open scalar to be 0 at first  首先让唇张开标量为0
        if flag_normalize_lip and inf_cfg.flag_relative_motion and source_lmk is not None:               # √
            c_d_lip_before_animation = [0.]       # 唇部张开度，初始值设置为零
            combined_lip_ratio_tensor_before_animation = self.adef_wrapper.calc_combined_lip_ratio(c_d_lip_before_animation, source_lmk)   # 计算最终的唇部张开度比例
            if combined_lip_ratio_tensor_before_animation[0][0] >= inf_cfg.lip_normalize_threshold:  # 唇部张开度大于等于某个阈值：进行后续的唇部调整操作
                lip_delta_before_animation = self.adef_wrapper.retarget_lip(x_s, combined_lip_ratio_tensor_before_animation)   # 重定向嘴唇

        if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:         # 需要粘贴回；需要被裁剪；需要被缝合  False
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(source_rgb_lst[0].shape[1], source_rgb_lst[0].shape[0]))

######## animate 动画化（逐帧处理音频）########
        for i in track(range(n_frames), description='🚀Animating Image with Generated Motions...', total=n_frames):

            x_d_i_info = driving_template_dct['motion'][i]   # 包括该帧的exp, scale, R, t, pitch, yaw, roll
            x_d_i_info = dct2device(x_d_i_info, device)

            # R  旋转矩阵
            R_d_i = x_d_i_info['R'] if 'R' in x_d_i_info.keys() else x_d_i_info['R_d']  # compatible with previous keys 与以前的键兼容
            if i == 0:  # cache the first frame     缓存第一帧
                R_d_0 = R_d_i                      # 第一帧的旋转矩阵
                x_d_0_info = x_d_i_info.copy()     # 第一帧的exp, scale, R, t, pitch, yaw, roll信息
            if inf_cfg.animation_region == "all" or inf_cfg.animation_region == "pose":       # "exp", "pose", "lip", "eyes", "all"
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s  # Rd_i * Rd_0 * Rs    (B=1, 3, 3)
            else:        # "exp", "lip", "eyes"
                R_new = R_s           # 直接copy参考图像的旋转矩阵 (B=1, 3, 3)

            delta_new = x_d_i_info['exp']

            # scale  缩放
            scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])   # （1,1）

            # translation  头部平移
            t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])    # 原图平移+相对首帧平移    （1,3）
            t_new[..., 2].fill_(0)  # zero tz        z坐标变为0

            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new         # 计算该帧的隐式关键点   （1,21,3）

            if inf_cfg.flag_relative_motion and inf_cfg.driving_option == "expression-friendly":   # 相对运动 & 表情友好型的
                if i == 0:
                    x_d_0_new = x_d_i_new          # 保留第一帧（第一帧不需要倍增）
                    motion_multiplier = calc_motion_multiplier(x_s, x_d_0_new)    # 基于源图像和第一驱动帧计算运动倍增器
                x_d_diff = (x_d_i_new - x_d_0_new) * motion_multiplier     # 相对第一帧的差异（乘以倍增器）
                x_d_i_new = x_d_diff + x_s

            # Algorithm 1 in Liveportrait:
            if not inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:     # 无缝合无重定向
                # without stitching or retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new += lip_delta_before_animation
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
                else:
                    pass
            elif inf_cfg.flag_stitching and not inf_cfg.flag_eye_retargeting and not inf_cfg.flag_lip_retargeting:        # 有缝合但无重定向
                # with stitching and without retargeting
                if flag_normalize_lip and lip_delta_before_animation is not None:
                    x_d_i_new = self.adef_wrapper.stitching(x_s, x_d_i_new) + lip_delta_before_animation
                else:
                    x_d_i_new = self.adef_wrapper.stitching(x_s, x_d_i_new)
                if flag_source_video_eye_retargeting and eye_delta_before_animation is not None:
                    x_d_i_new += eye_delta_before_animation
            else:
                if inf_cfg.flag_relative_motion:  # use x_s
                    x_d_i_new = x_s
                else:  # use x_d,i
                    x_d_i_new = x_d_i_new
                if inf_cfg.flag_stitching:
                    x_d_i_new = self.adef_wrapper.stitching(x_s, x_d_i_new)

            x_d_i_new = x_s + (x_d_i_new - x_s) * inf_cfg.driving_multiplier      # （1,21,3）
            # 扭曲解码（图像生成）：外观特征，源隐式关键点，当前帧驱动隐式关键点
            out = self.adef_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.adef_wrapper.parse_output(out['out'])[0]  # 512x512x3, uint8   生成的图像（可显示的那种0~255）
            if i > 1 and i < n_frames - 1:   # 只保存中间的帧
                I_p_lst.append(I_p_i)         # 图像列表（还没粘贴回完整原图空间）

            if inf_cfg.flag_pasteback and inf_cfg.flag_do_crop and inf_cfg.flag_stitching:     # 需要粘贴回；需要被裁剪；需要被缝合
                # TODO: the paste back procedure is slow, considering optimize it using multi-threading or GPU  回粘贴过程很慢；考虑使用多线程或GPU对其进行优化
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], source_rgb_lst[0], mask_ori_float)    # 将生成的图像粘贴回原空间后 的完整图像
                I_p_pstbk_lst.append(I_p_pstbk)

        # save the animated result       保存结果
        mkdir(args.output_dir)
        temp_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_temp.mp4')
        # 将图片列表保存为视频（没声音）
        if I_p_pstbk_lst is not None and len(I_p_pstbk_lst) > 0:
            print(f'图像数量（帧数）    {len(I_p_pstbk_lst)}')
            images2video(I_p_pstbk_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        else:
            print(f'图像数量（帧数）    {len(I_p_lst)}')
            images2video(I_p_lst, wfp=temp_video, fps=inf_cfg.output_fps)
        final_video = osp.join(args.output_dir, f'{basename(args.reference)}_{basename(args.audio)}_{args.emotype}.mp4')
        add_audio_to_video(temp_video, args.audio, final_video)     # 添加音轨
        return final_video


__all__ = ["ADEFPipeline"]

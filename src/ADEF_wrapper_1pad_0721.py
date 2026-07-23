# coding: utf-8

"""
Independent ADEF wrapper for the 1pad-canonical-token pipeline.

This is a verbatim copy of `src/ADEF_wrapper.py` with the import of
`load_model` / `concat_feat` redirected to `helper_1pad_0721` (so the
motion generator is built with the 1pad-prefixed `DitTalkingHead`),
plus two overrides:
    * `_extract_canonical_token` — derives a canonical keypoint token
      from the reference image and forwards it to the denoiser.
    * `gen_motion_sequence` — calls the override and clears the prior
      afterwards.

No inheritance from `ADEFWrapper`, so the 1pad pipeline does not depend
on the original `ADEF_wrapper.py` at all.
"""

import contextlib
import os.path as osp
import pickle
import numpy as np
import cv2
from src.modules.emotion_enhancer import EmotionTransformer
import torch
import yaml
import math
import librosa
import torch.nn.functional as F
from rich.progress import track

from .utils.timer import Timer
from .utils.helper_1pad_0721 import load_model, concat_feat
from .utils.camera import headpose_pred_to_degree, get_rotation_matrix
from .utils.retargeting_utils import calc_eye_close_ratio, calc_lip_close_ratio
from .config.inference_config import InferenceConfig
from .utils.rprint import rlog as log
from .utils.filter import smooth_
from .utils.io import load_image_rgb, resize_to_limit

from src.config.emotion_config import global_emo_list
emo_list = global_emo_list


class ADEFWrapper(object):
    def __init__(self, inference_cfg: InferenceConfig):

        self.inference_cfg = inference_cfg
        self.device_id = inference_cfg.device_id
        self.compile = inference_cfg.flag_do_torch_compile
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
        # init F   外观特征提取器
        self.appearance_feature_extractor = load_model(inference_cfg.checkpoint_F, model_config, self.device, 'appearance_feature_extractor')
        log(f'Load appearance_feature_extractor from {osp.realpath(inference_cfg.checkpoint_F)} done.')
        # init M    运动提取器
        self.motion_extractor = load_model(inference_cfg.checkpoint_M, model_config, self.device, 'motion_extractor')
        log(f'Load motion_extractor from {osp.realpath(inference_cfg.checkpoint_M)} done.')
        # init W    扭曲模块
        self.warping_module = load_model(inference_cfg.checkpoint_W, model_config, self.device, 'warping_module')
        log(f'Load warping_module from {osp.realpath(inference_cfg.checkpoint_W)} done.')
        # init G      图像生成器
        self.spade_generator = load_model(inference_cfg.checkpoint_G, model_config, self.device, 'spade_generator')
        log(f'Load spade_generator from {osp.realpath(inference_cfg.checkpoint_G)} done.')
        # init S and R     缝合模块 和 重定向模块
        if inference_cfg.checkpoint_S is not None and osp.exists(inference_cfg.checkpoint_S):
            self.stitching_retargeting_module = load_model(inference_cfg.checkpoint_S, model_config, self.device, 'stitching_retargeting_module')
            log(f'Load stitching_retargeting_module from {osp.realpath(inference_cfg.checkpoint_S)} done.')
        else:
            self.stitching_retargeting_module = None
        # Optimize for inference
        if self.compile:
            torch._dynamo.config.suppress_errors = True  # Suppress errors and fall back to eager execution
            self.warping_module = torch.compile(self.warping_module, mode='max-autotune')
            self.spade_generator = torch.compile(self.spade_generator, mode='max-autotune')

        self.model_config = model_config
        self.timer = Timer()

        # Motion Genertor     运动生成器 (1pad)
        self.motion_generator, self.motion_generator_args = load_model(inference_cfg.checkpoint_MotionGenerator, model_config, self.device, 'motion_generator')
        log(f'Load motion_generator from {osp.realpath(inference_cfg.checkpoint_MotionGenerator)} done.')
        self.n_motions = self.motion_generator_args.n_motions                  # 单个音频子序列（片段）的帧数
        self.n_prev_motions = self.motion_generator_args.n_prev_motions
        self.fps = self.motion_generator_args.fps
        self.audio_unit = 16000. / self.fps     # num of samples per frame     每帧样本数
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.motion_generator_args.pad_mode
        self.use_indicator = self.motion_generator_args.use_indicator
        self.template_dict = pickle.load(open(inference_cfg.motion_template_path, 'rb'))

        # #### 0420情感增强
        self.emo_ehance = inference_cfg.use_emo_enhancer
        if self.emo_ehance:
            transf_model = EmotionTransformer().to(self.device)
            enhancer_p = '/mnt/disk2/zhouxishi/ADEF/pretrained_weights/ADEF/emo_enhancer/emo_enhancer.pth'
            transf_model_data = torch.load(enhancer_p, map_location=self.device)
            transf_model.load_state_dict(transf_model_data, strict=False)
            transf_model.eval()
            transf_model.to(self.device)
            self.emo_enhancer = transf_model
            log(f'load enhancer_p from {enhancer_p}')

        # 1pad pipeline 用的前向人脸裁剪器（外部 attach）
        self.cropper = None

    # 获取推理上下文
    def inference_ctx(self):
        if self.device == "mps":
            ctx = contextlib.nullcontext()
        else:
            ctx = torch.autocast(device_type=self.device[:4], dtype=torch.float16,
                                 enabled=self.inference_cfg.flag_use_half_precision)
        return ctx

    # （根据用户修改后）更新配置
    def update_config(self, user_args):
        for k, v in user_args.items():
            if hasattr(self.inference_cfg, k):
                setattr(self.inference_cfg, k, v)

    # 处理源图像    OK
    def prepare_source(self, img: np.ndarray) -> torch.Tensor:
        """ construct the input as standard 将输入构建为标准
        img: H x W x 3, uint8（0~255）, 256x256         （256,256,3）
        输出x: B x 3 x H x W, float32, 256x256        B=1
        """
        h, w = img.shape[:2]
        if h != self.inference_cfg.input_shape[0] or w != self.inference_cfg.input_shape[1]:        # 256,256
            x = cv2.resize(img, (self.inference_cfg.input_shape[0], self.inference_cfg.input_shape[1]))
        else:
            x = img.copy()

        if x.ndim == 3:   # 使用np.newaxis增加维度
            x = x[np.newaxis].astype(np.float32) / 255.  # H x W x 3 -> 1 x H x W x 3, 归一化normalized to 0~1
        elif x.ndim == 4:        # 已经是一个图像批次了
            x = x.astype(np.float32) / 255.  # B x H x W x 3, 归一化normalized to 0~1
        else:
            raise ValueError(f'img ndim should be 3 or 4: {x.ndim}')
        x = np.clip(x, 0, 1)  # clip to 0~1     将图像的所有像素值限制在 [0, 1] 范围内，防止出现归一化后的值超过 1 或小于 0。
        x = torch.from_numpy(x).permute(0, 3, 1, 2)  # 1xHxWx3 -> 1x3xHxW
        x = x.to(self.device)
        return x     # 1 x 3 x H x W

    # 处理视频
    def prepare_videos(self, imgs) -> torch.Tensor:
        """ construct the input as standard 将输入构建为标准Tensor
        imgs: N x B x H x W x 3, uint8
        """
        if isinstance(imgs, list):   # 列表类型
            _imgs = np.array(imgs)[..., np.newaxis]    # T x H x W x 3 -> T x H x W x 3 x 1
        elif isinstance(imgs, np.ndarray):
            _imgs = imgs                               # T x H x W x 3 x 1
        else:
            raise ValueError(f'imgs type error: {type(imgs)}')

        y = _imgs.astype(np.float32) / 255.       # 归一化到 [0,1]
        y = np.clip(y, 0, 1)  # clip to 0~1       # 确保数据范围在 [0,1]，防止异常数据（np.clip 限制最大值 1，最小值 0）
        y = torch.from_numpy(y).permute(0, 4, 3, 1, 2)  # TxHxWx3x1 -> Tx1x3xHxW
        y = y.to(self.device)

        return y    # T x 1 x 3 x H x W   帧数、批次、、、

    # 提取参考图像的外观特征
    def extract_feature_3d(self, x: torch.Tensor) -> torch.Tensor:
        """ get the appearance feature of the image by F   通过F获取图像的外观特征
        x: Bx3xHxW, normalized to 0~1
        """
        with torch.no_grad(), self.inference_ctx():     # 关闭梯度计算
            feature_3d = self.appearance_feature_extractor(x)

        return feature_3d.float()

    # 获取 单张图像（参考图像or视频帧）中 计算隐式关键点相关的信息
    def get_kp_info(self, x: torch.Tensor, **kwargs) -> dict:     # x: (B,3,H=256,W=256)
        """ get the implicit keypoint information
        x: B x 3 x H x W, normalized to 0~1           B：批次（参考图像的个数）  B=1
        flag_refine_info: whether to trandform the pose to degrees and the dimention of the reshape  是否 将姿势转换为度数 和 重塑的尺寸
        return: A dict contains keys: 'pitch', 'yaw', 'roll', 't'（平移）, 'exp'（表情变形）, 'scale'（缩放）, 'kp'（规范关键点）
        """
        with torch.no_grad(), self.inference_ctx():     # 禁用梯度计算，这样在推理过程中不会计算梯度，从而节省内存和加速推理。
            kp_info = self.motion_extractor(x)             # 运动提取器M 返回 隐式关键点相关的信息
            if self.inference_cfg.flag_use_half_precision:     # 标志使用半精度  所有的张量值转换为 float32 类型
                # float the dict
                for k, v in kp_info.items():
                    if isinstance(v, torch.Tensor):
                        kp_info[k] = v.float()

        flag_refine_info: bool = kwargs.get('flag_refine_info', True)
        if flag_refine_info:
            bs = kp_info['kp'].shape[0]              # batch size：bs：单个批次的大小
            # 预测值转为角度          B x 1
            kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # B x 1    俯仰角
            kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # B x 1        偏航角
            kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # B x 1      横滚角
            #   N：关键点的个数（N=68?）    3代表xyz三维
            # -1表示在固定了bs和3后，根据剩下的参数自动计算N的个数
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # B x N x 3                     规范关键点
            kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # B x N x 3                   表情变换

        return kp_info

    # 通过姿势、位移和表情变形 计算 隐式关键点
    def transform_keypoint(self, kp_info: dict):
        """
        transform the implicit keypoints with the pose, shift, and expression deformation
        通过姿势、位移和表情变形转换隐式关键点
        kp: B x N x 3   N即k，关键点的个数
        """
        kp = kp_info['kp']    # (bs, k, 3)
        pitch, yaw, roll = kp_info['pitch'], kp_info['yaw'], kp_info['roll']

        t, exp = kp_info['t'], kp_info['exp']           # t: （bs，3）    exp： (bs, k, 3)
        scale = kp_info['scale']                        # (bs, 1)

        pitch = headpose_pred_to_degree(pitch)
        yaw = headpose_pred_to_degree(yaw)
        roll = headpose_pred_to_degree(roll)

        bs = kp.shape[0]
        if kp.ndim == 2:        # (bs, k * 3)
            num_kp = kp.shape[1] // 3  # Bx(num_kp x 3)
        else:
            num_kp = kp.shape[1]  # (bs, k, 3)

        rot_mat = get_rotation_matrix(pitch, yaw, roll)    # (bs, 3, 3)         即R

        # Eqn.2: s * (R * x_c,s + exp) + t
        kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat + exp.view(bs, num_kp, 3)     # x_c,s * R + exp        (bs, k, 3)
        # 使用None增加一维
        kp_transformed *= scale[..., None]  # (bs, k, 3) * (bs, 1, 1) = (bs, k, 3)      # * s                    (bs, k, 3)
        kp_transformed[:, :, 0:2] += t[:, None, 0:2]  # remove z, only apply tx ty      # + t                    (bs, k, 3)

        return kp_transformed    # 隐式关键点 (bs, k, 3)   bs图像个数；k关键点个数；3表示xyz三维

    # 重定向嘴唇
    def retarget_lip(self, kp_source: torch.Tensor, lip_close_ratio: torch.Tensor) -> torch.Tensor:
        """
        kp_source: B x N x 3
        lip_close_ratio: B x 2
        Return: B x (3 * num_kp)
        """
        feat_lip = concat_feat(kp_source, lip_close_ratio)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['lip'](feat_lip)

        return delta.reshape(-1, kp_source.shape[1], 3)

    # 缝合
    def stitch(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """
        kp_source: BxNx3
        kp_driving: BxNx3
        Return: Bx(3*num_kp+2)
        """
        feat_stiching = concat_feat(kp_source, kp_driving)

        with torch.no_grad():
            delta = self.stitching_retargeting_module['stitching'](feat_stiching)

        return delta

    def stitching(self, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ conduct the stitching
        kp_source: Bxnum_kpx3
        kp_driving: Bxnum_kpx3
        """

        if self.stitching_retargeting_module is not None:

            bs, num_kp = kp_source.shape[:2]

            kp_driving_new = kp_driving.clone()
            delta = self.stitch(kp_source, kp_driving_new)

            delta_exp = delta[..., :3*num_kp].reshape(bs, num_kp, 3)  # 1x20x3
            delta_tx_ty = delta[..., 3*num_kp:3*num_kp+2].reshape(bs, 1, 2)  # 1x1x2

            kp_driving_new += delta_exp
            kp_driving_new[..., :2] += delta_tx_ty

            return kp_driving_new

        return kp_driving

    # 扭曲解码（图像生成）     ！！！
    def warp_decode(self, feature_3d: torch.Tensor, kp_source: torch.Tensor, kp_driving: torch.Tensor) -> torch.Tensor:
        """ get the image after the warping of the implicit keypoints   获取隐式关键点扭曲后的图像
        feature_3d: B x 32 x 16 x 64 x 64, feature volume
        kp_source: B x N x 3
        kp_driving: B x N x 3
        """
        # The line 18 in Algorithm 1: D(W(f_s; x_s, x′_d,i)）
        with torch.no_grad(), self.inference_ctx():
            if self.compile:
                # Mark the beginning of a new CUDA Graph step   标记新CUDA图形步骤的开始
                torch.compiler.cudagraph_mark_step_begin()
            # get decoder input    获取被扭曲的图像    rec_det['out'] : Bx256x64x64
            ret_dct = self.warping_module(feature_3d, kp_source=kp_source, kp_driving=kp_driving)    # 预训练的扭曲模块 W
            # decode  生成最终的图像
            ret_dct['out'] = self.spade_generator(feature=ret_dct['out'])   # Bx3x512x512 归一化后的生成的图像                   # 预训练的生成器 G

            # float the dict 改变参数类型
            if self.inference_cfg.flag_use_half_precision:
                for k, v in ret_dct.items():
                    if isinstance(v, torch.Tensor):
                        ret_dct[k] = v.float()

        return ret_dct

    # 改变输出的格式
    def parse_output(self, out: torch.Tensor) -> np.ndarray:
        """ construct the output as standard   将输出构建为标准
        out： 1x3xHxW，标准化到0~1后的      Bx3x512x512
        return: 1xHxWx3, uint8
        """
        out = np.transpose(out.data.cpu().numpy(), [0, 2, 3, 1])  # 1x3xHxW -> 1xHxWx3
        out = np.clip(out, 0, 1)  # clip to 0~1
        out = np.clip(out * 255, 0, 255).astype(np.uint8)  # 0~1 -> 0~255

        return out  # Bx512x512x3

    def calc_ratio(self, lmk_lst):
        '''计算眼睛嘴巴close的比例'''
        input_eye_ratio_lst = []
        input_lip_ratio_lst = []
        for lmk in lmk_lst:
            # for eyes retargeting
            input_eye_ratio_lst.append(calc_eye_close_ratio(lmk[None]))
            # for lip retargeting
            input_lip_ratio_lst.append(calc_lip_close_ratio(lmk[None]))
        return input_eye_ratio_lst, input_lip_ratio_lst

    # 计算结合眼睛的比例
    def calc_combined_eye_ratio(self, c_d_eyes_i, source_lmk):
        c_s_eyes = calc_eye_close_ratio(source_lmk[None])
        c_s_eyes_tensor = torch.from_numpy(c_s_eyes).float().to(self.device)
        c_d_eyes_i_tensor = torch.Tensor([c_d_eyes_i[0][0]]).reshape(1, 1).to(self.device)
        # [c_s,eyes, c_d,eyes,i]
        combined_eye_ratio_tensor = torch.cat([c_s_eyes_tensor, c_d_eyes_i_tensor], dim=1)
        return combined_eye_ratio_tensor

    # 计算唇部张开度比例
    def calc_combined_lip_ratio(self, c_d_lip_i, source_lmk):
        c_s_lip = calc_lip_close_ratio(source_lmk[None])
        c_s_lip_tensor = torch.from_numpy(c_s_lip).float().to(self.device)
        c_d_lip_i_tensor = torch.Tensor([c_d_lip_i[0]]).to(self.device).reshape(1, 1) # 1x1
        # [c_s,lip, c_d,lip,i]
        combined_lip_ratio_tensor = torch.cat([c_s_lip_tensor, c_d_lip_i_tensor], dim=1) # 1x2
        return combined_lip_ratio_tensor

    # ───── 1pad-only overrides ──────────────────────────────────────────────

    def _extract_canonical_token(self, reference_path):
        if self.cropper is None:
            raise RuntimeError(
                "A cropper must be attached before extracting canonical keypoints"
            )

        img_rgb = load_image_rgb(reference_path)
        img_rgb = resize_to_limit(
            img_rgb,
            self.inference_cfg.source_max_dim,
            self.inference_cfg.source_division,
        )
        if self.inference_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(
                img_rgb, self.cropper.crop_cfg
            )
            if crop_info is None:
                raise RuntimeError("No face detected in the reference image")
            img_crop_256x256 = crop_info["img_crop_256x256"]
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))

        source = self.prepare_source(img_crop_256x256)
        source_info = self.get_kp_info(source)
        canonical_kp = source_info["kp"].reshape(1, -1)
        if canonical_kp.shape[-1] != 63:
            raise ValueError(
                f"Expected 63 canonical keypoint values, got {canonical_kp.shape[-1]}"
            )
        canonical_token = torch.cat(
            [
                torch.zeros(
                    1,
                    7,
                    dtype=canonical_kp.dtype,
                    device=canonical_kp.device,
                ),
                canonical_kp,
            ],
            dim=-1,
        ).unsqueeze(1)
        return canonical_token.detach()

    # （根据音频）生成运动序列！！！
    def gen_motion_sequence(self, args):
        # 1pad: 先从参考图像里抽出 canonical keypoint token，再让 denoiser 使用
        canonical_kp_feat = self._extract_canonical_token(args.reference)
        self.motion_generator.set_reference_priors(canonical_kp_feat)
        try:
            return self._gen_motion_sequence_impl(args)
        finally:
            self.motion_generator.clear_reference_priors()

    def _gen_motion_sequence_impl(self, args):
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

            # 情感类型
            emo_index = torch.tensor(emo_list.index(args.emotype))    # emo对应的索引值    # (B=1,)
            emo_index = emo_index.unsqueeze(0).to(self.device)   # 扩展维度并移动到指定设备（例如 GPU）

            if i == 0:    # 第一个片段没有先前帧的指导
                # 调用 motion_generator.sample 生成运动特征。返回的 motion_feat 是生成的运动特征，noise 是噪声，prev_audio_feat 是先前的音频特征。
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,        # (1, n_motion=100（即frame per seq), n_features=73) - 每个子序列生成的运动特征
                                                                        indicator=indicator, cfg_mode=args.cfg_mode,
                                                                        cfg_cond=args.cfg_cond, cfg_scale=args.cfg_scale,
                                                                        dynamic_threshold=0, emo_index=emo_index)
            else:
                motion_feat, noise, prev_audio_feat = self.motion_generator.sample(audio_in,
                                                                        prev_motion_feat, prev_audio_feat, noise,     # 需要添加先前的 运动特征，音频特征，噪声
                                                                        indicator=indicator, cfg_mode=args.cfg_mode,
                                                                        cfg_cond=args.cfg_cond, cfg_scale=args.cfg_scale,
                                                                        dynamic_threshold=0, emo_index=emo_index)
            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()        # copy临近的n_prev_motions=0帧，先前的运动特征  (N=1, L=10, motion_feat_dim=73)
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]             # copy，作为先前的音频特征                      (N=1, L=10, feature_dim=256)

            #### 情感增强
            if self.emo_ehance:
                emo_level = torch.tensor([args.enhance_level-1], dtype=torch.long).to(self.device)
                delta_emo = self.emo_enhancer(motion_feat[:, self.n_prev_motions:, :63], emo_index, emo_level)
                motion_feat[:, self.n_prev_motions:, :63] = motion_feat[:, self.n_prev_motions:, :63] + delta_emo.detach()

            motion_coef = motion_feat         #  运动系数 "coefficient"（系数）  (N=1, L=100, motion_feat_dim=73)
            if i == n_subdivision - 1 and n_padding_frames > 0:         # 最后一段音频序列
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames   删除右侧新增的n_padding_frames帧  (N=1, L - n_padding_frames, motion_feat_dim=73)
            coef_list.append(motion_coef)   # len = 3
            motion_coef = torch.cat(coef_list, dim=1)                # (1, n_frames视频总帧数, n_features=73) - 整个运动序列的运动系数      n_frames = n_motions * n_subdivision
            # motion_coef = self.reformat_motion(args, motion_coef)

        motion_coef = motion_coef.squeeze() #.cpu().numpy().astype(np.float32)     # 去除张量中所有尺寸为1的维度。(n_frames, n_features=70)

        motion_list = []              # 运动列表
        Emotion_template_dict = self.template_dict
        for idx in track(range(motion_coef.shape[0]), description='🚀Generating Motion Sequence...', total=motion_coef.shape[0]):    # 总帧数
            # 按照模板字典中的标准差和均值进行反归一化（与 exp 一致：正态分布归一化的逆变换）
            exp = motion_coef[idx][:63].cpu() * Emotion_template_dict["std_exp"] + Emotion_template_dict["mean_exp"]    # [63]
            scale = motion_coef[idx][63:64].cpu() * Emotion_template_dict["std_scale"] + Emotion_template_dict["mean_scale"]   # [1]
            t = motion_coef[idx][64:67].cpu() * Emotion_template_dict["std_t"] + Emotion_template_dict["mean_t"]    # [3]
            pitch = motion_coef[idx][67:68].cpu() * Emotion_template_dict["std_pitch"] + Emotion_template_dict["mean_pitch"]   # [1]
            yaw = motion_coef[idx][68:69].cpu() * Emotion_template_dict["std_yaw"] + Emotion_template_dict["mean_yaw"]   # [1]
            roll = motion_coef[idx][69:70].cpu() * Emotion_template_dict["std_roll"] + Emotion_template_dict["mean_roll"]   # [1]

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
        return tgt_motion


__all__ = ["ADEFWrapper"]

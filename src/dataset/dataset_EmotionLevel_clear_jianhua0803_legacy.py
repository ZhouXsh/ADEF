import os
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings

from src.config.emotion_config import global_emo_list
emo_list = global_emo_list

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class EmoLevelDataset(data.Dataset):
    def __init__(self,
                 root_dir='src/my_prepare/',                                        # 'prepare_data/'  数据集根目录
                 motion_filename="front_all_motions.pkl",              # 运动文件  'motions.pkl'
                 motion_template_filename="motion_template.pkl",   # 情感模板文件
                 split="train",
                 coef_fps=25,
                 n_motions=64,
                 n_prev_motions=16,
                 crop_strategy="random",
                 normalize_type="mix"):
        self.template_dir = os.path.join(root_dir, motion_template_filename)  # prepare_data/motion_template.pkl
        self.template_dict = pickle.load(open(self.template_dir, 'rb'))
        self.motion_dir = os.path.join(root_dir, motion_filename)     # prepare_data/motions.pkl
        self.eps = 1e-9
        self.normalize_type = normalize_type
        self.split = split

        if split == "train":
            self.root_dir = os.path.join(root_dir, "train.txt")  #  prepare_data/train.json
        else:
            self.root_dir = os.path.join(root_dir, "test.txt")  #  prepare_data/test.json

        # txt读取
        with open(self.root_dir, "r", encoding="utf-8") as file:
            lines = file.readlines()                    # 读取所有行，返回一个列表
            lines = [line.strip() for line in lines]                   # 名字之后改
            json_data = [{
                    "video_name": line,
                    "audio_name": line[:-4]+'.wav',
                    "motion_name": line[:-4]+'.pkl'
                }
                for line in lines
            ]

        self.all_data = json_data
        self.motion_data = pickle.load(open(self.motion_dir, "rb")) # prepare_data/motions.pkl  音频到Motion字典
        print("load all motion data done...")

        self.coef_fps = coef_fps            # 25
        self.audio_unit = 16000. / self.coef_fps  # num of samples per frame  每一帧的采样数  640
        self.n_motions = n_motions
        self.n_prev_motions = n_prev_motions
        self.n_audio_samples = round(self.audio_unit * self.n_motions)  # 音频子片段（current 段）的采样数（长度）
        self.coef_total_len = self.n_prev_motions + self.n_motions    # 系数总长度   16 + 64 = 80
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)  # 51200
        self.crop_strategy = crop_strategy

    def __len__(self, ):   # 视频的个数
        return len(self.all_data)

    def check_motion_length(self, motion_data, min_frames):
        """检查运动数据长度是否足够，不足则返回None（跳过该视频）"""
        if min_frames < self.coef_total_len + 4:
            return None

        exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
        for frame_index in range(min_frames):
            exp_list.append(motion_data["motion"][frame_index]["exp"])
            t_list.append(motion_data["motion"][frame_index]["t"])
            scale_list.append(motion_data["motion"][frame_index]["scale"])
            pitch_list.append(motion_data["motion"][frame_index]["pitch"])
            yaw_list.append(motion_data["motion"][frame_index]["yaw"])
            roll_list.append(motion_data["motion"][frame_index]["roll"])

        motion_new = {"motion": []}
        for i in range(len(exp_list)):
            motion = {
                "exp": exp_list[i],
                "t": t_list[i],
                "scale": scale_list[i],
                "pitch": pitch_list[i],
                "yaw": yaw_list[i],
                "roll": roll_list[i],
            }
            motion_new["motion"].append(motion)
        motion_new["n_frames"] = len(exp_list)
        return motion_new

    def augment_motion_data(self, coef_dict, emo_index):
        """
        运动系数数据增强（保持与音频的同步性）
        Args:
            coef_dict: 包含exp和pose的字典  [ exp:(200, 63),pose:(200,7) ]
            emo_index: 情感类别索引
        Returns:
            增强后的coef_dict
        """

        augmented_coef_dict = coef_dict.copy()
        augmented_coef_dict['exp'] = augmented_coef_dict['exp'] * 1.5

        return augmented_coef_dict


        augmented_coef_dict = coef_dict.copy()

        # 1. 随机噪声增强（保持时序同步）
        if np.random.random() < 0.3:  # 30%概率添加噪声
            noise_scale = np.random.uniform(0.01, 0.05)  # 噪声尺度
            noise = torch.randn_like(augmented_coef_dict['exp']) * noise_scale
            augmented_coef_dict['exp'] = augmented_coef_dict['exp'] + noise

        # 2. 幅度缩放增强（保持时序同步）
        if np.random.random() < 0.25:  # 25%概率进行幅度缩放
            scale_factor = np.random.uniform(0.8, 1.2)  # 缩放因子
            augmented_coef_dict['exp'] = augmented_coef_dict['exp'] * scale_factor

        # 3. 情感特定的增强（基于情感类别，保持时序同步）
        emotion_specific_augmentation = {
            0: ('angry', 0.1),    # 愤怒：增强幅度较大
            1: ('contempt', 0.05), # 轻蔑：轻微增强
            2: ('disgusted', 0.08), # 厌恶：中等增强
            3: ('fear', 0.07),     # 恐惧：中等增强
            4: ('happy', 0.12),    # 高兴：较大增强
            5: ('neutral', 0.02),  # 中性：很小增强
            6: ('sad', 0.06),      # 悲伤：中等增强
            7: ('surprised', 0.15) # 惊讶：最大增强
        }

        if np.random.random() < 0.4:  # 40%概率进行情感特定增强
            emotion, intensity = emotion_specific_augmentation.get(emo_index.item(), ('neutral', 0.05))

            if emotion in ['angry', 'happy', 'surprised']:
                # 增强表情幅度（全局缩放，不影响时序）
                emotion_scale = np.random.uniform(1.0, 1.0 + intensity)
                augmented_coef_dict['exp'] = augmented_coef_dict['exp'] * emotion_scale
            elif emotion in ['sad', 'fear', 'disgusted']:
                # 对这些情感添加平滑的表情模式（保持时序同步）
                pattern_strength = np.random.uniform(0.0, intensity)
                # 使用平滑的正弦波，不破坏时序关系
                pattern = torch.sin(torch.arange(augmented_coef_dict['exp'].shape[0]).float() * 0.1) * pattern_strength
                augmented_coef_dict['exp'] = augmented_coef_dict['exp'] + pattern.unsqueeze(1)

        # 4. 通道特定的噪声增强（不同表情通道添加不同强度的噪声）
        if np.random.random() < 0.2:  # 20%概率进行通道特定增强
            n_channels = augmented_coef_dict['exp'].shape[1]   # 63
            channel_noise_scale = torch.rand(n_channels) * 0.03  # 每个通道不同的噪声尺度
            channel_noise = torch.randn_like(augmented_coef_dict['exp']) * channel_noise_scale
            augmented_coef_dict['exp'] = augmented_coef_dict['exp'] + channel_noise

        # 5. 平滑处理（减少噪声增强可能引入的抖动）
        if np.random.random() < 0.15:  # 15%概率进行轻微平滑
            # 使用轻微的高斯平滑，保持整体时序形状
            kernel_size = 3
            sigma = 0.5
            exp_data = augmented_coef_dict['exp'].numpy()
            from scipy.ndimage import gaussian_filter1d
            for i in range(exp_data.shape[1]):  # 对每个通道单独平滑
                exp_data[:, i] = gaussian_filter1d(exp_data[:, i], sigma=sigma, mode='nearest')
            augmented_coef_dict['exp'] = torch.tensor(exp_data)

        # 确保数据范围合理（防止增强后出现极端值）
        augmented_coef_dict['exp'] = torch.clamp(augmented_coef_dict['exp'], -3.0, 3.0)

        return augmented_coef_dict

    # 获取具体的数据
    # 输入（视频id）索引，输出该视频的 音乐片段 及 对应的（标准化）运动系数片段，长度（帧数）为
    #   self.coef_total_len = self.n_prev_motions + self.n_motions = 16 + 64 = 80
    # 训练代码按"前 n_prev_motions / 后 n_motions"自行划分 prev / current。
    def __getitem__(self, index):
        has_valid_audio = False          # 没有有效的音频
        while not has_valid_audio:
            # read motion  读取运动系数
            metadata = self.all_data[index]   # 获取第index个视频对应的三文件：视频

            emotype = metadata['video_name'].split('/')[-1].split('_')[2]
            emo_index = torch.tensor(emo_list.index(emotype))    # emo对应的索引值

            emolevel = int(metadata['video_name'].split('/')[-1].split('_')[4])-1   # 1~3 -> 0~2
            emo_level = torch.tensor(emolevel)    # emo对应的索引值

            # 加载 motion
            motion_data = self.motion_data[metadata["audio_name"]]   # 单个视频的 运动系数字典motion_data

            # load audio & normalize  加载音频并标准化
            audio_path = metadata["audio_name"]
            audio_clip, sr = torchaudio.load(audio_path)  # 音频片段(1 or 2, sample_len)，采样率
            audio_clip = audio_clip.squeeze()    # (采样长度,)    .squeeze(): 去除多余的维度
            assert sr == 16000, f'Invalid sampling rate: {sr}'

            # 统计最小帧
            audio_frames = int(audio_clip.shape[0] / self.audio_unit)   # 计算音频对应的帧数
            motion_frames = motion_data["n_frames"]
            min_frames = min(audio_frames, motion_frames)   # 取最小帧数，避免不匹配

            # 根据最小帧 对motion进行裁剪（长度不足则跳过该视频）
            motion_data = self.check_motion_length(motion_data, min_frames)
            if motion_data is None:
                # print(f"skip short video: {os.path.basename(metadata['audio_name'])}, "
                #       f"min_frames: {min_frames}, required: {self.coef_total_len + 4}")
                index = np.random.randint(0, len(self.all_data))
                continue

            # 根据最小帧 对音频进行裁剪（不做复制填充）
            audio_clip = audio_clip[:int(min_frames * self.audio_unit)]

            # check裁剪前帧数是否匹配
            seq_len = motion_data["n_frames"]    # 系数总序列长度  seq_len > self.coef_total_len + 2
            assert int(seq_len * self.audio_unit) == audio_clip.shape[0], f'帧数不匹配: {seq_len * self.audio_unit} != {audio_clip.shape[0]}'

            # 计算裁剪的起始帧和结束帧 （反正一共要 self.coef_total_len = n_prev_motions + n_motions = 80 帧）
            if self.crop_strategy == 'random':  # 随机起始帧   this
                end = seq_len - self.coef_total_len   # 多余的部分
                if end < 0:   # 数据量不足self.coef_total_len，重新开始循环
                    print(f"current data invalid: {os.path.basename(metadata['audio_name'])}, n_frames: {seq_len}")
                    has_valid_audio = False
                    continue
                start_frame = np.random.randint(0, seq_len - self.coef_total_len - 2)  # 随机起始帧
            elif self.crop_strategy == 'begin':  # 从头开始
                start_frame = 0
            elif self.crop_strategy == 'end':   # 。。。直到结尾
                start_frame = seq_len - self.coef_total_len - 2
            else:
                raise ValueError(f'Unknown crop strategy: {self.crop_strategy}')
            end_frame = start_frame + self.coef_total_len   # 结束帧

            Emo_template_dict = self.template_dict   # 统一字典

            # 裁剪motion并标准化
            coef_keys = ["exp", "pose"] # exp - > exp, ['scale', 't', 'yaw', 'pitch', 'roll'] -> "pose"
            coef_dict = {k: [] for k in coef_keys}    # 空字典
            for frame_idx in range(start_frame, end_frame):   # 逐帧对两种运动系数进行标准化和归一化
                for coef_key in coef_keys:   # ["exp", "pose"]
                    if coef_key == "exp":
                        if self.normalize_type == "mix":
                            # 标准化
                            normalized_exp = (motion_data['motion'][frame_idx]["exp"].flatten() - Emo_template_dict["mean_exp"]) / (Emo_template_dict["std_exp"] + self.eps)
                        else:
                            raise RuntimeError("error")
                        coef_dict[coef_key].append([normalized_exp, ])   # (self.coef_total_len, n_exp)
                    elif coef_key == "pose":
                        if self.normalize_type == "mix":
                            pose_data = np.concatenate((   # (7,)
                                # 正态分布归一化（与 exp 一致）：(x - mean) / (std + eps)
                                (motion_data['motion'][frame_idx]["scale"].flatten() - Emo_template_dict["mean_scale"]) / (Emo_template_dict["std_scale"] + self.eps),
                                (motion_data['motion'][frame_idx]["t"].flatten() - Emo_template_dict["mean_t"]) / (Emo_template_dict["std_t"] + self.eps),
                                (motion_data['motion'][frame_idx]["pitch"].flatten() - Emo_template_dict["mean_pitch"]) / (Emo_template_dict["std_pitch"] + self.eps),
                                (motion_data['motion'][frame_idx]["yaw"].flatten() - Emo_template_dict["mean_yaw"]) / (Emo_template_dict["std_yaw"] + self.eps),
                                (motion_data['motion'][frame_idx]["roll"].flatten() - Emo_template_dict["mean_roll"]) / (Emo_template_dict["std_roll"] + self.eps),
                            ))
                        else:
                            raise RuntimeError("pose data error")
                        coef_dict[coef_key].append([pose_data, ])    # (self.coef_total_len, 7)
                    else:
                        raise RuntimeError("coef_key error: ", coef_key)
            coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}  # list->tensor       exp:(80, 63), pose:(80, 7)
            # ========== 添加数据增强模块（只在训练时） ==========
            # if self.split == "train":  # 注意：这里需要修改为 self.split
            #     coef_dict = self.augment_motion_data(coef_dict, emo_index)
            assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'

            # 裁剪 音频 并标准化
            audio = []
            audio.append(audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)])  # (self.coef_total_len * self.audio_unit,)   视频（采样长度）时长
            audio = torch.cat(audio, dim=0)             # (self.coef_total_len * self.audio_unit,)   视频（采样长度）时长   list->tensor
            if not (audio.shape[0] == self.audio_total_len):   # 不符合，重来
                print(f"audio length invalid! audio: {audio.shape[0]}, coef: {self.audio_total_len}")
                has_valid_audio = False
                continue

            # 提取一个连续的音频片段 (n_prev_motions + n_motions = 16 + 64 = 80 帧) 以及对应的运动系数片段。
            #   audio  : (audio_total_len,) = (80 * audio_unit,)，整段原始音频（对应 prev + current 共 80 帧）
            #   coef   : { 'exp': (80, 63), 'pose': (80, 7) }，整段归一化运动系数
            # 训练代码自行用 [: n_prev_motions] 作为 prev，用 [-n_motions:] 作为 current。
            keys = ['exp', 'pose']
            coef_single = {k: coef_dict[k].clone() for k in keys}    # exp:(80,63), pose:(80,7)
            has_valid_audio = True  # 有效的音乐片段。
            return audio, coef_single, emo_index, emo_level       # audio:(audio_total_len,)   coef_single:{exp:(80,63), pose:(80,7)}

import json
import os
import io
import torchaudio
import numpy as np
import torch
from torch.utils import data
import pickle
import warnings
import torch.nn.functional as F

'''
各情感独立分布的结果
'''

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

# torchaudio.set_audio_backend('soundfile')
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')

class EmoLevelDataset(data.Dataset):
    def __init__(self, 
                 root_dir,                                        # 'prepare_data/'  数据集根目录
                 motion_filename="talking_face.pkl",              # 运动文件  'motions.pkl'
                 motion_template_filename="front_all_Emotion_template.pkl",   # 情感模板文件
                 split="train", 
                 coef_fps=25, 
                 n_motions=100, 
                 crop_strategy="random", 
                 normalize_type="mix"): 
        self.template_dir = os.path.join(root_dir, motion_template_filename)  # prepare_data/motion_template.pkl
        print(self.template_dir)        # src/prepare_data/motion_template.pkl
        self.template_dict = pickle.load(open(self.template_dir, 'rb'))
        self.motion_dir = os.path.join(root_dir, motion_filename)     # prepare_data/motions.pkl
        self.eps = 1e-9
        self.normalize_type = normalize_type

        if split == "train":
            self.root_dir = os.path.join(root_dir, "all_train.txt")  #  prepare_data/train.json
        else:
            self.root_dir = os.path.join(root_dir, "front_test.txt")  #  prepare_data/test.json

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
        self.n_audio_samples = round(self.audio_unit * self.n_motions)  # 音频子片段的采样数（长度）
        self.coef_total_len = self.n_motions * 2    # 系数总长度   200
        self.audio_total_len = round(self.audio_unit * self.coef_total_len)  # 128000
        self.crop_strategy = crop_strategy
        
    def __len__(self, ):   # 视频的个数
        return len(self.all_data)
    
    def check_motion_length(self, motion_data, min_frames):
        # motion_data = {
        #     'n_frames': n_frames,         # 总帧数
        #     'output_fps': kwargs.get('output_fps', 25),
        #     'motion': [],
        #     'c_eyes_lst': [],
        #     'c_lip_lst': [],
        # }
        exp_list, t_list, scale_list, pitch_list, yaw_list, roll_list = [], [], [], [], [], []
        for frame_index in range(min_frames):
            exp_list.append(motion_data["motion"][frame_index]["exp"])
            t_list.append(motion_data["motion"][frame_index]["t"])
            scale_list.append(motion_data["motion"][frame_index]["scale"])
            pitch_list.append(motion_data["motion"][frame_index]["pitch"])
            yaw_list.append(motion_data["motion"][frame_index]["yaw"])
            roll_list.append(motion_data["motion"][frame_index]["roll"])

        if min_frames > self.coef_total_len + 4:    #  避免索引越界，并提供一定的缓冲
            motion_new = {"motion": []}
            for i in range(len(exp_list)):  # 填充
                motion = {
                    "exp": exp_list[i],
                    "t": t_list[i],
                    "scale": scale_list[i],
                    "pitch": pitch_list[i],
                    "yaw": yaw_list[i],
                    "roll": roll_list[i],
                }
                motion_new["motion"].append(motion)
            motion_new["n_frames"] = len(exp_list)    # 结算时的总帧数
            return motion_new   # 数据量足够，直接返回
        else:
            repeat = 0    # 重复次数
            while len(exp_list) < self.coef_total_len + 4:
                # 以翻倍的方式增加数据量，直到达标
                exp_list = exp_list * 2
                t_list = t_list * 2
                scale_list = scale_list * 2
                pitch_list = pitch_list * 2
                yaw_list = yaw_list * 2
                roll_list = roll_list * 2
                repeat += 1
            
            motion_new = {"motion": []}
            for i in range(len(exp_list)):  # 填充
                motion = {
                    "exp": exp_list[i],
                    "t": t_list[i],
                    "scale": scale_list[i],
                    "pitch": pitch_list[i],
                    "yaw": yaw_list[i],
                    "roll": roll_list[i],
                }
                motion_new["motion"].append(motion)
            motion_new["n_frames"] = len(exp_list)    # 结算时的总帧数
            motion_new["repeat"] = repeat             # 翻倍的次数
        return motion_new
     
    # 获取具体的数据
    # 输入（视频id）索引，输出该视频的 音乐片段 及 对应的（标准化）运动系数片段，长度（帧数）为self.coef_total_len = self.n_motions * 2 = 200
    def __getitem__(self, index): 
        has_valid_audio = False          # 没有有效的音频
        while not has_valid_audio:
            # read motion  读取运动系数
            metadata = self.all_data[index]   # 获取第index个视频对应的三文件：视频
            # metadata={
            #     "video_name": video_name,
            #     "audio_name": audio_name,
            #     "motion_name": motion_name,
            # }

### 新增：情感标签的索引映射 ###   zxs 20250314
# "video_name": "/mnt/disk2/zhouxishi/JoyVASA/single_video/M003_down_angry_level_1_001.mp4",
            emotype = metadata['video_name'].split('/')[-1].split('_')[2]
            emo_index = torch.tensor(emo_list.index(emotype))    # emo对应的索引值   

### 新增：情感增强 返回情感等级   zxs 20250419
            emolevel = int(metadata['video_name'].split('/')[-1].split('_')[4])-1   # 1~3 -> 0~2
            emo_level = torch.tensor(emolevel)    # emo对应的索引值     

            # 加载 motion
            motion_data = self.motion_data[metadata["audio_name"]]   # 单个视频的 运动系数字典motion_data
            # motion_data = {
            #     'n_frames': n_frames,         # 总帧数
            #     'output_fps': kwargs.get('output_fps', 25),
            #     'motion': [],
            #     'c_eyes_lst': [],
            #     'c_lip_lst': [],
            # }

            # load audio & normalize  加载音频并标准化
            audio_path = metadata["audio_name"]
            audio_clip, sr = torchaudio.load(audio_path)  # 音频片段(1 or 2, sample_len)，采样率
            audio_clip = audio_clip.squeeze()    # (采样长度,)    .squeeze(): 去除多余的维度
            assert sr == 16000, f'Invalid sampling rate: {sr}'

            # 统计最小帧
            audio_frames = int(audio_clip.shape[0] / self.audio_unit)   # 计算音频对应的帧数
            motion_frames = motion_data["n_frames"]
            min_frames = min(audio_frames, motion_frames)   # 取最小帧数，避免不匹配
            # print(f"min_frames: {min_frames}, audio_frames: {audio_frames}, motion_frames: {motion_frames}")

            # 根据最小帧 对motion进行裁剪+填充
            motion_data = self.check_motion_length(motion_data, min_frames)  # min_frames *2*2  > self.coef_total_len
            
            # 根据最小帧 对 音频  进行裁剪+填充
            audio_clip = audio_clip[:int(min_frames * self.audio_unit)]  # (self.coef_total_len * self.audio_unit,)   视频（采样长度）时长
            if "repeat" in motion_data:
                for _ in range(motion_data["repeat"]):
                    audio_clip = torch.cat((audio_clip, audio_clip), dim=0)  # 长度翻倍   (2 * 2**motion_data["repeat"], time_steps) or  (time_steps * 2**motion_data["repeat"], )
            
            # check裁剪前帧数是否匹配
            seq_len = motion_data["n_frames"]    # 系数总序列长度  seq_len > self.coef_total_len + 2
            assert int(seq_len * self.audio_unit) == audio_clip.shape[0], f'帧数不匹配: {seq_len * self.audio_unit} != {audio_clip.shape[0]}'

            # 计算裁剪的起始帧和结束帧 （反正一共要self.coef_total_len=200帧）
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

            # 0425 统一的template
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
                                # 各自《归一化》到[0, 1]
                                (motion_data['motion'][frame_idx]["scale"].flatten() - Emo_template_dict["min_scale"]) / (Emo_template_dict["max_scale"] - Emo_template_dict["min_scale"] + self.eps),
                                (motion_data['motion'][frame_idx]["t"].flatten() - Emo_template_dict["min_t"]) / (Emo_template_dict["max_t"] - Emo_template_dict["min_t"] + self.eps),
                                (motion_data['motion'][frame_idx]["pitch"].flatten() - Emo_template_dict["min_pitch"]) / (Emo_template_dict["max_pitch"] - Emo_template_dict["min_pitch"] + self.eps),
                                (motion_data['motion'][frame_idx]["yaw"].flatten() - Emo_template_dict["min_yaw"]) / (Emo_template_dict["max_yaw"] - Emo_template_dict["min_yaw"] + self.eps),
                                (motion_data['motion'][frame_idx]["roll"].flatten() - Emo_template_dict["min_roll"]) / (Emo_template_dict["max_roll"] - Emo_template_dict["min_roll"] + self.eps),
                            ))
                        else:
                            raise RuntimeError("pose data error")
                        coef_dict[coef_key].append([pose_data, ])    # (self.coef_total_len, 7)
                    else:
                        raise RuntimeError("coef_key error: ", coef_key)       
            coef_dict = {k: torch.tensor(np.concatenate(coef_dict[k], axis=0)) for k in coef_keys}  # list->tensor       [ exp:(200, 63),pose:(200,7) ] * 2
            assert coef_dict['exp'].shape[0] == self.coef_total_len, f'Invalid coef length: {coef_dict["exp"].shape[0]}'

            # 裁剪 音频 并标准化
            audio = []
            audio.append(audio_clip[round(start_frame * self.audio_unit):round(end_frame * self.audio_unit)])  # (self.coef_total_len * self.audio_unit,)   视频（采样长度）时长
            audio = torch.cat(audio, dim=0)             # (self.coef_total_len * self.audio_unit,)   视频（采样长度）时长   list->tensor
            if not (audio.shape[0] == self.audio_total_len):   # 不符合，重来
                print(f"audio length invalid! audio: {audio.shape[0]}, coef: {self.audio_total_len}")
                has_valid_audio = False 
                continue

            # Extract two consecutive audio/coef clips  提取两个连续的音频/音频片段
            keys = ['exp', 'pose']
            # audio_pair: self.coef_total_len * self.audio_unit = 200 * audio_unit = 2 * self.n_motions * audio_unit
            # audio_pair:[audio的前n_audio_samples，audio的后n_audio_samples]  其中n_audio_samples = round(self.audio_unit * self.n_motions=100)
            audio_pair = [audio[:self.n_audio_samples].clone(), audio[-self.n_audio_samples:].clone()]
            # coef_pair： self.coef_total_len = 200 = 2 * self.n_motions
            coef_pair = [{k: coef_dict[k][:self.n_motions].clone() for k in keys},   # 前n_motions=100
                        {k: coef_dict[k][-self.n_motions:].clone() for k in keys}]   # 后n_motions=100
            has_valid_audio = True  # 有效的音乐片段。
            return audio_pair, coef_pair, emo_index, emo_level       # [(采样长度,), (采样长度,)]   ，   [(exp: 100,63, pose: 100,7), (exp: 100,63, pose: 100,7)]

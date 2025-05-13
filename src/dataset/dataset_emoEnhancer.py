
from random import randint
import sys

import torch
import os
import pickle
from torch.utils import data
sys.path.append(os.path.dirname(os.path.abspath("../")))
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning
import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

# 同分布
class DiT_Emo_Dataset(data.Dataset):
    def __init__(self, root_dir='src/my_prepare', gt_motion_filename="front_all_motions.pkl", dit_motion_filename="front_dit_motions.pkl"):
        self.template_dict = pickle.load(open(os.path.join(root_dir, 'motion_template.pkl'), 'rb'))
        self.gt_motion_data = pickle.load(open(os.path.join(root_dir, gt_motion_filename), "rb"))
        self.dit_motion_data = pickle.load(open(os.path.join(root_dir, dit_motion_filename), "rb"))
        print("load all motion data done...")
        self.eps = 1e-9
        
        txt_path = os.path.join(root_dir, "all_train.txt")
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            self.all_data = [{
                "video_name": line,
                "audio_name": line[:-4]+'.wav'
            } for line in lines]
        self.all_num = 100

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        metadata = self.all_data[index] 
        # /W037/front/angry/level_3/W037_front_angry_level_3_034.mp4
        emotype = metadata['audio_name'].split('/')[-3]   # angry
        emo_index = torch.tensor(emo_list.index(emotype), dtype=torch.long)  # (1,)

        level_ = int(metadata['audio_name'].split('/')[-2].split('_')[-1])-1   # 0~2
        emo_level = torch.tensor(level_, dtype=torch.long)  # (1,)
    
        template_dict = self.template_dict         # 同分布

        gt_motions = self.gt_motion_data[metadata["audio_name"]]
        dit_name = metadata["audio_name"].replace('/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos', '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos')
        dit_motions = self.dit_motion_data[dit_name]
        # dit_motions = self.dit_motion_data[metadata["audio_name"]]

        min_frames = min(gt_motions['n_frames'], dit_motions['n_frames'])
        
        gt_list, dit_list = [],[]
        for i in range(min_frames):
            gt_motion_exp = gt_motions['motion'][i]['exp'] * 1.5
            dit_motion_exp = dit_motions['motion'][i]['exp']           # (1,21,3)

            normalized_exp11 = (gt_motion_exp.flatten() - template_dict["mean_exp"]) / (template_dict["std_exp"] + self.eps)
            normalized_exp22 = (dit_motion_exp.flatten() - template_dict["mean_exp"]) / (template_dict["std_exp"] + self.eps)

            gt_motion_exp = torch.tensor(normalized_exp11, dtype=torch.float32)           # (63)
            dit_motion_exp = torch.tensor(normalized_exp22, dtype=torch.float32)          # (63)

            gt_list.append(gt_motion_exp)
            dit_list.append(dit_motion_exp)

        gt_motion_exps =  torch.stack(gt_list, dim=0)      # (min_frames, 63)
        dit_motion_exps =  torch.stack(dit_list, dim=0)    # (min_frames, 63)

        while gt_motion_exps.shape[0] < self.all_num + 2:
            gt_motion_exps = torch.cat([gt_motion_exps, gt_motion_exps], dim=0)
            dit_motion_exps = torch.cat([dit_motion_exps, dit_motion_exps], dim=0)

        end_frame = randint(self.all_num, gt_motion_exps.shape[0] - 1)
        start_frame = end_frame - self.all_num

        dit_prev = dit_motion_exps[start_frame:end_frame]
        gt_prev = gt_motion_exps[start_frame:end_frame]

        return dit_prev, gt_prev, emo_index, emo_level

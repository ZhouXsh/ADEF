from random import randint
import sys
import torch
import os
import pickle
from torch.utils import data
sys.path.append(os.path.dirname(os.path.abspath("../")))

torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

class Motion2Emo_Dataset(data.Dataset):
    def __init__(self, root_dir='src/my_prepare', gt_motion_filename="front_all_motions.pkl", motion_template_filename="joyvasa_motion_template.pkl", train_dataset="all_train.txt"):
        self.template_dict = pickle.load(open(os.path.join(root_dir, motion_template_filename), 'rb'))
        self.gt_motion_data = pickle.load(open(os.path.join(root_dir, gt_motion_filename), "rb"))
        print("load all motion data done...")
        self.eps = 1e-9
        
        txt_path = os.path.join(root_dir, train_dataset)
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            self.all_data = [{
                "video_name": line,
                "audio_name": line[:-4]+'.wav',
                "motion_name": line[:-4]+'.pkl'
            } for line in lines]
        self.all_num = 100

    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, index):
        metadata = self.all_data[index] 
        # /W037/front/angry/level_3/W037_front_angry_level_3_034.mp4
        emotype = metadata['audio_name'].split('/')[-3]   # angry
        emo_index = torch.tensor(emo_list.index(emotype), dtype=torch.long)  # (1,)
        # template_dict = self.template_dict[emo_index.item()]    # 各自分布
        template_dict = self.template_dict                        # 真·同分布

        level = metadata['audio_name'].split('/')[-2].split('_')[-1]
        gt_level = torch.tensor(int(level)-1, dtype=torch.long)   # 1~3  变成 0~2

        gt_motions = self.gt_motion_data[metadata["audio_name"]]

        gt_list = []
        for i in range(gt_motions['n_frames']):
            gt_motion_exp = gt_motions['motion'][i]['exp']             # (1,21,3)
            normalized_exp11 = (gt_motion_exp.flatten() - template_dict["mean_exp"]) / (template_dict["std_exp"] + self.eps)
            gt_motion_exp = torch.tensor(normalized_exp11, dtype=torch.float32)           # (63)
            gt_list.append(gt_motion_exp)
        gt_motion_exps =  torch.stack(gt_list, dim=0)      # (min_frames, 63)

        while gt_motion_exps.shape[0] < self.all_num + 2:
            gt_motion_exps = torch.cat([gt_motion_exps, gt_motion_exps], dim=0)

        end_frame = randint(self.all_num, gt_motion_exps.shape[0] - 1)
        start_frame = end_frame - self.all_num

        gt_motion = gt_motion_exps[start_frame:end_frame]      # len = 100

        return gt_motion, emo_index, gt_level

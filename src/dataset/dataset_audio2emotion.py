import os
import pickle
import sys
import torch
from torch.utils.data import Dataset
sys.path.append(os.path.dirname(os.path.abspath("../")))

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

class AudioEmotionDataset(Dataset):
    def __init__(self, root_dir='src/my_prepare', np_dict = 'front_emotion2vec.pkl', train_txt="all_train.txt"):
        self.audio_data_list = pickle.load(open(os.path.join(root_dir, np_dict), 'rb'))
        txt_path = os.path.join(root_dir, train_txt)
        with open(txt_path, "r", encoding="utf-8") as file:
            lines = [line.strip() for line in file.readlines()]
            self.all_data = [{
                "video_name": line,
                "audio_name": line[:-4]+'.wav',
                "npy_name": line[:-4]+'.npy'
            } for line in lines]

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        datas = self.all_data[idx]['audio_name']         # [audio waveform tensor]
        emotype = datas.split('/')[-3]   # angry
        emo_index = torch.tensor(emo_list.index(emotype), dtype=torch.long)  # (1,)
        np_feat = torch.tensor(self.audio_data_list[datas])
        return np_feat, emo_index

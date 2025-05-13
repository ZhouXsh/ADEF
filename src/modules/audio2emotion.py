import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
sys.path.append(os.path.dirname(os.path.abspath("../")))
from src.dataset import infinite_data_loader

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

# 假设 emotion2vec 是一个可以将音频转换为 1024 维向量的模型
# 假设你已经有一个 Dataset，它返回 (audio_tensor, emotion_label)
class AudioEmotionDataset(Dataset):
    def __init__(self, root_dir='/mnt/disk2/zhouxishi/JoyVASA/src/my_prepare', np_dict = 'front_emotion2vec.pkl', train_txt="all_train.txt"):
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

class AudioEmotionClassifierModel(nn.Module):
    def __init__(self, num_classifier_layers=5, num_classifier_channels=2048, num_emotion_classes=8):
        super().__init__()
        self.num_emotion_classes = num_emotion_classes

        if num_classifier_layers == 1:
            self.layers = nn.Linear(1024, self.num_emotion_classes)
        else:
            layer_list = [
                nn.Linear(1024, num_classifier_channels),
                nn.ReLU()
            ]
            for _ in range(num_classifier_layers - 2):
                layer_list.append(nn.Linear(num_classifier_channels, num_classifier_channels))
                layer_list.append(nn.ReLU())
            layer_list.append(nn.Linear(num_classifier_channels, self.num_emotion_classes))
            self.layers = nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.layers(x)
        return x

# 准确率函数
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

# 示例主函数
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100000
    batch_size = 64
    lr = 1e-4

    dataset = AudioEmotionDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader = infinite_data_loader(dataloader)

    model = AudioEmotionClassifierModel().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 初始化 TensorBoard 写入器
    log_dir=f'emo2vec/lr{lr}_b{batch_size}'
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(num_epochs):
        vecs, labels = next(data_loader)
        vecs = vecs.to(device)      # [32, 1024]
        labels = labels.to(device)
        outputs = model(vecs)       # [32, 8]
        loss = criterion(outputs, labels)
        acc = accuracy(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 写入 TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch)
        writer.add_scalar('Accuracy/train', acc, epoch)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, acc: {acc:.4f}")

    writer.close()
    save_dir = f"{log_dir}/emo2vec2emo_{num_epochs}_lr{lr}_b{batch_size}.pth"
    torch.save(model.state_dict(), save_dir)

def eval():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    dataset = AudioEmotionDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioEmotionClassifierModel().to(device)
    dict = torch.load('/mnt/disk2/zhouxishi/JoyVASA/pretrained_weights/ADEF/audio2emo/audio2emo.pth',map_location='cuda:0')
    model.load_state_dict(dict)
    model.eval()

    num,acc = 0,0.0
    for vecs, labels in dataloader:
        
        vecs = vecs.to(device)      # [32, 1024]
        labels = labels.to(device)
        outputs = model(vecs)       # [32, 8]
        num += 1
        acc += accuracy(outputs, labels)

    print(f"{acc} / {num} = {acc/num}")  # 520.484375 / 521 = 0.9990

import os
import pickle

from funasr import AutoModel

# model_id = "iic/emotion2vec_plus_large"

# model = AutoModel(
#     model=model_id,
#     hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
# )

def extract_emo2vec(wav_file,output_dir):

    model_id = "iic/emotion2vec_plus_large"

    model = AutoModel(
        model=model_id,
        hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
    )
    model.generate(wav_file, output_dir=output_dir, granularity="utterance", extract_embedding=True)
    '''
    key: file_name
    labels: emotion type  ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']
    scores: possibility of each emotion
    feats: emo_vector       shape:(1024,)
    '''

def single():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # extract_emo2vec('/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/angry/level_3/M003_front_angry_level_3_001.wav', '.')

    np_path = '/mnt/disk2/zhouxishi/ADEF/src/modules/M003_front_angry_level_3_001.npy'

    emo_vec = torch.tensor(np.load(np_path)).unsqueeze(0).to(device)  # [1, 1024]

    a2e_model = AudioEmotionClassifierModel().to(device)
    dict = torch.load('/mnt/disk2/zhouxishi/ADEF/pretrained_weights/ADEF/audio2emo/audio2emo.pth',map_location=device)
    a2e_model.load_state_dict(dict)
    a2e_model.eval()

    outputs = a2e_model(emo_vec)
    print(outputs)

    argmax = outputs.argmax(dim=1).item()
    print(argmax)

    # num,acc = 0,0.0
    # for vecs, labels in dataloader:
        
    #     vecs = vecs.to(device)      # [32, 1024]
    #     labels = labels.to(device)
    #     outputs = a2e_model(vecs)       # [32, 8]
    #     num += 1
    #     acc += accuracy(outputs, labels)

    # print(f"{acc} / {num} = {acc/num}")  # 520.484375 / 521 = 0.9990


if __name__ == "__main__":
    # main()
    # eval()
    single()
import logging
from random import randint
import sys
import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath("../")))
from src.dataset import infinite_data_loader
from src.modules.common import PositionalEncoding

torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

class Motion2Emo_Dataset(data.Dataset):
    def __init__(self, root_dir='/mnt/disk2/zhouxishi/JoyVASA/src/my_prepare', gt_motion_filename="front_all_motions.pkl", motion_template_filename="joyvasa_motion_template.pkl", train_dataset="all_train.txt"):
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

class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=63, embed_dim=512, num_heads=8, ff_dim=4*512, num_layers=8, num_classes=8, level_classes=3):
        super(EmotionTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 运动参数嵌入层 (B, L, 63) -> (B, L, D)
        self.motion_embedding = nn.Linear(input_dim, embed_dim)
        
        # 位置编码
        self.PE = PositionalEncoding(embed_dim)

        # Transformer Encoder
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=num_layers)

        # 输出层 (B, D) -> (B, 8)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.classifier_level = nn.Linear(embed_dim, level_classes)

    def forward(self, motion_seq):
        """
        motion_seq: (B, L=100, 63)  待预测情感的运动序列
        """
        # 运动参数嵌入 + 位置编码
        motion_embed = self.motion_embedding(motion_seq)  # (B, L, D)
        motion_embed = self.PE(motion_embed)

        # 经过 Transformer Encoder
        encoded_features = self.encoder(motion_embed)  # (B, L, D=128)

        # 平均池化操作
        encoded_features = encoded_features.mean(dim=1)  # (B, L, D) -> (B, D)

        out_emo = self.classifier(encoded_features)   # (B, D) -> (B, 8)
        out_level = self.classifier_level(encoded_features)         # 情感等级输出 (B, 3)

        return out_emo, out_level   # (B, 8)   # (B, 3)

# 准确率函数
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

dates = '0430'

def transf_train(devices=1):
    # 训练超参数
    num_epochs = 50000
    learning_rate = 1e-4

    warm_iter = 10000
    decay_iter = 90000

    log_dir = f'./logs_motion2emolevel_Transf_{num_epochs}_{dates}_所有视频同分布_高dim'
    
    writer = SummaryWriter(log_dir)   # 路径

    # 初始化模型
    device = torch.device(f"cuda:{devices}" if torch.cuda.is_available() else "cpu")
    model = EmotionTransformer().to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, decay_iter, learning_rate * 0.02)
    # scheduler = GradualWarmupScheduler(optimizer, 1, warm_iter, after_scheduler)

    train_dataset = Motion2Emo_Dataset()
    train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    train_loader = infinite_data_loader(train_loader)   # 将数据加载器（train_loader）转换为一个无限循环的迭代器

    model.train()
    loss_log = {
        'total_loss': [],
        'emo_loss': [],
        'level_loss': [],
        'acc_emo': [],
        'acc_level': [],
        'acc_all': []
    }

    for epoch in range(num_epochs):
        gt_motion, emo_index, gt_level = next(train_loader)
        gt_motion = gt_motion.to(device)     # （B,100,63）
        emo_index = emo_index.to(device)     # （B,）
        gt_level = gt_level.to(device)       # （B,）
        optimizer.zero_grad()

        pred_emo,pred_level = model(gt_motion)     #   (B, 1, 63)

        loss_emo = criterion(pred_emo, emo_index)    # （B,）
        loss_level = criterion(pred_level, gt_level) # （B,）

        total_loss = loss_emo + loss_level
        total_loss.backward()

        optimizer.step()

        acc_emo = accuracy(pred_emo, emo_index)
        acc_level = accuracy(pred_level, gt_level)
        acc_all = (acc_emo + acc_level) / 2

        loss_log['emo_loss'].append(loss_emo.mean().item())
        loss_log['level_loss'].append(loss_level.mean().item())
        loss_log['total_loss'].append(total_loss.mean().item())
        loss_log['acc_emo'].append(acc_emo)
        loss_log['acc_level'].append(acc_level)
        loss_log['acc_all'].append(acc_all)

        # Create description string for logging
        description = f'Iter: {epoch}\t Train loss: [Total: {np.mean(loss_log["total_loss"]):.3e}'
        description += f", Emo_loss: {np.mean(loss_log['emo_loss']):.3e}"
        description += f", Level_loss: {np.mean(loss_log['level_loss']):.3e}"
        description += f", ACC_all: {np.mean(loss_log['acc_all']):.3e}"
        description += f", ACC_emo: {np.mean(loss_log['acc_emo']):.3e}"
        description += f", ACC_level: {np.mean(loss_log['acc_level']):.3e}"
        description += ']'
        logging.info(description)

        # Write to tensorboard
        if epoch % 50 == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['total_loss']), epoch)
            writer.add_scalar('train/emo_loss', np.mean(loss_log['emo_loss']), epoch)
            writer.add_scalar('train/level_loss', np.mean(loss_log['level_loss']), epoch)
            writer.add_scalar('train/acc_all', np.mean(loss_log['acc_all']), epoch)
            writer.add_scalar('train/acc_emo', np.mean(loss_log['acc_emo']), epoch)
            writer.add_scalar('train/acc_level', np.mean(loss_log['acc_level']), epoch)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # Clear the loss log for next interval
            for key in loss_log.keys():
                loss_log[key].clear()

        # update learning rate  更新学习率
        # if scheduler is not None and epoch < warm_iter + decay_iter:   # 调度器用于更新学习率  区分：优化器optimizor
        #     scheduler.step()

        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}, ACC: {acc_emo:.6f} + {acc_level:.6f}")

    # print("训练完成！")
    save_dir = f"{log_dir}/transf_motion2emolevel_{num_epochs}_{dates}.pth"
    torch.save(model.state_dict(), save_dir)

# 评估
@torch.no_grad()
def eval():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    transf_model = EmotionTransformer().to(device)
    # transf_model.load_state_dict(torch.load('/mnt/disk2/zhouxishi/JoyVASA/src/modules/logs_motion2emolevel_Transf_50000_0429_所有视频同分布_高dim/transf_motion2emolevel_50000_0429.pth'))
    transf_model.load_state_dict(torch.load('/mnt/disk2/zhouxishi/JoyVASA/pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth'))
    transf_model.eval()

    eval_dataset = Motion2Emo_Dataset()
    eval_loader = data.DataLoader(eval_dataset, batch_size=64, shuffle=True)

    epoch, acc, acc_level = 0, 0.0, 0.0
    for gt_motion, emo_index, gt_level in eval_loader:
        epoch +=1
        gt_motion = gt_motion.to(device)     # （B,100,63）
        emo_index = emo_index.to(device)     # （B,）
        gt_level = gt_level.to(device)

        pred_emo, pred_level = transf_model(gt_motion)
        acc += accuracy(pred_emo, emo_index)
        acc_level += accuracy(pred_level, gt_level)
    
    print(f'{acc} / {epoch} = {acc/epoch}')
    print(f'{acc_level} / {epoch} = {acc_level/epoch}')

    return None

if __name__ == '__main__':
    # transf_train()
    eval()
import logging
import sys
from src.dataset.dataset_emoClassifier import Motion2Emo_Dataset
from src.modules.emotion_level_classifier import EmotionTransformer
import torch
import torch.nn as nn
import os
import numpy as np
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
sys.path.append(os.path.dirname(os.path.abspath("../")))
from src.dataset import infinite_data_loader

torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2; cv2.setNumThreads(0); cv2.ocl.setUseOpenCL(False)

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
emo_label = ['ang',  'con',  'dis',  'fea',  'hap',  'neu',  'sad',  'sur']

# 准确率函数
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def train(devices=1):
    # 训练超参数
    num_epochs = 50000
    learning_rate = 1e-4

    log_dir = f'experiments/emo_classifier/'
    
    writer = SummaryWriter(log_dir)   # 路径

    # 初始化模型
    device = torch.device(f"cuda:{devices}" if torch.cuda.is_available() else "cpu")
    model = EmotionTransformer().to(device)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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

        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}, ACC: {acc_emo:.6f} + {acc_level:.6f}")

    # print("训练完成！")
    save_dir = f"{log_dir}/ckpt.pth"
    torch.save(model.state_dict(), save_dir)

# 评估
@torch.no_grad()
def eval():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    transf_model = EmotionTransformer().to(device)
    transf_model.load_state_dict(torch.load('pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth'))
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
    train()
    # eval()
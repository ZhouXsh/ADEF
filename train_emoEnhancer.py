import logging

from src.dataset.dataset_emoEnhancer import DiT_Emo_Dataset
from src.modules.emotion_enhancer import EmotionTransformer
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils import data
from tensorboardX import SummaryWriter
from src.dataset import infinite_data_loader
from src.scheduler import GradualWarmupScheduler
torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

from src.modules.emotion_level_classifier import EmotionTransformer as Classifier

def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']

def transf_train():
    # 训练超参数
    num_epochs = 120000
    learning_rate = 1e-4

    warm_iter = 12000
    decay_iter = 120000
    batch_size = 64
    # 同分布
    log_dir = f'experiments/emo_enhancer/'
    
    writer = SummaryWriter(log_dir)   # 路径

    # 初始化模型
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = EmotionTransformer().to(device)

    # 0417 情感分类器    0426高dim且同分布
    emo_classifier = Classifier().to(device)
    emo_classifier.load_state_dict(torch.load(f'pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth', map_location=device))
    emo_classifier.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # 定义损失函数
    mse_loss = nn.MSELoss()  # 主要损失，约束运动参数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    after_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, decay_iter, learning_rate * 0.02)
    scheduler = GradualWarmupScheduler(optimizer, 1, warm_iter, after_scheduler)

    train_dataset = DiT_Emo_Dataset()
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = infinite_data_loader(train_loader)   # 将数据加载器（train_loader）转换为一个无限循环的迭代器

    model.train()
    loss_log = {
        'total_loss': [],
        'mse_loss': []
        ,'emo_loss':[]
        ,'level_loss': []
    }

    for epoch in range(num_epochs):
        dit_prev, gt_prev, emo_index, emo_level = next(train_loader)
        dit_prev = dit_prev.to(device)     # （B,100,63）
        emo_index = emo_index.to(device)     # （B,）
        gt_prev = gt_prev.to(device)       # （B,100,63）
        emo_level = emo_level.to(device)

        optimizer.zero_grad()

        pred = model(dit_prev, emo_index, emo_level)     #   (B, 1, 63)

        _, L, _ = dit_prev.shape
        dit_prev = dit_prev + pred.expand(-1, L, -1)   # (B, 1, 63)  -> (B, L, 63)

        # === 计算各个损失 ===
        loss_mse = mse_loss(dit_prev, gt_prev)  # (B, 100, 63)

        pred_emo, pred_level = emo_classifier(dit_prev)   # (N,100,63)  -> (N,8)
        loss_emo = criterion(pred_emo, emo_index)
        loss_level = criterion(pred_level, emo_level)

        # === 计算总损失 ===
        total_loss = loss_mse + loss_emo + loss_level

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # Inside your training loop, after computing the losses:
        # Logging - Append current batch losses
        loss_log['total_loss'].append(total_loss.mean().item())
        loss_log['mse_loss'].append(loss_mse.mean().item())
        loss_log['emo_loss'].append(loss_emo.mean().item())
        loss_log['level_loss'].append(loss_emo.mean().item())

        # Create description string for logging
        description = f'Iter: {epoch}\t Train loss: [Total: {np.mean(loss_log["total_loss"]):.3e}'
        description += f", MSE: {np.mean(loss_log['mse_loss']):.3e}"
        description += f", Emo: {np.mean(loss_log['emo_loss']):.3e}"
        description += f", Level: {np.mean(loss_log['level_loss']):.3e}"
        description += ']'
        logging.info(description)

        # Write to tensorboard
        if epoch % 50 == 0 and writer is not None:
            writer.add_scalar('train/total_loss', np.mean(loss_log['total_loss']), epoch)
            writer.add_scalar('train/mse_loss', np.mean(loss_log['mse_loss']), epoch)
            writer.add_scalar('train/emo_loss', np.mean(loss_log['emo_loss']), epoch)
            writer.add_scalar('train/level_loss', np.mean(loss_log['level_loss']), epoch)
            writer.add_scalar('opt/lr', optimizer.param_groups[0]['lr'], epoch)
            
            # Clear the loss log for next interval
            for key in loss_log.keys():
                loss_log[key].clear()

        # update learning rate  更新学习率
        if scheduler is not None and epoch < warm_iter + decay_iter:   # 调度器用于更新学习率  区分：优化器optimizor
            scheduler.step()

        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss.item():.6f}")

    print("训练完成！")
    save_dir = f"{log_dir}/ckpt.pth"
    torch.save(model.state_dict(), save_dir)

if __name__ == "__main__":
    transf_train()
# 示例主函数
from src.dataset import infinite_data_loader
from src.dataset.dataset_audio2emotion import AudioEmotionDataset
from src.modules.audio2emotion import Audio2EmotionModel
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 准确率函数
def accuracy(preds, labels):
    return (preds.argmax(dim=1) == labels).float().mean().item()

def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100000
    batch_size = 64
    lr = 1e-4

    dataset = AudioEmotionDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    data_loader = infinite_data_loader(dataloader)

    model = Audio2EmotionModel().to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # 初始化 TensorBoard 写入器
    log_dir=f'experiments/audio2emo/'
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
    save_dir = f"{log_dir}/audio2emotion_{num_epochs}_lr{lr}_b{batch_size}.pth"
    torch.save(model.state_dict(), save_dir)

def eval():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    batch_size = 64

    dataset = AudioEmotionDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Audio2EmotionModel().to(device)
    dict = torch.load('pretrained_weights/ADEF/audio2emo/audio2emo.pth',map_location='cuda:0')
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

if __name__ == "__main__":
    # main()
    eval()
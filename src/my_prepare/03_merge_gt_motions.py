# 读取文件并将每行存入列表
import pickle
from tqdm import tqdm

with open(f"train.txt", "r") as f:
    train_data = [line.strip() for line in f.readlines()]

with open(f"test.txt", "r") as f:
    test_data = [line.strip() for line in f.readlines()]

data = train_data + test_data
all_items = {}
for item in tqdm(data, desc="加载运动数据"):
    key = item[:-4] + '.wav'               # 使用音频文件名作为键
    motion_name = item[:-4] + '.pkl'       # 获取运动数据文件名
    motions = pickle.load(open(motion_name, 'rb'))  # 读取运动数据（pkl 格式）
    value = motions  # 这里 motions 直接作为值（可以进行预处理）

    all_items[key] = value  # 存入字典

save_name = f'front_all_motions.pkl'
pickle.dump(all_items, open(save_name, 'wb'))       # 使用 pickle 序列化并保存到文件

print("运动数据处理完成")

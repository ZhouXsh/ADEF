import os
import random

def prefix(filename):  # tools
    """获取文件名的前缀（去掉扩展名）
    例如：'a.jpg' -> 'a'
    """
    pos = filename.rfind(".")  # 找到最后一个 "." 的位置
    if pos == -1:
        return filename  # 如果没有 "."，直接返回原文件名
    return filename[:pos]  # 返回去掉扩展名的部分

def basename(filename):  # tools
    """获取文件的基本名（去掉路径，只保留文件名，并去掉扩展名）
    例如：'a/b/c.jpg' -> 'c'
    """
    return prefix(os.path.basename(filename))  # 先获取文件名，再去掉扩展名

def remove_suffix(filepath):  # tools
    """去掉文件的后缀，保留路径和文件名
    例如：'a/b/c.jpg' -> 'a/b/c'
    """
    return os.path.join(os.path.dirname(filepath), basename(filepath))  # 拼接目录和无后缀的文件名


video_list = []
root_dir = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos'
for id in os.listdir(root_dir):    # 演员id
    # id = os.path.join(id)
    front = os.path.join(root_dir, id, 'front')
    for emo in os.listdir(front):
        emo_dir = os.path.join(front, emo)
        for level in os.listdir(emo_dir):
            level_dir = os.path.join(emo_dir, level)
            for video_file in os.listdir(level_dir):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(level_dir, video_file)
                    video_list.append(video_path)
                    if emo == 'neutral':  # 只有level1，扩充数据
                        video_list.append(video_path)
                        video_list.append(video_path)
random.shuffle(video_list)     # 打乱顺序
num_train_labels = int(0.9 * len(video_list)) 

# 打开文件并写入
with open(f"all_train.txt", "w") as f:
    for item in video_list[:num_train_labels]:
        f.write(f"{item}\n")

with open(f"all_test.txt", "w") as f:
    for item in video_list[num_train_labels:]:
        f.write(f"{item}\n")

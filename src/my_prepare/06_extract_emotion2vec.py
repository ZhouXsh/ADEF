'''emotion2vec'''

'''
Using the finetuned emotion recognization model

rec_result contains {'feats', 'labels', 'scores'}
	extract_embedding=False: 9-class emotions with scores
	extract_embedding=True: 9-class emotions with scores, along with features

9-class emotions: 
iic/emotion2vec_plus_seed, iic/emotion2vec_plus_base, iic/emotion2vec_plus_large (May. 2024 release)
iic/emotion2vec_base_finetuned (Jan. 2024 release)
    0: angry
    1: disgusted
    2: fearful
    3: happy
    4: neutral
    5: other
    6: sad
    7: surprised
    8: unknown
'''

import os
import pickle

import numpy as np
from tqdm import tqdm
from funasr import AutoModel

model_id = "iic/emotion2vec_plus_large"

model = AutoModel(
    model=model_id,
    hub="ms",  # "ms" or "modelscope" for China mainland users; "hf" or "huggingface" for other overseas users
)

# wav_file = f"{model.model_path}/example/test.wav"

def extract_emo2vec(wav_file,output_dir):
    model.generate(wav_file, output_dir=output_dir, granularity="utterance", extract_embedding=True)
    '''
    key: file_name
    labels: emotion type  ['生气/angry', '厌恶/disgusted', '恐惧/fearful', '开心/happy', '中立/neutral', '其他/other', '难过/sad', '吃惊/surprised', '<unk>']
    scores: possibility of each emotion
    feats: emo_vector       shape:(1024,)    # (768,)
    '''

def front_all():
    audio_list = []
    root_dir = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos'
    for id in os.listdir(root_dir):    # 演员id
        # id = os.path.join(id)
        front = os.path.join(root_dir, id, 'front')
        for emo in os.listdir(front):
            emo_dir = os.path.join(front, emo)
            for level in os.listdir(emo_dir):
                level_dir = os.path.join(emo_dir, level)
                for audio_file in os.listdir(level_dir):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(level_dir, audio_file)
                        audio_list.append((audio_path,level_dir))
    for audio, level_dir in tqdm(audio_list):
        extract_emo2vec(audio, level_dir)

def save2dict():
    audio_list = []
    root_dir = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos'
    for id in os.listdir(root_dir):    # 演员id
        # id = os.path.join(id)
        front = os.path.join(root_dir, id, 'front')
        for emo in os.listdir(front):
            emo_dir = os.path.join(front, emo)
            for level in os.listdir(emo_dir):
                level_dir = os.path.join(emo_dir, level)
                for audio_file in os.listdir(level_dir):
                    if audio_file.endswith('.wav'):
                        audio_path = os.path.join(level_dir, audio_file)
                        audio_list.append(audio_path)
    all_items = {}
    for audio_path in tqdm(audio_list, desc="加载运动数据"):
        key = audio_path              # 使用音频文件名作为键
        np_name = audio_path[:-4] + '.npy'       # 获取运动数据文件名
        emo_vec = np.load(np_name)  # 读取运动数据（pkl 格式）
        value = emo_vec  # 这里 motions 直接作为值（可以进行预处理）

        all_items[key] = value  # 存入字典

    save_name = f'front_emotion2vec.pkl'
    pickle.dump(all_items, open(save_name, 'wb'))       # 使用 pickle 序列化并保存到文件

    print("运动数据处理完成")

front_all()   # 29709
save2dict()  # 29709
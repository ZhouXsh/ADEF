'''
测试用例的生成
实验部分。。。
'''

import os
import subprocess

from tqdm import tqdm

image_dir = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/first_frame'

images_list = sorted([os.path.join(image_dir,image) for image in os.listdir(image_dir) if image.endswith('.png')])

def check(image_path,audio_path,out_dir):
    image_path = image_path.split('/')[-1][:-4]
    audio_path = audio_path.split('/')[-1][:-4]
    video = f'{out_dir}/{image_path}_{audio_path}.mp4'
    if os.path.exists(video):
        print(f'{video} 已存在')    
        return True
    return False    

def exec_emo(image_path,audio_path,out_dir = '.', emotion='angry',use_emo_enhancer=False,enhance_level=1,use_emo_analyzer = False,device_id = 1):
    # if check(image_path,audio_path,out_dir):
    #     return
    cmd = f'python inference.py -r {image_path} -a {audio_path} -e {emotion} --cfg_scale 1.5 --output_dir {out_dir} --use_emo_enhancer {use_emo_enhancer} --enhance_level {enhance_level} --use_emo_analyzer {use_emo_analyzer} --device_id {device_id}'
    subprocess.run(cmd,shell=True)

def DitOnly(device_id = 1):
    out_dir = '/mnt/disk2/zhouxishi/JoyVASA/eval/0513_DiT_Only'
    for i in tqdm(range(0,len(images_list),25)):
        image = images_list[i]
# M009/front/angry/level_2/M009_front_angry_level_2_001.wav
# /mnt/disk2/zhouxishi/JoyVASA/eval/11image/M003_front_angry_level_1_001.jpg
        image_names = image.split('/')[-1].split('.')[0]    # M009_front_angry_level_2_001
        image_sp = image_names.split('_')  # M009 front angry level 2 001
        emo_name = image_sp[2]  # angry
        emo_level = int(image_sp[4])-1  
        audio = f'/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/{image_sp[0]}/{image_sp[1]}/{image_sp[2]}/{image_sp[3]}_{image_sp[4]}/{image_names}.wav'
        exec_emo(image,audio,out_dir,emo_name,False,emo_level,False,device_id)

def Dit_Enhancer(device_id = 3):
    out_dir = '/mnt/disk2/zhouxishi/JoyVASA/eval/0513_DiT_Enhancer'
    for i in tqdm(range(0,len(images_list),25)):
        image = images_list[i]
        # XXXXX/M009_front_angry_level_2_001.wav
        image_names = image.split('/')[-1].split('.')[0]    # M009_front_angry_level_2_001
        image_sp = image_names.split('_')  # M009 front angry level 2 001
        emo_name = image_sp[2]  # angry
        emo_level = int(image_sp[4])-1  
        audio = f'/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/{image_sp[0]}/{image_sp[1]}/{image_sp[2]}/{image_sp[3]}_{image_sp[4]}/{image_names}.wav'
        exec_emo(image,audio,out_dir,emo_name,True,emo_level,False,device_id)

DitOnly()
# Dit_Enhancer()
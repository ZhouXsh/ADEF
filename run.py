import os
import subprocess
import csv
import time

'''
情感测试
'''

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


# traget = '/mnt/disk2/zhouxishi/JoyVASA/new'
# image_neu = '/mnt/disk2/zhouxishi/JoyVASA/single_image_test/images/开心.jpg'
# audio_ang = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD/raw_videos/M003/front/angry/level_3/M003_front_angry_level_3_001.wav'
# # exec(image_neu, audio_ang, out_dir=traget)         # 中性图 + 生气音
# exec_emo(image_neu, audio_ang, out_dir=traget)         # 中性图 + 生气音

# /mnt/disk2/zhouxishi/JoyVASA/animations/白人男__cyL5vt3pMc_2_0.mp4

image = '/mnt/disk2/zhouxishi/JoyVASA/myTest/image/白人男.jpg'

audio_angry = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/angry/level_3/M003_front_angry_level_3_001.wav'
audio_contempt = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/contempt/level_3/M003_front_contempt_level_3_001.wav'
audio_disgusted = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/disgusted/level_3/M003_front_disgusted_level_3_001.wav'
audio_fear = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/fear/level_3/M003_front_fear_level_3_001.wav'
audio_happy = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/happy/level_3/M003_front_happy_level_3_001.wav'
audio_neutral = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/neutral/level_1/M003_front_neutral_level_1_001.wav'
audio_sad = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/sad/level_3/M003_front_sad_level_3_001.wav'
audio_surprised = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD25/raw_videos/M003/front/surprised/level_3/M003_front_surprised_level_3_001.wav'

'''
0422一同训练_不增强_myAllevel
0422一同训练_一同推理_allevel
0422一同训练_一同推理_level3
0422一同训练_一同推理_myAllevel
0422_2Trans_level3
0422_enhance_emo_prevNo_allevel
'''

# 情感音频一一对应
def all_():
    # image = '/mnt/disk2/zhouxishi/JoyVASA/eval/11image/M003_front_neutral_level_1_014.jpg'
    # exp_name = '0425使用统一情感的模版/0425一同训练_不增强_myAllevel改版_leveloss'
    exp_name = '0512情感单独归一_' + '0511_allevel_120000'
    target = f'/mnt/disk2/zhouxishi/JoyVASA/测试效果/{exp_name}'
    exec_emo(image, audio_angry, out_dir=target, emotion='angry')
    exec_emo(image, audio_contempt, out_dir=target, emotion='contempt')
    exec_emo(image, audio_disgusted, out_dir=target, emotion='disgusted')
    exec_emo(image, audio_fear, out_dir=target, emotion='fear')
    exec_emo(image, audio_happy, out_dir=target, emotion='happy')
    exec_emo(image, audio_neutral, out_dir=target, emotion='neutral')
    exec_emo(image, audio_sad, out_dir=target, emotion='sad')
    exec_emo(image, audio_surprised, out_dir=target, emotion='surprised')

# 相同音频不同情感
def sAudio_dEmo_():
    exp_name = '0423_相同音频-中性-不同情感'
    target = f'/mnt/disk2/zhouxishi/JoyVASA/测试效果/{exp_name}'
    exec_emo(image, audio_neutral, out_dir=target+'/angry', emotion='angry')
    exec_emo(image, audio_neutral, out_dir=target+'/contempt', emotion='contempt')
    exec_emo(image, audio_neutral, out_dir=target+'/disgusted', emotion='disgusted')
    exec_emo(image, audio_neutral, out_dir=target+'/fear', emotion='fear')
    exec_emo(image, audio_neutral, out_dir=target+'/happy', emotion='happy')
    exec_emo(image, audio_neutral, out_dir=target+'/neutral', emotion='neutral')
    exec_emo(image, audio_neutral, out_dir=target+'/sad', emotion='sad')
    exec_emo(image, audio_neutral, out_dir=target+'/surprised', emotion='surprised')

# 不同音频相同情感
def dAudio_sEmo_():
    exp_name = '0423_不同音频相同情感-开心-'
    target = f'/mnt/disk2/zhouxishi/JoyVASA/测试效果/{exp_name}'
    exec_emo(image, audio_angry, out_dir=target, emotion='happy')
    exec_emo(image, audio_contempt, out_dir=target, emotion='happy')
    exec_emo(image, audio_disgusted, out_dir=target, emotion='happy')
    exec_emo(image, audio_fear, out_dir=target, emotion='happy')
    exec_emo(image, audio_happy, out_dir=target, emotion='happy')
    exec_emo(image, audio_neutral, out_dir=target, emotion='happy')
    exec_emo(image, audio_sad, out_dir=target, emotion='happy')
    exec_emo(image, audio_surprised, out_dir=target, emotion='happy')

# all_()
# sAudio_dEmo_()
# dAudio_sEmo_()
audio = '/mnt/disk2/zhouxishi/JoyVASA/myTest/audio/_cyL5vt3pMc_2_0.m4a'
exec_emo(image, audio, '.', 'surprised',False,1,True,1)

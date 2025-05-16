import subprocess

def exec_emo(image_path,audio_path,out_dir = '.', emotion='angry',use_emo_enhancer=False,enhance_level=1,use_emo_analyzer = False,device_id = 1):
    cmd = f'python inference.py -r {image_path} -a {audio_path} -e {emotion} --cfg_scale 1.5 --output_dir {out_dir} --use_emo_enhancer {use_emo_enhancer} --enhance_level {enhance_level} --use_emo_analyzer {use_emo_analyzer} --device_id {device_id}'
    subprocess.run(cmd,shell=True)

image = '/mnt/disk2/zhouxishi/JoyVASA/myTest/image/白人男.jpg'

audio = '/mnt/disk2/zhouxishi/JoyVASA/dataset/MEAD11/videos/M003/front/happy/level_3/M003_front_happy_level_3_001.wav'

exec_emo(image, audio, '.', 'happy',False,1,False,0)
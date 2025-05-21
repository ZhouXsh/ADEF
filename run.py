import subprocess

def exec_emo(image_path,audio_path,out_dir = '.', emotion='angry',use_emo_enhancer=False,enhance_level=1,use_emo_analyzer = False,device_id = 1):
    cmd = f'python inference.py -r {image_path} -a {audio_path} -e {emotion} --cfg_scale 1.5 2.0 --output_dir {out_dir} --use_emo_enhancer {use_emo_enhancer} --enhance_level {enhance_level} --use_emo_analyzer {use_emo_analyzer} --device_id {device_id}'
    subprocess.run(cmd,shell=True)

image = 'assets/examples/image/白人男.jpg'

audio = 'assets/examples/audio/_cyL5vt3pMc_2_0.m4a'

exec_emo(image, audio, '.', 'contempt',False,1,False,3)
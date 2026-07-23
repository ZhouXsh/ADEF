import os
import subprocess
import sys
import time

exam_name = '20260720_emotion_dit_Unification'

from src.config.emotion_config import global_emo_list
emo_list = global_emo_list
father = '/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake'
outdir = f'{father}/{exam_name}'
os.makedirs(outdir, exist_ok=True)


def exec_emo(image_path, audio_path, out_dir='.', emotion='angry',
             use_emo_enhancer=False, enhance_level=1,
             use_emo_analyzer=False, device_id=1, task_desc=''):
    cmd = [
        sys.executable, 'inference.py',
        '-r', image_path,
        '-a', audio_path,
        '-e', emotion,
        '--cfg_scale', '1.5', '2.0',
        '--output_dir', out_dir,
        '--device_id', str(device_id),
    ]
    print(f'\n{"="*60}')
    print(f'[START] {task_desc}')
    print(f'  image: {os.path.basename(image_path)}')
    print(f'  audio: {os.path.basename(audio_path)}')
    print(f'  emotion: {emotion}, device: {device_id}')
    print(f'{"="*60}')

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f'[FAIL] {task_desc} (returncode={result.returncode}, {elapsed:.1f}s)')
    else:
        print(f'[DONE] {task_desc} ({elapsed:.1f}s)')

    return result.returncode


def ouT():
    print('\n>>> ouT: cross-identity emotion transfer')
    image = 'assets/examples/image/白人男.jpg'
    for i in range(len(emo_list)):
        audio_angry = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/{emo_list[i]}/level_3/M003_front_{emo_list[i]}_level_3_001.wav'
        exec_emo(image, audio_angry, outdir, emo_list[i], False, 1, False, i % 2,
                 task_desc=f'ouT [{i+1}/{len(emo_list)}] {emo_list[i]}')
    audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/neutral/level_1/M003_front_neutral_level_1_001.wav'
    exec_emo(image, audio, outdir, 'neutral', False, 1, False, device_id=0,
             task_desc=f'ouT [{len(emo_list)+1}/{len(emo_list)+1}] neutral')


def inT():
    print('\n>>> inT: identity-preserving emotion transfer')
    for i in range(len(emo_list)):
        image = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/first_frame/M003_front_{emo_list[i]}_level_3_001.png'
        audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/{emo_list[i]}/level_3/M003_front_{emo_list[i]}_level_3_001.wav'
        exec_emo(image, audio, outdir, emo_list[i], False, 1, False, device_id=i % 2,
                 task_desc=f'inT [{i+1}/{len(emo_list)}] {emo_list[i]}')
    image = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/first_frame/M003_front_neutral_level_1_001.png'
    audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/neutral/level_1/M003_front_neutral_level_1_001.wav'
    exec_emo(image, audio, outdir, 'neutral', False, 1, False, device_id=0,
             task_desc=f'inT [{len(emo_list)+1}/{len(emo_list)+1}] neutral')


if __name__ == '__main__':
    print(f'exam: {exam_name}')
    print(f'output: {outdir}')
    print(f'emo_list: {emo_list}')
    total_start = time.time()

    ouT()
    inT()

    total = time.time() - total_start
    print(f'\nAll done in {total:.1f}s')

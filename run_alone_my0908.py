import argparse
import os
import subprocess
import sys
import time

exam_name = '20260905_ablation_cond_dit_adaln'

from src.config.emotion_config import global_emo_list
emo_list = global_emo_list
father = '/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake'
outdir = f'{father}/{exam_name}'
os.makedirs(outdir, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--motion_checkpoint', '--motion_ckpt',
        dest='motion_checkpoint',
        type=str,
        required=True,
        help='Motion-generator checkpoint fixed for this entire batch run.',
    )
    return parser.parse_args()


def exec_emo(image_path, audio_path, out_dir='.', emotion='angry',
             use_emo_enhancer=False, enhance_level=1,
             use_emo_analyzer=False, device_id=1, task_desc='',
             motion_checkpoint=None):
    if motion_checkpoint is None:
        raise ValueError('motion_checkpoint must be provided for batch inference')

    cmd = [
        sys.executable, 'inference.py',
        '-r', image_path,
        '-a', audio_path,
        '-e', emotion,
        '--cfg_scale', '2.0',
        '--output_dir', out_dir,
        '--device_id', str(device_id),
        '--motion_ckpt', motion_checkpoint,
    ]
    print(f'\n{"="*60}')
    print(f'[START] {task_desc}')
    print(f'  image: {os.path.basename(image_path)}')
    print(f'  audio: {os.path.basename(audio_path)}')
    print(f'  emotion: {emotion}, device: {device_id}')
    print(f'  motion checkpoint: {motion_checkpoint}')
    print(f'{"="*60}')

    start = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f'[FAIL] {task_desc} (returncode={result.returncode}, {elapsed:.1f}s)')
    else:
        print(f'[DONE] {task_desc} ({elapsed:.1f}s)')

    return result.returncode


def inT(motion_checkpoint):
    print('\n>>> inT: identity-preserving emotion transfer')
    for i in range(len(emo_list)):
        image = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/first_frame/M003_front_{emo_list[i]}_level_3_001.png'
        audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/{emo_list[i]}/level_3/M003_front_{emo_list[i]}_level_3_001.wav'
        exec_emo(image, audio, outdir, emo_list[i], False, 1, False, device_id=0,
                 task_desc=f'inT [{i+1}/{len(emo_list)}] {emo_list[i]}',
                 motion_checkpoint=motion_checkpoint)
    image = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/first_frame/M003_front_neutral_level_1_001.png'
    audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/neutral/level_1/M003_front_neutral_level_1_001.wav'
    exec_emo(image, audio, outdir, 'neutral', False, 1, False, device_id=0,
             task_desc=f'inT [{len(emo_list)+1}/{len(emo_list)+1}] neutral',
             motion_checkpoint=motion_checkpoint)


if __name__ == '__main__':
    args = parse_args()
    motion_checkpoint = os.path.abspath(os.path.expanduser(args.motion_checkpoint))
    if not os.path.isfile(motion_checkpoint):
        raise FileNotFoundError(f'motion checkpoint not found: {motion_checkpoint}')

    print(f'exam: {exam_name}')
    print(f'output: {outdir}')
    print(f'emo_list: {emo_list}')
    print(f'motion checkpoint: {motion_checkpoint}')
    total_start = time.time()

    inT(motion_checkpoint)

    total = time.time() - total_start
    print(f'\nAll done in {total:.1f}s')

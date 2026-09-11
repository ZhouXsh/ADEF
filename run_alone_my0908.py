import argparse
import os
import subprocess
import sys
import time

from src.config.emotion_config import global_emo_list

emo_list = global_emo_list
father = '/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake'
CHECKPOINT_ROOT = '/home/Zhouxishi/VirtualMan_proj/ADEF_remake/experiments/emo_dit'
CHECKPOINT_FILENAME = 'iter_0435000.pt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exam_name',
        type=str,
        required=True,
        help='Experiment name used to derive the checkpoint and output directory.',
    )
    parser.add_argument('--device_id', type=int, default=0,
                        help='GPU device id passed to inference.py (default: 0).')
    return parser.parse_args()


def build_motion_checkpoint(exam_name):
    return os.path.join(
        CHECKPOINT_ROOT,
        exam_name,
        'checkpoints',
        CHECKPOINT_FILENAME,
    )


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
        '--motion-checkpoint', motion_checkpoint,
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


def inT(outdir, motion_checkpoint, device_id=0):
    print('\n>>> inT: identity-preserving emotion transfer')
    for i in range(len(emo_list)):
        image = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/first_frame/M003_front_{emo_list[i]}_level_3_001.png'
        audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/{emo_list[i]}/level_3/M003_front_{emo_list[i]}_level_3_001.wav'
        exec_emo(image, audio, outdir, emo_list[i], False, 1, False, device_id=device_id,
                 task_desc=f'inT [{i+1}/{len(emo_list)}] {emo_list[i]}',
                 motion_checkpoint=motion_checkpoint)
    image = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/first_frame/M003_front_neutral_level_1_001.png'
    audio = f'/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/neutral/level_1/M003_front_neutral_level_1_001.wav'
    exec_emo(image, audio, outdir, 'neutral', False, 1, False, device_id=device_id,
             task_desc=f'inT [{len(emo_list)+1}/{len(emo_list)+1}] neutral',
             motion_checkpoint=motion_checkpoint)


if __name__ == '__main__':
    args = parse_args()
    exam_name = args.exam_name
    device_id = args.device_id
    outdir = os.path.join(father, exam_name)
    motion_checkpoint = build_motion_checkpoint(exam_name)

    if not os.path.isfile(motion_checkpoint):
        raise FileNotFoundError(f'motion checkpoint not found: {motion_checkpoint}')
    os.makedirs(outdir, exist_ok=True)

    print(f'exam: {exam_name}')
    print(f'output: {outdir}')
    print(f'emo_list: {emo_list}')
    print(f'device_id: {device_id}')
    print(f'motion checkpoint: {motion_checkpoint}')
    total_start = time.time()

    inT(outdir, motion_checkpoint, device_id=device_id)

    total = time.time() - total_start
    print(f'\nAll done in {total:.1f}s')

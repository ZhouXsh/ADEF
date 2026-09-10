import argparse
import os
import subprocess
import sys
import time

exam_name = '20260909_fusion_balanced_decay_ema_20cfg_full'

from src.config.emotion_config import global_emo_list
emo_list = global_emo_list
father = '/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake'
outdir = f'{father}/{exam_name}'
os.makedirs(outdir, exist_ok=True)

# Comma-separated triples of (reference_image, audio, gt_video).
# Emotion label is parsed from the audio filename (e.g.
# `M003_front_angry_level_3_001.wav` -> 'angry').
TRIPLES_FILE = '/home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/my_final_triples.txt'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--motion_checkpoint', '--motion_ckpt',
        dest='motion_checkpoint',
        type=str,
        required=True,
        help='Motion-generator checkpoint fixed for this entire batch run.',
    )
    parser.add_argument(
        '--triples_file',
        type=str,
        default=TRIPLES_FILE,
        help='Comma-separated (reference_image, audio, gt_video) list.',
    )
    parser.add_argument('--device_id', type=int, default=0)
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


def parse_emotion(audio_path):
    """Extract emotion label from an audio path like
    `.../M003/front/angry/level_3/M003_front_angry_level_3_001.wav` -> 'angry'.
    Filename pattern: <id>_front_<emotion>_level_<lvl>_<idx>.<ext>
    """
    stem = os.path.basename(audio_path).rsplit('.', 1)[0]
    parts = stem.split('_')
    # parts: ['M003', 'front', 'angry', 'level', '3', '001']
    if len(parts) < 6:
        raise ValueError(f'unexpected audio filename: {audio_path}')
    return parts[2]


def run_triples(triples_path=TRIPLES_FILE, device_id=0, motion_checkpoint=None):
    """Read each (image, audio, gt_video) line from `triples_path`, derive the
    emotion label from the audio filename, and run inference. Outputs land in
    `outdir`."""
    if motion_checkpoint is None:
        raise ValueError('motion_checkpoint must be provided for batch inference')

    print(f'\n>>> run_triples: {triples_path}')
    with open(triples_path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    total = len(lines)
    print(f'  total triples: {total}')
    success = 0
    for i, line in enumerate(lines, 1):
        parts = line.split(',')
        if len(parts) < 3:
            print(f'[SKIP] line {i}: expected 3 comma-separated paths, got {len(parts)} -> {line!r}')
            continue
        image_path, audio_path, gt_video_path = parts[0].strip(), parts[1].strip(), parts[2].strip()
        emotion = parse_emotion(audio_path)
        rc = exec_emo(image_path, audio_path, outdir, emotion, False, 1, False,
                      device_id=device_id,
                      task_desc=f'triples [{i}/{total}] {emotion} '
                                f'{os.path.basename(image_path)}',
                      motion_checkpoint=motion_checkpoint)
        if rc == 0:
            success += 1
    print(f'\n>>> run_triples done: {success}/{total} succeeded')


if __name__ == '__main__':
    args = parse_args()
    motion_checkpoint = os.path.abspath(os.path.expanduser(args.motion_checkpoint))
    triples_file = os.path.abspath(os.path.expanduser(args.triples_file))
    if not os.path.isfile(motion_checkpoint):
        raise FileNotFoundError(f'motion checkpoint not found: {motion_checkpoint}')
    if not os.path.isfile(triples_file):
        raise FileNotFoundError(f'triples file not found: {triples_file}')

    print(f'exam: {exam_name}')
    print(f'output: {outdir}')
    print(f'triples: {triples_file}')
    print(f'motion checkpoint: {motion_checkpoint}')
    total_start = time.time()

    run_triples(
        triples_path=triples_file,
        device_id=args.device_id,
        motion_checkpoint=motion_checkpoint,
    )

    total = time.time() - total_start
    print(f'\nAll done in {total:.1f}s')

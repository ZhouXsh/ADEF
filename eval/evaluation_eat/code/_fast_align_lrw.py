import glob
import os
import cv2
import argparse
import tqdm
import face_alignment
from scripts.align_68 import align_folder
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

frames_sav_rt = '../talking_head_testing/25fps_video/no_crop_frames'
vid_sav_rt    = '../talking_head_testing/25fps_video/pcavs_crop'
wav_rt        = '../talking_head_testing/lipread_test_wav'
# wav_rt        = ' /data3/vox/vox/mead/result/lipread_test_wav'


# CUDA_VISIBLE_DEVICES=0 python _fast_align_lrw.py --name LRW_vt2mel25_2_vox_head_555
# CUDA_VISIBLE_DEVICES=1 python _fast_align_lrw.py --name makeittalk
# CUDA_VISIBLE_DEVICES=2 python _fast_align_lrw.py --name atvg

# CUDA_VISIBLE_DEVICES=2 python _fast_align_lrw.py --name gt
# CUDA_VISIBLE_DEVICES=0 python _fast_align_lrw.py --name EAMM
# CUDA_VISIBLE_DEVICES=3 python _fast_align_lrw.py --name wav2lip
# CUDA_VISIBLE_DEVICES=0 python _fast_align_lrw.py --name audio2head
# CUDA_VISIBLE_DEVICES=1 python _fast_align_lrw.py --name AAAI22
# CUDA_VISIBLE_DEVICES=1 python _fast_align_lrw.py --name pcavs



def get_parser():
    parser = argparse.ArgumentParser('--')
    parser.add_argument('--name',type=str)
    parser.add_argument('--gpuid',type=int,default=0)
    return parser.parse_args()

if __name__ == '__main__':
    args = get_parser()
    name = args.name
    gpuid = args.gpuid


    frames_sav_dir = '{}/{}'.format(frames_sav_rt,name)
    vid_sav_dir = '{}/{}'.format(vid_sav_rt,name)
    os.makedirs(frames_sav_dir,exist_ok=True)
    os.makedirs(vid_sav_dir,exist_ok=True)

    vid_pths = glob.glob('../talking_head_testing/25fps_video/no_crop/{}/*.mp4'.format(name))


    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

    for vid_pth in tqdm.tqdm(vid_pths):
        vname = vid_pth.split('/')[-1].split('.')[0]
        sav_pth = '{}/{}.mp4'.format( vid_sav_dir , vname )

        if os.path.exists(sav_pth) : continue
        
        vid_frames_pth = '{}/{}'.format(frames_sav_dir,vname)
        os.makedirs(vid_frames_pth,exist_ok=True)
        
        # save frames
        vreader = cv2.VideoCapture(vid_pth)
        idx = 1
        while True :
            ret, frm = vreader.read()
            if not ret : break
            frame_save_path = vid_frames_pth + '/' + '%d.jpg' % idx
            cv2.imwrite( frame_save_path , frm )
            idx += 1
        
        # pcavs_crop
        align_folder(vid_frames_pth,fa=fa)
        # pcavs_align_cmd = f'CUDA_VISIBLE_DEVICES={gpuid} python scripts/align_68.py --folder_path {vid_frames_pth}'
        # print(pcavs_align_cmd)
        # os.system(pcavs_align_cmd)
        
        # gather into a video
        vid_frames_pth = vid_frames_pth + '_cropped'
        wav_pth = '{}/{}.wav'.format( wav_rt, vname )
        
        cmd = 'ffmpeg -loglevel error -f image2 -i {} -i {} -r 25 {} -y'.format( vid_frames_pth + '/%d.jpg' ,  wav_pth , sav_pth )
        # gather_video_cmds.append(cmd)
        os.system(cmd)
        # os.system(cmd)
        # assert(0)
    
        
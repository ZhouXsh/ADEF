##############################################################

# wav2lip 的 LSE-D 和 LSE-C    以单个视频为输入  已验证可行


cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Wav2Lip/evaluation
source venv/bin/activate
python eval_lipsync.py \
    --video /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --output_csv ./angry.csv \
    --output_json ./angry.json



cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Wav2Lip/evaluation
source venv/bin/activate
python eval_lipsync.py \
    --video /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_绝对运动/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --output_csv ./angry.csv \
    --output_json ./angry.json


cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Wav2Lip/evaluation
source venv/bin/activate
python eval_lipsync.py \
    --video /home/Zhouxishi/VirtualMan_proj/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --output_csv ./gt_angry.csv \
    --output_json ./gt_angry.json


python eval_lipsync.py \
    --video /home/Zhouxishi/VirtualMan_proj/dataset/HDTF_Processed/videos/RD_Radio4_000.mp4 \
    --output_csv ./gt_HDTF.csv \
    --output_json ./gt_HDTF.json

python eval_lipsync.py \
    --video /home/Zhouxishi/VirtualMan_proj/dataset/RAVDESS_Processed/videos/Actor01/front/angry/level_1/Actor01_front_angry_level_1_01_01_01.mp4 \
    --output_csv ./gt_RAVDESS.csv \
    --output_json ./gt_RAVDESS.json

python eval_lipsync.py \
    --video /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/001.mp4 \
    --output_csv ./重新下载的MEAD.csv \
    --output_json ./重新下载的MEAD.json

python eval_lipsync.py \
    --video /home/Zhouxishi/DATASET/MEAD/raw_videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --output_csv ./没处理的MEAD.csv \
    --output_json ./没处理的MEAD.json

##############################################################

# EAT 的评估函数

./venv/bin/python evaluate.py \
    --fake "/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4" \
    --gt   "/home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4" \
    --name bishe --device 0 --metrics all --auto-detect-name-mode --allow-all-pids



##############################################################

# FVD

conda activate fvd
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/frechet_video_distance
python evaluate_videos.py \
    --real_dir /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --fake_dir /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4

python evaluate_adef.py \
    --real_dir /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --fake_dir /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --video_length 80 \
    --output_file fvd_result.json


python evaluate_adef.py \
    --real_dir /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --fake_dir /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --video_length 16 \
    --pad_pairs_to_batch_size \
    --output_file fvd_result.json


##############################################################

# FID   已验证可行

cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/pytorch-fid

# 直接对比两个视频
.venv/bin/python evaluate_fid_video.py \
    --path1 /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --path2 /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 --output-json results.json


# 每隔 5 帧采 1 帧，最多 500 帧，并保存结果
.venv/bin/python evaluate_fid_video.py \
    --path1 /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --path2 /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --frame-stride 5 --max-frames 500 \
    --output-json results.json

##############################################################


# syncnet  已验证可行

cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/syncnet_python && \
source syncnet_venv/bin/activate && \
python evaluate_syncnet.py --mode pipeline \
    --videofile /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4 \
    --data_dir ./GT_Angry --reference test --overwrite

cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/syncnet_python && \
source syncnet_venv/bin/activate && \
python evaluate_syncnet.py --mode pipeline \
    --videofile /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --data_dir ./GT_Angry --reference test --overwrite

cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/syncnet_python && \
source syncnet_venv/bin/activate && \
python evaluate_syncnet.py --mode pipeline \
    --videofile /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_绝对运动/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --data_dir ./GT_Angry --reference test --overwrite

cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/syncnet_python && \
source syncnet_venv/bin/activate && \
python evaluate_syncnet.py --mode pipeline \
    --videofile /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260717_emotion_dit_SingleCFG_0717_youhua_绝对运动cfg_scale2/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --data_dir ./GT_Angry --reference test --overwrite

##############################################################


# EmoNet      可以单个视频情感分析，也可以两个视频情感逐帧对比  已验证可行
conda activate emonet
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/emonet

# 单视频 vs 单 GT 视频
python evaluate_emotion.py --gen /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 --gt /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3/M003_front_angry_level_3_001.mp4

# 单视频无 GT（仅输出每帧 EmoNet 预测）
python evaluate_emotion.py --gen /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4

##############################################################


# Emotion-FAN

conda activate emotion_fan
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Emotion-FAN
python evaluate_emotion_fan.py \
    --input /home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4 \
    --pretrain_fer ./pretrain_model/Resnet18_FER+_pytorch.pth.tar \
    --at_type 1 --device cuda:0
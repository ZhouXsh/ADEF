📖 评估脚本使用教程
快速上手

conda activate emonet
cd ~/VirtualMan_proj/ADEF_remake/eval/emonet
用法 1:单个生成视频 vs 单个 GT 视频(你最常用的)

python evaluate_emotion.py \
    --gen  /path/to/generated_video.mp4 \
    --gt   /path/to/ground_truth.mp4
文件名不需要一样,默认按输入顺序配对。

用法 2:单视频,无 GT(看生成视频的情绪分布)

python evaluate_emotion.py --gen /path/to/video.mp4
输出:每帧 emotion + valence + arousal,以及整体均值和类别直方图。

用法 3:批量评估(目录 vs 目录)

python evaluate_emotion.py \
    --gen /path/to/generated_dir \
    --gt  /path/to/ground_truth_dir
按顺序一一配对(扩展后排序)。如果文件数对不上会警告。

用法 4:用文件名 stem 配对(可选)

python evaluate_emotion.py --gen gen_dir --gt gt_dir --pair_mode stem
完整 CLI 参数
参数	默认值	说明
--gen	必填,可重复	生成视频文件路径或目录路径
--gt	可选,可重复	GT 视频文件路径或目录路径;不传则进入单视频模式
--nclasses	8	5 或 8
--device	cuda:0	推理设备
--batch_size	32	EmoNet 推理批大小
--pair_mode	positional	positional(按输入顺序)或 stem(按文件名)
--output	emonet_eval_results.json	结果保存路径
输出 JSON 结构
对比模式:


{
  "overall": {
    "n_videos": ..., "n_frames": ...,
    "emo_acc": ..., "emo_sim": ...,
    "valence": {"ccc": ..., "pcc": ..., "rmse": ..., "sagr": ...},
    "arousal": {"ccc": ..., "pcc": ..., "rmse": ..., "sagr": ...},
    "emo_acc_video_mean": ..., "emo_sim_video_mean": ...
  },
  "per_video": {
    "<gen视频名>": {
      "n_frames": ..., "emo_acc": ..., "emo_sim": ...,
      "valence": {...}, "arousal": {...}
    }
  },
  "config": {...}
}
单视频模式(无 overall 字段):


{
  "per_video": {
    "<视频名>": {
      "n_frames_with_face": ...,
      "frames": [{"frame_idx":..., "expression":..., "expression_name":..., "valence":..., "arousal":...}, ...],
      "mean_valence": ..., "mean_arousal": ...,
      "emotion_histogram": {"Anger": 41, "Disgust": 37, ...}
    }
  }
}



⚠️ 注意事项
1. /eval/ 是 gitignored 的,目录会被外部清理脚本删除/重建
我观察到 eval/ 下其它子目录也在频繁变动,且 git log 有 清理eval目录 提交。如果你重启工作流发现 emonet 目录又消失了,直接重新克隆:


cd ~/VirtualMan_proj/ADEF_remake/eval && git clone https://github.com/face-analysis/emonet.git
然后再把 evaluate_emotion.py 复制回去(我把它写在 eval/emonet/ 下,但如果需要把它放到 git 跟踪位置,告诉我我帮你挪)。

2. 配对逻辑
默认 positional:严格按下标配对,文件数必须相等(否则警告并按 min 长度截断)。
stem 模式:对目录批量友好,但要求两边文件名一致。
你之前看到 "no GT match" 警告,是因为脚本旧版本把"没配上"悄悄退回单视频模式。新版本会直接报错退出。
3. 情感类别定义(EmoNet 8 类)

0 Neutral | 1 Happy | 2 Sad | 3 Surprise
4 Fear    | 5 Disgust | 6 Anger | 7 Contempt
Anger(6) 与 Disgust(5) 在 EmoNet 中经常混淆(你的 angry 视频就跑出了 41 vs 37 的分布)。这是模型本身的局限。

4. Valence / Arousal 数值范围
连续值,理论范围 [-1, +1],脚本里已 .clamp(-1, 1)
Angry 类典型值:Valence ≈ -0.5,Arousal ≈ +0.5(你的视频是 -0.47 / +0.59,合理)
5. 帧对齐假设
脚本按帧索引对齐 gen / GT。假设两个视频帧数相同(或非常接近)。如果帧率/时长差异大,需要在外部先做时间对齐再传入。

6. 人脸检测
默认用 SFD 检测器(face_alignment 自带),只取第一张人脸
某些帧检测不到人脸时会跳过(frames_with_face 可能小于视频总帧数)
如果生成视频里脸部很糊 / 角度极端,检测失败率高
7. torch.load 的 weights_only 警告
首次跑会有一条 FutureWarning,是 torch 2.5 默认 weights_only=False 的提醒。官方权重是可信的,可以忽略。等以后 torch 默认改成 True 时再处理。

8. SFD 检测器权重
首次运行会自动从 https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth 下载 85MB,缓存在 ~/.cache/torch/hub/checkpoints/。
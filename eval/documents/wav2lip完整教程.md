第二部分：评估函数使用教程
1. 脚本定位
路径：/home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Wav2Lip/evaluation/eval_lipsync.py
依赖：同目录 venv/（已部署 Python 3.10.20 + torch+cu130 + opencv + librosa + scenedetect + python_speech_features + tqdm）
外部权重：../syncnet_python/data/syncnet_v2.model（自动定位，无需手动指定）
2. 激活环境

cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Wav2Lip/evaluation
source venv/bin/activate
3. 完整 CLI 参考

python eval_lipsync.py [-h]
    输入（至少选一个）:
      --video VIDEO           单个视频
      --videos V [V ...]      多个视频（shell 通配）
      --video_dir DIR         目录（含子目录一层）
      --filelist FILE         文本文件，每行一个视频路径
    输出:
      --output_csv CSV        写出每条记录 CSV
      --output_json JSON      写出汇总 JSON（含 aggregate）
      --save_frame_conf DIR   每帧置信度 .npy（可选，用于画曲线）
    模型:
      --syncnet_weights PATH  自定义权重，默认自动定位
      --vshift 15             音视频偏移搜索窗口（每侧帧数）
      --batch_size 20         SyncNet 推理 batch
      --device cuda|cuda:0|cpu
4. 典型用法
(a) ADEF 单次推理结果的快速检查

python eval_lipsync.py \
    --video /home/.../ADEF_remake/experiments/run_001/REF_AUDIO_angry.mp4
(b) 批量评估整个 results 目录

python eval_lipsync.py \
    --video_dir /home/.../ADEF_remake/experiments/run_001 \
    --output_csv  results/run_001_lipsync.csv \
    --output_json results/run_001_lipsync.json
(c) 按情感分组的对比
ADEF 输出命名形如 <ref>_<audio>_<emotion>.mp4，先用 mkdir 软链接分组：


mkdir -p grouped
ln -sf "$PWD"/experiments/run_001/*angry*.mp4 grouped/angry/
ln -sf "$PWD"/experiments/run_001/*happy*.mp4 grouped/happy/
ln -sf "$PWD"/experiments/run_001/*sad*.mp4   grouped/sad/

for emo in angry happy sad; do
    python eval_lipsync.py --video_dir grouped/$emo \
        --output_csv "grouped/${emo}_lipsync.csv"
done
(d) 把每帧置信度落盘用于画图

python eval_lipsync.py --video_dir results/run_001 \
    --save_frame_conf frame_conf/run_001/

# 之后用 matplotlib / numpy 加载
import numpy as np
data = np.load("frame_conf/run_001/REF_AUDIO_angry_conf.npy")
#   列: 0=frame_idx, 1=per-frame confidence, 2=per-frame min-distance
5. 输出文件结构
CSV（每行一条视频）：


video,lse_d,lse_c,av_offset,min_dist_raw,n_frames,duration_s,elapsed_s,error
/path/v1.mp4,11.332,0.221,15,4.543,120,4.8,0.90,
/path/v2.mp4,10.875,0.345,8,4.122,98,3.92,0.81,
JSON：


{
  "results": [
    {"video": "...", "lse_d": 11.332, "lse_c": 0.221, "av_offset": 15,
     "min_dist_raw": 4.543, "n_frames": 120, "duration_s": 4.8,
     "elapsed_s": 0.90, "error": null}
  ],
  "aggregate": {
    "n_total": 1, "n_success": 1, "n_failed": 0,
    "lse_d_mean": 11.332, "lse_d_std": 0,
    "lse_c_mean": 0.221, "lse_c_std": 0,
    "lse_c_min": 0.221, "lse_c_max": 0.221,
    "av_offset_mean": 15.0
  }
}
控制台表格：


==============================================================================
video                                               LSE-D    LSE-C  offset  frames
------------------------------------------------------------------------------
run1.mp4                                            11.332    0.221      15     120
run2.mp4                                            10.875    0.345       8      98
------------------------------------------------------------------------------
AGGREGATE  n=2/2  LSE-D = 11.104 ± 0.228  LSE-C = 0.283 ± 0.062  AV offset ≈ 11.50 frames
==============================================================================
第三部分：注意事项与已知坑
⚠️ 指标语义（与论文严格对齐）
指标	公式	取向	论文出处
LSE-D	dists.mean(axis=0).min()	越低越好	Wav2Lip / SyncNet 论文
LSE-C	median(mdist) - min(mdist)，其中 mdist = dists.mean(axis=0)	越高越好	同上
AV offset	最佳对齐的偏移（25 fps 帧数）	0 为完美对齐	同上
⚠️ 不要把 LSE-D 与 min(dists) 混淆！后者是全矩阵绝对最小，论文中不使用这个值。脚本里以 min_dist_raw 字段保留它，仅供诊断。

⚠️ 输入视频假设
假设	含义	不满足时的后果
25 fps	SyncNet 训练于 25 fps	ffmpeg 自动重采样到 25 fps，但建议生成时就用 25 fps
16 kHz 音轨	MFCC 用 16 kHz	ffmpeg 自动重采样
单张人脸为主	SyncNet 不做检测，直接吃全帧 resize 到 224×224	多脸 / 远景脸时分数不准，需先跑 syncnet_python/run_pipeline.py 做裁脸
人脸基本居中	全帧 resize 会丢失空间信息	若人脸不在中央，LSE 分数会被背景主导
对 ADEF_remake 的输出（talking-head 人脸居中且占帧主要面积）这些假设都成立。对电影片段或多人画面，需先跑 run_pipeline.py 做裁脸预处理。

⚠️ 性能与显存
单卡 RTX 4090：5 秒 25 fps 视频约 0.9 秒完成
长视频（>10 分钟）：显存够，单段约 1-2 秒
--batch_size 20 对 24 GB 显存安全；显存不够会自动降级（但本脚本未实现自动 fallback，需要手动调小）
⚠️ 输出格式
视频扩展名只识别：.mp4 .mov .mkv .avi .webm .flv .m4v
单视频失败不会中断批处理，错误写到 result.error，最终在控制台表格中显示 ERROR <原因>
单个失败不会写 output_csv 列，但脚本退出码为 2（全部成功时为 0），方便 CI 检查
⚠️ 与论文的可比对性
✅ 同一份 syncnet_v2.model 权重，与 Wav2Lip / SadTalker / MuseTalk 等论文中报告的 LSE-D/LSE-C 数值可直接比对
✅ vshift=15 是论文默认值（约 ±0.6 秒搜索窗口）
⚠️ 不同 vshift 会得到不同分数：搜索窗口越大，越可能找到更好的对齐，LSE-D 可能更低、LSE-C 可能更高
⚠️ 用 run_pipeline.py 裁脸后再喂 eval_lipsync.py，分数会有差异（人脸区域更纯净，背景干扰更少）
⚠️ 临时目录
每次运行在系统 /tmp 下创建 syncnet_eval_xxxxxx，结束后自动清理
如果 /tmp 空间不足，会失败；可用 TMPDIR=/path/to/big/disk 环境变量改路径
⚠️ 不要做的事
❌ 不要手动修改 syncnet_python/SyncNetInstance.py 里的 L79-86 行 resize 补丁 — 这是绕过官方裁脸流水线的关键
❌ 不要用 weights_only=False 加载旧权重 — 我们已经统一用 weights_only=True（torch ≥ 1.13 默认行为）
❌ 不要在 25 fps 之外的帧率下解读 AV offset 单位 — offset 永远是 25 fps 帧数
常用排错命令

# 1. 看完整错误堆栈
python eval_lipsync.py --video bad.mp4 2>&1 | grep -A 30 ERROR

# 2. 强制 CPU 排查显存问题
python eval_lipsync.py --video_dir results/ --device cpu

# 3. 验证 SyncNet 权重仍能加载
python -c "
import sys; sys.path.insert(0, '/home/.../ADEF_remake/eval/syncnet_python')
from SyncNetInstance import SyncNetInstance
s = SyncNetInstance(); s.loadParameters('/home/.../syncnet_python/data/syncnet_v2.model')
print('weights OK')
"

# 4. 单视频手工 sanity check
ffmpeg -i your_video.mp4 -ar 16000 -ac 1 -af apad test_16k.wav
ffmpeg -i your_video.mp4 -vf fps=25 test_25fps.mp4
# 再用 test_25fps.mp4 + test_16k.wav 跑 eval_lipsync.py


# 第四部分：快速参考卡片

部署完情况：
  Python  : 3.10.20  (venv)
  PyTorch : 2.11.0+cu130
  SyncNet : syncnet_v2.model (54.5 MB, 自动定位)
  SFD     : sfd_face.pth    (89.8 MB, 备用)
  
调用模板：
  cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/Wav2Lip/evaluation
  source venv/bin/activate
  python eval_lipsync.py \
      --video_dir <ADEF输出目录> \
      --output_csv <CSV报告> \
      --output_json <JSON报告>

期望数值范围（真实 talking-head）：
  LSE-D ≈  6.0 – 12.0   (越低越好)
  LSE-C ≈  1.0 – 8.0    (越高越好)
  AV offset ≈ -5 – +5   (越小越好，0 最佳)
# SyncNet 评估脚本 (evaluate_syncnet.py)

`evaluate_syncnet.py` 是一站式评估脚本,封装了 syncnet_python 仓库中三段式管线
(face detection → face tracking → SyncNet offset/confidence) 的常见用法,
并额外提供批量评估入口。

## 1. 激活虚拟环境

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/syncnet_python
source syncnet_venv/bin/activate
```

## 2. 三种评估模式

| 模式 | 输入 | 用途 |
| --- | --- | --- |
| `single` | 一个**已裁剪**的人脸视频 | 直接计算 AV offset / confidence / min dist |
| `batch` | 一个目录下多个**已裁剪**视频 | 批量计算,生成 JSON/CSV 报告 |
| `pipeline` | 一段原始视频 | 走完整管线:人脸检测+跟踪+裁剪+同步评估+可选可视化 |

### 2.1 single

```bash
python evaluate_syncnet.py --mode single \
    --videofile data/example.avi \
    --output data/example_result.json
```

输出:
```
AV offset: 3   Min dist: 5.348   Confidence: 10.081
```

`--reference` 参数控制中间临时目录命名 (默认 `demo`),多进程/多评估任务请用不同名字。

### 2.2 batch

```bash
python evaluate_syncnet.py --mode batch \
    --video_dir /path/to/cropped_videos \
    --output reports/batch_eval.csv
```

- 支持 `.mp4 .avi .mov .mkv .webm .flv`
- `--output` 以 `.csv` 结尾则写 CSV,否则写 JSON

### 2.3 pipeline (含人脸检测)

```bash
python evaluate_syncnet.py --mode pipeline \
    --videofile /path/to/raw_video.mp4 \
    --data_dir ./output \
    --reference my_video \
    --overwrite \
    --visualise
```

参数与 `run_pipeline.py` 对齐:`--facedet_scale`, `--crop_scale`, `--min_track`,
`--frame_rate`, `--num_failed_det`, `--min_face_size`。

`--visualise` 会额外生成 `video_out.avi`(带置信度框的可视化结果)。

## 3. 输出指标

`evaluate()` 的核心输出:
- `offset`:音频相对视频的偏移(以视频帧为单位,正值=音频滞后)
- `confidence`:越高代表同步越可靠
- `min_distance`:跨模态特征的最小欧氏距离(越低越好)

## 4. 输出文件

`pipeline` 模式会在 `$data_dir/` 下产出与原仓库一致的结构:

```
$DATA_DIR/
├── pyavi/$REFERENCE/video.avi, audio.wav, video_out.avi (--visualise 时)
├── pycrop/$REFERENCE/00000.avi, 00001.avi, ...
├── pyframes/$REFERENCE/000001.jpg, ...
└── pywork/$REFERENCE/{scene.pckl, faces.pckl, tracks.pckl, activesd.pckl}
```

## 5. 常见问题

- **CUDA OOM**:把 `--batch_size` 调小(例如 `10`)。
- **下载模型慢**:脚本默认指向 `data/syncnet_v2.model` 与 `detectors/s3fd/weights/sfd_face.pth`,
  若无模型可手动 `bash download_model.sh` 或 `wget` 到对应路径。
- **Python 3.13 兼容**:SyncNet 自带的代码已经在 3.13 环境下通过测试。

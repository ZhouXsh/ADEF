# pytorch-fid 评估部署说明

本目录是 FID (Fréchet Inception Distance) 评估工具的本地部署，用于评估生成模型的图像质量。

## 目录结构

```
pytorch-fid/
├── .venv/                        # Python 3.10 虚拟环境（含 torch 2.11.0+cu130）
├── evaluate_fid.py               # 主要的 FID 评估脚本
├── EVAL_README.md                # 本说明文档
├── models_cache/                 # Inception V3 FID 预训练权重副本
│   └── pt_inception-2015-12-05-6726825d.pth
├── src/pytorch_fid/              # pytorch-fid 源码（editable 安装）
└── ...
```

## 虚拟环境

- **路径**: `.venv/`
- **Python**: 3.10.20（来自 miniconda envs/eval）
- **PyTorch**: 2.11.0+cu130（通过 `--system-site-packages` 复用 eval 环境的 torch）
- **CUDA**: 支持，已自动检测

### 激活虚拟环境

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/pytorch-fid
source .venv/bin/activate
```

### 验证环境

```bash
.venv/bin/python -c "import torch; from pytorch_fid import fid_score, inception; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## 评估脚本用法

### 1. 基本用法：比较两个图像文件夹

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_images \
    --path2 /path/to/generated_images
```

### 2. 指定 GPU 和批量大小

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_images \
    --path2 /path/to/generated_images \
    --device cuda:0 \
    --batch-size 64
```

### 3. 缓存真实图像统计（节省时间）

只需要提取一次真实图像的统计，之后多次与不同的生成结果比较：

```bash
# 第一次：保存真实图像的统计
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_images \
    --path2 ./real_stats.npz \
    --save-stats

# 之后：用缓存的统计与多个生成结果比较
.venv/bin/python evaluate_fid.py \
    --path1 ./real_stats.npz \
    --path2 /path/to/generated_v1
```

### 4. 多模型对比（一次性评估多个候选）

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_images \
    --path2 /path/to/baseline \
    --multi /path/to/model_v1 /path/to/model_v2 /path/to/model_v3 \
    --output-json results.json
```

### 5. 选择不同的 Inception 特征维度

- `64` - 第一个 max pooling 特征
- `192` - 第二个 max pooling 特征
- `768` - 辅助分类器前特征
- `2048` - 最终平均池化特征（**默认，推荐**）

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real \
    --path2 /path/to/fake \
    --dims 2048
```

### 6. 保存结果为 JSON

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real \
    --path2 /path/to/fake \
    --output-json ./results/fid_eval.json
```

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--path1` | 必填 | 第一个图像文件夹路径或 `.npz` 统计文件 |
| `--path2` | 必填 | 第二个图像文件夹路径或 `.npz` 统计文件 |
| `--batch-size` | 50 | 提取 Inception 特征时的批量大小 |
| `--device` | 自动 | `cuda` / `cuda:0` / `cpu`，未指定则自动选择 |
| `--dims` | 2048 | 特征维度（64/192/768/2048）|
| `--num-workers` | 4 | DataLoader worker 数量 |
| `--save-stats` | False | 把 path1 的统计保存到 path2 |
| `--multi` | None | 一次比较多个候选目录 |
| `--output-json` | None | 把结果保存为 JSON |

## 输出说明

### 控制台输出示例

```
[config] device=cuda, dims=2048, batch_size=50
[config] path1=/path/to/real
[config] path2=/path/to/fake
[result] FID(/path/to/real vs /path/to/fake) = 23.456789  (took 12.34s)
```

### JSON 输出格式

```json
{
  "config": {
    "device": "cuda",
    "dims": 2048,
    "batch_size": 50,
    "num_workers": 4
  },
  "results": [
    {
      "path1": "/path/to/real",
      "path2": "/path/to/fake",
      "fid": 23.456789,
      "elapsed_sec": 12.34
    }
  ]
}
```

## FID 分数解读

- **FID = 0**：两组图像分布完全相同（实际中由于数值误差可能得到极小的负数）
- **FID 越小**：两组图像分布越接近，生成质量越高
- **FID > 50**：分布差异较大，生成质量较差
- 业界常用：高质量 GAN 通常 FID < 20，扩散模型通常 FID < 10

## 注意事项

1. **图像格式**：支持 `.bmp .jpg .jpeg .pgm .png .ppm .tif .tiff .webp`
2. **图像尺寸**：脚本会自动将图像 resize 到 299x299
3. **图像数量**：每组图像越多，统计越稳定；建议至少 1000+ 张以获得可靠结果
4. **批大小**：根据 GPU 显存调整；4090 推荐 50-128
5. **首次运行**：会从 GitHub 下载 Inception 权重（已预下载到 `~/.cache/torch/hub/checkpoints/`）

## 视频版评估脚本（evaluate_fid_video.py）

当输入是视频文件（`.mp4 / .avi / .mov / .mkv / .webm` 等）时，使用此脚本。
脚本会用 OpenCV 自动将视频抽帧到临时目录，计算 FID 后再删除临时目录。

### 用法示例

```bash
# 两个视频直接比较
.venv/bin/python evaluate_fid_video.py --path1 real.mp4 --path2 fake.mp4

# 视频 vs 帧目录
.venv/bin/python evaluate_fid_video.py --path1 real.mp4 --path2 /frames/fake

# 视频 vs 已缓存的 .npz 统计（推荐：真实视频只提取一次）
.venv/bin/python evaluate_fid_video.py --path1 real.mp4 --path2 real_stats.npz

# 抽帧参数：每隔 5 帧采 1 帧，最多 500 帧
.venv/bin/python evaluate_fid_video.py --path1 a.mp4 --path2 b.mp4 \
    --frame-stride 5 --max-frames 500

# 抽帧时统一 resize 到 256x256（短边缩放 + 中心裁剪）
.venv/bin/python evaluate_fid_video.py --path1 a.mp4 --path2 b.mp4 --resize 256

# 保存结构化结果
.venv/bin/python evaluate_fid_video.py --path1 a.mp4 --path2 b.mp4 \
    --output-json results.json
```

### 视频版新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--frame-stride` | 1 | 每隔 N 帧采一帧（视频版新增） |
| `--max-frames` | None | 单个视频最大采帧数 |
| `--resize` | None | 抽帧时 resize 短边到 N 并中心裁剪到 NxN |
| `--keep-temp` | False | 保留抽帧临时目录（默认计算完自动清理） |

输入自动识别规则：路径后缀在 `{mp4, avi, mov, mkv, webm, flv, wmv, m4v, mpeg, mpg}` 中 → 视为视频，其他视为目录或 `.npz`。

## 快速测试

```bash
# 已通过烟雾测试，验证模型加载、计算、缓存、JSON 输出均正常
# 随机生成的两组图像，FID 约为 75-230（差异越大分数越高，符合预期）

# 视频版测试示例（先生成测试视频）
ffmpeg -y -f lavfi -i "color=c=green:s=128x128:d=2:r=10" -pix_fmt yuv420p /tmp/real.mp4
ffmpeg -y -f lavfi -i "color=c=red:s=128x128:d=2:r=10" -pix_fmt yuv420p /tmp/fake.mp4
.venv/bin/python evaluate_fid_video.py --path1 /tmp/real.mp4 --path2 /tmp/fake.mp4
```

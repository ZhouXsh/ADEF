# pytorch-fid 评估完整使用教程

> 本教程涵盖本次部署的全部内容：环境搭建、模型部署、脚本使用、注意事项。

---

## 一、本次部署总结

### 1.1 完成的工作

| 步骤 | 内容 | 产出 |
|------|------|------|
| ① 创建虚拟环境 | Python 3.10 + torch 2.11.0+cu130 | `.venv/` (24MB) |
| ② 安装 pytorch-fid | editable 模式安装本地源码 | `src/pytorch_fid/` |
| ③ 部署 Inception V3 | 下载 FID 专用预训练权重（95.6MB）| `models_cache/pt_inception-2015-12-05-6726825d.pth`<br>`~/.cache/torch/hub/checkpoints/` |
| ④ 编写评估脚本 | 支持帧目录 / 视频文件 / .npz 统计 | `evaluate_fid.py`、`evaluate_fid_video.py` |

### 1.2 当前目录结构

```
pytorch-fid/
├── .venv/                  # 虚拟环境（Python 3.10 + torch 2.11.0+cu130）
├── evaluate_fid.py         # 帧目录版评估脚本
├── evaluate_fid_video.py   # 视频版评估脚本（自动抽帧）
├── EVAL_README.md          # 部署说明
├── USER_GUIDE.md           # 本教程
├── models_cache/           # Inception V3 FID 预训练权重副本
├── setup.py                # pytorch-fid 安装配置
└── src/pytorch_fid/        # pytorch-fid 源码（editable 安装）
```

### 1.3 环境信息

- **Python**: 3.10.20
- **PyTorch**: 2.11.0+cu130
- **CUDA**: 已检测 RTX 4090 可用
- **OpenCV**: 4.13.0（视频版脚本使用）
- **关键依赖**: numpy 2.2.5 / Pillow 12.2.0 / scipy 1.15.3

---

## 二、两个评估脚本对比

| 特性 | `evaluate_fid.py` | `evaluate_fid_video.py` |
|------|-------------------|--------------------------|
| 接受帧目录 | ✅ | ✅ |
| 接受视频文件 | ❌ | ✅ 自动抽帧 |
| 接受 `.npz` 统计 | ✅ | ✅ |
| 临时目录自动清理 | — | ✅ finally 保证清理 |
| `--save-stats` | ✅ | ❌（用 evaluate_fid.py） |
| `--multi` 多比较 | ✅ | ❌ |
| 帧采样参数 | — | ✅ stride/max-frames/resize |
| `--output-json` | ✅ | ✅ |

> **推荐工作流**：先用 `evaluate_fid.py --save-stats` 提取真实数据统计（只做一次），再用 `evaluate_fid_video.py` 把生成的视频和缓存的 `.npz` 比较。

---

## 三、evaluate_fid.py 使用教程（帧目录版）

### 3.1 最简单的用法

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/pytorch-fid
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_frames \
    --path2 /path/to/fake_frames
```

### 3.2 推荐：缓存真实数据统计

真实数据通常不变，提取一次统计后可反复与不同生成结果比较，节省大量时间：

```bash
# 第一次：把真实帧的统计保存到 .npz 文件
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_frames \
    --path2 ./real_stats.npz \
    --save-stats

# 之后：用缓存的统计与多个生成结果比较（只需提取生成端的特征）
.venv/bin/python evaluate_fid.py \
    --path1 ./real_stats.npz \
    --path2 /path/to/fake_v1

.venv/bin/python evaluate_fid.py \
    --path1 ./real_stats.npz \
    --path2 /path/to/fake_v2
```

### 3.3 多模型对比

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_frames \
    --path2 /path/to/baseline \
    --multi /path/to/model_v1 /path/to/model_v2 /path/to/model_v3 \
    --output-json ./results/multi_eval.json
```

### 3.4 GPU 与批量大小

```bash
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real \
    --path2 /path/to/fake \
    --device cuda:0 \
    --batch-size 64 \
    --num-workers 4
```

### 3.5 完整参数列表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--path1` | 必填 | 帧目录 / .npz 统计 |
| `--path2` | 必填 | 帧目录 / .npz 统计 |
| `--batch-size` | 50 | Inception 前向的批量大小 |
| `--device` | 自动 | cuda / cuda:0 / cpu |
| `--dims` | 2048 | 特征维度：64/192/768/2048 |
| `--num-workers` | 4 | DataLoader 进程数 |
| `--save-stats` | False | 把 path1 的统计存到 path2 |
| `--save-stats1` | None | 单独缓存 path1 的统计 |
| `--save-stats2` | None | 单独缓存 path2 的统计 |
| `--multi` | None | 一次比较多个候选目录 |
| `--output-json` | None | 保存结果为 JSON |

---

## 四、evaluate_fid_video.py 使用教程（视频版）

### 4.1 最简单的用法

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/pytorch-fid
.venv/bin/python evaluate_fid_video.py \
    --path1 /path/to/real_video.mp4 \
    --path2 /path/to/fake_video.mp4
```

脚本会：
1. 检测到两个输入都是视频
2. 自动抽帧到 `/tmp/fid_video_frames_xxx/`（两个独立临时目录）
3. 用 pytorch-fid 计算 FID
4. **自动删除**两个临时目录

### 4.2 视频 vs 帧目录（混合输入）

```bash
.venv/bin/python evaluate_fid_video.py \
    --path1 /path/to/real_video.mp4 \
    --path2 /path/to/extracted_frames/
```

### 4.3 视频 vs 缓存统计（推荐工作流）

先用 `evaluate_fid.py` 把真实帧目录（或真实视频抽帧后的目录）保存为 `.npz`：

```bash
# 用 evaluate_fid.py 缓存真实数据（必须先有帧目录）
.venv/bin/python evaluate_fid.py \
    --path1 /path/to/real_frames \
    --path2 ./real_stats.npz \
    --save-stats

# 然后用视频版脚本与 .npz 比较
.venv/bin/python evaluate_fid_video.py \
    --path1 ./real_stats.npz \
    --path2 /path/to/fake_video.mp4
```

### 4.4 帧采样控制

```bash
# 每隔 5 帧采 1 帧（视频 30fps 时相当于 6fps 采样）
.venv/bin/python evaluate_fid_video.py \
    --path1 a.mp4 --path2 b.mp4 \
    --frame-stride 5

# 限制最多采 500 帧（避免显存/速度问题）
.venv/bin/python evaluate_fid_video.py \
    --path1 a.mp4 --path2 b.mp4 \
    --max-frames 500

# 抽帧时 resize 到 256×256（短边缩放 + 中心裁剪）
.venv/bin/python evaluate_fid_video.py \
    --path1 a.mp4 --path2 b.mp4 \
    --resize 256
```

### 4.5 调试：保留临时目录

```bash
.venv/bin/python evaluate_fid_video.py \
    --path1 a.mp4 --path2 b.mp4 \
    --keep-temp
# 抽帧后保留在 /run/user/1007/mihomo-tmp/fid_video_frames_xxxxx/
```

### 4.6 视频版完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--path1` | 必填 | 视频 / 帧目录 / .npz |
| `--path2` | 必填 | 视频 / 帧目录 / .npz |
| `--batch-size` | 50 | Inception 批量 |
| `--device` | 自动 | cuda / cuda:0 / cpu |
| `--dims` | 2048 | 特征维度 |
| `--num-workers` | 4 | DataLoader 进程数 |
| `--frame-stride` | 1 | 每 N 帧采 1 帧 |
| `--max-frames` | None | 单视频最大采帧数 |
| `--resize` | None | 抽帧时 resize 到 NxN |
| `--keep-temp` | False | 保留临时目录（默认会清理）|
| `--output-json` | None | 保存结果为 JSON |

### 4.7 支持的视频格式

`.mp4 .avi .mov .mkv .webm .flv .wmv .m4v .mpeg .mpg`

后缀不在列表中的文件 → 视为帧目录或 `.npz`。

---

## 五、JSON 输出格式

```json
{
  "config": {
    "device": "cuda",
    "dims": 2048,
    "batch_size": 50,
    "num_workers": 4,
    "frame_stride": 1,
    "max_frames": null,
    "resize": null
  },
  "inputs": {
    "path1": {"raw": "...", "resolved": "...", "is_video": true},
    "path2": {"raw": "...", "resolved": "...", "is_video": true}
  },
  "fid": 23.456789,
  "elapsed_sec": 12.34
}
```

---

## 六、FID 分数解读

| FID 范围 | 含义 |
|---------|------|
| ≈ 0 | 分布几乎一致（同源对比通常 < 0.01）|
| < 10 | 极好（高质量扩散模型水平）|
| 10 ~ 30 | 良好（高质量 GAN 水平）|
| 30 ~ 100 | 一般（明显可分辨）|
| \> 100 | 较差（分布差异很大）|

⚠️ **注意**：FID 仅在相同 `--dims`、相同数据集、相同图像预处理下才可比较。论文里一般报 `dims=2048`。

---

## 七、注意事项与最佳实践

### 7.1 数据准备

1. **图像格式**：帧目录支持 `.bmp .jpg .jpeg .pgm .png .ppm .tif .tiff .webp`
2. **样本数量**：每组至少 1000+ 帧才能得到稳定统计；少于 2048 帧建议改用 `--dims 192` 或 `768`
3. **图像尺寸**：脚本内部统一 resize 到 299×299，无需手动处理
4. **彩色图像**：只支持 RGB，灰度图会自动转 3 通道

### 7.2 性能与显存

| GPU | 推荐 batch-size | 4090 单帧耗时 |
|-----|----------------|---------------|
| RTX 4090 (24GB) | 64 ~ 128 | ~10ms |
| RTX 3090 (24GB) | 32 ~ 64 | ~15ms |
| V100 (32GB) | 32 ~ 64 | ~25ms |

### 7.3 推荐工作流（重要）

```
1. 准备真实数据 → 抽帧到目录（或直接用视频）
2. 用 evaluate_fid.py --save-stats 一次性提取真实数据统计 → 存为 real_stats.npz
3. 对每个生成视频/帧目录，用 evaluate_fid_video.py 与 real_stats.npz 比较
4. 用 --output-json 汇总所有结果
```

理由：真实数据通常很大（几万帧），提取一次复用可节省 90%+ 时间。

### 7.4 命令行排错

- **路径含中文/空格** → 用双引号 `"..."` 包起来
- **长命令换行** → 每一行（最后一行除外）末尾必须有 `\`，且不能有多余空行
- **报错 `command not found`** → 续行符 `\` 缺失，shell 把后续行当成独立命令
- **报错 `Invalid path`** → 检查路径是否存在，是否被拼写错误

### 7.5 常见错误

| 错误信息 | 原因 | 解决 |
|---------|------|------|
| `RuntimeError: Invalid path` | 路径不存在或拼错 | 检查 `--path1` / `--path2` |
| `RuntimeError: --save-stats expects a directory` | 把视频当作目录传给 `--save-stats` | 先抽帧成目录 |
| `CUDA out of memory` | batch-size 太大 | 调小 `--batch-size` |
| `Connection timed out` (首次运行) | 下载 Inception 权重失败 | 手动下载到 `~/.cache/torch/hub/checkpoints/` |
| FID 出现 NaN | 样本数太少或分布退化 | 增加样本数，或加 eps |

### 7.6 不建议的事

- ❌ 不要跨数据集比较 FID（不同数据集的绝对值无意义）
- ❌ 不要在每轮训练时重新提取真实数据统计
- ❌ 不要把负的 FID 当成"比 0 更好"，那只是数值误差
- ❌ 不要混用 `--dims 64/192/768/2048` 的 FID 分数

---

## 八、实操示例

### 8.1 评估 MEAD 数据集生成结果

```bash
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/pytorch-fid
mkdir -p results

# 1) 缓存真实视频所在目录的统计
#    （前提：已经把真实视频抽帧到 frames_real/ 目录）
.venv/bin/python evaluate_fid.py \
    --path1 /home/Zhouxishi/VirtualMan_proj/ADEFv4/src/dataset/MEAD11/videos/M003/front/angry/level_3 \
    --path2 ./results/MEAD_M003_angry_level3_real_stats.npz \
    --batch-size 32 \
    --num-workers 4 \
    --save-stats

# 2) 把生成的视频与缓存统计比较
.venv/bin/python evaluate_fid_video.py \
    --path1 "./results/MEAD_M003_angry_level3_real_stats.npz" \
    --path2 "/home/Zhouxishi/VirtualMan_proj/ADEFv4_visual/ADEF_remake/20260630毕设去掉两层调制_修改train错误的loss累积_prev_modi/M003_front_angry_level_3_001_M003_front_angry_level_3_001_angry.mp4" \
    --batch-size 32 \
    --num-workers 4 \
    --frame-stride 2 \
    --output-json ./results/MEAD_M003_angry_level3_fid.json
```

### 8.2 批量评估多个生成视频

```bash
# 写一个小循环
for video in /path/to/generated/*.mp4; do
    name=$(basename "$video" .mp4)
    .venv/bin/python evaluate_fid_video.py \
        --path1 ./results/real_stats.npz \
        --path2 "$video" \
        --output-json "./results/${name}_fid.json"
done

# 汇总
python -c "
import json, glob
for f in sorted(glob.glob('./results/*_fid.json')):
    d = json.load(open(f))
    print(f'{f}: FID = {d[\"fid\"]:.4f}')
"
```

---

## 九、激活环境速查

```bash
# 激活虚拟环境
cd /home/Zhouxishi/VirtualMan_proj/ADEF_remake/eval/pytorch-fid
source .venv/bin/activate

# 或不激活直接调用
.venv/bin/python evaluate_fid.py --help
.venv/bin/python evaluate_fid_video.py --help

# 退出虚拟环境
deactivate
```

---

## 十、关键文件路径速查

| 文件 | 路径 |
|------|------|
| 虚拟环境 | `.venv/` |
| 评估脚本 1（帧目录） | `evaluate_fid.py` |
| 评估脚本 2（视频） | `evaluate_fid_video.py` |
| 部署说明 | `EVAL_README.md` |
| 本教程 | `USER_GUIDE.md`（当前文件） |
| Inception 权重（副本）| `models_cache/pt_inception-2015-12-05-6726825d.pth` |
| Inception 权重（实际使用）| `~/.cache/torch/hub/checkpoints/pt_inception-2015-12-05-6726825d.pth` |
| pytorch-fid 源码 | `src/pytorch_fid/` |

---

## 十一、版本信息

- 部署日期：2026-07-13
- pytorch-fid 版本：0.3.0
- PyTorch 版本：2.11.0+cu130
- Python 版本：3.10.20
- 主机 GPU：NVIDIA RTX 4090 × 4

> 如果未来 PyTorch/Inception 权重需要重新下载（断网等情况），可用国内镜像：
> `https://gh-proxy.com/https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth`
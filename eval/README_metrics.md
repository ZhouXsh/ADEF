# ADEF 论文评估协议（Paper Protocol v3）

统一评估引擎仍然是 `paper_evaluator.py`，正常入口不变：

```bash
python eval/final_evaluator.py --triples-file eval/my_triples.txt
python eval/ADEF_evaluator.py <exam_name> --pairs-file /path/to/pairs.txt
python eval/ADEF_all_evaluator.py --pairs-file /path/to/pairs.txt
```

## v3：失败样本不再导致整项清空

每个指标只使用该指标成功的样本计算最终值。成功样本数、状态、协议版本和 manifest fingerprint 属于运行元数据，不再写入 `paper_table.csv`，而是保留在 `paper_metrics.json` 和 ADEF 的内部 `summary.csv` 中。

每次评估额外生成：

```text
failed_samples.csv
```

逐条记录 `metric / name / fake / gt / error`，终端也会打印失败视频和原因。

状态含义：

- `complete`：所有请求指标对所有 eligible 样本均成功；
- `partial`：每个请求指标都有可用结果，但存在部分失败样本；表格数值已经按成功样本计算；
- `failed`：至少有一个请求指标一个可用 aggregate 都没有。

`ADEF_all_evaluator.py` 会把当前 v3 的 `complete` 和 `partial` 都视为已完成；只有 `failed` 会在下次自动重试。需要重跑 partial 时使用 `--include-done`。

## 指标协议

| 指标 | 方向 | v3 实现 |
|---|---:|---|
| LSE-D / LSE-C | ↓ / ↑ | 25 fps → S3FD track/crop → SyncNet-v2；逐视频评分，成功视频求均值 |
| FID | ↓ | 成功 pair 的所有帧汇成 real/fake 两个分布，只计算一次 pytorch-fid |
| FVD | ↓ | 成功 pair 均匀采样固定帧数 → Google I3D，完整成功视频分布只计算一次 FVD |
| PSNR / SSIM | ↑ | EAT `utils_crop_psnr.crop_and_align` + temporal linspace pairing；有效帧全局均值 |
| LPIPS | ↓ | 与 PSNR/SSIM 相同的 EAT 对齐帧对；官方 `lpips` AlexNet |
| M-LMD / F-LMD | ↓ | EAT `utils_crop` + dlib-68；嘴 20 点 / 全脸 68 点；先视频均值再数据集均值 |
| EmotiEff-Acc | ↑ | 8 类目标情感 accuracy；无有效主导情感的视频记为失败并排除分母 |
| DFER-CLIP-Acc | ↑ | DFEW fold-1 七类 accuracy；`contempt` 明确排除 |

### FID/FVD 的部分失败处理

FID/FVD 是分布级指标，不能逐视频计算后再平均。如果某个 fake 或对应 GT 无法读取，v3 会把该 sample 的 **real 与 fake 两侧同时剔除**，然后在剩余成功 sample 的完整分布上只计算一次 FID/FVD。

### Pairwise 的部分失败处理

PSNR/SSIM/LPIPS 和 LMD 分开记录成功覆盖。例如某视频 EAT pixel alignment 成功但 dlib landmarks 全部失败，它仍会进入 PSNR/SSIM/LPIPS，但不会进入 M-LMD/F-LMD。

## 输出

每个方法/实验目录包含：

```text
paper_table.csv       # 论文展示表：仅 Method + 评估指标值
paper_metrics.json    # 完整协议、status、coverage、aggregate、failures、子进程信息
per_video.csv         # 每视频结果
failed_samples.csv    # 失败样本和原因
work/                 # 各 evaluator 中间 JSON/manifest
```

`paper_table.csv` 当前列固定为：

```text
Method,
LSE-D,LSE-C,
FID,FVD,
PSNR,SSIM,LPIPS,
M-LMD,F-LMD,
EmotiEff-Acc,DFER-CLIP-Acc
```

以下信息不再写入论文表：

```text
Status
N
Evaluated-N
LSE-N / FID-N / FVD-N
PSNR-N / SSIM-N / LPIPS-N / LMD-N
EmotiEff-N / DFER-N
Protocol
Manifest-SHA256
```

这些运行元数据仍完整保留在 `paper_metrics.json`；ADEF 批量续跑所需的 `Status / Protocol` 等信息保留在内部 `summary.csv`。因此精简论文表不会影响 `ADEF_all_evaluator.py` 判断哪些实验需要重跑。

## 运行前检查

```bash
python eval/check_paper_eval.py --deep-hash
```

v3 preflight 除了检查权重/源码文件，还会使用各 evaluator 实际 Python 环境检查关键 import，包括 pairwise 环境的 `skimage/dlib/imutils/lpips`，避免长批处理运行到第一个实验才发现依赖缺失。

当前 `eval/evaluation_eat/` 和 `eval/emonet/` 是 vendored 普通目录，不需要 `git submodule update`。

EAT predictor 可放在：

```text
eval/evaluation_eat/code/shape_predictor_68_face_landmarks.dat
eval/evaluation_eat/checkpoints/shape_predictor_68_face_landmarks.dat
```

EmoNet 仍不是默认论文主表指标；单独使用时权重路径为：

```text
eval/emonet/pretrained/emonet_8.pth
```

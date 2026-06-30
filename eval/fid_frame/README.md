# 帧级 FID

帧级 FID 用于比较生成帧与真实 / 参考帧之间的图像分布差异。它适合评估分布级视觉质量，但不会建模时间动态。

该封装支持两种模式：

1. 如果安装了 `pytorch-fid`，可直接调用其命令行实现；
2. 使用本地轻量实现，基于 `torchvision.models.inception_v3` 提取特征并计算 Frechet distance。

## 准备帧文件夹

你需要准备两个图像文件夹：

```text
real_frames/
  000001.png
  ...
gen_frames/
  000001.png
  ...
```

可以用 ffmpeg 或你自己的脚本抽帧。不同方法之间必须保持一致的抽帧策略。

## 用法

```bash
python eval/fid_frame/eval_fid_frame.py \
  --real_dir path/to/real_frames \
  --gen_dir path/to/gen_frames \
  --out eval_results/fid_frame.json
```

如果已安装 `pytorch-fid`，可以强制使用外部模式：

```bash
python eval/fid_frame/eval_fid_frame.py \
  --real_dir real_frames --gen_dir gen_frames \
  --use_pytorch_fid \
  --out fid.json
```

## 注意事项

FID 在样本量很小时偏差较大。请使用足够数量的帧，并记录抽帧协议。
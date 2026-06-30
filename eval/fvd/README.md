# Frechet Video Distance / 视频分布质量

FVD 是生成视频领域常用的分布级指标。它通常使用 I3D 特征，并计算生成视频特征分布与真实视频特征分布之间的 Frechet distance。

本目录提供的是封装器，而不是直接复制某个 I3D 实现。你可以将它接入本地的 FVD 实现，例如 PyTorch / TensorFlow 版 FVD 仓库，或安装提供 FVD 命令行接口的包。

## 使用显式外部命令

```bash
python eval/fvd/eval_fvd.py \
  --real_dir path/to/real_videos \
  --gen_dir path/to/generated_videos \
  --external_cmd "python third_party/fvd/eval.py --real {real_dir} --fake {gen_dir} --out {out}" \
  --out eval_results/fvd.json
```

外部命令应写出 JSON，或在文本输出中包含数值型 `fvd` 字段。

## 为什么采用外部封装

FVD 对 I3D checkpoint、预处理、clip 长度和帧率非常敏感。为了保证实验可复现，最好显式指定你选择的官方 / 第三方实现，并记录其 commit。

## 注意事项

FVD 在样本量很小时不可靠。请同时汇报视频数量、帧采样方式、分辨率和 I3D checkpoint。
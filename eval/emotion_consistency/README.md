# 情感一致性

该评估器用于衡量生成帧是否表达了指定的目标情感标签。

由于不同数据集上的情感识别模型差异较大，本目录提供了一个通用封装，支持以下两种方式：

1. HuggingFace 图像分类模型；或
2. 自定义外部命令。

对于 ADEF，如果你有项目专用的情感分类器，并且它可以接收裁剪后的人脸帧，建议使用 `--external_cmd` 接入，并在所有实验中固定同一个模型。

## 使用 HuggingFace 图像分类器

```bash
python eval/emotion_consistency/eval_emotion_consistency.py \
  --manifest generated.csv \
  --hf_model your/emotion-classifier \
  --out eval_results/emotion_consistency.json
```

`generated.csv` 示例：

```csv
generated,label
/path/to/gen_happy.mp4,happy
```

## 使用外部命令

```bash
python eval/emotion_consistency/eval_emotion_consistency.py \
  --manifest generated.csv \
  --external_cmd "python my_eval.py --video {video} --label {label} --out {out}" \
  --out eval_results/emotion_consistency.json
```

外部命令应写出 JSON，或在文本中包含置信度 / 准确率信息。

## 输出字段

- `target_confidence_mean`：目标情感的平均置信度。
- `target_top1_ratio`：采样帧被预测为目标情感的比例。
- `emotion_entropy_mean`：时间维度情感不确定性的代理指标。
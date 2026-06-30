# 基于 ArcFace / InsightFace 的身份保持评估

该评估器用于衡量生成的说话人脸视频是否保持了输入身份。

它使用 `insightface`。该依赖已经列在 ADEF 的 requirements 中。对于每个视频，脚本会采样若干帧、提取人脸 embedding，并与以下目标进行比较：

- manifest 中提供的参考视频或参考图像；或
- 如果没有提供参考，则使用生成视频中第一帧成功检测到的人脸作为自一致性参考。

## 用法

基于参考图 / 参考视频：

```bash
python eval/identity_arcface/eval_identity_arcface.py \
  --manifest generated.csv \
  --out eval_results/identity_arcface.json
```

`generated.csv` 示例：

```csv
generated,reference
/path/to/gen.mp4,/path/to/source.png
```

单视频自一致性：

```bash
python eval/identity_arcface/eval_identity_arcface.py \
  --video path/to/generated.mp4 \
  --out eval_results/identity_self.json
```

## 输出字段

- `identity_cosine_mean`：生成帧与参考 embedding 的平均 cosine similarity。
- `identity_cosine_std`：身份在时间维度上的稳定性。
- `detected_frames`：成功检测到人脸的帧数。

cosine similarity 越高，通常表示身份保持越好。
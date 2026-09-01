# New_Emo — auxiliary emotion evaluators

本目录保留两个独立情感分类器：

- **EmotiEffLib**：AffectNet 8 类，可覆盖 MEAD 的 `contempt`；论文主表使用 target correctness / accuracy。
- **DFER-CLIP (BMVC 2023)**：DFEW 7 类，不含 `contempt`；unsupported label 必须排除 accuracy 分母并报告有效样本数。

论文统一入口请使用上一级 `paper_evaluator.py`，不要把 `dominant_fraction` 或两个分类器之间的 `agreement` 当作情感正确率。

## DFER-CLIP 配置

仓库中的 wrapper 与 release 训练配置保持一致：ViT-B/32、16 segments、8 context tokens、class-specific contexts、`class_descriptor`、1 temporal layer。

OpenAI 官方 ViT-B/32 权重 URL 为：

```text
https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```

注意：此前 README 中以 `be1cfb55...` 开头的 URL 实际对应 **RN50x64**，不是 ViT-B/32。

DFER-CLIP 的 DFEW checkpoint 请按上游仓库 `zengqunzhao/DFER-CLIP` 的说明下载并放到 `weights/DFEW_fold1.pth`。

## EmotiEffLib

安装：

```bash
pip install 'emotiefflib[torch]' facenet-pytorch
```

默认 evaluator 使用 face detection 后的视频级主导分类，并把目标标签统一成 AffectNet 的 8 类语义名称。主表使用 `accuracy`；`dominant_fraction`、face detection rate 等仅保留为诊断信息。

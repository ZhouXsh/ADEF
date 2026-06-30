# SyncNet / LSE 唇音同步评估

该封装用于 Wav2Lip 和许多说话人脸论文中常见的标准唇音同步指标：

- LSE-D：通常越低越好；
- LSE-C：通常越高越好。

Wav2Lip 论文提出了用于无约束视频的强唇音同步判别器 / 评估器，并开源了代码和模型。本目录不会把官方仓库代码直接复制进来，而是提供一个安全封装，使 ADEF 能够调用你本地克隆的官方仓库或兼容实现。

## 安装准备

请先克隆或安装官方 / 兼容的 SyncNet 实现，例如 Wav2Lip：

```bash
git clone https://github.com/Rudrabha/Wav2Lip third_party/Wav2Lip
```

然后按照官方说明下载 SyncNet expert checkpoint。

## 用法

```bash
python eval/sync_lse/eval_sync_lse.py \
  --manifest generated.csv \
  --wav2lip_root third_party/Wav2Lip \
  --syncnet_checkpoint path/to/lipsync_expert.pth \
  --out eval_results/sync_lse.json
```

`generated.csv` 至少应包含：

```csv
generated,audio
/path/to/gen.mp4,/path/to/audio.wav
```

如果 `audio` 为空，是否能从视频中自动提取音频取决于你接入的外部实现。

## 说明

不同 SyncNet fork 暴露的命令行参数并不一致。本封装支持两种模式：

1. `--external_cmd`：显式指定命令模板，并使用 `{video}`、`{audio}`、`{checkpoint}`、`{out}` 占位符。
2. `--wav2lip_root`：保留给常见 Wav2Lip 评估脚本的路径说明；实际推荐使用 `--external_cmd` 精确指定命令。

正式汇报时，请记录所使用 SyncNet 仓库的具体 commit 和 checkpoint 路径。
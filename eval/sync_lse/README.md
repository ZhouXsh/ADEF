# SyncNet / LSE lip-sync evaluation

This wrapper is for the standard lip-sync metrics used by Wav2Lip and many talking-head papers:

- LSE-D: lower is usually better;
- LSE-C: higher is usually better.

The official Wav2Lip paper introduced a strong lip-sync discriminator/evaluator for unconstrained videos and released code/models. This directory does not vendor the official repository. Instead, it provides a safe wrapper so you can point ADEF to a local clone/checkpoint.

## Setup

Clone or install an official/compatible SyncNet implementation, for example Wav2Lip:

```bash
git clone https://github.com/Rudrabha/Wav2Lip third_party/Wav2Lip
```

Download the SyncNet expert checkpoint following the official instructions.

## Usage

```bash
python eval/sync_lse/eval_sync_lse.py \
  --manifest generated.csv \
  --wav2lip_root third_party/Wav2Lip \
  --syncnet_checkpoint path/to/lipsync_expert.pth \
  --out eval_results/sync_lse.json
```

`generated.csv` should contain at least:

```csv
generated,audio
/path/to/gen.mp4,/path/to/audio.wav
```

If `audio` is empty, this wrapper lets the external script extract audio from the video if the implementation supports it.

## Notes

Different SyncNet forks expose different CLI names. This wrapper supports two modes:

1. `--external_cmd`: explicit command template with `{video}`, `{audio}`, `{checkpoint}`, `{out}` placeholders.
2. `--wav2lip_root`: best-effort call into common Wav2Lip evaluation scripts.

For rigorous reporting, record the exact SyncNet repo commit and checkpoint path.

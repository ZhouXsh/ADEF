# 0904 Fusion variants

Six complete independent training scripts branch from the completed 6009 recipe. No model architecture change and no Generic replay are introduced.

Recommended branch checkpoint:
`experiments/emo_dit/20260901_twostage_sharedcond_minsnr_balanced_ema/checkpoints/iter_0350000.pt`

| GPU | Script | Min-SNR final | Balance final | Audio after 400k |
|---|---|---:|---:|---|
|0|`train_Unification_twostage0904_fusion_conservative_ema.py`|0.25|0.50|normal|
|1|`train_Unification_twostage0904_fusion_plain_tail_ema.py`|0.00|0.50|normal|
|2|`train_Unification_twostage0904_fusion_balanced_decay_ema.py`|0.25|0.25|normal|
|3|`train_Unification_twostage0904_fusion_natural_tail_ema.py`|0.00|0.00|normal|
|4|`train_Unification_twostage0904_fusion_lowaudio_protect_ema.py`|0.25|0.25|LR <= 2e-7|
|5|`train_Unification_twostage0904_fusion_freezeaudio_protect_ema.py`|0.25|0.25|LR = 0|

Loss mix anneals 350k-420k; sampling power anneals 380k-430k; all run to 450k. Evaluate 400k/420k/430k/440k/450k, not training loss alone. Variant GPU4 is the current first-choice hypothesis.

```bash
BASE=experiments/emo_dit/20260901_twostage_sharedcond_minsnr_balanced_ema/checkpoints/iter_0350000.pt
python train_Unification_twostage0904_fusion_conservative_ema.py --device_id 0 --resume_checkpoint "$BASE"
python train_Unification_twostage0904_fusion_plain_tail_ema.py --device_id 1 --resume_checkpoint "$BASE"
python train_Unification_twostage0904_fusion_balanced_decay_ema.py --device_id 2 --resume_checkpoint "$BASE"
python train_Unification_twostage0904_fusion_natural_tail_ema.py --device_id 3 --resume_checkpoint "$BASE"
python train_Unification_twostage0904_fusion_lowaudio_protect_ema.py --device_id 4 --resume_checkpoint "$BASE"
python train_Unification_twostage0904_fusion_freezeaudio_protect_ema.py --device_id 5 --resume_checkpoint "$BASE"
```

Do not add `--resume_optimizer` when branching from an ordinary `iter_*.pt`; those checkpoints do not contain Adam state. The option exists for compatible `latest_train_state.pt` resumes.

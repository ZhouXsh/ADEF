# 0904 Fusion variants — reduced experiment set

This round intentionally keeps only **two necessary experiments** instead of filling all six available GPUs.

The previous completed runs already form an unusually informative 2x2 training-factor comparison:

- 6008: shared condition, no Min-SNR, no balanced MEAD — strongest spatial / landmark quality.
- 6007: shared condition, no Min-SNR, balanced MEAD — strongest lip-sync / FID side of the trade-off.
- 6010: shared condition, Min-SNR, no balanced MEAD — strongest FVD side of the trade-off.
- 6009: shared condition, Min-SNR, balanced MEAD — best overall Pareto starting point.
- Generic replay did not provide enough benefit to justify keeping it in this round.

Because the four core runs already isolate the two main factors (Min-SNR and balanced sampling), there is no reason to spend four additional GPUs on redundant endpoint combinations or speculative audio freezing. The only remaining question worth testing is whether a **late continuous transition away from the 6009 objective** can keep its temporal/sync strengths while recovering part of 6008's spatial quality.

## Recommended branch checkpoint

`experiments/emo_dit/20260901_twostage_sharedcond_minsnr_balanced_ema/checkpoints/iter_0350000.pt`

Both experiments resume from 6009 at 350k and continue to 450k. They do not reload the Generic stage.

## Experiment A — loss-only annealing control

`train_Unification_twostage0904_fusion_plain_tail_ema.py`

Purpose: isolate whether the spatial-quality loss of 6009 is mainly caused by keeping Min-SNR throughout Phase 3.

- Min-SNR mix: cosine 1.0 -> 0.0 from 350k to 420k.
- Balanced MEAD power: stays at 0.50.
- Audio / backbone learning-rate policy: unchanged from the corrected 0901 schedule.

This is the necessary control experiment. If spatial metrics recover while LSE remains strong, then Min-SNR persistence was the dominant cause of the 6009 trade-off.

## Experiment B — joint Pareto annealing

`train_Unification_twostage0904_fusion_balanced_decay_ema.py`

Purpose: the main candidate for the final model.

- Min-SNR mix: cosine 1.0 -> 0.25 from 350k to 420k.
- Balanced MEAD power: cosine 0.50 -> 0.25 from 380k to 430k.
- Audio / backbone learning-rate policy: unchanged from the corrected 0901 schedule.

This keeps a residual Min-SNR contribution for temporal robustness and a residual balancing contribution for emotion/lip-sync coverage, while relaxing both late enough to allow spatial refinement.

## Why the other four 0904 variants were removed

- `fusion_conservative_ema`: too close to 6009 and adds little information beyond Experiment A/B.
- `fusion_natural_tail_ema`: changes both factors all the way to zero, which is too aggressive and largely revisits the already-observed 6008 endpoint.
- `fusion_lowaudio_protect_ema`: audio-LR protection was a new speculative factor not supported by the completed six-run evidence.
- `fusion_freezeaudio_protect_ema`: hard audio freezing is an even stronger speculative intervention and is unnecessary before proving that audio drift is actually a problem.

## Parallel commands

```bash
BASE=experiments/emo_dit/20260901_twostage_sharedcond_minsnr_balanced_ema/checkpoints/iter_0350000.pt

python train_Unification_twostage0904_fusion_plain_tail_ema.py \
  --device_id 0 --resume_checkpoint "$BASE"

python train_Unification_twostage0904_fusion_balanced_decay_ema.py \
  --device_id 1 --resume_checkpoint "$BASE"
```

Do not add `--resume_optimizer` when branching from the ordinary `iter_0350000.pt`; these tail experiments intentionally start with fresh Adam state.

## Evaluation checkpoints

Evaluate at least 400k, 420k, 430k, 440k, and 450k with exactly the same evaluation pipeline used for 6007-6011.

Do not choose by training loss alone because the loss definition changes continuously during annealing. The final choice should be based on the same external metrics, with emphasis on whether Experiment B can retain 6009/6010-level FVD and 6007-level lip-sync while recovering toward 6008 on PSNR / SSIM / LPIPS / M-LMD.

If Experiment B dominates or nearly dominates the old Pareto frontier, stop there. Only if its result exposes a clear remaining failure mode should another GPU round be designed.
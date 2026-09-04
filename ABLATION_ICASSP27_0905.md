# ICASSP27 controlled ablations — 2026-09-05

## Final reference

The paper reference method is `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` (6009).
Every ablation below is a **physical training-file copy** and has a dedicated **physical model + legacy-model copy**. No ablation imports another ablation. This makes checkpoint/model provenance explicit and prevents nested variant dependencies.

## Why these six new runs

After re-checking the paper claims, only six additional controlled runs are necessary. The existing 6006/6008/6007/6010/6009 runs already isolate shared-prior warm-start, balanced MEAD sampling, and Min-SNR. Generic replay (6011) is diagnostic and is not part of the final method. Radius-0/global attention is not a headline contribution (the manuscript explicitly treats the local/global temporal design as architectural context), so it should not consume main ablation budget unless reviewers request it.

| ID | Training file | Scientific question | Primary metrics |
|---|---|---|---|
| `cond_additive` | `train_Ablation0905_cond_additive.py` | Conditioning control: replace pre-audio affine reparameterization with an additive target bias while retaining the final 6009 training recipe. | T-UAR, leakage, target-source margin, LSE-C |
| `cond_dit_adaln` | `train_Ablation0905_cond_dit_adaln.py` | Conditioning-placement control: keep acoustic tokens unmodulated and apply the same target affine inside every DiT block. | T-UAR, leakage, target-source margin, LSE-C |
| `motion_partition` | `train_Ablation0905_motion_partition.py` | Holistic-generation control: force a coordinate pseudo-partition between speech-only and target-conditioned predictions with shared parameters. | T-UAR, leakage, FVD, LSE-C |
| `emotion_residual` | `train_Ablation0905_emotion_residual.py` | Holistic-generation control: predict a category-agnostic speech trajectory and add an independently learned category motion residual. | T-UAR, leakage, FVD, LSE-C |
| `single_stage_mead` | `train_Ablation0905_single_stage_mead.py` | Training-schedule control: remove Generic Stage 1 and train only the 5k specialization transition plus 240k full MEAD updates. | FVD, LSE-C, T-UAR, cross-dataset generalization |
| `no_emotion_ce` | `train_Ablation0905_no_emotion_ce.py` | Regularization control: retain the final 6009 architecture and training schedule but set the frozen motion-classifier CE weight to zero. | T-UAR, leakage, target-source margin, LSE-C |

## Recommended paper comparisons

### Conditioning placement/operator

Compare the final 6009 model against:

- `cond_additive`: target information is an additive bias on frame-aligned acoustic tokens.
- `cond_dit_adaln`: acoustic tokens remain category-agnostic; the target affine is applied inside DiT blocks.

This directly tests whether **pre-motion affective acoustic reparameterization** is preferable to a simpler additive operator and to moving the affine condition into the motion backbone. Use the 8x8 source-to-target counterfactual protocol; matched MEAD emotion accuracy alone is not sufficient.

### Holistic generation vs forced factorization

Compare the final 6009 model against:

- `motion_partition`: the same denoiser parameters are evaluated through a speech-only path and a target-conditioned path; output coordinates are forced to come from one path or the other. By default keypoint indices 0--10 are target-conditioned and 11--20 are speech-only, while the 7 global motion dimensions remain target-conditioned. The exact index set is configurable with `--partition_keypoint_indices`. For the paper, keep one fixed mask for all identities and disclose it. If an image-space support diagnostic is available, pass its pre-registered mask rather than tuning the mask on test results.
- `emotion_residual`: target emotion is represented as a category-specific additive 80x70 motion residual on top of a category-agnostic speech trajectory. It cannot alter the speech generator through acoustic FiLM, so it is a direct residual-composition alternative.

Do **not** choose the partition mask after seeing test scores. That would invalidate the controlled comparison.

### Training protocol / semantic regularization

- `single_stage_mead`: removes Generic pretraining but preserves the same MEAD budget (5k emotion/start transition + 240k full MEAD), Min-SNR, balance, EMA, architecture, losses, and learning-rate family. This is the correct control for the two-data-stage claim.
- `no_emotion_ce`: changes only `lambda_emo` from 1 to 0. This isolates the frozen classifier regularizer.

## Existing training-recipe ablation (do not retrain)

Use the already trained runs on the **same final evaluation set**:

- 6006 `schedule_ema`: no shared-condition warm-start, no Min-SNR, no balanced sampling.
- 6008 `sharedcond_ema`: + shared generic prior.
- 6007 `sharedcond_balanced_ema`: + balanced MEAD sampling.
- 6010 `sharedcond_minsnr_ema`: + Min-SNR.
- 6009 `sharedcond_minsnr_balanced_ema`: final method.

This table is supporting evidence for the training recipe, not the main method-claim table.

## Run commands

```bash
python train_Ablation0905_cond_additive.py
python train_Ablation0905_cond_dit_adaln.py
python train_Ablation0905_motion_partition.py
python train_Ablation0905_emotion_residual.py
python train_Ablation0905_single_stage_mead.py
python train_Ablation0905_no_emotion_ce.py
```

For `motion_partition`, a different **pre-registered** pseudo-partition can be supplied, e.g.

```bash
python train_Ablation0905_motion_partition.py   --partition_keypoint_indices 0,2,4,7,9,11,13,15,18,19
```

## Evaluation rules

1. Use exactly the same held-out identities, renderer, source portraits, audio clips, CFG scale, diffusion steps, and preprocessing for every row.
2. For counterfactual control, hold portrait and audio fixed and generate all target emotions. Use the same initial diffusion seed/noise for corresponding comparisons.
3. Report target UAR (or the final independent RGB emotion metric), source-emotion leakage, target-source probability margin, LSE-C/LSE-D, and FVD.
4. Do not use the training-time motion classifier as the paper emotion evaluator.
5. Evaluate multiple fixed diffusion seeds when budget permits and report paired uncertainty.
6. The final 6009 Stage-1 manifest is assumed to exclude MEAD test identities, as verified before these ablations.

## Checkpoint provenance

Each training script writes its own `variant_name` and `model_variant` into checkpoint args. Keep those fields when generating videos so a checkpoint is always loaded with its dedicated model file.

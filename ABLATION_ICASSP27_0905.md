# ICASSP27 controlled ablations — 2026-09-05

## Reference implementation

The final paper model is `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` (run 6009). Its default `align_mask_width=2` corresponds to **audio-attention radius 1** because `enc_dec_mask` uses `expansion=align_mask_width-1`.

The Stage-1 `front_all_motions.pkl` manifest used by the final recipe has already been verified to exclude MEAD test identities. Therefore the two-stage control below tests the value of generic motion pretraining rather than repairing an identity-leakage problem.

## File organization

Each controlled variant has exactly two user-facing files:

```text
train_Ablation0905_<variant>.py
src/modules/emotion_dit_ablation0905_<variant>.py
```

The model file is self-contained. The former `*_legacy.py` companion files have been merged into the corresponding public model file and deleted. The private `_Core*` classes and the public runtime-corrected classes intentionally coexist **inside the same file** so checkpoint compatibility is preserved without cross-file nesting. No ablation model imports another ablation model; no ablation training script imports another ablation training script.

## A. Conditioning controls — main paper table

| Variant | Training file | What changes relative to full ADEF |
|---|---|---|
| `audio_only` | `train_Ablation0905_audio_only.py` | Target label is functionally disconnected; start priors are category agnostic. Target parameters remain allocated only to keep capacity/count matched. |
| `cond_late_token_concat` | `train_Ablation0905_cond_late_token_concat.py` | Acoustic tokens remain target agnostic; one target token is appended to cross-attention memory after acoustic encoding. No new projection or parameters are added. |
| `cond_additive` | `train_Ablation0905_cond_additive.py` | Replace feature-wise affine acoustic reparameterization with an additive target bias. |
| `cond_dit_adaln` | `train_Ablation0905_cond_dit_adaln.py` | Acoustic carrier stays target agnostic; the target affine acts inside DiT blocks. |
| **full 6009** | `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` | Target affine reparameterizes the dynamic acoustic carrier before motion decoding. |

These rows now exactly support the conditioning comparison stated in the manuscript. `cond_late_token_concat` is deliberately **token concatenation**, not feature-dimension concatenation: appending a 512-D target token keeps the denoiser width and trainable parameter count unchanged, whereas feature concatenation would require an extra projection and confound placement with capacity.

Primary evidence: counterfactual T-UAR/E-UAR, source-emotion leakage, target-source probability margin, LSE-C/LSE-D, FVD.

## B. Holistic-generation controls — main paper table

| Variant | Training file | Scientific question |
|---|---|---|
| `motion_partition` | `train_Ablation0905_motion_partition.py` | What happens when output coordinates are forced to come from speech-only vs target-conditioned paths instead of one holistic trajectory model? |
| `emotion_residual` | `train_Ablation0905_emotion_residual.py` | Is `speech trajectory + additive emotion residual` sufficient compared with holistic joint generation? |
| **full 6009** | final training file | Does one joint conditional diffusion distribution better preserve both target affect and speech synchronization? |

For `motion_partition`, pre-register the coordinate mask before evaluating test results. The default indices are only a runnable reference. If the image-space support diagnostic produces a justified pseudo-partition, use that fixed mask for the paper run; never select a mask after seeing test scores.

## C. Frame-local cross-attention controls — required by the current manuscript

The implementation mapping is:

| Paper radius | `align_mask_width` | Training file |
|---|---:|---|
| radius 0 | 1 | `train_Ablation0905_attn_radius0.py` |
| **radius 1 (full 6009)** | **2** | final training file; **do not retrain** |
| radius 2 | 3 | `train_Ablation0905_attn_radius2.py` |
| global / unrestricted | 0 | `train_Ablation0905_attn_global.py` |

This set is necessary because the current paper explicitly states that radius `{0,1,2}` and unrestricted attention are evaluated. Earlier advice to omit these runs was therefore too conservative.

Primary evidence: LSE-C/LSE-D first, then FVD and target emotion metrics. The expected scientific question is whether radius 1 gives enough local audiovisual tolerance without allowing unrelated acoustic frames to dominate.

## D. Training / regularization controls

| Variant | Training file | Change |
|---|---|---|
| `single_stage_mead` | `train_Ablation0905_single_stage_mead.py` | Remove 190k generic Stage 1; keep the same 5k specialization + 240k MEAD adaptation budget and final training recipe. |
| `no_emotion_ce` | `train_Ablation0905_no_emotion_ce.py` | Set the frozen motion-classifier CE weight to zero. |
| `shared_start_tokens` | `train_Ablation0905_shared_start_tokens.py` | Keep full ADEF conditioning, but force category-agnostic first-window start priors. This checks that target control is not merely coming from emotion-indexed initialization. |

`shared_start_tokens` is an additional confound check rather than a headline contribution. It is worth running because the Method explicitly mentions emotion-indexed start tokens; a strong result here can show that acoustic reparameterization remains effective without category-specific initialization.

## E. Existing recipe ablations — reuse; do not retrain

Evaluate these existing checkpoints on the **same final test protocol**:

- 6006 `schedule_ema`: no shared-condition warm-start, no Min-SNR, no balanced sampling.
- 6008 `sharedcond_ema`: adds shared generic condition warm-start.
- 6007 `sharedcond_balanced_ema`: adds balanced MEAD sampling without Min-SNR.
- 6010 `sharedcond_minsnr_ema`: adds Min-SNR without balancing.
- 6009 `sharedcond_minsnr_balanced_ema`: full recipe.

Together they isolate shared-condition warm-start, class/level balancing, Min-SNR, and their combination. 6011 generic replay remains a diagnostic rather than part of the final method.

## Recommended training batches for six GPUs

The number of experiments is **not** limited to six. A practical order is:

```text
Batch 1 (main conditioning/representation):
  audio_only
  cond_late_token_concat
  cond_additive
  cond_dit_adaln
  emotion_residual
  motion_partition   [only after the partition mask is frozen]

Batch 2 (temporal/training/confound):
  attn_radius0
  attn_radius2
  attn_global
  single_stage_mead
  no_emotion_ce
  shared_start_tokens
```

If the pseudo-partition is not ready, move `motion_partition` to Batch 2 and use the free Batch-1 GPU for one fixed-seed replication of a high-priority conditioning row.

## Run commands

```bash
python train_Ablation0905_audio_only.py
python train_Ablation0905_cond_late_token_concat.py
python train_Ablation0905_cond_additive.py
python train_Ablation0905_cond_dit_adaln.py
python train_Ablation0905_motion_partition.py
python train_Ablation0905_emotion_residual.py
python train_Ablation0905_attn_radius0.py
python train_Ablation0905_attn_radius2.py
python train_Ablation0905_attn_global.py
python train_Ablation0905_single_stage_mead.py
python train_Ablation0905_no_emotion_ce.py
python train_Ablation0905_shared_start_tokens.py
```

## Evaluation rules

1. Use the same held-out identities, renderer, portraits, audio clips, preprocessing, diffusion steps and CFG setting for all causal comparisons.
2. Counterfactual evaluation fixes portrait/audio and generates every target category. Use identical diffusion initial noise for paired variant comparisons.
3. Report target RGB-video UAR, source leakage, target-source probability margin, LSE-C/LSE-D, FVD, and motion magnitude. Do not use the training-time motion classifier as the paper emotion evaluator.
4. For attention-radius rows, emphasize LSE before affect scores; for conditioning rows, emphasize target adherence/leakage while confirming LSE is not materially degraded.
5. Use multiple fixed seeds or paired bootstrap confidence intervals when budget permits.
6. Keep `model_variant` from the checkpoint. `src/utils/helper.py` dynamically resolves only the controlled `emotion_dit_ablation0905_*` namespace so evaluation loads the correct physical model file.
7. Smoke-test each new entrypoint for 10–50 iterations before launching the full run.

## Scope decision

No extra ablations are added merely because GPUs are available. In particular, EMA itself is an optimization/stabilization device rather than a claimed methodological contribution, and generic replay is not in the final method. The 12 controlled variants above cover every comparison explicitly promised by the current manuscript plus the category-specific-start confound, while the already trained 6006/6007/6008/6010 checkpoints cover the remaining recipe factors.

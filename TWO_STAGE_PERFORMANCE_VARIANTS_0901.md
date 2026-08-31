# Two-stage performance training variants (2026-09-01)

These files are independent physical copies. They do not overwrite
`train_Unification_twostage0819_opt.py` or the current model implementation.
All versions use the existing
`src/modules/emotion_dit_Unification_jianhua0803.py` interface, so no new model
file is required.

## Why these versions were added

The 0819 curve confirmed that the separated learning-rate groups and the short
emotion-only transition substantially reduced the Stage-1/Stage-2 shock. It
also showed two inefficiencies: Generic training stopped while its losses were
still falling, and the long MEAD tail spent many iterations at an almost-zero
learning rate. The common defaults are therefore:

- Phase 1: 190,000 Generic updates;
- Phase 2: 5,000 MEAD emotion/start warm-up updates;
- Phase 3: 240,000 full MEAD updates;
- total: 435,000 useful updates (447,000 only for the replay version, preserving
  approximately 240,000 effective MEAD updates);
- a 40,000-update constant low-LR tail instead of decaying to nearly zero;
- EMA validation and `best.pt` selection;
- a resume-safe checkpoint path that never re-copies already specialized
  emotion rows.

The new Generic wrapper also supplies genuine first-64-frame samples. The
training stream explicitly reports whether a batch is a continuation or a true
start, removing the old dependence on global iteration parity.

## Files and intended use

| Priority | Training file | Main optimization |
|---|---|---|
| 1 | `train_Unification_twostage0901_schedule_ema.py` | Clean control: corrected budget/loading/LR tail, EMA and validation; the emotion branch is still frozen in Phase 1. |
| 2 | `train_Unification_twostage0901_sharedcond_ema.py` | Phase 1 learns a shared `emo_embed`, `adaLN_modulation`, and `null_emotion_feat`; the embedding is copied to all emotions before MEAD specialization. |
| 3 | `train_Unification_twostage0901_sharedcond_minsnr_ema.py` | Version 2 plus stratified diffusion timesteps and Min-SNR primary-loss weighting. |
| 4 | `train_Unification_twostage0901_sharedcond_balanced_ema.py` | Version 2 plus conservative inverse-square-root emotion-level sampling on MEAD. |
| 5 | `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` | Recommended high-performance candidate: shared warm-start + Min-SNR + balanced MEAD + EMA. |
| 6 | `train_Unification_twostage0901_sharedcond_minsnr_balanced_replay_ema.py` | Version 5 plus one unlabeled Generic continuation replay batch every 20 Phase-3 updates. Replay updates only backbone/audio and leaves emotion/start priors untouched. |

Shared dataset file:

- `src/dataset/dataset_GenericTalkingMotion_clear_start0901.py`

## Generic-to-emotion initialization

For the shared-condition versions, Phase 1 trains row 0 as a generic condition:

- `start_motion_feat[0]`;
- `start_audio_feat[0]`;
- `emo_embed.weight[0]`.

After every Phase-1 optimizer update, row 0 is copied to rows 1-7. The shared
`adaLN_modulation` and `null_emotion_feat` are trained directly and do not need
copying. At the MEAD boundary, the row-wise Adam state is cleared and the eight
identical rows begin emotion-specific specialization from the same generic
starting point.

## Common reliability/performance changes

- True video-start batches for both Generic and MEAD.
- Audio context is encoded once for a continuation window and then split into
  previous/current features.
- Explicit propagation of `n_heads`, `n_layers`, `mlp_ratio`, indicator and PE
  options into the model constructor.
- Logging only every `log_iter`, rather than writing one console/file record for
  every update.
- `iter_XXXXXXX.pt` stores EMA weights under `model` for existing inference
  compatibility and also stores `model_raw`.
- `latest_train_state.pt` additionally stores the optimizer for optimizer-aware recovery.
- Validation is deterministic and uses both true-start and continuation clips;
  the lowest validation total is saved as `best.pt`.

## Example

```bash
python train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py
```

Resume from the compact training-state checkpoint:

```bash
python train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py \
  --resume_checkpoint experiments/emo_dit/20260901_twostage_sharedcond_minsnr_balanced_ema/checkpoints/latest_train_state.pt
```

The recommended execution order is version 1 -> version 2 -> version 5. Run
versions 3 and 4 only when a separated comparison of Min-SNR and balanced
sampling is useful. Run version 6 when preservation of generic-domain motion
quality is important enough to justify about 2.8% extra updates.

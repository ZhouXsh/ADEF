from pathlib import Path
import re

BASE = Path('train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py')
base = BASE.read_text(encoding='utf-8')

VARIANTS = [
    ('train_Unification_twostage0904_fusion_conservative_ema.py', 'fusion_conservative_ema', '20260904_fusion_conservative_ema', 0.25, 0.50, False, 5e-7),
    ('train_Unification_twostage0904_fusion_plain_tail_ema.py', 'fusion_plain_tail_ema', '20260904_fusion_plain_tail_ema', 0.00, 0.50, False, 5e-7),
    ('train_Unification_twostage0904_fusion_balanced_decay_ema.py', 'fusion_balanced_decay_ema', '20260904_fusion_balanced_decay_ema', 0.25, 0.25, False, 5e-7),
    ('train_Unification_twostage0904_fusion_natural_tail_ema.py', 'fusion_natural_tail_ema', '20260904_fusion_natural_tail_ema', 0.00, 0.00, False, 5e-7),
    ('train_Unification_twostage0904_fusion_lowaudio_protect_ema.py', 'fusion_lowaudio_protect_ema', '20260904_fusion_lowaudio_protect_ema', 0.25, 0.25, True, 2e-7),
    ('train_Unification_twostage0904_fusion_freezeaudio_protect_ema.py', 'fusion_freezeaudio_protect_ema', '20260904_fusion_freezeaudio_protect_ema', 0.25, 0.25, True, 0.0),
]

DEFAULT_RE = re.compile(
    r'# -----------------------------------------------------------------------------\n'
    r'# Variant defaults\..*?DEFAULT_MAX_ITER = 435000\n', re.S)
assert DEFAULT_RE.search(base), 'variant defaults not found'

HELPERS = r'''

def cosine_anneal_value(iteration, start_iter, end_iter, start_value, end_value):
    """Cosine interpolation with smooth endpoints."""
    if iteration <= start_iter:
        return float(start_value)
    if iteration >= end_iter:
        return float(end_value)
    p = (iteration - start_iter) / float(max(1, end_iter - start_iter))
    retain = 0.5 * (1.0 + math.cos(math.pi * p))
    return float(end_value) + (float(start_value) - float(end_value)) * retain


def get_min_snr_mix_weight(args, iteration):
    if not args.use_min_snr:
        return 0.0
    return cosine_anneal_value(
        iteration, args.loss_anneal_start_iter, args.loss_anneal_end_iter,
        1.0, args.min_snr_final_weight,
    )


def get_balance_power_for_iteration(args, iteration):
    if not args.balance_mead:
        return 0.0
    power = cosine_anneal_value(
        iteration, args.balance_anneal_start_iter, args.balance_anneal_end_iter,
        args.balance_power, args.balance_final_power,
    )
    q = float(args.balance_power_quantum)
    if q > 0:
        power = round(power / q) * q
    lo = min(float(args.balance_power), float(args.balance_final_power))
    hi = max(float(args.balance_power), float(args.balance_final_power))
    return min(hi, max(lo, float(power)))


def apply_tail_lr_policy(args, iteration, lr_dict):
    if not args.tail_lr_protect or iteration < args.audio_protect_start_iter:
        return lr_dict
    out = dict(lr_dict)
    out['backbone'] = min(out['backbone'], args.tail_backbone_lr_cap)
    out['audio'] = min(out['audio'], args.tail_audio_lr_cap)
    out['emotion'] = min(out['emotion'], args.tail_emotion_lr_cap)
    out['start'] = min(out['start'], args.tail_start_lr_cap)
    return out
'''

REPL = [
('''    lr_dict = {
        "backbone": backbone_lr,
        "audio": audio_lr,
        "emotion": emotion_lr,
        "start": start_lr,
    }
    for group in optimizer.param_groups:
''', '''    lr_dict = {
        "backbone": backbone_lr,
        "audio": audio_lr,
        "emotion": emotion_lr,
        "start": start_lr,
    }
    lr_dict = apply_tail_lr_policy(args, iteration, lr_dict)
    for group in optimizer.param_groups:
'''),
('''def _compute_losses(args, model, classifier, use_emotion_loss,
                    is_starting_sample, current_motion, noise, target,
                    prev_motion_for_loss, end_idx, emo_index, time_step):
''', '''def _compute_losses(args, model, classifier, use_emotion_loss,
                    is_starting_sample, current_motion, noise, target,
                    prev_motion_for_loss, end_idx, emo_index, time_step,
                    iteration=None):
'''),
('''    if args.use_min_snr:
        if time_step is None:
            raise RuntimeError("Min-SNR requires explicit diffusion time steps")
        loss_primary = min_snr_primary_loss(
            args,
            model,
            is_starting_sample,
            current_motion,
            noise,
            target,
            prev_motion_for_loss,
            end_idx,
            time_step,
        )
''', '''    loss_primary_plain = loss_primary
    mix = get_min_snr_mix_weight(
        args, args.max_iter if iteration is None else iteration
    )
    loss_primary_minsnr = loss_primary_plain
    if args.use_min_snr:
        if time_step is None:
            raise RuntimeError("Min-SNR requires explicit diffusion time steps")
        loss_primary_minsnr = min_snr_primary_loss(
            args, model, is_starting_sample, current_motion, noise, target,
            prev_motion_for_loss, end_idx, time_step,
        )
    loss_primary = mix * loss_primary_minsnr + (1.0 - mix) * loss_primary_plain
'''),
('''        "primary": loss_primary,
        "emotion": loss_emotion if loss_emotion is not None else zero,
''', '''        "primary": loss_primary,
        "primary_plain": loss_primary_plain,
        "primary_minsnr": loss_primary_minsnr,
        "emotion": loss_emotion if loss_emotion is not None else zero,
'''),
('''    mead_weights = None
    if args.balance_mead:
        mead_weights, group_counts = build_mead_sample_weights(
            mead_dataset, args.balance_power
        )
        logging.info("MEAD emotion-level group counts: %s", dict(group_counts))
''', '''    active_balance_power = None
    if args.balance_mead:
        _, group_counts = build_mead_sample_weights(mead_dataset, args.balance_power)
        logging.info("MEAD emotion-level group counts: %s", dict(group_counts))
'''),
('''        else:
            if mead_stream is None:
                mead_stream = AlternatingBatchStream(
                    mead_dataset,
                    args.batch_size,
                    args.num_workers,
                    args.seed + 31,
                    start_interval=args.start_interval,
                    sample_weights=mead_weights,
                )
            (audio, coef_dict, emo_index, _), is_starting_sample = next(mead_stream)
            use_emotion_loss = True
            data_name = "mead"
''', '''        else:
            desired_power = get_balance_power_for_iteration(args, iteration)
            if (mead_stream is None or active_balance_power is None
                    or abs(desired_power - active_balance_power) > 1e-12):
                if mead_stream is not None:
                    mead_stream.close()
                mead_weights = None
                if args.balance_mead and desired_power > 0:
                    mead_weights, _ = build_mead_sample_weights(mead_dataset, desired_power)
                stream_seed = args.seed + 31 + iteration + int(round(desired_power * 1000.0)) * 100003
                mead_stream = AlternatingBatchStream(
                    mead_dataset, args.batch_size, args.num_workers, stream_seed,
                    start_interval=args.start_interval, sample_weights=mead_weights,
                )
                mead_stream.counter = max(0, iteration - args.stage1_iter - 1) % args.start_interval
                active_balance_power = desired_power
                logging.info("MEAD sampler rebuilt at iter %d with balance_power=%.3f", iteration, active_balance_power)
            (audio, coef_dict, emo_index, _), is_starting_sample = next(mead_stream)
            use_emotion_loss = True
            data_name = "mead"
'''),
('''            emo_index,
            time_step,
        )

        (loss_dict["total"] / args.gradient_accumulation_steps).backward()
''', '''            emo_index,
            time_step,
            iteration=iteration,
        )

        (loss_dict["total"] / args.gradient_accumulation_steps).backward()
'''),
('''            "primary": loss_dict["primary"].item(),
            "emotion": args.l_emo * loss_dict["emotion"].item(),
''', '''            "primary": loss_dict["primary"].item(),
            "primary_plain": loss_dict["primary_plain"].item(),
            "primary_minsnr": loss_dict["primary_minsnr"].item(),
            "emotion": args.l_emo * loss_dict["emotion"].item(),
'''),
('''            writer.add_scalar("train/is_generic_replay", int(is_generic_replay), iteration)
''', '''            writer.add_scalar("train/is_generic_replay", int(is_generic_replay), iteration)
            writer.add_scalar("train/min_snr_mix", get_min_snr_mix_weight(args, iteration), iteration)
            writer.add_scalar("train/balance_power", get_balance_power_for_iteration(args, iteration), iteration)
            writer.add_scalar("train/primary_plain_loss", np.mean(loss_log["primary_plain"]), iteration)
            writer.add_scalar("train/primary_minsnr_loss", np.mean(loss_log["primary_minsnr"]), iteration)
            writer.add_scalar("train/audio_protected", int(args.tail_lr_protect and iteration >= args.audio_protect_start_iter), iteration)
'''),
('''    parser.add_argument("--resume_checkpoint", type=Path, default=None)
''', '''    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    add_bool_argument(parser, "resume_optimizer", False,
                      "Load Adam state when present; off by default for 6009 tail branching.")
'''),
('''    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
''', '''    if checkpoint is not None and args.resume_optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
'''),
]

PARSER_OLD = '    parser.add_argument("--balance_power", type=float, default=0.5)\n'
PARSER_NEW = '''    parser.add_argument("--balance_power", type=float, default=0.5)
    parser.add_argument("--loss_anneal_start_iter", type=int, default=DEFAULT_LOSS_ANNEAL_START_ITER)
    parser.add_argument("--loss_anneal_end_iter", type=int, default=DEFAULT_LOSS_ANNEAL_END_ITER)
    parser.add_argument("--min_snr_final_weight", type=float, default=DEFAULT_MIN_SNR_FINAL_WEIGHT)
    parser.add_argument("--balance_anneal_start_iter", type=int, default=DEFAULT_BALANCE_ANNEAL_START_ITER)
    parser.add_argument("--balance_anneal_end_iter", type=int, default=DEFAULT_BALANCE_ANNEAL_END_ITER)
    parser.add_argument("--balance_final_power", type=float, default=DEFAULT_BALANCE_FINAL_POWER)
    parser.add_argument("--balance_power_quantum", type=float, default=0.05)
    add_bool_argument(parser, "tail_lr_protect", DEFAULT_TAIL_LR_PROTECT,
                      "Clamp final-stage LRs to protect learned speech-motion alignment.")
    parser.add_argument("--audio_protect_start_iter", type=int, default=DEFAULT_AUDIO_PROTECT_START_ITER)
    parser.add_argument("--tail_backbone_lr_cap", type=float, default=DEFAULT_TAIL_BACKBONE_LR_CAP)
    parser.add_argument("--tail_audio_lr_cap", type=float, default=DEFAULT_TAIL_AUDIO_LR_CAP)
    parser.add_argument("--tail_emotion_lr_cap", type=float, default=DEFAULT_TAIL_EMOTION_LR_CAP)
    parser.add_argument("--tail_start_lr_cap", type=float, default=DEFAULT_TAIL_START_LR_CAP)
'''

VALID_OLD = '''    if not 0 <= args.generic_replay_emo_index < 8:
        raise ValueError("generic_replay_emo_index must be in [0, 7]")
'''
VALID_NEW = VALID_OLD + '''    phase3_start = args.stage1_iter + args.stage2_emotion_only_iter + 1
    if not phase3_start <= args.loss_anneal_start_iter < args.loss_anneal_end_iter <= args.max_iter:
        raise ValueError("loss anneal range must lie inside Phase 3")
    if not phase3_start <= args.balance_anneal_start_iter < args.balance_anneal_end_iter <= args.max_iter:
        raise ValueError("balance anneal range must lie inside Phase 3")
    if not 0.0 <= args.min_snr_final_weight <= 1.0:
        raise ValueError("min_snr_final_weight must be in [0, 1]")
    if not 0.0 <= args.balance_final_power <= args.balance_power:
        raise ValueError("balance_final_power must be in [0, balance_power]")
    if args.balance_power_quantum <= 0:
        raise ValueError("balance_power_quantum must be positive")
'''

GENERIC_START = '    generic_template_path = args.generic_motion_template_path\n'
GENERIC_END = '    mead_dataset = EmoLevelDataset(\n'
GENERIC_BLOCK = '''    generic_dataset = None
    needs_generic_dataset = start_iter < args.stage1_iter or args.generic_replay_interval > 0
    if needs_generic_dataset:
        generic_template_path = args.generic_motion_template_path
        if generic_template_path is None:
            generic_template_path = Path(args.data_root) / args.motion_template_filename
        generic_dataset = GenericTalkingMotionDataset(
            motion_template_path=generic_template_path,
            motion_filenames=args.generic_motion_filenames or None,
            aggregate_motion_files=args.generic_aggregate_motion_files or None,
            split="train", split_file=args.generic_split_file,
            validation_ratio=args.generic_validation_ratio,
            split_seed=args.generic_split_seed, coef_fps=args.fps,
            n_motions=args.n_motions, n_prev_motions=args.n_prev_motions,
            crop_strategy=args.crop_strategy, normalize_type=args.normalize_type,
            strict_absolute_paths=not args.generic_allow_relative_paths,
            missing_audio_policy=args.generic_missing_audio_policy,
            duplicate_policy=args.generic_duplicate_policy,
        )
'''

for filename, name, exp, min_final, balance_final, protect, audio_cap in VARIANTS:
    text = base
    block = f'''# -----------------------------------------------------------------------------
# 0904 fusion defaults. Full independent copy derived from successful 6009.
# -----------------------------------------------------------------------------
VARIANT_NAME = "{name}"
VARIANT_DESCRIPTION = "6009-based late-stage Pareto refinement."
DEFAULT_EXP_NAME = "{exp}"
DEFAULT_SHARED_CONDITION_WARMSTART = True
DEFAULT_USE_MIN_SNR = True
DEFAULT_BALANCE_MEAD = True
DEFAULT_GENERIC_REPLAY_INTERVAL = 0
DEFAULT_MAX_ITER = 450000
DEFAULT_LOSS_ANNEAL_START_ITER = 350000
DEFAULT_LOSS_ANNEAL_END_ITER = 420000
DEFAULT_MIN_SNR_FINAL_WEIGHT = {min_final}
DEFAULT_BALANCE_ANNEAL_START_ITER = 380000
DEFAULT_BALANCE_ANNEAL_END_ITER = 430000
DEFAULT_BALANCE_FINAL_POWER = {balance_final}
DEFAULT_TAIL_LR_PROTECT = {protect}
DEFAULT_AUDIO_PROTECT_START_ITER = 400000
DEFAULT_TAIL_BACKBONE_LR_CAP = 2.5e-6
DEFAULT_TAIL_AUDIO_LR_CAP = {audio_cap}
DEFAULT_TAIL_EMOTION_LR_CAP = 8e-6
DEFAULT_TAIL_START_LR_CAP = 4e-6
'''
    text, n = DEFAULT_RE.subn(block, text, count=1)
    assert n == 1
    marker = '\ndef get_training_phase(args, iteration):\n'
    assert marker in text
    text = text.replace(marker, HELPERS + marker, 1)
    for old, new in REPL:
        assert old in text, old[:100]
        text = text.replace(old, new, 1)
    assert PARSER_OLD in text
    text = text.replace(PARSER_OLD, PARSER_NEW, 1)
    assert VALID_OLD in text
    text = text.replace(VALID_OLD, VALID_NEW, 1)

    gs = text.index(GENERIC_START)
    ge = text.index(GENERIC_END, gs)
    text = text[:gs] + GENERIC_BLOCK + text[ge:]

    old = '''    if start_iter < args.stage1_iter:
        generic_stream = AlternatingBatchStream(
            generic_dataset,
'''
    new = '''    if start_iter < args.stage1_iter:
        if generic_dataset is None:
            raise RuntimeError("Phase 1 requires Generic dataset")
        generic_stream = AlternatingBatchStream(
            generic_dataset,
'''
    assert old in text
    text = text.replace(old, new, 1)
    old = '''    audio_unit = mead_dataset.audio_unit
    if abs(generic_dataset.audio_unit - audio_unit) > 1e-6:
        raise RuntimeError("Generic and MEAD audio units do not match")
'''
    new = '''    audio_unit = mead_dataset.audio_unit
    if generic_dataset is not None and abs(generic_dataset.audio_unit - audio_unit) > 1e-6:
        raise RuntimeError("Generic and MEAD audio units do not match")
'''
    assert old in text
    text = text.replace(old, new, 1)
    old = '''            if generic_stream is None:
                generic_stream = AlternatingBatchStream(
                    generic_dataset,
'''
    new = '''            if generic_stream is None:
                if generic_dataset is None:
                    raise RuntimeError("Phase 1 requires Generic dataset")
                generic_stream = AlternatingBatchStream(
                    generic_dataset,
'''
    assert old in text
    text = text.replace(old, new, 1)
    Path(filename).write_text(text, encoding='utf-8')

Path('FUSION_VARIANTS_0904.md').write_text('''# 0904 Fusion variants

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
''', encoding='utf-8')

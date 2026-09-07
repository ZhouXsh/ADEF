from pathlib import Path
import re

files = sorted(Path('.').glob('train_Ablation0905_*.py'))
changed = []

main_pattern = re.compile(
    r'    generic_template_path = args\.generic_motion_template_path\n'
    r'.*?'
    r'    mead_dataset = EmoLevelDataset\(\n'
    r'.*?'
    r'    \)\n'
    r'(?=    # Validation is disabled)',
    re.S,
)
main_replacement = '''    # Stage-aware dataset lifecycle: do not materialize MEAD during Generic Stage 1.
    # On a Phase-2/3 resume, Generic is skipped entirely.
    generic_dataset = None
    if start_iter < args.stage1_iter:
        generic_template_path = args.generic_motion_template_path
        if generic_template_path is None:
            generic_template_path = Path(args.data_root) / args.motion_template_filename

        generic_dataset = GenericTalkingMotionDataset(
            motion_template_path=generic_template_path,
            motion_filenames=args.generic_motion_filenames or None,
            aggregate_motion_files=args.generic_aggregate_motion_files or None,
            split="train",
            split_file=args.generic_split_file,
            validation_ratio=args.generic_validation_ratio,
            split_seed=args.generic_split_seed,
            coef_fps=args.fps,
            n_motions=args.n_motions,
            n_prev_motions=args.n_prev_motions,
            crop_strategy=args.crop_strategy,
            normalize_type=args.normalize_type,
            strict_absolute_paths=not args.generic_allow_relative_paths,
            missing_audio_policy=args.generic_missing_audio_policy,
            duplicate_policy=args.generic_duplicate_policy,
        )

    # MEAD is intentionally constructed lazily inside train() only after the
    # Generic stream/dataset has been closed and released.
    mead_dataset = None
'''

init_pattern = re.compile(
    r'    mead_weights = None\n'
    r'    if args\.balance_mead:\n'
    r'        mead_weights, group_counts = build_mead_sample_weights\(\n'
    r'            mead_dataset, args\.balance_power\n'
    r'        \)\n'
    r'        logging\.info\("MEAD emotion-level group counts: %s", dict\(group_counts\)\)\n\n'
    r'    # Keep only the current stage.*?'
    r'    n_prev_audio_samples = round\(audio_unit \* args\.n_prev_motions\)\n',
    re.S,
)
init_replacement = '''    mead_weights = None

    # Derive the shared audio/frame unit without constructing MEAD.
    audio_unit = 16000.0 / float(args.fps)
    n_audio_samples = round(audio_unit * args.n_motions)
    n_prev_audio_samples = round(audio_unit * args.n_prev_motions)
    if generic_dataset is not None and abs(generic_dataset.audio_unit - audio_unit) > 1e-6:
        raise RuntimeError("Generic audio unit does not match the configured fps")

    def ensure_mead_dataset():
        nonlocal mead_dataset, mead_weights
        if mead_dataset is None:
            logging.info("Loading MEAD dataset after Generic resources have been released")
            mead_dataset = EmoLevelDataset(
                args.data_root,
                motion_filename=args.motion_filename,
                motion_template_filename=args.motion_template_filename,
                split="train",
                coef_fps=args.fps,
                n_motions=args.n_motions,
                n_prev_motions=args.n_prev_motions,
                crop_strategy=args.crop_strategy,
                normalize_type=args.normalize_type,
            )
            if abs(mead_dataset.audio_unit - audio_unit) > 1e-6:
                raise RuntimeError("MEAD audio unit does not match the configured fps")
            if args.balance_mead:
                mead_weights, group_counts = build_mead_sample_weights(
                    mead_dataset, args.balance_power
                )
                logging.info(
                    "MEAD emotion-level group counts: %s", dict(group_counts)
                )
        return mead_dataset

    # Keep only the current stage's persistent worker pool alive. Stage 1 uses
    # Generic only; MEAD workers and the MEAD motion pickle do not exist yet.
    generic_stream = None
    mead_stream = None
    generic_replay_stream = None
    if start_iter < args.stage1_iter:
        if generic_dataset is None:
            raise RuntimeError("Generic dataset is required before the Stage-1 boundary")
        generic_stream = AlternatingBatchStream(
            generic_dataset,
            args.batch_size,
            args.num_workers,
            args.seed + 11,
            start_interval=args.start_interval,
        )
    else:
        ensure_mead_dataset()
'''

mead_stream_old = '''            if mead_stream is None:
                mead_stream = AlternatingBatchStream(
                    mead_dataset,
'''
mead_stream_new = '''            if mead_stream is None:
                ensure_mead_dataset()
                mead_stream = AlternatingBatchStream(
                    mead_dataset,
'''

release_pattern = re.compile(
    r'        # The Stage-1 boundary saves two large checkpoints\..*?'
    r'            gc\.collect\(\)\n',
    re.S,
)
release_replacement = '''        # Strict stage boundary: terminate every Generic worker, release the
        # in-memory Generic motion dictionary, and return allocator pages to the
        # OS before MEAD is constructed on the next iteration.
        if iteration == args.stage1_iter:
            logging.info("Closing and releasing Generic resources at Stage-1 boundary")
            if generic_stream is not None:
                generic_stream.close()
                generic_stream = None
            if generic_dataset is not None:
                _release_dataset_storage(generic_dataset)
                generic_dataset = None
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except (OSError, AttributeError):
                pass
            logging.info("Generic resources released; MEAD will be loaded lazily")
'''

replay_validation_old = '''    if args.generic_replay_interval < 0:
        raise ValueError("generic_replay_interval cannot be negative")
    if args.generic_replay_interval > 0 and args.gradient_accumulation_steps != 1:
        raise ValueError("generic replay requires gradient_accumulation_steps=1")
'''
replay_validation_new = '''    if args.generic_replay_interval != 0:
        raise ValueError(
            "Ablation scripts use a strict Generic->release->MEAD lifecycle; "
            "generic replay must remain disabled."
        )
'''

for path in files:
    text = path.read_text(encoding='utf-8')
    original = text

    # single_stage_mead intentionally has no Generic Stage 1 and starts from MEAD.
    if path.name != 'train_Ablation0905_single_stage_mead.py':
        text, n_main = main_pattern.subn(main_replacement, text, count=1)
        if n_main != 1:
            raise RuntimeError(f'{path}: main dataset block match={n_main}')
        text, n_init = init_pattern.subn(init_replacement, text, count=1)
        if n_init != 1:
            raise RuntimeError(f'{path}: train init block match={n_init}')
        if text.count(mead_stream_old) != 1:
            raise RuntimeError(f'{path}: MEAD stream block mismatch')
        text = text.replace(mead_stream_old, mead_stream_new, 1)
        text, n_release = release_pattern.subn(release_replacement, text, count=1)
        if n_release != 1:
            raise RuntimeError(f'{path}: release block match={n_release}')
        if replay_validation_old not in text:
            raise RuntimeError(f'{path}: replay validation block missing')
        text = text.replace(replay_validation_old, replay_validation_new, 1)

    if path.name == 'train_Ablation0905_audio_only.py':
        backward_old = '''        (loss_dict["total"] / args.gradient_accumulation_steps).backward()
        micro_step += 1

        if is_generic_replay:
'''
        backward_new = '''        loss_for_backward = (
            loss_dict["total"] / args.gradient_accumulation_steps
        )
        has_backward_path = loss_for_backward.requires_grad
        if has_backward_path:
            loss_for_backward.backward()
            micro_step += 1
        else:
            # In audio-only Phase 2, continuation batches use neither the shared
            # start priors nor target-emotion parameters, while backbone/audio are
            # intentionally frozen. Such batches are expected no-ops.
            expected_audio_only_noop = phase == 2 and not is_starting_sample
            if not expected_audio_only_noop:
                raise RuntimeError(
                    "Loss has no gradient path outside the expected audio-only "
                    "Phase-2 continuation no-op."
                )

        if is_generic_replay:
'''
        if text.count(backward_old) != 1:
            raise RuntimeError('audio_only: backward block mismatch')
        text = text.replace(backward_old, backward_new, 1)
        step_old = '''        should_step = (
            micro_step % args.gradient_accumulation_steps == 0
            or iteration == args.max_iter
        )
'''
        step_new = '''        should_step = has_backward_path and (
            micro_step % args.gradient_accumulation_steps == 0
            or iteration == args.max_iter
        )
'''
        if text.count(step_old) != 1:
            raise RuntimeError('audio_only: optimizer step block mismatch')
        text = text.replace(step_old, step_new, 1)

    if text != original:
        path.write_text(text, encoding='utf-8')
        changed.append(path.name)

print('patched', len(changed), 'files')
for name in changed:
    print(name)
if len(changed) != 11:
    raise RuntimeError(f'expected 11 changed two-stage ablations, got {len(changed)}')

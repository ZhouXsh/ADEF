"""Minimal Stage-1 dataset adapter for the existing two-stage trainer.

The model implementations and ``train_two_stage.py`` remain unchanged. This
entrypoint only replaces the Stage-1 dataset construction so the newly added
Stage1TalkingMotionDataset is used, then delegates the complete training flow
to the original single-file trainer.
"""

import train_two_stage as trainer
from src.dataset.dataset_Stage1TalkingMotion import Stage1TalkingMotionDataset


_original_make_datasets = trainer.make_datasets
_original_unpack_batch = trainer.unpack_batch


def make_datasets(args):
    if args.stage != 1:
        return _original_make_datasets(args)

    common = dict(
        motion_template_path=args.motion_template_path,
        motion_filenames=args.generic_motion_files,
        aggregate_motion_files=args.aggregate_motion_files,
        validation_ratio=args.generic_validation_ratio,
        split_seed=args.split_seed,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        normalize_type=args.normalize_type,
    )
    train_dataset = Stage1TalkingMotionDataset(
        split="train",
        split_file=args.generic_train_split,
        crop_strategy=args.crop_strategy,
        **common,
    )
    val_dataset = Stage1TalkingMotionDataset(
        split="val",
        split_file=args.generic_val_split,
        crop_strategy="begin",
        **common,
    )
    return train_dataset, val_dataset


def unpack_batch(batch, args, device, predict_head_pose):
    if args.stage == 1:
        audio_pair, coefficient_pair, _, sample_name = batch
        batch = audio_pair, coefficient_pair, sample_name
    return _original_unpack_batch(batch, args, device, predict_head_pose)


def main():
    parser = trainer.build_parser()
    parser.add_argument(
        "--generic_motion_files",
        type=str,
        default=None,
        help="Comma-separated motion pickle files for Stage 1.",
    )
    args = parser.parse_args()

    if args.stage == 1 and not (
        args.generic_motion_files or args.aggregate_motion_files
    ):
        parser.error(
            "Stage 1 requires --generic_motion_files or "
            "--aggregate_motion_files"
        )

    # The original main function checks this legacy option. Set a harmless
    # compatibility value after the new, explicit validation above.
    if args.stage == 1 and not args.generic_video_roots:
        args.generic_video_roots = args.generic_motion_files or "aggregate"

    trainer.make_datasets = make_datasets
    trainer.unpack_batch = unpack_batch
    trainer.main(args)


if __name__ == "__main__":
    main()

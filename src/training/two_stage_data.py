from __future__ import annotations

import argparse

import torch
from torch.utils import data

from src.dataset.dataset_EmotionLevel import EmoLevelDataset
from src.dataset.dataset_GenericTalkingMotion import GenericTalkingMotionDataset


def make_generic_dataset(args: argparse.Namespace, split: str):
    split_file = (
        args.generic_train_split_file if split == "train" else args.generic_val_split_file
    )
    template_path = args.generic_motion_template
    if template_path is None:
        template_path = args.data_root / args.motion_template_filename
    return GenericTalkingMotionDataset(
        motion_template_path=template_path,
        motion_filenames=args.generic_motion_files,
        aggregate_motion_files=args.generic_aggregate_motion_files,
        split=split,
        split_file=split_file,
        validation_ratio=args.generic_validation_ratio,
        split_seed=args.generic_split_seed,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy if split == "train" else "begin",
        normalize_type=args.normalize_type,
        strict_absolute_paths=not args.generic_allow_relative_paths,
        missing_audio_policy=args.generic_missing_audio_policy,
        duplicate_policy=args.generic_duplicate_policy,
        max_retries=args.dataset_max_retries,
    )


def make_emotion_dataset(args: argparse.Namespace, split: str):
    return EmoLevelDataset(
        args.data_root,
        motion_filename=args.motion_filename,
        motion_template_filename=args.motion_template_filename,
        split=split,
        coef_fps=args.fps,
        n_motions=args.n_motions,
        crop_strategy=args.crop_strategy if split == "train" else "begin",
        normalize_type=args.normalize_type,
    )


def build_loaders(args: argparse.Namespace):
    if args.stage == "general":
        train_dataset = make_generic_dataset(args, "train")
        val_dataset = make_generic_dataset(args, "val")
    else:
        train_dataset = make_emotion_dataset(args, "train")
        val_dataset = make_emotion_dataset(args, "val")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader

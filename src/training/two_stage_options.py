from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["general", "emotion"], required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage1_ckpt", type=Path)

    parser.add_argument("--data_root", type=Path, default=Path("src/my_prepare"))
    parser.add_argument("--motion_filename", type=str, default="front_all_motions.pkl")
    parser.add_argument("--motion_template_filename", type=str, default="motion_template.pkl")
    parser.add_argument("--generic_motion_template", type=Path)
    parser.add_argument("--generic_motion_files", type=str)
    parser.add_argument("--generic_aggregate_motion_files", type=str)
    parser.add_argument("--generic_train_split_file", type=Path)
    parser.add_argument("--generic_val_split_file", type=Path)
    parser.add_argument("--generic_validation_ratio", type=float, default=0.05)
    parser.add_argument("--generic_split_seed", type=int, default=2026)
    parser.add_argument(
        "--generic_missing_audio_policy", choices=["skip", "error"], default="skip"
    )
    parser.add_argument(
        "--generic_duplicate_policy",
        choices=["error", "keep_first", "keep_last"],
        default="error",
    )
    parser.add_argument("--generic_allow_relative_paths", action="store_true")
    parser.add_argument("--dataset_max_retries", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--crop_strategy", choices=["random", "begin", "end"], default="random")
    parser.add_argument("--normalize_type", choices=["mix"], default="mix")

    parser.add_argument("--target", choices=["sample", "noise"], default="sample")
    parser.add_argument("--guiding_conditions", type=str, default="audio,emotion")
    parser.add_argument("--cfg_mode", choices=["incremental", "independent"], default="incremental")
    parser.add_argument("--n_diff_steps", type=int, default=50)
    parser.add_argument(
        "--diff_schedule",
        choices=["linear", "cosine", "quadratic", "sigmoid"],
        default="cosine",
    )
    parser.add_argument("--no_head_pose", action="store_true")
    parser.add_argument("--rot_repr", choices=["aa"], default="aa")
    parser.add_argument(
        "--audio_model",
        choices=["wav2vec2", "hubert", "hubert_zh", "hubert_zh_ori"],
        default="wav2vec2",
    )
    parser.add_argument("--architecture", choices=["decoder"], default="decoder")
    parser.add_argument("--align_mask_width", type=int, default=1)
    parser.add_argument("--use_learnable_pe", action="store_true")
    parser.add_argument("--use_indicator", action="store_true")
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--n_motions", type=int, default=100)
    parser.add_argument("--n_prev_motions", type=int, default=25)
    parser.add_argument("--motion_feat_dim", type=int, default=70)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--pad_mode", choices=["zero", "replicate"], default="zero")
    parser.add_argument("--general_audio_dropout", type=float, default=0.1)
    parser.add_argument("--emotion_dropout", type=float, default=0.5)
    parser.add_argument("--emotion_gate_init", type=float, default=0.1)
    parser.add_argument("--train_audio_encoder", action="store_true")
    parser.add_argument("--stage2_unfreeze_motion_decoder", action="store_true")

    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warm_iter", type=int, default=10000)
    parser.add_argument("--min_lr_ratio", type=float, default=0.02)
    parser.add_argument("--clip_grad_norm", type=float, default=2.0)

    parser.add_argument("--criterion", choices=["l1", "l2"], default="l2")
    parser.add_argument("--l_exp", type=float, default=0.1)
    parser.add_argument("--l_exp_vel", type=float, default=1e-4)
    parser.add_argument("--l_exp_smooth", type=float, default=1e-4)
    parser.add_argument("--l_head_angle", type=float, default=1e-2)
    parser.add_argument("--l_head_vel", type=float, default=1e-2)
    parser.add_argument("--l_head_smooth", type=float, default=1e-2)
    parser.add_argument("--l_head_trans", type=float, default=1e-2)
    parser.add_argument("--l_emotion", type=float, default=1.0)
    parser.add_argument("--l_sync_highfreq", type=float, default=0.5)
    parser.add_argument("--l_general_anchor", type=float, default=1.0)
    parser.add_argument("--highfreq_kernel", type=int, default=7)
    parser.add_argument("--no_constrain_prev", action="store_true")
    parser.add_argument("--use_context_audio_feat", action="store_true")
    parser.add_argument("--trunc_prob1", type=float, default=0.3)
    parser.add_argument("--trunc_prob2", type=float, default=0.4)

    parser.add_argument(
        "--emotion_classifier_ckpt",
        type=Path,
        default=Path("pretrained_weights/ADEF/emo_classifier/emo_level_classifier.pth"),
    )
    parser.add_argument("--save_iter", type=int, default=1000)
    parser.add_argument("--val_iter", type=int, default=1000)
    parser.add_argument("--log_iter", type=int, default=50)
    parser.add_argument("--log_smooth_win", type=int, default=50)

    args = parser.parse_args()
    validate_args(args, parser)
    return args


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.stage == "general":
        provided = int(bool(args.generic_motion_files)) + int(
            bool(args.generic_aggregate_motion_files)
        )
        if provided != 1:
            parser.error(
                "general stage requires exactly one of --generic_motion_files or "
                "--generic_aggregate_motion_files"
            )
    elif args.stage1_ckpt is None and args.resume is None:
        parser.error("emotion stage requires --stage1_ckpt unless --resume is used")

    if args.stage == "emotion" and args.target != "sample":
        parser.error("emotion stage requires --target sample")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be positive")
    if args.highfreq_kernel < 3 or args.highfreq_kernel % 2 == 0:
        parser.error("--highfreq_kernel must be an odd integer >= 3")
    if args.n_layers < 1 or args.n_heads < 1:
        parser.error("--n_layers and --n_heads must be positive")
    if args.feature_dim % args.n_heads:
        parser.error("--feature_dim must be divisible by --n_heads")

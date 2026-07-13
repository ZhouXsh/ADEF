from pathlib import Path

import torch

from src.training.two_stage_data import build_loaders
from src.training.two_stage_engine import (
    build_classifier,
    build_model,
    configure_logging,
    load_model_weights,
    lr_multiplier,
    resolve_device,
    set_seed,
    train,
)
from src.training.two_stage_options import parse_args


def main(args):
    set_seed(args.seed)
    exp_dir = Path("experiments/emo_dit") / args.exp_name
    writer = configure_logging(exp_dir)
    device = resolve_device(args.device)

    model = build_model(args, device)
    resume_checkpoint = None
    if args.resume is not None:
        resume_checkpoint = load_model_weights(model, args.resume, device, "resume")
    elif args.stage == "emotion":
        load_model_weights(model, args.stage1_ckpt, device, "stage-1")

    trainable_names = model.configure_training_stage(
        args.stage,
        train_audio_encoder=args.train_audio_encoder,
        stage2_unfreeze_motion_decoder=args.stage2_unfreeze_motion_decoder,
    )
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    writer.add_text("trainable_parameters", "\n".join(trainable_names))

    train_loader, val_loader = build_loaders(args)
    classifier = build_classifier(args, device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: lr_multiplier(step, args)
    )
    start_iter = 0
    if resume_checkpoint is not None:
        if "optimizer" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        start_iter = int(resume_checkpoint.get("iter", -1)) + 1

    options_path = exp_dir / "logs" / "options.log"
    with open(options_path, "w", encoding="utf-8") as file:
        for name, value in sorted(vars(args).items()):
            file.write(f"{name}: {value}\n")
    writer.add_text(
        "options", "\n".join(f"{k}: {v}" for k, v in sorted(vars(args).items()))
    )
    writer.add_text(
        "parameter_count", f"trainable={trainable_count}, total={total_count}"
    )

    train(
        args,
        model,
        train_loader,
        val_loader,
        classifier,
        optimizer,
        scheduler,
        writer,
        exp_dir / "checkpoints",
        start_iter,
    )
    writer.close()


if __name__ == "__main__":
    main(parse_args())

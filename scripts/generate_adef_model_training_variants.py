from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

BASE_MODEL = Path("src/modules/emotion_dit_Unification_jianhua0803_legacy.py")
BASE_TRAIN = Path("train_Unification_jianhua0803.py")

MODEL_HEADER = """# NOTE: This file is a complete, independent physical copy of
# emotion_dit_Unification_jianhua0803_legacy.py. It does not import, inherit,
# or wrap the sample implementation. All changes are implemented in this file.
"""

TRAIN_HEADER = """# NOTE: This file is a complete, independent physical copy of
# train_Unification_jianhua0803.py. It is not a launcher or wrapper; all
# training changes are implemented directly in this file.
"""

LIP_CONSTANTS = """
LIP_KEYPOINTS = (6, 12, 14, 17, 19, 20)
LIP_DIM_INDICES = tuple(
    coordinate
    for keypoint in LIP_KEYPOINTS
    for coordinate in range(keypoint * 3, keypoint * 3 + 3)
)
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def regex_once(text: str, pattern: str, replacement: str, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=flags)
    if count != 1:
        raise RuntimeError(f"{label}: expected one regex match, found {count}")
    return updated


def replace_region(text: str, start_marker: str, end_marker: str, replacement: str, label: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end = text.find(end_marker, start)
    if end < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start] + replacement + text[end:]


def insert_before(text: str, marker: str, addition: str, label: str, start: int = 0) -> str:
    index = text.find(marker, start)
    if index < 0:
        raise RuntimeError(f"{label}: marker not found")
    return text[:index] + addition + text[index:]


def prepare_model(base: str) -> str:
    text = MODEL_HEADER + base

    old_signature = """    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=64, n_prev_motions=16,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental",
                 guiding_conditions="audio,emotion", emo_classes=8,
                 align_mask_width=1):"""
    new_signature = """    def __init__(self, device='cuda', target="sample", architecture="decoder",
                 motion_feat_dim=70, fps=25, n_motions=64, n_prev_motions=16,
                 audio_model="hubert", feature_dim=512, n_diff_steps=500,
                 diff_schedule="cosine", cfg_mode="incremental",
                 guiding_conditions="audio,emotion", emo_classes=8,
                 align_mask_width=1, n_heads=8, n_layers=8, mlp_ratio=4,
                 use_indicator=True, no_use_learnable_pe=False):"""
    text = replace_once(text, old_signature, new_signature, "DitTalkingHead signature")

    text = replace_once(
        text,
        """        self.feature_dim = feature_dim

        # Audio encoder""",
        """        self.feature_dim = feature_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.mlp_ratio = mlp_ratio
        self.use_indicator = bool(use_indicator)
        self.no_use_learnable_pe = bool(no_use_learnable_pe)

        # Audio encoder""",
        "store architecture parameters",
    )

    denoise_start = text.find("        self.denoising_net = DenoisingNetwork(")
    denoise_end = text.find("        self.diffusion_sched =", denoise_start)
    if denoise_start < 0 or denoise_end < 0:
        raise RuntimeError("DenoisingNetwork construction block was not found")
    block = text[denoise_start:denoise_end]
    block = replace_once(
        block,
        "            motion_feat_dim=self.motion_feat_dim,\n",
        "            motion_feat_dim=self.motion_feat_dim,\n            use_indicator=use_indicator,\n",
        "pass use_indicator",
    )
    block = replace_once(
        block,
        "            feature_dim=feature_dim,\n",
        "            feature_dim=feature_dim,\n            n_heads=n_heads,\n            n_layers=n_layers,\n            mlp_ratio=mlp_ratio,\n",
        "pass transformer dimensions",
    )
    block = replace_once(
        block,
        "            align_mask_width=align_mask_width,\n",
        "            align_mask_width=align_mask_width,\n            no_use_learnable_pe=no_use_learnable_pe,\n",
        "pass positional encoding option",
    )
    text = text[:denoise_start] + block + text[denoise_end:]

    text = replace_once(
        text,
        """        # 指示器用于指示 最后一个音频片段 中 填充的部分。
        if indicator is not None:
            indicator = torch.cat([""",
        """        # 指示器用于指示最后一个音频片段中的有效帧。
        if self.use_indicator and indicator is None:
            indicator = torch.ones(
                motion_feat.shape[:2], device=motion_feat.device,
                dtype=motion_feat.dtype,
            )
        if indicator is not None:
            indicator = indicator.to(dtype=motion_feat.dtype)
            indicator = torch.cat([""",
        "indicator propagation",
    )

    text = replace_once(
        text,
        """        if ret_traj:
            return traj, motion_at_T, audio_feat_cond
        return traj[0], motion_at_T, audio_feat_cond""",
        """        if ret_traj:
            return traj, motion_at_T, audio_feat_saved
        return traj[0], motion_at_T, audio_feat_saved""",
        "return raw audio feature from sample",
    )
    return text


COMMON_FORWARD_BLOCK = """        # ---------- 单次上下文音频编码 ----------
        # 起始窗口仅编码真实的前 64 帧；续接窗口将 prev 16 + current 64
        # 拼接后一次编码，再按照帧索引切分，避免上下文分布不一致。
        if is_starting_sample:
            current_audio_feat = model.extract_audio_feature(
                current_audio, frame_num=args.n_motions
            )
            noise, target, _, _ = model(
                current_motion, current_audio_feat,
                indicator=indicator, emo_index=emo_index,
            )
            prev_motion_for_loss = None
        else:
            prev_motion_gt = motion_coef_full[:, :args.n_prev_motions]
            prev_audio_raw = audio[:, :n_prev_audio_samples].contiguous()
            context_audio = torch.cat([prev_audio_raw, current_audio], dim=1)
            context_audio_feat = model.extract_audio_feature(
                context_audio,
                frame_num=args.n_prev_motions + args.n_motions,
            )
            prev_audio_feat = context_audio_feat[:, :args.n_prev_motions].detach()
            current_audio_feat = context_audio_feat[:, args.n_prev_motions:]
            noise, target, _, _ = model(
                current_motion, current_audio_feat,
                prev_motion_gt, prev_audio_feat,
                indicator=indicator, emo_index=emo_index,
            )
            prev_motion_for_loss = prev_motion_gt

"""


def prepare_train(base: str, module_name: str, experiment_name: str) -> str:
    text = TRAIN_HEADER + base
    text = regex_once(
        text,
        r"^from src\.modules\.emotion_dit_Unification_jianhua0803 import DitTalkingHead$",
        f"from src.modules.{module_name} import DitTalkingHead",
        "active model import",
        flags=re.MULTILINE,
    )
    text = regex_once(
        text,
        r'^g_exp_name = ".*"$',
        f'g_exp_name = "{experiment_name}"',
        "experiment name",
        flags=re.MULTILINE,
    )
    text = replace_once(
        text,
        "from src.dataset.dataset_EmotionLevel_clear_jianhua0803 import EmoLevelDataset\n",
        "from src.dataset.dataset_EmotionLevel_clear_jianhua0803 import EmoLevelDataset\nfrom src.seed import set_global_seed\n",
        "seed import",
    )

    start_marker = "        # ---------- 模型前向：单数轮次不传 prev，双数轮次传 GT prev ----------\n"
    end_marker = "        # ---------- 损失 ----------\n"
    text = replace_region(text, start_marker, end_marker, COMMON_FORWARD_BLOCK + end_marker, "context audio block")

    kwargs_start = text.find("    model_kwargs = dict(")
    kwargs_end = text.find("    )\n\n    model = DitTalkingHead", kwargs_start)
    if kwargs_start < 0 or kwargs_end < 0:
        raise RuntimeError("model_kwargs block was not found")
    additions = """        n_heads             = args.n_heads,
        n_layers            = args.n_layers,
        mlp_ratio           = args.mlp_ratio,
        use_indicator       = args.use_indicator,
        no_use_learnable_pe = args.no_use_learnable_pe,
"""
    text = text[:kwargs_end] + additions + text[kwargs_end:]

    text = replace_once(
        text,
        "    classifier.eval()\n",
        "    classifier.eval()\n    classifier.requires_grad_(False)\n",
        "freeze classifier",
    )
    text = replace_once(
        text,
        "    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])\n",
        "    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])\n    parser.add_argument('--seed', type=int, default=2026)\n",
        "seed argument",
    )
    text = replace_once(
        text,
        "    args = parser.parse_args()\n\n    if args.mode == 'train':",
        "    args = parser.parse_args()\n    set_global_seed(args.seed)\n\n    if args.mode == 'train':",
        "apply global seed",
    )

    accumulation = re.compile(
        r"        loss\.backward\(\).*?\n\n"
        r"        if args\.clip_grad:.*?\n"
        r"            torch\.nn\.utils\.clip_grad_norm_\(model\.parameters\(\), max_norm=2\.0\).*?\n\n"
        r"        if it % args\.gradient_accumulation_steps == 0:.*?\n"
        r"            optimizer\.step\(\)\n"
        r"            optimizer\.zero_grad\(\)",
        flags=re.DOTALL,
    )
    replacement = """        (loss / args.gradient_accumulation_steps).backward()
        micro_step = it - start_iter + 1
        should_step = (
            micro_step % args.gradient_accumulation_steps == 0
            or it == args.max_iter
        )
        if should_step:
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)"""
    text, count = accumulation.subn(replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"gradient accumulation block: expected one match, found {count}")
    text = replace_once(
        text,
        "    optimizer.zero_grad()\n    for it in range(",
        "    optimizer.zero_grad(set_to_none=True)\n    for it in range(",
        "initial zero_grad",
    )
    text = text.replace(
        "        if scheduler is not None:\n            scheduler.step()",
        "        if scheduler is not None and should_step:\n            scheduler.step()",
        1,
    )
    return text


def add_lip_model(text: str) -> str:
    text = replace_once(
        text,
        "from ..config.base_config import make_abs_path\n",
        "from ..config.base_config import make_abs_path\n" + LIP_CONSTANTS,
        "lip constants",
    )
    old = """        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),
        )

        self.to(device)"""
    new = """        self.motion_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, self.motion_feat_dim),
        )
        self.lip_residual_dec = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Linear(self.feature_dim // 2, len(LIP_DIM_INDICES)),
        )
        nn.init.zeros_(self.lip_residual_dec[-1].weight)
        nn.init.zeros_(self.lip_residual_dec[-1].bias)

        self.to(device)"""
    text = replace_once(text, old, new, "lip residual decoder")
    text = replace_once(
        text,
        """        motion_feat_target = self.motion_dec(feat_out)
        return motion_feat_target""",
        """        motion_feat_target = self.motion_dec(feat_out)
        lip_residual = self.lip_residual_dec(feat_out)
        motion_feat_target = motion_feat_target.clone()
        motion_feat_target[..., list(LIP_DIM_INDICES)] += lip_residual
        return motion_feat_target""",
        "lip residual output",
    )
    return text


LIP_TRAIN_HELPERS = LIP_CONSTANTS + """

def compute_lip_losses(target, current_motion, prev_motion, n_prev):
    pred = target[:, n_prev:, :63][..., list(LIP_DIM_INDICES)]
    gt = current_motion[:, :, :63][..., list(LIP_DIM_INDICES)]
    loss_pos = torch.nn.functional.smooth_l1_loss(pred, gt)
    loss_vel = torch.nn.functional.smooth_l1_loss(
        pred[:, 1:] - pred[:, :-1], gt[:, 1:] - gt[:, :-1]
    )
    pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    gt_acc = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
    loss_acc = torch.nn.functional.smooth_l1_loss(pred_acc, gt_acc)
    if prev_motion is None:
        loss_boundary = pred.new_zeros(())
    else:
        prev_last = prev_motion[:, -1, :63][..., list(LIP_DIM_INDICES)]
        loss_boundary = torch.nn.functional.smooth_l1_loss(
            pred[:, 0] - prev_last, gt[:, 0] - prev_last
        )
    return loss_pos, loss_vel, loss_acc, loss_boundary
"""


def replace_emotion_with_lip_stop_gradient(text: str) -> str:
    start = "        # 情感分类损失：取 target 的后 64 帧（即去噪后的 current 段）的 exp 系数\n"
    end = "        # 累加各项损失（每轮只用一 clip，所以无需 /2）\n"
    block = """        # 情感分类损失不再向嘴部维度反传，避免情感张嘴与音素闭合冲突。
        exps = target[:, args.n_prev_motions:, :63].clone()
        emotion_grad_mask = torch.ones_like(exps)
        emotion_grad_mask[..., list(LIP_DIM_INDICES)] = 0
        emotion_input = exps.detach() + emotion_grad_mask * (exps - exps.detach())
        pred_emo, _ = classifier(emotion_input)
        loss_emo = cross_criterion(pred_emo, emo_index)

"""
    return replace_region(text, start, end, block + end, "emotion lip stop-gradient")


def add_lip_train(text: str) -> str:
    text = replace_once(
        text,
        "cross_criterion = torch.nn.CrossEntropyLoss()\n",
        "cross_criterion = torch.nn.CrossEntropyLoss()\n" + LIP_TRAIN_HELPERS,
        "lip training helpers",
    )
    text = replace_emotion_with_lip_stop_gradient(text)
    text = replace_once(
        text,
        "        loss_exp_smooth = loss_exp_s\n",
        """        loss_exp_smooth = loss_exp_s
        loss_lip_pos, loss_lip_vel, loss_lip_acc, loss_lip_boundary = compute_lip_losses(
            target, current_motion, prev_motion_for_loss, args.n_prev_motions
        )
""",
        "compute lip losses",
    )
    text = replace_once(
        text,
        """        loss_log['exp_smooth'].append(loss_exp_smooth.item() * args.l_exp_smooth)  # l_exp_smooth： 1e-4  权重
        loss = loss + args.l_exp_smooth * loss_exp_smooth""",
        """        loss_log['exp_smooth'].append(loss_exp_smooth.item() * args.l_exp_smooth)
        loss = loss + args.l_exp_smooth * loss_exp_smooth

        loss_log['lip_pos'].append(loss_lip_pos.item() * args.l_lip_pos)
        loss_log['lip_vel'].append(loss_lip_vel.item() * args.l_lip_vel)
        loss_log['lip_acc'].append(loss_lip_acc.item() * args.l_lip_acc)
        loss_log['lip_boundary'].append(loss_lip_boundary.item() * args.l_lip_boundary)
        loss = loss + args.l_lip_pos * loss_lip_pos
        loss = loss + args.l_lip_vel * loss_lip_vel
        loss = loss + args.l_lip_acc * loss_lip_acc
        loss = loss + args.l_lip_boundary * loss_lip_boundary""",
        "add lip losses",
    )
    text = replace_once(
        text,
        "    parser.add_argument('--l_exp_smooth', type=float, default=1e-4, help='weight of the head angle loss')  \n",
        """    parser.add_argument('--l_exp_smooth', type=float, default=1e-4, help='weight of the head angle loss')  
    parser.add_argument('--l_lip_pos', type=float, default=2.0)
    parser.add_argument('--l_lip_vel', type=float, default=0.75)
    parser.add_argument('--l_lip_acc', type=float, default=0.10)
    parser.add_argument('--l_lip_boundary', type=float, default=0.50)
""",
        "lip loss arguments",
    )
    return text


AUDIO_PYRAMID_CLASS = """

class AudioTemporalPyramid(nn.Module):
    """Frame-preserving multi-scale temporal refinement for speech features."""

    def __init__(self, feature_dim):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size,
                      padding=kernel_size // 2, groups=feature_dim)
            for kernel_size in (3, 5, 9)
        ])
        self.fuse = nn.Conv1d(3 * feature_dim, feature_dim, kernel_size=1)
        self.gate = nn.Sequential(
            nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1, feature_dim, 1))

    def forward(self, audio_feat):
        x = audio_feat.transpose(1, 2)
        context = self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))
        gate = self.gate(torch.cat([x, context], dim=1))
        refined = x + torch.tanh(self.residual_scale) * gate * context
        return refined.transpose(1, 2)
"""


def add_audio_pyramid_model(text: str) -> str:
    text = insert_before(text, "\n\nclass DiTDecoderLayer", AUDIO_PYRAMID_CLASS, "audio pyramid class")
    text = replace_once(
        text,
        "            self.audio_feature_map = nn.Linear(768, feature_dim)\n",
        "            self.audio_feature_map = nn.Linear(768, feature_dim)\n            self.audio_temporal_pyramid = AudioTemporalPyramid(feature_dim)\n",
        "audio pyramid initialization",
    )
    text = replace_once(
        text,
        """        audio_feat = self.audio_feature_map(hidden_states)
        return audio_feat""",
        """        audio_feat = self.audio_feature_map(hidden_states)
        audio_feat = self.audio_temporal_pyramid(audio_feat)
        return audio_feat""",
        "audio pyramid application",
    )
    return text


AUDIO_TRAIN_HELPERS = """

def freeze_audio_encoder(model):
    for parameter in model.audio_encoder.parameters():
        parameter.requires_grad_(False)


def unfreeze_audio_top_layers(model, num_layers):
    encoder = getattr(model.audio_encoder, 'encoder', None)
    layers = getattr(encoder, 'layers', None)
    if layers is None:
        for parameter in model.audio_encoder.parameters():
            parameter.requires_grad_(True)
        return
    for layer in layers[-max(1, num_layers):]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)
"""


def add_audio_pyramid_train(text: str) -> str:
    text = replace_once(
        text,
        "cross_criterion = torch.nn.CrossEntropyLoss()\n",
        "cross_criterion = torch.nn.CrossEntropyLoss()\n" + AUDIO_TRAIN_HELPERS,
        "audio training helpers",
    )
    old_optimizer = "    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)  # 选取需要训练的部分  lr=1e-4"
    new_optimizer = """    freeze_audio_encoder(model)
    audio_params = list(model.audio_encoder.parameters())
    audio_param_ids = {id(parameter) for parameter in audio_params}
    main_params = [
        parameter for parameter in model.parameters()
        if id(parameter) not in audio_param_ids
    ]
    optimizer = torch.optim.AdamW([
        {'params': main_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': audio_params, 'lr': args.audio_lr, 'weight_decay': args.weight_decay},
    ], betas=(0.9, 0.95))"""
    text = replace_once(text, old_optimizer, new_optimizer, "audio optimizer")
    text = replace_once(
        text,
        "    optimizer.zero_grad(set_to_none=True)\n    for it in range(",
        """    optimizer.zero_grad(set_to_none=True)
    audio_unfrozen = False
    if start_iter >= args.audio_freeze_iter:
        unfreeze_audio_top_layers(model, args.audio_unfreeze_layers)
        audio_unfrozen = True
    for it in range(""",
        "audio unfreeze state",
    )
    text = replace_once(
        text,
        "    for it in range(start_iter, args.max_iter + 1):   # 迭代次数  0 ~ max_iter\n",
        """    for it in range(start_iter, args.max_iter + 1):   # 迭代次数  0 ~ max_iter
        if not audio_unfrozen and it >= args.audio_freeze_iter:
            unfreeze_audio_top_layers(model, args.audio_unfreeze_layers)
            audio_unfrozen = True
            logging.info(
                f'Unfroze top {args.audio_unfreeze_layers} audio layers at iter {it}.'
            )
""",
        "staged audio unfreezing",
    )
    text = replace_once(
        text,
        "    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')\n",
        """    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--audio_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--audio_freeze_iter', type=int, default=40000)
    parser.add_argument('--audio_unfreeze_layers', type=int, default=4)
""",
        "audio optimizer arguments",
    )
    return text


def add_channel_gate_model(text: str) -> str:
    text = replace_once(
        text,
        """            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emotion_feat_dim, 2 * emotion_feat_dim, bias=True),
            )

        self.to(device)""",
        """            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emotion_feat_dim, 2 * emotion_feat_dim, bias=True),
            )
            self.emotion_channel_gate = nn.Parameter(
                torch.zeros(1, 1, emotion_feat_dim)
            )

        self.to(device)""",
        "emotion channel gate parameter",
    )
    class_start = text.find("class DitTalkingHead")
    property_index = text.find("    @property\n    def device", class_start)
    if property_index < 0:
        raise RuntimeError("DitTalkingHead device property was not found")
    method = """    def apply_emotion_modulation(self, audio_feat, shift, scale):
        normalized = self.audio_norm(audio_feat)
        gate = 2.0 * torch.sigmoid(self.emotion_channel_gate)
        return normalized * (1 + gate * scale) + gate * shift

"""
    text = text[:property_index] + method + text[property_index:]

    replacements = {
        "self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift": "self.apply_emotion_modulation(audio_feat, emo_shift, emo_scale)",
        "self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift": "self.apply_emotion_modulation(prev_audio_feat, emo_shift, emo_scale)",
        "self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift": "self.apply_emotion_modulation(null_audio_feat, null_shift, null_scale)",
        "self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift": "self.apply_emotion_modulation(audio_feat_saved, emo_shift, emo_scale)",
    }
    total = 0
    for old, new in replacements.items():
        count = text.count(old)
        total += count
        text = text.replace(old, new)
    if total < 7:
        raise RuntimeError(f"emotion modulation replacements: expected at least 7, found {total}")
    return text


GATE_TRAIN_HELPERS = LIP_CONSTANTS


def add_channel_gate_train(text: str) -> str:
    text = replace_once(
        text,
        "cross_criterion = torch.nn.CrossEntropyLoss()\n",
        "cross_criterion = torch.nn.CrossEntropyLoss()\n" + GATE_TRAIN_HELPERS,
        "gate constants",
    )
    text = replace_emotion_with_lip_stop_gradient(text)
    text = replace_once(
        text,
        """        loss_log['emo'].append(loss_emo.item())
        loss = loss + loss_emo""",
        """        loss_log['emo'].append(loss_emo.item())
        emotion_progress = min(1.0, (it + 1) / max(1, args.emo_warmup_iters))
        emotion_weight = args.l_emo * emotion_progress
        loss = loss + emotion_weight * loss_emo
        gate_mean = (2.0 * torch.sigmoid(model.emotion_channel_gate)).mean()
        loss_gate = (gate_mean - args.gate_target).square()
        loss_log['gate'].append(loss_gate.item() * args.l_gate)
        loss = loss + args.l_gate * loss_gate""",
        "emotion curriculum and gate regularization",
    )
    old_optimizer = "    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)  # 选取需要训练的部分  lr=1e-4"
    new_optimizer = """    emotion_params = (
        list(model.emo_embed.parameters())
        + list(model.adaLN_modulation.parameters())
        + [model.emotion_channel_gate]
    )
    emotion_ids = {id(parameter) for parameter in emotion_params}
    base_params = [
        parameter for parameter in model.parameters()
        if id(parameter) not in emotion_ids
    ]
    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.lr, 'weight_decay': args.weight_decay},
        {'params': emotion_params, 'lr': args.emotion_lr, 'weight_decay': args.weight_decay},
    ], betas=(0.9, 0.95))"""
    text = replace_once(text, old_optimizer, new_optimizer, "channel gate optimizer")
    text = replace_once(
        text,
        "    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')\n",
        """    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--emotion_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--l_emo', type=float, default=1.0)
    parser.add_argument('--emo_warmup_iters', type=int, default=30000)
    parser.add_argument('--l_gate', type=float, default=1e-3)
    parser.add_argument('--gate_target', type=float, default=1.0)
""",
        "channel gate arguments",
    )
    return text


MIN_SNR_HELPERS = """

def stratified_diffusion_timesteps(batch_size, num_steps, device):
    u = (torch.arange(batch_size, device=device) + torch.rand(batch_size, device=device)) / batch_size
    steps = torch.clamp((u * num_steps).long() + 1, max=num_steps)
    return steps[torch.randperm(batch_size, device=device)]


def min_snr_primary_loss(args, model, is_starting_sample, motion_gt, noise,
                         prediction, prev_motion, end_idx, time_step):
    if args.target == 'noise':
        pred = prediction[:, args.n_prev_motions:]
        gt = noise
        mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    elif args.target == 'sample':
        if is_starting_sample:
            pred = prediction[:, args.n_prev_motions:]
            gt = motion_gt
            mask = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
        else:
            pred = prediction
            gt = torch.cat([prev_motion, motion_gt], dim=1)
            prev_mask = torch.zeros(
                pred.shape[0], args.n_prev_motions,
                dtype=torch.bool, device=pred.device,
            ) if args.no_constrain_prev else torch.ones(
                pred.shape[0], args.n_prev_motions,
                dtype=torch.bool, device=pred.device,
            )
            mask = torch.cat([
                prev_mask,
                torch.ones(pred.shape[0], args.n_motions, dtype=torch.bool, device=pred.device),
            ], dim=1)
    else:
        raise ValueError(f'Unsupported target for Min-SNR: {args.target}')

    if end_idx is not None:
        current_mask = torch.arange(args.n_motions, device=pred.device).expand(
            pred.shape[0], -1
        ) < end_idx.unsqueeze(1)
        if args.target == 'sample' and not is_starting_sample:
            mask = torch.cat([mask[:, :args.n_prev_motions], current_mask], dim=1)
        else:
            mask = current_mask

    elementwise = (pred - gt).square() if args.criterion == 'l2' else (pred - gt).abs()
    valid = mask.unsqueeze(-1).to(elementwise.dtype)
    denom = valid.sum(dim=(1, 2)).clamp_min(1.0) * elementwise.shape[-1]
    per_sample = (elementwise * valid).sum(dim=(1, 2)) / denom

    alpha_bar = model.diffusion_sched.alpha_bars[time_step]
    snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
    gamma = torch.full_like(snr, args.min_snr_gamma)
    if args.target == 'sample':
        weight = torch.minimum(snr, gamma)
    else:
        weight = torch.minimum(snr, gamma) / snr.clamp_min(1e-8)
    weight = weight / weight.mean().detach().clamp_min(1e-8)
    return (per_sample * weight).mean()


@torch.no_grad()
def update_ema(ema_model, model, decay):
    for ema_parameter, parameter in zip(ema_model.parameters(), model.parameters()):
        ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)
"""


def add_minsnr_ema_train(text: str) -> str:
    text = replace_once(text, "import argparse\n", "import argparse\nimport copy\n", "copy import")
    text = replace_once(
        text,
        "cross_criterion = torch.nn.CrossEntropyLoss()\n",
        "cross_criterion = torch.nn.CrossEntropyLoss()\n" + MIN_SNR_HELPERS,
        "Min-SNR helpers",
    )
    text = replace_once(
        text,
        "def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None):",
        "def train(args, model, train_loader, val_loader, optimizer, save_dir, scheduler=None, writer=None, start_iter=0, classifier=None, ema_model=None):",
        "EMA train signature",
    )
    text = replace_once(
        text,
        "        batch_size = audio.shape[0]\n",
        """        batch_size = audio.shape[0]
        time_step = stratified_diffusion_timesteps(
            batch_size, model.diffusion_sched.num_steps, device
        )
""",
        "stratified timestep sampling",
    )
    call_token = "                indicator=indicator, emo_index=emo_index,\n"
    if text.count(call_token) < 2:
        raise RuntimeError("Expected two training model calls for Min-SNR")
    text = text.replace(
        call_token,
        "                time_step=time_step, indicator=indicator, emo_index=emo_index,\n",
        2,
    )
    loss_call = """        loss_n, loss_exp, loss_exp_v, loss_exp_s, loss_ha, loss_hc, loss_hs, loss_ht = utils.compute_loss_new(
            args, is_starting_sample, current_motion, noise, target, prev_motion_for_loss, end_idx,
        )
"""
    text = replace_once(
        text,
        loss_call,
        loss_call + """        loss_n = min_snr_primary_loss(
            args, model, is_starting_sample, current_motion, noise, target,
            prev_motion_for_loss, end_idx, time_step,
        )
""",
        "Min-SNR primary loss",
    )
    text = replace_once(
        text,
        "    model = DitTalkingHead(**model_kwargs)           \n",
        """    model = DitTalkingHead(**model_kwargs)
    ema_model = copy.deepcopy(model).eval()
    ema_model.requires_grad_(False)
""",
        "EMA model creation",
    )
    text = replace_once(
        text,
        """            optimizer.step()
            optimizer.zero_grad(set_to_none=True)""",
        """            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                update_ema(ema_model, model, args.ema_decay)""",
        "EMA update",
    )
    text = replace_once(
        text,
        "                'model': model.state_dict(),   # 模型\n",
        "                'model': (ema_model.state_dict() if ema_model is not None else model.state_dict()),\n                'model_raw': model.state_dict(),\n",
        "EMA checkpoint state",
    )
    text = replace_once(
        text,
        """          start_iter=start_iter,
          classifier=classifier)""",
        """          start_iter=start_iter,
          classifier=classifier,
          ema_model=ema_model)""",
        "EMA train call",
    )
    old_optimizer = "    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)  # 选取需要训练的部分  lr=1e-4"
    text = replace_once(
        text,
        old_optimizer,
        """    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )""",
        "Min-SNR optimizer",
    )
    text = replace_once(
        text,
        "    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')\n",
        """    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--min_snr_gamma', type=float, default=5.0)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
""",
        "Min-SNR arguments",
    )
    return text


DOCS = """# ADEF 模型与训练优化变体

四组实验均由 0803 完整模型文件和完整训练脚本物理复制后直接修改得到。每个模型文件都包含 `DiffusionSchedule`、`DiTDecoderLayer`、`DiTDecoder`、`DenoisingNetwork` 与 `DitTalkingHead` 的完整实现，不通过导入、继承或 wrapper 复用样例实现。

## 1. Lip-aware residual + 嘴部专项监督

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_lipaware.py`
- 训练：`train_Unification_jianhua0803_lipaware.py`
- 在共享 DiT 输出上增加零初始化的嘴部残差解码头，只修正六个嘴部隐式关键点对应的 18 个通道。
- 增加嘴部位置、速度、GT 加速度和窗口边界 Huber 损失。
- 情感分类损失不再向嘴部通道传梯度，减少情感张嘴与音素闭合之间的竞争。
- 这是最直接面向唇形同步的首选实验。

## 2. Multi-scale audio temporal pyramid + 分层微调

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_audio_pyramid.py`
- 训练：`train_Unification_jianhua0803_audio_pyramid.py`
- 在 Wav2Vec2/HuBERT 投影后加入 3、5、9 帧的深度卷积时序金字塔，保持输出帧率和接口不变。
- 残差尺度零初始化，训练起点等价于原音频特征。
- 前期冻结完整音频编码器，随后只解冻顶部若干 Transformer 层，并使用较低的音频学习率。

## 3. Learnable emotion channel gate + 情感课程学习

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_channelgate.py`
- 训练：`train_Unification_jianhua0803_channelgate.py`
- 保留“情感调制音频”核心逻辑，为每个音频特征通道学习 0 到 2 倍的调制强度，初始化时严格等价于原调制。
- 情感损失线性 warm-up，情感模块采用独立较小学习率。
- 同样阻断情感分类损失对嘴部通道的梯度，重点验证语音内容与情感表达的 Pareto 改善。

## 4. Stratified Min-SNR + AdamW + EMA

- 模型：`src/modules/emotion_dit_Unification_jianhua0803_minsnr_ema.py`
- 训练：`train_Unification_jianhua0803_minsnr_ema.py`
- 不改变网络主干，使用分层扩散时间步采样和适配 sample/noise target 的 Min-SNR 主损失。
- 优化器改为 AdamW，并维护 EMA 模型；checkpoint 的 `model` 保存 EMA 参数，`model_raw` 保存即时训练参数。
- 用于单独检验扩散优化稳定性，避免把结构收益与训练配方混在一起。

## 所有方案共有的修正

- `n_heads`、`n_layers`、`mlp_ratio`、`use_indicator` 与位置编码配置直接下传到 `DenoisingNetwork`。
- `sample()` 第三个返回值统一为未经情感调制的 `audio_feat_saved`。
- continuation 阶段将 prev 16 + current 64 波形合并后仅编码一次，再按帧切分。
- 冻结情感分类器参数；增加全局 seed 参数。
- 梯度累积时先除以 accumulation steps，只在真正 optimizer step 时裁剪梯度和更新 scheduler。

## 推荐顺序

1. Lip-aware；
2. Audio pyramid；
3. Channel gate；
4. Min-SNR + EMA。

先保持数据划分、batch size、训练步数、CFG 和随机种子一致，分别单独训练。不要一开始合并四组修改，否则无法判断指标提升来自哪里。
"""


def validate(path: Path, content: str, is_model: bool) -> None:
    ast.parse(content, filename=str(path))
    forbidden = (
        "from . import emotion_dit_Unification_jianhua0803_legacy",
        "from .emotion_dit_Unification_jianhua0803 import",
        "DitTalkingHead as _BaseDitTalkingHead",
    )
    if any(token in content for token in forbidden):
        raise RuntimeError(f"{path} imports or inherits the sample implementation")
    minimum = 700 if is_model else 500
    if len(content.splitlines()) < minimum:
        raise RuntimeError(f"{path} is not a complete copy: too few lines")
    if is_model:
        required = (
            "class DiffusionSchedule",
            "class DiTDecoderLayer",
            "class DiTDecoder",
            "class DenoisingNetwork",
            "class DitTalkingHead",
        )
        if not all(token in content for token in required):
            raise RuntimeError(f"{path} is missing a complete model component")


def main() -> None:
    if not BASE_MODEL.exists() or not BASE_TRAIN.exists():
        raise FileNotFoundError("0803 full sample model/training files were not found")
    base_model = BASE_MODEL.read_text(encoding="utf-8")
    base_train = BASE_TRAIN.read_text(encoding="utf-8")

    specifications = []

    module = "emotion_dit_Unification_jianhua0803_lipaware"
    specifications.append((
        Path(f"src/modules/{module}.py"),
        add_lip_model(prepare_model(base_model)),
        True,
    ))
    specifications.append((
        Path("train_Unification_jianhua0803_lipaware.py"),
        add_lip_train(prepare_train(base_train, module, "20260827_opt_lipaware_residual_losses")),
        False,
    ))

    module = "emotion_dit_Unification_jianhua0803_audio_pyramid"
    specifications.append((
        Path(f"src/modules/{module}.py"),
        add_audio_pyramid_model(prepare_model(base_model)),
        True,
    ))
    specifications.append((
        Path("train_Unification_jianhua0803_audio_pyramid.py"),
        add_audio_pyramid_train(prepare_train(base_train, module, "20260827_opt_audio_pyramid_finetune")),
        False,
    ))

    module = "emotion_dit_Unification_jianhua0803_channelgate"
    specifications.append((
        Path(f"src/modules/{module}.py"),
        add_channel_gate_model(prepare_model(base_model)),
        True,
    ))
    specifications.append((
        Path("train_Unification_jianhua0803_channelgate.py"),
        add_channel_gate_train(prepare_train(base_train, module, "20260827_opt_channelgate_curriculum")),
        False,
    ))

    module = "emotion_dit_Unification_jianhua0803_minsnr_ema"
    specifications.append((
        Path(f"src/modules/{module}.py"),
        prepare_model(base_model),
        True,
    ))
    specifications.append((
        Path("train_Unification_jianhua0803_minsnr_ema.py"),
        add_minsnr_ema_train(prepare_train(base_train, module, "20260827_opt_minsnr_ema")),
        False,
    ))

    for path, content, is_model in specifications:
        validate(path, content, is_model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"generated {path} ({len(content.splitlines())} lines)")

    docs_path = Path("docs/MODEL_TRAINING_OPTIMIZATION_VARIANTS.md")
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(textwrap.dedent(DOCS).strip() + "\n", encoding="utf-8")
    print(f"generated {docs_path}")


if __name__ == "__main__":
    main()

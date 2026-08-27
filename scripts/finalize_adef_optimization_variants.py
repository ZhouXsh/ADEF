from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

MODEL_FILES = [
    Path('src/modules/emotion_dit_Unification_jianhua0803_lipaware.py'),
    Path('src/modules/emotion_dit_Unification_jianhua0803_audio_pyramid.py'),
    Path('src/modules/emotion_dit_Unification_jianhua0803_channelgate.py'),
    Path('src/modules/emotion_dit_Unification_jianhua0803_minsnr_ema.py'),
]

TRAIN_FILES = [
    Path('train_Unification_jianhua0803_lipaware.py'),
    Path('train_Unification_jianhua0803_audio_pyramid.py'),
    Path('train_Unification_jianhua0803_channelgate.py'),
    Path('train_Unification_jianhua0803_minsnr_ema.py'),
]

DOC_PATH = Path('docs/MODEL_TRAINING_OPTIMIZATION_VARIANTS.md')


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, found {count}')
    return text.replace(old, new, 1)


def fix_model(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    old_pe = 'torch.randn(1, 1 + self.n_prev_motions + self.n_motions, self.feature_dim)'
    new_pe = 'torch.randn(1, self.n_prev_motions + self.n_motions, self.feature_dim)'
    text = replace_once(text, old_pe, new_pe, f'{path}: learnable PE length')
    ast.parse(text, filename=str(path))
    path.write_text(text, encoding='utf-8')


def add_optimizer_update(text: str, path: Path) -> str:
    old = '''        micro_step = it - start_iter + 1
        should_step = (
            micro_step % args.gradient_accumulation_steps == 0
            or it == args.max_iter
        )'''
    new = '''        micro_step = it - start_iter + 1
        should_step = (
            micro_step % args.gradient_accumulation_steps == 0
            or it == args.max_iter
        )
        optimizer_update = (
            micro_step + args.gradient_accumulation_steps - 1
        ) // args.gradient_accumulation_steps'''
    return replace_once(text, old, new, f'{path}: optimizer update counter')


def fix_scheduler(text: str, path: Path) -> str:
    old = '''        # update learning rate  更新学习率
        if scheduler is not None:   # 调度器用于更新学习率  区分：优化器optimizor
            if args.scheduler != 'WarmupThenDecay' or (args.scheduler == 'WarmupThenDecay' and it < args.cos_max_iter):
                scheduler.step()'''
    new = '''        # update learning rate only after a real optimizer update
        if scheduler is not None and should_step:
            if args.scheduler != 'WarmupThenDecay' or optimizer_update < args.cos_max_iter:
                scheduler.step()'''
    return replace_once(text, old, new, f'{path}: scheduler cadence')


def add_model_metadata(text: str, path: Path) -> str:
    marker = '    model = DitTalkingHead(**model_kwargs)'
    index = text.find(marker)
    if index < 0:
        raise RuntimeError(f'{path}: model construction was not found')
    line_end = text.find('\n', index)
    if line_end < 0:
        raise RuntimeError(f'{path}: malformed model construction line')
    addition = "\n    args.model_module = DitTalkingHead.__module__\n    args.optimization_variant = Path(__file__).stem\n"
    return text[:line_end] + addition + text[line_end:]


MASKED_LIP_FUNCTION = '''def _masked_smooth_l1(prediction, target, valid_mask):
    elementwise = torch.nn.functional.smooth_l1_loss(
        prediction, target, reduction='none'
    ).mean(dim=-1)
    if valid_mask.any():
        return elementwise[valid_mask].mean()
    return prediction.new_zeros(())


def compute_lip_losses(target, current_motion, prev_motion, n_prev, end_idx=None):
    pred = target[:, n_prev:, :63][..., list(LIP_DIM_INDICES)]
    gt = current_motion[:, :, :63][..., list(LIP_DIM_INDICES)]

    if end_idx is None:
        valid = torch.ones(pred.shape[:2], dtype=torch.bool, device=pred.device)
    else:
        valid = torch.arange(pred.shape[1], device=pred.device).expand(
            pred.shape[0], -1
        ) < end_idx.unsqueeze(1)

    loss_pos = _masked_smooth_l1(pred, gt, valid)

    velocity_valid = valid[:, 1:] & valid[:, :-1]
    loss_vel = _masked_smooth_l1(
        pred[:, 1:] - pred[:, :-1],
        gt[:, 1:] - gt[:, :-1],
        velocity_valid,
    )

    acceleration_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    pred_acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    gt_acc = gt[:, 2:] - 2 * gt[:, 1:-1] + gt[:, :-2]
    loss_acc = _masked_smooth_l1(pred_acc, gt_acc, acceleration_valid)

    if prev_motion is None:
        loss_boundary = pred.new_zeros(())
    else:
        prev_last = prev_motion[:, -1, :63][..., list(LIP_DIM_INDICES)]
        boundary_error = torch.nn.functional.smooth_l1_loss(
            pred[:, 0] - prev_last,
            gt[:, 0] - prev_last,
            reduction='none',
        ).mean(dim=-1)
        boundary_valid = valid[:, 0]
        loss_boundary = (
            boundary_error[boundary_valid].mean()
            if boundary_valid.any()
            else pred.new_zeros(())
        )

    return loss_pos, loss_vel, loss_acc, loss_boundary


'''


def fix_lip_masking(text: str, path: Path) -> str:
    start = text.find('def compute_lip_losses(')
    end = text.find('def train(', start)
    if start < 0 or end < 0:
        raise RuntimeError(f'{path}: lip helper region was not found')
    text = text[:start] + MASKED_LIP_FUNCTION + text[end:]
    old_call = '''        loss_lip_pos, loss_lip_vel, loss_lip_acc, loss_lip_boundary = compute_lip_losses(
            target, current_motion, prev_motion_for_loss, args.n_prev_motions
        )'''
    new_call = '''        loss_lip_pos, loss_lip_vel, loss_lip_acc, loss_lip_boundary = compute_lip_losses(
            target, current_motion, prev_motion_for_loss,
            args.n_prev_motions, end_idx=end_idx,
        )'''
    return replace_once(text, old_call, new_call, f'{path}: masked lip loss call')


def fix_train(path: Path) -> None:
    text = path.read_text(encoding='utf-8')
    text = add_optimizer_update(text, path)
    text = fix_scheduler(text, path)
    text = add_model_metadata(text, path)
    text = text.replace(
        '        # ---------- 损失 ----------\n        # ---------- 损失 ----------\n',
        '        # ---------- 损失 ----------\n',
        1,
    )
    if path.name.endswith('_lipaware.py'):
        text = fix_lip_masking(text, path)
    ast.parse(text, filename=str(path))
    path.write_text(text, encoding='utf-8')


def validate_full_copy(path: Path, is_model: bool) -> None:
    text = path.read_text(encoding='utf-8')
    forbidden = (
        'from . import emotion_dit_Unification_jianhua0803_legacy',
        'from .emotion_dit_Unification_jianhua0803 import',
        'DitTalkingHead as _BaseDitTalkingHead',
    )
    if any(token in text for token in forbidden):
        raise RuntimeError(f'{path}: variant imports or inherits a sample implementation')
    minimum = 700 if is_model else 500
    if len(text.splitlines()) < minimum:
        raise RuntimeError(f'{path}: file is not a complete physical copy')
    if is_model:
        required = (
            'class DiffusionSchedule',
            'class DiTDecoderLayer',
            'class DiTDecoder',
            'class DenoisingNetwork',
            'class DitTalkingHead',
        )
        if not all(token in text for token in required):
            raise RuntimeError(f'{path}: missing a core model class')
        if '1 + self.n_prev_motions + self.n_motions' in text:
            raise RuntimeError(f'{path}: learnable PE still has an extra token')
        if 'return traj[0], motion_at_T, audio_feat_saved' not in text:
            raise RuntimeError(f'{path}: sample does not return raw audio features')
    else:
        if 'if scheduler is not None and should_step:' not in text:
            raise RuntimeError(f'{path}: scheduler is not tied to optimizer updates')
        if 'args.model_module = DitTalkingHead.__module__' not in text:
            raise RuntimeError(f'{path}: checkpoint model-module metadata is missing')


def update_docs() -> None:
    text = DOC_PATH.read_text(encoding='utf-8')
    addition = '''

## 运行注意事项

- 四个训练脚本都会把 `args.model_module` 和 `args.optimization_variant` 写入 checkpoint，便于后续按对应的完整模型文件恢复。
- 四组模型均修正了 learnable positional encoding 的长度：位置参数与 `n_prev_motions + n_motions` 完全一致，不再多出一个未使用 token。
- 嘴部专项损失会依据 `end_idx` 屏蔽随机截断后的 padding 帧；速度和加速度损失使用相应的相邻有效帧掩码。
- 当 `gradient_accumulation_steps > 1` 时，scheduler 只在真实 `optimizer.step()` 后更新。
- Lip-aware、Audio pyramid 和 Channel gate 默认面向 `target=sample`；Min-SNR + EMA 同时兼容当前 `sample` 与 `noise` 目标，但建议先以 `sample` 做可比实验。
'''
    if '## 运行注意事项' not in text:
        text = text.rstrip() + textwrap.dedent(addition) + '\n'
    DOC_PATH.write_text(text, encoding='utf-8')


def main() -> None:
    for path in MODEL_FILES + TRAIN_FILES + [DOC_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in MODEL_FILES:
        fix_model(path)
    for path in TRAIN_FILES:
        fix_train(path)
    update_docs()
    for path in MODEL_FILES:
        validate_full_copy(path, is_model=True)
    for path in TRAIN_FILES:
        validate_full_copy(path, is_model=False)
    print('Finalized and validated all ADEF optimization variants.')


if __name__ == '__main__':
    main()

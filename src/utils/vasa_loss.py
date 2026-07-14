import torch
import torch.nn.functional as F


def _criterion(args):
    if args.criterion.lower() == 'l2':
        return F.mse_loss
    if args.criterion.lower() == 'l1':
        return F.l1_loss
    raise NotImplementedError(f'Criterion {args.criterion} not implemented.')


def _masked_mean(value, mask):
    selected = value[mask]
    return selected.mean() if selected.numel() > 0 else None


def _temporal_masks(current_mask, previous_valid, history_len=2):
    history_mask = previous_valid[:, None].expand(-1, history_len)
    sequence_mask = torch.cat([history_mask, current_mask], dim=1)
    velocity_mask = sequence_mask[:, 1:] & sequence_mask[:, :-1]
    smooth_mask = sequence_mask[:, 2:] & sequence_mask[:, 1:-1] & sequence_mask[:, :-2]
    return velocity_mask, smooth_mask


def compute_loss_vasa(
    args,
    motion_coef_gt,
    noise,
    target,
    prev_motion_coef,
    end_idx=None,
    previous_valid=None,
):
    """Compute losses for a current window conditioned by previous frames.

    The previous frames are conditions only. Direct diffusion and reconstruction
    losses supervise the current window, while temporal losses may cross the
    boundary when the previous condition was visible to the model.
    """
    criterion_func = _criterion(args)
    target_current = target[:, args.n_prev_motions:]
    if previous_valid is None:
        previous_valid = torch.ones(
            target_current.shape[0], dtype=torch.bool, device=target_current.device
        )

    if end_idx is None:
        current_mask = torch.ones(
            target_current.shape[:2], dtype=torch.bool, device=target_current.device
        )
    else:
        current_mask = (
            torch.arange(args.n_motions, device=target_current.device)[None, :]
            < end_idx[:, None]
        )

    empty = (None,) * 7
    if args.target == 'noise':
        loss_noise = _masked_mean(
            criterion_func(noise, target_current, reduction='none'), current_mask
        )
        return (loss_noise,) + empty
    if args.target != 'sample':
        raise ValueError(f'Unknown diffusion target: {args.target}')

    loss_noise = _masked_mean(
        criterion_func(motion_coef_gt, target_current, reduction='none'), current_mask
    )

    if args.rot_repr == 'aa':
        exp_gt = motion_coef_gt[..., :63]
        exp_pred = target_current[..., :63]
    elif args.rot_repr == 'emo':
        exp_gt = torch.cat([motion_coef_gt[..., :63], motion_coef_gt[..., -3:]], dim=-1)
        exp_pred = torch.cat([target_current[..., :63], target_current[..., -3:]], dim=-1)
    else:
        raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

    loss_exp = _masked_mean(
        criterion_func(exp_gt, exp_pred, reduction='none'), current_mask
    )
    loss_exp_vel = None
    loss_exp_smooth = None
    loss_head_angle = None
    loss_head_vel = None
    loss_head_smooth = None
    loss_head_trans = None

    history_len = min(2, prev_motion_coef.shape[1])
    velocity_mask, smooth_mask = _temporal_masks(
        current_mask, previous_valid, history_len
    )
    prev_exp = prev_motion_coef[:, -history_len:, :63]
    exp_gt_sequence = torch.cat([prev_exp, exp_gt[..., :63]], dim=1)
    exp_pred_sequence = torch.cat([prev_exp, exp_pred[..., :63]], dim=1)

    if args.l_exp_vel > 0:
        exp_vel_gt = exp_gt_sequence[:, 1:] - exp_gt_sequence[:, :-1]
        exp_vel_pred = exp_pred_sequence[:, 1:] - exp_pred_sequence[:, :-1]
        loss_exp_vel = _masked_mean(
            criterion_func(exp_vel_gt, exp_vel_pred, reduction='none'), velocity_mask
        )
    if args.l_exp_smooth > 0:
        exp_vel_pred = exp_pred_sequence[:, 1:] - exp_pred_sequence[:, :-1]
        exp_accel_pred = exp_vel_pred[:, 1:] - exp_vel_pred[:, :-1]
        loss_exp_smooth = _masked_mean(
            criterion_func(
                exp_accel_pred, torch.zeros_like(exp_accel_pred), reduction='none'
            ),
            smooth_mask,
        )

    if not args.no_head_pose:
        head_pose_gt = motion_coef_gt[..., 63:70]
        head_pose_pred = target_current[..., 63:70]
        if args.l_head_angle > 0:
            loss_head_angle = _masked_mean(
                criterion_func(head_pose_gt, head_pose_pred, reduction='none'),
                current_mask,
            )

        prev_head = prev_motion_coef[:, -history_len:, 63:70]
        head_gt_sequence = torch.cat([prev_head, head_pose_gt], dim=1)
        head_pred_sequence = torch.cat([prev_head, head_pose_pred], dim=1)
        if args.l_head_vel > 0:
            head_vel_gt = head_gt_sequence[:, 1:] - head_gt_sequence[:, :-1]
            head_vel_pred = head_pred_sequence[:, 1:] - head_pred_sequence[:, :-1]
            loss_head_vel = _masked_mean(
                criterion_func(head_vel_gt, head_vel_pred, reduction='none'),
                velocity_mask,
            )
        if args.l_head_smooth > 0:
            head_vel_pred = head_pred_sequence[:, 1:] - head_pred_sequence[:, :-1]
            head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]
            loss_head_smooth = _masked_mean(
                criterion_func(
                    head_accel_pred, torch.zeros_like(head_accel_pred), reduction='none'
                ),
                smooth_mask,
            )

        if args.l_head_trans > 0 and prev_motion_coef.shape[1] >= 3:
            transition = torch.cat(
                [prev_motion_coef[:, -3:, 63:70], head_pose_pred[:, :3]], dim=1
            )
            transition_velocity = transition[:, 1:] - transition[:, :-1]
            transition_accel = transition_velocity[:, 1:] - transition_velocity[:, :-1]
            transition_vel_loss = criterion_func(
                transition_velocity[:, 2:4], transition_velocity[:, 1:3], reduction='none'
            )
            transition_accel_loss = criterion_func(
                transition_accel[:, 1:], transition_accel[:, :-1], reduction='none'
            )
            valid_first = current_mask[:, :3] & previous_valid[:, None]
            vel_value = _masked_mean(transition_vel_loss, valid_first[:, :2])
            accel_value = _masked_mean(transition_accel_loss, valid_first)
            if vel_value is not None and accel_value is not None:
                loss_head_trans = vel_value + accel_value

    return (
        loss_noise,
        loss_exp,
        loss_exp_vel,
        loss_exp_smooth,
        loss_head_angle,
        loss_head_vel,
        loss_head_smooth,
        loss_head_trans,
    )

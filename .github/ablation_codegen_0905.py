from pathlib import Path
import re

ROOT = Path('.')
BASE_TRAIN = ROOT / 'train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py'
BASE_WRAP = ROOT / 'src/modules/emotion_dit_Unification_jianhua0803.py'
BASE_LEG = ROOT / 'src/modules/emotion_dit_Unification_jianhua0803_legacy.py'

base_train = BASE_TRAIN.read_text(encoding='utf-8')
base_wrap = BASE_WRAP.read_text(encoding='utf-8')
base_leg = BASE_LEG.read_text(encoding='utf-8')


def must_replace(text, old, new, expected=None, label='replace'):
    count = text.count(old)
    if expected is not None and count != expected:
        raise RuntimeError(f'{label}: expected {expected} occurrences, found {count}')
    if count == 0:
        raise RuntimeError(f'{label}: pattern not found')
    return text.replace(old, new)


def replace_once(text, old, new, label='replace_once'):
    return must_replace(text, old, new, expected=1, label=label)


def set_variant_constants(text, slug, description, model_module, max_iter=None):
    text = text.replace(
        'from src.modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead',
        f'from src.modules.{model_module} import DitTalkingHead',
    )
    text = re.sub(r'VARIANT_NAME = "[^"]+"', f'VARIANT_NAME = "ablation0905_{slug}"', text, count=1)
    text = re.sub(
        r'VARIANT_DESCRIPTION = \(\n.*?\n\)',
        'VARIANT_DESCRIPTION = (\n    ' + repr(description) + '\n)',
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r'DEFAULT_EXP_NAME = "[^"]+"',
        f'DEFAULT_EXP_NAME = "20260905_ablation_{slug}"',
        text,
        count=1,
    )
    text = text.replace(
        'parser.add_argument("--model_variant", type=str, default="jianhua0803")',
        f'parser.add_argument("--model_variant", type=str, default="{model_module}")',
    )
    if max_iter is not None:
        text = re.sub(r'DEFAULT_MAX_ITER = \d+', f'DEFAULT_MAX_ITER = {max_iter}', text, count=1)
    header = (
        '# ICASSP27 controlled ablation generated from the final 6009 recipe.\n'
        '# This file is a physical copy and never imports another training variant.\n'
    )
    return header + text


def make_model_pair(slug, legacy_transform=None, wrapper_transform=None):
    legacy_name = f'emotion_dit_ablation0905_{slug}_legacy'
    model_name = f'emotion_dit_ablation0905_{slug}'
    legacy = base_leg
    wrapper = base_wrap.replace(
        'emotion_dit_Unification_jianhua0803_legacy as _legacy',
        f'{legacy_name} as _legacy',
    )
    if legacy_transform:
        legacy = legacy_transform(legacy)
    if wrapper_transform:
        wrapper = wrapper_transform(wrapper)
    legacy = (
        f'# ICASSP27 ablation: {slug}. Independent legacy implementation copy.\n' + legacy
    )
    wrapper = (
        f'# ICASSP27 ablation: {slug}. Independent compatibility wrapper.\n' + wrapper
    )
    (ROOT / f'src/modules/{legacy_name}.py').write_text(legacy, encoding='utf-8')
    (ROOT / f'src/modules/{model_name}.py').write_text(wrapper, encoding='utf-8')
    return model_name


def additive_transform(text):
    replacements = {
        'self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift':
            'self.audio_norm(audio_feat) + (emo_shift + emo_scale)',
        'self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift':
            'self.audio_norm(prev_audio_feat) + (emo_shift + emo_scale)',
        'self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift':
            'self.audio_norm(audio_feat_saved) + (emo_shift + emo_scale)',
        'self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift':
            'self.audio_norm(null_audio_feat) + (null_shift + null_scale)',
    }
    changed = 0
    for old, new in replacements.items():
        n = text.count(old)
        if n:
            text = text.replace(old, new)
            changed += n
    if changed < 5:
        raise RuntimeError(f'additive transform changed only {changed} conditioning expressions')
    return text


def dit_adaln_legacy(text):
    text = replace_once(
        text,
        'def forward(self, tgt, memory, t_emb, memory_mask=None, tgt_mask=None):',
        'def forward(self, tgt, memory, t_emb, memory_mask=None, tgt_mask=None, emotion_shift=None, emotion_scale=None):',
        'DiTDecoderLayer.forward signature',
    )
    for old in [
        'h = modulate(self.norm1(tgt), shift_sa, scale_sa)',
        'h = modulate(self.norm2(tgt), shift_ca, scale_ca)',
        'h = modulate(self.norm3(tgt), shift_ff, scale_ff)',
    ]:
        new = old + '\n        if emotion_shift is not None:\n            h = modulate(h, emotion_shift, emotion_scale)'
        text = replace_once(text, old, new, old)

    text = replace_once(
        text,
        'def forward(self, tgt, memory, t_emb, memory_mask=None, tgt_mask=None):',
        'def forward(self, tgt, memory, t_emb, memory_mask=None, tgt_mask=None, emotion_shift=None, emotion_scale=None):',
        'DiTDecoder.forward signature',
    )
    text = replace_once(
        text,
        'tgt = layer(tgt, memory, t_emb, memory_mask=memory_mask, tgt_mask=tgt_mask)',
        'tgt = layer(tgt, memory, t_emb, memory_mask=memory_mask, tgt_mask=tgt_mask,\n'
        '                        emotion_shift=emotion_shift, emotion_scale=emotion_scale)',
        'DiTDecoder layer call',
    )
    text = replace_once(
        text,
        'def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):',
        'def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None,\n'
        '                emotion_shift=None, emotion_scale=None):',
        'DenoisingNetwork.forward signature',
    )
    text = replace_once(
        text,
        'feat_out = self.transformer(feats_in, audio_feat_in, diff_step_embedding,\n'
        '                                        memory_mask=self.alignment_mask)',
        'feat_out = self.transformer(\n'
        '                feats_in, audio_feat_in, diff_step_embedding,\n'
        '                memory_mask=self.alignment_mask,\n'
        '                emotion_shift=emotion_shift, emotion_scale=emotion_scale,\n'
        '            )',
        'DenoisingNetwork transformer call',
    )
    text = replace_once(
        text,
        'audio_feat_cond = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift',
        'audio_feat_cond = self.audio_norm(audio_feat)',
        'joint conditional audio',
    )
    text = text.replace(
        'self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift',
        'self.audio_norm(prev_audio_feat)',
    )
    text = text.replace(
        'self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift',
        'self.audio_norm(null_audio_feat)',
    )
    marker = '''        audio_feat = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_cond,
        )
'''
    insert = marker + '''        emotion_shift = torch.where(
            drop_joint_condition.view(-1, 1, 1), null_shift, emo_shift
        )
        emotion_scale = torch.where(
            drop_joint_condition.view(-1, 1, 1), null_scale, emo_scale
        )
'''
    text = replace_once(text, marker, insert, 'joint emotion selection')
    call = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
        )
'''
    call_new = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
            emotion_shift=emotion_shift,
            emotion_scale=emotion_scale,
        )
'''
    text = replace_once(text, call, call_new, 'joint denoiser call')
    independent_call = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy, audio_feat,
            prev_motion_feat, prev_audio_feat, time_step, indicator,
        )
'''
    independent_new = '''        emotion_shift = emotion_scale = None
        if 'emotion' in self.guiding_conditions:
            emotion_shift, emotion_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        motion_feat_target = self.denoising_net(
            motion_feat_noisy, audio_feat,
            prev_motion_feat, prev_audio_feat, time_step, indicator,
            emotion_shift=emotion_shift, emotion_scale=emotion_scale,
        )
'''
    text = replace_once(text, independent_call, independent_new, 'independent denoiser call')
    text = replace_once(
        text,
        'self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift',
        'self.audio_norm(audio_feat_saved)',
        'sample conditional audio',
    )
    sample_nentries = '''        if use_joint_cfg:
            audio_feat_in = torch.cat([audio_feat_uncond, audio_feat_cond], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            n_entries = 1
'''
    sample_nentries_new = sample_nentries + '''
        if use_joint_cfg:
            emotion_shift_in = torch.cat([null_shift, emo_shift], dim=0)
            emotion_scale_in = torch.cat([null_scale, emo_scale], dim=0)
        else:
            emotion_shift_in = emo_shift
            emotion_scale_in = emo_scale
'''
    text = replace_once(text, sample_nentries, sample_nentries_new, 'sample emotion entries')
    sample_call = '''            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
            )
'''
    sample_call_new = '''            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
                emotion_shift=emotion_shift_in,
                emotion_scale=emotion_scale_in,
            )
'''
    text = replace_once(text, sample_call, sample_call_new, 'sample denoiser call')
    return text


def dit_adaln_wrapper(text):
    old = '''    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None):
        if self.use_indicator and indicator is None:
            indicator = torch.ones(
                motion_feat.shape[:2],
                device=motion_feat.device,
                dtype=motion_feat.dtype,
            )
        return super().forward(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
            step, indicator=indicator,
        )
'''
    new = '''    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None, emotion_shift=None, emotion_scale=None):
        if self.use_indicator and indicator is None:
            indicator = torch.ones(
                motion_feat.shape[:2],
                device=motion_feat.device,
                dtype=motion_feat.dtype,
            )
        return super().forward(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
            step, indicator=indicator, emotion_shift=emotion_shift,
            emotion_scale=emotion_scale,
        )
'''
    return replace_once(text, old, new, 'wrapper DenoisingNetwork.forward')


def residual_legacy(text):
    insert_after = '''        self.start_motion_feat = nn.Parameter(
            torch.randn(emo_classes, self.n_prev_motions, self.motion_feat_dim))
'''
    insertion = insert_after + '''
        # Competing factorization: category-specific motion is represented as an
        # additive trajectory residual on top of a category-agnostic speech path.
        self.emotion_residual = nn.Parameter(torch.zeros(
            emo_classes, self.n_prev_motions + self.n_motions, self.motion_feat_dim
        ))
'''
    text = replace_once(text, insert_after, insertion, 'emotion residual parameter')
    text = text.replace(
        'torch.index_select(\n                self.start_motion_feat, 0, emo_index)',
        'self.start_motion_feat[0:1].expand(batch_size, -1, -1)',
    )
    text = text.replace(
        'torch.index_select(\n                self.start_audio_feat, 0, emo_index)',
        'self.start_audio_feat[0:1].expand(batch_size, -1, -1)',
    )
    joint_block = '''        # Conditional branch: real audio + real emotion.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat_cond = self.audio_norm(audio_feat) * (1 + emo_scale) + emo_shift

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        # Unconditional branch: null audio + null emotion.
        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(
            null_emotion_feat
        ).chunk(2, dim=2)
        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )
'''
    joint_new = '''        # Speech path is category agnostic; emotion is an additive motion residual.
        audio_feat_cond = self.audio_norm(audio_feat)
        prev_audio_feat = self.audio_norm(prev_audio_feat)
        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        audio_feat_uncond = self.audio_norm(null_audio_feat)
        residual_cond = torch.index_select(self.emotion_residual, 0, emo_index)
        residual_uncond = torch.zeros_like(residual_cond)
'''
    text = replace_once(text, joint_block, joint_new, 'residual joint conditioning block')
    marker = '''        audio_feat = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_cond,
        )
'''
    text = replace_once(
        text, marker,
        marker + '''        residual = torch.where(
            drop_joint_condition.view(-1, 1, 1), residual_uncond, residual_cond
        )
''',
        'residual dropout selection',
    )
    joint_call_end = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
        )

        return (
'''
    text = replace_once(
        text, joint_call_end,
        '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
        )
        motion_feat_target = motion_feat_target + residual

        return (
''',
        'residual forward add',
    )
    sample_block = '''        # Full joint condition.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        audio_feat_cond = (
            self.audio_norm(audio_feat_saved) * (1 + emo_scale) + emo_shift
        )

        if prev_audio_is_start:
            prev_audio_feat = self.audio_norm(prev_audio_feat)
        else:
            prev_audio_feat = (
                self.audio_norm(prev_audio_feat) * (1 + emo_scale) + emo_shift
            )

        # Fully dropped joint condition.
        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(
            null_emotion_feat
        ).chunk(2, dim=2)
        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )
'''
    sample_new = '''        # Speech trajectory plus an independently learned category residual.
        audio_feat_cond = self.audio_norm(audio_feat_saved)
        prev_audio_feat = self.audio_norm(prev_audio_feat)
        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        audio_feat_uncond = self.audio_norm(null_audio_feat)
        residual_cond = torch.index_select(self.emotion_residual, 0, emo_index)
'''
    text = replace_once(text, sample_block, sample_new, 'residual sample conditioning block')
    old_target = '''            if use_joint_cfg:
                uncond_target = results[0][:, -self.n_motions:]
                cond_target = results[1][:, -self.n_motions:]
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = results[0][:, -self.n_motions:]
'''
    new_target = '''            if use_joint_cfg:
                uncond_target = results[0][:, -self.n_motions:]
                cond_target = (
                    results[1][:, -self.n_motions:]
                    + residual_cond[:, -self.n_motions:]
                )
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = (
                    results[0][:, -self.n_motions:]
                    + residual_cond[:, -self.n_motions:]
                )
'''
    text = replace_once(text, old_target, new_target, 'residual sample target')
    return text


def partition_legacy(text):
    text = replace_once(
        text,
        'guiding_conditions="audio,emotion", emo_classes=8,\n                 align_mask_width=1):',
        'guiding_conditions="audio,emotion", emo_classes=8,\n                 align_mask_width=1, partition_keypoint_indices=None):',
        'partition legacy ctor signature',
    )
    insertion_point = '''        self.start_motion_feat = nn.Parameter(
            torch.randn(emo_classes, self.n_prev_motions, self.motion_feat_dim))
'''
    insertion = insertion_point + '''
        if partition_keypoint_indices is None:
            partition_keypoint_indices = list(range(11))
        elif isinstance(partition_keypoint_indices, str):
            partition_keypoint_indices = [
                int(v.strip()) for v in partition_keypoint_indices.split(',') if v.strip()
            ]
        partition_mask = torch.zeros(self.motion_feat_dim, dtype=torch.float32)
        for kp_index in partition_keypoint_indices:
            if kp_index < 0 or kp_index >= 21:
                raise ValueError(f'partition keypoint index out of range: {kp_index}')
            partition_mask[3 * kp_index:3 * kp_index + 3] = 1.0
        if self.motion_feat_dim > 63:
            partition_mask[63:] = 1.0
        self.register_buffer('emotion_partition_mask', partition_mask.view(1, 1, -1))
'''
    text = replace_once(text, insertion_point, insertion, 'partition mask init')
    marker = '''        prev_audio_is_start = prev_audio_feat is None
        if prev_audio_is_start:
            prev_audio_feat = torch.index_select(
                self.start_audio_feat, 0, emo_index)

        # Conditional branch: real audio + real emotion.
'''
    replacement = '''        prev_audio_is_start = prev_audio_feat is None
        if prev_audio_is_start:
            prev_audio_feat = torch.index_select(
                self.start_audio_feat, 0, emo_index)
        prev_audio_raw = prev_audio_feat

        # Conditional branch: real audio + real emotion.
'''
    text = replace_once(text, marker, replacement, 'partition save prev audio')
    uncond_end = '''        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )

        # One dropout decision controls both audio and emotion.
'''
    speech_insert = '''        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )
        audio_feat_speech = self.audio_norm(audio_feat)
        prev_audio_speech = self.audio_norm(prev_audio_raw)
        prev_audio_target = prev_audio_feat

        # One dropout decision controls both audio and emotion.
'''
    text = replace_once(text, uncond_end, speech_insert, 'partition speech branch')
    dropout_block = '''        audio_feat = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_cond,
        )
'''
    dropout_new = '''        audio_feat_target = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_cond,
        )
        audio_feat_speech = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_speech,
        )
'''
    text = replace_once(text, dropout_block, dropout_new, 'partition dropout')
    forward_call = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
        )
'''
    forward_new = '''        target_prediction = self.denoising_net(
            motion_feat_noisy, audio_feat_target, prev_motion_feat,
            prev_audio_target, time_step, indicator,
        )
        speech_prediction = self.denoising_net(
            motion_feat_noisy, audio_feat_speech, prev_motion_feat,
            prev_audio_speech, time_step, indicator,
        )
        partition_mask = self.emotion_partition_mask.to(target_prediction.dtype)
        motion_feat_target = (
            speech_prediction * (1.0 - partition_mask)
            + target_prediction * partition_mask
        )
'''
    text = replace_once(text, forward_call, forward_new, 'partition forward compose')
    sample_prev = '''        prev_audio_is_start = prev_audio_feat is None
        if prev_audio_is_start:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)

        if motion_at_T is None:
'''
    sample_prev_new = '''        prev_audio_is_start = prev_audio_feat is None
        if prev_audio_is_start:
            prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)
        prev_audio_raw = prev_audio_feat

        if motion_at_T is None:
'''
    text = replace_once(text, sample_prev, sample_prev_new, 'partition sample prev')
    sample_uncond_end = '''        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )

        if use_joint_cfg:
            audio_feat_in = torch.cat([audio_feat_uncond, audio_feat_cond], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            n_entries = 1

        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
        prev_audio_feat_in = torch.cat([prev_audio_feat] * n_entries, dim=0)
'''
    sample_branches = '''        audio_feat_uncond = (
            self.audio_norm(null_audio_feat) * (1 + null_scale) + null_shift
        )
        audio_feat_speech = self.audio_norm(audio_feat_saved)
        prev_audio_target = prev_audio_feat
        prev_audio_speech = self.audio_norm(prev_audio_raw)

        if use_joint_cfg:
            audio_feat_in = torch.cat(
                [audio_feat_uncond, audio_feat_speech, audio_feat_cond], dim=0
            )
            n_entries = 3
            prev_audio_feat_in = torch.cat(
                [prev_audio_target, prev_audio_speech, prev_audio_target], dim=0
            )
        else:
            audio_feat_in = torch.cat([audio_feat_speech, audio_feat_cond], dim=0)
            n_entries = 2
            prev_audio_feat_in = torch.cat(
                [prev_audio_speech, prev_audio_target], dim=0
            )

        prev_motion_feat_in = torch.cat([prev_motion_feat] * n_entries, dim=0)
'''
    text = replace_once(text, sample_uncond_end, sample_branches, 'partition sample branches')
    old_target = '''            results = results.chunk(n_entries)
            if use_joint_cfg:
                uncond_target = results[0][:, -self.n_motions:]
                cond_target = results[1][:, -self.n_motions:]
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                target_theta = results[0][:, -self.n_motions:]
'''
    new_target = '''            results = results.chunk(n_entries)
            current_mask = self.emotion_partition_mask.to(results[0].dtype)
            if use_joint_cfg:
                uncond_target = results[0][:, -self.n_motions:]
                speech_target = results[1][:, -self.n_motions:]
                emotion_target = results[2][:, -self.n_motions:]
                cond_target = (
                    speech_target * (1.0 - current_mask)
                    + emotion_target * current_mask
                )
                target_theta = uncond_target + joint_cfg_scale * (
                    cond_target - uncond_target
                )
            else:
                speech_target = results[0][:, -self.n_motions:]
                emotion_target = results[1][:, -self.n_motions:]
                target_theta = (
                    speech_target * (1.0 - current_mask)
                    + emotion_target * current_mask
                )
'''
    text = replace_once(text, old_target, new_target, 'partition sample compose')
    return text


def partition_wrapper(text):
    text = replace_once(
        text,
        '        no_use_learnable_pe=None,\n    ):',
        '        no_use_learnable_pe=None,\n        partition_keypoint_indices=None,\n    ):',
        'partition wrapper signature',
    )
    text = replace_once(
        text,
        '            align_mask_width=align_mask_width,\n        )',
        '            align_mask_width=align_mask_width,\n            partition_keypoint_indices=partition_keypoint_indices,\n        )',
        'partition wrapper super arg',
    )
    return text


def partition_train(text):
    text = text.replace(
        '        no_use_learnable_pe=args.no_use_learnable_pe,\n    )',
        '        no_use_learnable_pe=args.no_use_learnable_pe,\n'
        '        partition_keypoint_indices=args.partition_keypoint_indices,\n    )',
    )
    marker = '    parser.add_argument("--motion_feat_dim", type=int, default=70)\n'
    addition = marker + '''    parser.add_argument(
        "--partition_keypoint_indices", type=str,
        default="0,1,2,3,4,5,6,7,8,9,10",
        help=("21-point pseudo-partition: these implicit keypoint indices use "
              "target-conditioned prediction; the complement uses speech-only prediction."),
    )
'''
    return replace_once(text, marker, addition, 'partition train arg')


def residual_train(text):
    text = text.replace(
        'EMOTION_TABLE_PREFIXES = (\n    "emo_embed",\n)',
        'EMOTION_TABLE_PREFIXES = (\n    "emo_embed",\n    "emotion_residual",\n)',
    )
    init_marker = '''        if hasattr(model, "emo_embed"):
            model.emo_embed.weight.zero_()
'''
    text = replace_once(
        text, init_marker,
        init_marker + '''        if hasattr(model, "emotion_residual"):
            model.emotion_residual.zero_()
''',
        'residual init',
    )
    sync_marker = '''        if include_emotion_embedding and hasattr(model, "emo_embed"):
            model.emo_embed.weight[1:].copy_(
                model.emo_embed.weight[0:1].expand_as(model.emo_embed.weight[1:])
            )
'''
    text = replace_once(
        text, sync_marker,
        sync_marker + '''        if hasattr(model, "emotion_residual"):
            model.emotion_residual[1:].copy_(
                model.emotion_residual[0:1].expand_as(model.emotion_residual[1:])
            )
''',
        'residual sync',
    )
    return text


def single_stage_train(text):
    text = re.sub(r'DEFAULT_MAX_ITER = \d+', 'DEFAULT_MAX_ITER = 245000', text, count=1)
    text = text.replace(
        'parser.add_argument("--stage1_iter", type=int, default=190000)',
        'parser.add_argument("--stage1_iter", type=int, default=0)',
    )
    text = text.replace(
        '    if not (0 < args.stage1_iter < args.max_iter):\n        raise ValueError("stage1_iter must be inside the total training budget")',
        '    if not (0 <= args.stage1_iter < args.max_iter):\n        raise ValueError("stage1_iter must be in [0, max_iter)")',
    )
    start = text.index('    generic_template_path = args.generic_motion_template_path\n')
    end = text.index('    mead_dataset = EmoLevelDataset(\n', start)
    text = text[:start] + '    generic_dataset = None\n' + text[end:]
    text = text.replace(
        '    audio_unit = mead_dataset.audio_unit\n    if abs(generic_dataset.audio_unit - audio_unit) > 1e-6:\n        raise RuntimeError("Generic and MEAD audio units do not match")\n',
        '    audio_unit = mead_dataset.audio_unit\n',
    )
    return text


def no_ce_train(text):
    text = text.replace(
        'parser.add_argument("--l_emo", type=float, default=1.0)',
        'parser.add_argument("--l_emo", type=float, default=0.0)',
    )
    text = text.replace(
        'if use_emotion_loss and args.target == "sample":',
        'if use_emotion_loss and args.l_emo > 0 and args.target == "sample":',
    )
    return text


variants = []


def emit(slug, description, legacy_transform=None, wrapper_transform=None, train_transform=None, max_iter=None):
    model_module = make_model_pair(slug, legacy_transform, wrapper_transform)
    train = set_variant_constants(base_train, slug, description, model_module, max_iter=max_iter)
    if train_transform:
        train = train_transform(train)
    train_path = ROOT / f'train_Ablation0905_{slug}.py'
    train_path.write_text(train, encoding='utf-8')
    variants.append((slug, model_module, train_path.name, description))


emit(
    'cond_additive',
    'Conditioning control: replace pre-audio affine reparameterization with an additive target bias while retaining the final 6009 training recipe.',
    legacy_transform=additive_transform,
)
emit(
    'cond_dit_adaln',
    'Conditioning-placement control: keep acoustic tokens unmodulated and apply the same target affine inside every DiT block.',
    legacy_transform=dit_adaln_legacy,
    wrapper_transform=dit_adaln_wrapper,
)
emit(
    'motion_partition',
    'Holistic-generation control: force a coordinate pseudo-partition between speech-only and target-conditioned predictions with shared parameters.',
    legacy_transform=partition_legacy,
    wrapper_transform=partition_wrapper,
    train_transform=partition_train,
)
emit(
    'emotion_residual',
    'Holistic-generation control: predict a category-agnostic speech trajectory and add an independently learned category motion residual.',
    legacy_transform=residual_legacy,
    train_transform=residual_train,
)
emit(
    'single_stage_mead',
    'Training-schedule control: remove Generic Stage 1 and train only the 5k specialization transition plus 240k full MEAD updates.',
    train_transform=single_stage_train,
    max_iter=245000,
)
emit(
    'no_emotion_ce',
    'Regularization control: retain the final 6009 architecture and training schedule but set the frozen motion-classifier CE weight to zero.',
    train_transform=no_ce_train,
)

readme = '''# ICASSP27 controlled ablations — 2026-09-05

## Final reference

The paper reference method is `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` (6009).
Every ablation below is a **physical training-file copy** and has a dedicated **physical model + legacy-model copy**. No ablation imports another ablation. This makes checkpoint/model provenance explicit and prevents nested variant dependencies.

## Why these six new runs

After re-checking the paper claims, only six additional controlled runs are necessary. The existing 6006/6008/6007/6010/6009 runs already isolate shared-prior warm-start, balanced MEAD sampling, and Min-SNR. Generic replay (6011) is diagnostic and is not part of the final method. Radius-0/global attention is not a headline contribution (the manuscript explicitly treats the local/global temporal design as architectural context), so it should not consume main ablation budget unless reviewers request it.

| ID | Training file | Scientific question | Primary metrics |
|---|---|---|---|
'''
for slug, model_module, train_file, desc in variants:
    metrics = {
        'cond_additive': 'T-UAR, leakage, target-source margin, LSE-C',
        'cond_dit_adaln': 'T-UAR, leakage, target-source margin, LSE-C',
        'motion_partition': 'T-UAR, leakage, FVD, LSE-C',
        'emotion_residual': 'T-UAR, leakage, FVD, LSE-C',
        'single_stage_mead': 'FVD, LSE-C, T-UAR, cross-dataset generalization',
        'no_emotion_ce': 'T-UAR, leakage, target-source margin, LSE-C',
    }[slug]
    readme += f'| `{slug}` | `{train_file}` | {desc} | {metrics} |\n'

readme += '''
## Recommended paper comparisons

### Conditioning placement/operator

Compare the final 6009 model against:

- `cond_additive`: target information is an additive bias on frame-aligned acoustic tokens.
- `cond_dit_adaln`: acoustic tokens remain category-agnostic; the target affine is applied inside DiT blocks.

This directly tests whether **pre-motion affective acoustic reparameterization** is preferable to a simpler additive operator and to moving the affine condition into the motion backbone. Use the 8x8 source-to-target counterfactual protocol; matched MEAD emotion accuracy alone is not sufficient.

### Holistic generation vs forced factorization

Compare the final 6009 model against:

- `motion_partition`: the same denoiser parameters are evaluated through a speech-only path and a target-conditioned path; output coordinates are forced to come from one path or the other. By default keypoint indices 0--10 are target-conditioned and 11--20 are speech-only, while the 7 global motion dimensions remain target-conditioned. The exact index set is configurable with `--partition_keypoint_indices`. For the paper, keep one fixed mask for all identities and disclose it. If an image-space support diagnostic is available, pass its pre-registered mask rather than tuning the mask on test results.
- `emotion_residual`: target emotion is represented as a category-specific additive 80x70 motion residual on top of a category-agnostic speech trajectory. It cannot alter the speech generator through acoustic FiLM, so it is a direct residual-composition alternative.

Do **not** choose the partition mask after seeing test scores. That would invalidate the controlled comparison.

### Training protocol / semantic regularization

- `single_stage_mead`: removes Generic pretraining but preserves the same MEAD budget (5k emotion/start transition + 240k full MEAD), Min-SNR, balance, EMA, architecture, losses, and learning-rate family. This is the correct control for the two-data-stage claim.
- `no_emotion_ce`: changes only `lambda_emo` from 1 to 0. This isolates the frozen classifier regularizer.

## Existing training-recipe ablation (do not retrain)

Use the already trained runs on the **same final evaluation set**:

- 6006 `schedule_ema`: no shared-condition warm-start, no Min-SNR, no balanced sampling.
- 6008 `sharedcond_ema`: + shared generic prior.
- 6007 `sharedcond_balanced_ema`: + balanced MEAD sampling.
- 6010 `sharedcond_minsnr_ema`: + Min-SNR.
- 6009 `sharedcond_minsnr_balanced_ema`: final method.

This table is supporting evidence for the training recipe, not the main method-claim table.

## Run commands

```bash
python train_Ablation0905_cond_additive.py
python train_Ablation0905_cond_dit_adaln.py
python train_Ablation0905_motion_partition.py
python train_Ablation0905_emotion_residual.py
python train_Ablation0905_single_stage_mead.py
python train_Ablation0905_no_emotion_ce.py
```

For `motion_partition`, a different **pre-registered** pseudo-partition can be supplied, e.g.

```bash
python train_Ablation0905_motion_partition.py \
  --partition_keypoint_indices 0,2,4,7,9,11,13,15,18,19
```

## Evaluation rules

1. Use exactly the same held-out identities, renderer, source portraits, audio clips, CFG scale, diffusion steps, and preprocessing for every row.
2. For counterfactual control, hold portrait and audio fixed and generate all target emotions. Use the same initial diffusion seed/noise for corresponding comparisons.
3. Report target UAR (or the final independent RGB emotion metric), source-emotion leakage, target-source probability margin, LSE-C/LSE-D, and FVD.
4. Do not use the training-time motion classifier as the paper emotion evaluator.
5. Evaluate multiple fixed diffusion seeds when budget permits and report paired uncertainty.
6. The final 6009 Stage-1 manifest is assumed to exclude MEAD test identities, as verified before these ablations.

## Checkpoint provenance

Each training script writes its own `variant_name` and `model_variant` into checkpoint args. Keep those fields when generating videos so a checkpoint is always loaded with its dedicated model file.
'''
(ROOT / 'ABLATION_ICASSP27_0905.md').write_text(readme, encoding='utf-8')

for slug, model_module, train_file, _ in variants:
    train_text = (ROOT / train_file).read_text(encoding='utf-8')
    if 'train_Ablation0905_' in train_text.replace(train_file, ''):
        raise RuntimeError(f'{train_file} imports/references another ablation training file')
    wrapper_text = (ROOT / f'src/modules/{model_module}.py').read_text(encoding='utf-8')
    expected_legacy = f'{model_module}_legacy'
    if expected_legacy not in wrapper_text:
        raise RuntimeError(f'{model_module} does not point to its dedicated legacy copy')

print('Generated variants:')
for item in variants:
    print('  ', item)

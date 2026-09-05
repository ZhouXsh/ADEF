from pathlib import Path
import re

ROOT = Path('.')
MOD = ROOT / 'src/modules'
BASE_TRAIN = ROOT / 'train_Ablation0905_cond_additive.py'
BASE_WRAPPER = MOD / 'emotion_dit_Unification_jianhua0803.py'
BASE_LEGACY = MOD / 'emotion_dit_Unification_jianhua0803_legacy.py'


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected exactly one occurrence, found {n}')
    return text.replace(old, new, 1)


def merge_sources(legacy_text, wrapper_text, slug):
    """Put the implementation and runtime-correction layer in one module."""
    core = legacy_text
    core = replace_once(
        core,
        'class DenoisingNetwork(nn.Module):',
        'class _CoreDenoisingNetwork(nn.Module):',
        f'{slug}: core denoiser class',
    )
    core = replace_once(
        core,
        'class DitTalkingHead(nn.Module):',
        'class _CoreDitTalkingHead(nn.Module):',
        f'{slug}: core talking-head class',
    )
    core = replace_once(
        core,
        'self.denoising_net = DenoisingNetwork(',
        'self.denoising_net = _CoreDenoisingNetwork(',
        f'{slug}: core denoiser construction',
    )

    if 'import sys' not in wrapper_text:
        raise RuntimeError(f'{slug}: wrapper has no import sys anchor')
    public = wrapper_text[wrapper_text.index('import sys'):]
    public = re.sub(
        r'from \. import emotion_dit_ablation0905_[A-Za-z0-9_]+_legacy as _legacy\n',
        '',
        public,
        count=1,
    )
    # The base wrapper has a different legacy module name.
    public = public.replace(
        'from . import emotion_dit_Unification_jianhua0803_legacy as _legacy\n',
        '',
        1,
    )
    public = public.replace('DiffusionSchedule = _legacy.DiffusionSchedule\n\n', '', 1)
    public = public.replace('DiTDecoderLayer = _legacy.DiTDecoderLayer\n', '', 1)
    public = public.replace('DiTDecoder = _legacy.DiTDecoder\n\n', '', 1)
    public = public.replace(
        'class DenoisingNetwork(_legacy.DenoisingNetwork):',
        'class DenoisingNetwork(_CoreDenoisingNetwork):',
        1,
    )
    public = public.replace(
        'class DitTalkingHead(_legacy.DitTalkingHead):',
        'class DitTalkingHead(_CoreDitTalkingHead):',
        1,
    )
    if '_legacy' in public:
        raise RuntimeError(f'{slug}: merged public layer still references _legacy')

    header = f'''"""Self-contained ICASSP27 controlled ablation: {slug}.

The complete model implementation and the runtime-correction layer live in this
single file.  ``_CoreDenoisingNetwork`` and ``_CoreDitTalkingHead`` are private
implementation classes; ``DenoisingNetwork`` and ``DitTalkingHead`` are the
public checkpoint-compatible classes.  This module never imports another
ablation model and has no companion ``*_legacy.py`` file.
"""\n\n'''
    return header + core + '\n\n# ---- Runtime-correction/public compatibility layer (same file) ----\n\n' + public


def make_base_merged(slug, legacy_transform=None, wrapper_transform=None):
    legacy = BASE_LEGACY.read_text(encoding='utf-8')
    wrapper = BASE_WRAPPER.read_text(encoding='utf-8')
    if legacy_transform is not None:
        legacy = legacy_transform(legacy)
    if wrapper_transform is not None:
        wrapper = wrapper_transform(wrapper)
    return merge_sources(legacy, wrapper, slug)


def make_train(slug, description, module_name, align_mask_width=None):
    text = BASE_TRAIN.read_text(encoding='utf-8')
    text = re.sub(
        r'from src\.modules\.emotion_dit_ablation0905_[A-Za-z0-9_]+ import DitTalkingHead',
        f'from src.modules.{module_name} import DitTalkingHead',
        text,
        count=1,
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
    text = re.sub(
        r'parser\.add_argument\("--model_variant", type=str, default="[^"]+"\)',
        f'parser.add_argument("--model_variant", type=str, default="{module_name}")',
        text,
        count=1,
    )
    if align_mask_width is not None:
        text = re.sub(
            r'parser\.add_argument\("--align_mask_width", type=int, default=\d+\)',
            f'parser.add_argument("--align_mask_width", type=int, default={align_mask_width})',
            text,
            count=1,
        )
    text = (
        '# ICASSP27 controlled ablation generated from the final 6009 recipe.\n'
        '# Physical training copy; imports only its own physical model module.\n' + text
    )
    path = ROOT / f'train_Ablation0905_{slug}.py'
    path.write_text(text, encoding='utf-8')
    return path


def shared_start_legacy(text):
    # Retain the parameters for an exact architecture match, but every first
    # window uses the same row-0 start prior independent of target category.
    pat_motion = re.compile(r'torch\.index_select\(\s*self\.start_motion_feat,\s*0,\s*emo_index\s*\)')
    pat_audio = re.compile(r'torch\.index_select\(\s*self\.start_audio_feat,\s*0,\s*emo_index\s*\)')
    text, n_m = pat_motion.subn('self.start_motion_feat[0:1].expand(batch_size, -1, -1)', text)
    text, n_a = pat_audio.subn('self.start_audio_feat[0:1].expand(batch_size, -1, -1)', text)
    if n_m < 2 or n_a < 2:
        raise RuntimeError(f'shared_start: unexpected replacement counts motion={n_m}, audio={n_a}')
    return text


def late_concat_legacy(text):
    text = replace_once(
        text,
        'def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None):',
        'def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat, step, indicator=None, emotion_token=None):',
        'late_concat: denoiser signature',
    )
    old_memory = '''            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
            feat_out = self.transformer(feats_in, audio_feat_in, diff_step_embedding,
                                        memory_mask=self.alignment_mask)
'''
    new_memory = '''            audio_feat_in = torch.cat([prev_audio_feat, audio_feat], dim=1)
            memory_mask = self.alignment_mask
            if emotion_token is not None:
                # Parameter-matched late concatenation: append one target token
                # after acoustic encoding instead of changing the 512-D carrier.
                audio_feat_in = torch.cat([audio_feat_in, emotion_token], dim=1)
                if memory_mask is not None:
                    target_column = torch.zeros(
                        memory_mask.shape[0], 1,
                        dtype=memory_mask.dtype, device=memory_mask.device,
                    )
                    memory_mask = torch.cat([memory_mask, target_column], dim=1)
            feat_out = self.transformer(
                feats_in, audio_feat_in, diff_step_embedding,
                memory_mask=memory_mask,
            )
'''
    text = replace_once(text, old_memory, new_memory, 'late_concat: memory append')

    old_joint = '''        # Conditional branch: real audio + real emotion.
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
    new_joint = '''        # Late target token: acoustic features remain category agnostic.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        emotion_token_cond = emo_shift + emo_scale
        audio_feat_cond = self.audio_norm(audio_feat)
        prev_audio_feat = self.audio_norm(prev_audio_feat)

        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(null_emotion_feat).chunk(2, dim=2)
        emotion_token_uncond = null_shift + null_scale
        audio_feat_uncond = self.audio_norm(null_audio_feat)
'''
    text = replace_once(text, old_joint, new_joint, 'late_concat: joint condition')

    drop_audio = '''        audio_feat = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            audio_feat_uncond,
            audio_feat_cond,
        )
'''
    text = replace_once(
        text,
        drop_audio,
        drop_audio + '''        emotion_token = torch.where(
            drop_joint_condition.view(-1, 1, 1),
            emotion_token_uncond,
            emotion_token_cond,
        )
''',
        'late_concat: joint target dropout',
    )
    old_call = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
        )
'''
    new_call = '''        motion_feat_target = self.denoising_net(
            motion_feat_noisy,
            audio_feat,
            prev_motion_feat,
            prev_audio_feat,
            time_step,
            indicator,
            emotion_token=emotion_token,
        )
'''
    text = replace_once(text, old_call, new_call, 'late_concat: joint denoiser call')

    old_sample = '''        # Full joint condition.
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

        if use_joint_cfg:
            audio_feat_in = torch.cat([audio_feat_uncond, audio_feat_cond], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            n_entries = 1
'''
    new_sample = '''        # Acoustic tokens are untouched by target category; target is appended late.
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        emotion_token_cond = emo_shift + emo_scale
        audio_feat_cond = self.audio_norm(audio_feat_saved)
        prev_audio_feat = self.audio_norm(prev_audio_feat)

        null_audio_feat = self.null_audio_feat.expand(batch_size, self.n_motions, -1)
        null_emotion_feat = self.null_emotion_feat.expand(batch_size, -1, -1)
        null_shift, null_scale = self.adaLN_modulation(null_emotion_feat).chunk(2, dim=2)
        emotion_token_uncond = null_shift + null_scale
        audio_feat_uncond = self.audio_norm(null_audio_feat)

        if use_joint_cfg:
            audio_feat_in = torch.cat([audio_feat_uncond, audio_feat_cond], dim=0)
            emotion_token_in = torch.cat([emotion_token_uncond, emotion_token_cond], dim=0)
            n_entries = 2
        else:
            audio_feat_in = audio_feat_cond
            emotion_token_in = emotion_token_cond
            n_entries = 1
'''
    text = replace_once(text, old_sample, new_sample, 'late_concat: sample conditions')
    old_sample_call = '''            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
            )
'''
    new_sample_call = '''            results = self.denoising_net(
                motion_in,
                audio_feat_in,
                prev_motion_feat_in,
                prev_audio_feat_in,
                step_in,
                indicator_in,
                emotion_token=emotion_token_in,
            )
'''
    text = replace_once(text, old_sample_call, new_sample_call, 'late_concat: sample denoiser call')
    return text


def late_concat_wrapper(text):
    old = '''    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None):
'''
    new = '''    def forward(self, motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
                step, indicator=None, emotion_token=None):
'''
    text = replace_once(text, old, new, 'late_concat wrapper: signature')
    old_super = '''        return super().forward(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
            step, indicator=indicator,
        )
'''
    new_super = '''        return super().forward(
            motion_feat, audio_feat, prev_motion_feat, prev_audio_feat,
            step, indicator=indicator, emotion_token=emotion_token,
        )
'''
    return replace_once(text, old_super, new_super, 'late_concat wrapper: super call')


def audio_only_from_residual(legacy_text):
    # Residual model already uses a category-agnostic speech path and shared
    # start priors. Null the residual while retaining all target parameters so
    # reported parameter count/capacity is exactly matched to the target models.
    start = legacy_text.index('    def _build_emotion_residual(self, emo_index):')
    end = legacy_text.index('    def extract_audio_feature(self, audio, frame_num=None):', start)
    zero_method = '''    def _build_emotion_residual(self, emo_index):
        """Return a zero target residual; target labels have no effect."""
        total_len = self.n_prev_motions + self.n_motions
        return torch.zeros(
            emo_index.shape[0], total_len, self.motion_feat_dim,
            device=self.device, dtype=self.start_motion_feat.dtype,
        )

'''
    text = legacy_text[:start] + zero_method + legacy_text[end:]
    # Cover the fallback independent path too, even though the controlled run
    # uses the joint-CFG interface for an exact training-recipe match.
    text = shared_start_legacy(text)
    text = text.replace(
        '# Speech path is category agnostic; emotion is an additive motion residual.',
        '# Audio-only control: category label is accepted by the API but has no effect.',
    )
    return text


# -----------------------------------------------------------------------------
# 1) Collapse every existing ablation pair into one self-contained model file.
# -----------------------------------------------------------------------------
existing = [
    'cond_additive',
    'cond_dit_adaln',
    'motion_partition',
    'emotion_residual',
    'single_stage_mead',
    'no_emotion_ce',
]
# Save residual sources before deleting companion files; audio-only derives from
# this exact parameter-matched speech/residual factorization.
residual_legacy_source = (MOD / 'emotion_dit_ablation0905_emotion_residual_legacy.py').read_text(encoding='utf-8')
residual_wrapper_source = (MOD / 'emotion_dit_ablation0905_emotion_residual.py').read_text(encoding='utf-8')

for slug in existing:
    public_path = MOD / f'emotion_dit_ablation0905_{slug}.py'
    legacy_path = MOD / f'emotion_dit_ablation0905_{slug}_legacy.py'
    merged = merge_sources(
        legacy_path.read_text(encoding='utf-8'),
        public_path.read_text(encoding='utf-8'),
        slug,
    )
    public_path.write_text(merged, encoding='utf-8')
    legacy_path.unlink()


# -----------------------------------------------------------------------------
# 2) Add controls required by the manuscript plus one confound check.
# -----------------------------------------------------------------------------
new_variants = []

def emit_model_and_train(slug, description, merged_model, align_mask_width=None):
    module_name = f'emotion_dit_ablation0905_{slug}'
    model_path = MOD / f'{module_name}.py'
    model_path.write_text(merged_model, encoding='utf-8')
    train_path = make_train(
        slug, description, module_name,
        align_mask_width=align_mask_width,
    )
    new_variants.append((slug, model_path, train_path, description))


emit_model_and_train(
    'audio_only',
    'Controlled same-capacity audio-only baseline: target-label parameters remain allocated but are functionally disconnected, and all first-window start priors are category agnostic.',
    merge_sources(
        audio_only_from_residual(residual_legacy_source),
        residual_wrapper_source,
        'audio_only',
    ),
)

emit_model_and_train(
    'cond_late_token_concat',
    'Conditioning-placement control: append one parameter-matched target token to the acoustic memory after encoding; the 512-D acoustic carrier itself is not target-reparameterized.',
    make_base_merged(
        'cond_late_token_concat',
        legacy_transform=late_concat_legacy,
        wrapper_transform=late_concat_wrapper,
    ),
)

# Attention radius mapping follows enc_dec_mask(expansion=align_mask_width-1):
# width=1 -> radius 0, width=2 -> radius 1 (full 6009), width=3 -> radius 2,
# width=0 -> unrestricted/global cross-attention.
for slug, width, description in [
    ('attn_radius0', 1, 'Audio cross-attention control with strict same-frame radius 0; all other final-6009 settings are unchanged.'),
    ('attn_radius2', 3, 'Audio cross-attention control with frame-local radius 2; all other final-6009 settings are unchanged.'),
    ('attn_global', 0, 'Audio cross-attention control with no alignment mask (global acoustic memory); all other final-6009 settings are unchanged.'),
]:
    emit_model_and_train(
        slug,
        description,
        make_base_merged(slug),
        align_mask_width=width,
    )

emit_model_and_train(
    'shared_start_tokens',
    'Start-prior confound control: retain full ADEF acoustic reparameterization but make first-window motion/audio start priors category agnostic.',
    make_base_merged('shared_start_tokens', legacy_transform=shared_start_legacy),
)


# -----------------------------------------------------------------------------
# 3) Replace the README with a complete, paper-aligned experiment plan.
# -----------------------------------------------------------------------------
readme = r'''# ICASSP27 controlled ablations — 2026-09-05

## Reference implementation

The final paper model is `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` (run 6009). Its default `align_mask_width=2` corresponds to **audio-attention radius 1** because `enc_dec_mask` uses `expansion=align_mask_width-1`.

The Stage-1 `front_all_motions.pkl` manifest used by the final recipe has already been verified to exclude MEAD test identities. Therefore the two-stage control below tests the value of generic motion pretraining rather than repairing an identity-leakage problem.

## File organization

Each controlled variant has exactly two user-facing files:

```text
train_Ablation0905_<variant>.py
src/modules/emotion_dit_ablation0905_<variant>.py
```

The model file is self-contained. The former `*_legacy.py` companion files have been merged into the corresponding public model file and deleted. The private `_Core*` classes and the public runtime-corrected classes intentionally coexist **inside the same file** so checkpoint compatibility is preserved without cross-file nesting. No ablation model imports another ablation model; no ablation training script imports another ablation training script.

## A. Conditioning controls — main paper table

| Variant | Training file | What changes relative to full ADEF |
|---|---|---|
| `audio_only` | `train_Ablation0905_audio_only.py` | Target label is functionally disconnected; start priors are category agnostic. Target parameters remain allocated only to keep capacity/count matched. |
| `cond_late_token_concat` | `train_Ablation0905_cond_late_token_concat.py` | Acoustic tokens remain target agnostic; one target token is appended to cross-attention memory after acoustic encoding. No new projection or parameters are added. |
| `cond_additive` | `train_Ablation0905_cond_additive.py` | Replace feature-wise affine acoustic reparameterization with an additive target bias. |
| `cond_dit_adaln` | `train_Ablation0905_cond_dit_adaln.py` | Acoustic carrier stays target agnostic; the target affine acts inside DiT blocks. |
| **full 6009** | `train_Unification_twostage0901_sharedcond_minsnr_balanced_ema.py` | Target affine reparameterizes the dynamic acoustic carrier before motion decoding. |

These rows now exactly support the conditioning comparison stated in the manuscript. `cond_late_token_concat` is deliberately **token concatenation**, not feature-dimension concatenation: appending a 512-D target token keeps the denoiser width and trainable parameter count unchanged, whereas feature concatenation would require an extra projection and confound placement with capacity.

Primary evidence: counterfactual T-UAR/E-UAR, source-emotion leakage, target-source probability margin, LSE-C/LSE-D, FVD.

## B. Holistic-generation controls — main paper table

| Variant | Training file | Scientific question |
|---|---|---|
| `motion_partition` | `train_Ablation0905_motion_partition.py` | What happens when output coordinates are forced to come from speech-only vs target-conditioned paths instead of one holistic trajectory model? |
| `emotion_residual` | `train_Ablation0905_emotion_residual.py` | Is `speech trajectory + additive emotion residual` sufficient compared with holistic joint generation? |
| **full 6009** | final training file | Does one joint conditional diffusion distribution better preserve both target affect and speech synchronization? |

For `motion_partition`, pre-register the coordinate mask before evaluating test results. The default indices are only a runnable reference. If the image-space support diagnostic produces a justified pseudo-partition, use that fixed mask for the paper run; never select a mask after seeing test scores.

## C. Frame-local cross-attention controls — required by the current manuscript

The implementation mapping is:

| Paper radius | `align_mask_width` | Training file |
|---|---:|---|
| radius 0 | 1 | `train_Ablation0905_attn_radius0.py` |
| **radius 1 (full 6009)** | **2** | final training file; **do not retrain** |
| radius 2 | 3 | `train_Ablation0905_attn_radius2.py` |
| global / unrestricted | 0 | `train_Ablation0905_attn_global.py` |

This set is necessary because the current paper explicitly states that radius `{0,1,2}` and unrestricted attention are evaluated. Earlier advice to omit these runs was therefore too conservative.

Primary evidence: LSE-C/LSE-D first, then FVD and target emotion metrics. The expected scientific question is whether radius 1 gives enough local audiovisual tolerance without allowing unrelated acoustic frames to dominate.

## D. Training / regularization controls

| Variant | Training file | Change |
|---|---|---|
| `single_stage_mead` | `train_Ablation0905_single_stage_mead.py` | Remove 190k generic Stage 1; keep the same 5k specialization + 240k MEAD adaptation budget and final training recipe. |
| `no_emotion_ce` | `train_Ablation0905_no_emotion_ce.py` | Set the frozen motion-classifier CE weight to zero. |
| `shared_start_tokens` | `train_Ablation0905_shared_start_tokens.py` | Keep full ADEF conditioning, but force category-agnostic first-window start priors. This checks that target control is not merely coming from emotion-indexed initialization. |

`shared_start_tokens` is an additional confound check rather than a headline contribution. It is worth running because the Method explicitly mentions emotion-indexed start tokens; a strong result here can show that acoustic reparameterization remains effective without category-specific initialization.

## E. Existing recipe ablations — reuse; do not retrain

Evaluate these existing checkpoints on the **same final test protocol**:

- 6006 `schedule_ema`: no shared-condition warm-start, no Min-SNR, no balanced sampling.
- 6008 `sharedcond_ema`: adds shared generic condition warm-start.
- 6007 `sharedcond_balanced_ema`: adds balanced MEAD sampling without Min-SNR.
- 6010 `sharedcond_minsnr_ema`: adds Min-SNR without balancing.
- 6009 `sharedcond_minsnr_balanced_ema`: full recipe.

Together they isolate shared-condition warm-start, class/level balancing, Min-SNR, and their combination. 6011 generic replay remains a diagnostic rather than part of the final method.

## Recommended training batches for six GPUs

The number of experiments is **not** limited to six. A practical order is:

```text
Batch 1 (main conditioning/representation):
  audio_only
  cond_late_token_concat
  cond_additive
  cond_dit_adaln
  emotion_residual
  motion_partition   [only after the partition mask is frozen]

Batch 2 (temporal/training/confound):
  attn_radius0
  attn_radius2
  attn_global
  single_stage_mead
  no_emotion_ce
  shared_start_tokens
```

If the pseudo-partition is not ready, move `motion_partition` to Batch 2 and use the free Batch-1 GPU for one fixed-seed replication of a high-priority conditioning row.

## Run commands

```bash
python train_Ablation0905_audio_only.py
python train_Ablation0905_cond_late_token_concat.py
python train_Ablation0905_cond_additive.py
python train_Ablation0905_cond_dit_adaln.py
python train_Ablation0905_motion_partition.py
python train_Ablation0905_emotion_residual.py
python train_Ablation0905_attn_radius0.py
python train_Ablation0905_attn_radius2.py
python train_Ablation0905_attn_global.py
python train_Ablation0905_single_stage_mead.py
python train_Ablation0905_no_emotion_ce.py
python train_Ablation0905_shared_start_tokens.py
```

## Evaluation rules

1. Use the same held-out identities, renderer, portraits, audio clips, preprocessing, diffusion steps and CFG setting for all causal comparisons.
2. Counterfactual evaluation fixes portrait/audio and generates every target category. Use identical diffusion initial noise for paired variant comparisons.
3. Report target RGB-video UAR, source leakage, target-source probability margin, LSE-C/LSE-D, FVD, and motion magnitude. Do not use the training-time motion classifier as the paper emotion evaluator.
4. For attention-radius rows, emphasize LSE before affect scores; for conditioning rows, emphasize target adherence/leakage while confirming LSE is not materially degraded.
5. Use multiple fixed seeds or paired bootstrap confidence intervals when budget permits.
6. Keep `model_variant` from the checkpoint. `src/utils/helper.py` dynamically resolves only the controlled `emotion_dit_ablation0905_*` namespace so evaluation loads the correct physical model file.
7. Smoke-test each new entrypoint for 10–50 iterations before launching the full run.

## Scope decision

No extra ablations are added merely because GPUs are available. In particular, EMA itself is an optimization/stabilization device rather than a claimed methodological contribution, and generic replay is not in the final method. The 12 controlled variants above cover every comparison explicitly promised by the current manuscript plus the category-specific-start confound, while the already trained 6006/6007/6008/6010 checkpoints cover the remaining recipe factors.
'''
(ROOT / 'ABLATION_ICASSP27_0905.md').write_text(readme, encoding='utf-8')


# -----------------------------------------------------------------------------
# 4) Update loader wording for the new one-file organization.
# -----------------------------------------------------------------------------
helper_path = ROOT / 'src/utils/helper.py'
helper = helper_path.read_text(encoding='utf-8')
helper = helper.replace(
    'those variants are resolved through a strict allow-list prefix so inference never\n'
    'silently falls back to the final ADEF class.',
    'those self-contained variants are resolved through a strict allow-list prefix so\n'
    'inference never silently falls back to the final ADEF class.',
)
helper = helper.replace(
    '"model_variant must name the public ablation wrapper, not its "\n'
    '                f"legacy implementation: {model_variant}"',
    '"legacy ablation module names are no longer valid after the one-file "\n'
    '                f"refactor: {model_variant}"',
)
helper_path.write_text(helper, encoding='utf-8')


# -----------------------------------------------------------------------------
# 5) Static consistency checks before CI performs py_compile.
# -----------------------------------------------------------------------------
all_slugs = existing + [item[0] for item in new_variants]
assert len(all_slugs) == 12, all_slugs
for slug in all_slugs:
    model_path = MOD / f'emotion_dit_ablation0905_{slug}.py'
    train_path = ROOT / f'train_Ablation0905_{slug}.py'
    if not model_path.exists() or not train_path.exists():
        raise RuntimeError(f'missing pair for {slug}')
    model = model_path.read_text(encoding='utf-8')
    train = train_path.read_text(encoding='utf-8')
    if '_legacy as _legacy' in model:
        raise RuntimeError(f'{slug} still imports a companion legacy module')
    if f'emotion_dit_ablation0905_{slug}_legacy' in model:
        raise RuntimeError(f'{slug} still references its removed legacy filename')
    expected_import = f'from src.modules.emotion_dit_ablation0905_{slug} import DitTalkingHead'
    if expected_import not in train:
        raise RuntimeError(f'{slug} training script does not import its own model')
    for other in all_slugs:
        if other != slug and f'from src.modules.emotion_dit_ablation0905_{other} import' in train:
            raise RuntimeError(f'{slug} training imports sibling variant {other}')

legacy_left = sorted(MOD.glob('emotion_dit_ablation0905_*_legacy.py'))
if legacy_left:
    raise RuntimeError(f'legacy companion files remain: {legacy_left}')

# Explicit manuscript-critical defaults.
assert 'default=1)' in (ROOT / 'train_Ablation0905_attn_radius0.py').read_text()
assert 'default=3)' in (ROOT / 'train_Ablation0905_attn_radius2.py').read_text()
assert 'default=0)' in (ROOT / 'train_Ablation0905_attn_global.py').read_text()
late = (MOD / 'emotion_dit_ablation0905_cond_late_token_concat.py').read_text()
assert 'emotion_token=emotion_token' in late
assert 'target_column = torch.zeros' in late
audio_only = (MOD / 'emotion_dit_ablation0905_audio_only.py').read_text()
assert 'Return a zero target residual' in audio_only
shared = (MOD / 'emotion_dit_ablation0905_shared_start_tokens.py').read_text()
assert 'self.start_motion_feat[0:1].expand(batch_size, -1, -1)' in shared

print('Refactored/expanded ablation suite:', ', '.join(all_slugs))

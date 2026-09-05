from pathlib import Path

ROOT = Path('.')
MOD = ROOT / 'src/modules'
DATA = ROOT / 'src/dataset'
UTIL = ROOT / 'src/utils'


def one(text, old, new, label):
    if text.count(old) != 1:
        raise RuntimeError(f'{label}: expected one occurrence, got {text.count(old)}')
    return text.replace(old, new, 1)


def merge_wrapper(public_path, legacy_path, core_model=False, dataset=False):
    public = public_path.read_text(encoding='utf-8')
    legacy = legacy_path.read_text(encoding='utf-8')
    if core_model:
        legacy = one(legacy, 'class DenoisingNetwork(nn.Module):', 'class _CoreDenoisingNetwork(nn.Module):', 'model denoiser')
        legacy = one(legacy, 'class DitTalkingHead(nn.Module):', 'class _CoreDitTalkingHead(nn.Module):', 'model head')
        legacy = one(legacy, 'self.denoising_net = DenoisingNetwork(', 'self.denoising_net = _CoreDenoisingNetwork(', 'model construct')
        tail = public[public.index('import sys'):]
        tail = tail.replace('from . import emotion_dit_Unification_jianhua0803_legacy as _legacy\n', '', 1)
        tail = tail.replace('DiffusionSchedule = _legacy.DiffusionSchedule\n\n', '', 1)
        tail = tail.replace('DiTDecoderLayer = _legacy.DiTDecoderLayer\n', '', 1)
        tail = tail.replace('DiTDecoder = _legacy.DiTDecoder\n\n', '', 1)
        tail = tail.replace('class DenoisingNetwork(_legacy.DenoisingNetwork):', 'class DenoisingNetwork(_CoreDenoisingNetwork):', 1)
        tail = tail.replace('class DitTalkingHead(_legacy.DitTalkingHead):', 'class DitTalkingHead(_CoreDitTalkingHead):', 1)
        if '_legacy' in tail:
            raise RuntimeError('final model still references _legacy')
        header = '"""Self-contained runtime-corrected 0803 unified talking-head model."""\n\n'
        public_path.write_text(header + legacy + '\n\n# Runtime-correction/public layer\n\n' + tail, encoding='utf-8')
    elif dataset:
        legacy = one(legacy, 'class EmoLevelDataset(data.Dataset):', 'class _BaseEmoLevelDataset(data.Dataset):', 'dataset class')
        public = public.replace('from .dataset_EmotionLevel_clear_jianhua0803_legacy import EmoLevelDataset as _LegacyEmoLevelDataset\n\n', '', 1)
        public = public.replace('class EmoLevelDataset(_LegacyEmoLevelDataset):', 'class EmoLevelDataset(_BaseEmoLevelDataset):', 1)
        if '_legacy' in public or '_Legacy' in public:
            raise RuntimeError('dataset still references legacy')
        public_path.write_text('"""Self-contained MEAD dataset with true-start support."""\n\n' + legacy + '\n\n# True-start extension\n\n' + public, encoding='utf-8')
    legacy_path.unlink()


# Final model and MEAD dataset wrappers -> one file each.
merge_wrapper(MOD/'emotion_dit_Unification_jianhua0803.py', MOD/'emotion_dit_Unification_jianhua0803_legacy.py', core_model=True)
merge_wrapper(DATA/'dataset_EmotionLevel_clear_jianhua0803.py', DATA/'dataset_EmotionLevel_clear_jianhua0803_legacy.py', dataset=True)

# helper.py -> self-contained. Keep original utilities/non-motion loader, add only
# the necessary architecture-aware motion loader + ablation checkpoint dispatch.
base_path = UTIL/'helper_legacy.py'
base = base_path.read_text(encoding='utf-8')
base = base.replace('# from ..modules.emotion_dit_Unification_ditcond0819 import DitTalkingHead  # norm版本\nfrom ..modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead  # norm版本\n', '', 1)
base = one(base, 'def load_model(ckpt_path, model_config, device, model_type):', 'def _load_model_base(ckpt_path, model_config, device, model_type):', 'helper base loader')
append = r'''

# Public loader: normal components keep the original implementation. Motion
# checkpoints use the corrected architecture reconstruction; only controlled
# ablation checkpoints select a different model module.
def _arg(model_args, name, default=None):
    value = getattr(model_args, name, None)
    return default if value is None else value


def _motion_class(model_args):
    import importlib
    variant = str(_arg(model_args, 'model_variant', '')).strip()
    if variant.startswith('emotion_dit_ablation0905_'):
        if not all(ch.isalnum() or ch == '_' for ch in variant):
            raise ValueError(f'unsafe model_variant: {variant!r}')
        return importlib.import_module(f'src.modules.{variant}').DitTalkingHead
    from ..modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead
    return DitTalkingHead


def load_model(ckpt_path, model_config, device, model_type):
    if model_type != 'motion_generator':
        return _load_model_base(ckpt_path, model_config, device, model_type)

    import inspect
    model_data = torch.load(ckpt_path, map_location=device)
    model_args = NullableArgs(model_data['args'])
    feature_dim = int(_arg(model_args, 'feature_dim', 512))
    n_layers = int(_arg(model_args, 'n_layers', 8))
    n_heads = int(_arg(model_args, 'n_heads', 10 if feature_dim == 640 else 8))
    mlp_ratio = int(_arg(model_args, 'mlp_ratio', 4))
    if bool(_arg(model_args, 'model_params_propagated', False)):
        use_indicator = bool(_arg(model_args, 'use_indicator', True))
        no_use_learnable_pe = bool(_arg(model_args, 'no_use_learnable_pe', False))
    else:
        # Preserve the pre-ablation compatibility rule.
        use_indicator = bool(feature_dim != 512 or n_layers != 8 or n_heads != 8)
        no_use_learnable_pe = True

    cls = _motion_class(model_args)
    kwargs = dict(
        device=device,
        target=_arg(model_args, 'target', 'sample'),
        architecture=_arg(model_args, 'architecture', 'decoder'),
        motion_feat_dim=int(_arg(model_args, 'motion_feat_dim', 70)),
        fps=int(_arg(model_args, 'fps', 25)),
        n_motions=int(_arg(model_args, 'n_motions', 64)),
        n_prev_motions=int(_arg(model_args, 'n_prev_motions', 16)),
        audio_model=_arg(model_args, 'audio_model', 'wav2vec2'),
        feature_dim=feature_dim,
        n_diff_steps=int(_arg(model_args, 'n_diff_steps', 500)),
        diff_schedule=_arg(model_args, 'diff_schedule', 'cosine'),
        cfg_mode=_arg(model_args, 'cfg_mode', 'incremental'),
        guiding_conditions=_arg(model_args, 'guiding_conditions', 'audio,emotion'),
        align_mask_width=int(_arg(model_args, 'align_mask_width', 1)),
        n_heads=n_heads, n_layers=n_layers, mlp_ratio=mlp_ratio,
        use_indicator=use_indicator,
        no_use_learnable_pe=no_use_learnable_pe,
    )
    if 'partition_keypoint_indices' in inspect.signature(cls.__init__).parameters:
        kwargs['partition_keypoint_indices'] = _arg(model_args, 'partition_keypoint_indices', '0,1,2,3,4,5,6,7,8,9,10')
    model = cls(**kwargs)
    state = model_data['model']
    state.pop('denoising_net.TE.pe', None)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, model_args
'''
(UTIL/'helper.py').write_text(base + append, encoding='utf-8')
base_path.unlink()

# Stale companions left by previous refactor.
for p in MOD.glob('emotion_dit_ablation0905_*_legacy.py'):
    p.unlink()

legacy = sorted(ROOT.glob('src/**/*_legacy.py'))
if legacy:
    raise RuntimeError(f'legacy files remain: {legacy}')

helper = (UTIL/'helper.py').read_text(encoding='utf-8')
assert 'emotion_dit_Unification_jianhua0803_minsnr_ema' not in helper
assert 'helper_legacy' not in helper
assert 'emotion_dit_ablation0905_' in helper
assert 'from ..modules.emotion_dit_Unification_jianhua0803 import DitTalkingHead' in helper

models = sorted(MOD.glob('emotion_dit_ablation0905_*.py'))
trains = sorted(ROOT.glob('train_Ablation0905_*.py'))
assert len(models) == 12 and len(trains) == 12
print('legacy cleanup complete')

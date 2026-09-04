from pathlib import Path

legacy_path = Path('src/modules/emotion_dit_ablation0905_emotion_residual_legacy.py')
text = legacy_path.read_text(encoding='utf-8')

parameter_block = '''
        # Competing factorization: category-specific motion is represented as an
        # additive trajectory residual on top of a category-agnostic speech path.
        self.emotion_residual = nn.Parameter(torch.zeros(
            emo_classes, self.n_prev_motions + self.n_motions, self.motion_feat_dim
        ))
'''
if parameter_block not in text:
    raise RuntimeError('old free residual parameter block not found')
text = text.replace(parameter_block, '\n', 1)

method_anchor = '    def extract_audio_feature(self, audio, frame_num=None):\n'
method = '''    def _build_emotion_residual(self, emo_index):
        """Build a category residual with the final model's existing parameters.

        The competing method keeps exactly the same trainable parameter count as
        ADEF. Existing emotion parameters produce a category code, positional
        encoding makes it time-varying, and the shared motion decoder maps it to
        the 70-D trajectory. No residual table or extra residual network is added.
        """
        emo_feat = self.emo_embed(emo_index).unsqueeze(1)
        emo_shift, emo_scale = self.adaLN_modulation(emo_feat).chunk(2, dim=2)
        emotion_code = emo_shift + emo_scale
        total_len = self.n_prev_motions + self.n_motions
        if self.denoising_net.use_learnable_pe:
            position = self.denoising_net.PE[:, :total_len]
        else:
            position = self.denoising_net.PE.pe[:, :total_len]
        residual_hidden = emotion_code + position.to(emotion_code.dtype)
        return self.denoising_net.motion_dec(residual_hidden)

    def extract_audio_feature(self, audio, frame_num=None):
'''
if method_anchor not in text:
    raise RuntimeError('extract_audio_feature anchor not found')
text = text.replace(method_anchor, method, 1)

count = text.count('torch.index_select(self.emotion_residual, 0, emo_index)')
if count != 2:
    raise RuntimeError(f'expected two free residual lookups, found {count}')
text = text.replace(
    'torch.index_select(self.emotion_residual, 0, emo_index)',
    'self._build_emotion_residual(emo_index)',
)

sample_pos = text.index('    @torch.no_grad()\n    def sample(')
prefix, sample = text[:sample_pos], text[sample_pos:]
old_motion = 'prev_motion_feat = torch.index_select(self.start_motion_feat, 0, emo_index)'
old_audio = 'prev_audio_feat = torch.index_select(self.start_audio_feat, 0, emo_index)'
if old_motion not in sample or old_audio not in sample:
    raise RuntimeError('sample start-prior lookup not found')
sample = sample.replace(
    old_motion,
    'prev_motion_feat = self.start_motion_feat[0:1].expand(batch_size, -1, -1)',
    1,
)
sample = sample.replace(
    old_audio,
    'prev_audio_feat = self.start_audio_feat[0:1].expand(batch_size, -1, -1)',
    1,
)
legacy_path.write_text(prefix + sample, encoding='utf-8')

train_path = Path('train_Ablation0905_emotion_residual.py')
train = train_path.read_text(encoding='utf-8')
if '    "emotion_residual",\n' not in train:
    raise RuntimeError('residual optimizer-group marker not found')
train = train.replace('    "emotion_residual",\n', '', 1)
init_block = '''        if hasattr(model, "emotion_residual"):
            model.emotion_residual.zero_()
'''
sync_block = '''        if hasattr(model, "emotion_residual"):
            model.emotion_residual[1:].copy_(
                model.emotion_residual[0:1].expand_as(model.emotion_residual[1:])
            )
'''
if init_block not in train or sync_block not in train:
    raise RuntimeError('residual-specific initialization blocks not found')
train = train.replace(init_block, '', 1).replace(sync_block, '', 1)
train_path.write_text(train, encoding='utf-8')

doc_path = Path('ABLATION_ICASSP27_0905.md')
doc = doc_path.read_text(encoding='utf-8')
rationale = ('After re-checking the paper claims, only six additional controlled runs are necessary. '
             'The existing 6006/6008/6007/6010/6009 runs already isolate shared-prior warm-start, '
             'balanced MEAD sampling, and Min-SNR. Generic replay (6011) is diagnostic and is not '
             'part of the final method. Radius-0/global attention is not a headline contribution '
             '(the manuscript explicitly treats the local/global temporal design as architectural '
             'context), so it should not consume main ablation budget unless reviewers request it.\n')
extra = rationale + ('\nTwo originally drafted rows are intentionally not duplicated here. **Audio-only** should be '
                     'reported as the controlled KDTalker-style baseline rather than as another ADEF ablation. '
                     '**Late label concatenation** is omitted from the minimum suite because it changes the '
                     'denoiser input projection/capacity and is less clean than the same-parameter additive and '
                     'DiT-internal controls. If the manuscript retains a claim that late concatenation was tested, '
                     'that row must either be implemented separately or removed from the claim/table before submission.\n')
if rationale not in doc:
    raise RuntimeError('documentation rationale anchor not found')
doc = doc.replace(rationale, extra, 1)
old_residual = ('- `emotion_residual`: target emotion is represented as a category-specific additive 80x70 motion '
                'residual on top of a category-agnostic speech trajectory. It cannot alter the speech generator '
                'through acoustic FiLM, so it is a direct residual-composition alternative.\n')
new_residual = ('- `emotion_residual`: target emotion is represented as an additive 80x70 motion residual on top '
                'of a category-agnostic speech trajectory. The residual reuses the final model\'s existing emotion '
                'embedding, affine head, positional encoding, and shared motion decoder, so **no new trainable '
                'parameters are added**. It cannot alter the speech generator through acoustic FiLM, making it an '
                'exact-parameter-count residual-composition alternative.\n')
if old_residual not in doc:
    raise RuntimeError('residual documentation anchor not found')
doc = doc.replace(old_residual, new_residual, 1)
checkpoint = ('Each training script writes its own `variant_name` and `model_variant` into checkpoint args. Keep '
              'those fields when generating videos so a checkpoint is always loaded with its dedicated model file.\n')
checkpoint_new = checkpoint + ('\n`src/utils/helper.py` resolves the `emotion_dit_ablation0905_*` model namespace directly '
                               'from checkpoint metadata and passes the partition-mask argument only to the partition '
                               'model. This prevents evaluation from silently loading the final 6009 class for an '
                               'ablation checkpoint.\n')
if checkpoint not in doc:
    raise RuntimeError('checkpoint documentation anchor not found')
doc = doc.replace(checkpoint, checkpoint_new, 1)
doc_path.write_text(doc, encoding='utf-8')

print('refinement applied')

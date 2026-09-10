import torch

from .emotion_dit import DitTalkingHead as BaseDitTalkingHead


class DitTalkingHead(BaseDitTalkingHead):
    """Two-stage copy of the base emotion DiT with minimal stage controls.

    The denoising network and emotion-modulated audio path are inherited
    unchanged. Stage 1 constructs the same model with audio-only conditioning.
    Stage 2 restores audio+emotion conditioning, loads the general checkpoint,
    freezes the general backbone, and trains only emotion-related parameters.
    """

    def __init__(self, *args, training_stage='emotion', **kwargs):
        if training_stage not in ('general', 'emotion'):
            raise ValueError(f'Unknown training_stage: {training_stage}')
        self.training_stage = training_stage
        if training_stage == 'general':
            kwargs['guiding_conditions'] = 'audio'
        super().__init__(*args, **kwargs)

        if training_stage == 'emotion' and hasattr(self, 'adaLN_modulation'):
            torch.nn.init.zeros_(self.adaLN_modulation[-1].weight)
            torch.nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def load_general_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('model', checkpoint)
        incompatible = self.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys
                      if not key.startswith(('emo_embed.', 'adaLN_modulation.', 'null_emotion_feat'))]
        if unexpected:
            raise RuntimeError(f'Unexpected Stage-1 checkpoint keys: {unexpected}')
        return incompatible

    def set_train_stage(self, training_stage, train_motion_decoder=False):
        self.training_stage = training_stage
        for parameter in self.parameters():
            parameter.requires_grad = training_stage == 'general'

        for parameter in self.audio_encoder.parameters():
            parameter.requires_grad = False

        if training_stage == 'emotion':
            for parameter in self.parameters():
                parameter.requires_grad = False
            for module_name in ('emo_embed', 'adaLN_modulation'):
                module = getattr(self, module_name, None)
                if module is not None:
                    for parameter in module.parameters():
                        parameter.requires_grad = True
            self.start_audio_feat.requires_grad = True
            self.start_motion_feat.requires_grad = True
            if train_motion_decoder:
                for parameter in self.denoising_net.motion_dec.parameters():
                    parameter.requires_grad = True

        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

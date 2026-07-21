from .ADEF_wrapper_2pad_0721 import ADEFWrapper as TwoPadADEFWrapper
from .utils.helper_1pad_0721 import load_model


class ADEFWrapper(TwoPadADEFWrapper):
    def __init__(self, inference_cfg):
        super().__init__(inference_cfg)
        self.motion_generator, self.motion_generator_args = load_model(
            inference_cfg.checkpoint_MotionGenerator,
            self.model_config,
            self.device,
            "motion_generator",
        )
        self.n_motions = self.motion_generator_args.n_motions
        self.n_prev_motions = self.motion_generator_args.n_prev_motions
        self.fps = self.motion_generator_args.fps
        self.audio_unit = 16000.0 / self.fps
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.motion_generator_args.pad_mode
        self.use_indicator = self.motion_generator_args.use_indicator

    def _extract_reference_priors(self, reference_path):
        canonical_kp_feat, _first_motion_feat = super()._extract_reference_priors(
            reference_path
        )
        return canonical_kp_feat

    def gen_motion_sequence(self, args):
        canonical_kp_feat = self._extract_reference_priors(args.reference)
        self.motion_generator.set_reference_priors(canonical_kp_feat)
        try:
            # Call the original non-2pad generation implementation so that only
            # the registered canonical prior is consumed by sample().
            from .ADEF_wrapper import ADEFWrapper as BaseADEFWrapper
            return BaseADEFWrapper.gen_motion_sequence(self, args)
        finally:
            self.motion_generator.clear_reference_priors()


__all__ = ["ADEFWrapper"]

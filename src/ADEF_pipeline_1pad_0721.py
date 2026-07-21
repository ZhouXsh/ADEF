from .ADEF_pipeline import ADEFPipeline as BaseADEFPipeline
from .ADEF_wrapper_1pad_0721 import ADEFWrapper


class ADEFPipeline(BaseADEFPipeline):
    def __init__(self, inference_cfg, crop_cfg):
        super().__init__(inference_cfg, crop_cfg)
        self.adef_wrapper = ADEFWrapper(inference_cfg=inference_cfg)
        self.adef_wrapper.cropper = self.cropper


__all__ = ["ADEFPipeline"]

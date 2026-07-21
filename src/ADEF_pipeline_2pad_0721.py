from .ADEF_pipeline import ADEFPipeline as BaseADEFPipeline
from .ADEF_wrapper_2pad_0721 import ADEFWrapper
from .utils.cropper import Cropper


class ADEFPipeline(BaseADEFPipeline):
    def __init__(self, inference_cfg, crop_cfg):
        self.adef_wrapper = ADEFWrapper(inference_cfg=inference_cfg)
        self.cropper = Cropper(crop_cfg=crop_cfg)
        self.adef_wrapper.cropper = self.cropper

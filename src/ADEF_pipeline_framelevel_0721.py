# coding: utf-8

"""ADEF frame-level 推理 Pipeline 副本。"""

from . import ADEF_pipeline as base_pipeline_module
from .ADEF_pipeline import ADEFPipeline as BaseADEFPipeline
from .ADEF_wrapper_framelevel_0721 import ADEFWrapperFrameLevel0721
from .utils.cropper import Cropper


class ADEFPipelineFrameLevel0721(BaseADEFPipeline):
    def __init__(self, inference_cfg, crop_cfg):
        self.adef_wrapper = ADEFWrapperFrameLevel0721(
            inference_cfg=inference_cfg
        )
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def execute(self, args):
        # 防止 save_results=True 时误读旧模型生成的 <audio>.pkl。
        # frame-level 流程使用独立缓存：<audio>_framelevel_0721.pkl。
        original_remove_suffix = base_pipeline_module.remove_suffix
        base_pipeline_module.remove_suffix = (
            lambda path: original_remove_suffix(path) + '_framelevel_0721'
        )
        try:
            return super().execute(args)
        finally:
            base_pipeline_module.remove_suffix = original_remove_suffix


ADEFPipeline = ADEFPipelineFrameLevel0721

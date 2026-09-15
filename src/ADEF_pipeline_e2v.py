# coding: utf-8
"""Emotion2vec-conditioned ADEF inference pipeline."""

from __future__ import annotations

from .ADEF_pipeline import ADEFPipeline
from .ADEF_wrapper_e2v import ADEFE2VWrapper


class ADEFE2VPipeline(ADEFPipeline):
    """Reuse the portrait-animation pipeline with an e2v motion wrapper."""

    def __init__(self, inference_cfg, crop_cfg):
        # Avoid ADEFPipeline.__init__, which always constructs the legacy
        # emotion-label-only wrapper.
        self.adef_wrapper = ADEFE2VWrapper(inference_cfg=inference_cfg)
        from .utils.cropper import Cropper
        self.cropper = Cropper(crop_cfg=crop_cfg)

    def execute(self, args):
        original_save_results = args.save_results
        if original_save_results and not getattr(args, "allow_legacy_motion_cache", False):
            # The legacy cache filename is only derived from the audio path. It
            # is unsafe for e2v inference because target emotion, checkpoint,
            # and feature files can change independently.
            args.save_results = False
        try:
            return super().execute(args)
        finally:
            args.save_results = original_save_results

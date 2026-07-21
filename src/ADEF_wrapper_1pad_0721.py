import cv2
import torch

from . import ADEF_wrapper as base_wrapper_module
from .ADEF_wrapper import ADEFWrapper as BaseADEFWrapper
from .utils.helper_1pad_0721 import load_model
from .utils.io import load_image_rgb, resize_to_limit


class ADEFWrapper(BaseADEFWrapper):
    """Inference wrapper for the standalone canonical-token-only model."""

    def __init__(self, inference_cfg):
        base_load_model = base_wrapper_module.load_model
        base_wrapper_module.load_model = load_model
        try:
            super().__init__(inference_cfg)
        finally:
            base_wrapper_module.load_model = base_load_model
        self.cropper = None

    def _extract_canonical_token(self, reference_path):
        if self.cropper is None:
            raise RuntimeError(
                "A cropper must be attached before extracting canonical keypoints"
            )

        img_rgb = load_image_rgb(reference_path)
        img_rgb = resize_to_limit(
            img_rgb,
            self.inference_cfg.source_max_dim,
            self.inference_cfg.source_division,
        )
        if self.inference_cfg.flag_do_crop:
            crop_info = self.cropper.crop_source_image(
                img_rgb, self.cropper.crop_cfg
            )
            if crop_info is None:
                raise RuntimeError("No face detected in the reference image")
            img_crop_256x256 = crop_info["img_crop_256x256"]
        else:
            img_crop_256x256 = cv2.resize(img_rgb, (256, 256))

        source = self.prepare_source(img_crop_256x256)
        source_info = self.get_kp_info(source)
        canonical_kp = source_info["kp"].reshape(1, -1)
        if canonical_kp.shape[-1] != 63:
            raise ValueError(
                f"Expected 63 canonical keypoint values, got {canonical_kp.shape[-1]}"
            )
        canonical_token = torch.cat(
            [
                torch.zeros(
                    1,
                    7,
                    dtype=canonical_kp.dtype,
                    device=canonical_kp.device,
                ),
                canonical_kp,
            ],
            dim=-1,
        ).unsqueeze(1)
        return canonical_token.detach()

    def gen_motion_sequence(self, args):
        canonical_kp_feat = self._extract_canonical_token(args.reference)
        self.motion_generator.set_reference_priors(canonical_kp_feat)
        try:
            return super().gen_motion_sequence(args)
        finally:
            self.motion_generator.clear_reference_priors()


__all__ = ["ADEFWrapper"]

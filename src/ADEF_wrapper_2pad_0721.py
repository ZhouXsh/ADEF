import cv2
import torch

from . import ADEF_wrapper as base_wrapper_module
from .ADEF_wrapper import ADEFWrapper as BaseADEFWrapper
from .utils.helper_2pad_0721 import load_model
from .utils.io import load_image_rgb, resize_to_limit


class ADEFWrapper(BaseADEFWrapper):
    def __init__(self, inference_cfg):
        # BaseADEFWrapper resolves load_model from its own module namespace.
        # Replace it only during initialization so the 127-token checkpoint is
        # instantiated with the matching 2pad model instead of the old model.
        base_load_model = base_wrapper_module.load_model
        base_wrapper_module.load_model = load_model
        try:
            super().__init__(inference_cfg)
        finally:
            base_wrapper_module.load_model = base_load_model
        self.cropper = None

    def _stat_tensor(self, key, dtype, device):
        return torch.as_tensor(
            self.template_dict[key], dtype=dtype, device=device
        )

    def _extract_reference_priors(self, reference_path):
        if self.cropper is None:
            raise RuntimeError(
                "A cropper must be attached before generating reference priors"
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
        dtype = source_info["kp"].dtype
        device = source_info["kp"].device
        eps = 1e-9

        canonical_kp = source_info["kp"].reshape(1, -1)
        canonical_token = torch.cat(
            [
                torch.zeros(1, 7, dtype=dtype, device=device),
                canonical_kp,
            ],
            dim=-1,
        ).unsqueeze(1)

        exp = source_info["exp"].reshape(1, -1)
        exp = (
            exp - self._stat_tensor("mean_exp", dtype, device)
        ) / (self._stat_tensor("std_exp", dtype, device) + eps)

        scale = (
            source_info["scale"].reshape(1, -1)
            - self._stat_tensor("min_scale", dtype, device)
        ) / (
            self._stat_tensor("max_scale", dtype, device)
            - self._stat_tensor("min_scale", dtype, device)
            + eps
        )
        trans = (
            source_info["t"].reshape(1, -1)
            - self._stat_tensor("min_t", dtype, device)
        ) / (
            self._stat_tensor("max_t", dtype, device)
            - self._stat_tensor("min_t", dtype, device)
            + eps
        )
        pitch = (
            source_info["pitch"].reshape(1, -1)
            - self._stat_tensor("min_pitch", dtype, device)
        ) / (
            self._stat_tensor("max_pitch", dtype, device)
            - self._stat_tensor("min_pitch", dtype, device)
            + eps
        )
        yaw = (
            source_info["yaw"].reshape(1, -1)
            - self._stat_tensor("min_yaw", dtype, device)
        ) / (
            self._stat_tensor("max_yaw", dtype, device)
            - self._stat_tensor("min_yaw", dtype, device)
            + eps
        )
        roll = (
            source_info["roll"].reshape(1, -1)
            - self._stat_tensor("min_roll", dtype, device)
        ) / (
            self._stat_tensor("max_roll", dtype, device)
            - self._stat_tensor("min_roll", dtype, device)
            + eps
        )
        first_motion_token = torch.cat(
            [exp, scale, trans, pitch, yaw, roll], dim=-1
        ).unsqueeze(1)

        return canonical_token.detach(), first_motion_token.detach()

    def gen_motion_sequence(self, args):
        canonical_kp_feat, first_motion_feat = self._extract_reference_priors(
            args.reference
        )
        self.motion_generator.set_reference_priors(
            canonical_kp_feat, first_motion_feat
        )
        try:
            return super().gen_motion_sequence(args)
        finally:
            self.motion_generator.clear_reference_priors()

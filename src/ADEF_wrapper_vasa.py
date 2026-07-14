"""ADEF wrapper using the copied VASA-style motion generator."""

from threading import Lock

from . import ADEF_wrapper as base_wrapper
from .utils.helper_vasa import load_model as load_vasa_model


_INIT_LOCK = Lock()


class ADEFWrapper(base_wrapper.ADEFWrapper):
    """Keep the original wrapper behavior and swap only its model loader."""

    def __init__(self, inference_cfg):
        # The base wrapper resolves load_model from its module global. Limit the
        # swap to construction and serialize it so other wrapper instances are
        # never exposed to the temporary loader.
        with _INIT_LOCK:
            original_loader = base_wrapper.load_model
            base_wrapper.load_model = load_vasa_model
            try:
                super().__init__(inference_cfg)
            finally:
                base_wrapper.load_model = original_loader

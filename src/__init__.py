"""ADEF package initialization."""

from .seed import GLOBAL_SEED, set_global_seed

# One project-wide seed is applied before model/data modules are imported by any
# training or inference entrypoint under ``src``.
set_global_seed(GLOBAL_SEED)

__all__ = ["GLOBAL_SEED", "set_global_seed"]

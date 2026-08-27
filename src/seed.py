"""Project-wide reproducibility helpers for training and inference."""

import os
import random

import numpy as np
import torch

GLOBAL_SEED = 2026


def set_global_seed(seed=GLOBAL_SEED):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Deterministic cuDNN kernels where available. We intentionally do not call
    # torch.use_deterministic_algorithms(True), because several attention/warp
    # operators used by ADEF may not provide a deterministic CUDA implementation.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return seed


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed=GLOBAL_SEED):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

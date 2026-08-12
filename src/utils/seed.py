"""Centralized seeding for reproducibility."""

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import vllm

        # vLLM respects torch seeds; no separate API needed
        pass
    except ImportError:
        pass

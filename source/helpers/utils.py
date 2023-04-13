import os
import random

import numpy as np
import torch
from .config import Config


def seed_everything() -> None:
    """Set seed for reproducibility for all libraries"""
    random.seed(Config.SEED)
    os.environ["PYTHONHASHSEED"] = str(Config.SEED)
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_folders() -> None:
    """Create folders for logs and checkpoints"""
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)

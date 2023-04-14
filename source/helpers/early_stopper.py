import torch

from .config import Config
from .logger import logger


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model, comment=""):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, comment)
        elif score <= self.best_score:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                logger.info(
                    f"Score increased ({self.best_score:.4f} --> {score:.4f}).  Saving model ..."
                )
            self.best_score = score
            self.save_checkpoint(model, comment)
            self.counter = 0

    def save_checkpoint(self, model, comment=""):
        """Saves model when validation loss decrease."""
        model_checkpoint_components = [Config.MODEL, "checkpoint.pt"]
        if comment:
            model_checkpoint_components.insert(1, comment)
        model_checkpoint_name = "_".join(model_checkpoint_components)

        torch.save(model.state_dict(), Config.CHECKPOINTS_DIR / model_checkpoint_name)

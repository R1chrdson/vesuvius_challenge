"""
Configuration class for the project
"""

import json
import os
from pathlib import Path
from typing import Union, get_type_hints

import torch
from dotenv import load_dotenv

from .serialization import ConfigEncoder

load_dotenv()

TRAINING_KEYS = ["BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "PATIENCE", "CV_FOLDS", "FOLD_IDX", "TILE_SIZE"]

dataset_path_map = {
    "local": "dataset",
    "kaggle": "/kaggle/input/vesuvius-challenge-ink-detection",
}


class AppConfigError(Exception):
    """Raised when there is an error in the configuration"""


def _parse_bool(val: Union[str, bool]) -> bool:
    return val if isinstance(val, bool) else val.lower() in ["true", "yes", "1"]


class AppConfig:
    """
    General configuration class for the project
    Maps environment variables to class attributes
    """

    SEED: int = 777
    LOG_LEVEL: str
    ENVIRONMENT: str
    TILE_SIZE: int
    BATCH_SIZE: int
    EPOCHS: int
    MODEL: str
    LEARNING_RATE: float
    CHECKPOINTS_DIR: Path
    PATIENCE: int
    CV_FOLDS: int = 5
    FOLD_IDX: int = -1
    WANDB_API: str

    def __init__(self, env):
        for field in self.__annotations__:  # pylint: disable=no-member
            # Raise AppConfigError if required field not supplied
            default_value = getattr(self, field, None)
            if default_value is None and env.get(field) is None:
                raise AppConfigError(f"The {field} field is required")

            # Cast env var value to expected type and raise AppConfigError on failure
            try:
                var_type = get_type_hints(AppConfig)[field]
                if var_type == bool:
                    value = _parse_bool(env.get(field, default_value))
                else:
                    value = var_type(env.get(field, default_value))

                self.__setattr__(field, value)
            except ValueError as err:
                raise AppConfigError(
                    f'Unable to cast value of "{env[field]}" to type "{var_type}" for "{field}" field'
                ) from err

    @property
    def NUM_WORKERS(self) -> int:
        """Defines the number of workers for the DataLoader"""
        return os.cpu_count() or 0

    @property
    def DEVICE(self) -> torch.device:
        """Defines the device to use for training"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def DATASET_PATH(self) -> Path:
        """Defines the path to the dataset logic"""
        return Path(dataset_path_map[self.ENVIRONMENT])

    def WANDB_PROJECT(self) -> str:
        return "Vesuvius Challenge"

    def __repr__(self):
        attrs = {
            **vars(self),
            **{
                prop_name: str(getattr(self, prop_name))
                for prop_name in dir(self)
                if isinstance(getattr(type(self), prop_name, None), property)
            },
        }

        # Remove private attributes
        attrs.pop("WANDB_API", None)

        attrs_str = json.dumps(attrs, indent=4, sort_keys=True, cls=ConfigEncoder)
        return f"{type(self).__name__}({attrs_str})"

    def __getitem__(self, key):
        return self.__getattribute__(key)

Config = AppConfig(os.environ)

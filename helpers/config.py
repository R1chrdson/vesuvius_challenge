"""
Configuration class for the project
"""

import inspect
import os
from pathlib import Path
from typing import Union, get_type_hints

from dotenv import load_dotenv

load_dotenv()


dataset_path_map = {
    "local": "dataset",
    "kaggle": "/kaggle/input/vesuvius-challenge-ink-detection",
}


class AppConfigError(Exception):
    """Raised when there is an error in the configuration"""


def _parse_bool(val: Union[str, bool]) -> bool:
    return val if isinstance(val, bool) else val.lower() in ["true", "yes", "1"]


class AppConfig(object):
    """
    General configuration class for the project
    Maps environment variables to class attributes
    """

    SEED: int = 777
    LOG_LEVEL: str
    ENVIRONMENT: str

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
    def DATASET_PATH(self) -> Path:
        """Defines the path to the dataset logic"""
        return Path(dataset_path_map[self.ENVIRONMENT])

    def __repr__(self):
        attrs = vars(self)
        prop_values = {
            prop_name: str(getattr(self, prop_name))
            for prop_name in dir(self)
            if isinstance(getattr(type(self), prop_name, None), property)
        }
        attrs.update(prop_values)
        return f"{type(self).__name__}({attrs})"


Config = AppConfig(os.environ)

from source.models.example_model import InkDetector
from source.models.unet import UNet

from source.helpers.datasets import VesuviusOriginalDataSet, UnetVesuviusDataset

MODELS = {
    "InkDetector": {
        "model": InkDetector,
        "dataset": VesuviusOriginalDataSet,
    },
    "UNet": {
        "model": UNet,
        "dataset": UnetVesuviusDataset,
    }
}

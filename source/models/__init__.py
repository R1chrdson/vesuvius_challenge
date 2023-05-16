from source.models.example_model import InkDetector
from source.models.unet import UNet
from source.models.effnet import EffNet

from source.helpers.dataset import VesuviusOriginalDataSet, UnetVesuviusDataset, EffNetVesuviusDataset

MODELS = {
    "InkDetector": {
        "model": InkDetector,
        "dataset": VesuviusOriginalDataSet,
    },
    "UNet": {
        "model": UNet,
        "dataset": UnetVesuviusDataset,
    },
    "EffNet": {
        "model": EffNet,
        "dataset": VesuviusOriginalDataSet,
    }
}
